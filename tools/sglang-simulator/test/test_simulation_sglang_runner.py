import os
import sys
import types
import asyncio
import json
from dataclasses import asdict, dataclass, field

import pytest


os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = (
    os.path.dirname(__file__) + "/assets/config.json"
)
os.environ["CUDA_VISIBLE_DEVICES"] = ""


from sglang_simulator.simulation.sglang.bench_runner import (
    SGLangBenchmarkRunner,
)
from sglang_simulator.simulation.benchmark import BenchmarkConfig
from sglang_simulator.simulation.types import RequestStats


def test_create_simulator_engine_installs_hooks_before_importing_engine(monkeypatch):
    from sglang_simulator.simulation.sglang import bench_runner

    calls = []

    class FakeEngine:
        run_scheduler_process_func = None

        def __init__(self, **kwargs):
            calls.append(("engine_init", kwargs))

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sglang.srt.entrypoints.engine":
            calls.append(("import_engine",))
            module = types.ModuleType(name)
            module.Engine = FakeEngine
            return module
        if name == "sglang_simulator.simulation.sglang.hook_installer":
            module = types.ModuleType(name)

            def install_sglang_hooks():
                calls.append(("install_hooks",))

            module.install_sglang_hooks = install_sglang_hooks
            return module
        if name == "sglang_simulator.simulation.sglang.subprocess_entry":
            module = types.ModuleType(name)
            module.sim_run_scheduler_process = object()
            return module
        return original_import(name, globals, locals, fromlist, level)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", fake_import)

    @dataclass
    class FakeServerArgs:
        disable_cuda_graph: bool = False

    server_args = FakeServerArgs()

    engine = bench_runner._create_simulator_engine(server_args)

    assert isinstance(engine, FakeEngine)
    assert calls[:2] == [("install_hooks",), ("import_engine",)]
    assert server_args.disable_cuda_graph is True


def test_create_simulator_engine_patches_model_runner_imported_by_engine(monkeypatch):
    from sglang_simulator.simulation.sglang import bench_runner

    class FakeModelRunner:
        def initialize(self):
            return "original"

    fake_model_runner_module = types.ModuleType(
        "sglang.srt.model_executor.model_runner"
    )
    fake_model_runner_module.ModelRunner = FakeModelRunner
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.model_executor.model_runner",
        fake_model_runner_module,
    )

    class FakeEngine:
        run_scheduler_process_func = None

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sglang.srt.entrypoints.engine":
            module = types.ModuleType(name)
            module.Engine = FakeEngine
            return module
        if name == "sglang_simulator.simulation.sglang.subprocess_entry":
            module = types.ModuleType(name)
            module.sim_run_scheduler_process = object()
            return module
        return original_import(name, globals, locals, fromlist, level)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", fake_import)

    @dataclass
    class FakeServerArgs:
        disable_cuda_graph: bool = False

    bench_runner._create_simulator_engine(FakeServerArgs())

    assert FakeModelRunner.initialize.__name__ == "override_initialize"


@dataclass
class FakeRequest:
    prompt: str = "prompt"
    token_ids: list[int] = field(default_factory=lambda: [1])
    output_length: int = 1
    custom_params: dict = field(default_factory=dict)


class FakeDataset:
    def __init__(self, requests):
        self.requests = requests

    def __iter__(self):
        return iter(self.requests)

    def __len__(self):
        return len(self.requests)


def test_async_benchmark_ignores_internal_simulated_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_SIMULATOR_OUTPUT_DIR", str(tmp_path))

    class FakeTokenizerManager:
        async def start_profile(self, profile_prefix=None):
            if profile_prefix is None:
                (tmp_path / "metrics.json").write_text(
                    json.dumps({"completed": 2}),
                    encoding="utf-8",
                )

    class FakeEngine:
        def __init__(self):
            self.tokenizer_manager = FakeTokenizerManager()
            self.calls = 0

        async def async_generate(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("simulated gpu failure")
            return {"ok": True}

    runner = SGLangBenchmarkRunner.__new__(SGLangBenchmarkRunner)
    runner.engine = FakeEngine()
    dataset = FakeDataset(
        [
            FakeRequest(custom_params={"assigned_dp_rank": 0}),
            FakeRequest(custom_params={"assigned_dp_rank": 1}),
        ]
    )

    metrics = asyncio.run(runner.async_benchmark(BenchmarkConfig(), dataset))

    assert metrics == {"completed": 2}
    assert runner.engine.calls == 2


def test_async_benchmark_returns_aggregate_dp_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_SIMULATOR_OUTPUT_DIR", str(tmp_path))

    def write_request(path, stats):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(stats)) + "\n", encoding="utf-8")

    class FakeTokenizerManager:
        async def start_profile(self, profile_prefix=None):
            if profile_prefix is not None:
                return

            (tmp_path / "metrics.json").write_text(
                json.dumps({"completed": 1, "num_requests": 1, "time_cost": 7.0}),
                encoding="utf-8",
            )
            write_request(
                tmp_path / "dp_0" / "request.jsonl",
                RequestStats(
                    rid="dp0",
                    status="completed",
                    input_length=10,
                    output_length=2,
                    created_time=0.0,
                    queue_start=0.0,
                    queue_end=0.0,
                    last_event_time=1.0,
                    gen_token_latencies=[0.1, 0.2],
                ),
            )
            write_request(
                tmp_path / "dp_1" / "request.jsonl",
                RequestStats(
                    rid="dp1",
                    status="completed",
                    input_length=20,
                    output_length=3,
                    created_time=0.0,
                    queue_start=0.0,
                    queue_end=0.0,
                    last_event_time=2.0,
                    gen_token_latencies=[0.3, 0.4, 0.5],
                ),
            )

    class FakeEngine:
        tokenizer_manager = FakeTokenizerManager()

        async def async_generate(self, **kwargs):
            return {"ok": True}

    runner = SGLangBenchmarkRunner.__new__(SGLangBenchmarkRunner)
    runner.engine = FakeEngine()
    dataset = FakeDataset([FakeRequest(), FakeRequest()])

    metrics = asyncio.run(runner.async_benchmark(BenchmarkConfig(), dataset))

    assert metrics["num_requests"] == 2
    assert metrics["completed"] == 2
    assert metrics["total_input"] == 30
    assert metrics["total_output"] == 5
    assert metrics["time_cost"] == 7.0


def test_async_benchmark_reraises_unexpected_request_error(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_SIMULATOR_OUTPUT_DIR", str(tmp_path))

    class FakeTokenizerManager:
        async def start_profile(self, profile_prefix=None):
            pass

    class FakeEngine:
        tokenizer_manager = FakeTokenizerManager()

        async def async_generate(self, **kwargs):
            raise RuntimeError("unexpected")

    runner = SGLangBenchmarkRunner.__new__(SGLangBenchmarkRunner)
    runner.engine = FakeEngine()
    dataset = FakeDataset([FakeRequest()])

    with pytest.raises(RuntimeError, match="unexpected"):
        asyncio.run(runner.async_benchmark(BenchmarkConfig(), dataset))


def test_runner_preserves_custom_request_metadata():
    runner = SGLangBenchmarkRunner.__new__(SGLangBenchmarkRunner)
    dataset = FakeDataset(
        [
            FakeRequest(
                custom_params={
                    "job_id": "7",
                    "worker_id": "2",
                    "assigned_dp_rank": 1,
                }
            )
        ]
    )

    req, params, routed_dp_rank = next(
        runner.get_request(dataset, ignore_timestamp=True, request_rate=1.0)
    )

    assert req.custom_params["job_id"] == "7"
    assert params["job_id"] == "7"
    assert params["worker_id"] == "2"
    assert params["assigned_dp_rank"] == 1
    assert routed_dp_rank == 1
    assert params["total_request"] == 1
    assert "created_time" in params


def test_runner_preserves_custom_created_time():
    runner = SGLangBenchmarkRunner.__new__(SGLangBenchmarkRunner)
    dataset = FakeDataset(
        [
            FakeRequest(
                custom_params={
                    "job_id": "7",
                    "created_time": 12.5,
                }
            )
        ]
    )

    _, params, _ = next(runner.get_request(dataset, ignore_timestamp=True, request_rate=1.0))

    assert params["job_id"] == "7"
    assert params["created_time"] == 12.5


def test_runner_total_request_uses_dataset_length():
    runner = SGLangBenchmarkRunner.__new__(SGLangBenchmarkRunner)
    dataset = FakeDataset(
        [
            FakeRequest(custom_params={"total_request": 99}),
            FakeRequest(custom_params={}),
        ]
    )

    _, params, _ = next(runner.get_request(dataset, ignore_timestamp=True, request_rate=1.0))

    assert params["total_request"] == 2


def test_runner_total_request_uses_per_dp_count():
    runner = SGLangBenchmarkRunner.__new__(SGLangBenchmarkRunner)
    dataset = FakeDataset(
        [
            FakeRequest(custom_params={"assigned_dp_rank": 0}),
            FakeRequest(custom_params={"assigned_dp_rank": 1}),
            FakeRequest(custom_params={"assigned_dp_rank": 0}),
            FakeRequest(custom_params={"assigned_dp_rank": 1}),
        ]
    )

    requests = list(
        runner.get_request(dataset, ignore_timestamp=True, request_rate=1.0)
    )

    assert requests[0][1]["total_request"] == 2
    assert requests[1][1]["total_request"] == 2
    assert requests[2][1]["total_request"] == 2
    assert requests[3][1]["total_request"] == 2


def test_benchmark_sglang():
    pytest.importorskip("transformers")
    pytest.importorskip("sglang")

    from sglang_simulator.dataset import DatasetArgs, get_dataset
    from sglang_simulator.simulation.benchmark import BenchmarkConfig
    from sglang.srt.server_args import ServerArgs  # noqa
    from transformers import AutoTokenizer

    model_path = "Qwen/Qwen3-8B"
    runner = SGLangBenchmarkRunner(
        server_args=ServerArgs(
            model_path=model_path,
            load_format="dummy",
            device="cpu",
            enable_hierarchical_cache=True,
            hicache_storage_backend="file",
            max_total_tokens=8192,
            page_size=2,
        )
    )

    # Test with benchmark config
    benchmark_config = BenchmarkConfig(request_rate=10, ignore_request_timestamp=True)
    dataset_args = DatasetArgs(
        "random_ids",
        num_prompts=10,
        min_input_len=100,
        max_input_len=101,
        min_output_len=1,
        max_output_len=2,
    )
    dataset = get_dataset(
        dataset_args, tokenizer=AutoTokenizer.from_pretrained(model_path)
    )
    metrics = runner.benchmark(benchmark_config, dataset=dataset)
    assert metrics["completed"] == len(dataset)
    request_stats = runner.get_request_stats()
    for idx, req in enumerate(request_stats):
        assert (
            idx == 0 or req["created_time"] != 0
        ), "The created time should not be zero due to request_rate equal to 10"
        assert (
            dataset_args.min_input_len
            <= req["input_length"]
            <= dataset_args.max_input_len
        )

    first = request_stats[0]
    assert "queue_time_s" in first
    assert "prefetch_time_s" in first
    assert "backup_time_s" in first
    assert "inference_time_s" in first
    assert first["inference_time_s"] >= 0
    iteration_stats = runner.get_iteration_stats()
    assert iteration_stats
    assert "rids" in iteration_stats[0]
    assert "mode" in iteration_stats[0]
    runner.shutdown()


if __name__ == "__main__":
    test_benchmark_sglang()
