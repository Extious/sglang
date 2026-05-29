import json
import os
import sys
from pathlib import Path
import csv

import pytest

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT)]

CONFIG_DIR = (
    ROOT
    / "benchmark/fault_tolerance/simulator/synthesis_requests/configs/synthesis-a100"
)


class DummyTokenizer:
    vocab_size = 100


class FakeAutoTokenizer:
    calls = []

    @classmethod
    def from_pretrained(cls, model_path):
        cls.calls.append(model_path)
        return DummyTokenizer()


class FakeServerArgs:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__class__.calls.append(kwargs)


class FakeRunner:
    instances = []

    def __init__(self, server_args):
        self.server_args = server_args
        self.dataset = None
        self.benchmark_config = None
        self.shutdown_called = False
        self.__class__.instances.append(self)

    def benchmark(self, benchmark_config, dataset):
        assert hasattr(dataset, "__len__")
        assert len(dataset) == 32
        requests = list(dataset)
        assert len(requests) == len(dataset)
        assert requests[0].prompt == ""
        assert requests[0].output_length == 1000
        assert len(requests[0].token_ids) == 20000
        assert requests[0].custom_params["backup_policy"] == "host"
        assert requests[0].custom_params["total_request"] == 16
        assert requests[1].custom_params["total_request"] == 16
        self.dataset = dataset
        self.benchmark_config = benchmark_config

        raw_dir = Path(os.environ["SGLANG_SIMULATOR_OUTPUT_DIR"])
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "request.jsonl").write_text('{"rid": "r1"}\n', encoding="utf-8")
        return {"completed": len(dataset)}

    def get_request_stats(self):
        return [
            {
                "rid": f"r{index}",
                "job_id": str(index),
                "worker_id": request.custom_params["worker_id"],
                "assigned_dp_rank": request.custom_params["assigned_dp_rank"],
                "backup_policy": request.custom_params["backup_policy"],
                "status": "completed",
                "created_time": request.custom_params["created_time"],
                "finish_time": request.custom_params["created_time"] + 1.0,
            }
            for index, request in enumerate(self.dataset, start=1)
        ]

    def shutdown(self):
        self.shutdown_called = True


def _patch_runtime(monkeypatch):
    from benchmark.fault_tolerance.simulator.synthesis_requests.offline import run_one

    FakeAutoTokenizer.calls = []
    FakeServerArgs.calls = []
    FakeRunner.instances = []
    monkeypatch.setattr(run_one, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(run_one, "ServerArgs", FakeServerArgs)
    monkeypatch.setattr(run_one, "SGLangBenchmarkRunner", FakeRunner)
    return run_one


def test_run_offline_experiment_writes_outputs_and_returns_metrics(tmp_path, monkeypatch):
    run_one = _patch_runtime(monkeypatch)
    calls = []

    def fake_write_default_plots(output_dir):
        calls.append(output_dir)
        figures_dir = Path(output_dir) / "figures"
        figures_dir.mkdir()
        (figures_dir / "backup_cache_hit_profile.png").write_bytes(b"cache")
        (figures_dir / "trace_log_profile.png").write_bytes(b"runtime")

    monkeypatch.setattr(run_one, "write_default_plots", fake_write_default_plots)

    result = run_one.run_offline_experiment(
        CONFIG_DIR,
        "host_backup",
        tmp_path / "out",
    )

    simulator_config = json.loads(
        (tmp_path / "out" / "simulator_config.json").read_text(encoding="utf-8")
    )
    assert simulator_config["benchmark"]["execution_mode"] == "offline"
    assert simulator_config["failure"]["backup_policy"] == "host"
    assert result.strategy.value == "host_backup"
    assert result.output_dir == tmp_path / "out"
    assert result.metrics["completed"] == 32
    assert (tmp_path / "out" / "request_detail.csv").is_file()
    assert calls == [tmp_path / "out"]
    assert (tmp_path / "out" / "figures" / "backup_cache_hit_profile.png").is_file()
    assert (tmp_path / "out" / "figures" / "trace_log_profile.png").is_file()
    assert FakeAutoTokenizer.calls == ["Qwen/Qwen3-8B"]
    assert FakeServerArgs.calls[0]["model_path"] == "Qwen/Qwen3-8B"
    assert FakeServerArgs.calls[0]["load_format"] == "dummy"
    assert FakeServerArgs.calls[0]["device"] == "cpu"
    assert FakeServerArgs.calls[0]["enable_hierarchical_cache"] is True
    assert FakeServerArgs.calls[0]["hicache_storage_backend"] == "file"
    assert FakeServerArgs.calls[0]["tp_size"] == 1
    assert FakeServerArgs.calls[0]["pp_size"] == 1
    assert FakeServerArgs.calls[0]["dp_size"] == 2
    assert FakeServerArgs.calls[0]["page_size"] == 1
    assert FakeRunner.instances[0].benchmark_config is not None
    assert FakeRunner.instances[0].shutdown_called is True


def test_run_offline_experiment_sets_simulator_env(tmp_path, monkeypatch):
    run_one = _patch_runtime(monkeypatch)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("SGLANG_USE_CPU_ENGINE", raising=False)
    monkeypatch.delenv("FLASHINFER_DISABLE_VERSION_CHECK", raising=False)

    run_one.run_offline_experiment(CONFIG_DIR, "host_backup", tmp_path / "out")

    assert (
        run_one.os.environ["SGLANG_SIMULATOR_CONFIG_PATH"]
        == str(tmp_path / "out" / "simulator_config.json")
    )
    assert (
        run_one.os.environ["SGLANG_SIMULATOR_OUTPUT_DIR"]
        == str(tmp_path / "out" / "simulator_raw")
    )
    assert run_one.os.environ["SGLANG_SIMULATOR_OUTPUT_MODE"] == "OFFLINE"
    assert run_one.os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert run_one.os.environ["SGLANG_USE_CPU_ENGINE"] == "1"
    assert run_one.os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] == "1"


def test_run_offline_suite_uses_default_strategy_output_dirs(tmp_path, monkeypatch):
    from benchmark.fault_tolerance.simulator.synthesis_requests.offline import run_suite

    calls = []

    def fake_run_one(config_dir, strategy, output_dir):
        calls.append((config_dir, strategy, output_dir))
        return {"strategy": strategy, "output_dir": str(output_dir)}

    monkeypatch.setattr(run_suite, "run_offline_experiment", fake_run_one)

    results = run_suite.run_offline_suite(CONFIG_DIR, tmp_path / "suite-runs")

    assert [call[1] for call in calls] == [
        "baseline",
        "host_backup",
        "remote_backup",
    ]
    assert [call[2] for call in calls] == [
        tmp_path / "suite-runs" / "synthesis-a100" / "offline" / "baseline",
        tmp_path / "suite-runs" / "synthesis-a100" / "offline" / "host_backup",
        tmp_path / "suite-runs" / "synthesis-a100" / "offline" / "remote_backup",
    ]
    assert results == [
        {"strategy": "baseline", "output_dir": str(calls[0][2])},
        {"strategy": "host_backup", "output_dir": str(calls[1][2])},
        {"strategy": "remote_backup", "output_dir": str(calls[2][2])},
    ]


def test_run_offline_suite_writes_cross_strategy_summary(tmp_path, monkeypatch):
    from benchmark.fault_tolerance.simulator.synthesis_requests.offline import run_suite

    def fake_run_one(config_dir, strategy, output_dir):
        output_dir.mkdir(parents=True)
        latencies = {
            "baseline": (10.0, 30.0, 10.0, 2, 0),
            "host_backup": (10.0, 20.0, 10.0, 2, 100),
            "remote_backup": (10.0, 15.0, 10.0, 2, 120),
        }
        before, during, after, retry_count, backed_up = latencies[strategy]
        rows = [
            {
                "job_id": "1",
                "total_latency_s": before,
                "failure_impacted": "False",
                "is_failover_retried": "False",
                "pre_failover_backed_up_tokens": "0",
            },
            {
                "job_id": "11",
                "total_latency_s": during,
                "failure_impacted": "True",
                "is_failover_retried": "True" if retry_count else "False",
                "pre_failover_backed_up_tokens": str(backed_up),
            },
            {
                "job_id": "21",
                "total_latency_s": after,
                "failure_impacted": "False",
                "is_failover_retried": "False",
                "pre_failover_backed_up_tokens": "0",
            },
        ]
        with (output_dir / "request_detail.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        return {"strategy": strategy, "output_dir": str(output_dir)}

    monkeypatch.setattr(run_suite, "run_offline_experiment", fake_run_one)

    run_suite.run_offline_suite(CONFIG_DIR, tmp_path / "suite-runs")

    summary_path = (
        tmp_path
        / "suite-runs"
        / "synthesis-a100"
        / "offline"
        / "strategy_summary.csv"
    )
    assert summary_path.is_file()
    with summary_path.open(newline="", encoding="utf-8") as f:
        summary = {row["strategy"]: row for row in csv.DictReader(f)}

    assert summary["baseline"]["failure_window_avg_latency_s"] == "30.0"
    assert summary["baseline"]["retry_count"] == "1"
    assert summary["host_backup"]["failure_window_avg_latency_s"] == "20.0"
    assert summary["host_backup"]["backed_up_tokens"] == "100"
    assert summary["remote_backup"]["failure_window_avg_latency_s"] == "15.0"
    assert summary["remote_backup"]["backed_up_tokens"] == "120"


def test_run_offline_suite_cli_defaults_to_fault_tolerance_results(monkeypatch, capsys):
    from benchmark.fault_tolerance.simulator.synthesis_requests.offline import run_suite

    calls = []

    def fake_run_suite(config_dir, output_root):
        calls.append((config_dir, output_root))
        return []

    monkeypatch.setattr(run_suite, "run_offline_suite", fake_run_suite)

    assert run_suite.main([]) == 0

    assert calls == [
        (
            ROOT
            / "benchmark/fault_tolerance/simulator/synthesis_requests/configs/synthesis-a100",
            ROOT / "benchmark/fault_tolerance/results/synthesis_requests",
        )
    ]
    assert capsys.readouterr().out.strip() == "[]"


def test_prepare_artifact_paths_clears_stale_simulator_raw(tmp_path):
    from benchmark.fault_tolerance.simulator.synthesis_requests.common.suite import (
        prepare_artifact_paths,
    )

    stale = tmp_path / "out" / "simulator_raw" / "dp_99" / "request.jsonl"
    stale.parent.mkdir(parents=True)
    stale.write_text('{"rid": "stale"}\n', encoding="utf-8")

    paths = prepare_artifact_paths(tmp_path / "out")

    assert paths.simulator_raw_dir.is_dir()
    assert not stale.exists()


def test_get_strategy_raises_useful_error_for_missing_strategy():
    from benchmark.fault_tolerance.simulator.synthesis_requests.common.config import (
        load_experiment_suite,
    )
    from benchmark.fault_tolerance.simulator.synthesis_requests.common.suite import (
        get_strategy,
    )

    suite = load_experiment_suite(CONFIG_DIR)

    with pytest.raises(
        ValueError,
        match="missing.*baseline.*host_backup.*remote_backup",
    ):
        get_strategy(suite, "missing")
