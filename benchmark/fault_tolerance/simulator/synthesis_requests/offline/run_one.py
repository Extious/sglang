from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SIMULATOR_SRC = _REPO_ROOT / "tools" / "sglang-simulator" / "src"
for path in (str(_SIMULATOR_SRC), str(_REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from benchmark.fault_tolerance.simulator.synthesis_requests.common.artifacts import (
    copy_simulator_artifacts,
)
from benchmark.fault_tolerance.simulator.synthesis_requests.common.config import (
    build_simulator_config,
    load_experiment_suite,
)
from benchmark.fault_tolerance.simulator.synthesis_requests.common.dataset import (
    build_synthetic_requests,
)
from benchmark.fault_tolerance.simulator.synthesis_requests.common.payload import (
    with_total_request,
)
from benchmark.fault_tolerance.simulator.synthesis_requests.common.plot import (
    write_default_plots,
)
from benchmark.fault_tolerance.simulator.synthesis_requests.common.reporting import (
    write_outputs,
)
from benchmark.fault_tolerance.simulator.synthesis_requests.common.schema import (
    ExecutionMode,
    SyntheticRequest,
)
from benchmark.fault_tolerance.simulator.synthesis_requests.common.suite import (
    ExperimentResult,
    get_strategy,
    prepare_artifact_paths,
)


def _optional_runtime_imports():
    runtime: dict[str, Any] = {}
    try:
        from transformers import AutoTokenizer as imported_auto_tokenizer

        runtime["AutoTokenizer"] = imported_auto_tokenizer
    except ImportError:
        runtime["AutoTokenizer"] = None

    try:
        from sglang.srt.server_args import ServerArgs as imported_server_args

        runtime["ServerArgs"] = imported_server_args
    except ImportError:
        runtime["ServerArgs"] = None

    try:
        from sglang_simulator.simulation.sglang.bench_runner import (
            SGLangBenchmarkRunner as imported_runner,
        )

        runtime["SGLangBenchmarkRunner"] = imported_runner
    except ImportError:
        runtime["SGLangBenchmarkRunner"] = None

    try:
        from sglang_simulator.simulation.benchmark import (
            BenchmarkConfig as imported_benchmark_config,
        )

        runtime["BenchmarkConfig"] = imported_benchmark_config
    except ImportError:
        runtime["BenchmarkConfig"] = _FallbackBenchmarkConfig

    return runtime


@dataclass
class _FallbackBenchmarkConfig:
    request_rate: float = float("inf")
    max_concurrency: int | None = None
    ignore_request_timestamp: bool = False


_RUNTIME = _optional_runtime_imports()
AutoTokenizer = _RUNTIME["AutoTokenizer"]
SGLangBenchmarkRunner = _RUNTIME["SGLangBenchmarkRunner"]
ServerArgs = _RUNTIME["ServerArgs"]
BenchmarkConfig = _RUNTIME["BenchmarkConfig"]

_DEFAULT_CONFIG_DIR = (
    Path(__file__).resolve().parents[1] / "configs" / "synthesis-a100"
)


@dataclass
class OfflineRequest:
    prompt: str
    token_ids: list[int]
    output_length: int
    custom_params: dict[str, Any]


class OfflineDataset:
    def __init__(self, *, tokenizer: object, requests: Iterable[OfflineRequest]):
        self.tokenizer = tokenizer
        self.data = list(requests)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


def run_offline_experiment(
    config_dir: str | Path,
    strategy: str,
    output_dir: str | Path,
) -> ExperimentResult:
    suite = load_experiment_suite(Path(config_dir))
    experiment = get_strategy(suite, strategy)
    artifact_paths = prepare_artifact_paths(output_dir)
    simulator_config = build_simulator_config(
        suite=suite,
        experiment=experiment,
        mode=ExecutionMode.OFFLINE,
    )
    artifact_paths.simulator_config_path.write_text(
        json.dumps(simulator_config, indent=2),
        encoding="utf-8",
    )
    _set_simulator_environment(
        artifact_paths.simulator_config_path,
        artifact_paths.simulator_raw_dir,
    )

    _ensure_runtime_available()
    tokenizer = AutoTokenizer.from_pretrained(suite.server.model_path)
    synthetic_requests = build_synthetic_requests(
        tokenizer,
        suite.workload,
        suite.server.dp_size,
        experiment.backup_policy,
    )
    dataset = _build_offline_dataset(tokenizer, synthetic_requests)
    runner = SGLangBenchmarkRunner(
        server_args=ServerArgs(
            model_path=suite.server.model_path,
            load_format="dummy",
            device="cpu",
            enable_hierarchical_cache=True,
            hicache_storage_backend="file",
            tp_size=suite.server.tp_size,
            pp_size=suite.server.pp_size,
            dp_size=suite.server.dp_size,
            page_size=1,
        )
    )
    try:
        metrics = runner.benchmark(BenchmarkConfig(), dataset) or {}
        request_stats = runner.get_request_stats()
        write_outputs(artifact_paths.output_dir, request_stats)
        copy_simulator_artifacts(
            artifact_paths.simulator_raw_dir,
            artifact_paths.output_dir,
        )
        write_default_plots(artifact_paths.output_dir)
    finally:
        runner.shutdown()

    return ExperimentResult(
        strategy=experiment.strategy,
        output_dir=artifact_paths.output_dir,
        metrics=metrics,
    )


def _set_simulator_environment(simulator_config_path: Path, simulator_raw_dir: Path) -> None:
    os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = str(simulator_config_path)
    os.environ["SGLANG_SIMULATOR_OUTPUT_DIR"] = str(simulator_raw_dir)
    os.environ["SGLANG_SIMULATOR_OUTPUT_MODE"] = (
        ExecutionMode.OFFLINE.simulator_env_value
    )
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("SGLANG_USE_CPU_ENGINE", "1")
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")


def _ensure_runtime_available() -> None:
    missing = []
    for name in ("AutoTokenizer", "SGLangBenchmarkRunner", "ServerArgs"):
        if globals()[name] is None:
            missing.append(name)
    if missing:
        raise RuntimeError(
            "Offline synthesis runner requires optional runtime dependencies. "
            f"Missing symbols: {', '.join(missing)}"
        )


def _build_offline_dataset(
    tokenizer: object,
    synthetic_requests: list[SyntheticRequest],
) -> OfflineDataset:
    dp_counts: dict[int, int] = {}
    for request in synthetic_requests:
        dp_counts[request.assigned_dp_rank] = (
            dp_counts.get(request.assigned_dp_rank, 0) + 1
        )

    return OfflineDataset(
        tokenizer=tokenizer,
        requests=[
            OfflineRequest(
                prompt="",
                token_ids=list(request.token_ids),
                output_length=int(request.output_length),
                custom_params=with_total_request(
                    request,
                    total_request=dp_counts.get(
                        request.assigned_dp_rank,
                        len(synthetic_requests),
                    ),
                ),
            )
            for request in synthetic_requests
        ],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=_DEFAULT_CONFIG_DIR)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    result = run_offline_experiment(args.config_dir, args.strategy, args.output_dir)
    print(
        json.dumps(
            {
                "strategy": result.strategy.value,
                "output_dir": str(result.output_dir),
                "metrics": result.metrics,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
