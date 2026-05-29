from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SIMULATOR_SRC = _REPO_ROOT / "tools" / "sglang-simulator" / "src"
for path in (str(_SIMULATOR_SRC), str(_REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from benchmark.fault_tolerance.simulator.synthesis_requests.common.artifacts import (
    collect_jsonl_records,
    copy_simulator_artifacts,
)
from benchmark.fault_tolerance.simulator.synthesis_requests.common.config import (
    build_simulator_config,
    load_experiment_suite,
)
from benchmark.fault_tolerance.simulator.synthesis_requests.common.dataset import (
    build_synthetic_requests,
)
from benchmark.fault_tolerance.simulator.synthesis_requests.common.reporting import (
    write_outputs,
)
from benchmark.fault_tolerance.simulator.synthesis_requests.common.schema import (
    BackupStrategy,
    ExecutionMode,
)
from benchmark.fault_tolerance.simulator.synthesis_requests.common.suite import (
    ExperimentResult,
    get_strategy,
    prepare_artifact_paths,
)
from benchmark.fault_tolerance.simulator.synthesis_requests.online.client import (
    run_online_client,
    trigger_profile,
    wait_for_server_ready,
)
from benchmark.fault_tolerance.simulator.synthesis_requests.online.server import (
    start_server,
)

try:
    from transformers import AutoTokenizer as _AutoTokenizer
except ImportError:
    _AutoTokenizer = None

AutoTokenizer = _AutoTokenizer

_DEFAULT_CONFIG_DIR = (
    Path(__file__).resolve().parents[1] / "configs" / "synthesis-a100"
)
_SERVER_READY_TIMEOUT_S = 900.0


def run_online_experiment(
    config_dir: Path,
    strategy: BackupStrategy | str,
    output_dir: Path,
) -> ExperimentResult:
    suite = load_experiment_suite(Path(config_dir))
    experiment = get_strategy(suite, strategy)
    artifacts = prepare_artifact_paths(output_dir)
    simulator_config = build_simulator_config(
        suite=suite,
        experiment=experiment,
        mode=ExecutionMode.ONLINE,
    )
    artifacts.simulator_config_path.write_text(
        json.dumps(simulator_config, indent=2),
        encoding="utf-8",
    )

    if AutoTokenizer is None:
        raise RuntimeError(
            "Online synthesis runner requires optional runtime dependency "
            "AutoTokenizer from transformers."
        )

    tokenizer = AutoTokenizer.from_pretrained(suite.server.model_path)
    synthetic_requests = build_synthetic_requests(
        tokenizer=tokenizer,
        workload=suite.workload,
        dp_size=suite.server.dp_size,
        backup_policy=experiment.backup_policy,
    )

    process = start_server(
        suite=suite,
        sim_config_path=artifacts.simulator_config_path,
        simulator_raw_dir=artifacts.simulator_raw_dir,
    )
    server_log_path = artifacts.simulator_raw_dir / "server.log"
    try:
        wait_for_server_ready(
            host=suite.server.host,
            port=suite.server.port,
            timeout_s=_SERVER_READY_TIMEOUT_S,
            process=process,
            log_path=server_log_path,
        )
        raise_if_server_exited(process, log_path=server_log_path)
        client_results = run_online_client(
            suite.server.host,
            suite.server.port,
            synthetic_requests,
            total_request=len(synthetic_requests),
            app_workers=suite.workload.app_workers,
            request_rate=suite.workload.request_rate,
        )
        trigger_profile(suite.server.host, suite.server.port)
        request_stats = collect_jsonl_records(
            artifacts.simulator_raw_dir,
            "request.jsonl",
        )
        write_outputs(artifacts.output_dir, request_stats)
        copy_simulator_artifacts(
            artifacts.simulator_raw_dir,
            artifacts.output_dir,
        )
    finally:
        _stop_server(process)

    return ExperimentResult(
        strategy=experiment.strategy,
        output_dir=artifacts.output_dir,
        metrics={"client_completed": len(client_results)},
    )


def raise_if_server_exited(
    process: subprocess.Popen[Any],
    *,
    log_path: Path,
) -> None:
    exit_code = process.poll()
    if exit_code is None:
        return

    message = f"Online simulator server exited before readiness with exit code {exit_code}."
    log_tail = _read_log_tail(log_path)
    if log_tail:
        message = f"{message}\nLast server log lines:\n{log_tail}"
    raise RuntimeError(message)


def _read_log_tail(path: Path, max_lines: int = 80) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _stop_server(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=_DEFAULT_CONFIG_DIR)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    result = run_online_experiment(args.config_dir, args.strategy, args.output_dir)
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
