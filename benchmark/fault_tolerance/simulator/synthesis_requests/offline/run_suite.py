from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SIMULATOR_SRC = _REPO_ROOT / "tools" / "sglang-simulator" / "src"
for path in (str(_SIMULATOR_SRC), str(_REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from benchmark.fault_tolerance.simulator.synthesis_requests.common.config import (
    load_experiment_suite,
)
from benchmark.fault_tolerance.simulator.synthesis_requests.common.suite import (
    ExperimentResult,
    default_output_dir,
)
from benchmark.fault_tolerance.simulator.synthesis_requests.offline.run_one import (
    run_offline_experiment,
)

_DEFAULT_CONFIG_DIR = (
    Path(__file__).resolve().parents[1] / "configs" / "synthesis-a100"
)
_DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[3] / "results" / "synthesis_requests"
)


def run_offline_suite(config_dir: str | Path, output_root: str | Path) -> list[dict]:
    suite = load_experiment_suite(Path(config_dir))
    results = []
    strategy_root = Path(output_root) / suite.name / "offline"
    for experiment in suite.strategies:
        output_dir = default_output_dir(
            output_root,
            suite.name,
            "offline",
            experiment.strategy,
        )
        result = run_offline_experiment(
            config_dir,
            experiment.strategy.value,
            output_dir,
        )
        results.append(_result_to_dict(result))
    _write_strategy_summary(strategy_root, results)
    return results


def _result_to_dict(result: ExperimentResult | dict[str, Any]) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    return {
        "strategy": result.strategy.value,
        "output_dir": str(result.output_dir),
        "metrics": result.metrics,
    }


def _write_strategy_summary(strategy_root: Path, results: list[dict[str, Any]]) -> None:
    rows = []
    for result in results:
        output_dir = Path(result["output_dir"])
        detail_path = output_dir / "request_detail.csv"
        if not detail_path.is_file():
            continue
        rows.append(_strategy_summary_row(str(result["strategy"]), detail_path))

    if not rows:
        return

    strategy_root.mkdir(parents=True, exist_ok=True)
    columns = [
        "strategy",
        "request_count",
        "normal_avg_latency_s",
        "failure_window_avg_latency_s",
        "recovery_avg_latency_s",
        "failure_impacted_count",
        "retry_count",
        "backed_up_tokens",
    ]
    with (strategy_root / "strategy_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _strategy_summary_row(strategy: str, detail_path: Path) -> dict[str, Any]:
    with detail_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    normal = [row for row in rows if not _bool(row.get("failure_impacted"))]
    impacted = [row for row in rows if _bool(row.get("failure_impacted"))]
    retried = [row for row in rows if _bool(row.get("is_failover_retried"))]
    first_impacted_index = min(
        (_job_index(row) for row in impacted),
        default=None,
    )
    recovery = []
    if first_impacted_index is not None:
        recovery = [
            row
            for row in normal
            if _job_index(row) > first_impacted_index
        ]
    before = [
        row
        for row in normal
        if first_impacted_index is None or _job_index(row) < first_impacted_index
    ]

    return {
        "strategy": strategy,
        "request_count": len(rows),
        "normal_avg_latency_s": _avg_latency(before),
        "failure_window_avg_latency_s": _avg_latency(impacted),
        "recovery_avg_latency_s": _avg_latency(recovery),
        "failure_impacted_count": len(impacted),
        "retry_count": len(retried),
        "backed_up_tokens": sum(
            _int(row.get("pre_failover_backed_up_tokens"))
            for row in retried
        ),
    }


def _avg_latency(rows: list[dict[str, str]]) -> float:
    if not rows:
        return 0.0
    return sum(float(row.get("total_latency_s") or 0) for row in rows) / len(rows)


def _job_index(row: dict[str, str]) -> int:
    try:
        return int(row.get("job_id") or 0)
    except ValueError:
        return 0


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _int(value: Any) -> int:
    if value in (None, ""):
        return 0
    return int(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=_DEFAULT_CONFIG_DIR)
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)

    results = run_offline_suite(args.config_dir, args.output_root)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
