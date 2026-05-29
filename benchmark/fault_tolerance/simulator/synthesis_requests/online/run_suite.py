from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark.fault_tolerance.simulator.synthesis_requests.common.config import (
    load_experiment_suite,
)
from benchmark.fault_tolerance.simulator.synthesis_requests.common.suite import (
    ExperimentResult,
    default_output_dir,
)
from benchmark.fault_tolerance.simulator.synthesis_requests.online.run_one import (
    run_online_experiment,
)

_DEFAULT_CONFIG_DIR = (
    Path(__file__).resolve().parents[1] / "configs" / "synthesis-a100"
)
_DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[3] / "results" / "synthesis_requests"
)


def run_online_suite(config_dir: Path, output_root: Path) -> list[dict]:
    suite = load_experiment_suite(Path(config_dir))
    results = []
    for experiment in suite.strategies:
        output_dir = default_output_dir(
            root=output_root,
            suite_name=suite.name,
            mode="online",
            strategy=experiment.strategy,
        )
        result = run_online_experiment(
            Path(config_dir),
            experiment.strategy,
            output_dir,
        )
        results.append(_result_to_dict(result))
    return results


def _result_to_dict(result: ExperimentResult | dict[str, Any]) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    return {
        "strategy": result.strategy.value,
        "output_dir": str(result.output_dir),
        "metrics": result.metrics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=_DEFAULT_CONFIG_DIR)
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)

    results = run_online_suite(args.config_dir, args.output_root)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
