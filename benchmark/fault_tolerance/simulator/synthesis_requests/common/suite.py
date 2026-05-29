from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .schema import (
    ArtifactPaths,
    BackupStrategy,
    ExperimentSuite,
    StrategyExperiment,
)


@dataclass(frozen=True)
class ExperimentResult:
    strategy: BackupStrategy
    output_dir: Path
    metrics: dict


def get_strategy(
    suite: ExperimentSuite,
    strategy: BackupStrategy | str,
) -> StrategyExperiment:
    try:
        requested = (
            strategy if isinstance(strategy, BackupStrategy) else BackupStrategy(strategy)
        )
    except ValueError as exc:
        available = ", ".join(
            experiment.strategy.value for experiment in suite.strategies
        )
        raise ValueError(
            f"Unknown strategy {strategy!r} for suite {suite.name!r}. "
            f"Available strategies: {available}"
        ) from exc

    for experiment in suite.strategies:
        if experiment.strategy == requested:
            return experiment

    available = ", ".join(experiment.strategy.value for experiment in suite.strategies)
    raise ValueError(
        f"Unknown strategy {strategy!r} for suite {suite.name!r}. "
        f"Available strategies: {available}"
    )


def default_output_dir(
    root: str | Path,
    suite_name: str,
    mode: str,
    strategy: BackupStrategy | str,
) -> Path:
    strategy_name = (
        strategy.value if isinstance(strategy, BackupStrategy) else str(strategy)
    )
    return Path(root) / suite_name / mode / strategy_name


def prepare_artifact_paths(output_dir: str | Path) -> ArtifactPaths:
    output_path = Path(output_dir)
    simulator_raw_dir = output_path / "simulator_raw"
    output_path.mkdir(parents=True, exist_ok=True)
    if simulator_raw_dir.exists():
        shutil.rmtree(simulator_raw_dir)
    simulator_raw_dir.mkdir(parents=True, exist_ok=True)
    return ArtifactPaths(
        output_dir=output_path,
        simulator_raw_dir=simulator_raw_dir,
        simulator_config_path=output_path / "simulator_config.json",
    )
