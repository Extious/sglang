from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import (
    BackupPolicy,
    BackupStrategy,
    ExecutionMode,
    ExperimentSuite,
    FailureConfig,
    PlatformConfig,
    PredictorConfig,
    ServerConfig,
    StrategyExperiment,
    WorkloadConfig,
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing config file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_request_rate(value: Any) -> float:
    if isinstance(value, str) and value.lower() == "inf":
        return float("inf")
    return float(value)


def _parse_bool(value: Any, *, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "0", "false", "f", "no", "n", "off"}:
            return False
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _server_from_raw(raw: dict[str, Any]) -> ServerConfig:
    return ServerConfig(
        model_path=str(raw["model_path"]),
        dp_size=int(raw.get("dp_size", 1)),
        tp_size=int(raw.get("tp_size", 1)),
        pp_size=int(raw.get("pp_size", 1)),
        accelerator_name=str(raw.get("accelerator_name", "a100_sxm")),
        port=int(raw.get("port", 30000)),
        host=str(raw.get("host", "127.0.0.1")),
    )


def _workload_from_raw(raw: dict[str, Any]) -> WorkloadConfig:
    return WorkloadConfig(
        input_len=int(raw["input_len"]),
        output_len=int(raw["output_len"]),
        num_requests=int(raw["num_requests"]),
        app_workers=int(raw.get("app_workers", 1)),
        request_rate=_parse_request_rate(raw.get("request_rate", "inf")),
        seed=int(raw.get("seed", 1)),
        shared_prefix_len=int(raw.get("shared_prefix_len", 0) or 0),
    )


def _failure_from_raw(raw: dict[str, Any]) -> FailureConfig:
    return FailureConfig(
        enabled=_parse_bool(raw.get("enabled"), default=True),
        inject_after_job=int(raw.get("inject_after_job", 0) or 0),
        recover_after_job=int(raw.get("recover_after_job", 0) or 0),
        delay=float(raw.get("delay", 0) or 0),
        dp_rank=int(raw.get("dp_rank", raw.get("failed_dp_rank", 0)) or 0),
    )


def _predictor_from_raw(raw: dict[str, Any]) -> PredictorConfig:
    return PredictorConfig(
        name=str(raw.get("name", "aiconfigurator")),
        database_mode=str(raw.get("database_mode", "SILICON")),
        database_path=raw.get("database_path"),
        prefill_scale_factor=float(raw.get("prefill_scale_factor", 1.0)),
        decode_scale_factor=float(raw.get("decode_scale_factor", 1.0)),
    )


def _platform_from_raw(raw: dict[str, Any]) -> PlatformConfig:
    return PlatformConfig(
        disk_read_bandwidth_gb=float(raw.get("disk_read_bandwidth_gb", 8)),
        disk_write_bandwidth_gb=float(raw.get("disk_write_bandwidth_gb", 8)),
        memory_read_bandwidth_gb=float(raw.get("memory_read_bandwidth_gb", 64)),
        memory_write_bandwidth_gb=float(raw.get("memory_write_bandwidth_gb", 64)),
        num_device_per_node=int(raw.get("num_device_per_node", 8)),
    )


def load_experiment_suite(config_dir: Path) -> ExperimentSuite:
    base = _load_json(config_dir / "base.json")
    strategies_raw = _load_json(config_dir / "strategies.json")
    strategies = tuple(
        StrategyExperiment(
            strategy=BackupStrategy(name),
            backup_policy=BackupPolicy(raw["backup_policy"]),
        )
        for name, raw in strategies_raw["strategies"].items()
    )
    return ExperimentSuite(
        name=str(base["name"]),
        config_dir=config_dir,
        server=_server_from_raw(base["server"]),
        workload=_workload_from_raw(base["workload"]),
        failure=_failure_from_raw(base["failure"]),
        predictor=_predictor_from_raw(base.get("predictor", {})),
        platform=_platform_from_raw(base.get("platform", {})),
        strategies=strategies,
    )


def build_simulator_config(
    *,
    suite: ExperimentSuite,
    experiment: StrategyExperiment,
    mode: ExecutionMode,
) -> dict[str, Any]:
    predictor = {
        "name": suite.predictor.name,
        "database_mode": suite.predictor.database_mode,
        "prefill_scale_factor": suite.predictor.prefill_scale_factor,
        "decode_scale_factor": suite.predictor.decode_scale_factor,
    }
    if suite.predictor.database_path:
        predictor["database_path"] = suite.predictor.database_path

    return {
        "platform": {
            "accelerator": {
                "name": suite.server.accelerator_name,
                "hbm_capacity_gb": 80,
            },
            "disk_read_bandwidth_gb": suite.platform.disk_read_bandwidth_gb,
            "disk_write_bandwidth_gb": suite.platform.disk_write_bandwidth_gb,
            "memory_read_bandwidth_gb": suite.platform.memory_read_bandwidth_gb,
            "memory_write_bandwidth_gb": suite.platform.memory_write_bandwidth_gb,
            "num_device_per_node": suite.platform.num_device_per_node,
        },
        "predictor": predictor,
        "scheduler": {
            "tp_size": suite.server.tp_size,
            "pp_size": suite.server.pp_size,
            "dp_size": suite.server.dp_size,
            "backend_version": "0.5.9",
        },
        "failure": {
            "enabled": suite.failure.enabled,
            "backup_policy": experiment.backup_policy.value,
            "failed_dp_rank": suite.failure.dp_rank,
            "inject_after_job": suite.failure.inject_after_job,
            "inject_after_task": 0,
            "inject_delay_s": suite.failure.delay,
            "recover_after_job": suite.failure.recover_after_job,
            "recover_after_task": 0,
        },
        "benchmark": {
            "suite": suite.name,
            "workload": "synthesis_requests",
            "execution_mode": mode.value,
            "strategy": experiment.strategy.value,
            "request_rate": (
                "inf"
                if suite.workload.request_rate == float("inf")
                else suite.workload.request_rate
            ),
        },
    }
