from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class BackupStrategy(str, Enum):
    BASELINE = "baseline"
    HOST_BACKUP = "host_backup"
    REMOTE_BACKUP = "remote_backup"


class BackupPolicy(str, Enum):
    NONE = "none"
    HOST = "host"
    REMOTE_BACKUP = "remote_backup"


class ExecutionMode(str, Enum):
    OFFLINE = "offline"
    ONLINE = "online"

    @property
    def simulator_env_value(self) -> str:
        return "OFFLINE" if self is ExecutionMode.OFFLINE else "BLOCKING"


@dataclass(frozen=True)
class ServerConfig:
    model_path: str
    dp_size: int
    tp_size: int
    pp_size: int
    accelerator_name: str
    port: int = 30000
    host: str = "127.0.0.1"


@dataclass(frozen=True)
class WorkloadConfig:
    input_len: int
    output_len: int
    num_requests: int
    app_workers: int
    request_rate: float
    seed: int
    shared_prefix_len: int = 0


@dataclass(frozen=True)
class FailureConfig:
    enabled: bool
    inject_after_job: int
    recover_after_job: int
    delay: float
    dp_rank: int


@dataclass(frozen=True)
class PredictorConfig:
    name: str = "aiconfigurator"
    database_mode: str = "SILICON"
    database_path: str | None = None
    prefill_scale_factor: float = 1.0
    decode_scale_factor: float = 1.0


@dataclass(frozen=True)
class PlatformConfig:
    disk_read_bandwidth_gb: float
    disk_write_bandwidth_gb: float
    memory_read_bandwidth_gb: float
    memory_write_bandwidth_gb: float
    num_device_per_node: int


@dataclass(frozen=True)
class StrategyExperiment:
    strategy: BackupStrategy
    backup_policy: BackupPolicy


@dataclass(frozen=True)
class ExperimentSuite:
    name: str
    config_dir: Path
    server: ServerConfig
    workload: WorkloadConfig
    failure: FailureConfig
    predictor: PredictorConfig
    platform: PlatformConfig
    strategies: tuple[StrategyExperiment, ...]


@dataclass(frozen=True)
class SyntheticRequest:
    job_id: str
    worker_id: str
    worker_seq: int
    assigned_dp_rank: int
    token_ids: tuple[int, ...]
    output_length: int
    created_time: float
    backup_policy: BackupPolicy
    attempt: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_ids", tuple(self.token_ids))

    @property
    def custom_params(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "worker_id": self.worker_id,
            "worker_seq": self.worker_seq,
            "parallel_worker_client": True,
            "assigned_dp_rank": self.assigned_dp_rank,
            "backup_policy": self.backup_policy.value,
            "attempt": self.attempt,
            "created_time": self.created_time,
        }


@dataclass(frozen=True)
class ArtifactPaths:
    output_dir: Path
    simulator_raw_dir: Path
    simulator_config_path: Path
