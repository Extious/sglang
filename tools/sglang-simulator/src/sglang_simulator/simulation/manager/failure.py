from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class BackupPolicy(str, Enum):
    NONE = "none"
    HOST = "host"
    REMOTE_BACKUP = "remote_backup"


@dataclass(frozen=True)
class FailureConfig:
    enabled: bool = False
    backup_policy: BackupPolicy = BackupPolicy.NONE
    failed_dp_rank: int = 0
    inject_after_request: int = 0
    inject_after_task: int = 0
    inject_after_job: int = 0
    timeline_after_start_s: float = 0.0
    inject_delay_s: float = 0.0
    recover_after_request: int = 0
    recover_after_task: int = 0
    recover_after_job: int = 0
    recovery_delay_s: float = 0.0
    host_backup_ratio: float = 1.0
    remote_backup_ratio: float = 1.0

    @classmethod
    def from_dict(cls, raw: dict) -> "FailureConfig":
        if not raw:
            return cls()
        policy = BackupPolicy(str(raw.get("backup_policy", "none")))
        return cls(
            enabled=_parse_bool(raw.get("enabled", False)),
            backup_policy=policy,
            failed_dp_rank=int(raw.get("failed_dp_rank", raw.get("dp_rank", 0))),
            inject_after_request=int(raw.get("inject_after_request", 0) or 0),
            inject_after_task=int(raw.get("inject_after_task", 0) or 0),
            inject_after_job=int(raw.get("inject_after_job", 0) or 0),
            timeline_after_start_s=float(
                raw.get("timeline_after_start_s", 0.0) or 0.0
            ),
            inject_delay_s=float(raw.get("inject_delay_s", raw.get("delay", 0.0)) or 0.0),
            recover_after_request=int(raw.get("recover_after_request", 0) or 0),
            recover_after_task=int(raw.get("recover_after_task", 0) or 0),
            recover_after_job=int(raw.get("recover_after_job", 0) or 0),
            recovery_delay_s=float(
                raw.get("recovery_delay_s", raw.get("recovery_delay", 0.0)) or 0.0
            ),
            host_backup_ratio=_float_or_default(raw.get("host_backup_ratio"), 1.0),
            remote_backup_ratio=_float_or_default(
                raw.get("remote_backup_ratio"), 1.0
            ),
        )


@dataclass(frozen=True)
class FailureEvent:
    event_id: int
    fire_time_s: float
    failed_dp_rank: int
    backup_policy: BackupPolicy


class FailureManager:
    def __init__(self, config: FailureConfig):
        self.config = config
        self.completed_requests = 0
        self._scheduled: Optional[FailureEvent] = None
        self._recovery_fire_time_s: Optional[float] = None
        self._fired = False
        self._recovered = False

    def on_request_completed(
        self, rid: str, completion_time_s: float
    ) -> Optional[FailureEvent]:
        if not self.config.enabled:
            return None
        self.completed_requests += 1
        event = None
        threshold = _first_positive(
            self.config.inject_after_request,
            self.config.inject_after_task,
            self.config.inject_after_job,
        )
        if self._scheduled is None and threshold and self.completed_requests >= threshold:
            self._scheduled = FailureEvent(
                event_id=1,
                fire_time_s=completion_time_s + self.config.inject_delay_s,
                failed_dp_rank=self.config.failed_dp_rank,
                backup_policy=self.config.backup_policy,
            )
            event = self._scheduled

        recovery_threshold = _first_positive(
            self.config.recover_after_request,
            self.config.recover_after_task,
            self.config.recover_after_job,
        )
        if (
            self._fired
            and not self._recovered
            and self._recovery_fire_time_s is None
            and recovery_threshold
            and self.completed_requests >= recovery_threshold
        ):
            self._recovery_fire_time_s = (
                completion_time_s + self.config.recovery_delay_s
            )

        return event

    def maybe_schedule_timeline(self) -> Optional[FailureEvent]:
        if not self.config.enabled or self._scheduled is not None:
            return None
        if self.config.timeline_after_start_s > 0:
            self._scheduled = FailureEvent(
                event_id=1,
                fire_time_s=self.config.timeline_after_start_s
                + self.config.inject_delay_s,
                failed_dp_rank=self.config.failed_dp_rank,
                backup_policy=self.config.backup_policy,
            )
            return self._scheduled
        return None

    def maybe_schedule_request_start(
        self, simulation_args: dict, start_time_s: float
    ) -> Optional[FailureEvent]:
        if not self.config.enabled or self._scheduled is not None:
            return None

        threshold = _first_positive(
            self.config.inject_after_request,
            self.config.inject_after_task,
            self.config.inject_after_job,
        )
        if threshold <= 0:
            return None
        if not self._is_failed_rank_request(simulation_args):
            return None
        if _request_index(simulation_args) <= threshold:
            return None

        self._scheduled = FailureEvent(
            event_id=1,
            fire_time_s=start_time_s + self.config.inject_delay_s,
            failed_dp_rank=self.config.failed_dp_rank,
            backup_policy=self.config.backup_policy,
        )
        return self._scheduled

    def maybe_schedule_recovery_request_start(
        self, simulation_args: dict, start_time_s: float
    ) -> bool:
        if (
            not self.config.enabled
            or not self._fired
            or self._recovered
            or self._recovery_fire_time_s is not None
        ):
            return False

        threshold = _first_positive(
            self.config.recover_after_request,
            self.config.recover_after_task,
            self.config.recover_after_job,
        )
        if threshold <= 0:
            return False
        if not self._is_failed_rank_request(simulation_args):
            return False
        if _request_index(simulation_args) <= threshold:
            return False

        self._recovery_fire_time_s = start_time_s + self.config.recovery_delay_s
        return True

    def should_fire(self, now_s: float) -> Optional[FailureEvent]:
        if self._fired or self._scheduled is None:
            return None
        if now_s >= self._scheduled.fire_time_s:
            self._fired = True
            return self._scheduled
        return None

    def should_recover(self, now_s: float) -> bool:
        if self._recovered or self._recovery_fire_time_s is None:
            return False
        if now_s >= self._recovery_fire_time_s:
            self._recovered = True
            return True
        return False

    def estimate_backed_up_tokens(
        self, backup_policy: BackupPolicy, generated_tokens: int
    ) -> int:
        if backup_policy == BackupPolicy.NONE:
            return 0
        if backup_policy == BackupPolicy.HOST:
            return int(generated_tokens * self.config.host_backup_ratio)
        if backup_policy == BackupPolicy.REMOTE_BACKUP:
            return int(generated_tokens * self.config.remote_backup_ratio)
        return 0

    def restore_penalty_s(
        self,
        *,
        backup_policy: BackupPolicy,
        kv_bytes_per_token: int,
        restored_tokens: int,
        memory_read_bandwidth: float | None,
        disk_read_bandwidth: float | None,
    ) -> float:
        if restored_tokens <= 0 or backup_policy == BackupPolicy.NONE:
            return 0.0
        bytes_to_restore = kv_bytes_per_token * restored_tokens
        if backup_policy == BackupPolicy.HOST:
            bw = memory_read_bandwidth or 1.0
        else:
            bw = disk_read_bandwidth or 1.0
        return float(bytes_to_restore / bw)

    def pick_recovery_dp_rank(self, failed_dp_rank: int, dp_size: int) -> int:
        if dp_size <= 1:
            return failed_dp_rank
        return (failed_dp_rank + 1) % dp_size

    def _is_failed_rank_request(self, simulation_args: dict) -> bool:
        return (
            int(simulation_args.get("assigned_dp_rank", 0) or 0)
            == self.config.failed_dp_rank
        )


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "0", "false", "f", "no", "n", "off"}:
            return False
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        return float(normalized) != 0
    return bool(value)


def _float_or_default(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _first_positive(*values: int) -> int:
    for value in values:
        if value > 0:
            return value
    return 0


def _request_index(simulation_args: dict) -> int:
    for key in ("job_id", "request_id", "worker_seq"):
        value = simulation_args.get(key)
        if value in (None, ""):
            continue
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if key == "worker_seq":
            return index + 1
        return index
    return 0
