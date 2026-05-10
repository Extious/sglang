# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.

"""Fault injection configuration dataclass mirroring config/<name>/failure.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class FaultInjectionConfig:
    """Corresponds to config/<name>/failure.json."""
    after_job: str = ""
    after_task: str = ""
    timeline_after_start_s: str = ""
    delay: int = 0
    dp_rank: str = "0"
    pp_rank: str = "0"
    tp_rank: str = "0"
    method: str = "api"
    recovery_delay: str = ""
    recover_after_job: str = ""
    recover_after_task: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> FaultInjectionConfig:
        return cls(
            after_job=str(d.get("inject_after_job", "") or ""),
            after_task=str(d.get("inject_after_task", "") or ""),
            timeline_after_start_s=str(d.get("timeline_after_start_s", "") or ""),
            delay=int(d.get("delay", 0)),
            dp_rank=str(d.get("dp_rank", "0") or "0"),
            pp_rank=str(d.get("pp_rank", "0") or "0"),
            tp_rank=str(d.get("tp_rank", "0") or "0"),
            method=str(d.get("method", "api") or "api"),
            recovery_delay=str(d.get("recovery_delay", "") or ""),
            recover_after_job=str(d.get("recover_after_job", "") or ""),
            recover_after_task=str(d.get("recover_after_task", "") or ""),
        )


@dataclass
class FaultInjectorConfig:
    """Parameters for fault injection."""

    server_url: str
    stage_manifest: Optional[Path]
    slurm_job_id: str
    output_dir: Path
    inject_after_job: str = ""
    inject_after_task: str = ""
    timeline_after_start_s: str = ""
    inject_delay: int = 0
    fault_dp_rank: str = ""
    fault_pp_rank: str = "0"
    fault_tp_rank: str = "0"
    unhealthy_timeout: int = 20
    method: str = "api"
    recovery_delay: str = ""
    recover_after_job: str = ""
    recover_after_task: str = ""
    kv_backup: str = "none"
    events_file: Optional[Path] = None
    trace_file: Optional[Path] = None
    failover_events_file: Optional[str] = None
    stage_manifest_data: Optional[dict[str, Any]] = None
    server_topology: str = ""
    inject_match_task_label: str = ""
    inject_match_agent_role: str = ""
    inject_match_worker_id: str = ""

    @property
    def trigger_kind(self) -> str:
        if self.inject_after_task:
            return "task"
        return "job"

    @property
    def thresholds(self) -> list[int]:
        raw = self.inject_after_task if self.inject_after_task else self.inject_after_job
        return [int(x.strip()) for x in raw.split(",") if x.strip()]

    def build_worker_flags(self) -> list[str]:
        flags = [
            "--server-url",
            self.server_url,
            "--stage-manifest",
            str(self.stage_manifest) if self.stage_manifest else "",
            "--slurm-job-id",
            self.slurm_job_id,
            "--output-dir",
            str(self.output_dir),
            "--inject-after-job",
            self.inject_after_job,
            "--inject-after-task",
            self.inject_after_task,
            "--timeline-after-start-s",
            self.timeline_after_start_s,
            "--inject-delay",
            str(int(self.inject_delay)),
            "--fault-dp-rank",
            self.fault_dp_rank,
            "--fault-pp-rank",
            self.fault_pp_rank,
            "--fault-tp-rank",
            self.fault_tp_rank,
            "--unhealthy-timeout",
            str(int(self.unhealthy_timeout)),
            "--method",
            self.method,
            "--recovery-delay",
            self.recovery_delay,
            "--recover-after-job",
            self.recover_after_job,
            "--recover-after-task",
            self.recover_after_task,
            "--kv-backup",
            self.kv_backup,
            "--events-file",
            str(self.events_file) if self.events_file else "",
            "--failover-events-file",
            self.failover_events_file or "",
            "--stage-manifest-json",
            json.dumps(self.stage_manifest_data or {}, separators=(",", ":")),
            "--server-topology",
            self.server_topology,
            "--inject-match-task-label",
            self.inject_match_task_label,
            "--inject-match-agent-role",
            self.inject_match_agent_role,
            "--inject-match-worker-id",
            self.inject_match_worker_id,
        ]
        if self.trace_file:
            flags.extend(["--trace-file", str(self.trace_file)])
        return flags


def fault_injector_config_from_json(data: dict) -> FaultInjectorConfig:
    """Build FaultInjectorConfig from a JSON-serializable dict (paths as strings)."""

    def opt_path(key: str) -> Optional[Path]:
        val = data.get(key)
        if val in (None, ""):
            return None
        return Path(str(val))

    fe = data.get("failover_events_file")
    if fe in (None, ""):
        failover = None
    else:
        failover = str(fe).strip() or None

    return FaultInjectorConfig(
        server_url=str(data["server_url"]),
        stage_manifest=(
            Path(str(data["stage_manifest"]))
            if str(data.get("stage_manifest", "")).strip()
            else None
        ),
        slurm_job_id=str(data["slurm_job_id"]),
        output_dir=Path(str(data["output_dir"])),
        inject_after_job=str(data.get("inject_after_job", "")),
        inject_after_task=str(data.get("inject_after_task", "")),
        timeline_after_start_s=str(data.get("timeline_after_start_s", "")),
        inject_delay=int(data.get("inject_delay", 0)),
        fault_dp_rank=str(data.get("fault_dp_rank", "")),
        fault_pp_rank=str(data.get("fault_pp_rank", "0")),
        fault_tp_rank=str(data.get("fault_tp_rank", "0")),
        unhealthy_timeout=int(data.get("unhealthy_timeout", 20)),
        method=str(data.get("method", "api")),
        recovery_delay=str(data.get("recovery_delay", "")),
        recover_after_job=str(data.get("recover_after_job", "")),
        recover_after_task=str(data.get("recover_after_task", "")),
        kv_backup=str(data.get("kv_backup", "none")),
        events_file=opt_path("events_file"),
        trace_file=opt_path("trace_file"),
        failover_events_file=failover,
        stage_manifest_data=(
            data.get("stage_manifest_data")
            if isinstance(data.get("stage_manifest_data"), dict)
            else None
        ),
        server_topology=str(data.get("server_topology", "")),
        inject_match_task_label=str(data.get("inject_match_task_label", "")),
        inject_match_agent_role=str(data.get("inject_match_agent_role", "")),
        inject_match_worker_id=str(data.get("inject_match_worker_id", "")),
    )


def fault_injector_config_to_json(cfg: FaultInjectorConfig) -> dict:
    """Serialize config for the injector child process."""

    def path_or_empty(p: Optional[Path]) -> str:
        return str(p) if p else ""

    return {
        "server_url": cfg.server_url,
        "stage_manifest": str(cfg.stage_manifest) if cfg.stage_manifest else "",
        "slurm_job_id": cfg.slurm_job_id,
        "output_dir": str(cfg.output_dir),
        "inject_after_job": cfg.inject_after_job,
        "inject_after_task": cfg.inject_after_task,
        "timeline_after_start_s": cfg.timeline_after_start_s,
        "inject_delay": cfg.inject_delay,
        "fault_dp_rank": cfg.fault_dp_rank,
        "fault_pp_rank": cfg.fault_pp_rank,
        "fault_tp_rank": cfg.fault_tp_rank,
        "unhealthy_timeout": cfg.unhealthy_timeout,
        "method": cfg.method,
        "recovery_delay": cfg.recovery_delay,
        "recover_after_job": cfg.recover_after_job,
        "recover_after_task": cfg.recover_after_task,
        "kv_backup": cfg.kv_backup,
        "events_file": path_or_empty(cfg.events_file),
        "trace_file": path_or_empty(cfg.trace_file),
        "failover_events_file": cfg.failover_events_file or "",
        "stage_manifest_data": cfg.stage_manifest_data or {},
        "server_topology": cfg.server_topology,
        "inject_match_task_label": cfg.inject_match_task_label,
        "inject_match_agent_role": cfg.inject_match_agent_role,
        "inject_match_worker_id": cfg.inject_match_worker_id,
    }
