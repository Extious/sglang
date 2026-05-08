# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.

"""Client workload configuration dataclasses."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CrewAIClientConfig:
    """Corresponds to config/<name>/client.json -> crewai section."""

    client_mode: str = "crewai"
    jobs_csv: Path = Path("topics.csv")
    job_limit: int = 6
    app_workers: int = 2
    default_year: str = "2025"
    short_max_tokens: int = 1024
    long_max_tokens: int = 2048
    enable_stream: bool = False
    ignore_eos: int = 1
    worker_start_stagger_s: float = 0.0
    agent_dp_rank_map: dict[str, int] = field(default_factory=dict)
    extra_instructions_path: Optional[Path] = None

    @classmethod
    def from_dict(cls, d: dict, default_csv: Path = Path("topics.csv")) -> CrewAIClientConfig:
        raw = d.get("crewai", {})
        section = raw if isinstance(raw, dict) and raw else d
        _extra = (section.get("extra_instructions_file") or "").strip()
        return cls(
            client_mode=str(d.get("client_mode", "crewai") or "crewai"),
            jobs_csv=Path(section.get("topics_csv") or default_csv),
            job_limit=int(section.get("job_limit", 6)),
            app_workers=int(section.get("app_workers", 2)),
            default_year=str(section.get("default_year", 2025)),
            short_max_tokens=int(section.get("short_max_tokens", 1024)),
            long_max_tokens=int(section.get("long_max_tokens", 2048)),
            enable_stream=bool(section.get("enable_stream", False)),
            ignore_eos=int(section.get("ignore_eos", 1)),
            worker_start_stagger_s=float(section.get("worker_start_stagger_s", 0)),
            agent_dp_rank_map=dict(section.get("agent_dp_rank_map", {})),
            extra_instructions_path=Path(_extra) if _extra else None,
        )


@dataclass
class FixedClientConfig:
    """Fixed-length, non-agent client config from client.json -> fixed."""

    input_len: int = 1024
    output_len: int = 128
    num_requests: int = 64
    app_workers: int = 4
    seed: int = 1
    ignore_eos: int = 1

    @classmethod
    def from_dict(cls, d: dict) -> "FixedClientConfig":
        fixed = d.get("fixed", {})
        if not isinstance(fixed, dict):
            fixed = {}
        return cls(
            input_len=int(fixed.get("input_len", 1024)),
            output_len=int(fixed.get("output_len", 128)),
            num_requests=int(fixed.get("num_requests", 64)),
            app_workers=int(fixed.get("app_workers", 4)),
            seed=int(fixed.get("seed", 1)),
            ignore_eos=int(fixed.get("ignore_eos", d.get("ignore_eos", 1))),
        )


@dataclass
class CrewAIRunnerConfig:
    """Runtime parameters for the CrewAI workload runner."""

    server_url: str
    model_path: str
    jobs_csv: Path
    job_limit: int = 6
    app_workers: int = 2
    default_year: str = "2025"
    short_max_tokens: int = 1024
    long_max_tokens: int = 2048
    enable_stream: bool = False
    ignore_eos: int = 1
    worker_start_stagger_s: float = 0.0
    agent_dp_rank_map: dict[str, int] = field(default_factory=dict)
    extra_instructions_path: Optional[Path] = None
    output_dir: Path = Path(".")
    trace_file: Optional[Path] = None
    events_file: Optional[Path] = None
    control_url: str = ""

    def build_worker_flags(self) -> list[str]:
        flags = [
            "--server-url", self.server_url,
            "--model-path", self.model_path,
            "--jobs-csv", str(self.jobs_csv),
            "--job-limit", str(self.job_limit),
            "--app-workers", str(self.app_workers),
            "--default-year", self.default_year,
            "--short-max-tokens", str(self.short_max_tokens),
            "--long-max-tokens", str(self.long_max_tokens),
            "--enable-stream", str(int(self.enable_stream)),
            "--ignore-eos", str(self.ignore_eos),
            "--worker-start-stagger-s", str(self.worker_start_stagger_s),
            "--agent-dp-rank-map",
            json.dumps(dict(self.agent_dp_rank_map or {}), separators=(",", ":")),
            "--output-dir", str(self.output_dir),
        ]
        if self.trace_file:
            flags.extend(["--trace-file", str(self.trace_file)])
        if self.events_file:
            flags.extend(["--events-file", str(self.events_file)])
        if self.control_url:
            flags.extend(["--control-url", self.control_url])
        if self.extra_instructions_path:
            flags.extend(["--extra-instructions-file", str(self.extra_instructions_path)])
        return flags


@dataclass
class FixedRunnerConfig:
    """Runtime parameters for the fixed-length workload runner."""

    server_url: str
    model_path: str
    input_len: int = 1024
    output_len: int = 128
    num_requests: int = 64
    app_workers: int = 4
    seed: int = 1
    ignore_eos: int = 1
    output_dir: Path = Path(".")
    control_url: str = ""

    def build_worker_flags(self) -> list[str]:
        flags = [
            "--server-url", self.server_url,
            "--model-path", self.model_path,
            "--input-len", str(self.input_len),
            "--output-len", str(self.output_len),
            "--num-requests", str(self.num_requests),
            "--app-workers", str(self.app_workers),
            "--seed", str(self.seed),
            "--ignore-eos", str(self.ignore_eos),
            "--output-dir", str(self.output_dir),
        ]
        if self.control_url:
            flags.extend(["--control-url", self.control_url])
        return flags


def _path_or_empty(p: Optional[Path]) -> str:
    return str(p) if p else ""


def runner_config_from_json(data: dict) -> CrewAIRunnerConfig:
    """Build CrewAIRunnerConfig from a JSON-serializable dict."""
    dp_map = data.get("agent_dp_rank_map", {})
    if isinstance(dp_map, str):
        dp_map = json.loads(dp_map) if dp_map else {}
    trace = data.get("trace_file", "")
    events = data.get("events_file", "")
    extra_inst = str(data.get("extra_instructions_file", "") or "").strip()
    return CrewAIRunnerConfig(
        server_url=str(data.get("server_url", "")),
        model_path=str(data.get("model_path", "")),
        jobs_csv=Path(data.get("jobs_csv", "topics.csv")),
        job_limit=int(data.get("job_limit", 6)),
        app_workers=int(data.get("app_workers", 2)),
        default_year=str(data.get("default_year", "2025")),
        short_max_tokens=int(data.get("short_max_tokens", 1024)),
        long_max_tokens=int(data.get("long_max_tokens", 2048)),
        enable_stream=bool(int(data.get("enable_stream", 0))),
        ignore_eos=int(data.get("ignore_eos", 1)),
        worker_start_stagger_s=float(data.get("worker_start_stagger_s", 0)),
        agent_dp_rank_map=dict(dp_map),
        extra_instructions_path=Path(extra_inst) if extra_inst else None,
        output_dir=Path(data.get("output_dir", ".")),
        trace_file=Path(trace) if trace else None,
        events_file=Path(events) if events else None,
        control_url=str(data.get("control_url", "") or ""),
    )


def runner_config_to_json(cfg: CrewAIRunnerConfig) -> dict:
    """Serialize config for the client child process."""
    return {
        "server_url": cfg.server_url,
        "model_path": cfg.model_path,
        "jobs_csv": str(cfg.jobs_csv),
        "job_limit": cfg.job_limit,
        "app_workers": cfg.app_workers,
        "default_year": cfg.default_year,
        "short_max_tokens": cfg.short_max_tokens,
        "long_max_tokens": cfg.long_max_tokens,
        "enable_stream": int(cfg.enable_stream),
        "ignore_eos": cfg.ignore_eos,
        "worker_start_stagger_s": cfg.worker_start_stagger_s,
        "agent_dp_rank_map": dict(cfg.agent_dp_rank_map or {}),
        "extra_instructions_file": (
            str(cfg.extra_instructions_path) if cfg.extra_instructions_path else ""
        ),
        "output_dir": str(cfg.output_dir),
        "trace_file": _path_or_empty(cfg.trace_file),
        "events_file": _path_or_empty(cfg.events_file),
        "control_url": cfg.control_url,
    }
