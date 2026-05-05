# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.

"""Server configuration dataclass mirroring config/<name>/server.json."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ServerConfig:
    """Corresponds to config/<name>/server.json."""
    dp_size: int = 2
    pp_size: int = 1
    tp_size: int = 1
    nnodes: int = 1
    hicache_size_gb: int = 0
    quantization: str = ""
    model_path: str = ""
    tool_call_parser: str = ""
    enable_cache_report: bool = False
    enable_hicache: bool = False
    hicache_ratio: float = 0.0
    hicache_storage_backend: str = ""
    hicache_storage_prefetch_policy: str = ""
    remote_backup_port_base: int = 0
    remote_backup_buffer_size_gb: float = 0.0
    # SGLang --load-balance-method (e.g. total_tokens); empty = server default (auto)
    load_balance_method: str = ""
    # SGLang --reasoning-parser (e.g. qwen3 for Qwen3.5-*); empty = no override
    reasoning_parser: str = ""
    # SGLang --context-length override; <=0 = use model derived
    context_length: int = 0
    # SGLang --chat-template (path to a Jinja template). Used to override the
    # built-in template, e.g. relax Qwen3.5's "system at beginning" check that
    # breaks multi-agent (CrewAI) workflows. May be relative to the benchmark
    # src root.
    chat_template: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> ServerConfig:
        hs = d.get("hicache_storage", {})
        return cls(
            dp_size=int(d.get("dp_size", 2)),
            pp_size=int(d.get("pp_size", 1)),
            tp_size=int(d.get("tp_size", 1)),
            nnodes=int(d.get("nnodes", 1)),
            hicache_size_gb=int(d.get("hicache_size_gb", 0)),
            quantization=str(d.get("quantization", "")),
            model_path=str(d.get("model_path", "")),
            tool_call_parser=str(d.get("tool_call_parser", "")),
            enable_cache_report=bool(d.get("enable_cache_report", False)),
            enable_hicache=bool(d.get("enable_hicache", False)),
            hicache_ratio=float(d.get("hicache_ratio", 0.0)),
            hicache_storage_backend=str(hs.get("backend", "")),
            hicache_storage_prefetch_policy=str(hs.get("prefetch_policy", "")),
            remote_backup_port_base=int(hs.get("remote_backup_port_base", 0)),
            remote_backup_buffer_size_gb=float(hs.get("remote_backup_buffer_size_gb", 0.0)),
            load_balance_method=str(d.get("load_balance_method", "")),
            reasoning_parser=str(d.get("reasoning_parser", "")),
            context_length=int(d.get("context_length", 0) or 0),
            chat_template=str(d.get("chat_template", "")),
        )

    def build_deploy_flags(
        self,
        model_path: str,
        server_port: int,
        kv_backup: str,
        default_hicache_size_gb: int = 16,
        venv_dir: str = "",
    ) -> list[str]:
        flags = [
            "--dp-size", str(self.dp_size),
            "--pp-size", str(self.pp_size),
            "--tp-size", str(self.tp_size),
            "--model-path", model_path or "Qwen/Qwen3-8B",
            "--server-port-base", str(server_port),
            "--kv-backup", kv_backup,
        ]
        # Match deploy_server.py: explicit --hicache-size only when GB > 0; else --hicache-ratio.
        # Note: hicache_size_gb==0 must not fall back to a positive size here, or host pool stays
        # too small vs device_pool and HiRadixCache asserts (host must be strictly larger).
        # deploy_server.py only parses --hicache-size-gb (not --hicache-size). When using ratio,
        # pass --hicache-size-gb 0 so platform default (e.g. gpuhome 16) is overridden if --config
        # is missing.
        if self.hicache_size_gb > 0:
            flags.extend(["--hicache-size-gb", str(self.hicache_size_gb)])
        elif self.hicache_ratio > 0:
            flags.extend(
                ["--hicache-size-gb", "0", "--hicache-ratio", str(self.hicache_ratio)]
            )
        else:
            flags.extend(["--hicache-size-gb", str(default_hicache_size_gb)])
        if venv_dir:
            flags.extend(["--venv-dir", venv_dir])
        if self.quantization:
            flags.extend(["--quantization", self.quantization])
        if self.hicache_storage_backend:
            flags.extend(["--hicache-storage-backend", self.hicache_storage_backend])
        if self.hicache_storage_prefetch_policy:
            flags.extend(["--hicache-storage-prefetch-policy", self.hicache_storage_prefetch_policy])
        if kv_backup == "remote_backup" and self.remote_backup_port_base > 0:
            flags.extend(["--remote-backup-port-base", str(self.remote_backup_port_base)])
            if self.remote_backup_buffer_size_gb > 0:
                flags.extend(["--remote-backup-buffer-size-gb",
                              str(self.remote_backup_buffer_size_gb)])
        if self.load_balance_method:
            flags.extend(["--load-balance-method", self.load_balance_method])
        if self.reasoning_parser:
            flags.extend(["--reasoning-parser", self.reasoning_parser])
        if self.context_length and self.context_length > 0:
            flags.extend(["--context-length", str(self.context_length)])
        if self.chat_template:
            flags.extend(["--chat-template", self.chat_template])
        return flags

    def topology_tag(self) -> str:
        return f"DP{self.dp_size}_PP{self.pp_size}_TP{self.tp_size}_N{self.nnodes}"
