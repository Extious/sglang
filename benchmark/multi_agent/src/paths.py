# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.

"""
Unified path configuration.

PathConfig is the single source of truth for all directory paths used by the
experiment pipeline.  Only ``sglang_root`` is required; every other path is
derived automatically.  Optional overrides (venv, deploy_script, log / output
dirs) can be supplied via ``env.yml`` or constructor arguments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


def _find_repo_root(start: Optional[Path] = None) -> Path:
    """Walk up from *start* until we find a directory that contains ``.git``."""
    cur = (start or Path(__file__)).resolve().parent
    for _ in range(10):
        if (cur / ".git").exists():
            return cur
        if cur == cur.parent:
            break
        cur = cur.parent
    raise RuntimeError(
        "Cannot locate sglang repo root. "
        "Set SGLANG_REPO_ROOT or provide sglang_root in env.yml."
    )


def _resolve(base: Path, raw: str | Path) -> Path:
    """Return *raw* as-is when absolute, otherwise resolve relative to *base*."""
    p = Path(raw)
    return p if p.is_absolute() else (base / p).resolve()


@dataclass
class PathConfig:
    """All directory paths used by the experiment pipeline."""

    sglang_root: Path
    multi_agent_root: Path
    src_root: Path
    venv: Path
    deploy_script: Path
    log_dir: Path
    output_dir: Path

    @classmethod
    def from_env_yml(
        cls,
        env_yml_path: Optional[Path] = None,
        *,
        deploy_script: str = "",
        log_dir: str = "",
        output_dir: str = "",
    ) -> PathConfig:
        """Build a PathConfig from ``env.yml`` with automatic root discovery.

        Parameters
        ----------
        env_yml_path:
            Explicit path to ``env.yml``.  When *None* the file is located
            relative to this module (``src/config/env.yml``).
        deploy_script, log_dir, output_dir:
            CLI-level overrides that take precedence over ``env.yml`` values.
        """
        if env_yml_path is None:
            env_yml_path = Path(__file__).resolve().parent / "config" / "env.yml"

        env: dict = {}
        if env_yml_path.is_file():
            with env_yml_path.open(encoding="utf-8") as fh:
                env = yaml.safe_load(fh) or {}

        sglang_root_raw = env.get("sglang_root") or os.environ.get("SGLANG_REPO_ROOT")
        if sglang_root_raw:
            sglang_root = Path(sglang_root_raw).resolve()
        else:
            sglang_root = _find_repo_root()

        ma_root = sglang_root / "benchmark" / "multi_agent"
        src_root = ma_root / "src"

        venv_rel = env.get("venv", ".venv")
        venv = _resolve(sglang_root, venv_rel)

        ds = deploy_script or env.get("deploy_script", "")
        ds_path = _resolve(ma_root, ds) if ds else ma_root / "scripts" / "run_server.slurm"

        ld = log_dir or env.get("exp_log_dir", "logs")
        ld_path = _resolve(ma_root, ld)

        od = output_dir or env.get("exp_output_dir", "results")
        od_path = _resolve(ma_root, od)

        return cls(
            sglang_root=sglang_root,
            multi_agent_root=ma_root,
            src_root=src_root,
            venv=venv,
            deploy_script=ds_path,
            log_dir=ld_path,
            output_dir=od_path,
        )

    @property
    def config_dir(self) -> Path:
        return self.src_root / "config"
