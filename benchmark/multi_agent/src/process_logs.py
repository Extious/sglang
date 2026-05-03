from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


PROCESS_LOG_NAMES = ("main", "server", "backup_server", "injector", "client")
CONTROL_EVENT_PREFIX = "__MULTI_AGENT_CONTROL__ "

_PROCESS_LOG_ENV = {
    "main": "MULTI_AGENT_MAIN_LOG",
    "server": "MULTI_AGENT_SERVER_LOG",
    "backup_server": "MULTI_AGENT_BACKUP_SERVER_LOG",
    "injector": "MULTI_AGENT_INJECTOR_LOG",
    "client": "MULTI_AGENT_CLIENT_LOG",
}


def _job_id() -> str:
    return os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID") or "manual"


def get_process_log_path(log_dir: str | Path, process_name: str) -> Path:
    if process_name not in _PROCESS_LOG_ENV:
        raise ValueError(f"unknown process log name: {process_name}")

    override = os.environ.get(_PROCESS_LOG_ENV[process_name], "").strip()
    if override:
        return Path(override).expanduser()

    return Path(log_dir) / f"{process_name}_{_job_id()}.log"


def get_process_log_paths(log_dir: str | Path) -> dict[str, Path]:
    return {name: get_process_log_path(log_dir, name) for name in PROCESS_LOG_NAMES}


def process_log_env(log_dir: str | Path) -> dict[str, str]:
    return {
        _PROCESS_LOG_ENV[name]: str(get_process_log_path(log_dir, name))
        for name in PROCESS_LOG_NAMES
    }


def encode_control_event(event: str, **payload: Any) -> str:
    record = {"event": event, **payload}
    return CONTROL_EVENT_PREFIX + json.dumps(record, ensure_ascii=True, separators=(",", ":"))


def decode_control_event(line: str) -> dict[str, Any] | None:
    if not line.startswith(CONTROL_EVENT_PREFIX):
        return None
    try:
        record = json.loads(line[len(CONTROL_EVENT_PREFIX):].strip())
    except json.JSONDecodeError:
        return None
    return record if isinstance(record, dict) else None


def copy_process_logs(log_dir: str | Path, dest_dir: str | Path) -> list[Path]:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for path in get_process_log_paths(log_dir).values():
        if not path.is_file():
            continue
        target = dest / path.name
        shutil.copy2(path, target)
        copied.append(target)
    return copied


def redirect_current_process_to_log(log_path: str | Path):
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("ab", buffering=0)

    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(fh.fileno(), sys.stdout.fileno())
    os.dup2(fh.fileno(), sys.stderr.fileno())
    return fh
