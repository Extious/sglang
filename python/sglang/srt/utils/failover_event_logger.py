from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FAILOVER_EVENTS_FILE_ENV = "SGLANG_FAILOVER_EVENTS_FILE"


def append_failover_event(event: str, **payload: Any) -> None:
    path_str = os.environ.get(FAILOVER_EVENTS_FILE_ENV, "").strip()
    if not path_str:
        return

    path = Path(path_str).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with path.open("a", encoding="utf-8") as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        file.write(json.dumps(record, ensure_ascii=True) + "\n")
        file.flush()
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)
