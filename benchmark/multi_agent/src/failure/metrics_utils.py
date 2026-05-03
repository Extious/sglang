# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.

"""Failover metrics builder (simplified).

Produces internal_failover_metrics.json with only the fields consumed by
plot scripts: impacted_job_ids, benchmark_events, jobs[].recovered_checkpointed_tokens.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def _parse_ts(raw) -> Optional[datetime]:
    if raw is None:
        return None
    try:
        return datetime.fromisoformat(str(raw))
    except (ValueError, TypeError):
        return None


def _load_jsonl(path: Optional[Path]) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            result.append(json.loads(line))
    return result


def _load_trace(path: Optional[Path]) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _sort_key(job_id: str) -> tuple[int, str]:
    if job_id.isdigit():
        return (0, f"{int(job_id):010d}")
    return (1, job_id)


def _find_impacted_jobs(
    trace: list[dict], events: list[dict],
) -> list[str]:
    job_windows: dict[str, tuple[Optional[datetime], Optional[datetime]]] = {}
    for event in events:
        etype = event.get("event", "")
        job_id = str(event.get("job_id", "") or event.get("task_job_id", "") or "")
        ts = _parse_ts(event.get("timestamp"))
        if not job_id or ts is None:
            continue
        start, _ = job_windows.get(job_id, (None, None))
        if etype in ("job_completed", "job_failed"):
            job_windows[job_id] = (start, ts)

    for rec in trace:
        jid = str(rec.get("topic_id", ""))
        if not jid or jid in job_windows:
            continue
        timings = rec.get("agent_timings") or []
        starts = [_parse_ts(t.get("start")) for t in timings]
        ends = [_parse_ts(t.get("end")) for t in timings]
        starts = [s for s in starts if s]
        ends = [e for e in ends if e]
        job_windows[jid] = (min(starts) if starts else None, max(ends) if ends else None)

    impacted: set[str] = set()
    for event in events:
        if event.get("event") != "fault_injected":
            continue
        fault_ts = _parse_ts(event.get("timestamp"))
        if fault_ts is None:
            continue
        for jid, (start, end) in job_windows.items():
            if start and end and start <= fault_ts <= end:
                impacted.add(jid)
    return sorted(impacted, key=_sort_key)


def _build_job_recovery_stats(
    trace: list[dict],
    runtime_events: list[dict],
    impacted: list[str],
) -> list[dict]:
    recovered_by_job: dict[str, int] = {}
    rid_checkpointed: dict[str, int] = {}
    rid_job: dict[str, str] = {}
    for event in runtime_events:
        rid = str(event.get("rid", "") or "")
        if not rid:
            continue
        jid = str(event.get("job_id", "") or "")
        if jid:
            rid_job.setdefault(rid, jid)
        ckpt = int(event.get("checkpointed_output_len", 0) or 0)
        rid_checkpointed[rid] = max(rid_checkpointed.get(rid, 0), ckpt)

    for rid, ckpt in rid_checkpointed.items():
        jid = rid_job.get(rid, "")
        if jid:
            recovered_by_job[jid] = recovered_by_job.get(jid, 0) + ckpt

    impacted_set = set(impacted)
    jobs = []
    for rec in sorted(trace, key=lambda r: _sort_key(str(r.get("topic_id", "")))):
        jid = str(rec.get("topic_id", ""))
        jobs.append({
            "job_id": jid,
            "recovered_checkpointed_tokens": recovered_by_job.get(jid, 0),
        })
    return jobs


def build_internal_failover_metrics(
    trace_log: Optional[Path],
    events_file: Path,
    output: Path,
    runtime_events_file: Optional[Path] = None,
) -> None:
    trace = _load_trace(trace_log)
    events = _load_jsonl(events_file)
    runtime_events = _load_jsonl(runtime_events_file)

    impacted = _find_impacted_jobs(trace, events)
    jobs = _build_job_recovery_stats(trace, runtime_events, impacted)

    payload = {
        "impacted_job_ids": impacted,
        "benchmark_events": events,
        "jobs": jobs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
