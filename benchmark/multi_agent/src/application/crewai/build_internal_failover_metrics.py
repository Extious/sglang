#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


INTERESTING_RUNTIME_EVENTS = {
    "failover_dispatched",
    "failover_aborted",
    "resume_accepted",
    "resume_rejected",
    "resume_duplicate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build benchmark-facing internal failover metrics."
    )
    parser.add_argument("--trace-log", type=Path, required=True)
    parser.add_argument("--events-file", type=Path, required=True)
    parser.add_argument("--runtime-events-file", type=Path, required=False)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def topic_sort_key(job_id: str) -> tuple[int, str]:
    if job_id.isdigit():
        return (0, f"{int(job_id):010d}")
    return (1, job_id)


def load_json(path: Path) -> Any:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Optional[Path]) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_agent_context(raw_value: Any) -> dict[str, Any]:
    if isinstance(raw_value, dict):
        return raw_value
    if not raw_value:
        return {}
    if not isinstance(raw_value, str):
        return {}
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def build_job_windows(trace_records: list[dict[str, Any]]) -> dict[str, tuple[Optional[datetime], Optional[datetime]]]:
    windows: dict[str, tuple[Optional[datetime], Optional[datetime]]] = {}
    for record in trace_records:
        job_id = str(record.get("topic_id", ""))
        starts = []
        ends = []
        for timing in record.get("agent_timings", []):
            start = parse_timestamp(timing.get("start"))
            end = parse_timestamp(timing.get("end"))
            if start is not None:
                starts.append(start)
            if end is not None:
                ends.append(end)
        windows[job_id] = (
            min(starts) if starts else None,
            max(ends) if ends else None,
        )
    return windows


def enrich_runtime_events(runtime_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for event in runtime_events:
        if event.get("event") not in INTERESTING_RUNTIME_EVENTS:
            continue
        event = dict(event)
        request_context = parse_agent_context(event.get("agent_id"))
        event["request_context"] = request_context
        if "job_id" in request_context:
            event["job_id"] = str(request_context["job_id"])
        if "task_label" in request_context:
            event["task_label"] = request_context["task_label"]
        enriched.append(event)
    enriched.sort(key=lambda item: (item.get("timestamp", ""), item.get("rid", "")))
    return enriched


def build_rid_summary(runtime_events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rid_summary: dict[str, dict[str, Any]] = {}
    for event in runtime_events:
        rid = str(event.get("rid", "") or "")
        if not rid:
            continue
        job_id = str(event.get("job_id", "") or "")
        task_label = str(event.get("task_label", "") or "")
        summary = rid_summary.setdefault(
            rid,
            {
                "rid": rid,
                "job_id": job_id,
                "task_label": task_label,
                "agent_role": event.get("request_context", {}).get("agent_role"),
                "events": [],
                "checkpointed_output_len": 0,
                "visible_output_len": 0,
                "owner_dp_rank": event.get("failed_owner_dp_rank"),
                "backup_dp_rank": event.get("backup_dp_rank"),
                "target_dp_rank": event.get("target_dp_rank"),
                "failover_epoch": 0,
                "status": "unknown",
                "reason": event.get("reason"),
            },
        )
        summary["events"].append(event.get("event"))
        summary["job_id"] = summary["job_id"] or job_id
        summary["task_label"] = summary["task_label"] or task_label
        summary["checkpointed_output_len"] = max(
            int(summary["checkpointed_output_len"]),
            int(event.get("checkpointed_output_len", 0) or 0),
        )
        summary["visible_output_len"] = max(
            int(summary["visible_output_len"]),
            int(event.get("visible_output_len", 0) or 0),
        )
        summary["failover_epoch"] = max(
            int(summary["failover_epoch"]),
            int(event.get("failover_epoch", 0) or 0),
        )
        summary["reason"] = event.get("reason") or summary.get("reason")
        if event.get("event") == "resume_accepted":
            summary["status"] = "resumed"
        elif event.get("event") in {"failover_aborted", "resume_rejected"}:
            if summary["status"] != "resumed":
                summary["status"] = "aborted"
    return rid_summary


def build_impacted_job_ids(
    benchmark_events: list[dict[str, Any]],
    rid_summary: dict[str, dict[str, Any]],
    job_windows: dict[str, tuple[Optional[datetime], Optional[datetime]]],
) -> list[str]:
    impacted = {
        summary["job_id"]
        for summary in rid_summary.values()
        if summary.get("job_id")
    }
    for event in benchmark_events:
        if event.get("event") != "fault_injected":
            continue
        fault_ts = parse_timestamp(event.get("timestamp"))
        if fault_ts is None:
            continue
        for job_id, (start, end) in job_windows.items():
            if start is None or end is None:
                continue
            if start <= fault_ts <= end:
                impacted.add(job_id)
    return sorted(impacted, key=topic_sort_key)


def build_jobs_output(
    trace_records: list[dict[str, Any]],
    rid_summary: dict[str, dict[str, Any]],
    impacted_job_ids: set[str],
) -> list[dict[str, Any]]:
    per_job_rids: dict[str, list[dict[str, Any]]] = {}
    for summary in rid_summary.values():
        job_id = summary.get("job_id")
        if not job_id:
            continue
        per_job_rids.setdefault(job_id, []).append(summary)

    jobs: list[dict[str, Any]] = []
    for record in sorted(trace_records, key=lambda item: topic_sort_key(str(item.get("topic_id", "")))):
        job_id = str(record.get("topic_id", ""))
        rid_entries = per_job_rids.get(job_id, [])
        event_counter = Counter()
        recovered_checkpointed_tokens = 0
        recovered_visible_tokens = 0
        resumed_requests = 0
        aborted_requests = 0
        for entry in rid_entries:
            event_counter.update(entry.get("events", []))
            recovered_checkpointed_tokens += int(entry.get("checkpointed_output_len", 0) or 0)
            recovered_visible_tokens += int(entry.get("visible_output_len", 0) or 0)
            if entry.get("status") == "resumed":
                resumed_requests += 1
            elif entry.get("status") == "aborted":
                aborted_requests += 1

        jobs.append(
            {
                "job_id": job_id,
                "job": record.get("topic", ""),
                "year": record.get("year", ""),
                "worker_id": record.get("worker_id", ""),
                "status": record.get("status", ""),
                "wall_time_s": record.get("summary", {}).get("wall_time_s"),
                "impacted": job_id in impacted_job_ids,
                "runtime_event_counts": dict(event_counter),
                "recovered_checkpointed_tokens": recovered_checkpointed_tokens,
                "recovered_visible_tokens": recovered_visible_tokens,
                "resumed_requests": resumed_requests,
                "aborted_requests": aborted_requests,
            }
        )
    return jobs


def build_tasks_output(rid_summary: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    per_task: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in rid_summary.values():
        job_id = str(entry.get("job_id", "") or "")
        task_label = str(entry.get("task_label", "") or "")
        if not job_id or not task_label:
            continue
        key = (job_id, task_label)
        task_summary = per_task.setdefault(
            key,
            {
                "job_id": job_id,
                "task_label": task_label,
                "agent_role": entry.get("agent_role"),
                "request_count": 0,
                "resumed_requests": 0,
                "aborted_requests": 0,
                "recovered_checkpointed_tokens": 0,
                "recovered_visible_tokens": 0,
            },
        )
        task_summary["request_count"] += 1
        task_summary["recovered_checkpointed_tokens"] += int(
            entry.get("checkpointed_output_len", 0) or 0
        )
        task_summary["recovered_visible_tokens"] += int(
            entry.get("visible_output_len", 0) or 0
        )
        if entry.get("status") == "resumed":
            task_summary["resumed_requests"] += 1
        elif entry.get("status") == "aborted":
            task_summary["aborted_requests"] += 1

    return sorted(
        per_task.values(),
        key=lambda item: (topic_sort_key(str(item["job_id"])), str(item["task_label"])),
    )


def main() -> None:
    args = parse_args()
    trace_records = load_json(args.trace_log)
    if not isinstance(trace_records, list):
        raise ValueError("trace_log must contain a JSON list")

    benchmark_events = load_jsonl(args.events_file)
    runtime_events = enrich_runtime_events(load_jsonl(args.runtime_events_file))
    rid_summary = build_rid_summary(runtime_events)
    job_windows = build_job_windows(trace_records)
    impacted_job_ids = build_impacted_job_ids(
        benchmark_events, rid_summary, job_windows
    )

    jobs = build_jobs_output(trace_records, rid_summary, set(impacted_job_ids))
    tasks = build_tasks_output(rid_summary)

    payload = {
        "summary": {
            "jobs_total": len(trace_records),
            "benchmark_event_count": len(benchmark_events),
            "runtime_event_count": len(runtime_events),
            "tracked_request_count": len(rid_summary),
            "impacted_job_count": len(impacted_job_ids),
        },
        "impacted_job_ids": impacted_job_ids,
        "benchmark_events": benchmark_events,
        "runtime_events": runtime_events,
        "requests": sorted(rid_summary.values(), key=lambda item: item["rid"]),
        "jobs": jobs,
        "tasks": tasks,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
