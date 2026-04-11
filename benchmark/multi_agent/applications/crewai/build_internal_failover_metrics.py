#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


INTERESTING_RUNTIME_EVENTS = {
    "failover_dispatched",
    "failover_aborted",
    "resume_accepted",
    "resume_rejected",
    "resume_duplicate",
    "gpu_recovered",
    "recovery_applied",
}

AGENT_TO_TASK_LABEL = {
    "Planning Coordinator": "Phase 1 / Planning",
    "Data Collector": "Phase 2 / Data Collection",
    "Deep Analyst": "Phase 2 / Deep Analysis (critical)",
    "Trend Scout": "Phase 2 / Trend Scan",
    "Risk Assessor": "Phase 2 / Risk Assessment",
    "Report Synthesizer": "Phase 3 / Synthesis",
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



def build_task_instances(trace_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    task_instances: list[dict[str, Any]] = []
    for record in trace_records:
        job_id = str(record.get("topic_id", ""))
        for timing_idx, timing in enumerate(record.get("agent_timings", [])):
            agent_name = str(timing.get("agent", "") or "")
            task_label = AGENT_TO_TASK_LABEL.get(agent_name)
            start = parse_timestamp(timing.get("start"))
            end = parse_timestamp(timing.get("end"))
            if not task_label or start is None or end is None:
                continue
            task_instances.append(
                {
                    "instance_id": f"{job_id}:{task_label}:{timing_idx}",
                    "job_id": job_id,
                    "task_label": task_label,
                    "agent_role": agent_name,
                    "worker_id": record.get("worker_id"),
                    "start": start,
                    "end": end,
                    "prompt_tokens": int(timing.get("prompt_tokens", 0) or 0),
                    "cached_prompt_tokens": int(
                        timing.get("cached_prompt_tokens", 0) or 0
                    ),
                }
            )
    return task_instances




def _backup_cache_tokens_from_timing(timing: dict[str, Any]) -> int:
    v = int(timing.get("backup_cache_tokens", 0) or 0)
    if v > 0:
        return v
    return int(timing.get("l3_storage_cached_tokens", 0) or 0)


def build_trace_backup_cache_hits(
    trace_records: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[tuple[str, str], int]]:
    """Aggregate HiCache storage (L3 / backup) reuse tokens from trace agent_timings."""
    per_job_tokens: dict[str, int] = defaultdict(int)
    per_task_tokens: dict[tuple[str, str], int] = defaultdict(int)

    for record in trace_records:
        job_id = str(record.get("topic_id", "") or "")
        if not job_id:
            continue

        for timing in record.get("agent_timings", []):
            agent_name = str(timing.get("agent", "") or "")
            task_label = str(
                timing.get("task_label")
                or AGENT_TO_TASK_LABEL.get(agent_name)
                or ""
            )
            hit_tokens = _backup_cache_tokens_from_timing(timing)
            if hit_tokens <= 0:
                continue
            per_job_tokens[job_id] += hit_tokens
            if task_label:
                per_task_tokens[(job_id, task_label)] += hit_tokens

    return dict(per_job_tokens), dict(per_task_tokens)


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
                "resumed_input_len": 0,
                "first_failover_ts": None,
                "last_resume_accepted_ts": None,
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
        summary["resumed_input_len"] = max(
            int(summary["resumed_input_len"]),
            int(event.get("resumed_input_len", 0) or 0),
        )
        event_ts = parse_timestamp(event.get("timestamp"))
        if event.get("event") == "failover_dispatched" and event_ts is not None:
            prev = parse_timestamp(summary.get("first_failover_ts"))
            if prev is None or event_ts < prev:
                summary["first_failover_ts"] = event_ts.isoformat()
        if event.get("event") == "resume_accepted":
            summary["status"] = "resumed"
            if event_ts is not None:
                summary["last_resume_accepted_ts"] = event_ts.isoformat()
        elif event.get("event") in {"failover_aborted", "resume_rejected"}:
            if summary["status"] != "resumed":
                summary["status"] = "aborted"
    return rid_summary


def build_fault_events(benchmark_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fault_events: list[dict[str, Any]] = []
    for event in benchmark_events:
        if event.get("event") != "fault_injected":
            continue
        timestamp = parse_timestamp(event.get("timestamp"))
        if timestamp is None:
            continue
        fault_events.append(
            {
                "timestamp": timestamp,
                "event_id": event.get("event_id"),
                "job_id": str(event.get("job_id", "") or ""),
                "job": event.get("job"),
                "task_label": str(event.get("task_label", "") or ""),
                "agent_role": str(event.get("agent_role", "") or ""),
                "worker_id": str(event.get("worker_id", "") or ""),
                "dp_rank": event.get("dp_rank"),
                "pp_rank": event.get("pp_rank"),
                "tp_rank": event.get("tp_rank"),
            }
        )
    fault_events.sort(key=lambda item: item["timestamp"])
    return fault_events


def _task_match_score(
    summary: dict[str, Any], task: dict[str, Any]
) -> tuple[int, int, int, float]:
    summary_job_id = str(summary.get("job_id", "") or "")
    summary_task_label = str(summary.get("task_label", "") or "")
    summary_agent_role = str(summary.get("agent_role", "") or "")
    resumed_input_len = int(summary.get("resumed_input_len", 0) or 0)
    failover_ts = parse_timestamp(summary.get("first_failover_ts"))

    job_penalty = 0 if summary_job_id and task["job_id"] == summary_job_id else 1
    label_penalty = 0 if summary_task_label and task["task_label"] == summary_task_label else 1
    agent_penalty = 0 if summary_agent_role and task["agent_role"] == summary_agent_role else 1
    prompt_gap = (
        abs(int(task.get("prompt_tokens", 0) or 0) - resumed_input_len)
        if resumed_input_len > 0
        else 10**6
    )

    if failover_ts is not None:
        if task["start"] <= failover_ts <= task["end"]:
            time_penalty = 0
            edge_distance = min(
                (failover_ts - task["start"]).total_seconds(),
                (task["end"] - failover_ts).total_seconds(),
            )
        else:
            edge_distance = min(
                abs((failover_ts - task["start"]).total_seconds()),
                abs((failover_ts - task["end"]).total_seconds()),
            )
            time_penalty = 1 if edge_distance <= 30 else 2
    else:
        time_penalty = 1
        edge_distance = 10**6

    return (job_penalty + label_penalty + agent_penalty, time_penalty, prompt_gap, edge_distance)


def attach_request_recovery_context(
    rid_summary: dict[str, dict[str, Any]],
    task_instances: list[dict[str, Any]],
    fault_events: list[dict[str, Any]],
) -> None:
    for summary in rid_summary.values():
        if not task_instances:
            continue

        candidates = task_instances
        summary_job_id = str(summary.get("job_id", "") or "")
        if summary_job_id:
            same_job = [task for task in task_instances if task["job_id"] == summary_job_id]
            if same_job:
                candidates = same_job

        best_task = min(candidates, key=lambda task: _task_match_score(summary, task))
        summary["matched_job_id"] = best_task["job_id"]
        summary["matched_task_label"] = best_task["task_label"]
        summary["matched_agent_role"] = best_task["agent_role"]
        summary["matched_worker_id"] = best_task.get("worker_id")
        summary["matched_task_start"] = best_task["start"].isoformat()
        summary["matched_task_end"] = best_task["end"].isoformat()

        effective_job_id = str(summary.get("matched_job_id", "") or summary_job_id)
        effective_task_label = str(
            summary.get("matched_task_label", "") or summary.get("task_label", "") or ""
        )
        effective_agent_role = str(
            summary.get("matched_agent_role", "") or summary.get("agent_role", "") or ""
        )
        effective_worker_id = str(
            summary.get("matched_worker_id", "") or best_task.get("worker_id", "") or ""
        )

        failover_ts = parse_timestamp(summary.get("first_failover_ts"))
        task_end_ts = parse_timestamp(summary.get("matched_task_end"))
        resume_ts = parse_timestamp(summary.get("last_resume_accepted_ts"))

        fault_candidates = fault_events
        if effective_job_id:
            same_job_faults = [
                event for event in fault_events if str(event.get("job_id", "") or "") == effective_job_id
            ]
            if same_job_faults:
                fault_candidates = same_job_faults

        if effective_task_label:
            same_task_faults = [
                event
                for event in fault_candidates
                if str(event.get("task_label", "") or "") == effective_task_label
            ]
            if same_task_faults:
                fault_candidates = same_task_faults

        if effective_agent_role:
            same_agent_faults = [
                event
                for event in fault_candidates
                if str(event.get("agent_role", "") or "") == effective_agent_role
            ]
            if same_agent_faults:
                fault_candidates = same_agent_faults

        if effective_worker_id:
            same_worker_faults = [
                event
                for event in fault_candidates
                if str(event.get("worker_id", "") or "") == effective_worker_id
            ]
            if same_worker_faults:
                fault_candidates = same_worker_faults

        fault_anchor = failover_ts or task_end_ts
        best_fault = None
        if fault_anchor is not None and fault_candidates:
            best_fault = min(
                fault_candidates,
                key=lambda event: (
                    0 if event["timestamp"] <= fault_anchor else 1,
                    abs((fault_anchor - event["timestamp"]).total_seconds()),
                ),
            )
        elif fault_candidates:
            best_fault = fault_candidates[-1]

        if best_fault is not None:
            summary["fault_event_id"] = best_fault.get("event_id")
            summary["fault_injected_ts"] = best_fault["timestamp"].isoformat()
            summary["fault_task_label"] = best_fault.get("task_label")
            summary["fault_agent_role"] = best_fault.get("agent_role")

        if failover_ts is not None and task_end_ts is not None:
            summary["failover_to_task_completion_s"] = round(
                (task_end_ts - failover_ts).total_seconds(), 3
            )
        if resume_ts is not None and task_end_ts is not None:
            summary["resume_accept_to_task_completion_s"] = round(
                (task_end_ts - resume_ts).total_seconds(), 3
            )
        fault_ts = parse_timestamp(summary.get("fault_injected_ts"))
        if fault_ts is not None and task_end_ts is not None:
            summary["fault_to_task_completion_s"] = round(
                (task_end_ts - fault_ts).total_seconds(), 3
            )


def build_impacted_job_ids(
    benchmark_events: list[dict[str, Any]],
    rid_summary: dict[str, dict[str, Any]],
    job_windows: dict[str, tuple[Optional[datetime], Optional[datetime]]],
) -> list[str]:
    impacted = {
        str(summary.get("matched_job_id", "") or summary.get("job_id", "") or "")
        for summary in rid_summary.values()
        if summary.get("matched_job_id") or summary.get("job_id")
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
    backup_cache_hits_by_job: dict[str, int],
) -> list[dict[str, Any]]:
    per_job_rids: dict[str, list[dict[str, Any]]] = {}
    for summary in rid_summary.values():
        job_id = summary.get("matched_job_id") or summary.get("job_id")
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
        fault_to_completion_samples: list[float] = []
        resume_to_completion_samples: list[float] = []
        for entry in rid_entries:
            event_counter.update(entry.get("events", []))
            recovered_checkpointed_tokens += int(entry.get("checkpointed_output_len", 0) or 0)
            recovered_visible_tokens += int(entry.get("visible_output_len", 0) or 0)
            if entry.get("status") == "resumed":
                resumed_requests += 1
            elif entry.get("status") == "aborted":
                aborted_requests += 1
            if entry.get("fault_to_task_completion_s") is not None:
                fault_to_completion_samples.append(float(entry["fault_to_task_completion_s"]))
            if entry.get("resume_accept_to_task_completion_s") is not None:
                resume_to_completion_samples.append(
                    float(entry["resume_accept_to_task_completion_s"])
                )

        jobs.append(
            {
                "job_id": job_id,
                "job": record.get("topic", ""),
                "year": record.get("year", ""),
                "worker_id": record.get("worker_id", ""),
                "status": record.get("status", ""),
                "wall_time_s": record.get("summary", {}).get("wall_time_s"),
                "impacted": job_id in impacted_job_ids,
                "backup_cache_hit_tokens": int(
                    backup_cache_hits_by_job.get(job_id, 0) or 0
                ),
                "prompt_tokens": int(record.get("summary", {}).get("total_prompt_tokens", 0) or 0),
                "runtime_event_counts": dict(event_counter),
                "recovered_checkpointed_tokens": recovered_checkpointed_tokens,
                "recovered_visible_tokens": recovered_visible_tokens,
                "resumed_requests": resumed_requests,
                "aborted_requests": aborted_requests,
                "avg_fault_to_task_completion_s": (
                    round(sum(fault_to_completion_samples) / len(fault_to_completion_samples), 3)
                    if fault_to_completion_samples
                    else None
                ),
                "avg_resume_accept_to_task_completion_s": (
                    round(
                        sum(resume_to_completion_samples)
                        / len(resume_to_completion_samples),
                        3,
                    )
                    if resume_to_completion_samples
                    else None
                ),
            }
        )
    return jobs


def build_tasks_output(
    trace_records: list[dict[str, Any]],
    rid_summary: dict[str, dict[str, Any]],
    impacted_job_ids: set[str],
    backup_cache_hits_by_task: dict[tuple[str, str], int],
) -> list[dict[str, Any]]:
    per_task: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in rid_summary.values():
        job_id = str(entry.get("matched_job_id", "") or entry.get("job_id", "") or "")
        task_label = str(
            entry.get("matched_task_label", "") or entry.get("task_label", "") or ""
        )
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
                    "backup_cache_hit_tokens": 0,
                    "prompt_tokens": 0,
                    "recovered_checkpointed_tokens": 0,
                    "recovered_visible_tokens": 0,
                    "fault_to_task_completion_samples_s": [],
                    "resume_accept_to_task_completion_samples_s": [],
                },
        )
        task_summary["request_count"] += 1
        task_summary["agent_role"] = (
            task_summary.get("agent_role")
            or entry.get("matched_agent_role")
            or entry.get("agent_role")
        )
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
        if entry.get("fault_to_task_completion_s") is not None:
            task_summary["fault_to_task_completion_samples_s"].append(
                float(entry["fault_to_task_completion_s"])
            )
        if entry.get("resume_accept_to_task_completion_s") is not None:
            task_summary["resume_accept_to_task_completion_samples_s"].append(
                float(entry["resume_accept_to_task_completion_s"])
            )

    cache_hit_job_ids = {key[0] for key in backup_cache_hits_by_task}
    for record in trace_records:
        job_id = str(record.get("topic_id", ""))
        if job_id not in impacted_job_ids and job_id not in cache_hit_job_ids:
            continue
        for timing in record.get("agent_timings", []):
            agent_name = str(timing.get("agent", "") or "")
            task_label = AGENT_TO_TASK_LABEL.get(agent_name)
            if not task_label:
                continue
            key = (job_id, task_label)
            task_summary = per_task.setdefault(
                key,
                {
                    "job_id": job_id,
                    "task_label": task_label,
                    "agent_role": agent_name,
                    "request_count": 0,
                    "resumed_requests": 0,
                    "aborted_requests": 0,
                    "backup_cache_hit_tokens": 0,
                    "prompt_tokens": 0,
                    "recovered_checkpointed_tokens": 0,
                    "recovered_visible_tokens": 0,
                    "fault_to_task_completion_samples_s": [],
                    "resume_accept_to_task_completion_samples_s": [],
                },
            )
            task_summary["agent_role"] = task_summary.get("agent_role") or agent_name
            task_summary["backup_cache_hit_tokens"] = int(
                backup_cache_hits_by_task.get((job_id, task_label), 0) or 0
            )
            task_summary["prompt_tokens"] += int(timing.get("prompt_tokens", 0) or 0)

    for task_summary in per_task.values():
        prompt_tokens = int(task_summary.get("prompt_tokens", 0) or 0)
        hit_tokens = int(task_summary.get("backup_cache_hit_tokens", 0) or 0)
        task_summary["prefill_cache_hit_rate"] = (
            round(hit_tokens / prompt_tokens, 4) if prompt_tokens > 0 else None
        )
        fault_samples = task_summary["fault_to_task_completion_samples_s"]
        resume_samples = task_summary["resume_accept_to_task_completion_samples_s"]
        task_summary["avg_fault_to_task_completion_s"] = (
            round(sum(fault_samples) / len(fault_samples), 3) if fault_samples else None
        )
        task_summary["avg_resume_accept_to_task_completion_s"] = (
            round(sum(resume_samples) / len(resume_samples), 3)
            if resume_samples
            else None
        )

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
    task_instances = build_task_instances(trace_records)
    fault_events = build_fault_events(benchmark_events)
    attach_request_recovery_context(rid_summary, task_instances, fault_events)
    impacted_job_ids = build_impacted_job_ids(
        benchmark_events, rid_summary, job_windows
    )
    (
        backup_cache_hits_by_job,
        backup_cache_hits_by_task,
    ) = build_trace_backup_cache_hits(trace_records)

    jobs = build_jobs_output(
        trace_records,
        rid_summary,
        set(impacted_job_ids),
        backup_cache_hits_by_job,
    )
    tasks = build_tasks_output(
        trace_records,
        rid_summary,
        set(impacted_job_ids),
        backup_cache_hits_by_task,
    )

    trace_meta: dict[str, str] = {}
    if trace_records and isinstance(trace_records[0], dict):
        r0 = trace_records[0]
        for key in ("experiment_mode", "server_topology"):
            val = r0.get(key)
            if val:
                trace_meta[key] = str(val)

    payload = {
        "trace_meta": trace_meta,
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
