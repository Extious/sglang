# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.

"""CSV summary writers for the CrewAI client."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, Iterable


CACHE_HIT_HEADER = [
    "job_id", "task_label", "agent_role", "rid",
    "l1_match", "l2_match", "remote_match", "remote_prefetch",
    "reused_device", "reused_host", "reused_storage",
    "is_failover_retried", "pre_failover_output_tokens",
    "pre_failover_backed_up_tokens",
]


def _load_trace_records(trace_path: Path) -> list[dict]:
    if not trace_path.is_file():
        return []
    records = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _write_summary_records(
    records: Iterable[dict],
    csv_path: Path,
    header: list,
    row_fn: Callable,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for item in records:
            for row in row_fn(item):
                writer.writerow(row)


def write_trace_records(records: Iterable[dict], trace_path: Path) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trace_path, "w", encoding="utf-8") as fh:
        for item in records:
            fh.write(json.dumps(item, ensure_ascii=True) + "\n")


def write_cache_hit_records(records: Iterable[dict], csv_path: Path) -> None:
    _write_summary_records(
        records,
        csv_path,
        CACHE_HIT_HEADER,
        lambda item: ([item.get(name, "") for name in CACHE_HIT_HEADER],),
    )


def write_job_summary_records(records: Iterable[dict], csv_path: Path) -> None:
    header = [
        "job_id", "job", "year", "worker_id", "status",
        "job_completion_time_s", "llm_calls",
        "prefill_cached_tokens", "total_tokens", "error",
    ]

    def rows(item):
        s = item.get("summary") or {}
        yield [
            item.get("topic_id", ""), item.get("topic", ""),
            item.get("year", ""), item.get("worker_id", ""),
            item.get("status", ""),
            s.get("wall_time_s", ""), s.get("llm_calls", ""),
            s.get("prefill_cached_tokens", ""), s.get("total_tokens", ""),
            item.get("error") or "",
        ]

    _write_summary_records(records, csv_path, header, rows)


def write_job_summary_csv(trace_path: Path, csv_path: Path) -> None:
    write_job_summary_records(_load_trace_records(trace_path), csv_path)


def write_task_summary_records(
    records: Iterable[dict],
    csv_path: Path,
    role_map: dict[str, str] | None = None,
) -> None:
    if role_map is None:
        cfg_path = Path(__file__).resolve().parent / "crewai_config.json"
        if cfg_path.is_file():
            role_map = json.loads(cfg_path.read_text(encoding="utf-8")).get("role_to_task_label", {})
        else:
            role_map = {}
    assert role_map is not None

    header = [
        "job_id", "job", "year", "worker_id", "job_status",
        "agent", "task_label", "task_completion_time_s",
        "llm_calls", "prompt_tokens", "cached_prompt_tokens",
        "completion_tokens", "total_tokens", "prefill_cache_hit_rate",
        "rid",
        "error",
    ]

    def rows(item):
        timings = item.get("agent_timings") or []
        if timings:
            for timing in timings:
                agent = timing.get("agent") or ""
                rid = timing.get("rid", "")
                if not rid:
                    request_rids = timing.get("request_rids") or []
                    rid = ";".join(str(v) for v in request_rids if str(v))
                yield [
                    item.get("topic_id", ""), item.get("topic", ""),
                    item.get("year", ""), item.get("worker_id", ""),
                    item.get("status", ""), agent,
                    timing.get("task_label") or role_map.get(agent, agent),
                    timing.get("duration_s", ""),
                    timing.get("llm_calls", ""),
                    timing.get("prompt_tokens", ""),
                    timing.get("cached_prompt_tokens", ""),
                    timing.get("completion_tokens", ""),
                    timing.get("total_tokens", ""),
                    timing.get("prefill_cache_hit_rate", ""),
                    rid,
                    timing.get("error") or "",
                ]
        else:
            summary = item.get("summary") or {}
            prompt = summary.get("prompt_tokens", 0)
            cached = summary.get("prefill_cached_tokens", 0)
            completion = summary.get("total_tokens", 0) - prompt
            total = summary.get("total_tokens", 0)
            hit_rate = round(cached / prompt, 4) if prompt else ""
            yield [
                item.get("topic_id", ""), item.get("topic", ""),
                item.get("year", ""), item.get("worker_id", ""),
                item.get("status", ""), "",
                "(aggregated)",
                summary.get("wall_time_s", ""),
                summary.get("llm_calls", ""),
                prompt, cached, completion, total,
                hit_rate,
                "",
                item.get("error") or "",
            ]

    _write_summary_records(records, csv_path, header, rows)


def write_task_summary_csv(
    trace_path: Path,
    csv_path: Path,
    role_map: dict[str, str] | None = None,
) -> None:
    write_task_summary_records(_load_trace_records(trace_path), csv_path, role_map)
