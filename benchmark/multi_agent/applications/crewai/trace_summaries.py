# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.

"""Write job/task CSV summaries from CrewAI trace JSON."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

AGENT_TO_TASK = {
    "Planning Coordinator": "Phase 1 / Planning",
    "Data Collector": "Phase 2 / Data Collection",
    "Deep Analyst": "Phase 2 / Deep Analysis (critical)",
    "Trend Scout": "Phase 2 / Trend Scan",
    "Risk Assessor": "Phase 2 / Risk Assessment",
    "Report Synthesizer": "Phase 3 / Synthesis",
}


def write_job_summary_csv(trace_path: Path, csv_path: Path) -> None:
    if not trace_path.is_file():
        return
    records: list[dict[str, Any]] = json.loads(
        trace_path.read_text(encoding="utf-8")
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "job_id",
                "job",
                "year",
                "worker_id",
                "status",
                "job_completion_time_s",
                "llm_calls",
                "prefill_cached_tokens",
                "total_tokens",
                "error",
            ]
        )
        for item in records:
            summary = item.get("summary", {}) or {}
            writer.writerow(
                [
                    item.get("topic_id", ""),
                    item.get("topic", ""),
                    item.get("year", ""),
                    item.get("worker_id", ""),
                    item.get("status", ""),
                    summary.get("wall_time_s", ""),
                    summary.get("llm_calls", ""),
                    summary.get("prefill_cached_tokens", ""),
                    summary.get("total_tokens", ""),
                    item.get("error", "") or "",
                ]
            )


def write_task_summary_csv(trace_path: Path, csv_path: Path) -> None:
    if not trace_path.is_file():
        return
    records: list[dict[str, Any]] = json.loads(
        trace_path.read_text(encoding="utf-8")
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "job_id",
                "job",
                "year",
                "worker_id",
                "job_status",
                "agent",
                "task_label",
                "task_completion_time_s",
                "llm_calls",
                "prompt_tokens",
                "cached_prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "prefill_cache_hit_rate",
                "error",
            ]
        )
        for item in records:
            for timing in item.get("agent_timings", []):
                agent = timing.get("agent", "") or ""
                writer.writerow(
                    [
                        item.get("topic_id", ""),
                        item.get("topic", ""),
                        item.get("year", ""),
                        item.get("worker_id", ""),
                        item.get("status", ""),
                        agent,
                        AGENT_TO_TASK.get(agent, agent),
                        timing.get("duration_s", ""),
                        timing.get("llm_calls", ""),
                        timing.get("prompt_tokens", ""),
                        timing.get("cached_prompt_tokens", ""),
                        timing.get("completion_tokens", ""),
                        timing.get("total_tokens", ""),
                        timing.get("prefill_cache_hit_rate", ""),
                        timing.get("error", "") or "",
                    ]
                )
