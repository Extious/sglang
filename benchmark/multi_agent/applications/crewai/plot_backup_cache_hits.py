#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


TASK_LABEL_ORDER = [
    "Phase 1 / Planning",
    "Phase 2 / Data Collection",
    "Phase 2 / Deep Analysis (critical)",
    "Phase 2 / Trend Scan",
    "Phase 2 / Risk Assessment",
    "Phase 3 / Synthesis",
]

TASK_LABEL_DISPLAY = {
    "Phase 1 / Planning": "Planning",
    "Phase 2 / Data Collection": "Data Collection",
    "Phase 2 / Deep Analysis (critical)": "Deep Analysis",
    "Phase 2 / Trend Scan": "Trend Scan",
    "Phase 2 / Risk Assessment": "Risk Assessment",
    "Phase 3 / Synthesis": "Synthesis",
}

TASK_COLORS = {
    "Phase 1 / Planning": "#4c78a8",
    "Phase 2 / Data Collection": "#f58518",
    "Phase 2 / Deep Analysis (critical)": "#54a24b",
    "Phase 2 / Trend Scan": "#e45756",
    "Phase 2 / Risk Assessment": "#b279a2",
    "Phase 3 / Synthesis": "#72b7b2",
}

METRIC_KEY = "backup_cache_hit_tokens"
CHART_TITLE = "KevlarFlow-style backup (HiCache peer / L3) reuse prompt tokens by task"
CHART_YLABEL = "Backup cache-hit prompt tokens"
TOTAL_LABEL = "Total backup cache-hit tokens"
EMPTY_TEXT = "No backup cache-hit tokens were recorded."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot HiCache storage (DP KV backup / L3) reuse by task from "
            "internal_failover_metrics.json."
        )
    )
    parser.add_argument(
        "--failover-metrics",
        type=Path,
        required=True,
        help="Path to internal_failover_metrics.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the output figure.",
    )
    return parser.parse_args()


def topic_sort_key(job_id: str) -> tuple[int, str]:
    if job_id.isdigit():
        return (0, f"{int(job_id):010d}")
    return (1, job_id)


def load_metrics(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("failover metrics must be a JSON object")
    return payload


def build_task_matrix(
    metrics: dict,
) -> tuple[list[str], dict[str, list[int]], list[int]]:
    impacted_jobs = [
        job
        for job in metrics.get("jobs", [])
        if job.get("impacted") or job.get(METRIC_KEY, 0)
    ]
    impacted_jobs.sort(key=lambda item: topic_sort_key(str(item.get("job_id", ""))))
    job_ids = [str(job.get("job_id", "")) for job in impacted_jobs]
    index = {job_id: idx for idx, job_id in enumerate(job_ids)}

    per_task = {task: [0] * len(job_ids) for task in TASK_LABEL_ORDER}
    totals = [0] * len(job_ids)
    for item in metrics.get("tasks", []):
        task_label = item.get("task_label")
        job_id = str(item.get("job_id", ""))
        if task_label not in per_task or job_id not in index:
            continue
        value = int(item.get(METRIC_KEY, 0) or 0)
        row = index[job_id]
        per_task[task_label][row] += value
        totals[row] += value

    return job_ids, per_task, totals


def plot_chart(
    job_ids: list[str],
    per_task: dict[str, list[int]],
    totals: list[int],
    output_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(13, 5.5))

    if not job_ids:
        ax.text(
            0.5, 0.5, EMPTY_TEXT,
            ha="center", va="center", transform=ax.transAxes,
            fontsize=12, color="gray",
        )
        ax.set_axis_off()
    else:
        x = list(range(len(job_ids)))
        tick_step = max(1, len(job_ids) // 15)
        tick_positions = x[::tick_step]
        tick_labels = job_ids[::tick_step]

        bottoms = [0] * len(job_ids)
        for task_label in TASK_LABEL_ORDER:
            values = per_task[task_label]
            if not any(values):
                continue
            ax.bar(
                x, values, bottom=bottoms,
                color=TASK_COLORS[task_label],
                label=TASK_LABEL_DISPLAY[task_label],
                width=0.7, edgecolor="white", linewidth=0.5,
            )
            bottoms = [b + v for b, v in zip(bottoms, values)]

        ax.plot(
            x, totals,
            color="#222222", marker="o", linewidth=1.4, markersize=4,
            label=TOTAL_LABEL,
        )
        ax.set_ylabel(CHART_YLABEL)
        ax.set_title(CHART_TITLE)
        ax.legend(loc="upper left", frameon=True, ncol=2)
        ax.set_xlabel("Job ID")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    metrics = load_metrics(args.failover_metrics)
    job_ids, per_task, totals = build_task_matrix(metrics)
    plot_chart(job_ids, per_task, totals, args.output)


if __name__ == "__main__":
    main()
