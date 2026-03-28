#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


BASELINE_COLOR = "#c0392b"
TREATMENT_COLOR = "#2471a3"
FAULT_BG = "#fdf2e9"
FAULT_BORDER = "#d35400"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare routerless internal failover experiment outputs."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--baseline-label", type=str, default="No Backup")
    parser.add_argument("--treatment-label", type=str, default="With Backup")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 240,
            "savefig.dpi": 240,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def topic_sort_key(job_id: str) -> tuple[int, str]:
    if job_id.isdigit():
        return (0, f"{int(job_id):010d}")
    return (1, job_id)


def load_trace(path: Path) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Invalid trace file: {path}")
    return sorted(records, key=lambda item: topic_sort_key(str(item.get("topic_id", ""))))


def load_metrics(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid metrics file: {path}")
    return payload


def wall_time_map(trace: list[dict]) -> dict[str, float]:
    result: dict[str, float] = {}
    for record in trace:
        job_id = str(record.get("topic_id", ""))
        wall_time = record.get("summary", {}).get("wall_time_s")
        if record.get("status") == "success" and wall_time is not None:
            result[job_id] = float(wall_time)
        else:
            result[job_id] = float("nan")
    return result


def union_job_ids(*traces: list[dict]) -> list[str]:
    job_ids: set[str] = set()
    for trace in traces:
        for record in trace:
            job_ids.add(str(record.get("topic_id", "")))
    return sorted(job_ids, key=topic_sort_key)


def fault_window(job_ids: list[str], impacted: set[str]) -> tuple[int, int]:
    indices = [idx for idx, job_id in enumerate(job_ids) if job_id in impacted]
    if not indices:
        return (-1, -1)
    return min(indices), max(indices)


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.08,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="right",
    )


def panel_wall_time(
    axis: plt.Axes,
    job_ids: list[str],
    baseline_trace: list[dict],
    treatment_trace: list[dict],
    impacted: set[str],
    baseline_label: str,
    treatment_label: str,
) -> None:
    x = np.arange(len(job_ids))
    baseline_wt = wall_time_map(baseline_trace)
    treatment_wt = wall_time_map(treatment_trace)
    baseline_values = [baseline_wt.get(job_id, float("nan")) for job_id in job_ids]
    treatment_values = [treatment_wt.get(job_id, float("nan")) for job_id in job_ids]

    lo, hi = fault_window(job_ids, impacted)
    if lo >= 0:
        axis.axvspan(
            lo - 0.5,
            hi + 0.5,
            facecolor=FAULT_BG,
            edgecolor=FAULT_BORDER,
            linewidth=0.8,
            linestyle="--",
            zorder=0,
            label="Fault-impacted window",
        )

    axis.plot(x, baseline_values, color=BASELINE_COLOR, marker="o", label=baseline_label)
    axis.plot(x, treatment_values, color=TREATMENT_COLOR, marker="s", label=treatment_label)
    axis.set_ylabel("Completion Time (s)")
    axis.set_xlabel("Job ID")
    step = max(1, len(job_ids) // 18)
    axis.set_xticks(x[::step])
    axis.set_xticklabels([job_ids[idx] for idx in range(0, len(job_ids), step)], rotation=45, ha="right")
    axis.grid(axis="y", linestyle="--", alpha=0.5)
    axis.legend(loc="upper left", framealpha=0.92)
    add_panel_label(axis, "(a)")


def panel_cdf(
    axis: plt.Axes,
    baseline_trace: list[dict],
    treatment_trace: list[dict],
    baseline_label: str,
    treatment_label: str,
) -> None:
    baseline_values = sorted(
        value for value in wall_time_map(baseline_trace).values() if not math.isnan(value)
    )
    treatment_values = sorted(
        value for value in wall_time_map(treatment_trace).values() if not math.isnan(value)
    )

    if baseline_values:
        y = np.arange(1, len(baseline_values) + 1) / len(baseline_values)
        axis.step(
            baseline_values,
            y,
            where="post",
            color=BASELINE_COLOR,
            linewidth=1.4,
            label=f"{baseline_label} (n={len(baseline_values)})",
        )
    if treatment_values:
        y = np.arange(1, len(treatment_values) + 1) / len(treatment_values)
        axis.step(
            treatment_values,
            y,
            where="post",
            color=TREATMENT_COLOR,
            linewidth=1.4,
            label=f"{treatment_label} (n={len(treatment_values)})",
        )

    axis.set_xlabel("Completion Time (s)")
    axis.set_ylabel("CDF")
    axis.set_ylim(0, 1.05)
    axis.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    axis.grid(True, linestyle="--", alpha=0.5)
    axis.legend(loc="lower right", framealpha=0.92)
    add_panel_label(axis, "(b)")


def impacted_wall_times(trace: list[dict], impacted: set[str]) -> list[float]:
    wt_map = wall_time_map(trace)
    return [
        wt_map[job_id]
        for job_id in sorted(impacted, key=topic_sort_key)
        if job_id in wt_map and not math.isnan(wt_map[job_id])
    ]


def panel_impacted_box(
    axis: plt.Axes,
    baseline_trace: list[dict],
    treatment_trace: list[dict],
    impacted: set[str],
    baseline_label: str,
    treatment_label: str,
) -> None:
    baseline_values = impacted_wall_times(baseline_trace, impacted)
    treatment_values = impacted_wall_times(treatment_trace, impacted)

    if not baseline_values and not treatment_values:
        axis.text(
            0.5,
            0.5,
            "No impacted jobs",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="gray",
        )
        add_panel_label(axis, "(c)")
        return

    data = []
    labels = []
    colors = []
    if baseline_values:
        data.append(baseline_values)
        labels.append(baseline_label)
        colors.append(BASELINE_COLOR)
    if treatment_values:
        data.append(treatment_values)
        labels.append(treatment_label)
        colors.append(TREATMENT_COLOR)

    box = axis.boxplot(
        data,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 4},
        medianprops={"color": "black", "linewidth": 1.0},
        flierprops={"marker": "x", "markersize": 4},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axis.set_xticklabels(labels)
    axis.set_ylabel("Completion Time (s)")
    axis.set_title("Fault-impacted jobs")
    axis.grid(axis="y", linestyle="--", alpha=0.5)
    add_panel_label(axis, "(c)")


def recovered_token_map(metrics: dict) -> dict[str, int]:
    result = {}
    for job in metrics.get("jobs", []):
        job_id = str(job.get("job_id", ""))
        result[job_id] = int(job.get("recovered_checkpointed_tokens", 0) or 0)
    return result


def panel_recovery(
    axis: plt.Axes,
    job_ids: list[str],
    baseline_metrics: dict,
    treatment_metrics: dict,
    baseline_label: str,
    treatment_label: str,
) -> None:
    baseline_tokens = recovered_token_map(baseline_metrics)
    treatment_tokens = recovered_token_map(treatment_metrics)
    values = [
        max(baseline_tokens.get(job_id, 0), treatment_tokens.get(job_id, 0))
        for job_id in job_ids
    ]
    if not any(values):
        axis.text(
            0.5,
            0.5,
            "No recovery tokens recorded",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="gray",
        )
        add_panel_label(axis, "(d)")
        return

    x = np.arange(len(job_ids))
    width = 0.38
    baseline_vals = [baseline_tokens.get(job_id, 0) for job_id in job_ids]
    treatment_vals = [treatment_tokens.get(job_id, 0) for job_id in job_ids]
    axis.bar(x - width / 2, baseline_vals, width, color=BASELINE_COLOR, label=baseline_label)
    axis.bar(x + width / 2, treatment_vals, width, color=TREATMENT_COLOR, label=treatment_label)
    axis.set_xlabel("Job ID")
    axis.set_ylabel("Recovered checkpointed tokens")
    step = max(1, len(job_ids) // 18)
    axis.set_xticks(x[::step])
    axis.set_xticklabels([job_ids[idx] for idx in range(0, len(job_ids), step)], rotation=45, ha="right")
    axis.grid(axis="y", linestyle="--", alpha=0.5)
    axis.legend(loc="upper left", framealpha=0.92)
    add_panel_label(axis, "(d)")


def main() -> None:
    args = parse_args()
    apply_style()

    baseline_trace = load_trace(args.baseline / "trace_log.json")
    treatment_trace = load_trace(args.treatment / "trace_log.json")
    baseline_metrics = load_metrics(args.baseline / "internal_failover_metrics.json")
    treatment_metrics = load_metrics(args.treatment / "internal_failover_metrics.json")

    job_ids = union_job_ids(baseline_trace, treatment_trace)
    impacted = set(baseline_metrics.get("impacted_job_ids", [])) | set(
        treatment_metrics.get("impacted_job_ids", [])
    )
    impacted_job_ids = [job_id for job_id in job_ids if job_id in impacted]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    panel_wall_time(
        axes[0, 0],
        job_ids,
        baseline_trace,
        treatment_trace,
        impacted,
        args.baseline_label,
        args.treatment_label,
    )
    panel_cdf(
        axes[0, 1],
        baseline_trace,
        treatment_trace,
        args.baseline_label,
        args.treatment_label,
    )
    panel_impacted_box(
        axes[1, 0],
        baseline_trace,
        treatment_trace,
        impacted,
        args.baseline_label,
        args.treatment_label,
    )
    panel_recovery(
        axes[1, 1],
        impacted_job_ids,
        baseline_metrics,
        treatment_metrics,
        args.baseline_label,
        args.treatment_label,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output)
    plt.close(fig)


if __name__ == "__main__":
    main()
