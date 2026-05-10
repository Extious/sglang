#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


BASELINE_COLOR = "#c0392b"
BACKUP_COLOR = "#2471a3"
FAULT_BG = "#fdf2e9"
FAULT_BORDER = "#d35400"
LOCAL_COLOR = "#9ecae9"
REMOTE_COLOR = "#2471a3"
RECOMPUTE_COLOR = "#d9d9d9"

EXPOSURE_ORDER = ["Failure-unaffected", "Failure-affected"]


@dataclass(frozen=True)
class FixedRecord:
    context: str
    strategy: str
    exposure: str
    job_id: str
    completion_time_s: float
    prompt_tokens: int
    local_reuse_tokens: int
    remote_reuse_tokens: int

    @property
    def recomputed_tokens(self) -> int:
        return max(
            int(self.prompt_tokens) - int(self.local_reuse_tokens) - int(self.remote_reuse_tokens),
            0,
        )

    @property
    def cache_hit_ratio(self) -> float:
        if self.prompt_tokens <= 0:
            return 0.0
        return (self.local_reuse_tokens + self.remote_reuse_tokens) / float(self.prompt_tokens)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create synthesis-style figures for fixed-request baseline vs remote-backup "
            "failover experiments."
        )
    )
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--context-label", type=str, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--baseline-label", type=str, default="Baseline")
    parser.add_argument("--backup-label", type=str, default="Remote Backup")
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
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def topic_sort_key(job_id: str) -> tuple[int, str]:
    if job_id.isdigit():
        return (0, f"{int(job_id):010d}")
    return (1, job_id)


def classify_exposure(cache_row: dict[str, str] | None) -> str:
    row = cache_row or {}
    remote_reuse = int(row.get("remote_match") or 0)
    affected = truthy(row.get("is_failover_retried")) or remote_reuse > 0
    return "Failure-affected" if affected else "Failure-unaffected"


def build_cache_map(cache_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    cache_map: dict[str, dict[str, str]] = {}
    for row in cache_rows:
        job_id = str(row.get("job_id", "")).strip()
        if job_id:
            cache_map[job_id] = row
    return cache_map


def load_single_experiment(
    directory: Path,
    *,
    context_label: str,
    strategy_label: str,
) -> list[FixedRecord]:
    task_rows = read_csv(directory / "task_summary.csv")
    cache_map = build_cache_map(read_csv(directory / "cache_hits.csv"))

    request_records: list[FixedRecord] = []

    for row in task_rows:
        job_id = str(row.get("job_id", "")).strip()
        if not job_id:
            continue
        cache_row = cache_map.get(job_id, {})
        remote_reuse = int(cache_row.get("remote_match") or 0)
        local_reuse = int(cache_row.get("l1_match") or 0) + int(cache_row.get("l2_match") or 0)
        request_records.append(
            FixedRecord(
                context=context_label,
                strategy=strategy_label,
                exposure=classify_exposure(cache_row),
                job_id=job_id,
                completion_time_s=float(row["task_completion_time_s"]),
                prompt_tokens=int(row["prompt_tokens"]),
                local_reuse_tokens=local_reuse,
                remote_reuse_tokens=remote_reuse,
            )
        )
    return request_records


def load_records(
    *,
    baseline_dir: Path,
    backup_dir: Path,
    context_label: str,
    baseline_label: str,
    backup_label: str,
) -> list[FixedRecord]:
    baseline_requests = load_single_experiment(
        baseline_dir,
        context_label=context_label,
        strategy_label=baseline_label,
    )
    backup_requests = load_single_experiment(
        backup_dir,
        context_label=context_label,
        strategy_label=backup_label,
    )
    return baseline_requests + backup_requests


def group_records(records: list[object]) -> dict[tuple[str, str, str], list[object]]:
    grouped: dict[tuple[str, str, str], list[object]] = {}
    for record in records:
        key = (record.context, record.exposure, record.strategy)
        grouped.setdefault(key, []).append(record)
    return grouped


def collect_values(
    grouped: dict[tuple[str, str, str], list[object]],
    context: str,
    exposure: str,
    strategy: str,
    value_fn: Callable[[object], float],
) -> list[float]:
    return [value_fn(record) for record in grouped.get((context, exposure, strategy), [])]


def draw_box_with_points(
    axis: plt.Axes,
    position: float,
    box_values: list[float],
    point_values: list[float],
    color: str,
    marker: str,
    seed: int,
) -> None:
    if not box_values:
        return
    axis.boxplot(
        [box_values],
        positions=[position],
        widths=0.28,
        patch_artist=True,
        showmeans=False,
        showfliers=False,
        boxprops={"facecolor": color, "alpha": 0.30, "edgecolor": color, "linewidth": 1.1},
        medianprops={"color": color, "linewidth": 1.4},
        whiskerprops={"color": color, "linewidth": 1.0},
        capprops={"color": color, "linewidth": 1.0},
    )
    if not point_values:
        return
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-0.05, 0.05, size=len(point_values))
    axis.scatter(
        np.full(len(point_values), position) + jitter,
        point_values,
        s=26,
        marker=marker,
        facecolor=color,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.9,
        zorder=3,
    )


def annotate_group_counts(
    axis: plt.Axes,
    position: float,
    values: list[float],
    y_top: float,
    color: str,
) -> None:
    if not values:
        return
    axis.text(position, y_top, f"n={len(values)}", color=color, fontsize=7, ha="center", va="bottom")


def draw_latency_panel(
    axis: plt.Axes,
    grouped: dict[tuple[str, str, str], list[object]],
    *,
    context_label: str,
    baseline_label: str,
    backup_label: str,
    value_fn: Callable[[object], float],
    ylabel: str,
    improvement_label: str,
) -> None:
    x_centers = np.arange(len(EXPOSURE_ORDER), dtype=float)
    offset = 0.18

    axis.axvspan(0.5, 1.5, color=FAULT_BG, alpha=0.70, zorder=0)

    group_maxima: list[float] = []
    for group_index, exposure in enumerate(EXPOSURE_ORDER):
        for strategy, signed_offset, color, marker in [
            (baseline_label, -offset, BASELINE_COLOR, "o"),
            (backup_label, offset, BACKUP_COLOR, "s"),
        ]:
            values = collect_values(grouped, context_label, exposure, strategy, value_fn)
            draw_box_with_points(
                axis,
                x_centers[group_index] + signed_offset,
                values,
                values,
                color,
                marker,
                seed=1000 + group_index * 10 + (0 if strategy == baseline_label else 1),
            )
            if values:
                group_maxima.append(max(values))

    y_max = max(group_maxima) if group_maxima else 1.0
    count_y = y_max * 1.02
    for group_index, exposure in enumerate(EXPOSURE_ORDER):
        for strategy, signed_offset, color in [
            (baseline_label, -offset, BASELINE_COLOR),
            (backup_label, offset, BACKUP_COLOR),
        ]:
            values = collect_values(grouped, context_label, exposure, strategy, value_fn)
            annotate_group_counts(
                axis,
                x_centers[group_index] + signed_offset,
                values,
                count_y,
                color,
            )

    baseline_affected = collect_values(grouped, context_label, "Failure-affected", baseline_label, value_fn)
    backup_affected = collect_values(grouped, context_label, "Failure-affected", backup_label, value_fn)
    if baseline_affected and backup_affected:
        baseline_median = statistics.median(baseline_affected)
        backup_median = statistics.median(backup_affected)
        if baseline_median > 0:
            reduction = 100.0 * (baseline_median - backup_median) / baseline_median
            local_group_max = max(baseline_affected + backup_affected)
            axis.text(
                1.0,
                local_group_max + y_max * 0.08,
                f"{reduction:.0f}% lower\n{improvement_label}",
                color=BACKUP_COLOR,
                ha="center",
                va="bottom",
                fontsize=8,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": BACKUP_COLOR,
                    "alpha": 0.90,
                },
            )

    axis.set_ylabel(ylabel)
    axis.set_xlabel("Fault Status")
    axis.set_xticks(x_centers)
    axis.set_xticklabels([f"{context_label}\nUnaffected", f"{context_label}\nAffected"])
    axis.grid(axis="y", linestyle="--", alpha=0.45)
    axis.set_xlim(-0.55, 1.55)
    axis.set_ylim(0, y_max * 1.24)
    axis.text(1.0, y_max * 1.10, "Affected", ha="center", va="bottom", fontsize=8, color=FAULT_BORDER)
    axis.legend(
        handles=[
            mpatches.Patch(facecolor=BASELINE_COLOR, edgecolor=BASELINE_COLOR, alpha=0.30, label=baseline_label),
            mpatches.Patch(facecolor=BACKUP_COLOR, edgecolor=BACKUP_COLOR, alpha=0.30, label=backup_label),
            mpatches.Patch(facecolor=FAULT_BG, edgecolor=FAULT_BORDER, alpha=0.70, label="Affected"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        framealpha=0.95,
    )


def mean_or_zero(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def draw_prefill_panel(
    axis: plt.Axes,
    grouped: dict[tuple[str, str, str], list[object]],
    *,
    context_label: str,
    baseline_label: str,
    backup_label: str,
) -> None:
    positions = np.array([0.0, 0.9], dtype=float)
    pairs = [(baseline_label, 0.0), (backup_label, 0.9)]

    local_vals = []
    remote_vals = []
    recompute_vals = []
    cache_hit_labels = []
    xticklabels = []

    for strategy, _ in pairs:
        records = grouped.get((context_label, "Failure-affected", strategy), [])
        local_vals.append(mean_or_zero([record.local_reuse_tokens for record in records]))
        remote_vals.append(mean_or_zero([record.remote_reuse_tokens for record in records]))
        recompute_vals.append(mean_or_zero([record.recomputed_tokens for record in records]))
        cache_hit_labels.append(mean_or_zero([record.cache_hit_ratio for record in records]))
        xticklabels.append(f"{strategy}\n(n={len(records)})")

    axis.bar(positions, local_vals, width=0.56, color=LOCAL_COLOR, label="Local reuse")
    axis.bar(positions, remote_vals, width=0.56, bottom=local_vals, color=REMOTE_COLOR, label="Remote reuse")
    axis.bar(
        positions,
        recompute_vals,
        width=0.56,
        bottom=np.array(local_vals) + np.array(remote_vals),
        color=RECOMPUTE_COLOR,
        label="Recompute",
    )

    total_vals = np.array(local_vals) + np.array(remote_vals) + np.array(recompute_vals)
    ymax = float(np.max(total_vals)) if np.any(total_vals) else 1.0
    for pos, hit_ratio, total in zip(positions, cache_hit_labels, total_vals):
        if total <= 0:
            continue
        axis.text(pos, total + ymax * 0.02, f"{hit_ratio * 100:.1f}% hit", ha="center", va="bottom", fontsize=8)

    axis.set_xticks(positions)
    axis.set_xticklabels(xticklabels)
    axis.set_ylabel("Avg. Affected Prefill Tokens")
    axis.set_xlabel(f"System ({context_label})")
    axis.grid(axis="y", linestyle="--", alpha=0.45)
    axis.set_ylim(0, ymax * 1.22)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, framealpha=0.95)


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".pdf", ".png"]:
        fig.savefig(output_base.with_suffix(suffix))
    plt.close(fig)


def create_figures(
    request_records: list[FixedRecord],
    *,
    context_label: str,
    baseline_label: str,
    backup_label: str,
    output_prefix: Path,
) -> None:
    grouped_requests = group_records(list(request_records))

    fig, axis = plt.subplots(1, 1, figsize=(7.2, 4.8), constrained_layout=True)
    draw_latency_panel(
        axis,
        grouped_requests,
        context_label=context_label,
        baseline_label=baseline_label,
        backup_label=backup_label,
        value_fn=lambda record: record.completion_time_s,
        ylabel="Request Latency (s)",
        improvement_label="median latency",
    )
    axis.set_title(
        "Remote backup lowers latency\n"
        "under failover",
        pad=10,
    )
    save_figure(fig, output_prefix.parent / f"{output_prefix.name}_request_latency")

    fig, axis = plt.subplots(1, 1, figsize=(6.6, 4.8), constrained_layout=True)
    draw_prefill_panel(
        axis,
        grouped_requests,
        context_label=context_label,
        baseline_label=baseline_label,
        backup_label=backup_label,
    )
    axis.set_title(
        "Remote backup shifts prefill\n"
        "from recompute to reuse",
        pad=10,
    )
    save_figure(fig, output_prefix.parent / f"{output_prefix.name}_prefill")


def main() -> None:
    args = parse_args()
    apply_style()
    request_records = load_records(
        baseline_dir=args.baseline_dir,
        backup_dir=args.backup_dir,
        context_label=args.context_label,
        baseline_label=args.baseline_label,
        backup_label=args.backup_label,
    )
    if not request_records:
        raise SystemExit("No fixed-request records were found.")
    create_figures(
        request_records,
        context_label=args.context_label,
        baseline_label=args.baseline_label,
        backup_label=args.backup_label,
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    main()
