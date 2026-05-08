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

CONTEXT_ORDER = ["20k", "30k"]
EXPOSURE_ORDER = ["Failure-unaffected", "Failure-affected"]
STRATEGY_ORDER = ["Baseline", "Remote Backup"]


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    context: str
    strategy: str


@dataclass
class SynthesisRecord:
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


@dataclass
class JobRecord:
    context: str
    strategy: str
    exposure: str
    job_id: str
    completion_time_s: float


@dataclass(frozen=True)
class LatencyVariant:
    code: str
    description: str
    iqr_multiplier: float | None
    scatter_only: bool = False


@dataclass(frozen=True)
class LatencyDrawValues:
    box_values: list[float]
    scatter_values: list[float]
    summary_values: list[float]
    display_values: list[float]


LATENCY_VARIANTS = (
    LatencyVariant("A", "Variant A: 3.0 x IQR filter", 3.0),
    LatencyVariant("B", "Variant B: 2.0 x IQR filter", 2.0),
    LatencyVariant("C", "Variant C: 1.5 x IQR for scatter only", 1.5, scatter_only=True),
    LatencyVariant("D", "Variant D: no filtering", None),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create paper-style figures for synthesis latency, affected-synthesis "
            "prefill sources, and end-to-end job completion time."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Root directory containing baseline-20k, baseline-30k, remote_backup-20k, remote_backup-30k.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Output path prefix without file extension. The script saves multiple .pdf/.png files.",
    )
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


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def classify_exposure(cache_row: dict[str, str] | None) -> str:
    row = cache_row or {}
    remote_reuse = int(row.get("remote_match") or 0)
    affected = truthy(row.get("is_failover_retried")) or remote_reuse > 0
    return "Failure-affected" if affected else "Failure-unaffected"


def build_cache_map(cache_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    cache_map: dict[str, dict[str, str]] = {}
    for row in cache_rows:
        if row.get("task_label") != "Phase 3 / Synthesis":
            continue
        job_id = str(row.get("job_id", "")).strip()
        if job_id:
            cache_map[job_id] = row
    return cache_map


def load_records(results_root: Path) -> tuple[list[SynthesisRecord], list[JobRecord]]:
    experiments = [
        ExperimentSpec("baseline-20k", "20k", "Baseline"),
        ExperimentSpec("remote_backup-20k", "20k", "Remote Backup"),
        ExperimentSpec("baseline-30k", "30k", "Baseline"),
        ExperimentSpec("remote_backup-30k", "30k", "Remote Backup"),
    ]
    synthesis_records: list[SynthesisRecord] = []
    job_records: list[JobRecord] = []

    for experiment in experiments:
        directory = results_root / experiment.name
        task_rows = read_csv(directory / "task_summary.csv")
        job_rows = read_csv(directory / "job_summary.csv")
        cache_map = build_cache_map(read_csv(directory / "cache_hits.csv"))

        for row in task_rows:
            if row.get("task_label") != "Phase 3 / Synthesis":
                continue

            job_id = str(row.get("job_id", "")).strip()
            cache_row = cache_map.get(job_id, {})
            remote_reuse = int(cache_row.get("remote_match") or 0)
            local_reuse = int(cache_row.get("l1_match") or 0) + int(cache_row.get("l2_match") or 0)

            synthesis_records.append(
                SynthesisRecord(
                    context=experiment.context,
                    strategy=experiment.strategy,
                    exposure=classify_exposure(cache_row),
                    job_id=job_id,
                    completion_time_s=float(row["task_completion_time_s"]),
                    prompt_tokens=int(row["prompt_tokens"]),
                    local_reuse_tokens=local_reuse,
                    remote_reuse_tokens=remote_reuse,
                )
            )

        for row in job_rows:
            job_id = str(row.get("job_id", "")).strip()
            if not job_id:
                continue
            job_records.append(
                JobRecord(
                    context=experiment.context,
                    strategy=experiment.strategy,
                    exposure=classify_exposure(cache_map.get(job_id)),
                    job_id=job_id,
                    completion_time_s=float(row["job_completion_time_s"]),
                )
            )

    return synthesis_records, job_records


def group_records(records: list[object]) -> dict[tuple[str, str, str], list[object]]:
    grouped: dict[tuple[str, str, str], list[object]] = {}
    for record in records:
        key = (record.context, record.exposure, record.strategy)
        grouped.setdefault(key, []).append(record)
    return grouped


def strategy_style(strategy: str) -> dict[str, str]:
    return {
        "Baseline": {"color": BASELINE_COLOR, "marker": "o"},
        "Remote Backup": {"color": BACKUP_COLOR, "marker": "s"},
    }[strategy]


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
    axis.text(
        position,
        y_top,
        f"n={len(values)}",
        color=color,
        fontsize=7,
        ha="center",
        va="bottom",
    )


def filter_latency_outliers(values: list[float], iqr_multiplier: float = 1.5) -> list[float]:
    if len(values) < 4:
        return list(values)

    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr
    return [value for value in values if lower <= value <= upper]


def get_latency_draw_values(values: list[float], variant: LatencyVariant) -> LatencyDrawValues:
    raw_values = list(values)
    if variant.iqr_multiplier is None:
        return LatencyDrawValues(raw_values, raw_values, raw_values, raw_values)

    filtered_values = filter_latency_outliers(raw_values, iqr_multiplier=variant.iqr_multiplier)
    if variant.scatter_only:
        display_values = filtered_values if filtered_values else raw_values
        return LatencyDrawValues(raw_values, filtered_values, raw_values, display_values)

    return LatencyDrawValues(filtered_values, filtered_values, filtered_values, filtered_values)


def collect_values(
    grouped: dict[tuple[str, str, str], list[object]],
    context: str,
    exposure: str,
    strategy: str,
    value_fn: Callable[[object], float],
    filter_outliers: bool = False,
    iqr_multiplier: float = 1.5,
) -> list[float]:
    values = [value_fn(record) for record in grouped.get((context, exposure, strategy), [])]
    return (
        filter_latency_outliers(values, iqr_multiplier=iqr_multiplier)
        if filter_outliers
        else values
    )


def draw_latency_panel(
    axis: plt.Axes,
    grouped: dict[tuple[str, str, str], list[object]],
    value_fn: Callable[[object], float],
    ylabel: str,
    improvement_label: str,
    variant: LatencyVariant,
) -> None:
    x_centers = np.arange(len(CONTEXT_ORDER) * len(EXPOSURE_ORDER), dtype=float)
    offset = 0.18

    axis.axvspan(0.5, 1.5, color=FAULT_BG, alpha=0.70, zorder=0)
    axis.axvspan(2.5, 3.5, color=FAULT_BG, alpha=0.70, zorder=0)
    axis.axvline(1.95, color="#bbbbbb", linewidth=0.9, linestyle="--")

    group_maxima: list[float] = []
    for group_index, (context, exposure) in enumerate(
        [(c, e) for c in CONTEXT_ORDER for e in EXPOSURE_ORDER]
    ):
        for strategy, signed_offset in [("Baseline", -offset), ("Remote Backup", offset)]:
            raw_values = collect_values(grouped, context, exposure, strategy, value_fn)
            values = get_latency_draw_values(raw_values, variant)
            style = strategy_style(strategy)
            draw_box_with_points(
                axis,
                x_centers[group_index] + signed_offset,
                values.box_values,
                values.scatter_values,
                style["color"],
                style["marker"],
                seed=1000 + group_index * 10 + (0 if strategy == "Baseline" else 1),
            )
            if values.display_values:
                group_maxima.append(max(values.display_values))

    y_max = max(group_maxima) if group_maxima else 1.0
    count_y = y_max * 1.02
    for group_index, (context, exposure) in enumerate(
        [(c, e) for c in CONTEXT_ORDER for e in EXPOSURE_ORDER]
    ):
        for strategy, signed_offset in [("Baseline", -offset), ("Remote Backup", offset)]:
            raw_values = collect_values(grouped, context, exposure, strategy, value_fn)
            values = get_latency_draw_values(raw_values, variant)
            annotate_group_counts(
                axis,
                x_centers[group_index] + signed_offset,
                values.summary_values,
                count_y,
                strategy_style(strategy)["color"],
            )

    for group_index, context in zip([1, 3], CONTEXT_ORDER):
        baseline_values = get_latency_draw_values(
            collect_values(grouped, context, "Failure-affected", "Baseline", value_fn),
            variant,
        ).summary_values
        backup_values = get_latency_draw_values(
            collect_values(grouped, context, "Failure-affected", "Remote Backup", value_fn),
            variant,
        ).summary_values
        if baseline_values and backup_values:
            baseline_median = statistics.median(baseline_values)
            backup_median = statistics.median(backup_values)
            if baseline_median > 0:
                reduction = 100.0 * (baseline_median - backup_median) / baseline_median
                local_group_max = max(baseline_values + backup_values)
                text_y = local_group_max + y_max * 0.06
                axis.text(
                    group_index,
                    text_y,
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
    axis.set_xlabel("Context Size and Fault Exposure")
    axis.set_xticks(x_centers)
    axis.set_xticklabels(
        [
            "20k\nUnaffected",
            "20k\nAffected",
            "30k\nUnaffected",
            "30k\nAffected",
        ]
    )
    axis.grid(axis="y", linestyle="--", alpha=0.45)
    axis.set_xlim(-0.55, 3.55)
    axis.set_ylim(0, y_max * 1.22)
    label_y = y_max * 1.09
    axis.text(
        0.5,
        label_y,
        "Failure-affected groups",
        ha="center",
        va="bottom",
        fontsize=8,
        color=FAULT_BORDER,
    )
    axis.text(
        2.5,
        label_y,
        "Failure-affected groups",
        ha="center",
        va="bottom",
        fontsize=8,
        color=FAULT_BORDER,
    )

    legend_handles = [
        mpatches.Patch(facecolor=BASELINE_COLOR, edgecolor=BASELINE_COLOR, alpha=0.30, label="Baseline"),
        mpatches.Patch(facecolor=BACKUP_COLOR, edgecolor=BACKUP_COLOR, alpha=0.30, label="Remote Backup"),
        mpatches.Patch(facecolor=FAULT_BG, edgecolor=FAULT_BORDER, alpha=0.70, label="Failure-affected window"),
    ]
    axis.legend(
        handles=legend_handles,
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
) -> None:
    positions = np.array([0.0, 0.9, 2.3, 3.2], dtype=float)
    pairs = [
        ("20k", "Baseline"),
        ("20k", "Remote Backup"),
        ("30k", "Baseline"),
        ("30k", "Remote Backup"),
    ]

    local_vals = []
    remote_vals = []
    recompute_vals = []
    cache_hit_labels = []
    xticklabels = []

    for context, strategy in pairs:
        records = grouped.get((context, "Failure-affected", strategy), [])
        local_vals.append(mean_or_zero([record.local_reuse_tokens for record in records]))
        remote_vals.append(mean_or_zero([record.remote_reuse_tokens for record in records]))
        recompute_vals.append(mean_or_zero([record.recomputed_tokens for record in records]))
        cache_hit_labels.append(mean_or_zero([record.cache_hit_ratio for record in records]))
        xticklabels.append(f"{strategy}\n(n={len(records)})")

    axis.bar(positions, local_vals, width=0.56, color=LOCAL_COLOR, label="Local reuse (L1/L2)")
    axis.bar(
        positions,
        remote_vals,
        width=0.56,
        bottom=local_vals,
        color=REMOTE_COLOR,
        label="Remote backup reuse",
    )
    axis.bar(
        positions,
        recompute_vals,
        width=0.56,
        bottom=np.array(local_vals) + np.array(remote_vals),
        color=RECOMPUTE_COLOR,
        label="Recomputed prefill",
    )

    total_vals = np.array(local_vals) + np.array(remote_vals) + np.array(recompute_vals)
    ymax = float(np.max(total_vals)) if np.any(total_vals) else 1.0
    for pos, hit_ratio, total in zip(positions, cache_hit_labels, total_vals):
        if total <= 0:
            continue
        axis.text(
            pos,
            total + ymax * 0.02,
            f"{hit_ratio * 100:.1f}% hit",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    axis.axvline(1.6, color="#bbbbbb", linewidth=0.9, linestyle="--")
    axis.text(0.45, ymax * 1.09, "20k context", ha="center", va="bottom")
    axis.text(2.75, ymax * 1.09, "30k context", ha="center", va="bottom")
    axis.set_xticks(positions)
    axis.set_xticklabels(xticklabels)
    axis.set_ylabel("Average Affected-Synthesis Prefill Tokens")
    axis.set_xlabel("System")
    axis.grid(axis="y", linestyle="--", alpha=0.45)
    axis.set_ylim(0, ymax * 1.22)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        framealpha=0.95,
    )


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".pdf", ".png"]:
        fig.savefig(output_base.with_suffix(suffix))
    plt.close(fig)


def create_figures(
    synthesis_records: list[SynthesisRecord],
    job_records: list[JobRecord],
    output_prefix: Path,
) -> None:
    grouped_synthesis = group_records(list(synthesis_records))
    grouped_jobs = group_records(list(job_records))

    for variant in LATENCY_VARIANTS:
        fig, axis = plt.subplots(1, 1, figsize=(7.5, 4.8), constrained_layout=True)
        draw_latency_panel(
            axis,
            grouped_synthesis,
            value_fn=lambda record: record.completion_time_s,
            ylabel="Synthesis Completion Time (s)",
            improvement_label="median synthesis latency",
            variant=variant,
        )
        axis.set_title(
            "Remote backup reduces synthesis latency\n"
            "when failures directly disrupt synthesis",
            pad=10,
        )
        save_figure(
            fig,
            output_prefix.parent / f"{output_prefix.name}_synthesis_latency_{variant.code}",
        )

    fig, axis = plt.subplots(1, 1, figsize=(6.9, 4.8), constrained_layout=True)
    draw_prefill_panel(axis, grouped_synthesis)
    axis.set_title(
        "Remote backup replaces prefill recomputation\nwith remote reuse in failure-affected synthesis",
        pad=10,
    )
    save_figure(fig, output_prefix.parent / f"{output_prefix.name}_synthesis_prefill")

    for variant in LATENCY_VARIANTS:
        fig, axis = plt.subplots(1, 1, figsize=(7.5, 4.8), constrained_layout=True)
        draw_latency_panel(
            axis,
            grouped_jobs,
            value_fn=lambda record: record.completion_time_s,
            ylabel="Job Completion Time (s)",
            improvement_label="median job time",
            variant=variant,
        )
        axis.set_title(
            "Remote backup also lowers end-to-end job time\n"
            "for failure-affected runs",
            pad=10,
        )
        save_figure(
            fig,
            output_prefix.parent / f"{output_prefix.name}_job_latency_{variant.code}",
        )


def main() -> None:
    args = parse_args()
    apply_style()
    synthesis_records, job_records = load_records(args.results_root)
    if not synthesis_records:
        raise SystemExit("No synthesis records were found.")
    if not job_records:
        raise SystemExit("No job records were found.")
    create_figures(synthesis_records, job_records, args.output_prefix)


if __name__ == "__main__":
    main()
