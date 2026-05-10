#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


BASELINE_COLOR = "#c0392b"
BACKUP_COLOR = "#2471a3"
LOCAL_COLOR = "#9ecae9"
REMOTE_COLOR = "#2471a3"
RECOMPUTE_COLOR = "#d9d9d9"
DELTA_AFFECTED_COLOR = "#2471a3"
DELTA_UNAFFECTED_COLOR = "#bdbdbd"


@dataclass(frozen=True)
class FixedJobRecord:
    label: str
    job_id: str
    completion_time_s: float
    prompt_tokens: int
    cached_prompt_tokens: int
    remote_reuse_tokens: int
    affected: bool

    @property
    def local_reuse_tokens(self) -> int:
        return max(self.cached_prompt_tokens - self.remote_reuse_tokens, 0)

    @property
    def recomputed_tokens(self) -> int:
        return max(self.prompt_tokens - self.cached_prompt_tokens, 0)


@dataclass(frozen=True)
class ExperimentSnapshot:
    label: str
    jobs: dict[str, FixedJobRecord]


@dataclass(frozen=True)
class JobComparison:
    job_id: str
    baseline: FixedJobRecord
    backup: FixedJobRecord

    @property
    def affected(self) -> bool:
        return self.baseline.affected or self.backup.affected

    @property
    def completion_delta_s(self) -> float:
        return self.backup.completion_time_s - self.baseline.completion_time_s


@dataclass(frozen=True)
class ComparisonBundle:
    by_job_id: dict[str, JobComparison]
    ordered_job_ids: list[str]
    affected_job_ids: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a comparison figure that highlights the benefit of remote backup "
            "for fixed-request failover experiments."
        )
    )
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-label", default="Baseline")
    parser.add_argument("--backup-label", default="Remote Backup")
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
            "savefig.pad_inches": 0.06,
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def topic_sort_key(job_id: str) -> tuple[int, str]:
    if job_id.isdigit():
        return (0, f"{int(job_id):010d}")
    return (1, job_id)


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def _task_rows_by_job_id(task_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in task_rows:
        job_id = str(row.get("job_id", "")).strip()
        if not job_id:
            continue
        result[job_id] = row
    return result


def _cache_rows_by_job_id(cache_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in cache_rows:
        job_id = str(row.get("job_id", "")).strip()
        if not job_id:
            continue
        result[job_id] = row
    return result


def load_experiment(directory: Path, label: str) -> ExperimentSnapshot:
    task_by_job = _task_rows_by_job_id(read_csv(directory / "task_summary.csv"))
    cache_by_job = _cache_rows_by_job_id(read_csv(directory / "cache_hits.csv"))
    jobs: dict[str, FixedJobRecord] = {}

    for row in read_csv(directory / "job_summary.csv"):
        job_id = str(row.get("job_id", "")).strip()
        if not job_id:
            continue
        task_row = task_by_job.get(job_id, {})
        cache_row = cache_by_job.get(job_id, {})

        cached_prompt_tokens = int(
            task_row.get("cached_prompt_tokens")
            or row.get("prefill_cached_tokens")
            or 0
        )
        prompt_tokens = int(task_row.get("prompt_tokens") or cached_prompt_tokens or 0)
        remote_reuse_tokens = int(cache_row.get("remote_match") or 0)
        affected = truthy(cache_row.get("is_failover_retried")) or remote_reuse_tokens > 0

        jobs[job_id] = FixedJobRecord(
            label=label,
            job_id=job_id,
            completion_time_s=float(row.get("job_completion_time_s") or 0.0),
            prompt_tokens=prompt_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
            remote_reuse_tokens=remote_reuse_tokens,
            affected=affected,
        )

    return ExperimentSnapshot(label=label, jobs=jobs)


def build_comparison(
    baseline_snapshot: ExperimentSnapshot,
    backup_snapshot: ExperimentSnapshot,
) -> ComparisonBundle:
    ordered_job_ids = sorted(
        set(baseline_snapshot.jobs) & set(backup_snapshot.jobs),
        key=topic_sort_key,
    )
    by_job_id = {
        job_id: JobComparison(
            job_id=job_id,
            baseline=baseline_snapshot.jobs[job_id],
            backup=backup_snapshot.jobs[job_id],
        )
        for job_id in ordered_job_ids
    }
    affected_job_ids = [job_id for job_id in ordered_job_ids if by_job_id[job_id].affected]
    return ComparisonBundle(
        by_job_id=by_job_id,
        ordered_job_ids=ordered_job_ids,
        affected_job_ids=affected_job_ids,
    )


def _median(values: list[float]) -> float:
    sorted_values = sorted(values)
    if not sorted_values:
        return 0.0
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def draw_token_recovery_panel(axis: plt.Axes, comparison: ComparisonBundle) -> None:
    if not comparison.affected_job_ids:
        axis.text(0.5, 0.5, "No failover-affected jobs", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
        return

    positions = []
    local_values = []
    remote_values = []
    recompute_values = []
    tick_labels = []
    text_specs: list[tuple[float, float, str]] = []

    x = 0.0
    for job_id in comparison.affected_job_ids:
        record = comparison.by_job_id[job_id]
        baseline_pos = x
        backup_pos = x + 0.38
        positions.extend([baseline_pos, backup_pos])
        local_values.extend([record.baseline.local_reuse_tokens, record.backup.local_reuse_tokens])
        remote_values.extend([record.baseline.remote_reuse_tokens, record.backup.remote_reuse_tokens])
        recompute_values.extend([record.baseline.recomputed_tokens, record.backup.recomputed_tokens])
        tick_labels.extend([f"job {job_id}\nBase", f"job {job_id}\nBackup"])
        if record.backup.remote_reuse_tokens > 0:
            total = (
                record.backup.local_reuse_tokens
                + record.backup.remote_reuse_tokens
                + record.backup.recomputed_tokens
            )
            text_specs.append(
                (backup_pos, total, f"{record.backup.remote_reuse_tokens:,} remote")
            )
        x += 1.2

    positions_arr = np.array(positions)
    local_arr = np.array(local_values)
    remote_arr = np.array(remote_values)
    recompute_arr = np.array(recompute_values)
    total_arr = local_arr + remote_arr + recompute_arr

    axis.bar(positions_arr, local_arr, width=0.28, color=LOCAL_COLOR, label="Local reuse")
    axis.bar(
        positions_arr,
        remote_arr,
        width=0.28,
        bottom=local_arr,
        color=REMOTE_COLOR,
        label="Remote backup reuse",
    )
    axis.bar(
        positions_arr,
        recompute_arr,
        width=0.28,
        bottom=local_arr + remote_arr,
        color=RECOMPUTE_COLOR,
        label="Recomputed prefill",
    )

    ymax = float(np.max(total_arr)) if len(total_arr) else 1.0
    for x_pos, total, label in text_specs:
        axis.text(x_pos, total + ymax * 0.03, label, ha="center", va="bottom", fontsize=8, color=BACKUP_COLOR)

    axis.set_title("Recovered prefill work on failover-affected jobs")
    axis.set_ylabel("Prompt Tokens")
    axis.set_xticks(positions_arr)
    axis.set_xticklabels(tick_labels)
    axis.grid(axis="y", linestyle="--", alpha=0.45)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, framealpha=0.95)


def draw_affected_latency_panel(axis: plt.Axes, comparison: ComparisonBundle) -> None:
    if not comparison.affected_job_ids:
        axis.text(0.5, 0.5, "No failover-affected jobs", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
        return

    x = np.arange(len(comparison.affected_job_ids), dtype=float)
    width = 0.34
    baseline_values = [
        comparison.by_job_id[job_id].baseline.completion_time_s
        for job_id in comparison.affected_job_ids
    ]
    backup_values = [
        comparison.by_job_id[job_id].backup.completion_time_s
        for job_id in comparison.affected_job_ids
    ]
    axis.bar(x - width / 2, baseline_values, width, color=BASELINE_COLOR, label="Baseline")
    axis.bar(x + width / 2, backup_values, width, color=BACKUP_COLOR, label="Remote Backup")

    ymax = max(baseline_values + backup_values) if (baseline_values or backup_values) else 1.0
    for idx, job_id in enumerate(comparison.affected_job_ids):
        pair = comparison.by_job_id[job_id]
        if pair.baseline.completion_time_s <= 0:
            continue
        delta_pct = 100.0 * (
            pair.backup.completion_time_s - pair.baseline.completion_time_s
        ) / pair.baseline.completion_time_s
        axis.text(
            x[idx],
            max(pair.baseline.completion_time_s, pair.backup.completion_time_s) + ymax * 0.03,
            f"{delta_pct:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            color=BACKUP_COLOR if delta_pct < 0 else BASELINE_COLOR,
        )

    axis.set_title("Completion time for failover-affected jobs")
    axis.set_ylabel("Completion Time (s)")
    axis.set_xlabel("Affected Job ID")
    axis.set_xticks(x)
    axis.set_xticklabels([f"job {job_id}" for job_id in comparison.affected_job_ids])
    axis.grid(axis="y", linestyle="--", alpha=0.45)
    axis.legend(loc="upper right", framealpha=0.95)


def draw_delta_panel(axis: plt.Axes, comparison: ComparisonBundle) -> None:
    x = np.arange(len(comparison.ordered_job_ids), dtype=float)
    deltas = [comparison.by_job_id[job_id].completion_delta_s for job_id in comparison.ordered_job_ids]
    colors = [
        DELTA_AFFECTED_COLOR if comparison.by_job_id[job_id].affected else DELTA_UNAFFECTED_COLOR
        for job_id in comparison.ordered_job_ids
    ]
    axis.bar(x, deltas, width=0.62, color=colors)
    axis.axhline(0.0, color="#555555", linewidth=0.9)

    affected_deltas = [
        comparison.by_job_id[job_id].completion_delta_s
        for job_id in comparison.affected_job_ids
    ]
    unaffected_deltas = [
        comparison.by_job_id[job_id].completion_delta_s
        for job_id in comparison.ordered_job_ids
        if not comparison.by_job_id[job_id].affected
    ]
    affected_median = _median(affected_deltas)
    unaffected_median = _median(unaffected_deltas)
    summary = (
        f"Median delta: affected {affected_median:+.2f}s, "
        f"unaffected {unaffected_median:+.2f}s"
    )
    axis.text(
        0.01,
        0.96,
        summary,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.95},
    )

    axis.set_title("Per-job completion delta (backup - baseline)")
    axis.set_ylabel("Delta (s)")
    axis.set_xlabel("Job ID")
    step = max(1, len(comparison.ordered_job_ids) // 18)
    axis.set_xticks(x[::step])
    axis.set_xticklabels(
        [comparison.ordered_job_ids[idx] for idx in range(0, len(comparison.ordered_job_ids), step)],
        rotation=45,
        ha="right",
    )
    axis.grid(axis="y", linestyle="--", alpha=0.45)


def create_figure(comparison: ComparisonBundle, output: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 11.0), constrained_layout=True)
    draw_token_recovery_panel(axes[0], comparison)
    draw_affected_latency_panel(axes[1], comparison)
    draw_delta_panel(axes[2], comparison)
    fig.suptitle(
        "Remote backup advantage during failover\n"
        "Preserved prefill work, affected-job latency, and per-job deltas",
        fontsize=12,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    apply_style()
    comparison = build_comparison(
        load_experiment(args.baseline_dir, label=args.baseline_label),
        load_experiment(args.backup_dir, label=args.backup_label),
    )
    if not comparison.ordered_job_ids:
        raise SystemExit("No overlapping job IDs found between baseline and backup runs.")
    create_figure(comparison, args.output)


if __name__ == "__main__":
    main()
