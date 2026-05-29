#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

COMPONENT_KEYS = [
    "waiting_time_s",
    "prefetch_time_s",
    "backup_time_s",
    "inference_time_s",
    "failover_penalty_s",
]

COMPONENT_LABELS = {
    "waiting_time_s": "Queue",
    "prefetch_time_s": "Prefetch",
    "backup_time_s": "Backup",
    "inference_time_s": "Inference",
    "failover_penalty_s": "Failover Penalty",
}

COMPONENT_COLORS = {
    "waiting_time_s": "#4c78a8",
    "prefetch_time_s": "#72b7b2",
    "backup_time_s": "#f58518",
    "inference_time_s": "#54a24b",
    "failover_penalty_s": "#e45756",
}


def build_runtime_series(
    csv_path: str | Path,
) -> tuple[list[str], dict[str, list[float]], list[float], list[bool]]:
    rows_by_request: dict[str, list[dict[str, str]]] = {}
    with Path(csv_path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            request_id = _request_id(row)
            if not request_id:
                continue
            rows_by_request.setdefault(request_id, []).append(row)

    request_ids = sorted(rows_by_request, key=_sort_key)
    components = {key: [] for key in COMPONENT_KEYS}
    total_latency = []
    retried = []
    for request_id in request_ids:
        rows = rows_by_request[request_id]
        for key in COMPONENT_KEYS:
            components[key].append(sum(_float_value(row.get(key)) for row in rows))
        total_latency.append(
            sum(_float_value(row.get("total_latency_s")) for row in rows)
        )
        retried.append(any(_bool_value(row.get("is_failover_retried")) for row in rows))
    return request_ids, components, total_latency, retried


def plot_runtime_profile(csv_path: str | Path, output_path: str | Path) -> None:
    request_ids, components, total_latency, retried = build_runtime_series(csv_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(14, 6), constrained_layout=True)

    if not request_ids:
        axis.text(
            0.5,
            0.5,
            "No request runtime metrics were recorded.",
            ha="center",
            va="center",
            transform=axis.transAxes,
            fontsize=12,
            color="gray",
        )
    else:
        x_values = list(range(len(request_ids)))
        bottoms = [0.0] * len(request_ids)
        for key in COMPONENT_KEYS:
            values = components[key]
            axis.bar(
                x_values,
                values,
                bottom=bottoms,
                color=COMPONENT_COLORS[key],
                width=0.78,
                label=COMPONENT_LABELS[key],
            )
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

        axis.plot(
            x_values,
            total_latency,
            color="#111111",
            linewidth=1.5,
            marker="o",
            markersize=3,
            label="Total Latency",
        )
        for index, was_retried in enumerate(retried):
            if was_retried:
                axis.axvspan(index - 0.35, index + 0.35, color="#e45756", alpha=0.10)

        tick_positions, tick_labels = _build_tick_positions(request_ids)
        axis.set_xticks(tick_positions)
        axis.set_xticklabels(tick_labels, rotation=45, ha="right")
        axis.set_xlim(-0.5, len(request_ids) - 0.5)

    axis.set_title("Request runtime profile", fontsize=12)
    axis.set_xlabel("Request ID")
    axis.set_ylabel("Seconds")
    axis.grid(True, axis="y", linestyle="--", alpha=0.45)
    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend(loc="upper right", frameon=True, fontsize=8, ncol=2)

    fig.savefig(output, dpi=220)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot synthetic simulator request runtime metrics."
    )
    parser.add_argument("--request-detail-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plot_runtime_profile(args.request_detail_csv, args.output)
    return 0


def _request_id(row: dict[str, Any]) -> str:
    return str(row.get("job_id") or row.get("rid") or "").strip()


def _float_value(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(str(value))
    except ValueError:
        return 0.0


def _bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _sort_key(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(value):020d}")
    except ValueError:
        return (1, value)


def _build_tick_positions(
    values: list[str],
    max_ticks: int = 15,
) -> tuple[list[int], list[str]]:
    if not values:
        return [], []
    if len(values) <= max_ticks:
        positions = list(range(len(values)))
    else:
        step = max(1, (len(values) + max_ticks - 1) // max_ticks)
        positions = list(range(0, len(values), step))
        if positions[-1] != len(values) - 1:
            positions.append(len(values) - 1)
    return positions, [values[index] for index in positions]


if __name__ == "__main__":
    raise SystemExit(main())
