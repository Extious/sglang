#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

METRIC_KEYS = [
    "l1_match",
    "l2_match",
    "remote_match",
    "remote_prefetch",
    "reused_device",
    "reused_host",
    "reused_storage",
]

METRIC_LABELS = {
    "l1_match": "L1 Match (GPU)",
    "l2_match": "L2 Match (Host)",
    "remote_match": "Remote Match",
    "remote_prefetch": "Remote Prefetch",
    "reused_device": "Reused Device",
    "reused_host": "Reused Host",
    "reused_storage": "Reused Storage",
}

METRIC_COLORS = {
    "l1_match": "#4c78a8",
    "l2_match": "#f58518",
    "remote_match": "#54a24b",
    "remote_prefetch": "#e45756",
    "reused_device": "#9ecae9",
    "reused_host": "#ffbf79",
    "reused_storage": "#88d27a",
}

METRIC_MARKERS = {
    "l1_match": "o",
    "l2_match": "s",
    "remote_match": "^",
    "remote_prefetch": "D",
    "reused_device": "v",
    "reused_host": "<",
    "reused_storage": ">",
}


def build_request_matrix(
    csv_path: str | Path,
) -> tuple[list[str], dict[str, list[int]], list[bool]]:
    rows_by_request: dict[str, list[dict[str, str]]] = {}
    with Path(csv_path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            request_id = _request_id(row)
            if not request_id:
                continue
            rows_by_request.setdefault(request_id, []).append(row)

    request_ids = sorted(rows_by_request, key=_sort_key)
    metrics = {key: [] for key in METRIC_KEYS}
    retried = []
    for request_id in request_ids:
        rows = rows_by_request[request_id]
        for key in METRIC_KEYS:
            metrics[key].append(sum(_int_value(row.get(key)) for row in rows))
        retried.append(any(_bool_value(row.get("is_failover_retried")) for row in rows))
    return request_ids, metrics, retried


def plot_cache_hits(csv_path: str | Path, output_path: str | Path) -> None:
    request_ids, metrics, retried = build_request_matrix(csv_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(14, 5.5), constrained_layout=True)

    if not request_ids:
        axis.text(
            0.5,
            0.5,
            "No cache metrics were recorded.",
            ha="center",
            va="center",
            transform=axis.transAxes,
            fontsize=12,
            color="gray",
        )
    else:
        x_values = list(range(len(request_ids)))
        for key in METRIC_KEYS:
            values = metrics[key]
            if not any(values):
                continue
            axis.plot(
                x_values,
                values,
                color=METRIC_COLORS[key],
                marker=METRIC_MARKERS[key],
                linewidth=1.1,
                markersize=3,
                label=METRIC_LABELS[key],
            )

        for index, was_retried in enumerate(retried):
            if was_retried:
                axis.axvspan(index - 0.35, index + 0.35, color="#e45756", alpha=0.10)

        tick_positions, tick_labels = _build_tick_positions(request_ids)
        axis.set_xticks(tick_positions)
        axis.set_xticklabels(tick_labels, rotation=45, ha="right")
        axis.set_xlim(-0.5, len(request_ids) - 0.5)

    axis.set_title("Cache token profile by request", fontsize=12)
    axis.set_xlabel("Request ID")
    axis.set_ylabel("Token count")
    axis.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
    axis.grid(True, axis="y", linestyle="--", alpha=0.45)
    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend(loc="upper right", frameon=True, fontsize=8, ncol=2)

    fig.savefig(output, dpi=220)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot synthetic simulator cache token metrics by request."
    )
    parser.add_argument("--cache-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plot_cache_hits(args.cache_csv, args.output)
    return 0


def _request_id(row: dict[str, Any]) -> str:
    return str(row.get("job_id") or row.get("rid") or "").strip()


def _int_value(value: Any) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(float(str(value)))
    except ValueError:
        return 0


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
