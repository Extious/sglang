#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

AGENT_COLORS = {
    "Planning Coordinator": "#1f77b4",
    "Data Collector": "#ff7f0e",
    "Deep Analyst": "#2ca02c",
    "Trend Scout": "#d62728",
    "Risk Assessor": "#9467bd",
    "Report Synthesizer": "#8c564b",
}

AGENT_MARKERS = {
    "Planning Coordinator": "o",
    "Data Collector":       "s",
    "Deep Analyst":         "^",
    "Trend Scout":          "D",
    "Risk Assessor":        "v",
    "Report Synthesizer":   "P",
}

AGENT_ORDER = [
    "Planning Coordinator",
    "Data Collector",
    "Deep Analyst",
    "Trend Scout",
    "Risk Assessor",
    "Report Synthesizer",
]
AGENT_ORDER_INDEX = {a: i for i, a in enumerate(AGENT_ORDER)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot runtime and token profiling charts from trace_log.json."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("trace_log.json"),
        help="Path to the input trace log JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("trace_log_profile.png"),
        help="Path to the output image file.",
    )
    return parser.parse_args()


def topic_sort_key(item: dict) -> tuple[int, str]:
    topic_id = str(item.get("topic_id", ""))
    if topic_id.isdigit():
        return (0, f"{int(topic_id):010d}")
    return (1, topic_id)


def load_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise ValueError("Input JSON must be a list of job records.")
    return sorted(records, key=topic_sort_key)


def collect_agent_names(records: list[dict]) -> list[str]:
    seen = set()
    agent_names = []
    for record in records:
        for item in record.get("agent_timings", []):
            agent = item.get("agent")
            if agent and agent not in seen:
                seen.add(agent)
                agent_names.append(agent)
    return agent_names


def build_series(records: list[dict], agent_names: list[str]) -> tuple[list[str], dict, list[float]]:
    topic_ids = [str(record.get("topic_id", "")) for record in records]
    metrics = {
        agent: {
            "prompt_tokens": [],
            "completion_tokens": [],
            "duration_s": [],
        }
        for agent in agent_names
    }
    completion_times = []

    for record in records:
        timing_map = {
            item.get("agent"): item
            for item in record.get("agent_timings", [])
            if item.get("agent")
        }

        for agent in agent_names:
            item = timing_map.get(agent, {})
            metrics[agent]["prompt_tokens"].append(item.get("prompt_tokens", float("nan")))
            metrics[agent]["completion_tokens"].append(item.get("completion_tokens", float("nan")))
            metrics[agent]["duration_s"].append(item.get("duration_s", float("nan")))

        summary = record.get("summary", {})
        completion_times.append(summary.get("wall_time_s", float("nan")))

    return topic_ids, metrics, completion_times


def build_tick_positions(topic_ids: list[str], max_ticks: int = 12) -> tuple[list[int], list[str]]:
    count = len(topic_ids)
    if count == 0:
        return [], []

    if count <= max_ticks:
        positions = list(range(count))
    else:
        step = max(1, (count + max_ticks - 1) // max_ticks)
        positions = list(range(0, count, step))
        if positions[-1] != count - 1:
            positions.append(count - 1)

    labels = [topic_ids[index] for index in positions]
    return positions, labels


def style_job_axis(
    axis: plt.Axes,
    x_values: list[int],
    tick_positions: list[int],
    tick_labels: list[str],
    *,
    show_labels: bool,
) -> None:
    axis.set_xlim(-0.5, len(x_values) - 0.5)
    axis.set_xticks(tick_positions)
    if show_labels:
        axis.set_xticklabels(tick_labels, rotation=45, ha="right")
    else:
        axis.tick_params(axis="x", which="both", labelbottom=False)
    axis.margins(x=0.01)


def build_marker_indices(
    point_count: int,
    series_index: int,
    series_count: int,
) -> list[int]:
    if point_count <= 0:
        return []
    if point_count == 1:
        return [0]

    group_count = min(series_count, point_count, 4)
    start = series_index % group_count
    indices = list(range(start, point_count, group_count))
    return indices or [start % point_count]


def plot_chart(topic_ids: list[str], metrics: dict, completion_times: list[float], output_path: Path) -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass

    agent_names = list(metrics.keys())
    x_values = list(range(len(topic_ids)))
    tick_positions, tick_labels = build_tick_positions(topic_ids)
    figure_width = max(16, min(28, 12 + len(topic_ids) * 0.35))

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(figure_width, 10),
        sharex=True,
        constrained_layout=False,
    )
    panels = [
        ("prompt_tokens", "(A) Agent Prompt Tokens by Job", "Tokens"),
        ("completion_tokens", "(B) Agent Completion Tokens by Job", "Tokens"),
        ("duration_s", "(C) Agent Runtime by Job", "Time (s)"),
    ]
    legend_handles = []

    for axis, (metric_key, title, ylabel) in zip(axes.flat[:3], panels):
        for agent in agent_names:
            color = AGENT_COLORS.get(agent)
            marker = AGENT_MARKERS.get(agent, "o")
            idx = AGENT_ORDER_INDEX.get(agent, 0)
            marker_indices = build_marker_indices(len(x_values), idx, len(agent_names))
            axis.plot(
                x_values,
                metrics[agent][metric_key],
                color=color,
                label=agent,
                marker=marker,
                markevery=marker_indices,
                linestyle="-",
                linewidth=1.6,
                markersize=4.5,
                alpha=0.85,
            )
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Job ID", fontweight="bold")
        axis.set_ylabel(ylabel, fontweight="bold")
        style_job_axis(
            axis,
            x_values,
            tick_positions,
            tick_labels,
            show_labels=axis in (axes.flat[2], axes.flat[3]),
        )

    for agent in agent_names:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=AGENT_COLORS.get(agent),
                marker=AGENT_MARKERS.get(agent, "o"),
                linestyle="-",
                linewidth=1.6,
                markersize=4.5,
                alpha=0.9,
                label=agent,
            )
        )

    completion_axis = axes.flat[3]
    completion_axis.plot(
        x_values,
        completion_times,
        color="black",
        marker="o",
        linewidth=1.8,
        markersize=3.5,
        label="Job completion time",
    )

    valid_times = [value for value in completion_times if value == value]
    if valid_times:
        mean_time = sum(valid_times) / len(valid_times)
        median_time = sorted(valid_times)[len(valid_times) // 2]
        if len(valid_times) % 2 == 0:
            middle = len(valid_times) // 2
            median_time = (sorted(valid_times)[middle - 1] + sorted(valid_times)[middle]) / 2

        completion_axis.axhline(
            mean_time,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            label=f"Mean={mean_time:.1f}s",
        )
        completion_axis.axhline(
            median_time,
            color="gray",
            linestyle=":",
            linewidth=1.2,
            label=f"Median={median_time:.1f}s",
        )

    completion_axis.set_title("(D) Job Completion Time", fontweight="bold")
    completion_axis.set_xlabel("Job ID", fontweight="bold")
    completion_axis.set_ylabel("Time (s)", fontweight="bold")
    style_job_axis(
        completion_axis,
        x_values,
        tick_positions,
        tick_labels,
        show_labels=True,
    )
    completion_axis.legend(loc="upper right", fontsize=9)

    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="lower center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle("CrewAI Runtime and Token Profiling", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    agent_names = collect_agent_names(records)
    if len(agent_names) != 6:
        print(f"Warning: expected 6 agents, found {len(agent_names)}.")
    topic_ids, metrics, completion_times = build_series(records, agent_names)
    plot_chart(topic_ids, metrics, completion_times, args.output)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
