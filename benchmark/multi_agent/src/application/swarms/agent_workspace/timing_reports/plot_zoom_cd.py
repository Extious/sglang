"""Generate zoomed-in C & D charts for tasks 100–150."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

AGENT_ORDER = [
    "question",
    "research",
    "analysis",
    "alternatives",
    "verification",
    "synthesis",
]

agent_styles = {
    "question": ("#1f77b4", "o"),
    "research": ("#d99600", "s"),
    "analysis": ("#1b9e77", "^"),
    "alternatives": ("#d95f02", "D"),
    "verification": ("#cc79a7", "v"),
    "synthesis": ("#4ea8de", "o"),
}

paper_style = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444444",
    "axes.linewidth": 1.0,
    "grid.color": "#c7c7c7",
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.75,
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
plt.rcParams.update(paper_style)

HERE = Path(__file__).resolve().parent
json_path = HERE / "timing_metrics_20260305_143638.json"

with json_path.open("r", encoding="utf-8") as f:
    raw = json.load(f)

results = raw["results"]

TASK_LO, TASK_HI = 100, 150

summary_rows = []
for entry in results:
    ti = entry["task_index"]
    if ti < TASK_LO or ti > TASK_HI:
        continue
    timing = entry.get("timing", {})
    task_duration = timing.get("task_duration_seconds", 0.0)
    requests = timing.get("requests", [])

    per_agent = {}
    for req in requests:
        agent = req.get("agent", "unknown")
        if agent not in per_agent:
            per_agent[agent] = {
                "latency_seconds": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cached_tokens": 0,
                "cache_hit_ratio": None,
            }
        per_agent[agent]["latency_seconds"] += req.get("latency_seconds", 0.0)
        per_agent[agent]["prompt_tokens"] += req.get("prompt_tokens", 0)
        per_agent[agent]["completion_tokens"] += req.get("completion_tokens", 0)
        per_agent[agent]["total_tokens"] += req.get("total_tokens", 0)
        per_agent[agent]["cached_tokens"] += req.get("cached_tokens", 0)
        ratio = req.get("cache_hit_ratio")
        if ratio is not None:
            per_agent[agent]["cache_hit_ratio"] = ratio

    row = {
        "task_index": ti,
        "task_duration_seconds": task_duration,
    }
    for agent in AGENT_ORDER:
        m = per_agent.get(agent, {})
        row[f"{agent}_seconds"] = m.get("latency_seconds", 0.0)
    summary_rows.append(row)

summary_rows.sort(key=lambda r: r["task_index"])
task_indices = [r["task_index"] for r in summary_rows]
task_durations = [float(r["task_duration_seconds"]) for r in summary_rows]

fig, (ax_runtime, ax_completion) = plt.subplots(
    1, 2, figsize=(16, 6.5)
)
fig.suptitle(
    f"HeavySwarm Profiling — Tasks {TASK_LO}–{TASK_HI} (Zoomed C & D)",
    fontsize=17,
    fontweight="bold",
)

for agent in AGENT_ORDER:
    color, marker = agent_styles[agent]
    y = [r.get(f"{agent}_seconds", 0.0) for r in summary_rows]
    ax_runtime.plot(
        task_indices, y,
        label=agent.capitalize(),
        color=color,
        marker=marker,
        linewidth=1.8,
        markersize=5,
    )

ax_runtime.set_title("(C) Agent Runtime by Task", fontsize=13, fontweight="bold")
ax_runtime.set_xlabel("Task Index", fontsize=12, fontweight="bold")
ax_runtime.set_ylabel("Time (s)", fontsize=12, fontweight="bold")
ax_runtime.grid(True, axis="both")
ax_runtime.xaxis.set_major_locator(mticker.MultipleLocator(5))
ax_runtime.xaxis.set_minor_locator(mticker.MultipleLocator(1))
ax_runtime.set_xlim(TASK_LO - 0.5, TASK_HI + 0.5)
ax_runtime.set_axisbelow(True)
ax_runtime.legend(loc="upper right", fontsize=9, frameon=True)

mean_dur = float(np.mean(task_durations))
median_dur = float(np.median(task_durations))

ax_completion.plot(
    task_indices, task_durations,
    color="#222222",
    marker="o",
    linewidth=1.8,
    markersize=5,
    label="Task completion time",
)
ax_completion.axhline(
    mean_dur, color="#8f8f8f", linestyle="--", linewidth=1.2,
    label=f"Mean={mean_dur:.1f}s",
)
ax_completion.axhline(
    median_dur, color="#8f8f8f", linestyle=":", linewidth=1.2,
    label=f"Median={median_dur:.1f}s",
)
ax_completion.set_title("(D) Task Completion Time", fontsize=13, fontweight="bold")
ax_completion.set_xlabel("Task Index", fontsize=12, fontweight="bold")
ax_completion.set_ylabel("Time (s)", fontsize=12, fontweight="bold")
ax_completion.grid(True, axis="both")
ax_completion.xaxis.set_major_locator(mticker.MultipleLocator(5))
ax_completion.xaxis.set_minor_locator(mticker.MultipleLocator(1))
ax_completion.set_xlim(TASK_LO - 0.5, TASK_HI + 0.5)
ax_completion.set_axisbelow(True)
ax_completion.legend(loc="upper right", fontsize=9, frameon=True)

for tick in ax_runtime.get_xticklabels():
    tick.set_rotation(45)
    tick.set_ha("right")
for tick in ax_completion.get_xticklabels():
    tick.set_rotation(45)
    tick.set_ha("right")

fig.tight_layout(rect=(0, 0, 1, 0.94))

out_png = HERE / "zoom_task100_150_CD.png"
out_pdf = HERE / "zoom_task100_150_CD.pdf"
out_svg = HERE / "zoom_task100_150_CD.svg"

fig.savefig(out_png, dpi=500, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_svg, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
print(f"Saved: {out_svg}")
