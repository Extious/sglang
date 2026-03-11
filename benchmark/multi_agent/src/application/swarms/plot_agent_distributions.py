"""
Plot boxplot distributions of agent output length and inference latency
from HeavySwarm detailed timing CSV files.

Usage:
    python plot_agent_distributions.py <detailed_csv> [<detailed_csv2> ...] [-o OUTPUT_DIR]

    If no CSV paths are given, all agent_timing_detailed_*.csv files under the
    default timing_reports directory are loaded and merged.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

AGENT_ORDER = [
    "question",
    "research",
    "analysis",
    "alternatives",
    "verification",
    "synthesis",
]

AGENT_DISPLAY = {
    "question": "Question",
    "research": "Research",
    "analysis": "Analysis",
    "alternatives": "Alternatives",
    "verification": "Verification",
    "synthesis": "Synthesis",
}

AGENT_COLORS = {
    "question": "#8ecae6",
    "research": "#f4a261",
    "analysis": "#a7c957",
    "alternatives": "#e9c46a",
    "verification": "#c9b1d0",
    "synthesis": "#f4978e",
}

DEFAULT_TIMING_DIR = (
    Path(__file__).resolve().parent / "agent_workspace" / "timing_reports"
)


def _load_detailed_rows(paths: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                error = (row.get("error") or "").strip()
                if error:
                    continue
                agent = (row.get("agent") or "unknown").strip().lower()
                try:
                    completion_tokens = int(float(row.get("completion_tokens", 0)))
                    latency = float(row.get("latency_seconds", 0))
                except (TypeError, ValueError):
                    continue
                rows.append({
                    "agent": agent,
                    "completion_tokens": completion_tokens,
                    "latency_seconds": latency,
                })
    return rows


def _group_by_agent(
    rows: List[Dict[str, Any]],
) -> Dict[str, Dict[str, list]]:
    groups: Dict[str, Dict[str, list]] = defaultdict(
        lambda: {"completion_tokens": [], "latency_seconds": []}
    )
    for row in rows:
        agent = row["agent"]
        groups[agent]["completion_tokens"].append(row["completion_tokens"])
        groups[agent]["latency_seconds"].append(row["latency_seconds"])
    return groups


def _plot(
    groups: Dict[str, Dict[str, list]],
    output_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("[plot] matplotlib not available, cannot generate plots.")
        sys.exit(1)

    ordered = [a for a in AGENT_ORDER if a in groups]
    ordered += sorted(a for a in groups if a not in AGENT_ORDER)
    if not ordered:
        print("[plot] No agent data to plot.")
        return

    paper_style = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "axes.labelweight": "bold",
        "xtick.labelsize": 12,
        "ytick.labelsize": 11,
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

    colors = [AGENT_COLORS.get(a, "#aaaaaa") for a in ordered]

    def _make_boxplot(
        ax,
        data_lists: List[list],
        ylabel: str,
        title: str,
    ):
        bp = ax.boxplot(
            data_lists,
            patch_artist=True,
            widths=0.55,
            showfliers=True,
            flierprops=dict(
                marker="o",
                markerfacecolor="#999999",
                markeredgecolor="#999999",
                markersize=3,
                alpha=0.5,
            ),
            medianprops=dict(color="#d62728", linewidth=1.5),
            whiskerprops=dict(color="#444444", linewidth=1.0),
            capprops=dict(color="#444444", linewidth=1.0),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("#444444")
            patch.set_linewidth(1.0)
            patch.set_alpha(0.85)

        ax.set_xticklabels([""] * len(data_lists))
        ax.tick_params(axis="x", length=0)
        ax.set_xlabel("Agent", fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=8)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=False, nbins=6))
        ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)

    legend_patches = [
        mpatches.Patch(
            facecolor=c, edgecolor="#444444",
            label=AGENT_DISPLAY.get(a, a.capitalize()),
        )
        for a, c in zip(ordered, colors)
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    output_data = [groups[a]["completion_tokens"] for a in ordered]
    _make_boxplot(ax1, output_data,
                  ylabel="Sequence Length (tokens)",
                  title="(a) Output Length Distribution")

    latency_data = [groups[a]["latency_seconds"] for a in ordered]
    _make_boxplot(ax2, latency_data,
                  ylabel="Latency (s)",
                  title="(b) Inference Latency Distribution")

    fig.legend(
        handles=legend_patches,
        loc="upper center",
        ncol=min(len(ordered), 6),
        frameon=True,
        fontsize=11,
        bbox_to_anchor=(0.5, 1.0),
        borderpad=0.4,
        handlelength=1.2,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            output_dir / f"agent_distributions.{suffix}",
            dpi=500,
            bbox_inches="tight",
        )
    plt.close(fig)
    print(f"[plot] Agent distributions saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot agent output-length and latency boxplot distributions."
    )
    parser.add_argument(
        "csv_files",
        nargs="*",
        help=(
            "Paths to agent_timing_detailed_*.csv files. "
            "If omitted, all such files under the default timing_reports dir are used."
        ),
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="",
        help="Output directory for plots. Defaults to the timing_reports directory.",
    )
    args = parser.parse_args()

    if args.csv_files:
        csv_paths = [Path(f) for f in args.csv_files]
    else:
        csv_paths = sorted(DEFAULT_TIMING_DIR.glob("agent_timing_detailed_*.csv"))
        if not csv_paths:
            print(
                f"[plot] No agent_timing_detailed_*.csv found in {DEFAULT_TIMING_DIR}"
            )
            sys.exit(1)

    for p in csv_paths:
        if not p.exists():
            print(f"[plot] File not found: {p}")
            sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_TIMING_DIR
    output_dir = output_dir.resolve()

    print(f"[plot] Loading {len(csv_paths)} CSV file(s)...")
    rows = _load_detailed_rows(csv_paths)
    if not rows:
        print("[plot] No valid data rows found.")
        sys.exit(1)

    print(f"[plot] Loaded {len(rows)} records.")
    groups = _group_by_agent(rows)
    for agent, data in groups.items():
        n = len(data["completion_tokens"])
        print(f"  {agent}: {n} records")

    _plot(groups, output_dir)


if __name__ == "__main__":
    main()
