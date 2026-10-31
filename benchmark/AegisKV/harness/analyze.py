"""Compare policies under the same failure and draw the paper figures.

    cd benchmark/AegisKV && python -m harness.analyze \
        --nofail runs/<x>/nofail --runs rebuild=runs/<x>/rebuild lumen=... \
        --out results/<x>

A victim is a call interrupted by the failure. Its reference slack comes from the
No-Failure run (same seed, identical before the failure): the time its Workflow
could wait for it without finishing later, from actual call durations.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
from harness import workload

POLICY_LABELS = {
    "rebuild": "Rebuild",
    "lumen": "LUMEN-CP*",
    "hbm_replica": "HBM-Replica",
    "aegiskv": "AegisKV",
}


def _load(path: str) -> dict:
    with open(os.path.join(path, "result.json")) as f:
        return json.load(f)


def _reference_slack(result: dict) -> Dict[str, float]:
    """Actual-duration slack of every call in a No-Failure run (Eqs. 3-4)."""
    by_wf: Dict[int, Dict[str, dict]] = {}
    for c in result["calls"]:
        by_wf.setdefault(c["workflow"], {})[c["role"]] = c
    slack = {}
    for wf, calls in by_wf.items():
        if len(calls) != len(workload.ROLE_INDEX):
            continue
        end = max(c["t_done"] for c in calls.values())
        latest: Dict[str, float] = {}
        for stage in reversed(workload.STAGES):
            for r in stage:
                succ = workload.SUCCESSORS[r]
                latest[r] = min(
                    [end]
                    + [
                        latest[s] - (calls[s]["t_done"] - calls[s]["t_submit"])
                        for s in succ
                    ]
                )
                slack[calls[r]["key"]] = max(0.0, latest[r] - calls[r]["t_done"])
    return slack


def _wct(result: dict) -> Dict[int, float]:
    return {int(k): v[1] - v[0] for k, v in result["workflows"].items()}


def summarize(nofail: dict, run: dict) -> dict:
    ref_slack = _reference_slack(nofail)
    ref_wct, wct = _wct(nofail), _wct(run)
    t_fail = run["fail_time"]
    victims = [c for c in run["calls"] if c["failures"]]
    rows = []
    for c in victims:
        f = c["failures"][0]
        rows.append(
            {
                "key": c["key"],
                "role": c["role"],
                "critical": ref_slack.get(c["key"], 0.0) <= 1.0,
                "ref_slack": ref_slack.get(c["key"], 0.0),
                "first_output_s": f.get("t_resume_first", np.nan) - t_fail,
                "restored": f.get("restored_tokens", 0),
                "accepted": f["accepted_tokens"],
                "coverage": f.get("restored_tokens", 0) / max(1, f["accepted_tokens"]),
                "source": f.get("source", ""),
                "import_copy_s": f.get("import_copy_s"),
            }
        )
    affected = sorted({c["workflow"] for c in victims})
    delta = [wct[w] - ref_wct[w] for w in affected if w in wct and w in ref_wct]
    first = np.array([r["first_output_s"] for r in rows])
    crit = [r for r in rows if r["critical"]]
    return {
        "policy": run["policy"],
        "num_victims": len(rows),
        "num_affected_workflows": len(affected),
        "first_output_mean_s": float(np.nanmean(first)) if rows else None,
        "first_output_p95_s": float(np.nanpercentile(first, 95)) if rows else None,
        "exposed_delay_mean_s": float(
            np.nanmean([max(0.0, r["first_output_s"] - r["ref_slack"]) for r in rows])
        )
        if rows
        else None,
        "added_wct_mean_s": float(np.mean(delta)) if delta else None,
        "added_wct_p95_s": float(np.percentile(delta, 95)) if delta else None,
        "coverage_all": float(np.mean([r["coverage"] for r in rows])) if rows else None,
        "coverage_critical": float(np.mean([r["coverage"] for r in crit]))
        if crit
        else None,
        "num_critical_victims": len(crit),
        "wct_all_mean_s": float(np.mean(list(wct.values()))),
        "wct_all_p95_s": float(np.percentile(list(wct.values()), 95)),
        "victims": rows,
        "added_wct": delta,
    }


def _plot(summaries: List[dict], out: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    names = [POLICY_LABELS.get(s["policy"], s["policy"]) for s in summaries]
    hatches = ["", "//", "..", "xx", "\\\\"]
    colors = ["#bdbdbd", "#6baed6", "#fd8d3c", "#31a354", "#756bb1"]
    panels = [
        (
            "first_output",
            "First resumed\ntoken (s)",
            lambda s: [r["first_output_s"] for r in s["victims"]],
        ),
        ("added_wct", "Added WCT of affected\nWorkflows (s)", lambda s: s["added_wct"]),
        (
            "coverage_critical",
            "Restored KV of critical-\npath victims (%)",
            lambda s: [100 * r["coverage"] for r in s["victims"] if r["critical"]],
        ),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.0, 1.9))
    for ax, (_, ylabel, get) in zip(axes, panels):
        for i, s in enumerate(summaries):
            vals = np.array([v for v in get(s) if v is not None and not np.isnan(v)])
            mean = vals.mean() if len(vals) else 0.0
            ax.bar(
                i,
                mean,
                color=colors[i % 5],
                hatch=hatches[i % 5],
                edgecolor="black",
                linewidth=0.6,
                width=0.7,
            )
            if len(vals):
                jitter = (np.arange(len(vals)) % 7 - 3) * 0.04
                ax.scatter(i + jitter, vals, s=4, color="black", zorder=3, linewidths=0)
                ax.text(
                    i,
                    ax.get_ylim()[1] * 0.02,
                    f"n={len(vals)}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )
        ax.set_xticks(range(len(summaries)))
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linewidth=0.3, alpha=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout(w_pad=1.2)
    for ext in ("pdf", "png"):
        fig.savefig(
            os.path.join(out, f"failure_comparison.{ext}"), dpi=300, bbox_inches="tight"
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nofail", required=True)
    parser.add_argument("--runs", nargs="+", required=True, help="policy=path")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    nofail = _load(args.nofail)
    summaries = []
    for item in args.runs:
        _, path = item.split("=", 1)
        summaries.append(summarize(nofail, _load(path)))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)
    for s in summaries:
        print(
            json.dumps(
                {k: v for k, v in s.items() if k not in ("victims", "added_wct")}
            )
        )
    _plot(summaries, args.out)


if __name__ == "__main__":
    main()
