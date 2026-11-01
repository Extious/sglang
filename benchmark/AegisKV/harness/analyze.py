"""Compare policies under the same failure scenario and draw the paper figure.

    cd benchmark/AegisKV && python -m harness.analyze --nofail runs/<x>/nofail \
        --runs rebuild=runs/<x>/rebuild_w0,runs/<x>/rebuild_w1 aegiskv=... --out results/<x>

Each policy may have several runs (one failure each, e.g. a different failed
Worker); victims and affected Workflows are pooled across them. A victim is a
call interrupted by the failure. Its reference slack comes from the No-Failure
run with the same seed: how long its Workflow could wait for it without
finishing later, computed from actual call durations.
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
# Bytes of one token slot (MiniCPM4.1-8B KV, all layers), for GiB conversion.
SLOT_BYTES = 32 * 2 * 2 * 128 * 2
CRITICAL_SLACK_S = 1.0


def _load(path: str) -> dict:
    with open(os.path.join(path, "result.json")) as f:
        return json.load(f)


def _events(path: str) -> List[dict]:
    with open(os.path.join(path, "events.jsonl")) as f:
        return [json.loads(line) for line in f]


def _reference_slack(result: dict) -> Dict[str, float]:
    """Actual-duration slack of every call in a No-Failure run (Eqs. 3-4)."""
    by_wf: Dict[int, Dict[str, dict]] = {}
    for c in result["calls"]:
        by_wf.setdefault(c["workflow"], {})[c["role"]] = c
    slack = {}
    for calls in by_wf.values():
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


def _victim_rows(run: dict, ref_slack: Dict[str, float], tag: str) -> List[dict]:
    rows = []
    for c in run["calls"]:
        if not c["failures"]:
            continue
        f = c["failures"][0]
        slack = ref_slack.get(c["key"], 0.0)
        first = f.get("t_resume_first", np.nan) - run["fail_time"]
        rows.append(
            {
                "run": tag,
                "key": c["key"],
                "role": c["role"],
                "critical": slack <= CRITICAL_SLACK_S,
                "ref_slack": slack,
                "first_output_s": first,
                "exposed_delay_s": max(0.0, first - slack),
                "restored": f.get("restored_tokens", 0),
                "accepted": f["accepted_tokens"],
                "coverage": f.get("restored_tokens", 0) / max(1, f["accepted_tokens"]),
                "source": f.get("source", ""),
                "import_copy_s": f.get("import_copy_s"),
            }
        )
    return rows


def _usage(events: List[dict], t_end: float) -> Dict[str, float]:
    """Time-averaged backup footprint summed over holders, in GiB."""
    samples = [e for e in events if e["kind"] == "usage" and e["t"] <= t_end]
    if not samples:
        return {"hbm_gib": 0.0, "dram_gib": 0.0, "hbm_peak_gib": 0.0}
    hbm = [sum(e["hbm"].values()) * SLOT_BYTES / 2**30 for e in samples]
    dram = [sum(e["dram"].values()) * SLOT_BYTES / 2**30 for e in samples]
    return {
        "hbm_gib": float(np.mean(hbm)),
        "dram_gib": float(np.mean(dram)),
        "hbm_peak_gib": float(np.max(hbm)),
    }


def summarize(nofail: dict, runs: List[str]) -> dict:
    ref_slack, ref_wct = _reference_slack(nofail), _wct(nofail)
    rows, delta, pre_ratio, usages, policy = [], [], [], [], None
    for path in runs:
        run = _load(path)
        policy = run["policy"]
        wct = _wct(run)
        run_rows = _victim_rows(run, ref_slack, os.path.basename(path))
        rows += run_rows
        affected = {r["key"].split("-")[0][2:] for r in run_rows}
        delta += [wct[int(w)] - ref_wct[int(w)] for w in affected]
        # No-failure overhead: Workflows that finished before the failure.
        pre_ratio += [
            wct[w] / ref_wct[w]
            for w, (_, end) in ((int(k), v) for k, v in run["workflows"].items())
            if end < run["fail_time"] and w in ref_wct
        ]
        usages.append(_usage(_events(path), t_end=run["fail_time"]))
    crit = [r for r in rows if r["critical"]]
    arr = lambda key, rs: np.array([r[key] for r in rs], dtype=float)  # noqa: E731
    stat = lambda xs, fn: float(fn(xs)) if len(xs) else None  # noqa: E731
    return {
        "policy": policy,
        "num_runs": len(runs),
        "num_victims": len(rows),
        "num_critical_victims": len(crit),
        "num_affected_workflows": len(delta),
        "first_output_mean_s": stat(arr("first_output_s", rows), np.nanmean),
        "first_output_p95_s": stat(
            arr("first_output_s", rows), lambda x: np.nanpercentile(x, 95)
        ),
        "critical_first_output_mean_s": stat(arr("first_output_s", crit), np.nanmean),
        "exposed_delay_mean_s": stat(arr("exposed_delay_s", rows), np.nanmean),
        "critical_exposed_delay_mean_s": stat(arr("exposed_delay_s", crit), np.nanmean),
        "added_wct_mean_s": stat(np.array(delta), np.mean),
        "added_wct_p95_s": stat(np.array(delta), lambda x: np.percentile(x, 95)),
        "coverage_all": stat(arr("coverage", rows), np.mean),
        "coverage_critical": stat(arr("coverage", crit), np.mean),
        "prefailure_wct_ratio": stat(np.array(pre_ratio), np.mean),
        "hbm_gib": float(np.mean([u["hbm_gib"] for u in usages])),
        "hbm_peak_gib": float(np.max([u["hbm_peak_gib"] for u in usages])),
        "dram_gib": float(np.mean([u["dram_gib"] for u in usages])),
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
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    names = [POLICY_LABELS.get(s["policy"], s["policy"]) for s in summaries]
    styles = {
        "rebuild": ("#bdbdbd", ""),
        "lumen": ("#6baed6", "//"),
        "hbm_replica": ("#fd8d3c", ".."),
        "aegiskv": ("#31a354", "xx"),
    }
    panels = [
        (
            "First resumed\ntoken (s)",
            lambda s: [r["first_output_s"] for r in s["victims"]],
        ),
        ("Added WCT of\naffected Workflows (s)", lambda s: s["added_wct"]),
        (
            "Restored KV, critical-\npath victims (%)",
            lambda s: [100 * r["coverage"] for r in s["victims"] if r["critical"]],
        ),
        ("Peer-HBM backup,\nmean (GiB)", lambda s: [s["hbm_gib"]]),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.0, 1.8))
    for ax, (ylabel, get) in zip(axes, panels):
        for i, s in enumerate(summaries):
            color, hatch = styles.get(s["policy"], ("#999999", ""))
            vals = np.array([v for v in get(s) if v is not None and not np.isnan(v)])
            ax.bar(
                i,
                vals.mean() if len(vals) else 0.0,
                color=color,
                hatch=hatch,
                edgecolor="black",
                linewidth=0.6,
                width=0.7,
            )
            if len(vals) > 1:
                jitter = (np.arange(len(vals)) % 7 - 3) * 0.05
                ax.scatter(i + jitter, vals, s=3, color="black", zorder=3, linewidths=0)
        ax.set_xticks(range(len(summaries)))
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linewidth=0.3, alpha=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout(w_pad=0.8)
    for ext in ("pdf", "png"):
        fig.savefig(
            os.path.join(out, f"failure_comparison.{ext}"), dpi=300, bbox_inches="tight"
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nofail", required=True)
    parser.add_argument(
        "--runs", nargs="+", required=True, help="policy=path[,path...]"
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    nofail = _load(args.nofail)
    summaries = []
    for item in args.runs:
        _, paths = item.split("=", 1)
        summaries.append(summarize(nofail, paths.split(",")))
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
