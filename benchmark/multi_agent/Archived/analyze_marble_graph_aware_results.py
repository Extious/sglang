"""
MARBLE Graph-Aware Serving Experiments - Results Analyzer

This script analyzes the results from run_marble_graph_aware_experiments.py
and generates:
- CSV summaries for each experiment
- Visualization plots (cache hit, latency, etc.)
- Markdown report with tables and plot references

Usage:
    python analyze_marble_graph_aware_results.py --input-root ./results/20260202_123456_marble_graph_aware

Dependencies:
    pip install pandas matplotlib tabulate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Check dependencies
try:
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install pandas matplotlib tabulate")
    sys.exit(1)

# Check for tabulate (needed for to_markdown)
try:
    import tabulate  # noqa: F401
except ImportError:
    print("Missing dependency: tabulate")
    print("Run: pip install tabulate")
    sys.exit(1)


# =============================================================================
# Utility Functions
# =============================================================================

def _read_json(path: Path) -> Dict[str, Any]:
    """Read and parse a JSON file."""
    return json.loads(path.read_text())


def _maybe(path: Path) -> Dict[str, Any]:
    """Read JSON file if it exists, otherwise return empty dict."""
    return _read_json(path) if path.exists() else {}


def _ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def _flatten_latency(prefix: str, obj: Dict[str, Any]) -> Dict[str, Any]:
    """Extract latency metrics from nested structure into flat dict."""
    lat = obj.get(prefix, {}) or {}
    return {
        f"{prefix}_p50_ms": lat.get("p50_ms"),
        f"{prefix}_p95_ms": lat.get("p95_ms"),
        f"{prefix}_p99_ms": lat.get("p99_ms"),
        f"{prefix}_mean_ms": lat.get("mean_ms"),
    }


# =============================================================================
# Data Collection Functions
# =============================================================================

def collect_exp1(input_root: Path) -> pd.DataFrame:
    """Collect Experiment 1 (routing policies) results into a DataFrame.

    Args:
        input_root: Root directory containing experiment results

    Returns:
        DataFrame with columns: workload, policy, cache_hit_ratio, throughput_rps,
        total_time_s, successful_requests, failed_requests, ttft_*, latency_*
    """
    rows: List[Dict[str, Any]] = []
    base = input_root / "exp1_routing"
    if not base.exists():
        return pd.DataFrame()

    for workload_dir in sorted(base.iterdir()):
        if not workload_dir.is_dir():
            continue
        workload = workload_dir.name
        for policy_dir in sorted(workload_dir.iterdir()):
            if not policy_dir.is_dir():
                continue
            policy = policy_dir.name
            summary_path = policy_dir / "summary_with_cache.json"
            if not summary_path.exists():
                continue
            s = _read_json(summary_path)
            rows.append(
                {
                    "workload": workload,
                    "policy": policy,
                    "cache_hit_ratio": s.get("cache_hit_ratio"),
                    "throughput_rps": s.get("throughput_rps"),
                    "total_time_s": s.get("total_time_s"),
                    "successful_requests": s.get("successful_requests"),
                    "failed_requests": s.get("failed_requests"),
                    **_flatten_latency("ttft", s),
                    **_flatten_latency("latency", s),
                }
            )
    return pd.DataFrame(rows)


def collect_exp2(input_root: Path) -> pd.DataFrame:
    """Collect Experiment 2 (failure injection) results into a DataFrame.

    Args:
        input_root: Root directory containing experiment results

    Returns:
        DataFrame with columns: strategy, scenario, case, fail_gpu_ids,
        delta_total_time_pct, blast_radius_steps, recovery_time_s, etc.
    """
    rows: List[Dict[str, Any]] = []
    base = input_root / "exp2_failure"
    if not base.exists():
        return pd.DataFrame()

    def _read_records(path: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        if not path.exists():
            return records
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
        return records

    for strategy_dir in sorted(base.iterdir()):
        if not strategy_dir.is_dir():
            continue
        for scenario_dir in sorted(strategy_dir.iterdir()):
            if not scenario_dir.is_dir():
                continue
            for case_dir in sorted(scenario_dir.iterdir()):
                if not case_dir.is_dir():
                    continue
                summary_path = case_dir / "failure_summary.json"
                if not summary_path.exists():
                    continue
                s = _read_json(summary_path)
                wf = s.get("with_failure", {}) or {}
                base_s = s.get("baseline", {}) or {}

                records = _read_records(case_dir / "with_failure" / "records.jsonl")
                step_keys = {
                    (
                        r.get("workflow_id"),
                        r.get("step_id"),
                        r.get("agent_id"),
                    )
                    for r in records
                    if r.get("workflow_id") is not None and r.get("step_id") is not None
                }
                affected_step_keys = {
                    (
                        r.get("workflow_id"),
                        r.get("step_id"),
                        r.get("agent_id"),
                    )
                    for r in records
                    if (
                        r.get("workflow_id") is not None
                        and r.get("step_id") is not None
                        and (int(r.get("attempt", 0)) > 0 or not bool(r.get("success", False)))
                    )
                }
                total_steps = len(step_keys)
                affected_steps = len(affected_step_keys)
                blast_radius = (affected_steps / total_steps) if total_steps > 0 else None

                recompute_prompt_tokens = 0
                for r in records:
                    if int(r.get("attempt", 0)) <= 0:
                        continue
                    usage = r.get("usage")
                    if not isinstance(usage, dict):
                        continue
                    pt = usage.get("prompt_tokens")
                    if isinstance(pt, (int, float)):
                        recompute_prompt_tokens += int(pt)

                rows.append(
                    {
                        "strategy": s.get("strategy"),
                        "scenario": s.get("scenario"),
                        "case": s.get("case"),
                        "fail_gpu_ids": ",".join(str(x) for x in (s.get("fail_gpu_ids") or [])),
                        "delta_total_time_pct": s.get("delta_total_time_pct"),
                        "recompute_tokens_approx": s.get("recompute_tokens_approx"),
                        "recompute_prompt_tokens_retry": recompute_prompt_tokens,
                        "blast_radius_steps": blast_radius,
                        "recovery_time_s": s.get("recovery_time_s"),
                        "failed_requests": s.get("failed_requests"),
                        "retried_requests": s.get("retried_requests"),
                        "baseline_total_time_s": base_s.get("total_time_s"),
                        "with_failure_total_time_s": wf.get("total_time_s"),
                        "delta_latency_p95_ms": (
                            (wf.get("latency", {}) or {}).get("p95_ms")
                            - (base_s.get("latency", {}) or {}).get("p95_ms")
                            if (wf.get("latency", {}) or {}).get("p95_ms") is not None
                            and (base_s.get("latency", {}) or {}).get("p95_ms") is not None
                            else None
                        ),
                    }
                )
    return pd.DataFrame(rows)


def collect_exp3(input_root: Path) -> pd.DataFrame:
    """Collect Experiment 3 (scheduling) results into a DataFrame.

    Args:
        input_root: Root directory containing experiment results

    Returns:
        DataFrame with columns: concurrency, strategy, total_time_s,
        throughput_rps, latency_*
    """
    rows: List[Dict[str, Any]] = []
    base = input_root / "exp3_scheduling"
    if not base.exists():
        return pd.DataFrame()

    for conc_dir in sorted(base.iterdir()):
        if not conc_dir.is_dir() or not conc_dir.name.startswith("concurrency_"):
            continue
        conc = int(conc_dir.name.split("_", 1)[1])
        for strat_dir in sorted(conc_dir.iterdir()):
            if not strat_dir.is_dir():
                continue
            summary_path = strat_dir / "summary_with_cache.json"
            if not summary_path.exists():
                continue
            s = _read_json(summary_path)
            rows.append(
                {
                    "concurrency": conc,
                    "strategy": strat_dir.name,
                    "total_time_s": s.get("total_time_s"),
                    "throughput_rps": s.get("throughput_rps"),
                    "failed_requests": s.get("failed_requests"),
                    **_flatten_latency("latency", s),
                }
            )
    return pd.DataFrame(rows)


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    """Main entry point for results analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze MARBLE Graph-Aware Serving Experiment Results"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory containing experiment results"
    )
    args = parser.parse_args()

    input_root: Path = args.input_root
    if not input_root.exists():
        print(f"ERROR: Input directory not found: {input_root}")
        sys.exit(1)

    out_dir = input_root / "analysis"
    _ensure_dir(out_dir)

    print(f"Analyzing results from: {input_root}")
    print(f"Output directory: {out_dir}")

    # Collect data from all experiments
    print("\nCollecting experiment data...")
    exp1 = collect_exp1(input_root)
    exp2 = collect_exp2(input_root)
    exp3 = collect_exp3(input_root)

    print(f"  Experiment 1: {len(exp1)} results")
    print(f"  Experiment 2: {len(exp2)} results")
    print(f"  Experiment 3: {len(exp3)} results")

    # -------------------------------------------------------------------------
    # Experiment 1: Generate CSVs and plots
    # -------------------------------------------------------------------------
    if not exp1.empty:
        print("\nGenerating Experiment 1 outputs...")
        exp1.sort_values(["workload", "policy"]).to_csv(out_dir / "exp1_routing.csv", index=False)

        for workload in sorted(exp1["workload"].unique()):
            sub = exp1[exp1["workload"] == workload].sort_values("policy")

            # Cache hit bar chart
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(sub["policy"], sub["cache_hit_ratio"])
            ax.set_title(f"Exp1 Cache Hit Ratio - {workload}")
            ax.set_ylabel("cache_hit_ratio")
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            fig.tight_layout()
            fig.savefig(out_dir / f"exp1_{workload}_cache_hit.png", dpi=200)
            plt.close(fig)

            # TTFT P95 bar chart
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(sub["policy"], sub["ttft_p95_ms"])
            ax.set_title(f"Exp1 TTFT P95 (ms) - {workload}")
            ax.set_ylabel("ttft_p95_ms")
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            fig.tight_layout()
            fig.savefig(out_dir / f"exp1_{workload}_ttft_p95.png", dpi=200)
            plt.close(fig)

            # TTFT Mean bar chart
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(sub["policy"], sub["ttft_mean_ms"])
            ax.set_title(f"Exp1 TTFT Mean (ms) - {workload}")
            ax.set_ylabel("ttft_mean_ms")
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            fig.tight_layout()
            fig.savefig(out_dir / f"exp1_{workload}_ttft_mean.png", dpi=200)
            plt.close(fig)

            # Latency P95 bar chart
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(sub["policy"], sub["latency_p95_ms"])
            ax.set_title(f"Exp1 Latency P95 (ms) - {workload}")
            ax.set_ylabel("latency_p95_ms")
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            fig.tight_layout()
            fig.savefig(out_dir / f"exp1_{workload}_latency_p95.png", dpi=200)
            plt.close(fig)

        print(f"  Saved: exp1_routing.csv and {len(exp1['workload'].unique()) * 4} plots")

    # -------------------------------------------------------------------------
    # Experiment 2: Generate CSVs and plots
    # -------------------------------------------------------------------------
    if not exp2.empty:
        print("\nGenerating Experiment 2 outputs...")
        exp2.sort_values(["strategy", "scenario", "case"]).to_csv(out_dir / "exp2_failure.csv", index=False)

        # Delta total time plot
        fig, ax = plt.subplots(figsize=(12, 5))
        sub = exp2.sort_values(["strategy", "scenario", "case"])
        labels = (sub["strategy"] + "/" + sub["scenario"] + "/" + sub["case"]).tolist()
        ax.bar(range(len(sub)), sub["delta_total_time_pct"])
        ax.set_title("Exp2 ΔTotalTime (%) by Case")
        ax.set_ylabel("delta_total_time_pct")
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(out_dir / "exp2_delta_total_time_pct.png", dpi=200)
        plt.close(fig)

        print(f"  Saved: exp2_failure.csv and 1 plot")

    # -------------------------------------------------------------------------
    # Experiment 3: Generate CSVs and plots
    # -------------------------------------------------------------------------
    if not exp3.empty:
        print("\nGenerating Experiment 3 outputs...")
        exp3.sort_values(["concurrency", "strategy"]).to_csv(out_dir / "exp3_scheduling.csv", index=False)

        # Total time vs concurrency line plot
        fig, ax = plt.subplots(figsize=(8, 5))
        for strat in sorted(exp3["strategy"].unique()):
            sub = exp3[exp3["strategy"] == strat].sort_values("concurrency")
            ax.plot(sub["concurrency"], sub["total_time_s"], marker="o", label=strat)
        ax.set_title("Exp3 Total Time vs Workflow Concurrency")
        ax.set_xlabel("workflow_concurrency")
        ax.set_ylabel("total_time_s")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "exp3_total_time_vs_concurrency.png", dpi=200)
        plt.close(fig)

        print(f"  Saved: exp3_scheduling.csv and 1 plot")

    # -------------------------------------------------------------------------
    # Generate Markdown Report
    # -------------------------------------------------------------------------
    print("\nGenerating report...")
    report_lines: List[str] = []
    report_lines.append("# MARBLE Graph-Aware Serving Experiments - Results\n")
    report_lines.append(f"Input root: `{input_root}`\n")

    if (input_root / "run_meta.json").exists():
        meta = _read_json(input_root / "run_meta.json")
        report_lines.append("## Run Meta\n")
        report_lines.append("```json\n" + json.dumps(meta, ensure_ascii=False, indent=2) + "\n```\n")

    if not exp1.empty:
        report_lines.append("## Experiment 1 (Routing)\n")
        pivot = exp1.pivot_table(
            index=["workload"],
            columns=["policy"],
            values=["cache_hit_ratio", "ttft_p95_ms", "latency_p95_ms", "throughput_rps"],
            aggfunc="first",
        )
        report_lines.append(pivot.to_markdown())
        report_lines.append("")
        report_lines.append("### Plots\n")
        for workload in sorted(exp1["workload"].unique()):
            report_lines.append(f"- `analysis/exp1_{workload}_cache_hit.png`")
            report_lines.append(f"- `analysis/exp1_{workload}_ttft_p95.png`")
            report_lines.append(f"- `analysis/exp1_{workload}_ttft_mean.png`")
            report_lines.append(f"- `analysis/exp1_{workload}_latency_p95.png`")
        report_lines.append("")

    if not exp2.empty:
        report_lines.append("## Experiment 2 (Failure)\n")
        cols = [
            "strategy",
            "case",
            "delta_total_time_pct",
            "delta_latency_p95_ms",
            "blast_radius_steps",
            "recompute_prompt_tokens_retry",
            "recovery_time_s",
        ]
        report_lines.append(exp2[cols].to_markdown(index=False))
        report_lines.append("")
        report_lines.append("### Plots\n")
        report_lines.append("- `analysis/exp2_delta_total_time_pct.png`")
        report_lines.append("")

    if not exp3.empty:
        report_lines.append("## Experiment 3 (Scheduling)\n")
        cols = ["concurrency", "strategy", "total_time_s", "latency_p95_ms", "latency_p99_ms", "throughput_rps"]
        report_lines.append(exp3[cols].to_markdown(index=False))
        report_lines.append("")
        report_lines.append("### Plots\n")
        report_lines.append("- `analysis/exp3_total_time_vs_concurrency.png`")
        report_lines.append("")

    (out_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  Saved: report.md")

    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Results saved to: {out_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
