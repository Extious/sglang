"""
MARBLE Graph-Aware Serving Experiments Runner

This script runs the three main experiments from the research report:
1. Experiment 1: Routing policies vs cache hit & latency (H1)
2. Experiment 2: Failure injection + backup strategies (H2)
3. Experiment 3: Graph-aware (criticality) scheduling

Usage:
    python run_marble_graph_aware_experiments.py --model-path Qwen/Qwen3-4B-Thinking-2507

Environment Variables:
    MODEL_PATH: Default model path if not specified via CLI
    PYTHON_BIN: Python interpreter to use (default: python)
    SGLANG_EXTRA_PYTHONPATH: Additional PYTHONPATH for workers
    SGLANG_USE_REPO_SGL_KERNEL: Set to 1 to use repo sgl-kernel/python (default: use pip-installed sgl_kernel)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from marble_exp import (
    BenchmarkConfig,
    ClusterConfig,
    MarbleRequest,
    SGLangCluster,
    build_workflows_from_traces,
    clear_hicache_storage_backend,
    collect_worker_metrics,
    compute_cache_hit_ratio,
    flush_worker_cache,
    load_marble_traces,
    per_worker_deltas,
    run_benchmark,
)


@dataclass(frozen=True)
class WorkloadSpec:
    """Specification for a benchmark workload.

    Attributes:
        name: Human-readable workload name
        scenario: Scenario type (research, coding, or mixed)
        workflow_concurrency: Number of concurrent workflows
        num_traces: Number of traces to use (for single-scenario)
        num_traces_research: Number of research traces (for mixed)
        num_traces_coding: Number of coding traces (for mixed)
    """
    name: str
    scenario: str
    workflow_concurrency: int
    num_traces: int
    num_traces_research: int = 0
    num_traces_coding: int = 0


# =============================================================================
# Path Utilities
# =============================================================================

def _repo_root() -> Path:
    """Get the SGLang repository root directory."""
    return Path(__file__).resolve().parents[2]


def _script_dir() -> Path:
    """Get the directory containing this script."""
    return Path(__file__).resolve().parent


def _default_output_root() -> Path:
    """Generate default output directory with timestamp."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    return _script_dir() / "results" / f"{ts}_marble_graph_aware"


def _write_json(path: Path, obj: Any) -> None:
    """Write object as formatted JSON to file."""
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def _find_python_bin() -> str:
    """Find the Python interpreter to use.

    Priority:
    1. PYTHON_BIN environment variable
    2. sys.executable (current Python)
    """
    return os.getenv("PYTHON_BIN", sys.executable)


def _get_extra_pythonpath(repo_root: Path) -> str:
    """Build PYTHONPATH for worker processes.

    By default only SGLANG_EXTRA_PYTHONPATH is used. Set SGLANG_USE_REPO_SGL_KERNEL=1
    to prepend repo sgl-kernel/python (for dev when sgl-kernel is built in-repo).
    """
    parts = []

    # User-specified extra path
    user_path = os.getenv("SGLANG_EXTRA_PYTHONPATH", "")
    if user_path:
        parts.append(user_path)

    # sgl-kernel from repo only when explicitly requested (repo tree often has no compiled common_ops)
    if os.getenv("SGLANG_USE_REPO_SGL_KERNEL", "").strip() == "1":
        sgl_kernel_python = repo_root / "sgl-kernel" / "python"
        if sgl_kernel_python.exists():
            parts.append(str(sgl_kernel_python))

    return ":".join(parts) if parts else ""


# =============================================================================
# Experiment Helpers
# =============================================================================

def _run_one(
    *,
    worker_urls: List[str],
    workflows,
    cfg: BenchmarkConfig,
    out_dir: Path,
    priority_fn: Optional[Callable[[MarbleRequest], int]] = None,
) -> Dict[str, Any]:
    """Run a single benchmark configuration and collect metrics.

    Args:
        worker_urls: List of worker base URLs
        workflows: List of workflows to execute
        cfg: Benchmark configuration
        out_dir: Output directory for results
        priority_fn: Optional priority function for requests

    Returns:
        Statistics dictionary with cache hit ratio included
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    flush_worker_cache(worker_urls)

    metrics_before = collect_worker_metrics(worker_urls)
    import asyncio

    stats = asyncio.run(
        run_benchmark(workflows, worker_urls, cfg, output_dir=out_dir, priority_fn=priority_fn)
    )
    metrics_after = collect_worker_metrics(worker_urls)

    cache_hit = compute_cache_hit_ratio(metrics_before, metrics_after)
    deltas = per_worker_deltas(metrics_before, metrics_after)

    extra = {
        "cache_hit_ratio": cache_hit,
        "per_worker_deltas": deltas,
        "metrics_before": {k: {m: v for m, v in mv.items() if m.startswith("sglang:")} for k, mv in metrics_before.items()},
        "metrics_after": {k: {m: v for m, v in mv.items() if m.startswith("sglang:")} for k, mv in metrics_after.items()},
    }
    _write_json(out_dir / "metrics_summary.json", extra)

    stats_with_cache = dict(stats)
    stats_with_cache["cache_hit_ratio"] = cache_hit
    _write_json(out_dir / "summary_with_cache.json", stats_with_cache)
    return stats_with_cache


# =============================================================================
# Experiment 1: Routing Policies vs Cache Hit & Latency
# =============================================================================

def run_experiment_1(cluster: SGLangCluster, marble_dir: Path, out_root: Path) -> None:
    """Experiment 1: routing policies vs cache hit & latency (H1).

    Research dataset only, max_iter=1, workload research_para.
    Policies: random, agent_sticky_hash, workflow_sticky_hash, cache_aware.
    """
    exp_dir = out_root / "exp1_routing"
    exp_dir.mkdir(parents=True, exist_ok=True)

    workloads = [
        WorkloadSpec(name="research_para", scenario="research", workflow_concurrency=10, num_traces=20),
    ]

    policies = [
        "random",
        "agent_sticky_hash",
        "workflow_sticky_hash",
        "cache_aware",
    ]

    worker_urls = [ep.url for ep in cluster.worker_endpoints()]

    for wl in workloads:
        traces = load_marble_traces(marble_dir, wl.scenario, wl.num_traces)
        workflows = build_workflows_from_traces(traces, max_iter_override=1)

        for policy in policies:
            out_dir = exp_dir / wl.name / policy
            if (out_dir / "summary_with_cache.json").exists():
                continue

            cfg = BenchmarkConfig(
                routing_policy=policy,
                workflow_concurrency=wl.workflow_concurrency,
                # Keep decode short: focus on TTFT/prefix-cache effects
                max_completion_tokens=64,
                temperature=0.7,
                request_timeout_s=900,
                max_retries=0,
                extra_key_mode="none",
                router_url=None,
            )
            print(f"  Running: {wl.name} / {policy}")
            _run_one(worker_urls=worker_urls, workflows=workflows, cfg=cfg, out_dir=out_dir)
            print(f"  Completed: {wl.name} / {policy}")


def _sum_delta_prompt_tokens(metrics_summary_path: Path) -> float:
    """Sum delta_prompt_tokens across all workers from metrics summary."""
    data = json.loads(metrics_summary_path.read_text())
    return float(sum(v.get("delta_prompt_tokens", 0.0) for v in data.get("per_worker_deltas", {}).values()))


def _summarize_failure_records(records_path: Path, inject_ts: float) -> Dict[str, Any]:
    """Analyze failure records to compute recovery metrics.

    Args:
        records_path: Path to records.jsonl file
        inject_ts: Timestamp when failure was injected

    Returns:
        Dictionary with failed_requests, retried_requests, and recovery_time_s
    """
    failed = 0
    retried = 0
    first_success_after: Optional[float] = None
    with records_path.open() as f:
        for line in f:
            r = json.loads(line)
            if not r.get("success"):
                failed += 1
            if int(r.get("attempt", 0)) > 0:
                retried += 1
            if r.get("success") and r.get("finished_at") is not None:
                finished_at = float(r["finished_at"])
                if finished_at >= inject_ts:
                    first_success_after = finished_at if first_success_after is None else min(first_success_after, finished_at)

    recovery_time_s = (first_success_after - inject_ts) if first_success_after is not None else None
    return {
        "failed_requests": failed,
        "retried_requests": retried,
        "recovery_time_s": recovery_time_s,
    }


# =============================================================================
# Experiment 2: Failure Injection + Backup Strategies
# =============================================================================

def run_experiment_2_suite(
    repo_root: Path,
    marble_dir: Path,
    out_root: Path,
    model_path: str,
    log_dir: Path,
    python_bin: str,
    extra_pythonpath: str,
) -> None:
    """Experiment 2: failure injection + backup strategies (H2).

    Research dataset only. Tests KV cache backup strategies under worker failures.

    Strategies:
    - M0-NoBackup: Plain radix cache only
    - M1-AllBackup: HiCache on all workers
    - M2-TopK2 (research): HiCache on agent1+agent2 (GPU 0/5/1/6)

    Failure case: fail_agent1_gpu0 ([0]).
    """
    exp_dir = out_root / "exp2_failure"
    exp_dir.mkdir(parents=True, exist_ok=True)

    worker_base_port = int(os.getenv("WORKER_BASE_PORT", "8000"))
    print(f"Experiment 2 worker base port: {worker_base_port} (set WORKER_BASE_PORT to avoid port conflict)")

    print("Loading workflows for experiment 2 (research only)...")
    research_workflows = build_workflows_from_traces(load_marble_traces(marble_dir, "research", 20))

    failure_cases_research = [
        ("fail_agent1_gpu0", [0]),
    ]

    def _run_strategy(
        strategy_name: str,
        *,
        enable_storage_gpus: List[int],
        run_coding: bool,
        run_research: bool,
    ) -> None:
        """Run a single backup strategy configuration."""
        print(f"\n{'='*60}")
        print(f"Running strategy: {strategy_name}")
        print(f"{'='*60}")

        strat_dir = exp_dir / strategy_name
        strat_dir.mkdir(parents=True, exist_ok=True)

        extra_args_by_gpu: Dict[int, List[str]] = {}
        extra_env_by_gpu: Dict[int, Dict[str, str]] = {}

        # Configure HiCache for specified GPUs
        if enable_storage_gpus:
            storage_dir = out_root / "hicache_storage" / strategy_name
            storage_dir.mkdir(parents=True, exist_ok=True)
            for i in range(10):
                extra_env_by_gpu[i] = {"SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": str(storage_dir)}
                if i in enable_storage_gpus:
                    extra_args_by_gpu[i] = [
                        "--enable-hierarchical-cache",
                        "--hicache-write-policy",
                        "write_through",
                        "--hicache-storage-backend",
                        "file",
                        "--hicache-storage-backend-extra-config",
                        '{"prefetch_threshold":64}',
                    ]

        cfg = ClusterConfig(
            model_path=model_path,
            worker_base_port=worker_base_port,
            wait_ready_timeout_s=1800,
            worker_start_check_s=10,
            worker_startup_delay_s=8,
            python_bin=python_bin,
            extra_pythonpath=extra_pythonpath,
            include_repo_python=True,
            worker_common_args=[
                "--attention-backend", "triton",
                "--sampling-backend", "pytorch",
                "--grammar-backend", "none",
            ],
        )
        with SGLangCluster(repo_root=repo_root, cfg=cfg, log_dir=log_dir / "exp2" / strategy_name) as cluster:
            cluster.start_workers(extra_args_by_gpu=extra_args_by_gpu, extra_env_by_gpu=extra_env_by_gpu)
            worker_urls = [ep.url for ep in cluster.worker_endpoints()]

            def _run_baseline_and_cases(scenario_name: str, workflows, cases: List[Tuple[str, List[int]]]) -> None:
                base_cfg = BenchmarkConfig(
                    routing_policy="agent_primary",
                    workflow_concurrency=10,
                    max_completion_tokens=64,
                    temperature=0.7,
                    request_timeout_s=900,
                    max_retries=1,
                    retry_backoff_s=0.5,
                )

                scenario_dir = strat_dir / scenario_name
                scenario_dir.mkdir(parents=True, exist_ok=True)

                baseline_dir = scenario_dir / "baseline_no_failure"
                if not (baseline_dir / "summary_with_cache.json").exists():
                    print(f"    Running baseline_no_failure (sending requests to workers)...")
                    flush_worker_cache(worker_urls)
                    clear_hicache_storage_backend(worker_urls)
                    _run_one(worker_urls=worker_urls, workflows=workflows, cfg=base_cfg, out_dir=baseline_dir)

                baseline_stats = json.loads((baseline_dir / "summary_with_cache.json").read_text())
                baseline_prompt_tokens = _sum_delta_prompt_tokens(baseline_dir / "metrics_summary.json")
                baseline_total_time_s = float(baseline_stats.get("total_time_s") or 0.0)

                for case_name, fail_gpus in cases:
                    case_dir = scenario_dir / case_name
                    if (case_dir / "failure_summary.json").exists():
                        print(f"    Skipping {case_name} (already completed)")
                        continue

                    print(f"    Running failure case: {case_name} (killing GPUs {fail_gpus})")
                    flush_worker_cache(worker_urls)
                    clear_hicache_storage_backend(worker_urls)

                    # Calculate injection delay based on baseline duration
                    # Ensure injection happens during the benchmark, not after
                    if baseline_total_time_s > 2.0:
                        # Inject at 30% of baseline time to ensure it happens mid-run
                        inject_delay_s = max(1.0, min(30.0, baseline_total_time_s * 0.3))
                    else:
                        # For very fast benchmarks, inject after 1 second
                        inject_delay_s = 1.0

                    # Warn if injection might happen after benchmark completes
                    if inject_delay_s >= baseline_total_time_s * 0.8:
                        print(f"      Warning: inject_delay_s ({inject_delay_s:.1f}s) is close to "
                              f"baseline time ({baseline_total_time_s:.1f}s)")

                    fail_out_dir = case_dir / "with_failure"
                    fail_out_dir.mkdir(parents=True, exist_ok=True)

                    metrics_before = collect_worker_metrics(worker_urls)
                    start_epoch = time.time()

                    import asyncio

                    async def _run_with_injection():
                        injection_done = asyncio.Event()

                        async def _inject():
                            await asyncio.sleep(inject_delay_s)
                            print(f"      Injecting failure at t+{inject_delay_s:.1f}s...")
                            for gid in fail_gpus:
                                cluster.kill_worker(gid)
                            injection_done.set()

                        inj_task = asyncio.create_task(_inject())
                        try:
                            stats = await run_benchmark(
                                workflows, worker_urls, base_cfg, output_dir=fail_out_dir
                            )
                        finally:
                            # Wait for injection to complete if benchmark finished early
                            if not injection_done.is_set():
                                inj_task.cancel()
                                try:
                                    await inj_task
                                except asyncio.CancelledError:
                                    pass
                        return stats

                    stats = asyncio.run(_run_with_injection())
                    end_epoch = time.time()
                    metrics_after = collect_worker_metrics(worker_urls)

                    cache_hit = compute_cache_hit_ratio(metrics_before, metrics_after)
                    deltas = per_worker_deltas(metrics_before, metrics_after)
                    _write_json(
                        fail_out_dir / "metrics_summary.json",
                        {
                            "cache_hit_ratio": cache_hit,
                            "per_worker_deltas": deltas,
                        },
                    )
                    _write_json(
                        fail_out_dir / "summary_with_cache.json",
                        {**stats, "cache_hit_ratio": cache_hit},
                    )

                    fail_prompt_tokens = float(sum(v.get("delta_prompt_tokens", 0.0) for v in deltas.values()))
                    recompute_tokens = max(0.0, fail_prompt_tokens - baseline_prompt_tokens)

                    inject_ts = start_epoch + inject_delay_s
                    record_summary = _summarize_failure_records(fail_out_dir / "records.jsonl", inject_ts=inject_ts)

                    failure_summary = {
                        "strategy": strategy_name,
                        "scenario": scenario_name,
                        "case": case_name,
                        "fail_gpu_ids": fail_gpus,
                        "inject_delay_s": inject_delay_s,
                        "baseline": baseline_stats,
                        "with_failure": json.loads((fail_out_dir / "summary_with_cache.json").read_text()),
                        "delta_total_time_pct": (
                            (stats["total_time_s"] - baseline_stats["total_time_s"]) / baseline_stats["total_time_s"] * 100.0
                            if baseline_stats.get("total_time_s", 0) > 0
                            else 0.0
                        ),
                        "recompute_tokens_approx": recompute_tokens,
                        "started_at": start_epoch,
                        "injected_at": inject_ts,
                        "finished_at": end_epoch,
                        **record_summary,
                    }
                    _write_json(case_dir / "failure_summary.json", failure_summary)

                    print(f"    Completed: {case_name}")

                    # Restore cluster for the next case
                    print(f"    Restarting killed workers: {fail_gpus}")
                    cluster.restart_workers(fail_gpus)

            if run_research:
                print(f"  Running research scenario...")
                _run_baseline_and_cases("research", research_workflows, failure_cases_research)

    # Run strategies (research only): M0-NoBackup, M1-AllBackup, M2-TopK2 (research)
    print("\n" + "="*60)
    print("Experiment 2: Running backup strategy matrix (research only)")
    print("="*60)
    _run_strategy("M0_NoBackup", enable_storage_gpus=[], run_coding=False, run_research=True)
    _run_strategy("M1_AllBackup", enable_storage_gpus=list(range(10)), run_coding=False, run_research=True)
    _run_strategy("M2_TopK2_Research", enable_storage_gpus=[0, 5, 1, 6], run_coding=False, run_research=True)


# =============================================================================
# Experiment 3: Graph-Aware Scheduling
# =============================================================================

def run_experiment_3_scheduling(cluster: SGLangCluster, marble_dir: Path, out_root: Path) -> None:
    """Experiment 3: graph-aware (criticality) scheduling via request priority.

    Tests different scheduling strategies to prioritize critical agents.

    Strategies:
    - fifo: First-in-first-out (baseline)
    - criticality_priority: Prioritize agent1 > agent2 > agent3
    - sjf: Shortest job first (by prompt length)
    - criticality_plus_sjf: Combined criticality and SJF

    Concurrency levels tested: 10, 20, 50
    """
    exp_dir = out_root / "exp3_scheduling"
    exp_dir.mkdir(parents=True, exist_ok=True)

    worker_urls = [ep.url for ep in cluster.worker_endpoints()]

    coding_traces = load_marble_traces(marble_dir, "coding", 50)
    workflows = build_workflows_from_traces(coding_traces)

    def criticality_priority(req) -> int:
        # Higher integer = higher priority in SGLang by default.
        if req.agent_id == "agent1":
            return 300
        if req.agent_id == "agent2":
            return 200
        return 100

    def sjf_priority(req) -> int:
        # Approximate "shortest job first" using prompt char length as proxy.
        prompt_chars = sum(len(m.get("content", "")) for m in req.messages)
        return -prompt_chars

    def criticality_sjf_priority(req) -> int:
        return criticality_priority(req) * 10_000_000 + sjf_priority(req)

    strategies: List[Tuple[str, Optional[Any]]] = [
        ("fifo", None),
        ("criticality_priority", criticality_priority),
        ("sjf", sjf_priority),
        ("criticality_plus_sjf", criticality_sjf_priority),
    ]

    for conc in (10, 20, 50):
        for name, fn in strategies:
            out_dir = exp_dir / f"concurrency_{conc}" / name
            if (out_dir / "summary_with_cache.json").exists():
                print(f"  Skipping: concurrency={conc}, strategy={name} (already completed)")
                continue
            print(f"  Running: concurrency={conc}, strategy={name}")
            cfg = BenchmarkConfig(
                routing_policy="round_robin",
                workflow_concurrency=conc,
                max_completion_tokens=64,
                temperature=0.7,
                request_timeout_s=900,
                max_retries=0,
            )
            _run_one(worker_urls=worker_urls, workflows=workflows, cfg=cfg, out_dir=out_dir, priority_fn=fn)
            print(f"  Completed: concurrency={conc}, strategy={name}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    """Main entry point for running all experiments."""
    parser = argparse.ArgumentParser(
        description="Run MARBLE Graph-Aware Serving Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default settings
    python run_marble_graph_aware_experiments.py

    # Specify model path
    python run_marble_graph_aware_experiments.py --model-path Qwen/Qwen3-4B-Thinking-2507

    # Custom output directory
    python run_marble_graph_aware_experiments.py --output-root ./my_results

Environment Variables:
    MODEL_PATH              Default model path
    PYTHON_BIN              Python interpreter to use
    SGLANG_EXTRA_PYTHONPATH Additional PYTHONPATH for workers
        """
    )
    parser.add_argument(
        "--model-path",
        default=os.getenv("MODEL_PATH", "Qwen/Qwen3-4B-Thinking-2507"),
        help="HuggingFace model path (default: Qwen/Qwen3-4B-Thinking-2507)"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output directory for results (default: auto-generated with timestamp)"
    )
    parser.add_argument(
        "--skip-exp1",
        action="store_true",
        help="Skip experiment 1 (routing policies)"
    )
    parser.add_argument(
        "--skip-exp2",
        action="store_true",
        help="Skip experiment 2 (failure injection)"
    )
    parser.add_argument(
        "--skip-exp3",
        action="store_true",
        help="Skip experiment 3 (scheduling)"
    )
    args = parser.parse_args()

    # Setup paths
    repo_root = _repo_root()
    marble_dir = _script_dir() / "MARBLE"
    out_root: Path = args.output_root or _default_output_root()
    out_root.mkdir(parents=True, exist_ok=True)
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Get Python interpreter and PYTHONPATH
    python_bin = _find_python_bin()
    extra_pythonpath = _get_extra_pythonpath(repo_root)

    print("="*60)
    print("MARBLE Graph-Aware Serving Experiments")
    print("="*60)
    print(f"Model path:    {args.model_path}")
    print(f"Output root:   {out_root}")
    print(f"Python bin:    {python_bin}")
    print(f"MARBLE dir:    {marble_dir}")
    worker_base_port = int(os.getenv("WORKER_BASE_PORT", "8000"))
    print(f"Worker ports:  {worker_base_port}-{worker_base_port + 9} (set WORKER_BASE_PORT if port in use)")
    print("="*60)

    # Verify MARBLE dataset exists
    if not marble_dir.exists():
        print(f"ERROR: MARBLE dataset not found at {marble_dir}")
        print("Please run: python download.sh or manually download the dataset")
        sys.exit(1)

    # Save run metadata
    _write_json(
        out_root / "run_meta.json",
        {
            "model_path": args.model_path,
            "repo_root": str(repo_root),
            "marble_dir": str(marble_dir),
            "python_bin": python_bin,
            "extra_pythonpath": extra_pythonpath,
            "worker_base_port": worker_base_port,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    # Common cluster configuration
    base_cluster_cfg = ClusterConfig(
        model_path=args.model_path,
        worker_base_port=worker_base_port,
        wait_ready_timeout_s=1800,
        worker_start_check_s=10,
        worker_startup_delay_s=8,
        python_bin=python_bin,
        extra_pythonpath=extra_pythonpath,
        include_repo_python=True,
        worker_common_args=[
            "--attention-backend", "triton",
            "--sampling-backend", "pytorch",
            "--grammar-backend", "none",
        ],
    )

    # -------------------------------------------------------------------------
    # Experiment 1: Routing Policies
    # -------------------------------------------------------------------------
    if not args.skip_exp1:
        exp1_expected = 1 * 4  # 1 workload (research_para) × 4 policies
        exp1_existing = len(list((out_root / "exp1_routing").rglob("summary_with_cache.json")))
        if exp1_existing < exp1_expected:
            print("\n" + "="*60)
            print("Experiment 1: Routing Policies vs Cache Hit & Latency")
            print(f"Progress: {exp1_existing}/{exp1_expected} completed")
            print("="*60)
            with SGLangCluster(repo_root=repo_root, cfg=base_cluster_cfg, log_dir=log_dir / "exp1") as cluster:
                cluster.start_workers()
                run_experiment_1(cluster, marble_dir, out_root)
        else:
            print(f"\nExperiment 1: Already completed ({exp1_existing}/{exp1_expected})")

    # -------------------------------------------------------------------------
    # Experiment 2: Failure Injection
    # -------------------------------------------------------------------------
    if not args.skip_exp2:
        print("\n" + "="*60)
        print("Experiment 2: Failure Injection + Backup Strategies")
        print("="*60)
        run_experiment_2_suite(
            repo_root, marble_dir, out_root, args.model_path, log_dir,
            python_bin=python_bin,
            extra_pythonpath=extra_pythonpath,
        )

    # -------------------------------------------------------------------------
    # Experiment 3: Scheduling
    # -------------------------------------------------------------------------
    if not args.skip_exp3:
        exp3_expected = 3 * 4  # conc ∈ {10,20,50} × 4 strategies
        exp3_existing = len(list((out_root / "exp3_scheduling").rglob("summary_with_cache.json")))
        if exp3_existing < exp3_expected:
            print("\n" + "="*60)
            print("Experiment 3: Graph-Aware Scheduling")
            print(f"Progress: {exp3_existing}/{exp3_expected} completed")
            print("="*60)

            # Enable priority scheduling on all workers
            extra_args3 = {
                i: [
                    "--schedule-policy", "fcfs",
                    "--enable-priority-scheduling",
                    "--priority-scheduling-preemption-threshold", "1000000",
                ]
                for i in range(10)
            }

            cfg3 = ClusterConfig(
                model_path=args.model_path,
                worker_base_port=worker_base_port,
                wait_ready_timeout_s=1800,
                worker_start_check_s=10,
                worker_startup_delay_s=8,
                python_bin=python_bin,
                extra_pythonpath=extra_pythonpath,
                include_repo_python=True,
                worker_common_args=[
                    "--attention-backend", "triton",
                    "--sampling-backend", "pytorch",
                    "--grammar-backend", "none",
                ],
            )
            with SGLangCluster(repo_root=repo_root, cfg=cfg3, log_dir=log_dir / "exp3") as cluster:
                cluster.start_workers(extra_args_by_gpu=extra_args3)
                run_experiment_3_scheduling(cluster, marble_dir, out_root)
        else:
            print(f"\nExperiment 3: Already completed ({exp3_existing}/{exp3_expected})")

    print("\n" + "="*60)
    print("All experiments completed!")
    print(f"Results saved to: {out_root}")
    print("="*60)


if __name__ == "__main__":
    main()
