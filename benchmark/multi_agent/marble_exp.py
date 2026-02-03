"""
MARBLE Multi-Agent Benchmark Experiment Library

This module provides utilities for running multi-agent LLM serving experiments
using the MARBLE dataset. It includes:
- SGLang cluster management (start/stop/restart workers)
- Various routing policies (round-robin, agent-sticky, cache-aware, etc.)
- Metrics collection and cache hit ratio computation
- Workflow execution with configurable concurrency

Usage:
    from marble_exp import (
        SGLangCluster, ClusterConfig, BenchmarkConfig,
        load_marble_traces, build_workflows_from_traces, run_benchmark
    )
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

# Check dependencies
try:
    import aiohttp
    import numpy as np
    import requests
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install aiohttp numpy requests")
    sys.exit(1)

# Disable proxy for localhost connections
_LOCALHOST_NO_PROXY = "localhost,127.0.0.1,0.0.0.0,::1"
os.environ.setdefault("NO_PROXY", _LOCALHOST_NO_PROXY)
os.environ.setdefault("no_proxy", os.environ.get("NO_PROXY", _LOCALHOST_NO_PROXY))


def _stable_int_hash(text: str) -> int:
    """Generate a stable integer hash from text.

    Uses SHA256 to ensure consistency across processes and Python versions.
    """
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _now_ts() -> float:
    """Return current Unix timestamp."""
    return time.time()


def _ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def _percentile_ms(values_s: List[float], p: float) -> float:
    """Calculate percentile of values (in seconds) and return in milliseconds."""
    if not values_s:
        return 0.0
    return float(np.percentile(np.array(values_s) * 1000.0, p))


def _summarize_latencies(values_s: List[float]) -> Dict[str, float]:
    """Compute latency statistics (P50, P95, P99, mean) in milliseconds."""
    return {
        "p50_ms": _percentile_ms(values_s, 50),
        "p95_ms": _percentile_ms(values_s, 95),
        "p99_ms": _percentile_ms(values_s, 99),
        "mean_ms": float(np.mean(np.array(values_s) * 1000.0)) if values_s else 0.0,
    }


def _parse_prometheus_text(text: str) -> Dict[str, float]:
    """Parse Prometheus text format and sum samples by metric name.

    Args:
        text: Raw Prometheus metrics text

    Returns:
        Dictionary mapping metric names to their summed values
    """
    out: Dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Format: name{labels} value [ts]
        # Or:     name value [ts]
        name_and_labels, _, value_str = line.partition(" ")
        if not value_str:
            continue
        name = name_and_labels.split("{", 1)[0]
        try:
            value = float(value_str.split(" ", 1)[0])
        except ValueError:
            continue
        out[name] = out.get(name, 0.0) + value
    return out


@dataclasses.dataclass(frozen=True)
class WorkerEndpoint:
    """Represents a single SGLang worker endpoint."""
    gpu_id: int
    url: str
    port: int


@dataclasses.dataclass
class ClusterConfig:
    """Configuration for SGLang cluster deployment.

    Attributes:
        model_path: HuggingFace model path or local path
        num_workers: Number of worker processes (typically one per GPU)
        worker_base_port: Starting port number for workers
        router_port: Port for the optional router process
        host: Host address to bind workers
        python_bin: Path to Python interpreter
        python_flags: Additional flags for Python interpreter
        extra_pythonpath: Additional paths to prepend to PYTHONPATH
        include_repo_python: Whether to include repo's python/ in PYTHONPATH
        worker_common_args: Common arguments for all workers
        worker_startup_delay_s: Delay between starting workers (reduces NFS pressure)
        worker_start_check_s: Time to wait before checking if worker started
        worker_start_retries: Number of retries if worker fails to start
        worker_retry_delay_s: Delay between retries
        wait_ready_timeout_s: Timeout for waiting workers to be ready
        wait_ready_poll_s: Polling interval for health checks
    """
    model_path: str
    num_workers: int = 10
    worker_base_port: int = 8000
    router_port: int = 30000
    host: str = "127.0.0.1"
    python_bin: str = "python"  # Default to system python
    python_flags: List[str] = dataclasses.field(default_factory=list)
    extra_pythonpath: Optional[str] = None
    include_repo_python: bool = True
    worker_common_args: List[str] = dataclasses.field(default_factory=list)
    worker_startup_delay_s: int = 12
    worker_start_check_s: int = 30
    worker_start_retries: int = 3
    worker_retry_delay_s: int = 10
    wait_ready_timeout_s: int = 180
    wait_ready_poll_s: float = 2.0


class SGLangCluster:
    """Manages a cluster of SGLang worker processes.

    This class handles starting, stopping, and monitoring multiple SGLang
    worker processes, each typically running on a separate GPU.

    Example:
        cfg = ClusterConfig(model_path="Qwen/Qwen3-4B-Thinking-2507")
        with SGLangCluster(repo_root, cfg, log_dir) as cluster:
            cluster.start_workers()
            # Run experiments...
    """

    def __init__(self, repo_root: Path, cfg: ClusterConfig, log_dir: Path):
        """Initialize the cluster manager.

        Args:
            repo_root: Path to the SGLang repository root
            cfg: Cluster configuration
            log_dir: Directory for worker log files
        """
        self.repo_root = repo_root
        self.cfg = cfg
        self.log_dir = log_dir
        _ensure_dir(self.log_dir)

        self.worker_procs: Dict[int, subprocess.Popen] = {}
        self.router_proc: Optional[subprocess.Popen] = None
        self._extra_args_by_gpu: Dict[int, List[str]] = {}
        self._extra_env_by_gpu: Dict[int, Dict[str, str]] = {}

    def worker_endpoints(self) -> List[WorkerEndpoint]:
        """Get list of all worker endpoints."""
        eps: List[WorkerEndpoint] = []
        for gpu_id in range(self.cfg.num_workers):
            port = self.cfg.worker_base_port + gpu_id
            eps.append(
                WorkerEndpoint(
                    gpu_id=gpu_id,
                    port=port,
                    url=f"http://{self.cfg.host}:{port}",
                )
            )
        return eps

    def _base_env(self) -> Dict[str, str]:
        """Build base environment variables for worker processes."""
        env = os.environ.copy()
        pp_parts = []
        if self.cfg.extra_pythonpath:
            pp_parts.append(self.cfg.extra_pythonpath)
        if self.cfg.include_repo_python:
            pp_parts.append(str(self.repo_root / "python"))
        existing_pp = env.get("PYTHONPATH", "")
        if existing_pp:
            pp_parts.append(existing_pp)
        env["PYTHONPATH"] = ":".join(pp_parts)
        env["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0,::1"
        env["no_proxy"] = "localhost,127.0.0.1,0.0.0.0,::1"
        env.setdefault("LC_ALL", "C.UTF-8")
        env.setdefault("LANG", "C.UTF-8")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        return env

    def start_workers(
        self,
        extra_args_by_gpu: Optional[Dict[int, List[str]]] = None,
        extra_env_by_gpu: Optional[Dict[int, Dict[str, str]]] = None,
    ) -> None:
        """Start all worker processes.

        Args:
            extra_args_by_gpu: Additional CLI arguments per GPU
            extra_env_by_gpu: Additional environment variables per GPU
        """
        extra_args_by_gpu = extra_args_by_gpu or {}
        extra_env_by_gpu = extra_env_by_gpu or {}
        # Persist for potential restarts.
        self._extra_args_by_gpu = {k: list(v) for k, v in extra_args_by_gpu.items()}
        self._extra_env_by_gpu = {k: dict(v) for k, v in extra_env_by_gpu.items()}

        for gpu_id in range(self.cfg.num_workers):
            print(f"  Starting worker {gpu_id} (port {self.cfg.worker_base_port + gpu_id})...")
            self._start_one_worker(
                gpu_id,
                extra_args=extra_args_by_gpu.get(gpu_id, []),
                extra_env=extra_env_by_gpu.get(gpu_id, {}),
                truncate_log=True,
            )

        print(f"  Waiting for all {self.cfg.num_workers} workers to be ready (health check, timeout={self.cfg.wait_ready_timeout_s}s)...")
        self.wait_until_ready()
        print("  All workers ready.")

    def _start_one_worker(
        self,
        gpu_id: int,
        *,
        extra_args: List[str],
        extra_env: Dict[str, str],
        truncate_log: bool,
    ) -> None:
        port = self.cfg.worker_base_port + gpu_id
        log_path = self.log_dir / f"worker_{gpu_id}.log"
        if truncate_log:
            log_path.write_text("")

        env = self._base_env()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env.update(extra_env)

        args = [
            self.cfg.python_bin,
            *self.cfg.python_flags,
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.cfg.model_path,
            "--tp",
            "1",
            "--host",
            self.cfg.host,
            "--port",
            str(port),
            "--enable-metrics",
        ]
        args.extend(self.cfg.worker_common_args)
        args.extend(extra_args)

        proc: Optional[subprocess.Popen] = None
        for attempt in range(1, self.cfg.worker_start_retries + 1):
            proc = subprocess.Popen(
                args,
                env=env,
                stdout=log_path.open("ab"),
                stderr=subprocess.STDOUT,
            )
            time.sleep(self.cfg.worker_start_check_s)
            if proc.poll() is None:
                self.worker_procs[gpu_id] = proc
                break

            # Exited early; retry.
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
            if attempt < self.cfg.worker_start_retries:
                time.sleep(self.cfg.worker_retry_delay_s)

        if proc is None or proc.poll() is not None:
            raise RuntimeError(
                f"Worker {gpu_id} failed to start. Check {log_path} for details."
            )

        time.sleep(self.cfg.worker_startup_delay_s)

    def wait_until_ready(self) -> None:
        """Wait for all workers to be ready (health check passes)."""
        deadline = time.time() + self.cfg.wait_ready_timeout_s
        endpoints = self.worker_endpoints()
        for ep in endpoints:
            ok = False
            while time.time() < deadline:
                try:
                    r = requests.get(f"{ep.url}/health", timeout=5)
                    if r.status_code == 200:
                        ok = True
                        break
                except Exception:
                    pass
                time.sleep(self.cfg.wait_ready_poll_s)
            if not ok:
                raise TimeoutError(
                    f"Timeout waiting for worker GPU {ep.gpu_id} to be ready at {ep.url}"
                )
            print(f"    Worker {ep.gpu_id} ready.")

    def start_router(self, policy: str = "cache_aware") -> None:
        """Start the SGLang router process.

        Args:
            policy: Routing policy (e.g., "cache_aware", "round_robin")
        """
        if self.router_proc and self.router_proc.poll() is None:
            return

        worker_urls = [ep.url for ep in self.worker_endpoints()]
        log_path = self.log_dir / "router.log"
        log_path.write_text("")

        env = self._base_env()
        args = [
            self.cfg.python_bin,
            *self.cfg.python_flags,
            "-m",
            "sglang_router.launch_router",
            "--worker-urls",
            *worker_urls,
            "--policy",
            policy,
            "--host",
            "0.0.0.0",
            "--port",
            str(self.cfg.router_port),
            "--model-path",
            self.cfg.model_path,
        ]

        self.router_proc = subprocess.Popen(
            args, env=env, stdout=log_path.open("ab"), stderr=subprocess.STDOUT
        )

        # Wait until router is ready.
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                r = requests.get(
                    f"http://{self.cfg.host}:{self.cfg.router_port}/v1/models",
                    timeout=5,
                )
                if r.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(1)
        raise TimeoutError(f"Timeout waiting for router on port {self.cfg.router_port}")

    def stop_router(self) -> None:
        """Stop the router process if running."""
        if not self.router_proc:
            return
        if self.router_proc.poll() is None:
            self.router_proc.send_signal(signal.SIGTERM)
            try:
                self.router_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.router_proc.kill()
        self.router_proc = None

    def stop_workers(self) -> None:
        """Stop all worker processes."""
        for gpu_id, proc in list(self.worker_procs.items()):
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        for gpu_id, proc in list(self.worker_procs.items()):
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.worker_procs.clear()

    def kill_worker(self, gpu_id: int, sig: int = signal.SIGKILL) -> None:
        """Kill a specific worker process.

        Args:
            gpu_id: GPU ID of the worker to kill
            sig: Signal to send (default: SIGKILL)
        """
        proc = self.worker_procs.get(gpu_id)
        if not proc:
            return
        if proc.poll() is None:
            proc.send_signal(sig)

    def restart_workers(self, gpu_ids: List[int]) -> None:
        """Restart a subset of workers using the originally supplied args/env.

        Args:
            gpu_ids: List of GPU IDs to restart
        """
        for gpu_id in gpu_ids:
            old = self.worker_procs.get(gpu_id)
            if old and old.poll() is None:
                old.send_signal(signal.SIGTERM)
                try:
                    old.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    old.kill()
            # Remove stale proc entry.
            self.worker_procs.pop(gpu_id, None)

            self._start_one_worker(
                gpu_id,
                extra_args=self._extra_args_by_gpu.get(gpu_id, []),
                extra_env=self._extra_env_by_gpu.get(gpu_id, {}),
                truncate_log=False,
            )

        # Wait ready for restarted workers only.
        deadline = time.time() + self.cfg.wait_ready_timeout_s
        for gpu_id in gpu_ids:
            port = self.cfg.worker_base_port + gpu_id
            url = f"http://{self.cfg.host}:{port}"
            ok = False
            while time.time() < deadline:
                try:
                    r = requests.get(f"{url}/health", timeout=5)
                    if r.status_code == 200:
                        ok = True
                        break
                except Exception:
                    pass
                time.sleep(self.cfg.wait_ready_poll_s)
            if not ok:
                raise TimeoutError(f"Timeout waiting for restarted worker GPU {gpu_id}")

    def __enter__(self) -> "SGLangCluster":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop_router()
        self.stop_workers()


def flush_worker_cache(worker_urls: List[str]) -> None:
    """Flush the KV cache on all workers.

    Args:
        worker_urls: List of worker base URLs
    """
    for url in worker_urls:
        try:
            requests.post(f"{url}/flush_cache", timeout=10)
        except Exception:
            pass


def clear_hicache_storage_backend(worker_urls: List[str]) -> None:
    """Clear the HiCache storage backend on all workers.

    Args:
        worker_urls: List of worker base URLs
    """
    for url in worker_urls:
        try:
            requests.post(f"{url}/clear_hicache_storage_backend", timeout=10)
        except Exception:
            pass


def collect_worker_metrics(worker_urls: List[str]) -> Dict[str, Dict[str, float]]:
    """Collect Prometheus metrics from all workers.

    Args:
        worker_urls: List of worker base URLs

    Returns:
        Dictionary mapping worker URL to its metrics
    """
    out: Dict[str, Dict[str, float]] = {}
    for url in worker_urls:
        try:
            r = requests.get(f"{url}/metrics", timeout=10)
            r.raise_for_status()
            out[url] = _parse_prometheus_text(r.text)
        except Exception:
            out[url] = {}
    return out


def compute_cache_hit_ratio(
    before: Dict[str, Dict[str, float]],
    after: Dict[str, Dict[str, float]]
) -> float:
    """Compute cache hit ratio from before/after metrics snapshots.

    Cache hit ratio = cached_tokens / (cached_tokens + prompt_tokens)

    Args:
        before: Metrics snapshot before experiment
        after: Metrics snapshot after experiment

    Returns:
        Cache hit ratio (0.0 to 1.0)
    """
    delta_cached = 0.0
    delta_prompt = 0.0
    for url in before.keys() | after.keys():
        b = before.get(url, {})
        a = after.get(url, {})
        delta_cached += a.get("sglang:cached_tokens_total", 0.0) - b.get(
            "sglang:cached_tokens_total", 0.0
        )
        delta_prompt += a.get("sglang:prompt_tokens_total", 0.0) - b.get(
            "sglang:prompt_tokens_total", 0.0
        )
    denom = delta_cached + delta_prompt
    return float(delta_cached / denom) if denom > 0 else 0.0


def per_worker_deltas(
    before: Dict[str, Dict[str, float]],
    after: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """Compute per-worker metric deltas.

    Args:
        before: Metrics snapshot before experiment
        after: Metrics snapshot after experiment

    Returns:
        Dictionary mapping worker URL to its metric deltas
    """
    out: Dict[str, Dict[str, float]] = {}
    for url in before.keys() | after.keys():
        b = before.get(url, {})
        a = after.get(url, {})
        out[url] = {
            "delta_num_requests": a.get("sglang:num_requests_total", 0.0)
            - b.get("sglang:num_requests_total", 0.0),
            "delta_prompt_tokens": a.get("sglang:prompt_tokens_total", 0.0)
            - b.get("sglang:prompt_tokens_total", 0.0),
            "delta_cached_tokens": a.get("sglang:cached_tokens_total", 0.0)
            - b.get("sglang:cached_tokens_total", 0.0),
        }
    return out


@dataclasses.dataclass(frozen=True)
class MarbleRequest:
    """A single LLM request in a MARBLE workflow.

    Attributes:
        scenario: Scenario type (e.g., "research", "coding")
        agent_id: Identifier of the agent making the request
        workflow_id: Identifier of the parent workflow
        step_id: Unique step identifier within the workflow
        messages: Chat messages for the request
    """
    scenario: str
    agent_id: str
    workflow_id: str
    step_id: str
    messages: List[Dict[str, str]]


@dataclasses.dataclass
class MarbleWorkflow:
    """A complete MARBLE workflow consisting of multiple steps.

    Attributes:
        scenario: Scenario type (e.g., "research", "coding")
        workflow_id: Unique workflow identifier
        steps: List of requests in execution order
    """
    scenario: str
    workflow_id: str
    steps: List[MarbleRequest]


def load_marble_traces(
    marble_dir: Path,
    scenario: str,
    num_traces: int,
    start: int = 0
) -> List[Dict[str, Any]]:
    """Load MARBLE trace data from JSONL files.

    Args:
        marble_dir: Path to MARBLE dataset directory
        scenario: Scenario to load ("research" or "coding")
        num_traces: Maximum number of traces to load
        start: Starting index (for pagination)

    Returns:
        List of trace dictionaries
    """
    path = marble_dir / "multiagentbench" / scenario / f"{scenario}_main.jsonl"
    traces: List[Dict[str, Any]] = []
    with path.open() as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if len(traces) >= num_traces:
                break
            traces.append(json.loads(line))
    return traces


def build_workflows_from_traces(
    traces: Iterable[Dict[str, Any]],
    max_iter_override: Optional[int] = None,
) -> List[MarbleWorkflow]:
    """Convert raw MARBLE traces into workflow objects.

    Each trace is expanded into multiple steps based on the number of agents
    and iterations defined in the trace configuration.

    Args:
        traces: Iterable of raw trace dictionaries
        max_iter_override: If set, use this for all traces instead of trace config

    Returns:
        List of MarbleWorkflow objects ready for execution
    """
    workflows: List[MarbleWorkflow] = []
    for trace in traces:
        scenario = trace.get("scenario") or "unknown"
        workflow_id = str(trace.get("task_id") or trace.get("id") or "")
        scenario_l = str(scenario).lower()
        default_max_iter = 1
        if "research" in scenario_l:
            default_max_iter = 3
        elif "coding" in scenario_l:
            default_max_iter = 5

        if max_iter_override is not None:
            max_iter = max(1, max_iter_override)
        else:
            raw_max_iter = trace.get("environment", {}).get("max_iterations", default_max_iter)
            try:
                max_iter = int(raw_max_iter) if str(raw_max_iter).strip() else default_max_iter
            except Exception:
                max_iter = default_max_iter
            max_iter = max(1, max_iter)

        steps: List[MarbleRequest] = []
        agents = trace.get("agents", [])
        for iteration in range(max_iter):
            for agent in agents:
                agent_id = str(agent.get("agent_id"))
                step_id = f"iter{iteration}_{agent_id}"
                messages = [
                    {"role": "system", "content": agent.get("profile", "")},
                    {"role": "user", "content": trace.get("task", {}).get("content", "")},
                ]
                steps.append(
                    MarbleRequest(
                        scenario=scenario,
                        agent_id=agent_id,
                        workflow_id=workflow_id,
                        step_id=step_id,
                        messages=messages,
                    )
                )
        workflows.append(MarbleWorkflow(scenario=scenario, workflow_id=workflow_id, steps=steps))
    return workflows


class RoutingPolicy:
    """Base class for request routing policies.

    Routing policies determine which worker should handle each request.
    Subclasses implement different strategies (round-robin, sticky, etc.).
    """
    name: str

    def __init__(self, worker_urls: List[str]):
        """Initialize the routing policy.

        Args:
            worker_urls: List of worker base URLs
        """
        self.worker_urls = worker_urls
        self._inflight: Dict[str, int] = {u: 0 for u in worker_urls}
        self._dead: Set[str] = set()

    def mark_dead(self, url: str) -> None:
        """Mark a worker as dead (failed)."""
        self._dead.add(url)

    def is_alive(self, url: str) -> bool:
        """Check if a worker is alive."""
        return url not in self._dead

    def on_start(self, url: str) -> None:
        """Called when a request starts on a worker."""
        self._inflight[url] = self._inflight.get(url, 0) + 1

    def on_done(self, url: str) -> None:
        """Called when a request completes on a worker."""
        self._inflight[url] = max(0, self._inflight.get(url, 0) - 1)

    def route(self, req: MarbleRequest) -> str:
        """Route a request to a worker URL.

        Args:
            req: The request to route

        Returns:
            Worker URL to send the request to
        """
        raise NotImplementedError


class RoundRobinPolicy(RoutingPolicy):
    """Simple round-robin routing policy.

    Distributes requests evenly across all workers in order.
    No locality or cache awareness.
    """
    name = "round_robin"

    def __init__(self, worker_urls: List[str]):
        super().__init__(worker_urls)
        self._counter = 0

    def route(self, req: MarbleRequest) -> str:
        _ = req
        for _i in range(len(self.worker_urls)):
            url = self.worker_urls[self._counter % len(self.worker_urls)]
            self._counter += 1
            if self.is_alive(url):
                return url
        return self.worker_urls[self._counter % len(self.worker_urls)]


class RandomPolicy(RoutingPolicy):
    """Random routing policy.

    Randomly selects a worker for each request.
    No locality or cache awareness.
    """
    name = "random"

    def route(self, req: MarbleRequest) -> str:
        _ = req
        alive = [u for u in self.worker_urls if self.is_alive(u)]
        return random.choice(alive) if alive else random.choice(self.worker_urls)


class WorkflowStickyHashPolicy(RoutingPolicy):
    """Workflow-sticky routing policy.

    Routes all requests from the same workflow to the same worker.
    Good for workflow-level cache locality.
    """
    name = "workflow_sticky_hash"

    def route(self, req: MarbleRequest) -> str:
        idx = _stable_int_hash(req.workflow_id) % len(self.worker_urls)
        url = self.worker_urls[idx]
        if self.is_alive(url):
            return url
        # Fallback: pick any alive worker
        alive = [u for u in self.worker_urls if self.is_alive(u)]
        return alive[0] if alive else url


class AgentStickyHashPolicy(RoutingPolicy):
    """Agent-sticky routing policy.

    Routes requests from the same agent to dedicated GPU replicas.
    Maximizes prefix cache hits for agent system prompts.

    GPU mapping (10 GPUs):
    - Research (5 agents): agent1→[0,5], agent2→[1,6], agent3→[2,7], agent4→[3,8], agent5→[4,9]
    - Coding (3 agents): agent1→[0,1,2], agent2→[3,4,5], agent3→[6,7,8]
    """
    name = "agent_sticky_hash"

    def __init__(self, worker_urls: List[str], scenario: str):
        super().__init__(worker_urls)
        self.scenario = scenario
        self.mapping = self._build_mapping(scenario)

    def _build_mapping(self, scenario: str) -> Dict[str, List[int]]:
        if scenario == "research":
            return {
                "agent1": [0, 5],
                "agent2": [1, 6],
                "agent3": [2, 7],
                "agent4": [3, 8],
                "agent5": [4, 9],
            }
        if scenario == "coding":
            return {
                "agent1": [0, 1, 2],
                "agent2": [3, 4, 5],
                "agent3": [6, 7, 8],
            }
        # Mixed/unknown: use stable hash to preserve agent locality.
        return {}

    def _pick_from_candidates(self, candidates: List[int], req: MarbleRequest) -> str:
        if not candidates:
            idx = _stable_int_hash(f"{req.scenario}:{req.agent_id}") % len(self.worker_urls)
            url = self.worker_urls[idx]
            if self.is_alive(url):
                return url
            alive = [u for u in self.worker_urls if self.is_alive(u)]
            return alive[0] if alive else url
        ridx = _stable_int_hash(req.workflow_id) % len(candidates)
        gpu_id = candidates[ridx]
        url = self.worker_urls[gpu_id]
        if self.is_alive(url):
            return url
        # Fallback among candidate replicas, then any alive.
        for gid in candidates:
            u = self.worker_urls[gid]
            if self.is_alive(u):
                return u
        alive = [u for u in self.worker_urls if self.is_alive(u)]
        return alive[0] if alive else url

    def route(self, req: MarbleRequest) -> str:
        candidates = self.mapping.get(req.agent_id)
        if candidates is None:
            # Fallback for datasets with >5 agents (e.g., some MARBLE research traces).
            primary = _stable_int_hash(f"{req.scenario}:{req.agent_id}") % len(self.worker_urls)
            secondary = (primary + len(self.worker_urls) // 2) % len(self.worker_urls) if len(self.worker_urls) > 1 else primary
            candidates = [primary] if secondary == primary else [primary, secondary]
        return self._pick_from_candidates(candidates, req)


class AgentPrimaryPolicy(AgentStickyHashPolicy):
    """Agent-primary routing policy.

    Always routes to the primary replica (first GPU) for each agent.
    Useful for failure-injection experiments where we want a clear
    "primary" to fail, and deterministic fallback to secondary replicas.
    """
    name = "agent_primary"

    def route(self, req: MarbleRequest) -> str:
        candidates = self.mapping.get(req.agent_id, [])
        if not candidates:
            return super().route(req)

        primary_gpu = candidates[0]
        primary_url = self.worker_urls[primary_gpu]
        if self.is_alive(primary_url):
            return primary_url

        # Fallback among candidate replicas, then any alive.
        for gid in candidates[1:]:
            u = self.worker_urls[gid]
            if self.is_alive(u):
                return u
        alive = [u for u in self.worker_urls if self.is_alive(u)]
        return alive[0] if alive else primary_url


class AgentStickyLoadGuardPolicy(AgentStickyHashPolicy):
    """Agent-sticky routing with load balancing guard.

    Routes to agent's primary GPU unless it's overloaded, then falls back
    to secondary replicas or overflow GPU. Balances cache locality with
    load distribution.
    """
    name = "agent_sticky_load_guard"

    def __init__(
        self,
        worker_urls: List[str],
        scenario: str,
        max_inflight_on_primary: int = 4,
        overflow_gpu: Optional[int] = 9
    ):
        """Initialize the load-guarded agent-sticky policy.

        Args:
            worker_urls: List of worker base URLs
            scenario: Scenario type for GPU mapping
            max_inflight_on_primary: Max concurrent requests on primary before overflow
            overflow_gpu: GPU ID for overflow traffic (None to disable)
        """
        super().__init__(worker_urls, scenario)
        self.max_inflight_on_primary = max_inflight_on_primary
        self.overflow_gpu = overflow_gpu

    def route(self, req: MarbleRequest) -> str:
        candidates = self.mapping.get(req.agent_id, [])
        if not candidates:
            return super().route(req)

        primary_gpu = candidates[0]
        primary_url = self.worker_urls[primary_gpu]
        if self.is_alive(primary_url) and self._inflight.get(primary_url, 0) < self.max_inflight_on_primary:
            return primary_url

        # Prefer other replicas, then overflow, then any alive.
        replica_urls = [self.worker_urls[g] for g in candidates[1:] if self.is_alive(self.worker_urls[g])]
        if replica_urls:
            # Pick least in-flight among replicas.
            return min(replica_urls, key=lambda u: self._inflight.get(u, 0))

        if self.overflow_gpu is not None and 0 <= self.overflow_gpu < len(self.worker_urls):
            ov = self.worker_urls[self.overflow_gpu]
            if self.is_alive(ov):
                return ov

        alive = [u for u in self.worker_urls if self.is_alive(u)]
        return min(alive, key=lambda u: self._inflight.get(u, 0)) if alive else primary_url


class CacheAwareHashPolicy(RoutingPolicy):
    """Cache-aware routing policy.

    Hashes the request's stable prompt prefix (system message) to pick a worker,
    so identical agent profiles tend to land on the same GPU for higher
    prefix-cache hit rates.
    """
    name = "cache_aware"

    def route(self, req: MarbleRequest) -> str:
        sys_msg = req.messages[0]["content"] if req.messages else ""
        idx = _stable_int_hash(sys_msg) % len(self.worker_urls)
        url = self.worker_urls[idx]
        if self.is_alive(url):
            return url
        alive = [u for u in self.worker_urls if self.is_alive(u)]
        return alive[0] if alive else url


def create_routing_policy(
    name: str,
    worker_urls: List[str],
    *,
    scenario: str,
    router_url: Optional[str] = None
) -> RoutingPolicy:
    """Factory function to create routing policies.

    Args:
        name: Policy name (round_robin, random, agent_sticky_hash, etc.)
        worker_urls: List of worker base URLs
        scenario: Scenario type for agent-aware policies
        router_url: Optional router URL (unused, for future extension)

    Returns:
        Configured routing policy instance

    Raises:
        ValueError: If policy name is unknown
    """
    if name == RoundRobinPolicy.name:
        return RoundRobinPolicy(worker_urls)
    if name == RandomPolicy.name:
        return RandomPolicy(worker_urls)
    if name == AgentStickyHashPolicy.name:
        return AgentStickyHashPolicy(worker_urls, scenario=scenario)
    if name == AgentPrimaryPolicy.name:
        return AgentPrimaryPolicy(worker_urls, scenario=scenario)
    if name == WorkflowStickyHashPolicy.name:
        return WorkflowStickyHashPolicy(worker_urls)
    if name == AgentStickyLoadGuardPolicy.name:
        return AgentStickyLoadGuardPolicy(worker_urls, scenario=scenario)
    if name == CacheAwareHashPolicy.name:
        return CacheAwareHashPolicy(worker_urls)
    raise ValueError(f"Unknown routing policy: {name}")


async def _read_sse_json_events(resp: aiohttp.ClientResponse) -> List[Dict[str, Any]]:
    """Read and parse Server-Sent Events (SSE) from an HTTP response.

    Args:
        resp: aiohttp response object with SSE stream

    Returns:
        List of parsed JSON event objects
    """
    events: List[Dict[str, Any]] = []
    buf = b""
    async for chunk in resp.content.iter_any():
        buf += chunk
        while b"\n" in buf:
            line, _, buf = buf.partition(b"\n")
            line = line.strip()
            if not line:
                continue
            if not line.startswith(b"data:"):
                continue
            payload = line[len(b"data:"):].strip()
            if payload == b"[DONE]":
                return events
            try:
                events.append(json.loads(payload.decode("utf-8")))
            except Exception:
                continue
    return events


async def send_chat_completion(
    session: aiohttp.ClientSession,
    base_url: str,
    messages: List[Dict[str, str]],
    *,
    max_completion_tokens: int = 512,
    temperature: float = 0.7,
    extra_key: Optional[str] = None,
    priority: Optional[int] = None,
    timeout_s: int = 600,
) -> Dict[str, Any]:
    """Send a chat completion request to an SGLang worker.

    Args:
        session: aiohttp client session
        base_url: Worker base URL
        messages: Chat messages
        max_completion_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        extra_key: Optional extra key for cache routing
        priority: Optional request priority
        timeout_s: Request timeout in seconds

    Returns:
        Dictionary with success status, latency, TTFT, and usage info
    """
    payload: Dict[str, Any] = {
        "model": "default",
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_completion_tokens": max_completion_tokens,
        "temperature": temperature,
    }
    if extra_key is not None:
        payload["extra_key"] = extra_key
    if priority is not None:
        payload["priority"] = priority

    start = time.perf_counter()
    ttft_s: Optional[float] = None
    try:
        async with session.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as resp:
            resp.raise_for_status()
            # TTFT: first SSE "data:" line.
            first_chunk = await resp.content.readline()
            if first_chunk:
                ttft_s = time.perf_counter() - start

            # Continue reading the stream from remaining bytes (including first line content).
            # Put back the first chunk into a fake buffer by concatenating it with the rest.
            # aiohttp doesn't support un-reading, so we parse including the first chunk manually.
            rest = await resp.content.read()
            all_bytes = first_chunk + rest
            # Parse events from all_bytes.
            events: List[Dict[str, Any]] = []
            buf = all_bytes
            for raw_line in buf.splitlines():
                raw_line = raw_line.strip()
                if not raw_line.startswith(b"data:"):
                    continue
                data = raw_line[len(b"data:") :].strip()
                if data == b"[DONE]":
                    break
                try:
                    events.append(json.loads(data.decode("utf-8")))
                except Exception:
                    continue

            end = time.perf_counter()

        usage = {}
        for ev in reversed(events):
            u = ev.get("usage")
            if isinstance(u, dict):
                usage = u
                break

        return {
            "success": True,
            "latency_s": end - start,
            "ttft_s": ttft_s if ttft_s is not None else (end - start),
            "usage": usage,
        }
    except Exception as e:
        end = time.perf_counter()
        return {
            "success": False,
            "latency_s": end - start,
            "ttft_s": ttft_s,
            "error": repr(e),
        }


@dataclasses.dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution.

    Attributes:
        routing_policy: Name of the routing policy to use
        workflow_concurrency: Maximum concurrent workflows
        max_completion_tokens: Maximum tokens to generate per request
        temperature: Sampling temperature
        request_timeout_s: Request timeout in seconds
        max_retries: Maximum retry attempts on failure
        retry_backoff_s: Base backoff time between retries
        extra_key_mode: Extra key mode for cache routing (none|agent|workflow|agent_workflow)
        router_url: Optional router URL for cache_aware routing
    """
    routing_policy: str
    workflow_concurrency: int
    max_completion_tokens: int = 512
    temperature: float = 0.7
    request_timeout_s: int = 600
    max_retries: int = 0
    retry_backoff_s: float = 0.2
    extra_key_mode: str = "none"
    router_url: Optional[str] = None


def _make_extra_key(req: MarbleRequest, mode: str) -> Optional[str]:
    """Generate extra key for cache routing based on mode."""
    if mode == "none":
        return None
    if mode == "agent":
        return f"agent:{req.scenario}:{req.agent_id}"
    if mode == "workflow":
        return f"wf:{req.scenario}:{req.workflow_id}"
    if mode == "agent_workflow":
        return f"agent_wf:{req.scenario}:{req.agent_id}:{req.workflow_id}"
    raise ValueError(f"Unknown extra_key_mode: {mode}")


async def run_workflow(
    session: aiohttp.ClientSession,
    workflow: MarbleWorkflow,
    policy: RoutingPolicy,
    cfg: BenchmarkConfig,
    *,
    priority_fn: Optional[Callable[[MarbleRequest], int]] = None,
    results_out: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Execute a single workflow's steps sequentially.

    Args:
        session: aiohttp client session
        workflow: Workflow to execute
        policy: Routing policy for request distribution
        cfg: Benchmark configuration
        priority_fn: Optional function to compute request priority
        results_out: Optional list to append results to (for aggregation)

    Returns:
        List of result records for each step
    """
    out: List[Dict[str, Any]] = []
    for step in workflow.steps:
        attempt = 0
        while True:
            url = policy.route(step)
            policy.on_start(url)
            started_at = _now_ts()
            res = await send_chat_completion(
                session,
                url,
                step.messages,
                max_completion_tokens=cfg.max_completion_tokens,
                temperature=cfg.temperature,
                extra_key=_make_extra_key(step, cfg.extra_key_mode),
                priority=priority_fn(step) if priority_fn else None,
                timeout_s=cfg.request_timeout_s,
            )
            finished_at = _now_ts()
            policy.on_done(url)

            record: Dict[str, Any] = {
                "scenario": step.scenario,
                "agent_id": step.agent_id,
                "workflow_id": step.workflow_id,
                "step_id": step.step_id,
                "url": url,
                "attempt": attempt,
                "started_at": started_at,
                "finished_at": finished_at,
                **res,
            }
            out.append(record)
            if results_out is not None:
                results_out.append(record)

            if record.get("success"):
                break
            attempt += 1
            policy.mark_dead(url)
            if attempt > cfg.max_retries:
                break
            await asyncio.sleep(cfg.retry_backoff_s * (attempt + 1))
    return out


async def run_benchmark(
    workflows: List[MarbleWorkflow],
    worker_urls: List[str],
    cfg: BenchmarkConfig,
    *,
    output_dir: Path,
    priority_fn: Optional[Callable[[MarbleRequest], int]] = None,
) -> Dict[str, Any]:
    """Run a complete benchmark with multiple workflows.

    Executes workflows concurrently (up to workflow_concurrency limit),
    collects results, and computes statistics.

    Args:
        workflows: List of workflows to execute
        worker_urls: List of worker base URLs
        cfg: Benchmark configuration
        output_dir: Directory to save results
        priority_fn: Optional function to compute request priority

    Returns:
        Dictionary with benchmark statistics (latency, throughput, etc.)
    """
    _ensure_dir(output_dir)
    scenario_set = {wf.scenario for wf in workflows}
    scenario = next(iter(scenario_set)) if len(scenario_set) == 1 else "mixed"
    policy = create_routing_policy(
        cfg.routing_policy,
        worker_urls,
        scenario=scenario,
        router_url=cfg.router_url,
    )

    all_records: List[Dict[str, Any]] = []
    sem = asyncio.Semaphore(cfg.workflow_concurrency)

    async def _run_one(wf: MarbleWorkflow):
        async with sem:
            return await run_workflow(
                session, wf, policy, cfg, priority_fn=priority_fn, results_out=all_records
            )

    started = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        tasks = [asyncio.create_task(_run_one(wf)) for wf in workflows]
        await asyncio.gather(*tasks)
    total_time_s = time.perf_counter() - started

    # Persist raw records.
    with (output_dir / "records.jsonl").open("w") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    successful = [r for r in all_records if r.get("success")]
    failed = [r for r in all_records if not r.get("success")]

    latencies = [float(r["latency_s"]) for r in successful]
    ttfts = [float(r["ttft_s"]) for r in successful if r.get("ttft_s") is not None]

    stats: Dict[str, Any] = {
        "routing_policy": cfg.routing_policy,
        "workflow_concurrency": cfg.workflow_concurrency,
        "num_workflows": len(workflows),
        "total_requests": len(all_records),
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "total_time_s": total_time_s,
        "throughput_rps": (len(successful) / total_time_s) if total_time_s > 0 else 0.0,
        "latency": _summarize_latencies(latencies),
        "ttft": _summarize_latencies(ttfts),
    }

    (output_dir / "summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats
