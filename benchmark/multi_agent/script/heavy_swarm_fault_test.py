#!/usr/bin/env python3
"""Fault-injection wrapper around heavy_swarm.py.

Runs all tasks in a single ProcessPoolExecutor and injects worker failures
based on real-time task completion count.  After every N completed tasks one
worker is flushed (requests aborted, KV cache cleared, removed from router)
and re-added after a configurable delay.

Request-level retry is handled by the Python router: when a worker is
removed/aborted, the router automatically re-routes in-flight requests
(including already-generated tokens) to a healthy worker.  With peer
replication the healthy worker already holds replicated KV cache pages and
can skip prefix recomputation; without peer replication it must recompute
from scratch — this is the key difference the A/B experiment measures.

Configuration (environment variables):
  # --- inherited from heavy_swarm.py ---
  HEAVY_SWARM_TASK_LIMIT       Total tasks (default 200)
  HEAVY_SWARM_TASK_PROCESSES   Parallel workers (default 18)
  LLM_BASE_URL                 Router URL

  # --- fault injection specific ---
  FAULT_WORKER_URLS_FILE       File with worker URLs, one per line
  FAULT_INJECT_EVERY_N         Inject fault every N completed tasks (default 30)
  FAULT_INJECT_DELAY           Seconds to wait after threshold before injecting (default 10)
  FAULT_RECOVER_AFTER          Seconds before auto-recovery (default 30)
  FAULT_SLURM_JOB_ID           Target SLURM job ID (optional, auto-detected)
"""

import json
import os
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests as _requests

_SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Import heavy_swarm core (adds ~/swarms to sys.path)
# ---------------------------------------------------------------------------
_SWARM_DIR = Path(
    os.environ.get(
        "HEAVY_SWARM_DIR",
        os.environ.get("HEAVY_SWARM_SCRIPT", str(Path.home() / "swarms" / "heavy_swarm.py")),
    )
)
if _SWARM_DIR.is_file():
    _SWARM_DIR = _SWARM_DIR.parent
sys.path.insert(0, str(_SWARM_DIR))

import heavy_swarm  # noqa: E402
from heavy_swarm import (  # noqa: E402
    CFG,
    ENABLE_TIMING_REPORTS,
    TASK_LIMIT,
    TASK_PROCESSES,
    TASK_START_OFFSET,
    _run_one_task as _orig_run_one_task,
    _save_timing_reports,
    tasks,
)

CFG["agent_max_tokens"] = {
    "question": 1024,
    "research": 1024,
    "analysis": 1024,
    "alternatives": 2048,
    "verification": 1024,
    "synthesis": 2048,
}


def _run_one_task(task_item):
    """Wrap _run_one_task to inject ``user`` field into every LLM request.

    The ``user`` field is set to ``task_{index}__{agent}`` so the router
    can record which task/agent was retried in retry_metrics.json.
    """
    task_index = task_item[0]

    from swarms.utils import litellm_wrapper

    _pre_original = litellm_wrapper.completion

    def _inject_user(*args, **kwargs):
        agent = heavy_swarm._extract_agent_from_kwargs(kwargs)
        kwargs.setdefault("user", f"task_{task_index}__{agent}")
        return _pre_original(*args, **kwargs)

    litellm_wrapper.completion = _inject_user
    try:
        return _orig_run_one_task(task_item)
    finally:
        litellm_wrapper.completion = _pre_original

# Re-export so ProcessPoolExecutor workers can find _run_one_task via this
# module's namespace (needed for spawn start method; fork inherits anyway).
__all__ = ["main"]


# ============================================================================
# FaultInjector
# ============================================================================

class FaultInjector:
    """Inject and recover worker faults based on task completion count.

    Fault monitoring runs on a dedicated background thread, fully decoupled
    from the task execution loop.  The main loop only calls
    ``notify_completed()`` to bump an atomic counter; a separate monitor
    thread watches the counter and triggers faults when thresholds are
    crossed.

    fault_mode:
      - "suspend": SIGSTOP / SIGCONT (original behavior)
      - "kill": SIGKILL + router deregister + worker restart + router re-register
      - "flush": abort requests + flush KV cache + re-add after delay
    """

    def __init__(
        self,
        worker_urls: List[str],
        inject_every_n: int = 30,
        inject_delay: float = 10,
        recover_after: int = 30,
        slurm_job_id: Optional[str] = None,
        fault_mode: str = "suspend",
        router_url: str = "",
        worker_restart_env: Optional[Dict[str, str]] = None,
        max_faults: Optional[int] = None,
    ):
        self.workers: List[Tuple[str, str]] = []
        self.worker_urls_raw: List[str] = []
        for url in worker_urls:
            full = url if "://" in url else f"http://{url}"
            parsed = urlparse(full)
            node = parsed.hostname or ""
            port = str(parsed.port) if parsed.port else ""
            if node and port:
                self.workers.append((node, port))
                self.worker_urls_raw.append(full)

        self.inject_every_n = inject_every_n
        self.inject_delay = inject_delay
        self.recover_after = recover_after
        self.slurm_job_id = slurm_job_id
        self.fault_mode = fault_mode
        self.router_url = router_url.rstrip("/")
        self.worker_restart_env = worker_restart_env or {}

        self.fault_index = 0
        self.completed_count = 0
        self._lock = threading.Lock()
        self._fault_threads: List[threading.Thread] = []
        self._recovery_threads: List[threading.Thread] = []
        self._suspended: List[Tuple[str, str]] = []  # (node, pid)
        self._restart_procs: List[subprocess.Popen] = []

        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

    @property
    def fault_count(self) -> int:
        return len(self.workers)

    # ------------------------------------------------------------------
    # Public API — fully async: main loop only touches notify_completed()
    # ------------------------------------------------------------------

    def notify_completed(self) -> None:
        """Bump the completed-task counter.  Thread-safe, non-blocking."""
        with self._lock:
            self.completed_count += 1

    def start_monitor(self) -> None:
        """Launch background thread that watches completed_count and
        triggers fault injection independently of the task loop."""
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True,
        )
        self._monitor_thread.start()

    def stop_monitor(self) -> None:
        """Signal the monitor thread to exit."""
        self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=5)

    def _monitor_loop(self) -> None:
        """Poll completed_count and fire faults when thresholds are crossed."""
        while not self._stop_event.is_set():
            should_inject = False
            idx = -1
            trigger_count = 0

            with self._lock:
                if self.fault_index < len(self.workers):
                    threshold = (self.fault_index + 1) * self.inject_every_n
                    if self.completed_count >= threshold:
                        idx = self.fault_index
                        self.fault_index += 1
                        should_inject = True
                        trigger_count = self.completed_count

            if should_inject:
                node, port = self.workers[idx]
                t = threading.Thread(
                    target=self._delayed_inject,
                    args=(idx, node, port, trigger_count),
                    daemon=True,
                )
                t.start()
                with self._lock:
                    self._fault_threads.append(t)

            self._stop_event.wait(timeout=0.5)

    # ------------------------------------------------------------------

    def _delayed_inject(
        self, idx: int, node: str, port: str, trigger_count: int,
    ) -> None:
        """Wait inject_delay seconds, then inject the fault."""
        _log(
            f"Fault {idx + 1}/{self.fault_count} scheduled: "
            f"will flush {node}:{port} in {self.inject_delay}s "
            f"(triggered at {trigger_count} completed tasks)"
        )
        time.sleep(self.inject_delay)
        self._inject_fault(idx, node, port)

    def wait_all_recovered(self) -> None:
        self.stop_monitor()
        for t in self._fault_threads:
            t.join(timeout=self.inject_delay + 30)
        for t in self._recovery_threads:
            t.join(timeout=self.recover_after + 30)

    def cleanup(self) -> None:
        with self._lock:
            still_suspended = list(self._suspended)
        for node, pid in still_suspended:
            _log(f"Cleanup: resuming worker PID {pid} on {node}")
            self._exec_on_node(node, f"kill -CONT {pid} 2>/dev/null")
        for proc in self._restart_procs:
            if proc.poll() is None:
                proc.terminate()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _exec_on_node(self, node: str, cmd: str) -> str:
        srun_args = ["srun", "-w", node, "--nodes=1", "--ntasks=1", "--overlap"]
        if self.slurm_job_id:
            srun_args.insert(1, f"--jobid={self.slurm_job_id}")
        srun_args += ["bash", "-c", cmd]

        env = os.environ.copy()
        env["NO_PROXY"] = "*"
        env["no_proxy"] = "*"
        try:
            proc = subprocess.run(
                srun_args, capture_output=True, text=True, timeout=30, env=env,
            )
            return proc.stdout.strip()
        except Exception as exc:
            _log(f"srun failed on {node}: {exc}")
            return ""

    def _find_worker_pid(self, node: str, port: str) -> Optional[str]:
        cmd = (
            f"ss -ltnp 'sport = :{port}' 2>/dev/null "
            f"| awk -F'pid=' '/pid=/{{split($2,a,\",\"); print a[1]}}' | head -1"
        )
        pid = self._exec_on_node(node, cmd)
        return pid if pid and pid != "0" else None

    def _inject_fault(self, idx: int, node: str, port: str) -> None:
        if self.fault_mode == "kill":
            self._inject_fault_kill(idx, node, port)
        elif self.fault_mode == "flush":
            self._inject_fault_flush(idx, node, port)
        else:
            self._inject_fault_suspend(idx, node, port)

    # ---- Suspend mode (original) ----

    def _inject_fault_suspend(self, idx: int, node: str, port: str) -> None:
        _log(
            f">>> Fault {idx + 1}/{self.fault_count}: "
            f"suspending {node}:{port} (after {self.completed_count} tasks)"
        )

        pid = self._find_worker_pid(node, port)
        if not pid:
            _log(f"ERROR: PID not found for {node}:{port}")
            return

        result = self._exec_on_node(
            node, f"kill -STOP {pid} 2>/dev/null && echo ok || echo failed"
        )
        if result != "ok":
            _log(f"ERROR: Failed to suspend worker PID {pid} on {node}")
            return

        _log(f">>> FAULT INJECTED: Worker suspended (PID {pid} on {node}:{port})")

        with self._lock:
            self._suspended.append((node, pid))

        t = threading.Thread(
            target=self._recover_after_delay,
            args=(idx, node, port, pid),
            daemon=True,
        )
        t.start()
        self._recovery_threads.append(t)

    def _recover_after_delay(
        self, idx: int, node: str, port: str, pid: str,
    ) -> None:
        time.sleep(self.recover_after)

        result = self._exec_on_node(
            node, f"kill -CONT {pid} 2>/dev/null && echo ok || echo failed"
        )

        with self._lock:
            self._suspended = [
                (n, p) for n, p in self._suspended if not (n == node and p == pid)
            ]

        if result == "ok":
            _log(f">>> FAULT RECOVERED: Worker resumed (PID {pid} on {node}:{port})")
        else:
            _log(f">>> ERROR: Failed to resume worker PID {pid} on {node}:{port}")

    # ---- Kill mode ----

    def _inject_fault_kill(self, idx: int, node: str, port: str) -> None:
        worker_url = (
            self.worker_urls_raw[idx]
            if idx < len(self.worker_urls_raw)
            else f"http://{node}:{port}"
        )
        _log(
            f">>> Fault {idx + 1}/{self.fault_count}: "
            f"KILLING {node}:{port} (after {self.completed_count} tasks)"
        )

        pid = self._find_worker_pid(node, port)
        if not pid:
            _log(f"ERROR: PID not found for {node}:{port}")
            return

        result = self._exec_on_node(
            node, f"kill -9 {pid} 2>/dev/null && echo ok || echo failed"
        )
        if result != "ok":
            _log(f"ERROR: Failed to kill worker PID {pid} on {node}")
            return

        _log(f">>> FAULT INJECTED: Worker KILLED (PID {pid} on {node}:{port})")

        if self.router_url:
            self._router_remove_worker(worker_url)

        t = threading.Thread(
            target=self._restart_worker,
            args=(idx, node, port, worker_url),
            daemon=True,
        )
        t.start()
        self._recovery_threads.append(t)

    # ---- Flush mode (cache wipe without process restart) ----

    def _inject_fault_flush(self, idx: int, node: str, port: str) -> None:
        worker_url = (
            self.worker_urls_raw[idx]
            if idx < len(self.worker_urls_raw)
            else f"http://{node}:{port}"
        )
        _log(
            f">>> Fault {idx + 1}/{self.fault_count}: "
            f"FLUSHING {node}:{port} (after {self.completed_count} tasks)"
        )

        if self.router_url:
            self._router_remove_worker(worker_url)

        try:
            resp = _requests.post(
                f"{worker_url}/abort_request",
                json={"abort_all": True},
                timeout=10,
            )
            _log(f">>> Abort all requests on {worker_url}: {resp.status_code}")
        except Exception as exc:
            _log(f">>> ERROR: abort_request failed: {exc}")

        time.sleep(2)

        try:
            resp = _requests.post(f"{worker_url}/flush_cache", timeout=30)
            _log(f">>> Flush cache on {worker_url}: {resp.status_code}")
        except Exception as exc:
            _log(f">>> ERROR: flush_cache failed: {exc}")

        _log(f">>> FAULT INJECTED: Cache flushed on {node}:{port}")

        t = threading.Thread(
            target=self._readd_worker_after_delay,
            args=(idx, node, port, worker_url),
            daemon=True,
        )
        t.start()
        self._recovery_threads.append(t)

    def _readd_worker_after_delay(
        self, idx: int, node: str, port: str, worker_url: str,
    ) -> None:
        time.sleep(self.recover_after)

        health_url = f"{worker_url}/health"
        try:
            r = _requests.get(health_url, timeout=5)
            if r.status_code == 200:
                if self.router_url:
                    self._router_add_worker(worker_url)
                _log(
                    f">>> FAULT RECOVERED: Worker {node}:{port} re-added to router"
                )
                return
        except Exception:
            pass

        _log(f">>> ERROR: Worker {worker_url} not healthy after flush recovery")

    def _router_remove_worker(self, worker_url: str) -> None:
        try:
            resp = _requests.post(
                f"{self.router_url}/remove_worker",
                json={"url": worker_url},
                timeout=5,
            )
            _log(f">>> Router /remove_worker {worker_url}: {resp.status_code}")
        except Exception as exc:
            _log(f">>> ERROR: Router /remove_worker failed: {exc}")

    def _router_add_worker(self, worker_url: str) -> None:
        try:
            resp = _requests.post(
                f"{self.router_url}/add_worker",
                json={"url": worker_url},
                timeout=5,
            )
            _log(f">>> Router /add_worker {worker_url}: {resp.status_code}")
        except Exception as exc:
            _log(f">>> ERROR: Router /add_worker failed: {exc}")

    def _restart_worker(
        self, idx: int, node: str, port: str, worker_url: str,
    ) -> None:
        _log(f">>> Restarting worker {node}:{port}...")

        restart_cmd = self.worker_restart_env.get("WORKER_RESTART_CMD", "")
        if not restart_cmd:
            _log("ERROR: WORKER_RESTART_CMD not set, cannot restart worker")
            return

        cmd = restart_cmd.replace("{PORT}", port).replace("{GPU_ID}", str(idx))

        srun_args = ["srun", "-w", node, "--nodes=1", "--ntasks=1", "--overlap"]
        if self.slurm_job_id:
            srun_args.insert(1, f"--jobid={self.slurm_job_id}")
        srun_args += ["bash", "-c", cmd]

        env = os.environ.copy()
        env["NO_PROXY"] = "*"
        env["no_proxy"] = "*"

        try:
            proc = subprocess.Popen(
                srun_args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            with self._lock:
                self._restart_procs.append(proc)
        except Exception as exc:
            _log(f"ERROR: Failed to start worker restart: {exc}")
            return

        health_url = f"{worker_url}/health"
        deadline = time.time() + 600
        _log(f">>> Waiting for restarted worker {worker_url} to become healthy...")
        while time.time() < deadline:
            try:
                r = _requests.get(health_url, timeout=3)
                if r.status_code == 200:
                    _log(f">>> Worker {worker_url} is healthy again!")
                    if self.router_url:
                        self._router_add_worker(worker_url)
                    return
            except Exception:
                pass
            time.sleep(2)

        _log(f">>> ERROR: Worker {worker_url} did not become healthy within 600s")


# ============================================================================
# Helpers
# ============================================================================

def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[fault-test] [{ts}] {msg}"
    print(line, file=sys.stderr, flush=True)


def _load_worker_urls(filepath: str) -> List[str]:
    path = Path(filepath).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Worker URLs file not found: {path}")
    urls: List[str] = []
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return urls


def _auto_detect_slurm_job_id(node: str) -> Optional[str]:
    try:
        import getpass
        user = getpass.getuser()
        proc = subprocess.run(
            ["squeue", "-u", user, "-h", "-w", node, "-o", "%.10i"],
            capture_output=True, text=True, timeout=10,
        )
        job_id = proc.stdout.strip().split("\n")[0].strip()
        return job_id if job_id else None
    except Exception:
        return None


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    # --- Fault injection config ---
    default_urls_file = str(_SCRIPT_DIR / ".." / "logs" / "worker_urls.txt")
    worker_urls_file = os.environ.get("FAULT_WORKER_URLS_FILE", default_urls_file)
    inject_every_n = int(os.environ.get("FAULT_INJECT_EVERY_N", "30"))
    inject_delay = float(os.environ.get("FAULT_INJECT_DELAY", "10"))
    recover_after = int(os.environ.get("FAULT_RECOVER_AFTER", "30"))
    slurm_job_id = os.environ.get("FAULT_SLURM_JOB_ID", "").strip() or None
    fault_mode = os.environ.get("FAULT_MODE", "suspend")
    router_url = os.environ.get("FAULT_ROUTER_URL", "").strip()
    worker_restart_cmd = os.environ.get("WORKER_RESTART_CMD", "").strip()

    worker_urls = _load_worker_urls(worker_urls_file)
    if not worker_urls:
        _log(f"No worker URLs found in {worker_urls_file}")
        return

    # Auto-detect SLURM job ID if needed
    if not slurm_job_id and not os.environ.get("SLURM_JOB_ID"):
        first_node = urlparse(
            worker_urls[0] if "://" in worker_urls[0] else f"http://{worker_urls[0]}"
        ).hostname
        if first_node:
            slurm_job_id = _auto_detect_slurm_job_id(first_node)
            if slurm_job_id:
                _log(f"Auto-detected SLURM job ID: {slurm_job_id}")

    # --- Task list (reuse heavy_swarm's loaded config) ---
    all_candidate_tasks = tasks[: TASK_START_OFFSET + TASK_LIMIT]
    selected_tasks = all_candidate_tasks[TASK_START_OFFSET:]
    if not selected_tasks:
        _log("No tasks selected.")
        return

    indexed_tasks = [
        (TASK_START_OFFSET + i, task) for i, task in enumerate(selected_tasks)
    ]
    num_workers = min(TASK_PROCESSES, len(indexed_tasks))

    # --- Fault injector ---
    worker_restart_env = {}
    if worker_restart_cmd:
        worker_restart_env["WORKER_RESTART_CMD"] = worker_restart_cmd

    injector = FaultInjector(
        worker_urls=worker_urls,
        inject_every_n=inject_every_n,
        inject_delay=inject_delay,
        recover_after=recover_after,
        slurm_job_id=slurm_job_id,
        fault_mode=fault_mode,
        router_url=router_url,
        worker_restart_env=worker_restart_env,
    )

    stagger_delay = float(os.environ.get("TASK_STAGGER_DELAY", "0"))

    _log(f"Tasks: {len(indexed_tasks)}, Processes: {num_workers}")
    _log(
        f"Fault injection: mode={fault_mode}, every {inject_every_n} tasks, "
        f"delay {inject_delay}s before inject, "
        f"recover after {recover_after}s, {injector.fault_count} worker(s)"
    )
    if stagger_delay > 0:
        _log(f"Staggered startup: {stagger_delay}s between each of the first {num_workers} processes")
    for i, (node, port) in enumerate(injector.workers):
        _log(f"  Worker {i}: {node}:{port}")

    # Suppress agent/swarm verbose output — redirect stdout of child
    # processes to a log file so only progress (written to stderr) shows
    # on the terminal.
    suppress_agent_output = os.environ.get("SUPPRESS_AGENT_OUTPUT", "0") == "1"
    _saved_stdout = None
    _devnull = None
    if suppress_agent_output:
        _saved_stdout = sys.stdout
        _devnull = open(os.devnull, "w")
        sys.stdout = _devnull

    # --- Run all tasks with fault injection ---
    results: List[Dict[str, Any]] = []
    total_tasks = len(indexed_tasks)
    completed_count = 0
    failed_count = 0
    start_time = time.time()

    injector.start_monitor()

    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_item = {}
            for i, item in enumerate(indexed_tasks):
                if stagger_delay > 0 and 0 < i < num_workers:
                    _log(f"[STAGGER] Waiting {stagger_delay}s before submitting task {item[0]} (process {i+1}/{num_workers})")
                    time.sleep(stagger_delay)
                future = executor.submit(_run_one_task, item)
                future_to_item[future] = item

            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1
                    injector.notify_completed()
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed_count
                    remaining = avg_time * (total_tasks - completed_count)
                    _log(
                        f"[PROGRESS] Task {item[0]} completed "
                        f"({completed_count}/{total_tasks}, "
                        f"{100 * completed_count / total_tasks:.0f}%) "
                        f"elapsed={elapsed:.0f}s eta={remaining:.0f}s"
                    )
                except Exception as exc:
                    failed_count += 1
                    _log(f"Task {item[0]} raised an exception: {exc}")

        injector.wait_all_recovered()
    except KeyboardInterrupt:
        _log("Interrupted — cleaning up suspended workers")
    finally:
        injector.cleanup()
        if _saved_stdout is not None:
            sys.stdout = _saved_stdout
        if _devnull is not None:
            _devnull.close()

    total_elapsed = time.time() - start_time
    _log(
        f"[SUMMARY] {completed_count}/{total_tasks} tasks completed, "
        f"{failed_count} failed, total {total_elapsed:.0f}s"
    )

    results.sort(key=lambda item: item["task_index"])

    # --- Save results ---
    output_path = CFG.get("results_path", "")
    if output_path:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
        _log(f"Results written to {output_path}")

    if ENABLE_TIMING_REPORTS:
        _save_timing_reports(results)

    _log(f"Completed {len(results)} task(s).")


if __name__ == "__main__":
    main()
