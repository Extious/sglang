"""
Deploy SGLang Server (Workers)

Starts N SGLang workers on the current node, binds to 0.0.0.0,
and writes worker URLs (using internal network hostname/IP) to a shared file.

Usage:
    python deploy_server.py --model-path Qwen/Qwen3-4B-Thinking-2507 --num-workers 4
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@dataclasses.dataclass(frozen=True)
class WorkerEndpoint:
    gpu_id: int
    url: str
    port: int


@dataclasses.dataclass
class ClusterConfig:
    model_path: str
    num_workers: int = 10
    worker_base_port: int = 8000
    router_port: int = 30000
    host: str = "127.0.0.1"
    python_bin: str = "python"
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
    def __init__(self, repo_root: Path, cfg: ClusterConfig, log_dir: Path):
        self.repo_root = repo_root
        self.cfg = cfg
        self.log_dir = log_dir
        _ensure_dir(self.log_dir)

        self.worker_procs: Dict[int, subprocess.Popen] = {}
        self._extra_args_by_gpu: Dict[int, List[str]] = {}
        self._extra_env_by_gpu: Dict[int, Dict[str, str]] = {}
        self._errno5_pattern = re.compile(
            r"OSError:\s*\[Errno 5\]\s*Input/output error:\s*['\"]([^'\"]+)['\"]"
        )

    def worker_endpoints(self) -> List[WorkerEndpoint]:
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
        extra_args_by_gpu = extra_args_by_gpu or {}
        extra_env_by_gpu = extra_env_by_gpu or {}
        self._extra_args_by_gpu = {k: list(v) for k, v in extra_args_by_gpu.items()}
        self._extra_env_by_gpu = {k: dict(v) for k, v in extra_env_by_gpu.items()}

        for gpu_id in range(self.cfg.num_workers):
            print(
                f"  Starting worker {gpu_id} (port {self.cfg.worker_base_port + gpu_id})..."
            )
            self._start_one_worker(
                gpu_id,
                extra_args=extra_args_by_gpu.get(gpu_id, []),
                extra_env=extra_env_by_gpu.get(gpu_id, {}),
                truncate_log=True,
            )

        print(
            f"  Waiting for all {self.cfg.num_workers} workers to be ready "
            f"(health check, timeout={self.cfg.wait_ready_timeout_s}s)..."
        )
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

            try:
                proc.wait(timeout=5)
            except Exception:
                pass

            self._mitigate_errno5_from_log(log_path, gpu_id, attempt)
            if attempt < self.cfg.worker_start_retries:
                time.sleep(self.cfg.worker_retry_delay_s)

        if proc is None or proc.poll() is not None:
            raise RuntimeError(
                f"Worker {gpu_id} failed to start. Check {log_path} for details."
            )

        time.sleep(self.cfg.worker_startup_delay_s)

    def _mitigate_errno5_from_log(self, log_path: Path, gpu_id: int, attempt: int) -> None:
        try:
            text = log_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return

        matches = self._errno5_pattern.findall(text)
        if not matches:
            return

        touched = 0
        seen = set()
        for raw in reversed(matches):
            if raw in seen:
                continue
            seen.add(raw)
            if self._touch_errno5_path(Path(raw)):
                touched += 1

        if touched > 0:
            print(
                f"    Worker {gpu_id}: detected Errno 5 and touched {touched} path(s) "
                f"after failed attempt {attempt}."
            )

    @staticmethod
    def _touch_errno5_path(path: Path) -> bool:
        try:
            if path.exists():
                os.utime(path, None)
                return True
        except Exception:
            pass

        try:
            probe_dir = path if path.is_dir() else path.parent
            if probe_dir and probe_dir.exists():
                probe = probe_dir / ".sglang_errno5_probe"
                probe.touch(exist_ok=True)
                os.utime(probe_dir, None)
                return True
        except Exception:
            pass

        return False

    def wait_until_ready(self) -> None:
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

    def stop_workers(self) -> None:
        for _, proc in list(self.worker_procs.items()):
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        for _, proc in list(self.worker_procs.items()):
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.worker_procs.clear()


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _get_internal_host() -> str:
    """Get internal network hostname or IP for worker URLs."""
    hostname = socket.gethostname()
    hostname_short = hostname.split(".")[0]
    try:
        host_ip = socket.gethostbyname(hostname)
        if host_ip.startswith("127."):
            host_ip = socket.gethostbyname(hostname_short)
        return hostname_short if host_ip.startswith("127.") else hostname_short
    except Exception:
        return hostname_short


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy SGLang workers on current node")
    parser.add_argument(
        "--model-path",
        default=os.getenv("MODEL_PATH", "Qwen/Qwen3-4B-Thinking-2507"),
        help="HuggingFace model path",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(os.getenv("NUM_WORKERS", "2")),
        help="Number of workers (GPUs)",
    )
    parser.add_argument(
        "--worker-base-port",
        type=int,
        default=int(os.getenv("WORKER_BASE_PORT", "8000")),
        help="Base port for workers",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for logs and worker_urls.txt (default: experiment1/logs)",
    )
    parser.add_argument(
        "--tool-call-parser",
        default=os.getenv("TOOL_CALL_PARSER", "qwen"),
        help="Tool call parser for OpenAI server (default: qwen)",
    )
    parser.add_argument(
        "--enable-hicache",
        dest="enable_hicache",
        action="store_true",
        help="Enable hierarchical KV cache (HiCache) on each worker.",
    )
    parser.add_argument(
        "--disable-hicache",
        dest="enable_hicache",
        action="store_false",
        help="Disable hierarchical KV cache (HiCache) on each worker.",
    )
    parser.add_argument(
        "--hicache-size",
        type=int,
        default=int(os.getenv("HICACHE_SIZE", "8")),
        help="HiCache size per worker in GB. If >0, overrides hicache ratio.",
    )
    parser.add_argument(
        "--hicache-ratio",
        type=float,
        default=float(os.getenv("HICACHE_RATIO", "2.0")),
        help="HiCache ratio used when hicache-size <= 0.",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Exit immediately after starting workers (for use in scripts)",
    )
    parser.add_argument(
        "--enable-cache-report",
        dest="enable_cache_report",
        action="store_true",
        help="Enable usage.prompt_tokens_details.cached_tokens in OpenAI responses.",
    )
    parser.add_argument(
        "--disable-cache-report",
        dest="enable_cache_report",
        action="store_false",
        help="Disable usage.prompt_tokens_details.cached_tokens in OpenAI responses.",
    )
    parser.set_defaults(
        enable_hicache=os.getenv("ENABLE_HICACHE", "1").strip().lower()
        not in {"0", "false", "no"},
        enable_cache_report=os.getenv(
            "ENABLE_CACHE_REPORT", "1"
        ).strip().lower()
        not in {"0", "false", "no"},
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    script_dir = _script_dir()
    output_dir = args.output_dir or (script_dir / "logs")
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = output_dir
    worker_host = _get_internal_host()

    # Use Python from virtual environment if available
    venv_python = repo_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        python_bin = str(venv_python)
    else:
        venv_python_win = repo_root / ".venv" / "Scripts" / "python.exe"
        if venv_python_win.exists():
            python_bin = str(venv_python_win)
        else:
            python_bin = os.getenv("PYTHON_BIN", sys.executable)
    
    extra_pythonpath = os.getenv("SGLANG_EXTRA_PYTHONPATH", "")

    worker_common_args = [
        "--attention-backend", "triton",
        "--sampling-backend", "pytorch",
        "--enable-metrics",
        "--tool-call-parser", args.tool_call_parser,
    ]
    if args.enable_cache_report:
        worker_common_args.append("--enable-cache-report")
    if args.enable_hicache:
        worker_common_args.append("--enable-hierarchical-cache")
        if args.hicache_size > 0:
            worker_common_args.extend(["--hicache-size", str(args.hicache_size)])
        elif args.hicache_ratio > 0:
            worker_common_args.extend(["--hicache-ratio", str(args.hicache_ratio)])

    cluster_cfg = ClusterConfig(
        model_path=args.model_path,
        num_workers=args.num_workers,
        worker_base_port=args.worker_base_port,
        host="0.0.0.0",
        python_bin=python_bin,
        extra_pythonpath=extra_pythonpath or None,
        include_repo_python=True,
        worker_common_args=worker_common_args,
        wait_ready_timeout_s=1800,
        worker_start_check_s=10,
        worker_startup_delay_s=8,
    )

    print(f"Deploying {args.num_workers} SGLang workers on {worker_host}")
    print(f"Model: {args.model_path}")
    print(f"Worker base port: {args.worker_base_port}")
    print(f"Log directory: {log_dir}")
    print(f"HiCache enabled: {args.enable_hicache}")
    print(f"Cache report enabled: {args.enable_cache_report}")
    if args.enable_hicache:
        if args.hicache_size > 0:
            print(f"HiCache size per worker: {args.hicache_size} GB")
        else:
            print(f"HiCache ratio: {args.hicache_ratio}")

    cluster = SGLangCluster(repo_root=repo_root, cfg=cluster_cfg, log_dir=log_dir)
    
    try:
        cluster.start_workers()
        worker_endpoints = cluster.worker_endpoints()

        worker_urls: List[str] = []
        for ep in worker_endpoints:
            url = f"http://{worker_host}:{ep.port}"
            worker_urls.append(url)
            print(f"  Worker {ep.gpu_id}: {url}")

        worker_urls_file = output_dir / "worker_urls.txt"
        worker_urls_file.write_text("\n".join(worker_urls) + "\n")
        print(f"\nWorker URLs written to: {worker_urls_file}")
        print("Worker URLs:")
        for url in worker_urls:
            print(f"  {url}")

        server_node_file = output_dir / "server_node.txt"
        server_node_file.write_text(f"{worker_host}\n")
        print(f"\nServer node written to: {server_node_file}")

        if args.no_wait:
            print("\nWorkers started. Exiting (workers will continue running).")
            # Don't call cleanup in no-wait mode
            return 0

        print("\nWorkers are running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\nShutting down workers...")
    finally:
        if not args.no_wait:
            cluster.stop_workers()

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
