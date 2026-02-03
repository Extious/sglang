"""
Deploy SGLang Server (Workers)

Starts N SGLang workers on the current node, binds to 0.0.0.0,
and writes worker URLs (using internal network hostname/IP) to a shared file.

Usage:
    python deploy_server.py --model-path Qwen/Qwen3-4B-Thinking-2507 --num-workers 4
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from pathlib import Path
from typing import List

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from marble_exp import ClusterConfig, SGLangCluster


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
        "--no-wait",
        action="store_true",
        help="Exit immediately after starting workers (for use in scripts)",
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

    cluster_cfg = ClusterConfig(
        model_path=args.model_path,
        num_workers=args.num_workers,
        worker_base_port=args.worker_base_port,
        host="0.0.0.0",
        python_bin=python_bin,
        extra_pythonpath=extra_pythonpath or None,
        include_repo_python=True,
        worker_common_args=[
            "--attention-backend", "triton",
            "--sampling-backend", "pytorch",
        ],
        wait_ready_timeout_s=1800,
        worker_start_check_s=10,
        worker_startup_delay_s=8,
    )

    print(f"Deploying {args.num_workers} SGLang workers on {worker_host}")
    print(f"Model: {args.model_path}")
    print(f"Worker base port: {args.worker_base_port}")
    print(f"Log directory: {log_dir}")

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
