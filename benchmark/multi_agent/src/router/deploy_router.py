"""
Deploy SGLang Router

Starts SGLang router on current node, reads worker URLs from shared file,
and writes router URL (using internal network hostname/IP) to shared file.

Usage:
    python deploy_router.py --worker-urls-file logs/worker_urls.txt
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests


def _split_no_proxy(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _expand_no_proxy_hosts(hosts: list[str]) -> list[str]:
    """Expand hosts with their short hostname + resolved IPv4 when possible."""
    expanded: list[str] = []
    for h in hosts:
        if not h:
            continue
        expanded.append(h)
        if "." in h:
            expanded.append(h.split(".", 1)[0])
        try:
            ip = socket.gethostbyname(h)
            if ip and not ip.startswith("127."):
                expanded.append(ip)
        except Exception:
            pass
    return expanded


def _setup_no_proxy(extra_hosts: list[str]) -> None:
    """Set NO_PROXY/no_proxy for internal cluster communication.

    Notes:
    - Wildcards like `gpu*` are not reliably supported across HTTP clients.
      Prefer exact hostnames/IPs derived from worker URLs.
    """
    base_hosts = ["localhost", "127.0.0.1", "0.0.0.0", "::1"]
    existing = _split_no_proxy(os.environ.get("NO_PROXY", "")) + _split_no_proxy(
        os.environ.get("no_proxy", "")
    )

    merged = _dedupe_preserve_order(
        _expand_no_proxy_hosts(existing + base_hosts + extra_hosts)
    )
    no_proxy_hosts = ",".join(merged)
    os.environ["NO_PROXY"] = no_proxy_hosts
    os.environ["no_proxy"] = no_proxy_hosts


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _get_internal_host() -> str:
    """Get internal network hostname or IP for router URL."""
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
    parser = argparse.ArgumentParser(description="Deploy SGLang router")
    parser.add_argument(
        "--model-path",
        default=os.getenv("MODEL_PATH", "Qwen/Qwen3-4B-Thinking-2507"),
        help="HuggingFace model path",
    )
    parser.add_argument(
        "--router-port",
        type=int,
        default=int(os.getenv("ROUTER_PORT", "30000")),
        help="Router port",
    )
    parser.add_argument(
        "--router-policy",
        type=str,
        default=os.getenv("ROUTER_POLICY", "round_robin"),
        help="Routing policy (round_robin, agent_sticky_hash, workflow_sticky_hash, cache_aware)",
    )
    parser.add_argument(
        "--worker-urls-file",
        type=Path,
        default=None,
        help="Path to worker_urls.txt (default: experiment1/logs/worker_urls.txt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for logs and router_url.txt (default: experiment1/logs)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    script_dir = _script_dir()
    output_dir = args.output_dir or (script_dir / "logs")
    output_dir.mkdir(parents=True, exist_ok=True)

    worker_urls_file = args.worker_urls_file or (output_dir / "worker_urls.txt")
    if not worker_urls_file.exists():
        print(f"ERROR: Worker URLs file not found: {worker_urls_file}", file=sys.stderr)
        return 2

    worker_urls_text = worker_urls_file.read_text().strip()
    worker_urls = [url.strip() for url in worker_urls_text.split("\n") if url.strip()]
    if not worker_urls:
        print(f"ERROR: No worker URLs found in {worker_urls_file}", file=sys.stderr)
        return 2

    router_host = _get_internal_host()
    worker_hosts: list[str] = []
    for url in worker_urls:
        if "://" not in url:
            url = f"http://{url}"
        try:
            parsed = urlparse(url)
            if parsed.hostname:
                worker_hosts.append(parsed.hostname)
        except Exception:
            continue

    # Ensure internal worker/router traffic bypasses HTTP proxy.
    _setup_no_proxy([router_host, *worker_hosts])
    log_dir = output_dir

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

    repo_python = str(repo_root / "python")
    env = os.environ.copy()
    pp_parts = []
    if extra_pythonpath:
        pp_parts.append(extra_pythonpath)
    pp_parts.append(repo_python)
    if env.get("PYTHONPATH"):
        pp_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pp_parts)
    # Inherit NO_PROXY/no_proxy from this process (after _setup_no_proxy()).
    env["NO_PROXY"] = os.environ.get("NO_PROXY", env.get("NO_PROXY", ""))
    env["no_proxy"] = os.environ.get("no_proxy", env.get("no_proxy", env["NO_PROXY"]))
    env.setdefault("LC_ALL", "C.UTF-8")
    env.setdefault("LANG", "C.UTF-8")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    print(f"Deploying SGLang router on {router_host}")
    print(f"Router port: {args.router_port}")
    print(f"Routing policy: {args.router_policy}")
    print(f"Worker URLs ({len(worker_urls)}):")
    for url in worker_urls:
        print(f"  {url}")
    print(f"Log directory: {log_dir}")

    log_path = log_dir / "router.log"
    log_path.write_text("")

    args_list = [
        python_bin,
        "-m",
        "sglang_router.launch_router",
        "--worker-urls",
        *worker_urls,
        "--policy",
        args.router_policy,
        "--host",
        "0.0.0.0",
        "--port",
        str(args.router_port),
        "--model-path",
        args.model_path,
    ]

    print(f"\nStarting router: {' '.join(args_list)}")
    router_proc = subprocess.Popen(
        args_list,
        env=env,
        stdout=log_path.open("ab"),
        stderr=subprocess.STDOUT,
    )

    router_url = f"http://{router_host}:{args.router_port}"

    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            r = requests.get(f"{router_url}/v1/models", timeout=5)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        print(f"ERROR: Router failed to start on {router_url}", file=sys.stderr)
        router_proc.terminate()
        router_proc.wait(timeout=5)
        return 1

    router_url_file = output_dir / "router_url.txt"
    router_url_file.write_text(f"{router_url}\n")
    print(f"\nRouter URL written to: {router_url_file}")
    print(f"Router URL: {router_url}")

    router_node_file = output_dir / "router_node.txt"
    router_node_file.write_text(f"{router_host}\n")
    print(f"\nRouter node written to: {router_node_file}")
    print(f"Router PID: {router_proc.pid}")

    print("\nRouter is running. Press Ctrl+C to stop.")
    try:
        router_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down router...")
        router_proc.terminate()
        try:
            router_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            router_proc.kill()

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
