#!/usr/bin/env python3
"""
Standalone SGLang server deployer for multi-node clusters.

Reads configuration from a JSON file (or CLI args) and:
  1. Configures environment and cluster topology.
  2. Starts sglang.launch_server on every node via srun.
  3. Polls each node's /health endpoint.
  4. Emits a server_ready control event on stdout.
  5. Blocks until interrupted.

Usage:
  python deploy_server.py config/baseline/server.json
  python deploy_server.py --model-path Qwen/Qwen3-8B --dp-size 2 --pp-size 2 --nnodes 2

Output:
  stdout control events    In-memory orchestration data consumed by main.py
  process logs             Written through MULTI_AGENT_*_LOG paths
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import queue
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from process_logs import decode_control_event, encode_control_event, get_process_log_path

# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

DEFAULT_PLATFORMS = {
    "gpuhome": {
        "model_path": "Qwen/Qwen3-8B",
        "dp_size": 2,
        "pp_size": 1,
        "tp_size": 1,
        "nnodes": 1,
        "server_port_base": 28000,
        "dist_init_port_offset": 100,
        "tool_call_parser": "qwen",
        "reasoning_parser": "",
        "context_length": 0,
        "json_model_override_args": "{}",
        "retry_queue_same_priority_as_waiting": False,
        "chat_template": "",
        "enable_cache_report": True,
        "enable_hicache": True,
        "hicache_size_gb": 16,
        "hicache_ratio": 2.0,
        "insert_step": 0,
        "kv_backup": "none",
        "hicache_storage_backend": "",
        "hicache_storage_prefetch_policy": "",
        "remote_backup_port_base": 30000,
        "remote_backup_buffer_size_gb": 32.0,
        "quantization": "",
        "wait_timeout": 1800,
        "slurm_job_name": "sglang_server",
        "log_dir": "",
        "venv_dir": ".venv",
        "model_local_root": "",
        "venv_local_root": "",
        "dist_free_port_daai": False,
    },
    "daai": {
        "model_path": "Qwen/Qwen3-32B",
        "dp_size": 2,
        "pp_size": 2,
        "tp_size": 1,
        "nnodes": 2,
        "server_port_base": 34000,
        "dist_init_port_offset": 100,
        "tool_call_parser": "qwen",
        "reasoning_parser": "",
        "context_length": 0,
        "json_model_override_args": "{}",
        "retry_queue_same_priority_as_waiting": False,
        "chat_template": "",
        "enable_cache_report": True,
        "enable_hicache": True,
        "hicache_size_gb": 40,
        "hicache_ratio": 2.0,
        "insert_step": 0,
        "kv_backup": "none",
        "hicache_storage_backend": "",
        "hicache_storage_prefetch_policy": "",
        "remote_backup_port_base": 35000,
        "remote_backup_buffer_size_gb": 32.0,
        "quantization": "",
        "wait_timeout": 1800,
        "slurm_job_name": "sglang_server",
        "log_dir": "",
        "venv_dir": ".venv",
        "model_local_root": "/tmp/sglang-model-cache",
        "venv_local_root": "/tmp/sglang-runtime-cache",
        "dist_free_port_daai": True,
    },
}


def load_json_config(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))

    # Resolve platform defaults
    platform_name = raw.get("platform", "gpuhome")
    defaults = dict(DEFAULT_PLATFORMS.get(platform_name, DEFAULT_PLATFORMS["gpuhome"]))

    # Server section
    srv = raw.get("server", {})
    for k, v in srv.items():
        defaults[k] = v

    # Top-level overrides
    defaults["kv_backup"] = raw.get("kv_backup", defaults["kv_backup"])

    # hicache_storage section
    hs = raw.get("hicache_storage", {})
    for k, v in hs.items():
        defaults[f"hicache_storage_{k}"] = v

    defaults["hicache_storage_backend"] = hs.get("backend", "")
    defaults["hicache_storage_prefetch_policy"] = hs.get("prefetch_policy", "")
    defaults["remote_backup_port_base"] = hs.get("remote_backup_port_base", defaults["remote_backup_port_base"])
    defaults["remote_backup_buffer_size_gb"] = hs.get("remote_backup_buffer_size_gb", defaults["remote_backup_buffer_size_gb"])

    defaults["name"] = raw.get("name", "")

    return defaults


def merge_cli_overrides(cfg: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    result = dict(cfg)
    for k, v in args.items():
        if v is not None:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def log_err(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] ERROR: {msg}", flush=True, file=sys.stderr)


# ---------------------------------------------------------------------------
# SLURM helpers
# ---------------------------------------------------------------------------

def get_slurm_env() -> dict[str, str]:
    return {
        "SLURM_JOB_NUM_NODES": os.environ.get("SLURM_JOB_NUM_NODES", "1"),
        "SLURM_JOB_NODELIST": os.environ.get("SLURM_JOB_NODELIST", ""),
        "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID", ""),
        "SLURM_SUBMIT_DIR": os.environ.get("SLURM_SUBMIT_DIR", os.getcwd()),
    }


def get_node_list() -> list[str]:
    slurm_env = get_slurm_env()
    nodelist = slurm_env["SLURM_JOB_NODELIST"]
    if not nodelist:
        return ["localhost"]
    result = subprocess.run(
        ["scontrol", "show", "hostnames", nodelist],
        capture_output=True, text=True, check=True,
    )
    return [n.strip() for n in result.stdout.strip().split("\n") if n.strip()]


def get_total_nodes() -> int:
    return int(get_slurm_env().get("SLURM_JOB_NUM_NODES", "1"))


def find_free_port(node: str, start: int, end: int) -> int:
    """Find a free TCP port on the given node by trying to bind each one."""
    for port in range(start, end + 1):
        result = subprocess.run(
            ["srun", "-w", node, "--nodes=1", "--ntasks=1", "--kill-on-bad-exit=0",
             "python3", "-c",
             f"import socket; s=socket.socket(); s.setsockopt(1, 15, 1); "
             f"s.bind(('0.0.0.0', {port})); s.close(); print({port})"],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and result.stdout.strip().isdigit():
            return int(result.stdout.strip())
    # Fallback: let OS pick
    result = subprocess.run(
        ["srun", "-w", node, "--nodes=1", "--ntasks=1", "--kill-on-bad-exit=0",
         "python3", "-c",
         "import socket; s=socket.socket(); s.bind(('0.0.0.0',0)); print(s.getsockname()[1]); s.close()"],
        capture_output=True, text=True,
    )
    return int(result.stdout.strip())


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(argv: list[str]) -> dict[str, Any]:
    """Load config from JSON file and merge CLI overrides."""
    import argparse

    parser = argparse.ArgumentParser(description="SGLang server deployer")
    parser.add_argument("config", nargs="?", help="Path to server.json config file")
    parser.add_argument("--platform", default=None)
    parser.add_argument("--model-path", dest="model_path", default=None)
    parser.add_argument("--dp-size", dest="dp_size", type=int, default=None)
    parser.add_argument("--pp-size", dest="pp_size", type=int, default=None)
    parser.add_argument("--tp-size", dest="tp_size", type=int, default=None)
    parser.add_argument("--nnodes", dest="nnodes", type=int, default=None)
    parser.add_argument("--hicache-size-gb", dest="hicache_size_gb", type=int, default=None)
    parser.add_argument("--hicache-ratio", dest="hicache_ratio", type=float, default=None)
    parser.add_argument("--insert-step", dest="insert_step", type=int, default=None)
    parser.add_argument("--tool-call-parser", dest="tool_call_parser", default=None)
    parser.add_argument("--reasoning-parser", dest="reasoning_parser", default=None,
                        help="SGLang --reasoning-parser (e.g. qwen3 for Qwen3.5)")
    parser.add_argument("--context-length", dest="context_length", type=int, default=None,
                        help="SGLang --context-length override (tokens). 0/empty = model default")
    parser.add_argument(
        "--json-model-override-args",
        dest="json_model_override_args",
        default=None,
        help="Raw JSON string forwarded to SGLang --json-model-override-args.",
    )
    parser.add_argument(
        "--retry-queue-same-priority-as-waiting",
        dest="retry_queue_same_priority_as_waiting",
        action="store_true",
        default=None,
        help="Use same scheduling priority for retry_queue and waiting_queue.",
    )
    parser.add_argument("--chat-template", dest="chat_template", default=None,
                        help="Path to a Jinja chat template (overrides the model's built-in). "
                             "Relative paths are resolved against the benchmark src root.")
    parser.add_argument("--enable-cache-report", dest="enable_cache_report",
                        action="store_true", default=None)
    parser.add_argument("--disable-cache-report", dest="disable_cache_report",
                        action="store_true", default=None)
    parser.add_argument("--enable-hicache", dest="enable_hicache",
                        action="store_true", default=None)
    parser.add_argument("--disable-hicache", dest="disable_hicache",
                        action="store_true", default=None)
    parser.add_argument("--kv-backup", dest="kv_backup", default=None)
    parser.add_argument("--hicache-storage-backend", dest="hicache_storage_backend", default=None)
    parser.add_argument("--hicache-storage-prefetch-policy",
                        dest="hicache_storage_prefetch_policy", default=None)
    parser.add_argument("--remote-backup-port-base", dest="remote_backup_port_base", type=int, default=None)
    parser.add_argument("--remote-backup-buffer-size-gb", dest="remote_backup_buffer_size_gb",
                        type=float, default=None)
    parser.add_argument("--quantization", dest="quantization", default=None)
    parser.add_argument(
        "--load-balance-method",
        dest="load_balance_method",
        default=None,
        help="SGLang DP load balance: round_robin, total_requests, total_tokens, ...",
    )
    parser.add_argument("--server-port-base", dest="server_port_base", type=int, default=None)
    parser.add_argument("--wait-timeout", dest="wait_timeout", type=int, default=None)
    parser.add_argument("--log-dir", dest="log_dir", default=None)
    parser.add_argument("--venv-dir", dest="venv_dir", default=None)
    parser.add_argument("--model-local-root", dest="model_local_root", default=None)
    parser.add_argument("--venv-local-root", dest="venv_local_root", default=None)
    parser.add_argument("--dist-free-port", dest="dist_free_port_daai",
                        action="store_true", default=None)
    args = parser.parse_args(argv)

    # Load base config
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            cfg_path = (
                Path(__file__).resolve().parent.parent / "config" / args.config / "server.json"
            )
        if not cfg_path.exists():
            cfg_path = Path(args.config)
        cfg = load_json_config(cfg_path)
    else:
        platform = args.platform or "gpuhome"
        cfg = dict(DEFAULT_PLATFORMS.get(platform, DEFAULT_PLATFORMS["gpuhome"]))

    # CLI overrides
    overrides = {k: v for k, v in vars(args).items()
                 if v is not None and k not in ("config",)}
    cfg = merge_cli_overrides(cfg, overrides)

    # Handle --disable-* flags
    if args.disable_cache_report:
        cfg["enable_cache_report"] = False
    if args.disable_hicache:
        cfg["enable_hicache"] = False

    return cfg


# ---------------------------------------------------------------------------
# Peer target computation
# ---------------------------------------------------------------------------

def build_stages(
    nnodes: int,
    dp_size: int,
    pp_size: int,
    tp_size: int,
    kv_backup: str,
    node_rank: int,
    node_name: str,
    server_port: int,
    head_node: str,
) -> list[dict[str, Any]]:
    pp_size_per_node = max(pp_size // nnodes, 1)
    nnodes_per_pp_rank = max(nnodes // pp_size, 1)
    nnodes_per_tp_group = nnodes_per_pp_rank
    tp_size_per_node = max(tp_size // nnodes_per_tp_group, 1)
    pp_rank_range = range(
        pp_size_per_node * (node_rank // nnodes_per_pp_rank),
        pp_size_per_node * (node_rank // nnodes_per_pp_rank + 1),
    )
    tp_rank_range = range(
        tp_size_per_node * (node_rank % nnodes_per_tp_group),
        tp_size_per_node * (node_rank % nnodes_per_tp_group + 1),
    )
    stages = []
    for dp_rank in range(dp_size):
        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                kill_pattern = f"sglang::scheduler_DP{dp_rank}"
                if pp_size > 1:
                    kill_pattern += f"_PP{pp_rank}"
                if tp_size > 1:
                    kill_pattern += f"_TP{tp_rank}"
                stages.append({
                    "dp_rank": dp_rank,
                    "pp_rank": pp_rank,
                    "tp_rank": tp_rank,
                    "node_rank": node_rank,
                    "node": node_name,
                    "node_url": f"http://{node_name}:{server_port}",
                    "head_server_url": f"http://{head_node}:{server_port}",
                    "kill_pattern": kill_pattern,
                    "kv_backup": kv_backup,
                })
    return stages


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def wait_http_healthy(url: str, pid: int, timeout_s: int = 1800) -> bool:
    start = time.time()
    while True:
        r = subprocess.run(
            ["curl", "--noproxy", "*", "-fsS", "-m", "3", url],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            return True
        # Check if process is still alive
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        if time.time() - start >= timeout_s:
            return False
        time.sleep(2)


# ---------------------------------------------------------------------------
# Remote backup server launch planning
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RemoteBackupLaunchPlan:
    mode: str
    client_host: str
    cmd: list[str]
    env: dict[str, str]
    cleanup_cmd: list[str] | None = None


def build_remote_backup_launch_plan(
    *,
    repo_root: Path,
    venv_python: Path,
    backup_port: int,
    backup_buffer_gb: float,
) -> RemoteBackupLaunchPlan:
    pythonpath = f"{repo_root / 'python'}:{repo_root}"
    env = {
        **os.environ,
        "PYTHONPATH": pythonpath,
        "PYTHONUNBUFFERED": "1",
        "NO_PROXY": os.environ.get("NO_PROXY", "localhost,127.0.0.1,0.0.0.0,::1"),
        "no_proxy": os.environ.get("no_proxy", "localhost,127.0.0.1,0.0.0.0,::1"),
    }
    module_args = [
        str(venv_python),
        "-u",
        "-m",
        "sglang.srt.mem_cache.storage.remote_backup.launch_remote_backup_server",
        "--port",
        str(backup_port),
        "--buffer-size-gb",
        str(backup_buffer_gb),
    ]
    return RemoteBackupLaunchPlan(
        mode="direct",
        client_host="127.0.0.1",
        cmd=module_args,
        env=env,
        cleanup_cmd=None,
    )


# ---------------------------------------------------------------------------
# Build launch command
# ---------------------------------------------------------------------------

def build_launch_cmd(
    cfg: dict[str, Any],
    model_path: str,
    node_rank: int,
    total_nodes: int,
    server_port: int,
    dist_init_addr: str,
    effective_hicache_extra_config: str,
    py_bin: str,
) -> list[str]:
    cmd = [
        py_bin, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--dp-size", str(cfg["dp_size"]),
        "--host", "0.0.0.0",
        "--port", str(server_port),
        "--nnodes", str(total_nodes),
        "--node-rank", str(node_rank),
        "--tp", str(cfg["tp_size"]),
        "--pp-size", str(cfg["pp_size"]),
        "--disable-overlap-schedule",
        "--attention-backend", "triton",
        "--sampling-backend", "pytorch",
        "--enable-metrics",
        "--tool-call-parser", cfg["tool_call_parser"],
    ]
    if cfg.get("reasoning_parser"):
        cmd.extend(["--reasoning-parser", str(cfg["reasoning_parser"])])
    ctx_len = int(cfg.get("context_length", 0) or 0)
    if ctx_len > 0:
        cmd.extend(["--context-length", str(ctx_len)])
    model_override_args = str(cfg.get("json_model_override_args", "") or "").strip()
    if model_override_args and model_override_args != "{}":
        cmd.extend(["--json-model-override-args", model_override_args])
    if cfg.get("retry_queue_same_priority_as_waiting"):
        cmd.append("--retry-queue-same-priority-as-waiting")
    chat_tmpl = cfg.get("chat_template", "") or ""
    if chat_tmpl:
        # Resolve relative paths against the benchmark src root so configs can
        # reference templates shipped with the experiment.
        chat_tmpl_path = chat_tmpl
        if not os.path.isabs(chat_tmpl_path):
            src_root = Path(__file__).resolve().parents[1]
            candidate = (src_root / chat_tmpl_path).resolve()
            if candidate.exists():
                chat_tmpl_path = str(candidate)
        cmd.extend(["--chat-template", chat_tmpl_path])
    if total_nodes > 1:
        cmd.extend(["--dist-init-addr", dist_init_addr])
    if cfg.get("enable_cache_report"):
        cmd.append("--enable-cache-report")
    if cfg.get("enable_hicache"):
        cmd.append("--enable-hierarchical-cache")
        if cfg.get("hicache_size_gb", 0) > 0:
            cmd.extend(["--hicache-size", str(cfg["hicache_size_gb"])])
        elif cfg.get("hicache_ratio"):
            cmd.extend(["--hicache-ratio", str(cfg["hicache_ratio"])])
        if cfg.get("insert_step", 0) > 0:
            cmd.extend(["--insert-step", str(cfg["insert_step"])])
    if cfg.get("quantization"):
        cmd.extend(["--quantization", cfg["quantization"]])
    if cfg.get("kv_backup"):
        cmd.extend(["--kv-backup-strategy", str(cfg["kv_backup"])])
    backend = cfg.get("hicache_storage_backend", "")
    if not backend and cfg.get("kv_backup", "") == "remote_backup":
        backend = "remote_backup"
    if backend in ("file", "mooncake", "hf3fs", "nixl", "aibrix", "dynamic", "eic", "remote_backup"):
        cmd.extend(["--hicache-storage-backend", backend])
    if cfg.get("hicache_storage_prefetch_policy"):
        cmd.extend(["--hicache-storage-prefetch-policy",
                     cfg["hicache_storage_prefetch_policy"]])
    if effective_hicache_extra_config:
        cmd.extend(["--hicache-storage-backend-extra-config",
                    effective_hicache_extra_config])
    lb = cfg.get("load_balance_method")
    if lb:
        cmd.extend(["--load-balance-method", str(lb)])
    return cmd


# ---------------------------------------------------------------------------
# Per-node server launcher (run on each node via srun)
# ---------------------------------------------------------------------------

def run_node_server(
    cfg: dict[str, Any],
    node_rank: int,
    node_name: str,
    log_dir: Path,
    total_nodes: int,
    server_port: int,
    dist_init_addr: str,
    head_node: str,
    remote_backup_url: str,
    server_pid_out: dict[str, int],
) -> None:
    server_log = get_process_log_path(log_dir, "server")

    # Environment
    os.environ["NO_PROXY"] = f"localhost,127.0.0.1,0.0.0.0,::1,{node_name}"
    os.environ["no_proxy"] = os.environ["NO_PROXY"]
    os.environ["LC_ALL"] = os.environ.get("LC_ALL", "C.UTF-8")
    os.environ["LANG"] = os.environ.get("LANG", "C.UTF-8")
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["SGLANG_DISABLE_CUDNN_CHECK"] = "1"
    os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"
    os.environ["TORCH_NCCL_DUMP_ON_TIMEOUT"] = "0"

    # Resolve python binary
    venv_dir = Path(cfg.get("venv_dir", ".venv"))
    if venv_dir.is_absolute() or (venv_dir / "bin" / "python").exists():
        py_bin = str(venv_dir / "bin" / "python")
    elif (Path.home() / venv_dir.name).exists():
        py_bin = str(Path.home() / venv_dir.name / "bin" / "python")
    else:
        py_bin = sys.executable

    # Remote backup extra config
    effective_hicache_extra_config = ""
    if cfg.get("kv_backup", "") == "remote_backup":
        remote_backup_port = cfg.get("remote_backup_port_base", 30000)
        if not remote_backup_url:
            remote_backup_url = f"{head_node}:{remote_backup_port}"
        extra = {
            "remote_backup_url": remote_backup_url,
            "remote_backup_buffer_size_gb": float(cfg.get("remote_backup_buffer_size_gb", 32.0)),
            "remote_backup_mode": "client",
        }
        effective_hicache_extra_config = json.dumps(extra, separators=(",", ":"))
        os.environ["SGLANG_REMOTE_BACKUP_URL"] = remote_backup_url

    # Resolve model path
    model_path = cfg["model_path"]
    model_local_root = cfg.get("model_local_root", "")
    if model_local_root:
        model_path = resolve_model_path(
            model_path, model_local_root, py_bin, log_dir
        )

    local_health_url = f"http://127.0.0.1:{server_port}/health"
    external_server_url = f"http://{node_name}:{server_port}"

    cmd = build_launch_cmd(
        cfg, model_path, node_rank, total_nodes, server_port,
        dist_init_addr, effective_hicache_extra_config, py_bin,
    )
    log(f"[node {node_rank}] launch cmd: {' '.join(repr(a) for a in cmd)}")

    log(f"[node {node_rank}] Launching server on {node_name}...")
    proc = subprocess.Popen(
        cmd,
        stdout=server_log.open("ab"),
        stderr=subprocess.STDOUT,
        env=os.environ,
    )
    pid = proc.pid
    server_pid_out["pid"] = pid

    timeout = cfg.get("wait_timeout", 1800)
    if not wait_http_healthy(local_health_url, pid, timeout):
        log_err(f"[node {node_rank}] Server failed to become healthy at {external_server_url}")
        proc.terminate()
        proc.wait()
        sys.exit(1)

    log(f"[node {node_rank}] Server healthy at {external_server_url}")

    kv_backup = cfg.get("kv_backup", "none")
    stages = build_stages(
        nnodes=total_nodes,
        dp_size=cfg["dp_size"],
        pp_size=cfg["pp_size"],
        tp_size=cfg["tp_size"],
        kv_backup=kv_backup,
        node_rank=node_rank,
        node_name=node_name,
        server_port=server_port,
        head_node=head_node,
    )
    manifest = {
        "node_rank": node_rank,
        "node": node_name,
        "node_url": external_server_url,
        "head_server_url": f"http://{head_node}:{server_port}",
        "server_port": server_port,
        "server_main_pid": pid,
        "kv_backup": kv_backup,
        "stages": stages,
    }
    print(encode_control_event("node_ready", manifest=manifest), flush=True)

    proc.wait()


def resolve_model_path(
    model_path: str,
    local_root: str,
    py_bin: str,
    log_dir: Path,
) -> str:
    """Resolve model path, caching locally if local_root is set."""
    from pathlib import Path as P
    import hashlib, re

    # Sanitize model name to cache key
    key = re.sub(r"[^A-Za-z0-9._-]+", "__", P(model_path).name).strip("._") or "model"
    cache_dir = P(local_root) / key
    ready_marker = cache_dir / ".model_cache_ready"

    if ready_marker.exists():
        log(f"  Reusing local model cache: {cache_dir}")
        return str(cache_dir)

    log(f"  Caching model to local disk: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            [py_bin, "-c",
             f"from huggingface_hub import snapshot_download; "
             f"print(snapshot_download(repo_id='{model_path}', local_files_only=True))"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0 and P(result.stdout.strip()).exists():
            src = P(result.stdout.strip())
            subprocess.run(["cp", "-aL", f"{src}/.", f"{cache_dir}/"],
                          check=True, capture_output=True)
        else:
            subprocess.run(
                [py_bin, "-c",
                 f"from huggingface_hub import snapshot_download; "
                 f"snapshot_download(repo_id='{model_path}', "
                 f"local_dir='{cache_dir}', "
                 f"local_dir_use_symlinks=False, resume_download=True)"],
                check=True, capture_output=True, timeout=3600,
            )
    except Exception as ex:
        log(f"  Model copy/download failed, using original path: {ex}")

    ready_marker.write_text(datetime.now().isoformat() + "\n", encoding="utf-8")
    return str(cache_dir)


# ---------------------------------------------------------------------------
# Aggregate manifests and write shared artifacts
# ---------------------------------------------------------------------------

def build_shared_manifest(
    slurm_job_id: str,
    node_list: list[str],
    server_port: int,
    manifests: list[dict[str, Any]],
    remote_backup_url: str = "",
) -> dict[str, Any]:
    all_stages: list[dict] = []
    for m in manifests:
        all_stages.extend(m.get("stages", []))
    all_stages.sort(key=lambda s: (s["dp_rank"], s["pp_rank"], s["tp_rank"]))

    combined: dict[str, Any] = {
        "slurm_job_id": slurm_job_id,
        "kv_backup": manifests[0].get("kv_backup", "none") if manifests else "none",
        "server_url": f"http://{node_list[0]}:{server_port}",
        "head_server_url": manifests[0].get("head_server_url", "") if manifests else "",
        "server_port": server_port,
        "servers": manifests,
        "stages": all_stages,
    }
    if remote_backup_url:
        combined["remote_backup_url"] = remote_backup_url
    return combined


# ---------------------------------------------------------------------------
# Main deployment
# ---------------------------------------------------------------------------

def deploy(cfg: dict[str, Any]) -> int:
    slurm_env = get_slurm_env()
    slurm_job_id = slurm_env.get("SLURM_JOB_ID", "")

    # Capture full SLURM node list at top-level (before any srun spawns children).
    # Once srun children start, SLURM_JOB_NODELIST only covers that node's allocation.
    if slurm_env["SLURM_JOB_NODELIST"]:
        node_list = get_node_list()
        total_nodes = get_total_nodes()
    else:
        node_list = ["localhost"]
        total_nodes = 1

    nnodes = cfg.get("nnodes", total_nodes)
    dp_size = cfg.get("dp_size", 1)
    pp_size = cfg.get("pp_size", 1)
    tp_size = cfg.get("tp_size", 1)

    log_dir = Path(cfg.get("log_dir", f"logs/{cfg.get('name', 'server')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    log_dir.mkdir(parents=True, exist_ok=True)

    server_port_base = int(cfg.get("server_port_base", 28000))
    remote_backup_port_base = int(cfg.get("remote_backup_port_base", 30000))
    dist_init_port_offset = cfg.get("dist_init_port_offset", 100)

    server_port = server_port_base
    dist_init_port = server_port_base + dist_init_port_offset
    if cfg.get("dist_free_port_daai") and slurm_env["SLURM_JOB_NODELIST"]:
        dist_start = server_port_base + dist_init_port_offset
        dist_end = remote_backup_port_base - 1
        dist_init_port = find_free_port(node_list[0], dist_start, dist_end)
        log(f"Using free dist-init port {dist_init_port}")
    dist_init_addr = f"{node_list[0]}:{dist_init_port}"

    # Remote backup URL (head node hosts the backup server)
    remote_backup_url = ""
    backup_port = remote_backup_port_base
    log(f"[DEBUG] kv_backup='{cfg.get('kv_backup', 'none')}', "
        f"remote_backup_port_base={cfg.get('remote_backup_port_base', 'N/A')}, "
        f"remote_backup_buffer_size_gb={cfg.get('remote_backup_buffer_size_gb', 'N/A')}, "
        f"hicache_storage_backend='{cfg.get('hicache_storage_backend', '')}'")
    log(
        f"[DEBUG] resolved ports: server_port={server_port}, "
        f"dist_init_addr={dist_init_addr}, backup_port={backup_port}"
    )
    if cfg.get("kv_backup", "") == "remote_backup":
        # Resolve venv_dir and repo_root early (needed for backup server launch)
        _repo_root = Path(__file__).resolve().parents[4]
        _venv_dir = Path(cfg.get("venv_dir", ".venv"))
        if not _venv_dir.is_absolute():
            _venv_dir = _repo_root / _venv_dir
        backup_buffer_gb = cfg.get("remote_backup_buffer_size_gb", 32.0)
        backup_log = get_process_log_path(log_dir, "backup_server")
        # Clear previous log
        backup_log.write_bytes(b"")
        venv_python = _venv_dir / "bin" / "python"

        # Build the launch plan and start the backup server as a direct subprocess.
        plan = build_remote_backup_launch_plan(
            repo_root=_repo_root,
            venv_python=venv_python,
            backup_port=backup_port,
            backup_buffer_gb=backup_buffer_gb,
        )
        log(f"Remote backup server launching directly at 127.0.0.1:{backup_port} ...")
        backup_proc = subprocess.Popen(
            plan.cmd,
            stdout=backup_log.open("ab"),
            stderr=subprocess.STDOUT,
            env=plan.env,
        )

        # Poll with raw socket (server speaks binary protocol, not HTTP)
        import socket
        backup_ready = False
        for attempt in range(120):
            if backup_proc.poll() is not None:
                stdout, _ = backup_proc.communicate()
                log_err(f"  backup process exited early (rc={backup_proc.returncode})")
                if stdout:
                    log_err(f"  stdout: {stdout[:500].decode(errors='replace')}")
                break
            try:
                with socket.create_connection((plan.client_host, backup_port), timeout=2) as sock:
                    sock.sendall(b"\xff")  # CMD_HEALTH
                    resp = sock.recv(1)
                    if resp == b"\x01":
                        remote_backup_url = f"127.0.0.1:{backup_port}"
                        log(f"  Remote backup server ready at {remote_backup_url}")
                        backup_ready = True
                        break
            except (OSError, ConnectionRefusedError, socket.timeout):
                pass
            time.sleep(2)
        if not backup_ready:
            log_err(f"Remote backup server failed to respond at 127.0.0.1:{backup_port}")
            if backup_log.exists() and backup_log.stat().st_size > 0:
                log_err(f"  Backup server log content:")
                for line in backup_log.read_text(encoding="utf-8").strip().split("\n")[-20:]:
                    log_err(f"    {line}")
            if backup_proc.poll() is None:
                backup_proc.terminate()
                try:
                    backup_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    backup_proc.kill()
            return 1

    log(f"Deploying SGLang server (DP={dp_size} PP={pp_size} TP={tp_size} nodes={total_nodes})")
    log(f"  kv_backup:       {cfg.get('kv_backup', 'none')}")
    log(f"  nnodes:          {total_nodes}")
    log(f"  nodes:           {node_list}")

    # Launch per-node servers via srun -w (each gets a fresh SLURM env for its node)
    srun_procs: list[subprocess.Popen[str]] = []
    manifests_data: list[dict[str, Any]] = []
    ready_queue: queue.Queue[tuple[int, dict[str, Any]]] = queue.Queue()

    repo_root = Path(__file__).resolve().parents[4]
    venv_dir = Path(cfg.get("venv_dir", ".venv"))
    if not venv_dir.is_absolute():
        venv_dir = repo_root / venv_dir
    cfg_json = json.dumps(cfg, separators=(",", ":"))

    def _forward_node_output(node_rank: int, proc: subprocess.Popen[str]) -> None:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            print(line, end="", flush=True)
            record = decode_control_event(line.rstrip("\n"))
            if record and record.get("event") == "node_ready":
                manifest = record.get("manifest")
                if isinstance(manifest, dict):
                    ready_queue.put((node_rank, manifest))

    for node_rank, node_name in enumerate(node_list):
        cmd = [
            "srun", "-w", node_name, "--nodes=1", "--ntasks=1",
            "--overlap",
            "--kill-on-bad-exit=0",
            "/usr/bin/env",
            f"PATH={venv_dir / 'bin'}:{os.environ.get('PATH', '')}",
            f"PYTHONPATH={os.environ.get('PYTHONPATH', '')}",
            f"NODE_RANK={node_rank}",
            f"NODE_NAME={node_name}",
            f"LOG_DIR={log_dir}",
            f"TOTAL_NODES={total_nodes}",
            f"SERVER_PORT={server_port}",
            f"DIST_INIT_ADDR={dist_init_addr}",
            f"HEAD_NODE={node_list[0]}",
            f"REMOTE_BACKUP_URL={remote_backup_url}",
            f"SCRIPT_DIR={Path(__file__).parent}",
            f"REPO_ROOT={repo_root}",
            f"VENV_DIR={venv_dir}",
            f"CFG_JSON={cfg_json}",
            str(venv_dir / "bin" / "python"), __file__, "__run_node_server",
        ]

        log(f"Starting node_rank={node_rank} on {node_name}")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        srun_procs.append(proc)
        threading.Thread(
            target=_forward_node_output,
            args=(node_rank, proc),
            daemon=True,
        ).start()

    # Wait for all nodes
    timeout = cfg.get("wait_timeout", 1800)
    wait_start = time.time()
    ready = [False] * total_nodes
    failed_nodes: list[str] = []

    while not all(ready) and time.time() - wait_start < timeout:
        try:
            node_rank, manifest = ready_queue.get(timeout=1.0)
        except queue.Empty:
            for i, proc in enumerate(srun_procs):
                if not ready[i] and proc.poll() is not None:
                    failed_nodes.append(node_list[i])
            if failed_nodes:
                break
            continue
        if not ready[node_rank]:
            manifests_data.append(manifest)
            ready[node_rank] = True
            log(f"  Node node_rank={node_rank} ready: {manifest.get('node_url', '')}")

    if not all(ready):
        elapsed = int(time.time() - wait_start)
        log_err(f"Timeout ({elapsed}s) waiting for nodes to become healthy.")
        for proc in srun_procs:
            if proc.poll() is None:
                proc.kill()
        # Kill backup server if it was launched
        if remote_backup_url:
            if backup_proc and backup_proc.poll() is None:
                backup_proc.terminate()
                try:
                    backup_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    backup_proc.kill()
        return 1

    manifest = build_shared_manifest(
        slurm_job_id, node_list, server_port, manifests_data, remote_backup_url,
    )

    log(f"Head server URL: http://{node_list[0]}:{server_port}")
    print(encode_control_event("server_ready", manifest=manifest), flush=True)
    log(f"SLURM job {slurm_job_id} is ready. Blocking until interrupted.")

    # Wait for srun processes
    for proc in srun_procs:
        proc.wait()

    log("All node servers have exited.")
    if backup_proc and backup_proc.poll() is None:
        backup_proc.terminate()
        try:
            backup_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backup_proc.kill()
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    argv = sys.argv[1:]

    # __run_node_server mode: invoked by srun on each node.
    # Reads config from CFG_JSON to avoid writing per-run config files.
    if argv and argv[0] == "__run_node_server":
        cfg_json = os.environ.get("CFG_JSON", "")
        if cfg_json:
            cfg = json.loads(cfg_json)
        else:
            cfg = {
                "dp_size": int(os.environ.get("EFF_DP_SIZE", "1")),
                "pp_size": int(os.environ.get("EFF_PP_SIZE", "1")),
                "tp_size": int(os.environ.get("EFF_TP_SIZE", "1")),
                "model_path": os.environ.get("EFF_MODEL_PATH", "Qwen/Qwen3-8B"),
                "tool_call_parser": os.environ.get("EFF_TOOL_CALL_PARSER", "qwen"),
                "enable_cache_report": os.environ.get("EFF_ENABLE_CACHE_REPORT", "0") == "1",
                "enable_hicache": os.environ.get("EFF_ENABLE_HICACHE", "0") == "1",
                "hicache_size_gb": int(os.environ.get("EFF_HICACHE_SIZE", "0")),
                "hicache_ratio": float(os.environ.get("EFF_HICACHE_RATIO", "2.0")),
                "insert_step": int(os.environ.get("EFF_INSERT_STEP", "0")),
                "kv_backup": os.environ.get("EFF_KV_BACKUP", "none"),
                "remote_backup_port_base": int(os.environ.get("EFF_REMOTE_BACKUP_PORT_BASE", "30000")),
                "remote_backup_buffer_size_gb": float(os.environ.get("EFF_REMOTE_BACKUP_BUFFER_SIZE_GB", "32.0")),
                "hicache_storage_backend": os.environ.get("EFF_HICACHE_STORAGE_BACKEND", ""),
                "hicache_storage_prefetch_policy": os.environ.get("EFF_HICACHE_STORAGE_PREFETCH_POLICY", ""),
                "quantization": os.environ.get("EFF_QUANTIZATION", ""),
                "wait_timeout": int(os.environ.get("EFF_WAIT_TIMEOUT", "1800")),
                "venv_dir": os.environ.get("EFF_VENV_DIR", ".venv"),
                "model_local_root": os.environ.get("EFF_MODEL_LOCAL_ROOT", ""),
            }

        pid_holder: dict[str, int] = {}
        run_node_server(
            cfg=cfg,
            node_rank=int(os.environ.get("NODE_RANK", "0")),
            node_name=os.environ.get("NODE_NAME", "localhost"),
            log_dir=Path(os.environ.get("LOG_DIR", "logs")),
            total_nodes=int(os.environ.get("TOTAL_NODES", "1")),
            server_port=int(os.environ.get("SERVER_PORT", "28000")),
            dist_init_addr=os.environ.get("DIST_INIT_ADDR", "localhost:28100"),
            head_node=os.environ.get("HEAD_NODE", "localhost"),
            remote_backup_url=os.environ.get("REMOTE_BACKUP_URL", ""),
            server_pid_out=pid_holder,
        )
        return 0

    # Normal mode: load config and deploy
    cfg = load_config(argv)
    return deploy(cfg)


if __name__ == "__main__":
    sys.exit(main())
