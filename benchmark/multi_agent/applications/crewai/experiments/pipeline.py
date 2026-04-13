# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.

"""Orchestrate baseline vs gpu_backup CrewAI experiments (replaces run_ab_experiment_crewai_qwen3_32b.sh)."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

_CREWAI_ROOT = Path(__file__).resolve().parents[1]
if str(_CREWAI_ROOT) not in sys.path:
    sys.path.insert(0, str(_CREWAI_ROOT))
from experiments.fault_injection import FaultInjectionConfig, FaultInjectionRunner


def multi_agent_dir() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass
class ExperimentArgs:
    job_limit: int
    app_workers: int
    inject_after_job: str
    inject_after_task: str
    inject_delay: int
    dp_size: int
    pp_size: int
    tp_size: int
    nnodes: int
    fault_dp_rank: str
    fault_pp_rank: str
    fault_tp_rank: str
    model_path: str
    model_local_root: str
    hicache_size: int
    quantization: str
    jobs_csv: Path
    default_year: str
    output_base_dir: Path
    groups: list[str]
    server_port_base: int
    peer_port_base: int
    slurm_gres: str
    fault_injection_method: str
    recovery_delay: str
    short_max_tokens: int
    long_max_tokens: int
    ignore_eos: int
    artifact_timeout_s: int
    deploy_slurm: Path
    log_dir: Path


_ACTIVE_SLURM_JOB: str = ""


def _cancel_slurm_job(job_id: str) -> None:
    if not job_id:
        return
    r = subprocess.run(
        ["squeue", "-j", job_id, "-h"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return
    subprocess.run(["scancel", job_id], check=False)
    time.sleep(5)


def _on_signal(_sig: int, _frame: object) -> None:
    global _ACTIVE_SLURM_JOB
    if _ACTIVE_SLURM_JOB:
        print(
            f"\nSignal caught: cancelling SLURM job {_ACTIVE_SLURM_JOB}",
            file=sys.stderr,
            flush=True,
        )
        _cancel_slurm_job(_ACTIVE_SLURM_JOB)
    raise SystemExit(130)


def _parse_sbatch_job_id(stdout: str) -> str:
    m = re.search(r"Submitted batch job (\d+)", stdout)
    return m.group(1) if m else ""


def _wait_slurm_running(job_id: str, timeout_s: int = 86400) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        r = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%T"],
            capture_output=True,
            text=True,
        )
        state = (r.stdout or "").strip()
        if state == "RUNNING":
            return True
        if state in ("FAILED", "COMPLETED", "CANCELLED", ""):
            return False
        time.sleep(10)
    return False


def _wait_server_artifacts(
    job_id: str,
    server_url_file: Path,
    stage_manifest_file: Path,
    timeout_s: int,
    log_dir: Path,
    log_fn: Optional[Callable[[str], None]] = None,
) -> bool:
    start = time.time()
    err_hint = log_dir / f"run_server_{job_id}.err"

    def log_err(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    while time.time() - start < timeout_s:
        if server_url_file.is_file() and server_url_file.stat().st_size > 0:
            if stage_manifest_file.is_file() and stage_manifest_file.stat().st_size > 0:
                return True
        r = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%T"],
            capture_output=True,
            text=True,
        )
        state = (r.stdout or "").strip()
        if state in ("FAILED", "COMPLETED", "CANCELLED", ""):
            log_err(
                f"Deploy job {job_id} left squeue (state={state!r}) before artifacts appeared. "
                f"Check {err_hint} on the cluster node."
            )
            return False
        time.sleep(5)
    log_err(
        f"Timeout {timeout_s}s waiting for {server_url_file.name} and {stage_manifest_file.name}. "
        f"See {err_hint}."
    )
    return False


def _venv_python() -> Path:
    venv = _CREWAI_ROOT / ".venv" / "bin" / "python"
    if venv.is_file():
        return venv
    return Path(sys.executable)


def _run_plot(script: Path, args: list[str], log_file: Optional[Path] = None) -> int:
    if not script.is_file():
        return 1
    py = _venv_python()
    cmd = [str(py), str(script), *args]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as fh:
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
        return r.returncode
    return subprocess.run(cmd).returncode


def run_one_group(
    ns: ExperimentArgs,
    label: str,
    peer_enabled: bool,
    result_dir: Path,
    experiment_log: Path,
) -> int:
    global _ACTIVE_SLURM_JOB
    figures_dir = result_dir / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        experiment_log.parent.mkdir(parents=True, exist_ok=True)
        with experiment_log.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    log("=" * 60)
    log(f"EXPERIMENT: {label}")
    log("=" * 60)
    log(
        "kv-backup (peer): "
        + ("YES" if peer_enabled else "NO (baseline)")
    )
    log(f"results: {result_dir}")

    server_url_file = ns.log_dir / "server_url.txt"
    stage_manifest_file = ns.log_dir / "stage_manifest.json"
    for p in (server_url_file, stage_manifest_file):
        if p.exists():
            p.unlink()

    sbatch_cmd: list[str] = [
        "sbatch",
        f"--nodes={ns.nnodes}",
    ]
    if ns.slurm_gres:
        sbatch_cmd.append(f"--gres={ns.slurm_gres}")
    sbatch_cmd.extend(
        [
            str(ns.deploy_slurm),
            "--dp-size",
            str(ns.dp_size),
            "--pp-size",
            str(ns.pp_size),
            "--tp-size",
            str(ns.tp_size),
            "--model-path",
            ns.model_path,
            "--model-local-root",
            ns.model_local_root,
            "--venv-local-root",
            "",
            "--server-port-base",
            str(ns.server_port_base),
            "--peer-port-base",
            str(ns.peer_port_base),
            "--enable-hicache",
            "--hicache-size",
            str(ns.hicache_size),
        ]
    )
    if peer_enabled:
        sbatch_cmd.append("--enable-peer-replication")
    if ns.quantization:
        sbatch_cmd.extend(["--quantization", ns.quantization])

    log(f"[{label}] Step 1: Submitting SLURM deploy job...")
    r = subprocess.run(sbatch_cmd, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    job_id = _parse_sbatch_job_id(out)
    if r.returncode != 0 or not job_id:
        log(f"ERROR: sbatch failed: {out}")
        return 1
    log(f"  SLURM job submitted: {job_id}")
    _ACTIVE_SLURM_JOB = job_id

    log(f"[{label}] Step 2: Waiting for job {job_id} to start...")
    if not _wait_slurm_running(job_id):
        log("ERROR: SLURM job did not reach RUNNING")
        _cancel_slurm_job(job_id)
        _ACTIVE_SLURM_JOB = ""
        return 1

    log(f"[{label}] Step 3: Waiting for server_url.txt and stage_manifest.json...")
    if not _wait_server_artifacts(
        job_id,
        server_url_file,
        stage_manifest_file,
        ns.artifact_timeout_s,
        ns.log_dir,
        log,
    ):
        log("ERROR: Server artifacts not ready")
        _cancel_slurm_job(job_id)
        _ACTIVE_SLURM_JOB = ""
        return 1

    head_url = server_url_file.read_text(encoding="utf-8").strip()
    shutil.copy2(server_url_file, result_dir / "server_url.txt")
    shutil.copy2(stage_manifest_file, result_dir / "stage_manifest.json")
    (result_dir / "slurm_job_id.txt").write_text(job_id + "\n", encoding="utf-8")

    trace_file = result_dir / "trace_log.json"
    topo = f"DP{ns.dp_size}_PP{ns.pp_size}_TP{ns.tp_size}_N{ns.nnodes}x4gpu"
    fi_cfg = FaultInjectionConfig(
        server_url=head_url,
        stage_manifest=stage_manifest_file,
        slurm_job_id=job_id,
        model_path=ns.model_path,
        jobs_csv=ns.jobs_csv,
        job_limit=ns.job_limit,
        app_workers=ns.app_workers,
        default_year=ns.default_year,
        output_dir=result_dir,
        trace_file=trace_file,
        inject_after_job=ns.inject_after_job if not ns.inject_after_task else "",
        inject_after_task=ns.inject_after_task,
        inject_delay=ns.inject_delay,
        fault_dp_rank=ns.fault_dp_rank,
        fault_pp_rank=ns.fault_pp_rank,
        fault_tp_rank=ns.fault_tp_rank,
        fault_injection_method=ns.fault_injection_method,
        recovery_delay=ns.recovery_delay,
        short_max_tokens=ns.short_max_tokens,
        long_max_tokens=ns.long_max_tokens,
        ignore_eos=ns.ignore_eos,
        experiment_mode=label,
        server_topology=topo,
    )
    runner = FaultInjectionRunner(fi_cfg)
    log(f"[{label}] Step 4: CrewAI fault injection against {head_url}...")
    test_rc = runner.run()
    if test_rc != 0:
        log(f"WARNING: CrewAI run exited with code {test_rc}")

    if trace_file.is_file() and trace_file.stat().st_size > 0:
        prof = _CREWAI_ROOT / "plots" / "trace_log_profile.py"
        _run_plot(
            prof,
            [
                "--input",
                str(trace_file),
                "--output",
                str(figures_dir / "trace_log_profile.png"),
            ],
            experiment_log,
        )
        metrics_json = result_dir / "internal_failover_metrics.json"
        if metrics_json.is_file():
            backup_plot = _CREWAI_ROOT / "plots" / "backup_cache_hits.py"
            _run_plot(
                backup_plot,
                [
                    "--failover-metrics",
                    str(metrics_json),
                    "--output",
                    str(figures_dir / "backup_cache_hit_profile.png"),
                ],
                experiment_log,
            )

    log(f"[{label}] Step 5: Cancelling deploy job {job_id}...")
    _cancel_slurm_job(job_id)
    _ACTIVE_SLURM_JOB = ""

    log(f"[{label}] Experiment complete.")
    if trace_file.is_file() and trace_file.stat().st_size > 0:
        return 0
    return test_rc


def parse_args() -> ExperimentArgs:
    ma = multi_agent_dir()
    default_log = ma / "logs/qwen3_32b_a100"
    default_results = ma / "applications/crewai/results/crewai_ab_qwen3_32b_a100"
    default_deploy = ma / "scripts" / "deploy_sglang.slurm"
    default_jobs = ma / "applications/crewai/topics.csv"
    p = argparse.ArgumentParser(
        description="Baseline vs gpu_backup CrewAI experiments (CPU orchestration)."
    )
    p.add_argument("--job-limit", type=int, default=6)
    p.add_argument("--app-workers", type=int, default=2)
    p.add_argument("--inject-after-job", default="1")
    p.add_argument("--inject-after-task", default="")
    p.add_argument("--inject-delay", type=int, default=10)
    p.add_argument("--dp-size", type=int, default=2)
    p.add_argument("--pp-size", type=int, default=2)
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--nnodes", type=int, default=1)
    p.add_argument("--fault-dp-rank", default="0")
    p.add_argument("--fault-pp-rank", default="0")
    p.add_argument("--fault-tp-rank", default="0")
    p.add_argument("--model-path", default="Qwen/Qwen3-32B")
    p.add_argument(
        "--model-local-root",
        default=os.environ.get("MODEL_LOCAL_ROOT", f"/tmp/{os.environ.get('USER', 'user')}/sglang-model-cache"),
    )
    p.add_argument("--hicache-size", type=int, default=40)
    p.add_argument("--min-hicache-size", type=int, default=40)
    p.add_argument("--quantization", default="")
    p.add_argument("--jobs-csv", type=Path, default=default_jobs)
    p.add_argument("--default-year", default="2025")
    p.add_argument("--output-dir", type=Path, default=default_results)
    p.add_argument(
        "--group",
        default="all",
        help="Comma-separated: baseline, gpu_backup, or all",
    )
    p.add_argument("--server-port-base", type=int, default=34000)
    p.add_argument("--peer-port-base", type=int, default=35000)
    p.add_argument(
        "--slurm-gres",
        default=os.environ.get("SGLANG_BENCH_GRES", "gpu:a100:4"),
    )
    p.add_argument(
        "--fault-injection-method",
        default=os.environ.get("FAULT_INJECTION_METHOD", "srun_pkill"),
    )
    p.add_argument("--recovery-delay", default="")
    p.add_argument("--short-max-tokens", type=int, default=int(os.environ.get("CREWAI_SHORT_MAX_TOKENS", "1024")))
    p.add_argument("--long-max-tokens", type=int, default=int(os.environ.get("CREWAI_LONG_MAX_TOKENS", "2048")))
    p.add_argument("--ignore-eos", type=int, default=int(os.environ.get("CREWAI_IGNORE_EOS", "1")))
    p.add_argument("--artifact-timeout", type=int, default=1200)
    p.add_argument("--deploy-slurm", type=Path, default=default_deploy)
    p.add_argument("--log-dir", type=Path, default=default_log)
    a = p.parse_args()
    hicache = a.hicache_size
    if hicache > 0 and hicache < a.min_hicache_size:
        hicache = a.min_hicache_size
    inj_job = a.inject_after_job
    inj_task = a.inject_after_task
    if inj_task:
        inj_job = ""
    elif not inj_job:
        inj_job = "1"
    raw_groups = [x.strip() for x in a.group.split(",") if x.strip()]
    valid = {"baseline", "gpu_backup", "all"}
    if not raw_groups:
        raw_groups = ["all"]
    if "all" in raw_groups:
        groups = ["baseline", "gpu_backup"]
    else:
        for g in raw_groups:
            if g not in valid:
                p.error(f"Invalid --group entry: {g}")
        groups = [g for g in raw_groups if g != "all"]
    return ExperimentArgs(
        job_limit=a.job_limit,
        app_workers=a.app_workers,
        inject_after_job=inj_job,
        inject_after_task=inj_task,
        inject_delay=a.inject_delay,
        dp_size=a.dp_size,
        pp_size=a.pp_size,
        tp_size=a.tp_size,
        nnodes=a.nnodes,
        fault_dp_rank=a.fault_dp_rank,
        fault_pp_rank=a.fault_pp_rank,
        fault_tp_rank=a.fault_tp_rank,
        model_path=a.model_path,
        model_local_root=a.model_local_root,
        hicache_size=hicache,
        quantization=a.quantization,
        jobs_csv=a.jobs_csv,
        default_year=a.default_year,
        output_base_dir=a.output_dir,
        groups=groups,
        server_port_base=a.server_port_base,
        peer_port_base=a.peer_port_base,
        slurm_gres=a.slurm_gres,
        fault_injection_method=a.fault_injection_method,
        recovery_delay=a.recovery_delay,
        short_max_tokens=a.short_max_tokens,
        long_max_tokens=a.long_max_tokens,
        ignore_eos=a.ignore_eos,
        artifact_timeout_s=a.artifact_timeout,
        deploy_slurm=a.deploy_slurm,
        log_dir=a.log_dir,
    )


def main() -> int:
    global _ACTIVE_SLURM_JOB
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    ns = parse_args()
    if not ns.deploy_slurm.is_file():
        print(f"ERROR: deploy slurm not found: {ns.deploy_slurm}", file=sys.stderr)
        return 1
    ts = time.strftime("%Y%m%d_%H%M%S")
    ns.log_dir.mkdir(parents=True, exist_ok=True)
    ns.output_base_dir.mkdir(parents=True, exist_ok=True)
    experiment_log = ns.log_dir / f"ab_experiment_crewai_32b_a100_{ts}.log"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with experiment_log.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    log("CrewAI baseline vs gpu_backup (DP2 PP2)")
    log(f"Model: {ns.model_path}")
    log(f"Groups: {ns.groups}")
    log(f"Output base: {ns.output_base_dir}")
    log(f"Experiment log: {experiment_log}")

    overall = 0
    for idx, group in enumerate(ns.groups):
        if idx > 0:
            log("Cooldown 15s before next experiment...")
            time.sleep(15)
        if group == "baseline":
            rc = run_one_group(
                ns,
                "baseline",
                False,
                ns.output_base_dir / "baseline",
                experiment_log,
            )
        elif group == "gpu_backup":
            rc = run_one_group(
                ns,
                "gpu_backup",
                True,
                ns.output_base_dir / "gpu_backup",
                experiment_log,
            )
        else:
            continue
        if rc != 0:
            overall = 1

    baseline_dir = ns.output_base_dir / "baseline"
    gpu_dir = ns.output_base_dir / "gpu_backup"
    if baseline_dir.is_dir() and gpu_dir.is_dir():
        ab_plot = _CREWAI_ROOT / "plots" / "ab_comparison.py"
        for ext in ("pdf", "png"):
            _run_plot(
                ab_plot,
                [
                    "--baseline",
                    str(baseline_dir),
                    "--treatment",
                    str(gpu_dir),
                    "--baseline-label",
                    "Baseline (no peer KV backup)",
                    "--treatment-label",
                    "GPU backup (peer KV replication)",
                    "--output",
                    str(ns.output_base_dir / f"ab_comparison__baseline_vs_gpu_backup.{ext}"),
                ],
                experiment_log,
            )

    log(f"Overall exit code: {overall}")
    return overall


if __name__ == "__main__":
    sys.exit(main())
