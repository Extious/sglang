# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.

"""CrewAI workload with GPU fault injection and recovery (replaces run_crewai_fault_injection_qwen3_32b.sh)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_CREWAI_DIR = Path(__file__).resolve().parent
if str(_CREWAI_DIR) not in sys.path:
    sys.path.insert(0, str(_CREWAI_DIR))
from trace_summaries import (
    write_job_summary_csv,
    write_task_summary_csv,
)


def append_events_record(events_path: Path, event_type: str, **fields: Any) -> None:
    record: dict[str, Any] = {
        "event": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    for k, v in fields.items():
        record[k] = int(v) if isinstance(v, str) and v.isdigit() else v
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=True) + "\n")


def load_stage_targets(
    manifest_path: Path,
    fault_dp: str,
    fault_pp: str,
    fault_tp: str,
) -> list[dict[str, str]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: list[dict[str, str]] = []
    for stage in payload.get("stages", []):
        if fault_dp and str(stage.get("dp_rank", "")) != fault_dp:
            continue
        if fault_pp and fault_pp not in {"all", "*"} and str(
            stage.get("pp_rank", "")
        ) != str(fault_pp):
            continue
        if fault_tp and fault_tp not in {"all", "*"} and str(
            stage.get("tp_rank", "")
        ) != str(fault_tp):
            continue
        out.append(
            {
                "dp_rank": str(stage.get("dp_rank", "")),
                "pp_rank": str(stage.get("pp_rank", "")),
                "tp_rank": str(stage.get("tp_rank", "")),
                "node": str(stage.get("node", "")),
                "node_url": str(stage.get("node_url", "")),
                "kill_pattern": str(stage.get("kill_pattern", "")),
            }
        )
    return out


def http_post(url: str, run_log: Optional[Path] = None) -> int:
    req = urllib.request.Request(url, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            _ = resp.read()
        return 0
    except (urllib.error.URLError, OSError) as e:
        if run_log:
            with run_log.open("a", encoding="utf-8") as fh:
                fh.write(f"http_post failed: {url} err={e}\n")
        return 1


def inject_gpu_failure_api(server_url: str, dp_rank: int, run_log: Path) -> int:
    base = server_url.rstrip("/")
    url = f"{base}/simulate_gpu_failure?dp_rank={dp_rank}"
    with run_log.open("a", encoding="utf-8") as fh:
        fh.write(f"POST {url}\n")
    return http_post(url, run_log)


def recover_gpu_api(server_url: str, dp_rank: int, run_log: Path) -> int:
    base = server_url.rstrip("/")
    url = f"{base}/simulate_gpu_recovery?dp_rank={dp_rank}"
    with run_log.open("a", encoding="utf-8") as fh:
        fh.write(f"POST {url}\n")
    return http_post(url, run_log)


def srun_pkill_stop(node_name: str, kill_pattern: str, run_log: Path) -> int:
    cmd = [
        "srun",
        "-w",
        node_name,
        "--nodes=1",
        "--ntasks=1",
        "--overlap",
        "--kill-on-bad-exit=0",
        "bash",
        "-lc",
        f'pkill -STOP -f "{kill_pattern}" || true',
    ]
    with run_log.open("a", encoding="utf-8") as fh:
        fh.write(f"srun SIGSTOP: {cmd}\n")
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    return r.returncode


def srun_pkill_cont(node_name: str, kill_pattern: str, run_log: Path) -> int:
    cmd = [
        "srun",
        "-w",
        node_name,
        "--nodes=1",
        "--ntasks=1",
        "--overlap",
        "--kill-on-bad-exit=0",
        "bash",
        "-lc",
        f'pkill -CONT -f "{kill_pattern}" || true',
    ]
    with run_log.open("a", encoding="utf-8") as fh:
        fh.write(f"srun SIGCONT: {cmd}\n")
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    return r.returncode


def wait_for_runtime_reaction(
    events_file: Optional[Path],
    failed_dp_rank: str,
    not_before_epoch: float,
    timeout_s: float,
) -> bool:
    if not events_file or not events_file.is_file():
        return False
    interesting = {
        "failover_dispatched",
        "failover_aborted",
        "resume_accepted",
        "resume_rejected",
    }
    deadline = time.time() + timeout_s

    def parse_ts(value: str) -> float:
        if not value:
            return 0.0
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()

    while time.time() < deadline:
        if events_file.exists():
            for line in events_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("event") not in interesting:
                    continue
                if str(record.get("failed_owner_dp_rank", "")) != failed_dp_rank:
                    continue
                if parse_ts(str(record.get("timestamp", ""))) < not_before_epoch:
                    continue
                return True
        time.sleep(1.0)
    return False


def snapshot_server_logs(runtime_events_file: str, output_dir: Path) -> None:
    if not runtime_events_file:
        return
    log_dir = Path(runtime_events_file).parent
    if not log_dir.is_dir():
        return
    snapshot_dir = output_dir / "server_logs"
    for path in sorted(log_dir.glob("node_*/server.log")):
        node_name = path.parent.name
        dest = snapshot_dir / node_name / "server.log"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)


@dataclass
class FaultInjectionConfig:
    server_url: str
    stage_manifest: Path
    slurm_job_id: str
    model_path: str
    jobs_csv: Path
    job_limit: int
    app_workers: int
    default_year: str
    output_dir: Path
    trace_file: Path
    inject_after_job: str = ""
    inject_after_task: str = ""
    inject_delay: int = 10
    inject_match_task_label: str = ""
    inject_match_agent_role: str = ""
    inject_match_worker_id: str = ""
    fault_dp_rank: str = ""
    fault_pp_rank: str = "0"
    fault_tp_rank: str = "0"
    unhealthy_timeout: int = 20
    fault_injection_method: str = "srun_pkill"
    recovery_delay: str = ""
    crewai_script: Optional[Path] = None
    failover_metrics_script: Optional[Path] = None
    short_max_tokens: int = 1024
    long_max_tokens: int = 2048
    ignore_eos: int = 1
    experiment_mode: str = ""
    server_topology: str = ""


class FaultInjectionRunner:
    def __init__(self, cfg: FaultInjectionConfig):
        self.cfg = cfg
        self.crewai_dir = Path(__file__).resolve().parent
        self.venv_dir = self.crewai_dir / ".venv"
        self.crewai_py = cfg.crewai_script or (self.crewai_dir / "crewai_collaboration.py")
        self.events_file = cfg.output_dir / "events.jsonl"
        self.run_log = cfg.output_dir / "run.log"
        self.app_stdout = cfg.output_dir / "crewai_stdout.log"
        self.job_summary_csv = cfg.output_dir / "job_summary.csv"
        self.task_summary_csv = cfg.output_dir / "task_summary.csv"
        self.manifest_data = json.loads(cfg.stage_manifest.read_text(encoding="utf-8"))
        self.runtime_failover_events_file: Optional[str] = (
            self.manifest_data.get("runtime_failover_events_file") or ""
        ).strip() or None
        self.stage_targets = load_stage_targets(
            cfg.stage_manifest,
            cfg.fault_dp_rank,
            cfg.fault_pp_rank,
            cfg.fault_tp_rank,
        )
        if not self.stage_targets:
            raise RuntimeError("No stage targets matched the (dp, pp, tp) filter")
        if cfg.inject_after_job and cfg.inject_after_task:
            raise ValueError("Use only one of inject_after_job or inject_after_task")
        if not cfg.inject_after_job and not cfg.inject_after_task:
            cfg.inject_after_job = "1"
        if cfg.inject_after_task:
            self.inject_kind = "task"
            thresholds_raw = cfg.inject_after_task
            os.environ["CREWAI_PRINT_COMPLETED_ONLY"] = "0"
        else:
            self.inject_kind = "job"
            thresholds_raw = cfg.inject_after_job
            os.environ["CREWAI_PRINT_COMPLETED_ONLY"] = "1"
        self.inject_thresholds = [int(x.strip()) for x in thresholds_raw.split(",") if x.strip()]
        for t in self.inject_thresholds:
            if t <= 0:
                raise ValueError("Injection threshold must be > 0")
        self.next_inject_idx = 0
        self.job_done_count = 0
        self.task_done_count = 0
        self.fault_threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._log_lines: list[str] = []

    def log(self, msg: str) -> None:
        line = f"[{datetime.now().isoformat()}] {msg}"
        self._log_lines.append(line)
        self.run_log.parent.mkdir(parents=True, exist_ok=True)
        with self.run_log.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        print(line, flush=True)

    def _delayed_inject(
        self,
        event_id: int,
        trigger_kind: str,
        trigger_count: int,
        stage: dict[str, str],
    ) -> None:
        cfg = self.cfg
        time.sleep(cfg.inject_delay)
        dp_rank = stage["dp_rank"]
        pp_rank = stage["pp_rank"]
        tp_rank = stage["tp_rank"]
        node_name = stage["node"]
        node_url = stage["node_url"]
        kill_pattern = stage["kill_pattern"]
        append_events_record(
            self.events_file,
            "fault_injected",
            event_id=event_id,
            trigger_kind=trigger_kind,
            trigger_count=trigger_count,
            dp_rank=dp_rank,
            pp_rank=pp_rank,
            tp_rank=tp_rank,
            node=node_name,
            node_url=node_url,
            method=cfg.fault_injection_method,
            slurm_job_id=cfg.slurm_job_id,
        )
        fault_start = time.time()
        if cfg.fault_injection_method == "api":
            rc = inject_gpu_failure_api(cfg.server_url, int(dp_rank), self.run_log)
            if rc != 0:
                append_events_record(
                    self.events_file,
                    "fault_injection_failed",
                    event_id=event_id,
                    dp_rank=dp_rank,
                    node=node_name,
                    method="api",
                    status="api_failed",
                )
                return
        else:
            rc = srun_pkill_stop(node_name, kill_pattern, self.run_log)
            if rc != 0:
                append_events_record(
                    self.events_file,
                    "fault_injection_failed",
                    event_id=event_id,
                    dp_rank=dp_rank,
                    node=node_name,
                    method="srun_pkill",
                    status="pkill_failed",
                )
                return
        self.log(
            f"Fault {event_id}: GPU fault applied dp={dp_rank} pp={pp_rank} tp={tp_rank}"
        )
        ev_path = (
            Path(self.runtime_failover_events_file)
            if self.runtime_failover_events_file
            else None
        )
        if wait_for_runtime_reaction(
            ev_path, dp_rank, fault_start, float(cfg.unhealthy_timeout)
        ):
            append_events_record(
                self.events_file,
                "runtime_failover_reaction",
                event_id=event_id,
                dp_rank=dp_rank,
                node_url=node_url,
                status="runtime_reaction",
            )
        else:
            append_events_record(
                self.events_file,
                "gpu_failure_injected",
                event_id=event_id,
                dp_rank=dp_rank,
                node_url=node_url,
                status="api_success",
            )
        if cfg.recovery_delay.isdigit() and int(cfg.recovery_delay) > 0:
            self.log(
                f"Fault {event_id}: recovery scheduled in {cfg.recovery_delay}s"
            )
            time.sleep(int(cfg.recovery_delay))
            if cfg.fault_injection_method == "api":
                ok = recover_gpu_api(cfg.server_url, int(dp_rank), self.run_log) == 0
                ev = "gpu_recovered" if ok else "recovery_failed"
                append_events_record(
                    self.events_file,
                    ev,
                    event_id=event_id,
                    dp_rank=dp_rank,
                    method="api",
                    status="recovery_applied" if ok else "failed",
                )
            else:
                ok = srun_pkill_cont(node_name, kill_pattern, self.run_log) == 0
                ev = "gpu_recovered" if ok else "recovery_failed"
                append_events_record(
                    self.events_file,
                    ev,
                    event_id=event_id,
                    dp_rank=dp_rank,
                    method="sigcont",
                    status="recovery_applied" if ok else "failed",
                )

    def schedule_injection_if_needed(
        self,
        current_count: int,
        task_label: str = "",
        agent_role: str = "",
        worker_id: str = "",
        job_id: str = "",
        job_name: str = "",
    ) -> None:
        with self._lock:
            if self.next_inject_idx >= len(self.inject_thresholds):
                return
            threshold = self.inject_thresholds[self.next_inject_idx]
            if current_count < threshold:
                return
            event_id = self.next_inject_idx + 1
            stage_index = self.next_inject_idx % len(self.stage_targets)
            stage = dict(self.stage_targets[stage_index])
            self.next_inject_idx += 1
        append_events_record(
            self.events_file,
            "fault_scheduled",
            event_id=event_id,
            trigger_kind=self.inject_kind,
            trigger_count=current_count,
            dp_rank=stage["dp_rank"],
            pp_rank=stage["pp_rank"],
            tp_rank=stage["tp_rank"],
            node=stage["node"],
            node_url=stage["node_url"],
            inject_delay=self.cfg.inject_delay,
            task_label=task_label,
            agent_role=agent_role,
            worker_id=worker_id,
            job_id=job_id,
            job=job_name,
        )
        self.log(
            f"Fault {event_id}: trigger matched {self.inject_kind}={current_count}, "
            f"target=dp{stage['dp_rank']}/pp{stage['pp_rank']}/tp{stage['tp_rank']} on {stage['node']}"
        )
        t = threading.Thread(
            target=self._delayed_inject,
            args=(event_id, self.inject_kind, current_count, stage),
            daemon=True,
        )
        t.start()
        self.fault_threads.append(t)

    def task_matches_filters(
        self, task_label: str, agent_role: str, worker_id: str
    ) -> bool:
        c = self.cfg
        if c.inject_match_task_label and task_label != c.inject_match_task_label:
            return False
        if c.inject_match_agent_role and agent_role != c.inject_match_agent_role:
            return False
        if c.inject_match_worker_id and worker_id != c.inject_match_worker_id:
            return False
        return True

    def process_event_line(self, record: dict[str, Any]) -> None:
        ev = record.get("event", "")
        if ev == "job_completed":
            self.job_done_count += 1
            self.log(
                f"Observed job completion {self.job_done_count}: job_id={record.get('job_id')} job={record.get('job')}"
            )
            if self.inject_kind == "job":
                self.schedule_injection_if_needed(
                    self.job_done_count,
                    job_id=str(record.get("job_id", "")),
                    job_name=str(record.get("job", "")),
                    worker_id=str(record.get("worker_id", "")),
                )
        elif ev == "job_failed":
            self.log(f"Observed job failure: job_id={record.get('job_id')}")
        elif ev == "task_completed":
            tl = str(record.get("task_label", ""))
            ar = str(record.get("agent_role", ""))
            wid = str(record.get("worker_id", ""))
            if not self.task_matches_filters(tl, ar, wid):
                self.log(f"Observed task completion (ignored by filter): {tl} -> {ar}")
                return
            self.task_done_count += 1
            self.log(
                f"Observed task completion {self.task_done_count}: {tl} -> {ar}"
            )
            if self.inject_kind == "task":
                self.schedule_injection_if_needed(
                    self.task_done_count,
                    task_label=tl,
                    agent_role=ar,
                    worker_id=wid,
                    job_id=str(record.get("job_id", "")),
                    job_name=str(record.get("job", "")),
                )
        elif ev == "task_failed":
            self.log(f"Observed task failure: {record.get('task_label')}")

    def run(self) -> int:
        cfg = self.cfg
        for p in (self.run_log, self.app_stdout, self.events_file):
            if p.exists():
                p.unlink()
        self.log("Starting CrewAI fault injection run")
        self.log(f"Server URL: {cfg.server_url}")
        py = sys.executable
        if self.venv_dir.is_dir() and (self.venv_dir / "bin" / "python").is_file():
            py = str(self.venv_dir / "bin" / "python")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["CREWAI_SERVER_BASE_URL"] = cfg.server_url
        env["CREWAI_MODEL_PATH"] = cfg.model_path
        env["CREWAI_EVENTS_FILE"] = str(self.events_file)
        env["CREWAI_ENABLE_STREAM"] = "0"
        if cfg.experiment_mode:
            env["CREWAI_EXPERIMENT_MODE"] = cfg.experiment_mode
        if cfg.server_topology:
            env["CREWAI_SERVER_TOPOLOGY"] = cfg.server_topology
        env["CREWAI_SHORT_MAX_TOKENS"] = str(cfg.short_max_tokens)
        env["CREWAI_LONG_MAX_TOKENS"] = str(cfg.long_max_tokens)
        env["CREWAI_IGNORE_EOS"] = str(cfg.ignore_eos)
        env["FAULT_INJECTION_METHOD"] = cfg.fault_injection_method
        tmp_base = Path(os.environ.get("TMPDIR", "/tmp")) / os.environ.get("USER", "user") / "crewai-storage"
        db_name = f"job_{cfg.slurm_job_id}_{cfg.output_dir.name}"
        env["XDG_DATA_HOME"] = str(tmp_base / "xdg")
        env["SQLITE_TMPDIR"] = str(tmp_base / "sqlite-tmp")
        env["CREWAI_STORAGE_DIR"] = db_name
        Path(env["XDG_DATA_HOME"]).mkdir(parents=True, exist_ok=True)
        Path(env["SQLITE_TMPDIR"]).mkdir(parents=True, exist_ok=True)
        cmd = [
            py,
            str(self.crewai_py),
            "--csv",
            str(cfg.jobs_csv),
            "--limit",
            str(cfg.job_limit),
            "--workers",
            str(cfg.app_workers),
            "--trace-file",
            str(cfg.trace_file),
            "--default-year",
            cfg.default_year,
        ]
        self.log(f"CrewAI cmd: {cmd}")
        with self.app_stdout.open("wb") as out:
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.crewai_dir),
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
            )
        last_line = 0
        while proc.poll() is None:
            if self.events_file.is_file():
                lines = self.events_file.read_text(encoding="utf-8").splitlines()
                if len(lines) > last_line:
                    for line in lines[last_line:]:
                        if not line.strip():
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        self.process_event_line(rec)
                    last_line = len(lines)
            time.sleep(1.0)
        if self.events_file.is_file():
            lines = self.events_file.read_text(encoding="utf-8").splitlines()
            if len(lines) > last_line:
                for line in lines[last_line:]:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.process_event_line(rec)
        for t in self.fault_threads:
            t.join(timeout=600)
        rc = proc.wait()
        write_job_summary_csv(cfg.trace_file, self.job_summary_csv)
        write_task_summary_csv(cfg.trace_file, self.task_summary_csv)
        if self.runtime_failover_events_file:
            snapshot_server_logs(self.runtime_failover_events_file, cfg.output_dir)
        metrics_script = cfg.failover_metrics_script or (
            self.crewai_dir / "build_internal_failover_metrics.py"
        )
        if metrics_script.is_file():
            out_json = cfg.output_dir / "internal_failover_metrics.json"
            mcmd = [
                py,
                str(metrics_script),
                "--trace-log",
                str(cfg.trace_file),
                "--events-file",
                str(self.events_file),
                "--output",
                str(out_json),
            ]
            if self.runtime_failover_events_file:
                mcmd.extend(
                    ["--runtime-events-file", self.runtime_failover_events_file]
                )
            subprocess.run(mcmd, cwd=str(self.crewai_dir), check=False)
        self.log(f"CrewAI run finished with exit code {rc}")
        return rc


def parse_args() -> FaultInjectionConfig:
    p = argparse.ArgumentParser(description="CrewAI fault injection runner")
    p.add_argument("--server-url", required=True)
    p.add_argument("--stage-manifest", type=Path, required=True)
    p.add_argument("--slurm-job-id", required=True)
    p.add_argument("--model-path", default="Qwen/Qwen3-32B")
    p.add_argument("--jobs-csv", type=Path, required=True)
    p.add_argument("--job-limit", type=int, default=10)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--default-year", default="2025")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--trace-file", type=Path, required=True)
    p.add_argument("--inject-after-job", default="")
    p.add_argument("--inject-after-task", default="")
    p.add_argument("--inject-delay", type=int, default=10)
    p.add_argument("--inject-match-task-label", default="")
    p.add_argument("--inject-match-agent-role", default="")
    p.add_argument("--inject-match-worker-id", default="")
    p.add_argument("--fault-dp-rank", default="")
    p.add_argument("--fault-pp-rank", default="0")
    p.add_argument("--fault-tp-rank", default="0")
    p.add_argument("--unhealthy-timeout", type=int, default=20)
    p.add_argument("--fault-injection-method", default="srun_pkill")
    p.add_argument("--recovery-delay", default="")
    p.add_argument("--short-max-tokens", type=int, default=1024)
    p.add_argument("--long-max-tokens", type=int, default=2048)
    p.add_argument("--ignore-eos", type=int, default=1)
    p.add_argument("--experiment-mode", default="")
    p.add_argument("--server-topology", default="")
    a = p.parse_args()
    return FaultInjectionConfig(
        server_url=a.server_url,
        stage_manifest=a.stage_manifest,
        slurm_job_id=a.slurm_job_id,
        model_path=a.model_path,
        jobs_csv=a.jobs_csv,
        job_limit=a.job_limit,
        app_workers=a.workers,
        default_year=a.default_year,
        output_dir=a.output_dir,
        trace_file=a.trace_file,
        inject_after_job=a.inject_after_job,
        inject_after_task=a.inject_after_task,
        inject_delay=a.inject_delay,
        inject_match_task_label=a.inject_match_task_label,
        inject_match_agent_role=a.inject_match_agent_role,
        inject_match_worker_id=a.inject_match_worker_id,
        fault_dp_rank=a.fault_dp_rank,
        fault_pp_rank=a.fault_pp_rank,
        fault_tp_rank=a.fault_tp_rank,
        unhealthy_timeout=a.unhealthy_timeout,
        fault_injection_method=a.fault_injection_method,
        recovery_delay=a.recovery_delay,
        short_max_tokens=a.short_max_tokens,
        long_max_tokens=a.long_max_tokens,
        ignore_eos=a.ignore_eos,
        experiment_mode=a.experiment_mode,
        server_topology=a.server_topology,
    )


def main() -> int:
    cfg = parse_args()
    runner = FaultInjectionRunner(cfg)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
