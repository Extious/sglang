# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.

"""
Pure fault injection engine.

Usage::

    from experiments.injector import FaultInjector, FaultInjectorConfig

    cfg = FaultInjectorConfig(
        server_url="http://node0:28000",
        stage_manifest=Path("stage_manifest.json"),
        slurm_job_id="12345",
        inject_after_job="1",
        output_dir=Path("results/run_001"),
    )
    injector = FaultInjector(cfg)
    injector.monitor(events_file=Path("events.jsonl"))
    # returns when events_file reaches EOF or app process exits

Does NOT:
  - Deploy the server
  - Run the CrewAI client
  - Manage SLURM beyond injecting faults
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from failure.config import FaultInjectorConfig
from failure.metrics_utils import build_internal_failover_metrics


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

# Append event to events.jsonl file
def append_event(events_file: Path, event_type: str, **fields: Any) -> None:
    """Append a structured record to the events JSONL file."""
    record: dict[str, Any] = {
        "event": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    for k, v in fields.items():
        record[k] = int(v) if isinstance(v, str) and v.isdigit() else v
    events_file.parent.mkdir(parents=True, exist_ok=True)
    with events_file.open("a", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        fh.write(json.dumps(record, ensure_ascii=True) + "\n")
        fh.flush()
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

# Load stage targets from stage_manifest.json
def load_stage_targets(manifest_path: Path | dict, dp: str, pp: str, tp: str) -> list[dict]:
    """Extract matching stages from stage_manifest.json."""
    payload = (
        manifest_path
        if isinstance(manifest_path, dict)
        else json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    out = []
    for stage in payload.get("stages", []):
        if dp and str(stage.get("dp_rank", "")) != dp:
            continue
        if pp and pp not in ("all", "*") and str(stage.get("pp_rank", "")) != pp:
            continue
        if tp and tp not in ("all", "*") and str(stage.get("tp_rank", "")) != tp:
            continue
        out.append({
            "dp_rank": str(stage.get("dp_rank", "")),
            "pp_rank": str(stage.get("pp_rank", "")),
            "tp_rank": str(stage.get("tp_rank", "")),
            "node": str(stage.get("node", "")),
            "node_url": str(stage.get("node_url", "")),
            "kill_pattern": str(stage.get("kill_pattern", "")),
        })
    return out


# ---------------------------------------------------------------------------
# FaultTrigger: pluggable condition that decides when to fire
# ---------------------------------------------------------------------------

class TriggerContext:
    """Snapshot of current state passed to trigger.check()."""

    def __init__(
        self,
        job_count: int,
        task_count: int,
        event: str,
        rec: dict,
        elapsed_wall_s: float,
    ) -> None:
        self.job_count = job_count
        self.task_count = task_count
        self.event = event
        self.rec = rec
        self.elapsed_wall_s = elapsed_wall_s


class TriggerResult:
    """Returned by a trigger when it decides to fire."""

    def __init__(
        self,
        event_id: int,
        stage: dict,
        inject_delay: float = 0.0,
        recovery_delay: float = 0.0,
        **fields: Any,
    ) -> None:
        self.event_id = event_id
        self.stage = stage
        self.inject_delay = inject_delay
        self.recovery_delay = recovery_delay
        self.fields = fields


class FaultTrigger(ABC):
    """Abstract base for a fault trigger condition."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name, e.g. 'completion-job', 'timeline', 'custom'."""

    @abstractmethod
    def check(self, ctx: TriggerContext) -> Optional[TriggerResult]:
        """
        Called on every event or tick. Return TriggerResult to fire,
        or None to keep waiting.
        """
        ...

    def close(self) -> None:
        """Called when monitoring ends. Override for cleanup."""
        pass


# ---------------------------------------------------------------------------
# CompletionCountTrigger: fires after N job or task completions
# ---------------------------------------------------------------------------

class CompletionCountTrigger(FaultTrigger):
    """
    Fires when job_completed / task_completed reaches the next threshold.

    Filters task events by agent_role, task_label, and worker_id.
    Compatible with thresholds like "1,3,5" (fire on 1st, 3rd, 5th completion).
    """

    def __init__(
        self,
        kind: str,  # "job" or "task"
        thresholds: list[int],
        stage_targets: list[dict],
        inject_delay: float = 10.0,
        recovery_delay: float = 0.0,
        inject_match_task_label: str = "",
        inject_match_agent_role: str = "",
        inject_match_worker_id: str = "",
    ) -> None:
        self._name = f"completion-{kind}"
        self.kind = kind
        self.thresholds = thresholds
        self.stage_targets = stage_targets
        self.inject_delay = inject_delay
        self.recovery_delay = recovery_delay
        self.match_task_label = inject_match_task_label
        self.match_agent_role = inject_match_agent_role
        self.match_worker_id = inject_match_worker_id

        self._next_idx = 0
        self._job_count = 0
        self._task_count = 0

    @property
    def name(self) -> str:
        return self._name

    def check(self, ctx: TriggerContext) -> Optional[TriggerResult]:
        ev = ctx.event
        rec = ctx.rec

        if self.kind == "job":
            if ev != "job_completed":
                return None
            self._job_count += 1
            count = self._job_count
            fields = {
                "job_id": rec.get("job_id", ""),
                "job": rec.get("job", ""),
                "worker_id": rec.get("worker_id", ""),
            }
        else:
            if ev != "task_completed":
                return None
            tl = str(rec.get("task_label", ""))
            ar = str(rec.get("agent_role", ""))
            wid = str(rec.get("worker_id", ""))
            if self.match_task_label and tl != self.match_task_label:
                return None
            if self.match_agent_role and ar != self.match_agent_role:
                return None
            if self.match_worker_id and wid != self.match_worker_id:
                return None
            self._task_count += 1
            count = self._task_count
            fields = {
                "task_label": tl, "agent_role": ar,
                "worker_id": wid,
                "job_id": rec.get("job_id", ""),
                "job": rec.get("job", ""),
            }

        if self._next_idx >= len(self.thresholds):
            return None
        if count < self.thresholds[self._next_idx]:
            return None

        event_id = self._next_idx + 1
        self._next_idx += 1
        stage = self.stage_targets[(event_id - 1) % len(self.stage_targets)]
        return TriggerResult(
            event_id=event_id, stage=stage,
            inject_delay=self.inject_delay,
            recovery_delay=self.recovery_delay,
            trigger_kind=self.kind, trigger_count=count,
            **fields,
        )


# ---------------------------------------------------------------------------
# TimelineTrigger: fires at specific wall-clock times
# ---------------------------------------------------------------------------

class TimelineTrigger(FaultTrigger):
    """
    Fires at pre-configured wall-clock timestamps after experiment start.

    Usage example in fault_injection.json:
      "trigger_type": "timeline",
      "timeline_after_start_s": [60, 180, 300]
    """

    def __init__(
        self,
        fire_at_s: list[float],
        stage_targets: list[dict],
        inject_delay: float = 10.0,
    ) -> None:
        self._name = "timeline"
        self.fire_at_s = sorted(fire_at_s)
        self.stage_targets = stage_targets
        self.inject_delay = inject_delay
        self._next_idx = 0
        self._fired: list[int] = []

    @property
    def name(self) -> str:
        return self._name

    def check(self, ctx: TriggerContext) -> Optional[TriggerResult]:
        elapsed = ctx.elapsed_wall_s
        if self._next_idx >= len(self.fire_at_s):
            return None
        if elapsed < self.fire_at_s[self._next_idx]:
            return None
        event_id = self._next_idx + 1
        self._next_idx += 1
        stage = self.stage_targets[(event_id - 1) % len(self.stage_targets)]
        return TriggerResult(
            event_id=event_id, stage=stage,
            inject_delay=self.inject_delay,
            trigger_kind="timeline", trigger_time_s=elapsed,
        )

class FaultMethod(ABC):
    """Abstract fault injector."""

    @abstractmethod
    def inject(
        self,
        server_url: str,
        dp_rank: int,
        stage: dict,
        log: Callable[[str], None],
    ) -> bool: ...


class ApiFaultMethod(FaultMethod):
    """Inject via the SGLang /simulate_gpu_failure HTTP API."""

    def inject(
        self,
        server_url: str,
        dp_rank: int,
        stage: dict,
        log: Callable[[str], None],
    ) -> bool:
        url = f"{server_url.rstrip('/')}/simulate_gpu_failure"
        body = json.dumps({"dp_rank": dp_rank}).encode("utf-8")
        log(f"POST {url} body={body!r}")
        try:
            proxy_handler = urllib.request.ProxyHandler({})
            opener = urllib.request.build_opener(proxy_handler)
            req = urllib.request.Request(
                url, data=body, method="POST",
                headers={"Content-Type": "application/json"},
            )
            with opener.open(req, timeout=120) as resp:
                resp.read()
            return True
        except (urllib.error.URLError, OSError) as e:
            log(f"API inject failed: {e}")
            return False


class SlurmKillFaultMethod(FaultMethod):
    """Kill a server process via srun pkill on the target node."""

    def inject(
        self,
        server_url: str,
        dp_rank: int,
        stage: dict,
        log: Callable[[str], None],
    ) -> bool:
        node = stage.get("node", "")
        pattern = stage.get("kill_pattern", "")
        if not pattern or not node:
            return False

        script = "set -euo pipefail; " \
                 "matches=$(pgrep -f \"${KILL_PATTERN}\" || true); " \
                 '[ -z "$matches" ] && exit 3; ' \
                 'pkill -9 -f "${KILL_PATTERN}"'

        for gpu_opt in ("--gres=none", "--gpus=0", ""):
            cmd = ["srun", "--jobid", stage.get("_slurm_job_id", ""),
                   "-w", node, "--nodes=1", "--ntasks=1",
                   "--overlap", "--kill-on-bad-exit=0"]
            if gpu_opt:
                cmd.append(gpu_opt)
            cmd.extend([
                "/usr/bin/env", f"KILL_PATTERN={pattern}",
                "bash", "-lc", script,
            ])
            log(f"pkill srun: node={node} pattern={pattern}")
            result = subprocess.run(
                cmd, env={**os.environ, "NO_PROXY": "*", "no_proxy": "*"},
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                return True

        return False

    def _set_slurm_job_id(self, job_id: str) -> None:
        self._slurm_job_id = job_id


# ---------------------------------------------------------------------------
# FaultInjector
# ---------------------------------------------------------------------------

class FaultInjector:
    """
    Monitors events.jsonl and schedules GPU fault injections via pluggable triggers.

    Does NOT:
      - Deploy the server
      - Run the CrewAI client
      - Manage SLURM beyond using it to kill processes

    Recovery is handled per TriggerResult.recovery_delay — each trigger
    can prescribe its own recovery window. The injector appends events
    (fault_scheduled, fault_injected, runtime_failover_reaction,
    gpu_recovered, etc.) to events.jsonl for post-run analysis.
    """

    def __init__(self, cfg: FaultInjectorConfig) -> None:
        self.cfg = cfg
        self.events_file = cfg.events_file
        self.trace_file = cfg.trace_file

        if cfg.stage_manifest_data:
            manifest_data = cfg.stage_manifest_data
        elif cfg.stage_manifest:
            manifest_data = json.loads(cfg.stage_manifest.read_text(encoding="utf-8"))
        else:
            raise ValueError("FaultInjector requires stage_manifest_data or stage_manifest")
        self.failover_events_file = (
            cfg.failover_events_file
            or (manifest_data.get("runtime_failover_events_file") or "").strip()
            or None
        )
        self._start_time = time.time()
        self._counts_cond = threading.Condition()
        self._job_completed_count = 0
        self._task_completed_count = 0
        self._recorded_events: list[dict[str, Any]] = []

        stage_targets = load_stage_targets(
            manifest_data, cfg.fault_dp_rank,
            cfg.fault_pp_rank, cfg.fault_tp_rank,
        )
        if not stage_targets:
            raise ValueError(
                f"No stages match dp={cfg.fault_dp_rank} "
                f"pp={cfg.fault_pp_rank} tp={cfg.fault_tp_rank}"
            )

        # Attach slurm_job_id to each stage for SlurmKillFaultMethod
        if cfg.method == "slurm":
            method: FaultMethod = SlurmKillFaultMethod()
            method._set_slurm_job_id(cfg.slurm_job_id)
        else:
            method = ApiFaultMethod()
        self._method = method

        self._triggers = self._build_triggers(stage_targets)
        self._fault_threads: list[threading.Thread] = []

    def log_msg(self, msg: str) -> None:
        """Write a timestamped message to stdout."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)

    def _emit_event(self, event_type: str, **fields: Any) -> None:
        record: dict[str, Any] = {
            "event": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        for k, v in fields.items():
            record[k] = int(v) if isinstance(v, str) and v.isdigit() else v
        self._recorded_events.append(record)
        if self.events_file:
            append_event(self.events_file, event_type, **fields)
        else:
            self.log_msg("event " + json.dumps(record, ensure_ascii=True, separators=(",", ":")))

    # ------------------------------------------------------------------
    # Trigger construction
    # ------------------------------------------------------------------

    def _build_triggers(self, stage_targets: list[dict]) -> list[FaultTrigger]:
        """Build the list of FaultTrigger instances from config."""
        cfg = self.cfg
        triggers: list[FaultTrigger] = []

        kind = cfg.trigger_kind
        thresholds = cfg.thresholds
        if thresholds:
            recovery = float(cfg.recovery_delay) if str(cfg.recovery_delay).isdigit() else 0.0
            triggers.append(CompletionCountTrigger(
                kind=kind,
                thresholds=thresholds,
                stage_targets=stage_targets,
                inject_delay=float(cfg.inject_delay),
                recovery_delay=recovery,
                inject_match_task_label=getattr(cfg, "inject_match_task_label", ""),
                inject_match_agent_role=getattr(cfg, "inject_match_agent_role", ""),
                inject_match_worker_id=getattr(cfg, "inject_match_worker_id", ""),
            ))

        tl_raw = getattr(cfg, "timeline_after_start_s", "")
        if tl_raw:
            times = [float(x.strip()) for x in str(tl_raw).split(",") if x.strip()]
            if times:
                triggers.append(TimelineTrigger(
                    fire_at_s=times,
                    stage_targets=stage_targets,
                    inject_delay=float(cfg.inject_delay),
                ))

        if not triggers:
            raise ValueError(
                "No fault triggers configured. "
                "Set inject_after_job/after_task or timeline_after_start_s."
            )
        return triggers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _tail_new_events(self, last_line: int) -> int:
        if not self.events_file or not self.events_file.is_file():
            return last_line
        lines = self.events_file.read_text(encoding="utf-8").splitlines()
        if len(lines) <= last_line:
            return last_line
        for line in lines[last_line:]:
            if line.strip():
                try:
                    self.handle_event_record(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return len(lines)

    def monitor(self) -> None:
        """Monitor events.jsonl in a loop. Call this in the main thread."""
        if not self.events_file:
            raise ValueError("monitor() requires events_file; use handle_event_record() for control mode")
        last_line = 0
        self.log_msg(f"Monitoring {self.events_file}, triggers: {[t.name for t in self._triggers]}")

        while True:
            last_line = self._tail_new_events(last_line)
            time.sleep(1.0)

    def monitor_until(
        self,
        should_stop: Callable[[], bool],
        timeout_s: float = 86400.0,
        poll_s: float = 1.0,
    ) -> None:
        """Process events.jsonl until should_stop() is true, then drain and join fault threads."""
        start = time.time()
        if not self.events_file:
            while not should_stop() and time.time() - start < timeout_s:
                time.sleep(poll_s)
            for t in self._fault_threads:
                t.join(timeout=600)
            for trigger in self._triggers:
                trigger.close()
            return
        last_line = 0
        self.log_msg(f"Monitoring {self.events_file}, triggers: {[t.name for t in self._triggers]}")
        while time.time() - start < timeout_s:
            last_line = self._tail_new_events(last_line)
            if should_stop():
                break
            time.sleep(poll_s)
        last_line = self._tail_new_events(last_line)
        for t in self._fault_threads:
            t.join(timeout=600)
        for trigger in self._triggers:
            trigger.close()

    def wait_for_completion(self, poll_func, timeout_s: int = 86400) -> bool:
        """Poll poll_func() and process events until it returns True."""
        self.monitor_until(poll_func, timeout_s=float(timeout_s), poll_s=1.0)
        return True

    def snapshot_server_logs(self) -> None:
        """Copy server log files to output_dir."""
        if not self.failover_events_file:
            return
        log_dir = Path(self.failover_events_file).parent
        if not log_dir.is_dir():
            return
        snap = self.cfg.output_dir / "server_logs"
        try:
            from process_logs import get_process_log_path

            central_server_log = get_process_log_path(log_dir, "server")
            if central_server_log.is_file():
                dest = snap / "server.log"
                dest.parent.mkdir(parents=True, exist_ok=True)
                import shutil as _shutil

                _shutil.copy2(central_server_log, dest)
        except Exception:
            pass
        for path in sorted(log_dir.glob("node_*/server.log")):
            node_name = path.parent.name
            dest = snap / node_name / "server.log"
            dest.parent.mkdir(parents=True, exist_ok=True)
            import shutil as _shutil
            _shutil.copy2(path, dest)

    def build_metrics(self) -> None:
        """Run the metrics build script."""
        out_json = self.cfg.output_dir / "internal_failover_metrics.json"
        try:
            build_internal_failover_metrics(
                trace_log=self.trace_file,
                events_file=self.events_file,
                output=out_json,
                runtime_events_file=(
                    Path(self.failover_events_file)
                    if self.failover_events_file
                    else None
                ),
            )
        except Exception as exc:
            self.log_msg(f"Metrics build failed: {exc!r}")
            return
        if not out_json.is_file() or out_json.stat().st_size == 0:
            self.log_msg("Metrics build finished but output is missing or empty.")

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def handle_event_record(self, rec: dict) -> None:
        ev = str(rec.get("event", ""))
        with self._counts_cond:
            if ev == "job_completed":
                self._job_completed_count += 1
            elif ev == "task_completed":
                self._task_completed_count += 1
            self._counts_cond.notify_all()
        self._on_event(rec)

    def _on_event(self, rec: dict) -> None:
        ev = rec.get("event", "")
        ctx = TriggerContext(
            job_count=0, task_count=0, event=ev, rec=rec,
            elapsed_wall_s=time.time() - self._start_time,
        )
        self._dispatch_to_triggers(ctx)

    def _dispatch_to_triggers(self, ctx: TriggerContext) -> None:
        for trigger in self._triggers:
            result = trigger.check(ctx)
            if result:
                self._schedule_injection(result)

    def _schedule_injection(self, result: TriggerResult) -> None:
        stage = result.stage
        _explicit_keys = {"trigger_kind", "trigger_count", "trigger_time_s"}
        self._emit_event(
            "fault_scheduled",
            event_id=result.event_id,
            trigger_kind=result.fields.get("trigger_kind", ""),
            trigger_count=result.fields.get("trigger_count", ""),
            trigger_time_s=result.fields.get("trigger_time_s", ""),
            inject_delay=result.inject_delay,
            **stage,
            **{k: str(v) for k, v in result.fields.items()
               if v and k not in _explicit_keys},
        )
        self.log_msg(
            f"[fault #{result.event_id}] trigger={result.fields.get('trigger_kind', '')} "
            f"-> dp{stage['dp_rank']}/pp{stage['pp_rank']}/tp{stage['tp_rank']} "
            f"on {stage['node']} (delay={result.inject_delay}s)"
        )
        t = threading.Thread(
            target=self._do_inject,
            args=(result,),
            daemon=True,
        )
        t.start()
        self._fault_threads.append(t)

    # ------------------------------------------------------------------
    # Injection
    # ------------------------------------------------------------------

    def _do_inject(self, result: TriggerResult) -> None:
        cfg = self.cfg
        time.sleep(result.inject_delay)

        stage = result.stage
        dp_rank = int(stage["dp_rank"])
        self._emit_event(
            "fault_injected",
            event_id=result.event_id,
            trigger_kind=result.fields.get("trigger_kind", ""),
            trigger_count=result.fields.get("trigger_count", ""),
            trigger_time_s=result.fields.get("trigger_time_s", ""),
            **stage, method=cfg.method, slurm_job_id=cfg.slurm_job_id, kv_backup=cfg.kv_backup,
        )

        fault_start = time.time()
        ok = self._method.inject(cfg.server_url, dp_rank, stage, self.log_msg)
        if not ok:
            self._emit_event("fault_injection_failed",
                             event_id=result.event_id, **stage, method=cfg.method, status="inject_failed")
            return

        self.log_msg(f"[fault #{result.event_id}] injected dp={stage['dp_rank']} on {stage['node']}")

        # Wait for runtime reaction
        ev_path = Path(self.failover_events_file) if self.failover_events_file else None
        reaction_ok = self._wait_reaction(ev_path, stage["dp_rank"], fault_start)
        if reaction_ok:
            self._emit_event("runtime_failover_reaction",
                             event_id=result.event_id, **stage, status="runtime_reaction")
        else:
            self._emit_event("gpu_failure_injected",
                             event_id=result.event_id, **stage, status="injected_no_reaction")

        # Recovery: recover_after_task > recover_after_job > delay
        recover_after_task = (
            int(cfg.recover_after_task) if cfg.recover_after_task.isdigit() else 0
        )
        recover_after_job = int(cfg.recover_after_job) if cfg.recover_after_job.isdigit() else 0
        if recover_after_task > 0:
            recovery_task_target = self._resolve_recovery_task_target(
                recover_after_task
            )
            self.log_msg(
                f"[fault #{result.event_id}] waiting for task #{recovery_task_target} "
                f"to complete before recovery (configured={recover_after_task})"
            )
            self._wait_for_task_completion(recovery_task_target)
            recovered = self._recover(cfg.server_url, dp_rank)
            self._emit_event(
                "gpu_recovered" if recovered else "recovery_failed",
                event_id=result.event_id, **stage, method=cfg.method,
                status="recovered" if recovered else "failed",
                recover_trigger="after_task",
                recover_after_task=recover_after_task,
                recover_task_target=recovery_task_target,
            )
        elif recover_after_job > 0:
            self.log_msg(f"[fault #{result.event_id}] waiting for job #{recover_after_job} to complete before recovery")
            self._wait_for_job_completion(recover_after_job)
            recovered = self._recover(cfg.server_url, dp_rank)
            self._emit_event(
                "gpu_recovered" if recovered else "recovery_failed",
                event_id=result.event_id, **stage, method=cfg.method,
                status="recovered" if recovered else "failed",
                recover_trigger="after_job",
                recover_after_job=recover_after_job,
            )
        else:
            recovery_delay = result.recovery_delay
            if recovery_delay <= 0 and cfg.recovery_delay.isdigit():
                recovery_delay = float(cfg.recovery_delay)
            if recovery_delay > 0:
                self.log_msg(f"[fault #{result.event_id}] recovering in {recovery_delay}s")
                time.sleep(recovery_delay)
                recovered = self._recover(cfg.server_url, dp_rank)
                self._emit_event(
                    "gpu_recovered" if recovered else "recovery_failed",
                    event_id=result.event_id, **stage, method=cfg.method,
                    status="recovered" if recovered else "failed",
                )

    def _wait_for_job_completion(self, target_job_count: int) -> None:
        """Block until the Nth job_completed event appears in events.jsonl."""
        if not self.events_file:
            with self._counts_cond:
                while self._job_completed_count < target_job_count:
                    self._counts_cond.wait(timeout=2.0)
            return
        while True:
            count = 0
            if self.events_file and self.events_file.is_file():
                for line in self.events_file.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("event") == "job_completed":
                        count += 1
                        if count >= target_job_count:
                            return
            time.sleep(2.0)

    def _count_task_completions(self) -> int:
        if not self.events_file:
            with self._counts_cond:
                return self._task_completed_count

        count = 0
        if self.events_file.is_file():
            for line in self.events_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("event") == "task_completed":
                    count += 1
        return count

    def _resolve_recovery_task_target(self, configured_target: int) -> int:
        """Return a recovery threshold that cannot be satisfied before injection."""
        current_count = self._count_task_completions()
        if current_count >= configured_target:
            return current_count + 1
        return configured_target

    def _wait_for_task_completion(self, target_task_count: int) -> None:
        """Block until the Nth task_completed event appears in events.jsonl."""
        if not self.events_file:
            with self._counts_cond:
                while self._task_completed_count < target_task_count:
                    self._counts_cond.wait(timeout=2.0)
            return
        while True:
            count = 0
            if self.events_file and self.events_file.is_file():
                for line in self.events_file.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("event") == "task_completed":
                        count += 1
                        if count >= target_task_count:
                            return
            time.sleep(2.0)

    def _recover(self, server_url: str, dp_rank: int) -> bool:
        url = f"{server_url.rstrip('/')}/simulate_gpu_recovery"
        body = json.dumps({"dp_rank": dp_rank}).encode("utf-8")
        try:
            proxy_handler = urllib.request.ProxyHandler({})
            opener = urllib.request.build_opener(proxy_handler)
            req = urllib.request.Request(
                url, data=body, method="POST",
                headers={"Content-Type": "application/json"},
            )
            with opener.open(req, timeout=120):
                pass
            return True
        except (urllib.error.URLError, OSError):
            return False

    def _wait_reaction(
        self,
        events_file: Optional[Path],
        failed_dp_rank: str,
        not_before: float,
    ) -> bool:
        if not events_file:
            return False
        interesting = {
            "failover_dispatched", "failover_aborted",
            "resume_accepted", "resume_rejected",
        }
        deadline = time.time() + self.cfg.unhealthy_timeout

        def parse_ts(val: str) -> float:
            if not val:
                return 0.0
            return datetime.fromisoformat(val.replace("Z", "+00:00")).timestamp()

        while time.time() < deadline:
            if not events_file.exists():
                time.sleep(1.0)
                continue
            for line in events_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("event") not in interesting:
                    continue
                event_ts = parse_ts(rec.get("timestamp", ""))
                owner_dp_rank = str(rec.get("failed_owner_dp_rank", ""))
                dp_rank = str(rec.get("dp_rank", ""))
                owner_or_dp_matched = (
                    owner_dp_rank == failed_dp_rank or dp_rank == failed_dp_rank
                )
                if not owner_or_dp_matched:
                    continue
                if event_ts < not_before:
                    continue
                return True
            time.sleep(1.0)
        return False
