"""Thin orchestration pipeline: server + client + fault injection in one run."""

from __future__ import annotations

import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from client.config import CrewAIClientConfig, CrewAIRunnerConfig
from failure.config import FaultInjectionConfig, FaultInjectorConfig
from process_logs import (
    copy_process_logs,
    decode_control_event,
    get_process_log_path,
    process_log_env,
    redirect_current_process_to_log,
)
from server.config import ServerConfig
from utils import build_experiment_config  # pyright: ignore[reportMissingImports]


def _repo_root() -> Path:
    env = os.environ.get("SGLANG_REPO_ROOT", "").strip()
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[3]


class ExperimentPipeline:
    """
    One-shot experiment: deploy server, run workload with fault injection, teardown.
    """

    def __init__(
        self,
        server_cfg: ServerConfig,
        fi_cfg: FaultInjectionConfig,
        crewai_cfg: CrewAIClientConfig,
        kv_backup: str,
        log_dir: Path,
        output_dir: Path,
        timestamp: str,
        venv: Path,
        deploy_config_name: str,
    ) -> None:
        self.server_cfg = server_cfg
        self.fi_cfg = fi_cfg
        self.crewai_cfg = crewai_cfg
        self.kv_backup = kv_backup
        self.log_dir = log_dir
        run_dir_name = Path(deploy_config_name).name if deploy_config_name else timestamp
        self.output_dir = output_dir / run_dir_name
        self.timestamp = timestamp
        self._venv = venv
        self._deploy_config_name = deploy_config_name

        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)

        self._log_path = get_process_log_path(log_dir, "main")
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._src_root = Path(__file__).resolve().parent
        self._server_proc: subprocess.Popen | None = None
        self._client_proc: subprocess.Popen | None = None
        self._injector_proc: subprocess.Popen | None = None
        self._logs_copied = False

        def _sig_handler(_sig: int, _frame) -> None:
            print("\nInterrupted -- cleaning up...", flush=True)
            for proc in (self._client_proc, self._injector_proc, self._server_proc):
                if proc and proc.poll() is None:
                    proc.terminate()
            raise SystemExit(130)

        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)

    def log(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)

    def _start_log_reader(
        self,
        proc: subprocess.Popen,
        log_path: Path,
        events: "queue.Queue[dict] | None" = None,
    ) -> threading.Thread:
        def _reader() -> None:
            assert proc.stdout is not None
            with log_path.open("ab") as fh:
                for raw_line in proc.stdout:
                    data = raw_line if isinstance(raw_line, bytes) else raw_line.encode("utf-8", errors="replace")
                    fh.write(data)
                    fh.flush()
                    line = data.decode("utf-8", errors="replace").rstrip("\n")
                    record = decode_control_event(line)
                    if record is not None and events is not None:
                        events.put(record)

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()
        return thread

    def _wait_for_event(
        self,
        proc: subprocess.Popen,
        events: "queue.Queue[dict]",
        event_name: str,
        timeout_s: int,
    ) -> dict | None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if proc.poll() is not None:
                self.log(f"ERROR: subprocess exited early while waiting for {event_name} (rc={proc.returncode})")
                return None
            try:
                record = events.get(timeout=1.0)
            except queue.Empty:
                continue
            if record.get("event") == event_name:
                return record
        self.log(f"ERROR: timed out waiting for {event_name} after {timeout_s}s")
        return None

    def run(self) -> int:
        """All-in-one mode: server, client, injector as child processes."""
        # server config
        srv_cfg = self.server_cfg
        # fault injection config
        fi_cfg = self.fi_cfg
        # crewai config
        crewai_cfg = self.crewai_cfg
        # output directory
        out = self.output_dir
        # log directory
        log = self.log
        log_dir = self.log_dir
        # source root
        src_root = self._src_root
        # repository root
        repo = _repo_root()
        # python binary (from env.yml PathConfig.venv, optional SGLANG_VENV)
        py = str(self._venv / "bin" / "python")
        # deploy script
        deploy_py = str(repo / "benchmark" / "multi_agent" / "src" / "server" / "deploy_server.py")

        log(f"{'=' * 60}")
        log(f"EXPERIMENT  topology={srv_cfg.topology_tag()}  kv_backup={self.kv_backup}")
        log(f"  result_dir: {out}")
        log(f"  log_dir:    {log_dir}")

        client_rc = 1

        try:
            # ------------------------------------------------------------------
            # Step 1: start server as subprocess
            # ------------------------------------------------------------------
            log("Step 1: Start SGLang server subprocess...")
            server_flags = srv_cfg.build_deploy_flags(
                model_path=srv_cfg.model_path,
                server_port=28000,
                kv_backup=self.kv_backup,
                venv_dir=str(self._venv),
            )
            # build server command (optional positional config loads config/<name>/server.json)
            server_cmd = [py, deploy_py, "--log-dir", str(log_dir)]
            if self._deploy_config_name:
                server_cmd.append(self._deploy_config_name)
            server_cmd.extend(server_flags)
            log_env = os.environ.copy()
            log_env.update(process_log_env(log_dir))
            server_log = get_process_log_path(log_dir, "server")
            server_log.write_bytes(b"")
            server_events: queue.Queue[dict] = queue.Queue()
            # start server as subprocess
            self._server_proc = subprocess.Popen(
                server_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=log_env,
                text=True,
                bufsize=1,
            )
            self._start_log_reader(self._server_proc, server_log, server_events)
            log(f"  Server PID: {self._server_proc.pid}")
            server_ready = self._wait_for_event(
                self._server_proc, server_events, "server_ready", timeout_s=1200,
            )
            if server_ready is None:
                return 1

            manifest = server_ready.get("manifest") or {}
            if not isinstance(manifest, dict) or not manifest.get("server_url"):
                log("ERROR: malformed server_ready control event")
                return 1
            head_url = str(manifest["server_url"])
            log(f"  Server ready at {head_url}")

            # ------------------------------------------------------------------
            # Step 2: start injector + client as subprocesses
            # ------------------------------------------------------------------
            log("Step 2: Start injector and client subprocesses...")
            # build injector config
            injector_cfg = FaultInjectorConfig(
                server_url=head_url,
                stage_manifest=None,
                slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
                kv_backup=self.kv_backup,
                output_dir=out,
                inject_after_job=fi_cfg.after_job,
                inject_after_task=fi_cfg.after_task,
                timeline_after_start_s=getattr(fi_cfg, "timeline_after_start_s", ""),
                inject_delay=fi_cfg.delay,
                fault_dp_rank=fi_cfg.dp_rank,
                fault_pp_rank=fi_cfg.pp_rank,
                fault_tp_rank=fi_cfg.tp_rank,
                method=fi_cfg.method,
                recovery_delay=fi_cfg.recovery_delay,
                recover_after_job=fi_cfg.recover_after_job,
                recover_after_task=getattr(fi_cfg, "recover_after_task", ""),
                server_topology=srv_cfg.topology_tag(),
                inject_match_task_label=getattr(fi_cfg, "inject_match_task_label", ""),
                inject_match_agent_role=getattr(fi_cfg, "inject_match_agent_role", ""),
                inject_match_worker_id=getattr(fi_cfg, "inject_match_worker_id", ""),
                stage_manifest_data=manifest,
            )
            injector_flags = injector_cfg.build_worker_flags()

            # Launch injector subprocess
            inj_log = get_process_log_path(log_dir, "injector")
            inj_log.write_bytes(b"")
            injector_events: queue.Queue[dict] = queue.Queue()
            self._injector_proc = subprocess.Popen(
                [py, "-m", "failure.injector_worker"] + injector_flags
                + ["--control-host", "127.0.0.1", "--control-port", "0"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(src_root),
                env=log_env,
                text=True,
                bufsize=1,
            )
            self._start_log_reader(self._injector_proc, inj_log, injector_events)
            log(f"  Injector PID: {self._injector_proc.pid}")
            injector_ready = self._wait_for_event(
                self._injector_proc, injector_events, "injector_ready", timeout_s=60,
            )
            if injector_ready is None:
                return 1
            control_url = str(injector_ready.get("control_url", ""))
            if not control_url:
                log("ERROR: malformed injector_ready control event")
                return 1

            worker_cfg = CrewAIRunnerConfig(
                server_url=head_url,
                model_path=srv_cfg.model_path,
                jobs_csv=Path(crewai_cfg.jobs_csv).resolve(),
                job_limit=int(crewai_cfg.job_limit),
                app_workers=int(crewai_cfg.app_workers),
                default_year=crewai_cfg.default_year,
                short_max_tokens=int(crewai_cfg.short_max_tokens),
                long_max_tokens=int(crewai_cfg.long_max_tokens),
                enable_stream=bool(crewai_cfg.enable_stream),
                ignore_eos=int(crewai_cfg.ignore_eos),
                worker_start_stagger_s=float(crewai_cfg.worker_start_stagger_s),
                agent_dp_rank_map=dict(crewai_cfg.agent_dp_rank_map or {}),
                extra_instructions_path=crewai_cfg.extra_instructions_path,
                output_dir=out.resolve(),
                control_url=control_url,
            )
            client_flags = worker_cfg.build_worker_flags()

            # Launch client subprocess
            cli_log = get_process_log_path(log_dir, "client")
            cli_log.write_bytes(b"")
            self._client_proc = subprocess.Popen(
                [py, "-m", "client.crewai_worker"] + client_flags,
                stdout=cli_log.open("ab"),
                stderr=subprocess.STDOUT,
                cwd=str(src_root),
                env=log_env,
            )
            log(f"  Client PID: {self._client_proc.pid}")

            # Wait for client to finish
            client_rc = self._client_proc.wait()
            log(f"  Client exited (rc={client_rc})")
            if self._injector_proc and self._injector_proc.poll() is None:
                self._injector_proc.terminate()

            # Wait for injector to finish (with timeout)
            inj_timeout = 60
            try:
                self._injector_proc.wait(timeout=inj_timeout)
            except subprocess.TimeoutExpired:
                log("WARNING: injector timed out, terminating.")
                self._injector_proc.terminate()
                try:
                    self._injector_proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._injector_proc.kill()

            # ------------------------------------------------------------------
            # Step 3: generate plots
            # ------------------------------------------------------------------
            log("Step 3: Generate plots...")
            self._generate_plots(out)

            log(f"Experiment complete. Results in {out}")
        finally:
            for proc in (self._client_proc, self._injector_proc):
                if proc and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
            self._client_proc = None
            self._injector_proc = None
            if self._server_proc and self._server_proc.poll() is None:
                self._server_proc.terminate()
                try:
                    self._server_proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self._server_proc.kill()
                    self._server_proc.wait()
            self._server_proc = None
            self._copy_logs_to_result()

        return client_rc

    def _copy_logs_to_result(self) -> None:
        if self._logs_copied:
            return
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
        copy_process_logs(self.log_dir, self.output_dir / "logs")
        self._logs_copied = True

    def _generate_plots(self, out: Path) -> None:
        job_summary = out / "job_summary.csv"
        task_summary = out / "task_summary.csv"
        if job_summary.is_file() and task_summary.is_file():
            self._run_plot(
                "trace_log_profile.py",
                out / "figures" / "trace_log_profile.png",
                "--job-summary", str(job_summary),
                "--task-summary", str(task_summary),
            )

        cache_csv = out / "cache_hits.csv"
        if cache_csv.is_file() and cache_csv.stat().st_size > 0:
            self._run_plot(
                "backup_cache_hits.py",
                out / "figures" / "backup_cache_hit_profile.png",
                "--cache-csv", str(cache_csv),
            )

    def _run_plot(self, script_name: str, output: Path, *extra_args: str) -> None:
        script = self._src_root / "plot" / script_name
        if not script.is_file():
            return
        cmd = [sys.executable, str(script), "--output", str(output), *extra_args]
        log_fh = self._log_path.open("a")
        subprocess.run(cmd, check=False, cwd=str(self._src_root), stdout=log_fh, stderr=subprocess.STDOUT)
        log_fh.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    try:
        cfg = build_experiment_config(sys.argv[1:])
    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    log_dir = cfg.log_dir
    output_dir = cfg.output_dir

    log_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    main_log_handle = redirect_current_process_to_log(
        get_process_log_path(log_dir, "main")
    )

    try:
        pipeline = ExperimentPipeline(
            server_cfg=cfg.server,
            fi_cfg=cfg.fault_injection,
            crewai_cfg=cfg.crewai,
            kv_backup=cfg.kv_backup,
            log_dir=log_dir,
            output_dir=output_dir,
            timestamp=time.strftime("%Y%m%d_%H%M%S"),
            venv=cfg.venv,
            deploy_config_name=cfg.deploy_config_name,
        )
        return pipeline.run()
    finally:
        main_log_handle.close()


if __name__ == "__main__":
    sys.exit(main())
