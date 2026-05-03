# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.

"""Child process: tail events.jsonl and run fault injection until signaled."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Fault injector worker process")
    parser.add_argument("--config-json-file", type=Path, default=None)
    parser.add_argument("--config-json", type=str, default="")
    parser.add_argument("--server-url", type=str, default="")
    parser.add_argument("--stage-manifest", type=Path, default=None)
    parser.add_argument("--stage-manifest-json", type=str, default="")
    parser.add_argument("--slurm-job-id", type=str, default="")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--inject-after-job", type=str, default="1")
    parser.add_argument("--inject-after-task", type=str, default="")
    parser.add_argument("--timeline-after-start-s", type=str, default="")
    parser.add_argument("--inject-delay", type=int, default=10)
    parser.add_argument("--fault-dp-rank", type=str, default="")
    parser.add_argument("--fault-pp-rank", type=str, default="0")
    parser.add_argument("--fault-tp-rank", type=str, default="0")
    parser.add_argument("--unhealthy-timeout", type=int, default=20)
    parser.add_argument("--method", type=str, default="api")
    parser.add_argument("--recovery-delay", type=str, default="")
    parser.add_argument("--recover-after-job", type=str, default="")
    parser.add_argument("--recover-after-task", type=str, default="")
    parser.add_argument("--kv-backup", type=str, default="none")
    parser.add_argument("--events-file", type=str, default="")
    parser.add_argument("--trace-file", type=str, default="")
    parser.add_argument("--failover-events-file", type=str, default="")
    parser.add_argument("--server-topology", type=str, default="")
    parser.add_argument("--inject-match-task-label", type=str, default="")
    parser.add_argument("--inject-match-agent-role", type=str, default="")
    parser.add_argument("--inject-match-worker-id", type=str, default="")
    parser.add_argument("--control-host", type=str, default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=-1)
    args = parser.parse_args()
    stage_manifest_data = json.loads(args.stage_manifest_json) if args.stage_manifest_json else {}
    if args.server_url and (args.stage_manifest is not None or stage_manifest_data) and args.output_dir is not None:
        payload = {
            "server_url": args.server_url,
            "stage_manifest": str(args.stage_manifest) if args.stage_manifest else "",
            "stage_manifest_data": stage_manifest_data,
            "slurm_job_id": args.slurm_job_id,
            "output_dir": str(args.output_dir),
            "inject_after_job": args.inject_after_job,
            "inject_after_task": args.inject_after_task,
            "timeline_after_start_s": args.timeline_after_start_s,
            "inject_delay": int(args.inject_delay),
            "fault_dp_rank": args.fault_dp_rank,
            "fault_pp_rank": args.fault_pp_rank,
            "fault_tp_rank": args.fault_tp_rank,
            "unhealthy_timeout": int(args.unhealthy_timeout),
            "method": args.method,
            "recovery_delay": args.recovery_delay,
            "recover_after_job": args.recover_after_job,
            "recover_after_task": args.recover_after_task,
            "kv_backup": args.kv_backup,
            "events_file": args.events_file,
            "trace_file": args.trace_file,
            "failover_events_file": args.failover_events_file,
            "server_topology": args.server_topology,
            "inject_match_task_label": args.inject_match_task_label,
            "inject_match_agent_role": args.inject_match_agent_role,
            "inject_match_worker_id": args.inject_match_worker_id,
        }
    elif args.config_json:
        payload = json.loads(args.config_json)
    elif args.config_json_file is not None:
        payload = json.loads(args.config_json_file.read_text(encoding="utf-8"))
    else:
        parser.error("injector flags or one of --config-json/--config-json-file is required")
    from failure.config import fault_injector_config_from_json
    from failure.injector_core import FaultInjector  # pyright: ignore[reportMissingImports]

    stop_requested = False

    def _request_stop(_sig: int, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    cfg = fault_injector_config_from_json(payload)
    injector = FaultInjector(cfg)
    if args.control_port >= 0:
        from process_logs import encode_control_event

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *handler_args) -> None:
                return

            def do_POST(self) -> None:
                if self.path != "/event":
                    self.send_response(404)
                    self.end_headers()
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload_bytes = self.rfile.read(length)
                    record = json.loads(payload_bytes.decode("utf-8"))
                    if not isinstance(record, dict):
                        raise ValueError("event payload must be an object")
                    injector.handle_event_record(record)
                except Exception as exc:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(str(exc).encode("utf-8", errors="replace"))
                    return
                self.send_response(204)
                self.end_headers()

        httpd = ThreadingHTTPServer((args.control_host, args.control_port), Handler)
        host, port = httpd.server_address
        print(encode_control_event("injector_ready", control_url=f"http://{host}:{port}"), flush=True)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            while not stop_requested:
                signal.pause()
        except AttributeError:
            while not stop_requested:
                threading.Event().wait(0.5)
        finally:
            httpd.shutdown()
            thread.join(timeout=10)
        return 0
    injector.monitor_until(lambda: stop_requested)
    return 0


if __name__ == "__main__":
    sys.exit(main())
