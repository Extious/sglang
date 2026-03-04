#!/usr/bin/env python3
"""
Run heavy_swarm.py with router failure injection enabled.

Default behavior:
- total tasks: 60
- inject failure after completed tasks: 30
- blocked worker: http://hkbugpusrv09:8001
- recover after: 30 seconds
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run heavy_swarm with timed router failure injection"
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to run heavy_swarm.py",
    )
    parser.add_argument(
        "--task-limit",
        type=int,
        default=60,
        help="HEAVY_SWARM_TASK_LIMIT",
    )
    parser.add_argument(
        "--task-processes",
        type=int,
        default=6,
        help="HEAVY_SWARM_TASK_PROCESSES",
    )
    parser.add_argument(
        "--inject-after",
        type=int,
        default=30,
        help="Inject failure after this many completed tasks",
    )
    parser.add_argument(
        "--recover-after",
        type=float,
        default=30.0,
        help="Seconds before recovering the blocked worker",
    )
    parser.add_argument(
        "--failed-worker-url",
        default="http://hkbugpusrv09:8001",
        help="Worker URL to mark down at injection point",
    )
    parser.add_argument(
        "--router-control-base-url",
        default="",
        help=(
            "Router control base URL, e.g. http://router-host:30000. "
            "If empty, heavy_swarm derives it from LLM_BASE_URL."
        ),
    )
    parser.add_argument(
        "--llm-base-url",
        default="",
        help=(
            "Optional override for LLM_BASE_URL, e.g. "
            "http://router-host:30000/v1"
        ),
    )
    parser.add_argument(
        "--results-path",
        default="",
        help="Optional HEAVY_SWARM_RESULTS_PATH output JSON path",
    )
    parser.add_argument(
        "--enable-timing-reports",
        choices=["0", "1"],
        default="1",
        help="HEAVY_SWARM_ENABLE_TIMING_REPORTS",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    heavy_swarm_file = script_dir / "heavy_swarm.py"
    if not heavy_swarm_file.exists():
        print(f"ERROR: heavy_swarm.py not found: {heavy_swarm_file}", file=sys.stderr)
        return 2

    env = os.environ.copy()
    env["HEAVY_SWARM_TASK_LIMIT"] = str(max(1, args.task_limit))
    env["HEAVY_SWARM_TASK_PROCESSES"] = str(max(1, args.task_processes))
    env["HEAVY_SWARM_ENABLE_TIMING_REPORTS"] = args.enable_timing_reports

    env["HEAVY_SWARM_ENABLE_ROUTER_FAILURE_INJECTION"] = "1"
    env["HEAVY_SWARM_FAILURE_INJECT_AFTER_TASKS"] = str(max(1, args.inject_after))
    env["HEAVY_SWARM_FAILURE_WORKER_URL"] = args.failed_worker_url.strip()
    env["HEAVY_SWARM_FAILURE_RECOVER_AFTER_S"] = str(max(0.0, args.recover_after))

    if args.router_control_base_url.strip():
        env["HEAVY_SWARM_ROUTER_CONTROL_BASE_URL"] = args.router_control_base_url.strip()
    if args.llm_base_url.strip():
        env["LLM_BASE_URL"] = args.llm_base_url.strip()
    if args.results_path.strip():
        env["HEAVY_SWARM_RESULTS_PATH"] = args.results_path.strip()

    cmd = [args.python_bin, str(heavy_swarm_file)]

    print("[run_heavy_swarm_fault_injection] config:")
    print(f"  python_bin={args.python_bin}")
    print(f"  task_limit={env['HEAVY_SWARM_TASK_LIMIT']}")
    print(f"  task_processes={env['HEAVY_SWARM_TASK_PROCESSES']}")
    print(f"  inject_after={env['HEAVY_SWARM_FAILURE_INJECT_AFTER_TASKS']}")
    print(f"  recover_after={env['HEAVY_SWARM_FAILURE_RECOVER_AFTER_S']}")
    print(f"  failed_worker={env['HEAVY_SWARM_FAILURE_WORKER_URL']}")
    print(
        "  router_control_base_url="
        f"{env.get('HEAVY_SWARM_ROUTER_CONTROL_BASE_URL', '(auto-from-LLM_BASE_URL)')}"
    )
    print(f"  llm_base_url={env.get('LLM_BASE_URL', '(inherit-env)')}")
    print("[run_heavy_swarm_fault_injection] starting heavy_swarm.py ...")

    proc = subprocess.run(
        cmd,
        env=env,
        cwd=str(script_dir),
    )
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
