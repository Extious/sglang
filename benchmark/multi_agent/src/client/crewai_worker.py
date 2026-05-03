# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.

"""Child process: run CrewAI jobs (thread-parallel) and write summaries."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from client.config import CrewAIRunnerConfig, runner_config_from_json


def main() -> int:
    parser = argparse.ArgumentParser(description="CrewAI workload worker process")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--config-json", type=str, default="")
    parser.add_argument("--server-url", type=str, default="")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--jobs-csv", type=Path, default=None)
    parser.add_argument("--job-limit", type=int, default=6)
    parser.add_argument("--app-workers", type=int, default=2)
    parser.add_argument("--default-year", type=str, default="2025")
    parser.add_argument("--short-max-tokens", type=int, default=1024)
    parser.add_argument("--long-max-tokens", type=int, default=2048)
    parser.add_argument("--enable-stream", type=int, default=0)
    parser.add_argument("--ignore-eos", type=int, default=1)
    parser.add_argument("--worker-start-stagger-s", type=float, default=0.0)
    parser.add_argument("--agent-dp-rank-map", type=str, default="{}")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--trace-file", type=Path, default=None)
    parser.add_argument("--events-file", type=Path, default=None)
    parser.add_argument("--control-url", type=str, default="")
    parser.add_argument("--extra-instructions-file", type=Path, default=None)
    args = parser.parse_args()

    if args.server_url and args.jobs_csv is not None and args.output_dir is not None:
        cfg = CrewAIRunnerConfig(
            server_url=args.server_url,
            model_path=args.model_path,
            jobs_csv=args.jobs_csv,
            job_limit=int(args.job_limit),
            app_workers=int(args.app_workers),
            default_year=args.default_year,
            short_max_tokens=int(args.short_max_tokens),
            long_max_tokens=int(args.long_max_tokens),
            enable_stream=bool(int(args.enable_stream)),
            ignore_eos=int(args.ignore_eos),
            worker_start_stagger_s=float(args.worker_start_stagger_s),
            agent_dp_rank_map=dict(json.loads(args.agent_dp_rank_map or "{}")),
            extra_instructions_path=args.extra_instructions_file,
            output_dir=args.output_dir,
            trace_file=args.trace_file,
            events_file=args.events_file,
            control_url=args.control_url,
        )
    elif args.config_json:
        cfg = runner_config_from_json(json.loads(args.config_json))
    elif args.config is not None:
        cfg = runner_config_from_json(json.loads(args.config.read_text(encoding="utf-8")))
    else:
        parser.error("one of --config-json or --config is required")

    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / os.environ.get("USER", "user") / "crewai-storage"
    db_name = f"job_{cfg.output_dir.name}"
    os.environ["XDG_DATA_HOME"] = str(tmp / "xdg")
    os.environ["SQLITE_TMPDIR"] = str(tmp / "sqlite-tmp")
    os.environ["CREWAI_STORAGE_DIR"] = db_name
    (tmp / "xdg").mkdir(parents=True, exist_ok=True)
    (tmp / "sqlite-tmp").mkdir(parents=True, exist_ok=True)

    from client.crewai_core import CrewAIRunner
    from client.metrics_utils import (
        write_job_summary_records,
        write_cache_hit_records,
        write_task_summary_records,
    )

    # run all jobs
    runner = CrewAIRunner(cfg)
    results = runner.run_all_jobs()

    # write summaries
    out = cfg.output_dir
    records = [m.to_trace_record() for m in results]
    cache_rows = runner.cache_hit_rows() if hasattr(runner, "cache_hit_rows") else []
    write_cache_hit_records(cache_rows, out / "cache_hits.csv")
    write_job_summary_records(records, out / "job_summary.csv")
    write_task_summary_records(records, out / "task_summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
