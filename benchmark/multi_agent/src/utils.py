from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import NamedTuple, Optional

from client.config import CrewAIClientConfig, FixedClientConfig
from failure.config import FaultInjectionConfig
from paths import PathConfig  # pyright: ignore[reportMissingImports]
from server.config import ServerConfig

class ExperimentConfig(NamedTuple):
    server: ServerConfig
    fault_injection: FaultInjectionConfig
    crewai: CrewAIClientConfig
    fixed: FixedClientConfig
    client_mode: str
    kv_backup: str
    run_server_slurm: Path
    log_dir: Path
    output_dir: Path
    venv: Path
    deploy_config_name: str


def _default_jobs_csv() -> Path:
    return Path(__file__).resolve().parent / "client" / "topics.csv"


def _resolve_venv(paths: PathConfig) -> Path:
    override = os.environ.get("SGLANG_VENV", "").strip()
    if override:
        return Path(override).resolve()
    return paths.venv


def build_experiment_config(
    argv: Optional[list[str]] = None,
    paths: Optional[PathConfig] = None,
) -> ExperimentConfig:
    if argv is None:
        argv = sys.argv[1:]

    p = argparse.ArgumentParser(description="CrewAI fault-injection experiment")
    p.add_argument("--config", required=True, help="Config directory name or absolute path")
    p.add_argument("--dp-size", type=int, default=0)
    p.add_argument("--pp-size", type=int, default=0)
    p.add_argument("--tp-size", type=int, default=0)
    p.add_argument("--nnodes", type=int, default=0)
    p.add_argument("--hicache-size", type=int, default=0)
    p.add_argument("--quantization", default="")
    p.add_argument(
        "--load-balance-method",
        default="",
        help="SGLang DP scheduling, e.g. total_tokens (empty = use server.json / default)",
    )
    p.add_argument(
        "--kv-backup",
        default="none",
        choices=["none", "host", "host_backup", "device", "remote_backup"],
    )
    p.add_argument("--job-limit", type=int, default=0)
    p.add_argument("--app-workers", type=int, default=0)
    p.add_argument("--default-year", default="")
    p.add_argument("--short-max-tokens", type=int, default=0)
    p.add_argument("--long-max-tokens", type=int, default=0)
    p.add_argument("--ignore-eos", type=int, default=-1)
    p.add_argument("--worker-start-stagger-s", type=float, default=-1.0)
    p.add_argument("--client-mode", default="", choices=["crewai", "fixed"])
    p.add_argument("--fixed-input-len", type=int, default=0)
    p.add_argument("--fixed-output-len", type=int, default=0)
    p.add_argument("--fixed-num-requests", type=int, default=0)
    p.add_argument("--fixed-app-workers", type=int, default=0)
    p.add_argument("--fixed-seed", type=int, default=-1)
    p.add_argument("--inject-after-job", default="")
    p.add_argument("--inject-after-task", default="")
    p.add_argument("--timeline-after-start-s", default="")
    p.add_argument("--inject-delay", type=int, default=-1)
    p.add_argument("--fault-dp-rank", default="")
    p.add_argument("--fault-pp-rank", default="")
    p.add_argument("--fault-tp-rank", default="")
    p.add_argument("--fault-injection-method", default="")
    p.add_argument("--recovery-delay", default="")
    p.add_argument("--recover-after-job", default="")
    p.add_argument("--recover-after-task", default="")
    p.add_argument("--log-dir", default="")
    p.add_argument("--output-dir", default="")
    p.add_argument("--deploy-script", default="")
    args = p.parse_args(argv)

    # path config from env.yml
    if paths is None:
        paths = PathConfig.from_env_yml(
            deploy_script=args.deploy_script,
            log_dir=args.log_dir,
            output_dir=args.output_dir,
        )

    # config directory
    cfg_dir = Path(args.config)
    if not cfg_dir.is_absolute():
        cfg_dir = paths.config_dir / cfg_dir.name

    # load json config
    def load_json(name: str) -> dict:
        path = cfg_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    # load server config
    top = load_json("server.json")
    srv_json = dict(top.get("server", {}))
    srv_json["hicache_storage"] = top.get("hicache_storage", {})
    # load fault injection config
    fi_json = load_json("failure.json")
    # load client config
    client_json = load_json("client.json")

    # build server config
    srv = ServerConfig.from_dict(srv_json)
    if args.dp_size > 0:
        srv.dp_size = args.dp_size
    if args.pp_size > 0:
        srv.pp_size = args.pp_size
    if args.tp_size > 0:
        srv.tp_size = args.tp_size
    if args.nnodes > 0:
        srv.nnodes = args.nnodes
    if args.hicache_size > 0:
        srv.hicache_size_gb = args.hicache_size
    if args.quantization:
        srv.quantization = args.quantization
    if args.load_balance_method:
        srv.load_balance_method = args.load_balance_method.strip()

    # build fault injection config
    fi = FaultInjectionConfig.from_dict(fi_json)
    if args.inject_after_job:
        fi.after_job = args.inject_after_job
    if args.inject_after_task:
        fi.after_task = args.inject_after_task
    if args.timeline_after_start_s:
        fi.timeline_after_start_s = args.timeline_after_start_s
    if args.inject_delay >= 0:
        fi.delay = args.inject_delay
    if args.fault_dp_rank:
        fi.dp_rank = args.fault_dp_rank
    if args.fault_pp_rank:
        fi.pp_rank = args.fault_pp_rank
    if args.fault_tp_rank:
        fi.tp_rank = args.fault_tp_rank
    if args.fault_injection_method:
        fi.method = args.fault_injection_method
    if args.recovery_delay:
        fi.recovery_delay = args.recovery_delay
    if args.recover_after_job:
        fi.recover_after_job = args.recover_after_job
    if args.recover_after_task:
        fi.recover_after_task = args.recover_after_task

    # build crewai config
    crewai = CrewAIClientConfig.from_dict(client_json, default_csv=_default_jobs_csv())
    if crewai.extra_instructions_path and not crewai.extra_instructions_path.is_absolute():
        crewai.extra_instructions_path = (cfg_dir / crewai.extra_instructions_path).resolve()
    if args.job_limit > 0:
        crewai.job_limit = args.job_limit
    if args.app_workers > 0:
        crewai.app_workers = args.app_workers
    if args.default_year:
        crewai.default_year = args.default_year
    if args.short_max_tokens > 0:
        crewai.short_max_tokens = args.short_max_tokens
    if args.long_max_tokens > 0:
        crewai.long_max_tokens = args.long_max_tokens
    if args.ignore_eos >= 0:
        crewai.ignore_eos = args.ignore_eos
    if args.worker_start_stagger_s >= 0:
        crewai.worker_start_stagger_s = args.worker_start_stagger_s
    if args.client_mode:
        crewai.client_mode = args.client_mode
    if crewai.client_mode not in {"crewai", "fixed"}:
        crewai.client_mode = "crewai"

    fixed = FixedClientConfig.from_dict(client_json)
    if args.fixed_input_len > 0:
        fixed.input_len = args.fixed_input_len
    if args.fixed_output_len > 0:
        fixed.output_len = args.fixed_output_len
    if args.fixed_num_requests > 0:
        fixed.num_requests = args.fixed_num_requests
    if args.fixed_app_workers > 0:
        fixed.app_workers = args.fixed_app_workers
    if args.fixed_seed >= 0:
        fixed.seed = args.fixed_seed

    # build kv backup
    kv_backup = top.get("kv_backup", args.kv_backup)
    if isinstance(kv_backup, str) and kv_backup not in {
        "none",
        "host",
        "host_backup",
        "device",
        "remote_backup",
    }:
        kv_backup = args.kv_backup

    # build experiment config
    return ExperimentConfig(
        server=srv,
        fault_injection=fi,
        crewai=crewai,
        fixed=fixed,
        client_mode=crewai.client_mode,
        kv_backup=str(kv_backup),
        run_server_slurm=paths.deploy_script,
        log_dir=paths.log_dir,
        output_dir=paths.output_dir,
        venv=_resolve_venv(paths),
        deploy_config_name=str(args.config),
    )
