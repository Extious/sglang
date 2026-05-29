from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark.sglsim.synthesis_requests.client import (  # noqa: E402
    build_prepared_requests,
    run_client,
    wait_for_server_ready,
)
from benchmark.sglsim.synthesis_requests.server import (  # noqa: E402
    build_server_command,
    start_server,
    stop_server,
)
from benchmark.sglsim.synthesis_requests.utils import (  # noqa: E402
    CLIENT_LOG_PATH,
    DEFAULT_CONFIG_DIR,
    DEFAULT_RESULTS_ROOT,
    SERVER_LOG_PATH,
    aggregate_metrics,
    load_config_dir,
    new_run_dir,
    setup_file_logger,
    write_metrics_json,
    write_request_details_csv,
    write_run_config,
)

_SERVER_READY_TIMEOUT_S = 1800.0


def run_experiment(
    config_dir: Path,
    results_root: Path,
    *,
    log_level: str = "info",
    skip_server: bool = False,
) -> Path:
    client_cfg, server_cfg = load_config_dir(config_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = new_run_dir(results_root)

    runner_log = setup_file_logger(
        "sglang.benchmark.sglsim.synthesis_requests.runner",
        CLIENT_LOG_PATH,
        log_level,
    )
    setup_file_logger(
        "sglang.benchmark.sglsim.synthesis_requests.client",
        CLIENT_LOG_PATH,
        log_level,
    )
    setup_file_logger(
        "sglang.benchmark.sglsim.synthesis_requests.server",
        SERVER_LOG_PATH,
        log_level,
    )

    server_log_path = run_dir / "server.log"
    write_run_config(
        run_dir,
        client=client_cfg,
        server=server_cfg,
        timestamp=timestamp,
    )
    runner_log.info(
        "Experiment config_dir=%s run_dir=%s skip_server=%s",
        config_dir,
        run_dir,
        skip_server,
    )

    prepared = build_prepared_requests(client_cfg, server_cfg)
    runner_log.info("Prepared %d requests", len(prepared))

    process = None
    if not skip_server:
        process = start_server(
            build_server_command(server=server_cfg, log_level=log_level),
            server_log_path,
        )
    try:
        wait_for_server_ready(
            server_cfg.host,
            server_cfg.port,
            timeout_s=_SERVER_READY_TIMEOUT_S,
            process=process,
            log_path=server_log_path,
        )
        workload_start = time.monotonic()
        rows = run_client(
            server_cfg.host,
            server_cfg.port,
            prepared,
            client_cfg,
        )
        workload_end = time.monotonic()
        metrics = aggregate_metrics(
            rows,
            workload_start_s=workload_start,
            workload_end_s=workload_end,
        )
        write_request_details_csv(run_dir, rows)
        write_metrics_json(run_dir, metrics)
        runner_log.info(
            "Done completed=%s failed=%s duration_s=%.4f mean_e2e_ms=%.2f",
            metrics.get("completed", 0),
            metrics.get("failed", 0),
            float(metrics.get("duration", 0.0)),
            float(metrics.get("mean_e2e_latency_ms", 0.0)),
        )
        runner_log.info("results_dir=%s", run_dir)
        return run_dir
    finally:
        if process is not None:
            stop_server(process, server_log_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run synthesis_requests GSP workload on real SGLang GPU inference.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help="Directory containing client.json and server.json",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
    )
    parser.add_argument(
        "--skip-server",
        action="store_true",
        help="Assume SGLang server is already running at host:port from config.",
    )
    parser.add_argument("--log-level", type=str, default="info")
    args = parser.parse_args(argv)

    try:
        run_dir = run_experiment(
            args.config_dir,
            args.results_root,
            log_level=args.log_level,
            skip_server=args.skip_server,
        )
        print(json.dumps({"results_dir": str(run_dir)}, indent=2))
    except Exception as exc:
        setup_file_logger(
            "sglang.benchmark.sglsim.synthesis_requests.runner",
            CLIENT_LOG_PATH,
            args.log_level,
        ).exception("Experiment failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
