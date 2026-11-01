"""Run experiments on one set of Workers.

    cd benchmark/AegisKV && python -m harness.run --config configs/<name>.json \
        --suite nofail,rebuild,aegiskv --warmup 4 --out-dir runs/<name>

Workers start once; each suite entry resets backups and the radix cache,
restarts a Worker killed by the previous entry, then runs the full workload.
`nofail` runs the Rebuild policy without the failure. Workers are always torn
down on exit so no GPU sits idle after a crash.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from typing import List

import msgspec

from harness.cluster import Cluster
from harness.config import ExperimentConfig
from harness.master import Master
from harness.policy import make_policy


async def _ensure_alive(cluster: Cluster) -> None:
    for worker in cluster.workers:
        if worker.alive:
            continue
        if worker.proc is None or worker.proc.poll() is not None:
            cluster.start(worker)
        await cluster.wait_ready(worker)


def _entry_config(cfg: ExperimentConfig, entry: str, out_dir: str) -> ExperimentConfig:
    if entry == "nofail":
        return msgspec.structs.replace(
            cfg, policy="rebuild", out_dir=out_dir, fail_worker=None, fail_at_s=None
        )
    return msgspec.structs.replace(cfg, policy=entry, out_dir=out_dir)


async def _main(cfg: ExperimentConfig, suite: List[str], warmup: int) -> None:
    cluster = Cluster(cfg, run_tag=f"{cfg.name}_{os.getpid()}")
    try:
        t = time.monotonic()
        await cluster.start_all()
        logging.info("workers ready in %.1fs", time.monotonic() - t)
        if warmup:
            warm = msgspec.structs.replace(
                _entry_config(cfg, "nofail", os.path.join(cfg.out_dir, "warmup")),
                num_workflows=warmup,
            )
            await Master(cfg=warm, cluster=cluster, policy=make_policy("rebuild", warm)).run()
        for entry in suite:
            await _ensure_alive(cluster)
            run_cfg = _entry_config(cfg, entry, os.path.join(cfg.out_dir, entry))
            os.makedirs(run_cfg.out_dir, exist_ok=True)
            run_cfg.dump(os.path.join(run_cfg.out_dir, "config.json"))
            t = time.monotonic()
            policy = make_policy(run_cfg.policy, run_cfg)
            await Master(cfg=run_cfg, cluster=cluster, policy=policy).run()
            logging.info("RUN_DONE %s in %.1fs", entry, time.monotonic() - t)
    finally:
        cluster.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--suite", default=None, help="comma list; default: cfg.policy")
    parser.add_argument("--warmup", type=int, default=0, help="warm-up Workflows")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = ExperimentConfig.load(args.config)
    if args.out_dir:
        cfg = msgspec.structs.replace(cfg, out_dir=args.out_dir)
    suite = args.suite.split(",") if args.suite else [cfg.policy]
    os.makedirs(cfg.out_dir, exist_ok=True)
    asyncio.run(_main(cfg, suite, args.warmup))


if __name__ == "__main__":
    main()
