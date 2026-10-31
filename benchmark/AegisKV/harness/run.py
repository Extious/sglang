"""Run one experiment: start Workers, drive the workload, inject the failure.

    cd benchmark/AegisKV && python -m harness.run --config configs/<name>.json

Workers are always torn down on exit so no GPU sits idle after a crash.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time

from harness.cluster import Cluster
from harness.config import ExperimentConfig
from harness.master import Master
from harness.policy import make_policy


async def _main(cfg: ExperimentConfig) -> None:
    cluster = Cluster(cfg, run_tag=f"{cfg.name}_{os.getpid()}")
    try:
        t = time.monotonic()
        await cluster.start_all()
        logging.info("workers ready in %.1fs", time.monotonic() - t)
        master = Master(cfg=cfg, cluster=cluster, policy=make_policy(cfg.policy, cfg))
        await master.run()
    finally:
        cluster.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--policy")
    parser.add_argument("--no-failure", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = ExperimentConfig.load(args.config)
    if args.out_dir:
        cfg.out_dir = args.out_dir
    if args.policy:
        cfg.policy = args.policy
    if args.no_failure:
        cfg.fail_worker = cfg.fail_at_s = None
    os.makedirs(cfg.out_dir, exist_ok=True)
    cfg.dump(os.path.join(cfg.out_dir, "config.json"))
    asyncio.run(_main(cfg))


if __name__ == "__main__":
    main()
