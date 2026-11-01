"""Run experiments on one set of Workers.

    cd benchmark/AegisKV && python -m harness.run --config configs/<name>.json \
        --suite nofail,rebuild,aegiskv --warmup 4 --out-dir runs/<name>

Workers start once. The warm-up runs No-Failure Workflows with a different seed
(it also trains AegisKV's Tracker). Entries: `nofail` (Rebuild, no failure), a
policy name (with the failure), `profile` (fit T^D on workers 0-1) and
`tracker` (train the GBDT on the warm-up). Each run resets backups and the
radix cache and restarts a Worker killed by the previous run. Workers are always
torn down on exit so no GPU sits idle after a crash.
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
    """`nofail`, `<policy>`, or `<policy>@<worker>` to choose the failed Worker."""
    if entry == "nofail":
        return msgspec.structs.replace(
            cfg, policy="rebuild", out_dir=out_dir, fail_worker=None, fail_at_s=None
        )
    policy, _, worker = entry.partition("@")
    fail_worker = int(worker) if worker else cfg.fail_worker
    return msgspec.structs.replace(
        cfg, policy=policy, out_dir=out_dir, fail_worker=fail_worker
    )


async def _prepare(
    cfg: ExperimentConfig, cluster: Cluster, entry: str
) -> ExperimentConfig:
    """Run an offline-input entry and return the config later runs should use."""
    if entry == "profile":
        from harness.profile import collect_profile

        path = os.path.join(cfg.out_dir, "profile.json")
        await collect_profile(cluster, output=path, max_len=cfg.context_length)
        return msgspec.structs.replace(cfg, profile_path=path)
    if entry == "tracker":
        from harness.train_tracker import train

        path = os.path.join(cfg.out_dir, "tracker.pkl")
        train([os.path.join(cfg.out_dir, "warmup", "result.json")], path)
        return msgspec.structs.replace(cfg, tracker_path=path)
    raise ValueError(entry)


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
                seed=cfg.seed + 1000,
            )
            await Master(
                cfg=warm, cluster=cluster, policy=make_policy("rebuild", warm)
            ).run()
            logging.info("RUN_DONE warmup")
        for entry in suite:
            await _ensure_alive(cluster)
            if entry in ("profile", "tracker"):
                cfg = await _prepare(cfg, cluster, entry)
                logging.info("RUN_DONE %s", entry)
                continue
            run_cfg = _entry_config(
                cfg, entry, os.path.join(cfg.out_dir, entry.replace("@", "_w"))
            )
            os.makedirs(run_cfg.out_dir, exist_ok=True)
            run_cfg.dump(os.path.join(run_cfg.out_dir, "config.json"))
            t = time.monotonic()
            policy = make_policy(run_cfg.policy, run_cfg)
            master = Master(cfg=run_cfg, cluster=cluster, policy=policy)
            await master.run()
            logging.info("RUN_DONE %s in %.1fs", entry, time.monotonic() - t)
            if entry == "nofail" and cfg.fail_fraction is not None:
                makespan = max(end for _, end in master.workflow_times.values())
                cfg = msgspec.structs.replace(
                    cfg, fail_at_s=round(cfg.fail_fraction * makespan, 1)
                )
                with open(os.path.join(cfg.out_dir, "fail_at_s.txt"), "w") as f:
                    f.write(f"{cfg.fail_at_s}\n")
                logging.info("fail_at_s=%s", cfg.fail_at_s)
    finally:
        cluster.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--suite", default=None, help="comma list; default: cfg.policy")
    parser.add_argument("--warmup", type=int, default=0, help="warm-up Workflows")
    parser.add_argument(
        "--fail-at", type=float, default=None, help="fixed failure time (s)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = ExperimentConfig.load(args.config)
    if args.out_dir:
        cfg = msgspec.structs.replace(cfg, out_dir=args.out_dir)
    if args.fail_at is not None:
        cfg = msgspec.structs.replace(cfg, fail_at_s=args.fail_at, fail_fraction=None)
    suite = args.suite.split(",") if args.suite else [cfg.policy]
    os.makedirs(cfg.out_dir, exist_ok=True)
    asyncio.run(_main(cfg, suite, args.warmup))


if __name__ == "__main__":
    main()
