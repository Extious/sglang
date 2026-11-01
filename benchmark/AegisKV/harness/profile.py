"""Offline recovery profile on two Workers of the target GPU.

Measures (1) cold prefill latency for several input lengths, (2) prefill of a
short suffix on top of a cached prefix, and (3) the takeover import of a peer-DRAM
checkpoint, then fits `RecoveryProfile` (cost_model.py) and writes it as JSON.

    cd benchmark/AegisKV && python -m harness.profile --config configs/<name>.json \
        --output profiles/<gpu>.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from typing import List, Tuple

import aiohttp
import msgspec
import numpy as np
from harness import workload
from harness.client import control, generate_stream
from harness.cluster import Cluster
from harness.config import ExperimentConfig
from harness.cost_model import RecoveryProfile

LENGTHS = [2048, 4096, 8192, 16384, 24576]
SUFFIXES = [64, 512, 2048]
REPEATS = 2


async def _ttft(session, url, rid, input_ids) -> float:
    t = time.monotonic()
    async for _ in generate_stream(
        session, url, rid=rid, input_ids=input_ids, max_new_tokens=1
    ):
        break
    return time.monotonic() - t


async def _flush(session, url):
    async with session.post(f"{url}/flush_cache") as r:
        await r.read()


async def _prefill_samples(session, url, lengths) -> List[Tuple[int, int, float]]:
    samples = []
    for rep in range(REPEATS):
        for n in lengths:
            ids = workload.own_history(seed=100 + rep, workflow=n, role="P", length=n)
            await _flush(session, url)
            samples.append((n, 0, await _ttft(session, url, f"p{n}r{rep}", ids)))
            for s in SUFFIXES:
                ext = workload.own_history(
                    seed=200 + rep, workflow=s, role="S", length=s
                )
                t = await _ttft(session, url, f"p{n}s{s}r{rep}", ids + ext)
                samples.append((s, n, t))
    return samples


async def _import_samples(session, src, dst, lengths) -> List[Tuple[int, float]]:
    samples = []
    for n in lengths:
        rid = f"imp{n}"
        ids = workload.own_history(seed=300, workflow=n, role="I", length=n)
        spec = {
            "target_id": rid,
            "arena": dst.arena_path,
            "slot_offset": 0,
            "capacity": n + 8,
            "layer_lo": 0,
            "layer_hi": 32,
            "max_tokens": n,
        }
        await control(session, src.url, "protect", {"rid": rid, "targets": [spec]})
        async for _ in generate_stream(
            session, src.url, rid=rid, input_ids=ids, max_new_tokens=4
        ):
            pass
        while (await control(session, src.url, "poll", {}))["targets"][rid][0] < n:
            await asyncio.sleep(0.05)
        await control(session, src.url, "unprotect", {"target_ids": [rid]})
        await _flush(session, dst.url)
        args = {
            "import_id": rid,
            "token_ids": ids,
            "arena": {
                "path": dst.arena_path,
                "slot_offset": 0,
                "layer_lo": 0,
                "layer_hi": 32,
            },
        }
        await control(session, dst.url, "import", args)
        while True:
            state = (await control(session, dst.url, "poll", {}))["imports"][rid]
            if state["status"] != "copying":
                break
            await asyncio.sleep(0.01)
        samples.append((n, state["copy_seconds"]))
        await control(session, dst.url, "release_import", {"import_id": rid})
    return samples


def _fit(prefill, imports) -> RecoveryProfile:
    a = np.array([[1.0, new, new * (ctx + new / 2)] for new, ctx, _ in prefill])
    y = np.array([t for *_, t in prefill])
    p, *_ = np.linalg.lstsq(a, y, rcond=None)
    b = np.array([[1.0, n] for n, _ in imports])
    r, *_ = np.linalg.lstsq(b, np.array([t for _, t in imports]), rcond=None)
    return RecoveryProfile(
        p0=max(0.0, float(p[0])),
        p1=max(0.0, float(p[1])),
        p2=max(0.0, float(p[2])),
        r0=max(0.0, float(r[0])),
        r1=max(0.0, float(r[1])),
    )


async def collect_profile(
    cluster: Cluster, *, output: str, max_len: int
) -> RecoveryProfile:
    """Profile on the first two live Workers of an already started cluster."""
    w0, w1 = cluster.workers[:2]
    lengths = [n for n in LENGTHS if n + max(SUFFIXES) <= max_len]
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None)
    ) as session:
        for w in (w0, w1):
            await control(session, w.url, "reset", {})
            await control(session, w.url, "map_arenas", {"paths": [w1.arena_path]})
        # Warm up kernels and JIT caches before timing.
        await _prefill_samples(session, w0.url, [1024])
        prefill = await _prefill_samples(session, w0.url, lengths)
        imports = await _import_samples(session, w0, w1, lengths)
        for w in (w0, w1):
            await _flush(session, w.url)
    profile = _fit(prefill, imports)
    record = {
        **msgspec.to_builtins(profile),
        "samples": {"prefill": prefill, "import": imports},
        "resident_depth": {n: profile.resident_depth(n=n) for n in lengths},
    }
    with open(output, "w") as f:
        json.dump(record, f, indent=2)
    logging.info("profile: %s", json.dumps({k: v for k, v in record.items() if k != "samples"}))
    return profile


async def _main(cfg: ExperimentConfig, output: str) -> None:
    cluster = Cluster(cfg, run_tag=f"profile_{int(time.time())}")
    try:
        await cluster.start_all()
        await collect_profile(cluster, output=output, max_len=cfg.context_length)
    finally:
        cluster.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_main(ExperimentConfig.load(args.config), args.output))


if __name__ == "__main__":
    main()
