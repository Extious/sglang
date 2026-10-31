"""Data-plane smoke test on two Workers: checkpoint on worker 0, import on worker 1.

For each import source (peer DRAM only, HBM first-L layers + DRAM, full HBM
replica) it checks that the resumed request on worker 1 hits the imported
prefix and that its greedy continuation equals a from-scratch run.

    cd benchmark/AegisKV && python -m harness.smoke --config configs/smoke_local.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time

import aiohttp
from harness import workload
from harness.client import control, generate_stream
from harness.cluster import Cluster
from harness.config import ExperimentConfig

PROMPT_LEN = 3000
GEN_LEN = 160
CONTINUE = 48


async def _generate(session, url, rid, input_ids, n):
    out, meta = [], {}
    async for chunk in generate_stream(
        session, url, rid=rid, input_ids=input_ids, max_new_tokens=n
    ):
        out, meta = chunk.get("output_ids") or out, chunk["meta_info"]
    return out, meta


async def _wait(session, url, pred, what, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        res = await control(session, url, "poll", {})
        if pred(res):
            return res
        await asyncio.sleep(0.05)
    raise TimeoutError(what)


async def _case(session, cluster, name, layers_hbm, use_dram, seed):
    w0, w1 = cluster.workers
    prompt = workload.own_history(seed=seed, workflow=0, role=name, length=PROMPT_LEN)
    total = PROMPT_LEN + GEN_LEN
    rid = f"smoke-{name}"
    targets = []
    if use_dram or layers_hbm:
        targets.append(
            {
                "target_id": f"{rid}#d",
                "arena": w1.arena_path,
                "slot_offset": 0,
                "capacity": total,
                "layer_lo": 0,
                "layer_hi": 32,
                "max_tokens": total,
            }
        )
    await control(session, w0.url, "protect", {"rid": rid, "targets": targets})
    if layers_hbm:
        res = await control(
            session,
            w1.url,
            "reserve_region",
            {"region_id": rid, "token_capacity": total, "resident_layers": layers_hbm},
        )
        assert res["reserved"], res
    output, _ = await _generate(session, w0.url, rid, prompt, GEN_LEN)
    seq = prompt + output
    # The target ends once the request leaves the batch; a tail shorter than the
    # checkpoint interval is never copied.
    await _wait(session, w0.url, lambda r: r["targets"][f"{rid}#d"][2], "target end")
    await asyncio.sleep(0.2)
    res = await control(session, w0.url, "poll", {})
    committed = res["targets"][f"{rid}#d"][0]
    assert committed >= len(seq) - 64, (committed, len(seq))
    if layers_hbm:
        await control(
            session,
            w1.url,
            "stage_region",
            {
                "task_id": f"{rid}#st",
                "region_id": rid,
                "arena": w1.arena_path,
                "slot_offset": 0,
                "token_lo": 0,
                "token_hi": committed,
            },
        )
        await _wait(session, w1.url, lambda r: f"{rid}#st" in r["stages"], "stage")
    args = {"import_id": rid, "token_ids": seq[:committed]}
    if layers_hbm:
        args["region_id"] = rid
    if use_dram:
        args["arena"] = {
            "path": w1.arena_path,
            "slot_offset": 0,
            "layer_lo": layers_hbm,
            "layer_hi": 32,
        }
    t = time.monotonic()
    assert (await control(session, w1.url, "import", args))["accepted"]
    res = await _wait(
        session,
        w1.url,
        lambda r: r["imports"].get(rid, {}).get("status") != "copying",
        "import",
    )
    import_s = time.monotonic() - t
    state = res["imports"][rid]
    resumed, meta = await _generate(session, w1.url, f"{rid}#resume", seq, CONTINUE)
    await control(session, w1.url, "release_import", {"import_id": rid})
    await control(session, w0.url, "unprotect", {"target_ids": [f"{rid}#d"]})
    async with session.post(f"{w1.url}/flush_cache") as r:
        await r.read()
    reference, _ = await _generate(session, w1.url, f"{rid}#ref", seq, CONTINUE)
    report = {
        "case": name,
        "hbm_layers": layers_hbm,
        "dram": use_dram,
        "committed": committed,
        "state": state,
        "import_wall_s": round(import_s, 4),
        "cached_tokens": meta.get("cached_tokens"),
        "continuation_matches": resumed == reference,
    }
    print(json.dumps(report))
    return (
        report["continuation_matches"] and meta.get("cached_tokens", 0) >= committed - 1
    )


async def _main(cfg: ExperimentConfig) -> bool:
    cluster = Cluster(cfg, run_tag=f"smoke_{int(time.time())}")
    try:
        await cluster.start_all()
        async with aiohttp.ClientSession() as session:
            ok = True
            for name, hbm, dram, seed in (
                ("dram", 0, True, 1),
                ("split", 4, True, 2),
                ("replica", 32, False, 3),
            ):
                ok &= await _case(session, cluster, name, hbm, dram, seed)
            print("SMOKE_OK" if ok else "SMOKE_FAILED")
            return ok
    finally:
        cluster.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = ExperimentConfig.load(args.config)
    assert len(cfg.gpus) == 2
    raise SystemExit(0 if asyncio.run(_main(cfg)) else 1)


if __name__ == "__main__":
    main()
