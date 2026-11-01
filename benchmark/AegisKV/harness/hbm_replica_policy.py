"""HBM-Replica (DejaVu / KevlarFlow style): replicate every Agent's full KV into
the holder's GPU KV pool, first come first served, under a static per-holder
quota; takeover adopts the replica without copying."""

from __future__ import annotations

from typing import Any

from harness.policy import Policy

NUM_LAYERS = 32


class HbmReplicaPolicy(Policy):
    name = "hbm_replica"

    def __init__(self, cfg: Any):
        self.quota_tokens = cfg.hbm_quota_tokens

    async def tick(self, master: Any) -> None:
        for call in sorted(master.running.values(), key=lambda c: c.t_submit):
            if (call.prot is not None and call.prot.has_region) or not master.alive(
                call.holder
            ):
                continue
            if (
                master.region_slots_used[call.holder] + call.total_len
                > self.quota_tokens
            ):
                continue
            await master.add_region(call, NUM_LAYERS)


def make(cfg: Any) -> HbmReplicaPolicy:
    return HbmReplicaPolicy(cfg)
