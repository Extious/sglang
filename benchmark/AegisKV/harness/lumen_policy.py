"""LUMEN-CP*: checkpoint every Agent's full history to peer DRAM, first come first
served, until the holder's DRAM budget is full; no HBM replicas."""

from __future__ import annotations

from typing import Any

from harness.policy import Policy


class LumenPolicy(Policy):
    name = "lumen"

    async def tick(self, master: Any) -> None:
        for call in sorted(master.running.values(), key=lambda c: c.t_submit):
            if call.prot is None or not call.prot.has_dram:
                await master.protect_dram(call, call.total_len)


def make(cfg: Any) -> LumenPolicy:
    return LumenPolicy()
