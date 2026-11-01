"""AegisKV placement: Agent Selection, Region Selection and the Adapter.

Per poll tick, for every running call:
  * Tracker gives the call's slack s.
  * Zero slack (critical path): checkpoint the full history to peer DRAM and keep
    the first L layers in peer HBM, if the holder's Adapter quota allows.
  * Positive slack: checkpoint only the smallest prefix d* whose modeled recovery
    T^D(n, d*) fits in s (often d* = 0, i.e. no backup); DRAM only.
  * DRAM admission serves zero-slack calls first and may evict a positive-slack
    checkpoint on the same holder to make room.
  * Adapter: quota Q_r = clip(M_r - D_r - B_r, 0, Q_max) per holder, with D_r the
    forecast foreground KV demand; Reclaim demotes the earliest regions over Q_r.
Attention is dense here, so Region Selection keeps all blocks (K = 100%) of the
first L layers.
"""

from __future__ import annotations

import asyncio
import collections
from typing import Any, Deque, Dict, List, Tuple

from harness.cost_model import RecoveryProfile
from harness.master import NUM_LAYERS
from harness.policy import Policy
from harness.tracker import Tracker

# A call is on the critical path when its slack is below this, in seconds.
# Arbitrary; it only absorbs floating-point noise in the EF/LF computation.
_ZERO_SLACK_S = 1e-3
# Checkpoint prefixes are rounded up to this many tokens to avoid re-copying
# from scratch each time d* creeps up by a few tokens.
_PREFIX_ROUND = 2048
# Demand forecast window for the Adapter prior, in poll ticks.
_DEMAND_WINDOW = 200


class AegisKVPolicy(Policy):
    name = "aegiskv"

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.profile = RecoveryProfile.load(cfg.profile_path)
        self.tracker = Tracker(
            model_path=cfg.tracker_path,
            role_prior=self._role_prior(cfg),
        )
        # -1 selects L from the profiled zero-wait depth at the input length.
        self.resident_layers = (
            self.profile.resident_depth(n=cfg.input_len)
            if cfg.resident_layers < 0
            else cfg.resident_layers
        )
        self._demand: Dict[int, Deque[int]] = collections.defaultdict(
            lambda: collections.deque(maxlen=_DEMAND_WINDOW)
        )
        self._last_slack: Dict[str, float] = {}
        self._quotas: Dict[int, int] = {}

    def _role_prior(self, cfg: Any) -> Dict[str, float]:
        # Fallback when no GBDT is trained: prefill plus a nominal 30 ms/token decode.
        prior = {}
        for role, out in cfg.output_lens.items():
            prior[role] = self.profile.prefill(cfg.input_len, 0) + 0.03 * out
        return prior

    def describe(self, call: Any) -> Dict[str, Any]:
        info = super().describe(call)
        info["slack"] = self._last_slack.get(call.key)
        return info

    async def tick(self, master: Any) -> None:
        slack = self.tracker.slack(master)
        self._last_slack = slack
        self._quotas = {w: self._quota(master, w) for w in master.stats}
        plans = self._plan(master, slack)
        await self._apply(master, plans, slack)
        await self._adapt(master)

    def _plan(
        self, master: Any, slack: Dict[str, float]
    ) -> List[Tuple[Any, bool, int]]:
        plans = []
        for call in master.running.values():
            s = slack.get(call.key, 0.0)
            zero = s <= _ZERO_SLACK_S
            if zero:
                prefix = call.total_len
            else:
                prefix = self.profile.min_prefix(n=call.history_len, slack=s)
                if prefix > 0:
                    prefix = min(
                        call.total_len, -(-prefix // _PREFIX_ROUND) * _PREFIX_ROUND
                    )
            plans.append((call, zero, prefix))
        plans.sort(key=lambda p: (not p[1], slack.get(p[0].key, 0.0)))
        return plans

    async def _apply(
        self, master: Any, plans: List[Tuple[Any, bool, int]], slack: Dict[str, float]
    ) -> None:
        # Coroutines start in plan order and allocate before their first await,
        # so zero-slack calls still get DRAM first.
        await asyncio.gather(
            *(
                self._apply_one(master, call, zero, prefix, slack)
                for call, zero, prefix in plans
                if master.alive(call.holder)
            )
        )

    async def _apply_one(
        self, master: Any, call: Any, zero: bool, prefix: int, slack: Dict[str, float]
    ) -> None:
        prot = call.prot
        if prefix == 0:
            if prot is not None:
                await master.release_protection(call)
            return
        if prot is None or not prot.has_dram or prot.max_tokens != prefix:
            ok = await master.protect_dram(call, prefix)
            if not ok and zero and await self._evict_positive(master, call.holder, slack):
                await master.protect_dram(call, prefix)
        prot = call.prot
        if prot is None:
            return
        if not zero:
            if prot.has_region:
                await master.drop_region(call)
            return
        # HBM and DRAM together must cover the prefix: without a DRAM checkpoint
        # the region has to hold every layer.
        layers = self.resident_layers if prot.has_dram else NUM_LAYERS
        if prot.has_region and prot.region_layers and prot.region_layers < layers:
            await master.drop_region(call)
        if layers > 0 and not call.prot.has_region and self._region_fits(master, call, layers):
            await master.add_region(call, layers)

    async def _evict_positive(
        self, master: Any, holder: int, slack: Dict[str, float]
    ) -> bool:
        victims = [
            c
            for c in master.running.values()
            if c.holder == holder
            and c.prot is not None
            and c.prot.has_dram
            and slack.get(c.key, 0.0) > _ZERO_SLACK_S
        ]
        if not victims:
            return False
        victim = max(victims, key=lambda c: slack.get(c.key, 0.0))
        await master.release_protection(victim)
        return True

    # ------------------------------------------------------------------ #
    # Adapter
    # ------------------------------------------------------------------ #

    @staticmethod
    def _region_slots(call: Any, layers: int) -> int:
        return -(-layers * call.total_len // NUM_LAYERS)

    def _quota(self, master: Any, w: int) -> int:
        stats = master.stats.get(w)
        if not stats:
            return 0
        total = stats["total"]
        foreground = total - stats["available"] - stats["evictable"] - stats["held"]
        window = self._demand[w]
        window.append(foreground)
        forecast = max(foreground, sorted(window)[int(0.95 * (len(window) - 1))])
        reserve = int(self.cfg.hbm_reserve_fraction * total)
        return max(0, min(self.cfg.hbm_quota_tokens, total - forecast - reserve))

    def _region_fits(self, master: Any, call: Any, layers: int) -> bool:
        w = call.holder
        need = self._region_slots(call, layers)
        return master.region_slots_used[w] + need <= self._quotas.get(w, 0)

    async def _adapt(self, master: Any) -> None:
        for worker in master.cluster.workers:
            w = worker.index
            if not worker.alive:
                continue
            quota = self._quotas.get(w, 0)
            if master.region_slots_used[w] <= quota:
                continue
            holders = sorted(
                (
                    c
                    for c in master.running.values()
                    if c.prot is not None and c.prot.has_region and c.prot.holder == w
                ),
                key=lambda c: c.t_submit,
            )
            for call in holders:
                if master.region_slots_used[w] <= quota:
                    break
                await master.drop_region(call)


def make(cfg: Any) -> AegisKVPolicy:
    return AegisKVPolicy(cfg)
