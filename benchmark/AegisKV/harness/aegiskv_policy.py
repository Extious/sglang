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

import collections
from typing import Any, Deque, Dict, List, Tuple

from harness.cost_model import RecoveryProfile
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
        self.resident_layers = cfg.resident_layers
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
        for call, zero, prefix in plans:
            if not master.alive(call.holder):
                continue
            prot = call.prot
            if prefix == 0:
                if prot is not None:
                    await master.release_protection(call)
                continue
            if prot is None or not prot.has_dram or prot.max_tokens != prefix:
                ok = await master.protect_dram(call, prefix)
                if not ok and zero:
                    if await self._evict_positive(master, call.holder, slack):
                        await master.protect_dram(call, prefix)
            prot = call.prot
            if prot is None:
                continue
            if zero and not prot.has_region and self.resident_layers > 0:
                if self._region_fits(master, call):
                    await master.add_region(call, self.resident_layers)
            elif not zero and prot.has_region:
                await master.drop_region(call)

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

    def _region_slots(self, call: Any) -> int:
        return -(-self.resident_layers * call.total_len // 32)

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

    def _region_fits(self, master: Any, call: Any) -> bool:
        w = call.holder
        return master.region_slots_used[w] + self._region_slots(
            call
        ) <= self._quotas.get(w, 0)

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
