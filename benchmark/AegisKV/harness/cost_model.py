"""Offline-profiled recovery costs: T^D(n, d) and the resident depth L.

`profile.py` measures, on the target GPU, prefill latency as a function of new
and cached tokens and the takeover import throughput, and writes the fitted
coefficients consumed here.
"""

from __future__ import annotations

import json
import math
from typing import Optional

import msgspec


class RecoveryProfile(msgspec.Struct, kw_only=True):
    # prefill(new, ctx) = p0 + p1 * new + p2 * new * (ctx + new / 2), seconds.
    p0: float = 0.05
    p1: float = 1.0e-4
    p2: float = 1.0e-9
    # Import of d tokens from peer DRAM: r0 + r1 * d seconds (all layers).
    r0: float = 0.01
    r1: float = 1.5e-6
    # Fixed takeover overhead: resubmission plus the retry Worker's batch boundary.
    q0: float = 0.3
    num_layers: int = 32

    @classmethod
    def load(cls, path: Optional[str]) -> RecoveryProfile:
        if path is None:
            return cls()
        with open(path) as f:
            return msgspec.convert(json.load(f), type=cls)

    def prefill(self, new: int, ctx: int) -> float:
        if new <= 0:
            return 0.0
        return self.p0 + self.p1 * new + self.p2 * new * (ctx + new / 2)

    def restore(self, d: int) -> float:
        return self.r0 + self.r1 * d if d > 0 else 0.0

    def recovery_delay(self, n: int, d: int, slowdown: float = 1.0) -> float:
        """T^D(n, d): restore a d-token prefix, recompute the other n - d tokens.

        `slowdown` scales the profiled (unloaded) recompute by the retry
        Worker's recent contention, as in the paper's takeover estimate.
        """
        return self.q0 + self.restore(d) + slowdown * self.prefill(n - d, d)

    def min_prefix(
        self, *, n: int, slack: float, slowdown: float = 1.0, step: int = 1024
    ) -> int:
        """Smallest checkpoint prefix d whose recovery fits the slack (Eq. 5)."""
        best, best_cost = n, self.recovery_delay(n, n, slowdown)
        for d in list(range(0, n, step)) + [n]:
            cost = self.recovery_delay(n, d, slowdown)
            if cost <= slack:
                return d
            if cost < best_cost:
                best, best_cost = d, cost
        return best

    def resident_depth(self, *, n: int, suffix: int = 256) -> int:
        """Smallest L with zero modeled wait when layers >= L stream in during replay.

        Layer l (>= L) arrives at (l - L + 1) * D and is needed at l * C, where C is
        the per-layer replay compute and D the per-layer copy of n tokens.
        """
        per_layer_copy = self.r1 * n / self.num_layers
        per_layer_compute = self.prefill(suffix, n) / self.num_layers
        if per_layer_copy <= per_layer_compute:
            return 0
        last = self.num_layers - 1
        need = self.num_layers - last * per_layer_compute / per_layer_copy
        return min(self.num_layers, max(0, math.ceil(need)))
