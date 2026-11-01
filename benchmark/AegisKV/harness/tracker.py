"""Tracker: GBDT remaining-time prediction plus critical-path slack (Eqs. 2-4).

The GBDT predicts an Agent call's duration from features known at submission
(role, input length, execution Worker load). It is trained on No-Failure runs
(`train_tracker.py`). Slack comes from the Workflow DAG: EF_a = R_a + max EF_pred,
LF_a = min(F, min over successors of LF_c - R_c), s_a = LF_a - EF_a.
"""

from __future__ import annotations

import pickle
from typing import Any, Dict, List, Optional

from harness import workload


def call_features(*, role: str, prompt_len: int, stats: Dict[str, int]) -> List[float]:
    return [
        float(workload.ROLE_INDEX[role]),
        float(prompt_len),
        float(stats.get("running_reqs", 0)),
        float(stats.get("waiting_reqs", 0)),
    ]


class Tracker:
    def __init__(self, *, model_path: Optional[str], role_prior: Dict[str, float]):
        self.model = None
        if model_path is not None:
            # Predictions run on the master's event loop; OpenMP fan-out on a
            # many-core host shared with the Workers stalls it for seconds.
            from threadpoolctl import threadpool_limits

            threadpool_limits(limits=1)
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
        self.role_prior = role_prior
        # A call's prediction uses submission-time features only, so it is fixed.
        self._call_totals: Dict[str, float] = {}

    def _predict(self, rows: List[List[float]], roles: List[str]) -> List[float]:
        if self.model is None:
            return [self.role_prior[r] for r in roles]
        return [max(0.1, float(y)) for y in self.model.predict(rows)]

    def _call_total(self, call: Any) -> float:
        total = self._call_totals.get(call.key)
        if total is None:
            row = call_features(
                role=call.role, prompt_len=len(call.prompt), stats=call.submit_stats
            )
            total = self._predict([row], [call.role])[0]
            self._call_totals[call.key] = total
        return total

    def _future_totals(self, master: Any, avg: Dict[str, int]) -> Dict[str, float]:
        """One batched prediction per tick for calls not submitted yet, by role."""
        roles = list(workload.ROLE_INDEX)
        rows = [
            call_features(
                role=r,
                prompt_len=master.cfg.input_len
                + sum(master.cfg.output_lens[p] for p in workload.PREDECESSORS[r]),
                stats=avg,
            )
            for r in roles
        ]
        return dict(zip(roles, self._predict(rows, roles)))

    def _remaining(
        self, master: Any, wf: int, role: str, future: Dict[str, float]
    ) -> float:
        call = master.calls.get(f"wf{wf}-{role}")
        if call is None:
            return future[role]
        if call.t_done:
            return 0.0
        total = self._call_total(call)
        elapsed = master.now() - call.t_submit
        return max(total - elapsed, 0.05 * total)

    def slack(self, master: Any) -> Dict[str, float]:
        stats = list(master.stats.values())
        avg = {
            k: int(sum(s.get(k, 0) for s in stats) / max(1, len(stats)))
            for k in ("running_reqs", "waiting_reqs")
        }
        future = self._future_totals(master, avg)
        result: Dict[str, float] = {}
        for wf in {c.workflow for c in master.running.values()}:
            rem = {
                r: self._remaining(master, wf, r, future) for r in workload.ROLE_INDEX
            }
            ef: Dict[str, float] = {}
            for stage in workload.STAGES:
                for r in stage:
                    preds = workload.PREDECESSORS[r]
                    ef[r] = rem[r] + max([ef[p] for p in preds], default=0.0)
            finish = max(ef.values())
            lf: Dict[str, float] = {}
            for stage in reversed(workload.STAGES):
                for r in stage:
                    succ = workload.SUCCESSORS[r]
                    lf[r] = min([finish] + [lf[c] - rem[c] for c in succ])
            for call in master.running.values():
                if call.workflow == wf:
                    result[call.key] = max(0.0, lf[call.role] - ef[call.role])
        return result
