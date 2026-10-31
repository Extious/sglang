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
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
        self.role_prior = role_prior

    def predict_duration(
        self, *, role: str, prompt_len: int, stats: Dict[str, int]
    ) -> float:
        if self.model is None:
            return self.role_prior[role]
        x = call_features(role=role, prompt_len=prompt_len, stats=stats)
        return max(0.1, float(self.model.predict([x])[0]))

    def _remaining(
        self, master: Any, wf: int, role: str, avg_stats: Dict[str, int]
    ) -> float:
        call = master.calls.get(f"wf{wf}-{role}")
        if call is not None and call.t_done:
            return 0.0
        if call is None:
            prompt_len = master.cfg.input_len + sum(
                master.cfg.output_lens[p] for p in workload.PREDECESSORS[role]
            )
            return self.predict_duration(
                role=role, prompt_len=prompt_len, stats=avg_stats
            )
        total = self.predict_duration(
            role=role, prompt_len=len(call.prompt), stats=call.submit_stats
        )
        elapsed = master.now() - call.t_submit
        return max(total - elapsed, 0.05 * total)

    def slack(self, master: Any) -> Dict[str, float]:
        stats = list(master.stats.values())
        avg = {
            k: int(sum(s.get(k, 0) for s in stats) / max(1, len(stats)))
            for k in ("running_reqs", "waiting_reqs")
        }
        result: Dict[str, float] = {}
        for wf in {c.workflow for c in master.running.values()}:
            rem = {r: self._remaining(master, wf, r, avg) for r in workload.ROLE_INDEX}
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
