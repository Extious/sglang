"""Train the Tracker GBDT on No-Failure runs (features known at submission only).

    cd benchmark/AegisKV && python -m harness.train_tracker \
        --results runs/<nofail>/result.json [...] --output trackers/<name>.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
from harness.tracker import call_features
from sklearn.ensemble import HistGradientBoostingRegressor


def _dataset(paths):
    xs, ys = [], []
    for path in paths:
        with open(path) as f:
            result = json.load(f)
        if result["fail_time"] is not None:
            raise ValueError(f"{path} contains a failure; train on No-Failure runs")
        for c in result["calls"]:
            xs.append(
                call_features(
                    role=c["role"], prompt_len=c["prompt_len"], stats=c["submit_stats"]
                )
            )
            ys.append(c["t_done"] - c["t_submit"])
    return np.array(xs), np.array(ys)


def train(results, output: str) -> dict:
    x, y = _dataset(results)
    model = HistGradientBoostingRegressor(max_iter=300, max_leaf_nodes=31)
    model.fit(x, y)
    with open(output, "wb") as f:
        pickle.dump(model, f)
    report = {
        "samples": len(y),
        "train_mae_s": float(np.mean(np.abs(model.predict(x) - y))),
        "mean_s": float(y.mean()),
    }
    with open(output + ".json", "w") as f:
        json.dump(report, f)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(train(args.results, args.output)))


if __name__ == "__main__":
    main()
