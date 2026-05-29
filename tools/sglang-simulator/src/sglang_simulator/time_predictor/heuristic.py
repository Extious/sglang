import numpy as np

from sglang_simulator.time_predictor.base import (
    InferTimePredictor,
    ScheduleBatch,
)


class HeuristicTimePredictor(InferTimePredictor):
    """Fallback latency model when AIConfigurator data is unavailable."""

    name = "heuristic"

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        layers = max(self.model.num_hidden_layers, 1)
        hidden = max(self.model.hidden_size, 1)

        if batch.is_decode():
            base_ms = 6.0 + 0.003 * hidden / 1024.0 + 0.002 * layers
            return batch.batch_size * base_ms / 1000.0

        cost = 0.0
        for req in batch.reqs:
            seq_len = req.past_kv_length + req.extend_length
            cost += seq_len * req.extend_length
        scale = (hidden * layers) / (4096.0 * 32.0)
        return max(cost * scale * 2.5e-8, 1e-4)
