from sglang_simulator.time_predictor.base import (
    InferTimePredictor,
    ScheduleBatch,
    ScheduleRequest,
)

__all__ = (
    "ScheduleRequest",
    "ScheduleBatch",
    "InferTimePredictor",
    "AIConfiguratorTimePredictor",
)


def __getattr__(name):
    if name == "AIConfiguratorTimePredictor":
        from sglang_simulator.time_predictor.aiconfigurator import (
            AIConfiguratorTimePredictor,
        )

        return AIConfiguratorTimePredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
