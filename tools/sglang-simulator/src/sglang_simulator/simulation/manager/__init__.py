from sglang_simulator.simulation.manager.failure import (
    BackupPolicy,
    FailureConfig,
    FailureEvent,
    FailureManager,
)

__all__ = [
    "StateManager",
    "Envs",
    "ConfigManager",
    "BackupPolicy",
    "FailureConfig",
    "FailureEvent",
    "FailureManager",
]


def __dir__():
    return sorted(__all__)


def __getattr__(name):
    if name == "ConfigManager":
        from sglang_simulator.simulation.manager.config import ConfigManager

        return ConfigManager
    if name == "Envs":
        from sglang_simulator.simulation.manager.env import Envs

        return Envs
    if name == "StateManager":
        from sglang_simulator.simulation.manager.state import StateManager

        return StateManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
