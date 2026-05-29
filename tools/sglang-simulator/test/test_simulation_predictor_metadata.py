import builtins
import json
import sys
from types import SimpleNamespace

import pytest

from sglang_simulator.simulation.manager import ConfigManager
from sglang_simulator.simulation.sglang import scheduler as scheduler_module
from sglang_simulator.simulation.sglang.scheduler import C_SchedulerHook
from sglang_simulator.simulation.types import RequestStats, SchedulerConfig
from sglang_simulator.spec.accelerator import AcceleratorInfo
from sglang_simulator.spec.model import ModelInfo
from sglang_simulator.time_predictor.heuristic import HeuristicTimePredictor


@pytest.fixture(autouse=True)
def reset_config_manager():
    ConfigManager.reset_config_cache()
    yield
    ConfigManager.reset_config_cache()


@pytest.fixture
def scheduler_hook_state():
    state = {
        "INFERENCE_PREDICTOR": C_SchedulerHook.INFERENCE_PREDICTOR,
        "FAILURE_MANAGER": C_SchedulerHook.FAILURE_MANAGER,
        "REQUEST_STATS": dict(C_SchedulerHook.REQUEST_STATS),
        "ITERATION_STATS": list(C_SchedulerHook.ITERATION_STATS),
        "FAILURE_EVENTS": list(C_SchedulerHook.FAILURE_EVENTS),
        "FAILED_DP_RANKS": set(C_SchedulerHook.FAILED_DP_RANKS),
    }
    yield
    C_SchedulerHook.INFERENCE_PREDICTOR = state["INFERENCE_PREDICTOR"]
    C_SchedulerHook.FAILURE_MANAGER = state["FAILURE_MANAGER"]
    C_SchedulerHook.REQUEST_STATS.clear()
    C_SchedulerHook.REQUEST_STATS.update(state["REQUEST_STATS"])
    C_SchedulerHook.ITERATION_STATS.clear()
    C_SchedulerHook.ITERATION_STATS.extend(state["ITERATION_STATS"])
    C_SchedulerHook.FAILURE_EVENTS.clear()
    C_SchedulerHook.FAILURE_EVENTS.extend(state["FAILURE_EVENTS"])
    C_SchedulerHook.FAILED_DP_RANKS.clear()
    C_SchedulerHook.FAILED_DP_RANKS.update(state["FAILED_DP_RANKS"])


def _write_config(tmp_path, monkeypatch, config: dict):
    config_path = tmp_path / "sim.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setenv("SGLANG_SIMULATOR_CONFIG_PATH", str(config_path))


def _model() -> ModelInfo:
    return ModelInfo(
        model_path="test-model",
        hidden_size=4096,
        num_hidden_layers=32,
    )


def _accelerator() -> AcceleratorInfo:
    return AcceleratorInfo(
        name="test-gpu",
        vendor="test",
        hbm_capacity_gb=80,
        hbm_bandwidth_gb=2000,
    )


def test_predictor_config_defaults_to_aiconfigurator_when_section_absent(
    tmp_path, monkeypatch
):
    _write_config(tmp_path, monkeypatch, {"scheduler": {"backend_name": "sglang"}})

    predictor_config = ConfigManager.get_predictor_config()

    assert predictor_config["name"] == "aiconfigurator"


def test_predictor_config_defaults_to_aiconfigurator_when_section_is_malformed(
    tmp_path, monkeypatch
):
    _write_config(tmp_path, monkeypatch, {"predictor": "heuristic"})

    predictor_config = ConfigManager.get_predictor_config()

    assert predictor_config == {"name": "aiconfigurator"}


def test_get_inference_time_predictor_uses_centralized_predictor_config(monkeypatch):
    monkeypatch.setattr(
        ConfigManager,
        "get_predictor_config",
        classmethod(lambda cls: {"name": "heuristic"}),
    )
    monkeypatch.setattr(
        ConfigManager,
        "_get_raw_config",
        classmethod(
            lambda cls: pytest.fail(
                "get_inference_time_predictor should use get_predictor_config"
            )
        ),
    )

    predictor = ConfigManager.get_inference_time_predictor(
        _model(), _accelerator(), SchedulerConfig()
    )

    assert predictor.name == "heuristic"


def test_default_aiconfigurator_falls_back_to_heuristic_when_import_fails(
    tmp_path, monkeypatch
):
    _write_config(tmp_path, monkeypatch, {"scheduler": {"backend_name": "sglang"}})
    sys.modules.pop("sglang_simulator.time_predictor.aiconfigurator", None)
    original_import = builtins.__import__

    def fail_aiconfigurator_import(name, *args, **kwargs):
        if name == "sglang_simulator.time_predictor.aiconfigurator":
            raise ModuleNotFoundError("No module named 'aiconfigurator'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_aiconfigurator_import)

    predictor = ConfigManager.get_inference_time_predictor(
        _model(), _accelerator(), SchedulerConfig()
    )

    assert isinstance(predictor, HeuristicTimePredictor)


def test_heuristic_predictor_exposes_stable_name():
    predictor = HeuristicTimePredictor(_model(), _accelerator(), SchedulerConfig())

    assert predictor.name == "heuristic"


def test_scheduler_profile_writes_predictor_metadata_sidecar(
    tmp_path, monkeypatch, scheduler_hook_state
):
    monkeypatch.setenv("SGLANG_SIMULATOR_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(scheduler_module, "_reset_failure_manager_state", lambda: None)

    class FakeProfileReqOutput:
        def __init__(self, success, message):
            self.success = success
            self.message = message

    original_import_module = scheduler_module.importlib.import_module

    def fake_import_module(name, package=None):
        if name == "sglang.srt.managers.io_struct":
            return SimpleNamespace(ProfileReqOutput=FakeProfileReqOutput)
        return original_import_module(name, package)

    class FakeTarget:
        def __init__(self):
            pass

        def recv_requests(self):
            return []

        def get_new_batch_prefill(self):
            return None

        def run_batch(self):
            return None

        def process_batch_result(self):
            return None

        def event_loop_normal(self):
            return None

        def profile(self, req, *args, **kwargs):
            return None

    monkeypatch.setattr(scheduler_module.importlib, "import_module", fake_import_module)
    C_SchedulerHook.hook(FakeTarget)

    C_SchedulerHook.REQUEST_STATS.clear()
    C_SchedulerHook.ITERATION_STATS.clear()
    C_SchedulerHook.FAILURE_EVENTS.clear()
    C_SchedulerHook.FAILED_DP_RANKS.clear()
    C_SchedulerHook.INFERENCE_PREDICTOR = SimpleNamespace(name="heuristic")
    C_SchedulerHook.FAILURE_MANAGER = object()
    C_SchedulerHook.REQUEST_STATS["r1"] = RequestStats(
        rid="r1",
        input_length=4,
        output_length=2,
        created_time=10.0,
        queue_start=10.0,
        queue_end=10.5,
        last_event_time=11.0,
        status="completed",
        gen_token_latencies=[0.25, 0.5],
    )

    object.__new__(FakeTarget).profile(None)

    metadata_path = tmp_path / "simulator_metadata.json"
    metadata_lines = metadata_path.read_text().splitlines()

    assert len(metadata_lines) == 1
    assert json.loads(metadata_lines[0]) == {
        "predictor": "heuristic",
        "simulation_mode": C_SchedulerHook.SIM_MODE.value,
        "failure_enabled": True,
    }
