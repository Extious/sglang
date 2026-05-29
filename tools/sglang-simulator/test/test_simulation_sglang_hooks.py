import sys
import ast
from pathlib import Path
import types


ROOT = Path(__file__).resolve().parents[3]


def _top_level_imports(nodes):
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
            continue
        for child_field in ("body", "orelse", "finalbody"):
            child_nodes = getattr(node, child_field, None)
            if child_nodes:
                yield from _top_level_imports(child_nodes)
        handlers = getattr(node, "handlers", None)
        if handlers:
            for handler in handlers:
                yield from _top_level_imports(handler.body)


def test_sglang_config_helpers_do_not_eagerly_import_transformers():
    source_paths = [
        ROOT / "python/sglang/srt/utils/hf_transformers_utils.py",
        ROOT / "python/sglang/srt/configs/model_config.py",
        ROOT / "python/sglang/srt/multimodal/customized_mm_processor_utils.py",
    ]

    offenders = []
    for source_path in source_paths:
        tree = ast.parse(source_path.read_text())
        for node in _top_level_imports(tree.body):
            if isinstance(node, ast.Import):
                offenders.extend(
                    f"{source_path.name}:{alias.name}"
                    for alias in node.names
                    if alias.name == "transformers"
                )
            elif node.module and (
                node.module == "transformers" or node.module.startswith("transformers.")
            ):
                offenders.append(f"{source_path.name}:{node.module}")

    assert offenders == []


def test_launch_server_defers_hooks_to_scheduler_subprocess():
    source_path = (
        ROOT
        / "tools/sglang-simulator/src/sglang_simulator/simulation/sglang/launch_server.py"
    )
    tree = ast.parse(source_path.read_text())

    top_level_calls = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Name):
                top_level_calls.append(func.id)

    assert "install_sglang_hooks" not in top_level_calls


def test_model_config_does_not_eagerly_import_quantization_registry():
    source_path = ROOT / "python/sglang/srt/configs/model_config.py"
    tree = ast.parse(source_path.read_text())

    offenders = []
    for node in _top_level_imports(tree.body):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "sglang.srt.layers.quantization":
                offenders.append(node.module)

    assert offenders == []


def test_install_sglang_hooks_patches_already_imported_model_runner(monkeypatch):
    from sglang_simulator.simulation.sglang import hook_installer

    class FakeModelRunner:
        def initialize(self):
            return "original"

    fake_module = types.ModuleType("sglang.srt.model_executor.model_runner")
    fake_module.ModelRunner = FakeModelRunner
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.model_executor.model_runner",
        fake_module,
    )
    monkeypatch.setattr(hook_installer, "_HOOKS_INSTALLED", False)

    hook_installer.install_sglang_hooks()

    assert FakeModelRunner.initialize.__name__ == "override_initialize"


def test_install_sglang_hooks_patches_loaded_model_runner_when_already_installed(
    monkeypatch,
):
    from sglang_simulator.simulation.sglang import hook_installer

    class FakeModelRunner:
        def initialize(self):
            return "original"

    fake_module = types.ModuleType("sglang.srt.model_executor.model_runner")
    fake_module.ModelRunner = FakeModelRunner
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.model_executor.model_runner",
        fake_module,
    )
    monkeypatch.setattr(hook_installer, "_HOOKS_INSTALLED", True)

    hook_installer.install_sglang_hooks()

    assert FakeModelRunner.initialize.__name__ == "override_initialize"


def test_scheduler_hook_skips_tokenizer_only_during_tp_worker_init(monkeypatch):
    from sglang_simulator.simulation.sglang.scheduler import C_SchedulerHook

    calls = []

    class FakeScheduler:
        def __init__(self):
            self.server_args = types.SimpleNamespace(skip_tokenizer_init=False)

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

        def init_tp_model_worker(self):
            calls.append(self.server_args.skip_tokenizer_init)

    C_SchedulerHook.hook(FakeScheduler)
    scheduler = FakeScheduler.__new__(FakeScheduler)
    scheduler.server_args = types.SimpleNamespace(skip_tokenizer_init=False)
    scheduler.init_tp_model_worker()

    assert calls == [True]
    assert scheduler.server_args.skip_tokenizer_init is False


def test_scheduler_hook_ignores_generate_requests_without_simulation_metadata(
    monkeypatch,
):
    from sglang_simulator.simulation.sglang import scheduler as scheduler_module
    from sglang_simulator.simulation.sglang.scheduler import C_SchedulerHook
    from sglang_simulator.simulation.types import SimulationMode

    GenerateReq = type("TokenizedGenerateReqInput", (), {})
    request = GenerateReq()
    request.rid = "health-check"
    request.input_ids = [0]
    request.sampling_params = types.SimpleNamespace(
        custom_params=None,
        max_new_tokens=1,
    )

    class FakeScheduler:
        def __init__(self):
            pass

        def recv_requests(self):
            return [request]

        def get_new_batch_prefill(self):
            return None

        def run_batch(self):
            return None

        def process_batch_result(self):
            return None

        def event_loop_normal(self):
            return None

        def init_tp_model_worker(self):
            return None

    monkeypatch.setattr(scheduler_module, "_ensure_failure_manager", lambda: None)
    monkeypatch.setattr(C_SchedulerHook, "SIM_MODE", SimulationMode.BLOCKING)
    C_SchedulerHook.REQUEST_STATS.clear()

    C_SchedulerHook.hook(FakeScheduler)
    scheduler = FakeScheduler.__new__(FakeScheduler)

    assert scheduler.recv_requests() == [request]
    assert "health-check" not in C_SchedulerHook.REQUEST_STATS


def test_scheduler_hook_offline_uses_last_valid_simulation_metadata(monkeypatch):
    from sglang_simulator.simulation.sglang import scheduler as scheduler_module
    from sglang_simulator.simulation.sglang.scheduler import C_SchedulerHook
    from sglang_simulator.simulation.types import SimulationMode

    GenerateReq = type("TokenizedGenerateReqInput", (), {})
    real_request = GenerateReq()
    real_request.rid = "real-request"
    real_request.input_ids = [1, 2, 3]
    real_request.sampling_params = types.SimpleNamespace(
        custom_params={
            "simulation": {
                "created_time": 0.0,
                "total_request": 1,
                "job_id": "1",
            }
        },
        max_new_tokens=1,
    )
    internal_request = GenerateReq()
    internal_request.rid = "health-check"
    internal_request.input_ids = [0]
    internal_request.sampling_params = types.SimpleNamespace(
        custom_params=None,
        max_new_tokens=1,
    )

    class FakeScheduler:
        def __init__(self):
            pass

        def recv_requests(self):
            return [real_request, internal_request]

        def get_new_batch_prefill(self):
            return None

        def run_batch(self):
            return None

        def process_batch_result(self):
            return None

        def event_loop_normal(self):
            return None

        def init_tp_model_worker(self):
            return None

    monkeypatch.setattr(scheduler_module, "_ensure_failure_manager", lambda: None)
    monkeypatch.setattr(C_SchedulerHook, "SIM_MODE", SimulationMode.OFFLINE)
    C_SchedulerHook.REQUEST_STATS.clear()
    C_SchedulerHook.FUTURE_QUEUE.clear()
    C_SchedulerHook.WORKER_PENDING.clear()
    C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST = False

    C_SchedulerHook.hook(FakeScheduler)
    scheduler = FakeScheduler.__new__(FakeScheduler)

    assert scheduler.recv_requests() == [internal_request]
    assert C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST is True
    assert "health-check" in C_SchedulerHook.IGNORED_REQUEST_RIDS
