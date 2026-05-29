import sys
from types import ModuleType


def test_sim_run_scheduler_process_calls_install_first(monkeypatch):
    import sglang_simulator.simulation.sglang.hook_installer as hook_installer
    import sglang_simulator.simulation.sglang.subprocess_entry as subprocess_entry

    order = []

    monkeypatch.setattr(
        hook_installer,
        "install_sglang_hooks",
        lambda: order.append("install"),
    )

    scheduler_module = ModuleType("sglang.srt.managers.scheduler")
    scheduler_module.run_scheduler_process = lambda *args, **kwargs: order.append(
        "run"
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.scheduler", scheduler_module)

    subprocess_entry.sim_run_scheduler_process(1, kw=2)

    assert order == ["install", "run"]
