"""Subprocess entry points for SGLang simulator hooks.

SGLang scheduler workers are spawned with multiprocessing "spawn", so hooks must
be installed in each child process before SGLang model classes are imported.
"""

import faulthandler
import os


def _maybe_enable_stack_dump() -> None:
    interval = os.getenv("SGLANG_SIMULATOR_STACK_DUMP_INTERVAL")
    if not interval:
        return

    try:
        interval_s = int(interval)
    except ValueError:
        return
    if interval_s <= 0:
        return

    output_dir = os.getenv("SGLANG_SIMULATOR_OUTPUT_DIR", "/tmp/sglang_simulator")
    os.makedirs(output_dir, exist_ok=True)
    stack_path = os.path.join(output_dir, f"scheduler_stack_{os.getpid()}.log")
    stack_file = open(stack_path, "a", buffering=1)
    faulthandler.enable(file=stack_file, all_threads=True)
    faulthandler.dump_traceback_later(
        interval_s,
        repeat=True,
        file=stack_file,
        exit=False,
    )


def sim_run_scheduler_process(*args, **kwargs):
    _maybe_enable_stack_dump()

    from sglang_simulator.simulation.sglang.hook_installer import install_sglang_hooks

    install_sglang_hooks()
    from sglang.srt.managers.scheduler import run_scheduler_process

    return run_scheduler_process(*args, **kwargs)
