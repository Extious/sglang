import argparse
import dataclasses
import faulthandler
import os
import sys
from typing import Optional


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
    stack_path = os.path.join(output_dir, "server_stack.log")
    stack_file = open(stack_path, "a", buffering=1)
    faulthandler.enable(file=stack_file, all_threads=True)
    faulthandler.dump_traceback_later(
        interval_s,
        repeat=True,
        file=stack_file,
        exit=False,
    )


_maybe_enable_stack_dump()

from sglang_simulator.simulation.sglang.subprocess_entry import sim_run_scheduler_process
from sglang_simulator.utils import get_logger

logger = get_logger("sgl_simulator")


@dataclasses.dataclass
class SimulationArgs:
    sim_config_path: Optional[str] = None

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--sim-config-path",
            type=str,
            default=None,
            help="Path to simulation JSON config (same as SGLANG_SIMULATOR_CONFIG_PATH).",
        )

    @classmethod
    def from_cli_args(cls, ns: argparse.Namespace) -> "SimulationArgs":
        return SimulationArgs(sim_config_path=ns.sim_config_path)


# Ref: https://github.com/sgl-project/sglang/blob/v0.5.6.post2/python/sglang/launch_server.py
if __name__ == "__main__":
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import kill_process_tree

    parser = argparse.ArgumentParser()

    g = parser.add_argument_group("sglang")
    ServerArgs.add_cli_args(g)

    g = parser.add_argument_group("simulation")
    SimulationArgs.add_cli_args(g)

    raw_args = parser.parse_args(sys.argv[1:])
    server_args = ServerArgs.from_cli_args(raw_args)
    simulation_args = SimulationArgs.from_cli_args(raw_args)

    config_path = os.getenv("SGLANG_SIMULATOR_CONFIG_PATH")
    if config_path and os.path.exists(config_path):
        logger.info(f"Using config from {config_path}")
    elif simulation_args.sim_config_path:
        os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = simulation_args.sim_config_path

    try:
        launch_server(
            server_args,
            run_scheduler_process_func=sim_run_scheduler_process,
        )
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
