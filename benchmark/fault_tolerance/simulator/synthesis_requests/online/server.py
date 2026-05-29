from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from benchmark.fault_tolerance.simulator.synthesis_requests.common.schema import (
    ExperimentSuite,
)

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SIMULATOR_SRC = _REPO_ROOT / "tools" / "sglang-simulator" / "src"
_SGLANG_PYTHON = _REPO_ROOT / "python"


def build_server_command(*, suite: ExperimentSuite, sim_config_path: Path) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "sglang_simulator.simulation.sglang.launch_server",
        "--model-path",
        suite.server.model_path,
        "--load-format",
        "dummy",
        "--host",
        suite.server.host,
        "--port",
        str(suite.server.port),
        "--tp-size",
        str(suite.server.tp_size),
        "--pp-size",
        str(suite.server.pp_size),
        "--dp-size",
        str(suite.server.dp_size),
        "--enable-hierarchical-cache",
        "--hicache-storage-backend",
        "file",
        "--skip-server-warmup",
        "--skip-tokenizer-init",
        "--sim-config-path",
        str(sim_config_path),
    ]


def build_server_env(
    *,
    sim_config_path: Path,
    simulator_raw_dir: Path,
) -> dict[str, str]:
    env = os.environ.copy()
    env["SGLANG_SIMULATOR_CONFIG_PATH"] = str(sim_config_path)
    env["SGLANG_SIMULATOR_OUTPUT_DIR"] = str(simulator_raw_dir)
    env["SGLANG_SIMULATOR_OUTPUT_MODE"] = "BLOCKING"
    env["SGLANG_SIMULATOR_SKIP_REMOTE_QUANT_CONFIG"] = "1"
    env["SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION"] = "0"
    env.setdefault("SGLANG_SIMULATOR_STACK_DUMP_INTERVAL", "0")
    env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    env["PYTHONPATH"] = _prepend_pythonpath(
        env.get("PYTHONPATH", ""),
        [str(_SIMULATOR_SRC), str(_SGLANG_PYTHON), str(_REPO_ROOT)],
    )
    env.pop("SGLANG_USE_CPU_ENGINE", None)
    return env


def _prepend_pythonpath(current: str, paths: list[str]) -> str:
    entries = [entry for entry in current.split(os.pathsep) if entry]
    for path in reversed(paths):
        if path in entries:
            entries.remove(path)
        entries.insert(0, path)
    return os.pathsep.join(entries)


def start_server(
    *,
    suite: ExperimentSuite,
    sim_config_path: Path,
    simulator_raw_dir: Path,
) -> subprocess.Popen:
    simulator_raw_dir.mkdir(parents=True, exist_ok=True)
    log_file = (simulator_raw_dir / "server.log").open("w", encoding="utf-8")
    return subprocess.Popen(
        build_server_command(suite=suite, sim_config_path=sim_config_path),
        env=build_server_env(
            sim_config_path=sim_config_path,
            simulator_raw_dir=simulator_raw_dir,
        ),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
