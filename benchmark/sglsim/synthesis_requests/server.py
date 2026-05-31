from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

from .utils import ServerConfig, read_log_tail

logger = logging.getLogger("sglang.benchmark.sglsim.synthesis_requests.server")


def build_server_command(
    *,
    server: ServerConfig,
    log_level: str = "info",
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "sglang.launch_server",
        "--model-path",
        server.model_path,
        "--host",
        server.host,
        "--port",
        str(server.port),
        "--tp",
        str(server.tp_size),
        "--dp-size",
        str(server.dp_size),
        "--pp-size",
        str(server.pp_size),
        "--attention-backend",
        server.attention_backend,
        "--sampling-backend",
        "pytorch",
        "--enable-metrics",
        "--log-level",
        log_level,
    ]
    command.extend(server.extra_args)
    return command


def _server_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("SGLANG_DISABLE_CUDNN_CHECK", "1")
    return env


def _tee_output(process: subprocess.Popen, log_path: Path) -> threading.Thread:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")

    def _reader() -> None:
        if process.stdout is None:
            return
        try:
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                sys.stderr.write(line)
                sys.stderr.flush()
        finally:
            log_file.close()

    thread = threading.Thread(target=_reader, name="server-log-tee", daemon=True)
    thread.start()
    return thread


def start_server(command: list[str], log_path: Path) -> subprocess.Popen:
    logger.info("Starting SGLang server: %s", " ".join(command))
    logger.info("Server log file: %s", log_path)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=_server_env(),
    )
    _tee_output(process, log_path)
    return process


def stop_server(process: subprocess.Popen, log_path: Path | None = None) -> None:
    if process.poll() is not None:
        logger.info("Server subprocess exited with code %s", process.returncode)
        tail = read_log_tail(log_path)
        if tail:
            logger.info("Last server log lines:\n%s", tail)
        return
    logger.info("Terminating server subprocess")
    process.terminate()
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    logger.info("Server subprocess stopped with code %s", process.returncode)
    tail = read_log_tail(log_path)
    if tail and process.returncode != 0:
        logger.error("Last server log lines:\n%s", tail)
