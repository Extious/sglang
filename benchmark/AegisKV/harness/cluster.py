"""Worker processes and their peer-DRAM arenas.

Each Worker is an independent `sglang.launch_server` process (TP=1) started in
its own process group, so a fail-stop failure is a SIGKILL of that group. Each
Worker owns one /dev/shm arena: other Workers write checkpoints into it, and it
reads them back on takeover. A failed Worker's arena is deleted with it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import time
from typing import Dict, List, Optional

import aiohttp
from harness.config import ExperimentConfig

logger = logging.getLogger(__name__)

# MiniCPM4.1-8B: 32 layers x K/V x 2 KV heads x 128 dims x bf16.
SLOT_BYTES = 32 * 2 * 2 * 128 * 2


class Worker:
    def __init__(self, *, index: int, gpu: int, port: int, run_tag: str):
        self.index = index
        self.gpu = gpu
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self.run_tag = run_tag
        self.generation = 0
        self.proc: Optional[subprocess.Popen] = None
        self.alive = False
        self.arena_path = ""
        self.log_path = ""

    def next_arena_path(self) -> str:
        self.generation += 1
        return f"/dev/shm/aegiskv_{self.run_tag}_w{self.index}_g{self.generation}"


class Cluster:
    def __init__(self, cfg: ExperimentConfig, *, run_tag: str):
        self.cfg = cfg
        self.run_tag = run_tag
        self.arena_slots = cfg.dram_backup_tokens + cfg.staging_tokens
        self.workers: List[Worker] = [
            Worker(index=i, gpu=gpu, port=cfg.base_port + i, run_tag=run_tag)
            for i, gpu in enumerate(cfg.gpus)
        ]

    def _command(self, worker: Worker) -> List[str]:
        cfg = self.cfg
        return [
            cfg.python,
            "-m",
            "sglang.launch_server",
            "--model-path",
            cfg.model_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(worker.port),
            "--enable-aegiskv",
            "--page-size",
            "1",
            "--mem-fraction-static",
            str(cfg.mem_fraction_static),
            "--context-length",
            str(cfg.context_length),
            "--chunked-prefill-size",
            str(cfg.chunked_prefill_size),
            "--log-level",
            "warning",
            *cfg.extra_server_args,
        ]

    def _create_arena(self, worker: Worker) -> None:
        worker.arena_path = worker.next_arena_path()
        with open(worker.arena_path, "wb") as f:
            f.truncate(self.arena_slots * SLOT_BYTES)

    def _remove_arena(self, worker: Worker) -> None:
        if worker.arena_path and os.path.exists(worker.arena_path):
            os.unlink(worker.arena_path)

    def start(self, worker: Worker) -> None:
        self._create_arena(worker)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(worker.gpu))
        # JIT kernels invoke `ninja`; take it from the interpreter's environment.
        bin_dir = os.path.dirname(os.path.abspath(self.cfg.python))
        env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
        env.update(self.cfg.worker_env)
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        worker.log_path = os.path.join(
            self.cfg.out_dir, f"worker{worker.index}_g{worker.generation}.log"
        )
        log = open(worker.log_path, "w")
        worker.proc = subprocess.Popen(
            self._command(worker),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def kill(self, worker: Worker) -> None:
        """Fail-stop: kill the whole process group and lose the Worker's arena."""
        worker.alive = False
        if worker.proc is not None and worker.proc.poll() is None:
            os.killpg(worker.proc.pid, signal.SIGKILL)
            worker.proc.wait()
        self._remove_arena(worker)

    def shutdown(self) -> None:
        for worker in self.workers:
            self.kill(worker)

    async def wait_ready(self, worker: Worker, timeout_s: float = 1800.0) -> None:
        deadline = time.monotonic() + timeout_s
        async with aiohttp.ClientSession() as session:
            while time.monotonic() < deadline:
                if worker.proc is not None and worker.proc.poll() is not None:
                    raise RuntimeError(
                        f"Worker {worker.index} exited; see {worker.log_path}"
                    )
                try:
                    async with session.get(
                        f"{worker.url}/health_generate", timeout=5
                    ) as resp:
                        if resp.status == 200:
                            worker.alive = True
                            return
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    pass
                await asyncio.sleep(2.0)
        raise TimeoutError(f"Worker {worker.index} not ready; see {worker.log_path}")

    async def start_all(self) -> None:
        for worker in self.workers:
            self.start(worker)
        await asyncio.gather(*(self.wait_ready(w) for w in self.workers))

    def arenas(self) -> Dict[int, str]:
        return {w.index: w.arena_path for w in self.workers if w.alive}
