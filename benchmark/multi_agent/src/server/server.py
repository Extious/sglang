# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.

"""
Server lifecycle: submit, wait for readiness, yield, cleanup.

The server can be used as a context manager so the caller never has to
remember to cancel the SLURM job::

    with ServerRunner.start(cfg, slurm_script) as srv:
        print(srv.head_url)   # http://node0:28000
        print(srv.manifest)   # parsed stage_manifest.json
    # SLURM job is cancelled automatically on exit

Consumers (client, injector) only see server_url + stage_manifest.
They know nothing about SLURM or deployment details.
"""

from __future__ import annotations

import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import json


@dataclass
class ServerHandle:
    """Artifacts produced by a running server."""
    head_url: str
    manifest_path: Path
    slurm_job_id: str
    log_dir: Path

    @property
    def manifest(self) -> dict:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    @property
    def kv_backup(self) -> str:
        return self.manifest.get("kv_backup", "none")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_sbatch(stdout: str) -> str:
    m = re.search(r"Submitted batch job (\d+)", stdout)
    return m.group(1) if m else ""


def _slurm_state(job_id: str) -> str:
    r = subprocess.run(
        ["squeue", "-j", job_id, "-h", "-o", "%T"],
        capture_output=True, text=True,
    )
    return (r.stdout or "").strip()


# squeue -o "%T" uses compact state codes (e.g. R, PD). Never treat empty as terminal:
# the job may not appear in squeue for a short window right after sbatch.
_SLURM_FAILED_STATES = frozenset({
    "FAILED", "F", "CANCELLED", "CA", "CD", "COMPLETED",
    "TIMEOUT", "TO", "NODE_FAIL", "NF", "OUT_OF_MEMORY", "OM",
    "BOOT_FAIL", "BF", "DEADLINE", "DL", "STOPPED", "ST",
})


def _cancel(job_id: str) -> None:
    if not job_id:
        return
    subprocess.run(["squeue", "-j", job_id, "-h"], capture_output=True)
    subprocess.run(["scancel", job_id], check=False, capture_output=True)
    time.sleep(3)


# ---------------------------------------------------------------------------
# ServerRunner
# ---------------------------------------------------------------------------

class ServerRunner:
    """
    Manages the full server lifecycle within a SLURM job.

    Only cares about server_url.txt + stage_manifest.json appearing on disk.
    Does not know what backup strategy is configured (kv_backup is read
    from the manifest, not hardcoded).
    """

    def __init__(
        self,
        slurm_script: Path,
        server_flags: list[str],
        log_dir: Path,
        server_port: int = 28000,
        backup_port: int = 29000,
    ) -> None:
        self.slurm_script = slurm_script
        self.server_flags = server_flags
        self.log_dir = log_dir
        self.server_port = server_port
        self.backup_port = backup_port

        self._job_id: str = ""
        self._handle: Optional[ServerHandle] = None
        self._cancelled = False

    # ------------------------------------------------------------------
    # Public entry point (mirrors subprocess.Popen)
    # ------------------------------------------------------------------

    @classmethod
    def start(
        cls,
        slurm_script: Path,
        server_flags: list[str],
        log_dir: Path,
        server_port: int = 28000,
        backup_port: int = 29000,
    ) -> ServerRunner:
        inst = cls(slurm_script, server_flags, log_dir, server_port, backup_port)
        inst._submit(register_signals=True)
        inst._wait_ready()
        return inst

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _submit(self, register_signals: bool = True) -> None:
        log_dir = self.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        url_file = log_dir / "server_url.txt"
        manifest_file = log_dir / "stage_manifest.json"
        for p in (url_file, manifest_file):
            p.unlink(missing_ok=True)

        r = subprocess.run(
            ["sbatch", str(self.slurm_script), *self.server_flags],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            sys.exit(f"sbatch failed:\n{r.stdout}\n{r.stderr}")
        self._job_id = _parse_sbatch(r.stdout + r.stderr)
        if not self._job_id:
            sys.exit(f"Could not parse SLURM job id from:\n{r.stdout}\n{r.stderr}")

        print(f"[ServerRunner] SLURM job submitted: {self._job_id}", flush=True)

        if register_signals:
            def _cancel_on_signal(_sig: int, _frame) -> None:
                print(f"\n[ServerRunner] Signal caught, cancelling job {self._job_id}...", flush=True)
                _cancel(self._job_id)
                raise SystemExit(130)
            signal.signal(signal.SIGINT, _cancel_on_signal)
            signal.signal(signal.SIGTERM, _cancel_on_signal)

    def submit_and_wait_ready(self, *, register_signals: bool = False) -> ServerHandle:
        """Submit server via sbatch and block until URL/manifest exist (or fatal Slurm state)."""
        self._submit(register_signals=register_signals)
        self._wait_ready()
        if self._handle is None:
            sys.exit("ServerRunner: internal error, handle not set after wait.")
        return self._handle

    def _wait_ready(self, timeout_s: int = 1200) -> None:
        job_id = self._job_id
        url_file = self.log_dir / "server_url.txt"
        manifest_file = self.log_dir / "stage_manifest.json"

        start = time.time()
        while time.time() - start < timeout_s:
            if (
                url_file.is_file()
                and url_file.stat().st_size > 0
                and manifest_file.is_file()
                and manifest_file.stat().st_size > 0
            ):
                self._handle = ServerHandle(
                    head_url=url_file.read_text(encoding="utf-8").strip(),
                    manifest_path=manifest_file,
                    slurm_job_id=job_id,
                    log_dir=self.log_dir,
                )
                (self.log_dir / "slurm_job_id.txt").write_text(job_id + "\n", encoding="utf-8")
                print(
                    f"[ServerRunner] Server ready at {self._handle.head_url}", flush=True,
                )
                return

            state = _slurm_state(job_id)
            if state and state in _SLURM_FAILED_STATES:
                sys.exit(
                    f"SLURM job {job_id} ended (state={state!r}) before server artifacts were ready."
                )
            time.sleep(5)

        _cancel(job_id)
        sys.exit(f"Timeout after {timeout_s}s waiting for server artifacts.")

    def close(self) -> None:
        """Synchronously cancel the SLURM job and clean up."""
        if self._cancelled:
            return
        self._cancelled = True
        _cancel(self._job_id)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> ServerHandle:
        if self._handle is None:
            sys.exit("ServerRunner.start() must be called before entering context")
        return self._handle

    def __exit__(self, *_) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Simple CLI: deploy and block (used by run_server*.slurm wrapper)
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="Server lifecycle CLI")
    p.add_argument("--log-dir", type=Path, default=Path("logs/server"))
    p.add_argument("--slurm-script", type=Path, required=True)
    p.add_argument("server_flags", nargs="*", default=[])
    args = p.parse_args()

    with ServerRunner.start(
        slurm_script=args.slurm_script,
        server_flags=args.server_flags,
        log_dir=args.log_dir,
    ) as srv:
        print(f"Server head: {srv.head_url}")
        print(f"Manifest:   {srv.manifest_path}")
        print(f"Blocking until interrupted...")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
