#!/usr/bin/env python3
"""
Standalone remote backup KV Cache server launcher.

This process maintains a global radix buffer in DRAM and accepts TCP connections
from SGLang worker nodes. It runs independently of the SGLang server processes.

Usage:
  python launch_remote_backup_server.py --port 30000 --buffer-size-gb 32 --page-size 1
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

# Add sglang to path — parents[4] is the python/ directory inside the repo.
# PYTHONPATH must contain the python/ dir so that "import sglang ..." resolves.
_repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_repo_root))

# Import the server module directly, bypassing the package __init__.py which
# pulls in the full storage backend chain (torch, flashinfer, etc.) that
# requires CUDA and a writable filesystem.
import importlib.util

_self_dir = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "remote_backup_server",
    _self_dir / "remote_backup_server.py",
)
_remote_backup_mod = importlib.util.module_from_spec(_spec)
# Must register in sys.modules BEFORE exec_module so that @dataclass decorators
# can resolve the module via sys.modules[cls.__module__].
sys.modules[_spec.name] = _remote_backup_mod
_spec.loader.exec_module(_remote_backup_mod)
RemoteBackupServer = _remote_backup_mod.RemoteBackupServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Remote Backup KV Cache Server")
    parser.add_argument(
        "--port", type=int, default=30000,
        help="TCP port to listen on (default: 30000)"
    )
    parser.add_argument(
        "--buffer-size-gb", type=float, default=32.0,
        help="Radix buffer size in GB (default: 32.0)"
    )
    parser.add_argument(
        "--page-size", type=int, default=1,
        help="KV page size in tokens (default: 1)"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info(
        "Starting Remote Backup KV Cache Server: port=%d, buffer_size_gb=%.1f, page_size=%d",
        args.port, args.buffer_size_gb, args.page_size
    )

    server = RemoteBackupServer(
        port=args.port,
        max_buffer_size_gb=args.buffer_size_gb,
        page_size=args.page_size,
    )
    server.start()

    def shutdown(signum, frame):
        logger.info("Shutting down remote backup server...")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    logger.info("Remote Backup KV Cache Server is running. Press Ctrl+C to stop.")
    try:
        signal.pause()
    except AttributeError:
        # Windows doesn't have signal.pause()
        import time
        while True:
            time.sleep(1)


if __name__ == "__main__":
    main()
