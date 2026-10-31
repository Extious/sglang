"""Experiment configuration, loaded from a JSON file."""

from __future__ import annotations

from typing import Dict, List, Optional

import msgspec


class ExperimentConfig(msgspec.Struct, kw_only=True):
    name: str
    policy: str
    model_path: str
    out_dir: str

    # Workers: one TP=1 SGLang server per GPU.
    gpus: List[int]
    base_port: int = 31000
    mem_fraction_static: float = 0.85
    context_length: int = 40960
    chunked_prefill_size: int = 8192
    extra_server_args: List[str] = []
    python: str = "python"
    worker_env: Dict[str, str] = {}

    # Closed-loop workflow workload.
    num_workflows: int = 32
    concurrency: int = 8
    input_len: int = 16384
    output_lens: Dict[str, int] = msgspec.field(
        default_factory=lambda: {
            "A": 256,
            "B": 256,
            "C": 768,
            "D": 256,
            "E": 256,
            "F": 256,
        }
    )
    seed: int = 0
    stagger_s: float = 1.0

    # One fail-stop Worker failure; None disables it.
    fail_worker: Optional[int] = None
    fail_at_s: Optional[float] = None
    rejoin_after_s: float = 60.0

    # Backup capacity per holder Worker, in token slots (all layers per slot).
    dram_backup_tokens: int = 200000
    staging_tokens: int = 200000
    hbm_quota_tokens: int = 200000
    hbm_reserve_fraction: float = 0.05
    checkpoint_interval: int = 64
    poll_interval_s: float = 0.05

    # AegisKV inputs produced offline (profile.py, train_tracker.py).
    profile_path: Optional[str] = None
    tracker_path: Optional[str] = None
    resident_layers: int = 4

    @classmethod
    def load(cls, path: str) -> ExperimentConfig:
        with open(path, "rb") as f:
            return msgspec.json.decode(f.read(), type=cls)

    def dump(self, path: str) -> None:
        with open(path, "wb") as f:
            f.write(msgspec.json.format(msgspec.json.encode(self)))
