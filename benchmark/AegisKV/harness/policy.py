"""Backup policies. Each policy only decides placement through Master primitives:
`protect_dram`, `unprotect_dram`, `add_region`, `drop_region`.

A policy named `<name>` other than `rebuild` lives in `harness/<name>_policy.py`
and exposes `make(cfg) -> Policy`, so each baseline branch adds one module.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict


class Policy:
    name = "base"

    async def tick(self, master: Any) -> None:
        """Called every poll interval with the Master's live state."""

    def describe(self, call: Any) -> Dict[str, Any]:
        prot = call.prot
        if prot is None:
            return {"protected": False}
        return {
            "protected": True,
            "dram_committed": prot.dram_committed,
            "max_tokens": prot.max_tokens,
            "region_layers": prot.region_layers,
            "region_committed": prot.region_committed,
        }


class RebuildPolicy(Policy):
    """No backups: a takeover re-prefills the whole accepted history."""

    name = "rebuild"


def make_policy(name: str, cfg: Any) -> Policy:
    if name == "rebuild":
        return RebuildPolicy()
    return importlib.import_module(f"harness.{name}_policy").make(cfg)
