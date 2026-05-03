from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


PYTHON_ROOT = Path(__file__).resolve().parents[3] / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


def _req(**overrides):
    data = {
        "prefix_indices": [0] * 100,
        "host_hit_length": 0,
        "storage_hit_length": 0,
        "storage_query_tokens": 0,
        "cached_tokens_device": 0,
        "cached_tokens_host": 0,
        "cached_tokens_storage": 0,
        "reused_tokens_device": 0,
        "reused_tokens_host": 0,
        "reused_tokens_storage": 0,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


class _BoolAmbiguousSeq:
    def __init__(self, length: int) -> None:
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __bool__(self) -> bool:
        raise RuntimeError("Boolean value is ambiguous")


def test_remote_prefetch_breakdown_handles_tensor_like_prefix_indices():
    from sglang.srt.managers.cache_detail_utils import apply_cache_source_breakdown

    req = _req(prefix_indices=_BoolAmbiguousSeq(0))

    apply_cache_source_breakdown(req)

    assert req.cached_tokens_device == 0


def test_remote_prefetch_breakdown_keeps_expected_and_completed_tokens():
    from sglang.srt.managers.cache_detail_utils import apply_cache_source_breakdown

    req = _req(
        prefix_indices=[0] * 5389,
        host_hit_length=4992,
        storage_hit_length=4992,
        storage_query_tokens=5061,
    )

    apply_cache_source_breakdown(req)

    assert req.cached_tokens_device == 397
    assert req.cached_tokens_host == 0
    assert req.cached_tokens_storage == 4992
    assert req.reused_tokens_host == 0
    assert req.reused_tokens_storage == 4992
    assert req.storage_query_tokens == 5061


def test_remote_prefetch_completed_can_exceed_reused_tokens():
    from sglang.srt.managers.cache_detail_utils import apply_cache_source_breakdown

    req = _req(
        prefix_indices=[0] * 5445,
        host_hit_length=5045,
        storage_hit_length=5048,
    )

    apply_cache_source_breakdown(req)

    assert req.cached_tokens_device == 400
    assert req.cached_tokens_host == 0
    assert req.cached_tokens_storage == 5048
    assert req.reused_tokens_storage == 5045


def test_cache_details_include_remote_query_even_without_reuse():
    from sglang.srt.managers.cache_detail_utils import (
        build_cached_tokens_details,
    )

    req = _req(
        prefix_indices=[],
        storage_query_tokens=5121,
        cached_tokens=0,
    )

    details = build_cached_tokens_details(
        req,
        enable_hicache_storage=True,
        storage_backend_type="RemoteBackupStorage",
    )

    assert details is not None
    assert details["storage_query"] == 5121
    assert details["storage"] == 0


def test_remote_prefetch_stats_survive_later_empty_scheduler_pop():
    from sglang.srt.managers.cache_detail_utils import (
        apply_cache_source_breakdown,
        build_cached_tokens_details,
        record_remote_prefetch_stats,
    )

    req = _req(
        prefix_indices=[0] * 5454,
        host_hit_length=5057,
    )

    record_remote_prefetch_stats(req, completed_tokens=5057, expected_tokens=5057)
    record_remote_prefetch_stats(req, completed_tokens=0, expected_tokens=0)
    apply_cache_source_breakdown(req)

    details = build_cached_tokens_details(
        req,
        enable_hicache_storage=True,
        storage_backend_type="RemoteBackupStorage",
    )

    assert details is not None
    assert details["device"] == 397
    assert details["host"] == 0
    assert details["storage_query"] == 5057
    assert details["storage"] == 5057
    assert details["reused_host"] == 0
    assert details["reused_storage"] == 5057


def test_remote_query_survives_best_effort_prefetch_with_no_completed_tokens():
    from sglang.srt.managers.cache_detail_utils import (
        apply_cache_source_breakdown,
        build_cached_tokens_details,
        record_remote_prefetch_stats,
    )

    req = _req(
        prefix_indices=[0] * 406,
        host_hit_length=0,
    )

    record_remote_prefetch_stats(req, completed_tokens=0, expected_tokens=5117)
    record_remote_prefetch_stats(req, completed_tokens=0, expected_tokens=0)
    apply_cache_source_breakdown(req)

    details = build_cached_tokens_details(
        req,
        enable_hicache_storage=True,
        storage_backend_type="RemoteBackupStorage",
    )

    assert details is not None
    assert details["device"] == 406
    assert details["host"] == 0
    assert details["storage_query"] == 5117
    assert details["storage"] == 0
    assert details["reused_storage"] == 0
