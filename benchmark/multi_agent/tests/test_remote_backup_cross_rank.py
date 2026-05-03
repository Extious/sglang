from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


PYTHON_ROOT = Path(__file__).resolve().parents[3] / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


def _load_remote_backup_server_module():
    path = (
        Path(__file__).resolve().parents[3]
        / "python"
        / "sglang"
        / "srt"
        / "mem_cache"
        / "storage"
        / "remote_backup"
        / "remote_backup_server.py"
    )
    spec = importlib.util.spec_from_file_location("remote_backup_server_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_remote_backup_storage_module(monkeypatch, server_module):
    for package_name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.mem_cache",
        "sglang.srt.mem_cache.storage",
        "sglang.srt.mem_cache.storage.remote_backup",
    ):
        package = types.ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)

    hicache_storage = types.ModuleType("sglang.srt.mem_cache.hicache_storage")
    hicache_storage.HiCacheStorage = object
    hicache_storage.HiCacheStorageConfig = object
    hicache_storage.HiCacheStorageExtraInfo = object
    monkeypatch.setitem(
        sys.modules, "sglang.srt.mem_cache.hicache_storage", hicache_storage
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.mem_cache.storage.remote_backup.remote_backup_server",
        server_module,
    )

    path = (
        Path(__file__).resolve().parents[3]
        / "python"
        / "sglang"
        / "srt"
        / "mem_cache"
        / "storage"
        / "remote_backup"
        / "remote_backup_storage.py"
    )
    spec = importlib.util.spec_from_file_location("remote_backup_storage_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_remote_backup_match_and_get_can_use_pages_from_any_dp_rank():
    module = _load_remote_backup_server_module()
    server = module.RemoteBackupServer(port=0, max_buffer_size_gb=1.0, page_size=1)

    server.insert_pages(
        dp_rank=1,
        token_ids=[10, 20, 30],
        page_start=0,
        pages=[b"rank-1-page-10", b"rank-1-page-20", b"rank-1-page-30"],
    )

    source_rank, data_count, navigable = server.match_prefix_from_with_source(
        module.REMOTE_BACKUP_ALL_DP_RANKS,
        token_ids=[10, 20, 30],
        start_page=0,
    )

    assert source_rank == 1
    assert (data_count, navigable) == (3, 3)
    assert server.get_pages_by_tokens(
        module.REMOTE_BACKUP_ALL_DP_RANKS,
        token_ids=[10, 20, 30],
        start_page=0,
        count=3,
    ) == [b"rank-1-page-10", b"rank-1-page-20", b"rank-1-page-30"]


def test_remote_backup_match_data_count_stops_at_first_missing_data_page():
    module = _load_remote_backup_server_module()
    server = module.RemoteBackupServer(port=0, max_buffer_size_gb=1.0, page_size=1)

    server.insert_pages(
        dp_rank=1,
        token_ids=[10, 20, 30],
        page_start=1,
        pages=[b"rank-1-page-20", b"rank-1-page-30"],
    )

    source_rank, data_count, navigable = server.match_prefix_from_with_source(
        module.REMOTE_BACKUP_ALL_DP_RANKS,
        token_ids=[10, 20, 30],
        start_page=0,
    )
    assert source_rank == 1
    assert (data_count, navigable) == (0, 3)

    source_rank, data_count, navigable = server.match_prefix_from_with_source(
        module.REMOTE_BACKUP_ALL_DP_RANKS,
        token_ids=[10, 20, 30],
        start_page=1,
    )
    assert source_rank == 1
    assert (data_count, navigable) == (2, 2)


def test_remote_backup_all_rank_get_can_combine_pages_from_multiple_dp_ranks():
    module = _load_remote_backup_server_module()
    server = module.RemoteBackupServer(port=0, max_buffer_size_gb=1.0, page_size=1)

    server.insert_pages(
        dp_rank=0,
        token_ids=[10, 20],
        page_start=0,
        pages=[b"rank-0-page-10"],
    )
    server.insert_pages(
        dp_rank=1,
        token_ids=[10, 20],
        page_start=1,
        pages=[b"rank-1-page-20"],
    )

    _source_rank, data_count, navigable = server.match_prefix_from_with_source(
        module.REMOTE_BACKUP_ALL_DP_RANKS,
        token_ids=[10, 20],
        start_page=0,
    )

    assert (data_count, navigable) == (2, 2)
    assert server.get_pages_by_tokens(
        module.REMOTE_BACKUP_ALL_DP_RANKS,
        token_ids=[10, 20],
        start_page=0,
        count=2,
    ) == [b"rank-0-page-10", b"rank-1-page-20"]


def test_remote_backup_storage_read_lookup_uses_all_dp_ranks(monkeypatch):
    server_module = _load_remote_backup_server_module()
    storage_module = _load_remote_backup_storage_module(monkeypatch, server_module)

    storage = storage_module.RemoteBackupStorage.__new__(
        storage_module.RemoteBackupStorage
    )
    storage._dp_rank = 0

    assert storage._resolve_lookup_dp_rank(None) == server_module.REMOTE_BACKUP_ALL_DP_RANKS
    assert storage._resolve_lookup_dp_rank(1) == server_module.REMOTE_BACKUP_ALL_DP_RANKS
