from __future__ import annotations

import importlib.util
import logging
from pathlib import Path


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


def test_remote_backup_server_logs_backup_and_prefetch(caplog):
    module = _load_remote_backup_server_module()
    server = module.RemoteBackupServer(port=0, max_buffer_size_gb=1.0, page_size=1)

    caplog.set_level(logging.INFO, logger=module.__name__)

    results = server.insert_pages(
        dp_rank=1,
        token_ids=[101, 102, 103],
        page_start=1,
        pages=[b"page-102", b"page-103"],
    )
    assert results == [True, True]

    data_count, navigable = server.match_prefix_from(
        dp_rank=1,
        token_ids=[101, 102, 103],
        start_page=1,
    )
    assert (data_count, navigable) == (2, 2)

    pages = server.get_pages_by_tokens(
        dp_rank=1,
        token_ids=[101, 102, 103],
        start_page=1,
        count=2,
    )
    assert pages == [b"page-102", b"page-103"]

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "REMOTE_BACKUP backup_put" in messages
    assert "dp_rank=1" in messages
    assert "page_start=1" in messages
    assert "ok=2/2" in messages
    assert "REMOTE_BACKUP prefetch_match" in messages
    assert "data_pages=2" in messages
    assert "REMOTE_BACKUP prefetch_get" in messages
    assert "hit_pages=2/2" in messages
