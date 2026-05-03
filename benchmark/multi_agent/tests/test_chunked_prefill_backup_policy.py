from __future__ import annotations

import sys
from pathlib import Path


PYTHON_ROOT = Path(__file__).resolve().parents[3] / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


def test_chunked_prefill_counts_for_write_through_backup():
    from sglang.srt.mem_cache.hicache_backup_policy import should_count_backup_hit

    assert should_count_backup_hit("write_through", chunked=True)
    assert should_count_backup_hit("write_through_selective", chunked=True)


def test_write_back_still_skips_backup_hit_count_for_chunked_prefill():
    from sglang.srt.mem_cache.hicache_backup_policy import should_count_backup_hit

    assert not should_count_backup_hit("write_back", chunked=True)
