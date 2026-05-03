from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


PYTHON_ROOT = Path(__file__).resolve().parents[3] / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

for package_name, package_path in (
    ("sglang", PYTHON_ROOT / "sglang"),
    ("sglang.srt", PYTHON_ROOT / "sglang" / "srt"),
):
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_path)]
    sys.modules.setdefault(package_name, package)

if "tqdm" not in sys.modules:
    tqdm_stub = types.ModuleType("tqdm")
    tqdm_stub.tqdm = lambda iterable=None, *args, **kwargs: iterable
    sys.modules["tqdm"] = tqdm_stub


def _stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _install_cache_controller_dependency_stubs() -> None:
    if "sglang.srt.managers.cache_controller" in sys.modules:
        return

    torch_stub = types.ModuleType("torch")
    torch_stub.distributed = SimpleNamespace(
        ProcessGroup=object,
        ReduceOp=SimpleNamespace(MAX="max", MIN="min"),
        all_reduce=lambda *args, **kwargs: None,
    )
    torch_stub.int = int
    torch_stub.tensor = lambda value, dtype=None: SimpleNamespace(
        item=lambda: value[0] if isinstance(value, list) else value
    )
    sys.modules["torch"] = torch_stub

    _stub_module(
        "sglang.srt.mem_cache.hicache_storage",
        HiCacheStorageConfig=type("HiCacheStorageConfig", (), {}),
        HiCacheStorageExtraInfo=type("HiCacheStorageExtraInfo", (), {}),
    )
    _stub_module(
        "sglang.srt.distributed",
        get_tensor_model_parallel_rank=lambda: 0,
        get_tensor_model_parallel_world_size=lambda: 1,
    )
    _stub_module(
        "sglang.srt.layers.dp_attention",
        get_attention_dp_rank=lambda: 0,
        get_attention_tp_rank=lambda: 0,
        get_attention_tp_size=lambda: 1,
        is_dp_attention_enabled=lambda: False,
    )
    _stub_module(
        "sglang.srt.mem_cache.memory_pool",
        MLATokenToKVPool=type("MLATokenToKVPool", (), {}),
    )
    _stub_module(
        "sglang.srt.utils",
        get_device_module=lambda: SimpleNamespace(
            Event=lambda: SimpleNamespace(record=lambda: None),
            current_stream=lambda: SimpleNamespace(wait_event=lambda event: None),
        ),
    )


class _StorageBackend:
    def match_prefix_from(self, full_tokens, prefix_pages, lookup_dp_rank=None):
        self.full_tokens = list(full_tokens)
        self.prefix_pages = prefix_pages
        self.lookup_dp_rank = lookup_dp_rank
        return 3, 5


class RemoteBackupPrefetchTest(unittest.TestCase):
    def test_token_prefetch_expected_tokens_use_contiguous_data_pages(self):
        _install_cache_controller_dependency_stubs()
        from sglang.srt.managers.cache_controller import HiCacheController

        controller = HiCacheController.__new__(HiCacheController)
        controller.page_size = 1
        controller.storage_backend = _StorageBackend()
        controller.get_hash_str = lambda tokens, last_hash: f"{last_hash}:{tokens[0]}"

        operation = SimpleNamespace(
            request_id="rid",
            token_ids=[10, 11, 12, 13, 14, 15],
            prefix_token_ids=[1],
            last_hash="h",
            lookup_dp_rank=1,
        )

        hash_value, storage_query_count = controller._storage_hit_query_tokens(
            operation
        )

        self.assertEqual(len(hash_value), 3)
        self.assertEqual(storage_query_count, 3)
        self.assertEqual(operation.full_token_ids, [1, 10, 11, 12, 13, 14, 15])
        self.assertEqual(operation.page_start, 1)


if __name__ == "__main__":
    unittest.main()
