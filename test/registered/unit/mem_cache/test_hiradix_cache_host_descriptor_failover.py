"""Unit tests for descriptor-based host backup failover in HiRadixCache."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import (
    RadixKey,
    TreeNode,
    _key_match_page_size1,
    get_child_key,
)

register_cpu_ci(5, "stage-a-test-cpu")


class _FakeDescriptorHostPool:
    def __init__(self):
        self.shared_arena_id = "arena-1"
        self.alloc = MagicMock()
        self.free = MagicMock()
        self.evict_host = MagicMock(return_value=0)
        self.get_data_page = MagicMock()

    def is_shared_across_workers(self) -> bool:
        return True

    def descriptor_context(self):
        return {
            "arena_id": self.shared_arena_id,
            "layout": "layer_first",
            "page_size": 1,
            "owner_dp_rank": 0,
        }

    def descriptor_is_compatible(self, *, arena_id, layout, page_size) -> bool:
        return (
            arena_id == self.shared_arena_id
            and layout == "layer_first"
            and page_size == 1
        )


def _make_cache() -> HiRadixCache:
    cache = HiRadixCache.__new__(HiRadixCache)
    cache.page_size = 1
    cache.root_node = TreeNode(id=0)
    cache.root_node.key = RadixKey([])
    cache.root_node.host_value = None
    cache.root_node.hash_value = []
    cache.root_node.value = None
    cache.key_match_fn = _key_match_page_size1
    cache.get_child_key_fn = get_child_key
    cache.maybe_bigram_convert = lambda key, value=None: (key, value)
    cache.imported_host_keys_by_owner_gen = {}
    cache.evictable_host_leaves = set()
    cache._update_leaf_status = lambda node: None
    cache._update_host_leaf_status = lambda node: None
    cache.cache_controller = SimpleNamespace(
        dp_rank=1,
        mem_pool_host=_FakeDescriptorHostPool(),
    )
    return cache


class TestHiRadixCacheHostDescriptorFailover(CustomTestCase):
    def test_export_failure_checkpoints_uses_descriptor_payload(self):
        cache = _make_cache()
        child = TreeNode()
        child.parent = cache.root_node
        child.key = RadixKey([1, 2, 3, 4])
        child.value = None
        child.host_value = torch.tensor([100, 101, 102, 103], dtype=torch.int64)
        child.hash_value = ["h0", "h1", "h2", "h3"]
        child.host_owner_dp_rank = 1
        child.host_arena_id = "arena-1"
        cache.root_node.children[get_child_key(child.key)] = child

        req = SimpleNamespace(
            rid="rid-export",
            origin_input_ids=[1, 2],
            output_ids=[3, 4],
            extra_key=None,
            remote_backup_generation=7,
        )
        metadata_batch = cache.export_failure_checkpoints(req)

        self.assertEqual(len(metadata_batch), 1)
        metadata = metadata_batch[0]
        self.assertEqual(metadata["checkpoint_len"], 4)
        self.assertEqual(metadata["owner_dp_rank"], 1)
        self.assertEqual(metadata["generation"], 7)
        self.assertEqual(metadata["arena_id"], "arena-1")
        self.assertEqual(
            metadata["descriptors"],
            [{"slot_start": 100, "slot_count": 4}],
        )
        self.assertEqual(metadata["pages"], [])
        cache.cache_controller.mem_pool_host.get_data_page.assert_not_called()

    def test_import_host_checkpoints_attaches_shared_slots_without_alloc(self):
        cache = _make_cache()
        imported = cache.import_host_checkpoints(
            [
                {
                    "owner_dp_rank": 0,
                    "generation": 5,
                    "rid": "rid-import",
                    "extra_key": None,
                    "checkpoint_len": 4,
                    "prefix_token_ids": [10, 11, 12, 13],
                    "arena_id": "arena-1",
                    "layout": "layer_first",
                    "page_size": 1,
                    "descriptors": [{"slot_start": 200, "slot_count": 4}],
                }
            ]
        )

        self.assertEqual(imported, 4)
        child = next(iter(cache.root_node.children.values()))
        self.assertTrue(torch.equal(child.host_value, torch.tensor([200, 201, 202, 203])))
        self.assertTrue(child.host_imported)
        self.assertEqual(child.host_owner_dp_rank, 0)
        self.assertEqual(child.host_generation, 5)
        self.assertEqual(child.host_arena_id, "arena-1")
        cache.cache_controller.mem_pool_host.alloc.assert_not_called()
        cache.cache_controller.mem_pool_host.free.assert_not_called()

    def test_invalidate_imported_generation_does_not_free_foreign_slots(self):
        cache = _make_cache()
        cache.import_host_checkpoints(
            [
                {
                    "owner_dp_rank": 0,
                    "generation": 9,
                    "rid": "rid-cleanup",
                    "extra_key": None,
                    "checkpoint_len": 2,
                    "prefix_token_ids": [21, 22],
                    "arena_id": "arena-1",
                    "layout": "layer_first",
                    "page_size": 1,
                    "descriptors": [{"slot_start": 300, "slot_count": 2}],
                }
            ]
        )

        cache.invalidate_imported_generation(owner_dp_rank=0, generation=9)

        self.assertEqual(cache.root_node.children, {})
        cache.cache_controller.mem_pool_host.evict_host.assert_not_called()


if __name__ == "__main__":
    unittest.main()
