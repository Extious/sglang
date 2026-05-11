"""Unit tests for remote-acked token tracking in HiRadixCache."""

import queue
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.cache_controller import StorageOperation
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import (
    RadixKey,
    TreeNode,
    _key_match_page_size1,
    get_child_key,
)

register_cpu_ci(5, "stage-a-test-cpu")


def _make_cache() -> HiRadixCache:
    cache = HiRadixCache.__new__(HiRadixCache)
    cache.is_eagle = False
    cache.page_size = 1
    cache.enable_storage = False
    cache.enable_storage_metrics = False
    cache.hicache_storage_pass_prefix_keys = False
    cache.root_node = TreeNode(id=0)
    cache.root_node.key = RadixKey([])
    cache.root_node.host_value = []
    cache.root_node.hash_value = []
    cache.key_match_fn = _key_match_page_size1
    cache.get_child_key_fn = get_child_key
    cache.maybe_bigram_convert = lambda key, value=None: (key, value)
    cache.ongoing_backup = {}
    cache.ongoing_request_namespace_backup = {}
    cache.request_namespace_synced_tokens_by_reqid_gen = {}
    cache.ongoing_prefetch = {}
    cache.cache_controller = SimpleNamespace(
        ack_backup_queue=queue.Queue(),
        prefetch_revoke_queue=queue.Queue(),
        host_mem_release_queue=queue.Queue(),
    )
    return cache


class TestHiRadixCacheRemoteAck(CustomTestCase):
    def test_backup_ack_updates_remote_acked_count(self):
        cache = _make_cache()
        child = TreeNode()
        child.parent = cache.root_node
        child.key = RadixKey([1, 2, 3, 4])
        child.value = None
        child.host_value = torch.tensor([10, 11, 12, 13], dtype=torch.int64)
        child.hash_value = ["h1", "h2", "h3", "h4"]
        cache.root_node.children[get_child_key(child.key)] = child
        child.protect_host()

        operation = StorageOperation(
            host_indices=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
            token_ids=[1, 2, 3, 4],
        )
        operation.completed_tokens = 3
        cache.ongoing_backup[operation.id] = child
        cache.cache_controller.ack_backup_queue.put(operation)

        cache._drain_storage_control_queues_impl(
            n_revoke=0, n_backup=1, n_release=0, log_metrics=False
        )

        self.assertEqual(child.storage_acked_len, 3)
        self.assertEqual(child.host_ref_counter, 0)
        self.assertEqual(cache.count_remote_acked_tokens([1, 2, 3, 4]), 3)

    def test_split_node_preserves_remote_acked_lengths(self):
        cache = _make_cache()
        child = TreeNode()
        child.parent = cache.root_node
        child.key = RadixKey([1, 2, 3, 4])
        child.value = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        child.host_value = torch.tensor([5, 6, 7, 8], dtype=torch.int64)
        child.hash_value = ["h1", "h2", "h3", "h4"]
        child.storage_acked_len = 3
        cache.root_node.children[get_child_key(child.key)] = child

        new_node = cache._split_node(child.key, child, 2)

        self.assertEqual(new_node.storage_acked_len, 2)
        self.assertEqual(child.storage_acked_len, 1)
        self.assertEqual(cache.count_remote_acked_tokens([1, 2, 3, 4]), 3)

    def test_write_backup_storage_uses_full_token_context(self):
        cache = _make_cache()
        cache.hicache_storage_pass_prefix_keys = True
        cache.ongoing_backup = {}
        cache.cache_controller.write_storage = MagicMock(return_value=17)

        prefix = TreeNode()
        prefix.parent = cache.root_node
        prefix.key = RadixKey([10, 11])
        prefix.host_value = torch.tensor([100, 101], dtype=torch.int64)
        prefix.hash_value = ["p0", "p1"]
        cache.root_node.children[get_child_key(prefix.key)] = prefix

        child = TreeNode()
        child.parent = prefix
        child.key = RadixKey([12, 13, 14])
        child.host_value = torch.tensor([200, 201, 202], dtype=torch.int64)
        child.hash_value = ["c0", "c1", "c2"]
        child.request_id = "rid-123"
        child.request_generation = 7
        prefix.children[get_child_key(child.key)] = child

        cache.write_backup_storage(child)

        cache.cache_controller.write_storage.assert_called_once_with(
            child.host_value,
            child.key,
            child.hash_value,
            ["p0", "p1"],
            full_token_ids=[10, 11, 12, 13, 14],
            page_start=2,
            request_id="rid-123",
            request_generation=7,
        )
        self.assertIs(cache.ongoing_backup[17], child)
        self.assertEqual(child.host_ref_counter, 1)

    def test_request_namespace_mirror_tracks_synced_prefix_after_ack(self):
        cache = _make_cache()
        cache.enable_storage = True
        cache.cache_controller.storage_backend_type = "remote_backup"
        cache.cache_controller.write_storage = MagicMock(return_value=23)

        prefix = TreeNode()
        prefix.parent = cache.root_node
        prefix.key = RadixKey([1, 2, 3])
        prefix.host_value = torch.tensor([10, 11, 12], dtype=torch.int64)
        prefix.hash_value = ["h0", "h1", "h2"]
        cache.root_node.children[get_child_key(prefix.key)] = prefix

        req = SimpleNamespace(
            rid="rid-ns",
            remote_backup_generation=2,
            extra_key=None,
        )
        cache._mirror_request_namespace_prefix(req, [1, 2, 3, 4, 5])

        cache.cache_controller.write_storage.assert_called_once()
        args, kwargs = cache.cache_controller.write_storage.call_args
        self.assertTrue(torch.equal(args[0], prefix.host_value))
        self.assertEqual(args[1], [1, 2, 3])
        self.assertEqual(kwargs["hash_value"], ["ns:rid-ns:0", "ns:rid-ns:1", "ns:rid-ns:2"])
        self.assertEqual(kwargs["full_token_ids"], [1, 2, 3])
        self.assertEqual(kwargs["page_start"], 0)
        self.assertEqual(kwargs["request_id"], "rid-ns")
        self.assertEqual(kwargs["request_generation"], 2)
        self.assertEqual(
            cache.ongoing_request_namespace_backup[23], ("rid-ns", 2, 3)
        )

        operation = StorageOperation(
            host_indices=torch.tensor([10, 11, 12], dtype=torch.int64),
            token_ids=[1, 2, 3],
        )
        operation.id = 23
        operation.completed_tokens = 3
        cache.cache_controller.ack_backup_queue.put(operation)
        cache._drain_storage_control_queues_impl(
            n_revoke=0, n_backup=1, n_release=0, log_metrics=False
        )

        self.assertEqual(
            cache.request_namespace_synced_tokens_by_reqid_gen[("rid-ns", 2)], 3
        )


if __name__ == "__main__":
    unittest.main()
