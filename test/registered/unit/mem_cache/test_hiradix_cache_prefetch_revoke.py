"""Unit tests for prefetch revoke bookkeeping in HiRadixCache."""

import queue
import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.cache_controller import PrefetchOperation
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
    cache.root_node = TreeNode(id=0)
    cache.root_node.key = RadixKey([])
    cache.root_node.host_value = []
    cache.root_node.hash_value = []
    cache.key_match_fn = _key_match_page_size1
    cache.get_child_key_fn = get_child_key
    cache.maybe_bigram_convert = lambda key, value=None: (key, value)
    cache.ongoing_backup = {}
    cache.ongoing_prefetch = {}
    cache.enable_storage_metrics = False
    cache.cache_controller = SimpleNamespace(
        ack_backup_queue=queue.Queue(),
        prefetch_revoke_queue=queue.Queue(),
        host_mem_release_queue=queue.Queue(),
        prefetch_tokens_occupied=0,
    )
    return cache


class TestHiRadixCachePrefetchRevoke(CustomTestCase):
    def test_prefetch_operation_keeps_request_id(self):
        operation = PrefetchOperation(
            request_id="rid-123",
            host_indices=torch.tensor([0, 1], dtype=torch.int64),
            token_ids=[10, 11],
        )

        self.assertEqual(operation.request_id, "rid-123")

    def test_revoke_queue_releases_matching_ongoing_prefetch(self):
        cache = _make_cache()
        node = TreeNode()
        node.parent = cache.root_node
        node.key = RadixKey([1, 2])
        node.host_value = torch.tensor([7, 8], dtype=torch.int64)
        node.hash_value = ["h1", "h2"]
        cache.root_node.children[get_child_key(node.key)] = node
        node.protect_host()

        token_ids = [21, 22]
        host_indices = torch.tensor([4, 5], dtype=torch.int64)
        operation = PrefetchOperation(
            request_id="rid-revoke",
            host_indices=host_indices,
            token_ids=token_ids,
        )
        cache.ongoing_prefetch["rid-revoke"] = (
            node,
            token_ids,
            host_indices,
            operation,
        )
        cache.cache_controller.prefetch_tokens_occupied = len(token_ids)
        cache.cache_controller.prefetch_revoke_queue.put(operation.request_id)

        cache._drain_storage_control_queues_impl(
            n_revoke=1, n_backup=0, n_release=0, log_metrics=False
        )

        self.assertNotIn("rid-revoke", cache.ongoing_prefetch)
        self.assertEqual(node.host_ref_counter, 0)
        self.assertEqual(cache.cache_controller.prefetch_tokens_occupied, 0)


if __name__ == "__main__":
    unittest.main()
