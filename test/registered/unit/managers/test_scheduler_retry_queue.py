"""Unit tests for failover retry queue handling in scheduler."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import SimulateGpuFailureReqInput
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.disaggregation.utils import DisaggregationMode

register_cpu_ci(5, "stage-a-test-cpu")


def _make_req(rid: str):
    return SimpleNamespace(rid=rid, init_next_round_input=MagicMock())


class TestSchedulerRetryQueue(CustomTestCase):
    """Test that retry_queue preserves unprocessed failover requests."""

    def setUp(self):
        self.scheduler = Scheduler.__new__(Scheduler)
        self.scheduler.retry_queue = []
        self.scheduler.enable_hicache_storage = False
        self.scheduler.chunked_req = None
        self.scheduler.truncation_align_size = 1
        self.scheduler.tree_cache = MagicMock()

    def test_preserve_tail_when_current_req_already_added(self):
        first = _make_req("rid-1")
        second = _make_req("rid-2")
        third = _make_req("rid-3")
        self.scheduler.retry_queue = [first, second, third]

        adder = MagicMock()
        adder.budget_state.return_value = AddReqResult.CONTINUE
        adder.add_one_req.return_value = AddReqResult.NO_TOKEN
        adder.can_run_list = [first]

        self.scheduler._process_retry_queue(adder)

        self.assertEqual(self.scheduler.retry_queue, [second, third])
        first.init_next_round_input.assert_called_once_with(self.scheduler.tree_cache)
        second.init_next_round_input.assert_not_called()
        third.init_next_round_input.assert_not_called()

    def test_preserve_current_and_tail_when_current_req_not_added(self):
        first = _make_req("rid-1")
        second = _make_req("rid-2")
        third = _make_req("rid-3")
        self.scheduler.retry_queue = [first, second, third]

        adder = MagicMock()
        adder.budget_state.return_value = AddReqResult.CONTINUE
        adder.add_one_req.return_value = AddReqResult.OTHER
        adder.can_run_list = []

        self.scheduler._process_retry_queue(adder)

        self.assertEqual(self.scheduler.retry_queue, [first, second, third])
        first.init_next_round_input.assert_called_once_with(self.scheduler.tree_cache)
        second.init_next_round_input.assert_not_called()
        third.init_next_round_input.assert_not_called()

    def test_failover_prefetch_uses_failed_rank_bucket(self):
        last_host_node = MagicMock()
        last_host_node.backuped = True
        last_host_node.get_last_hash_value.return_value = "hash"
        last_host_node.get_prefix_hash_values.return_value = ["prefix-hash"]
        last_host_node.parent = object()

        req = SimpleNamespace(
            rid="rid-1",
            init_next_round_input=MagicMock(),
            last_host_node=last_host_node,
            prefix_indices=[101, 102],
            host_hit_length=2,
            fill_ids=[101, 102, 103, 104, 105],
            is_failover_retried=True,
            failover_source_dp_rank=1,
        )

        self.scheduler.enable_hicache_storage = True
        self.scheduler.tree_cache.root_node = object()
        self.scheduler.tree_cache.hicache_storage_pass_prefix_keys = True

        self.scheduler._prefetch_kvcache(req)

        req.init_next_round_input.assert_called_once_with(
            self.scheduler.tree_cache, cow_mamba=False
        )
        self.scheduler.tree_cache.prefetch_from_storage.assert_called_once_with(
            "rid-1",
            last_host_node,
            [105],
            "hash",
            ["prefix-hash"],
            prefix_token_ids=[101, 102, 103, 104],
            lookup_dp_rank=1,
        )

    def test_snapshot_uses_remote_acked_tokens(self):
        req = SimpleNamespace(
            rid="rid-1",
            origin_input_ids=[1, 2],
            output_ids=[3, 4],
            sampling_params=SimpleNamespace(max_new_tokens=12),
            stream=False,
        )
        self.scheduler.enable_hicache_storage = True
        self.scheduler.tree_cache.count_remote_acked_tokens = MagicMock(return_value=6)

        snapshot = self.scheduler._snapshot_req(req)

        self.assertEqual(snapshot.backed_up_tokens, 6)
        self.assertEqual(snapshot.origin_input_ids, [1, 2])
        self.assertEqual(snapshot.output_ids, [3, 4])
        self.scheduler.tree_cache.count_remote_acked_tokens.assert_called_once_with(
            [1, 2, 3, 4]
        )

    def test_pending_hicache_load_queue_blocks_idle_memory_check(self):
        cache_controller = SimpleNamespace(
            prefetch_revoke_queue=None,
            ack_backup_queue=None,
            host_mem_release_queue=None,
            prefetch_queue=None,
            prefetch_buffer=None,
            backup_queue=None,
            ack_write_queue=[],
            ack_load_queue=[],
            load_queue=[object()],
            load_buffer=None,
        )
        self.scheduler.tree_cache = SimpleNamespace(
            check_hicache_events=MagicMock(),
            ongoing_prefetch={},
            ongoing_write_through={},
            ongoing_backup={},
            ongoing_load_back={},
            cache_controller=cache_controller,
        )

        self.assertTrue(self.scheduler._has_pending_hicache_io())

    def test_pending_hicache_load_ack_blocks_idle_memory_check(self):
        cache_controller = SimpleNamespace(
            prefetch_revoke_queue=None,
            ack_backup_queue=None,
            host_mem_release_queue=None,
            prefetch_queue=None,
            prefetch_buffer=None,
            backup_queue=None,
            ack_write_queue=[],
            ack_load_queue=[object()],
            load_queue=[],
            load_buffer=None,
        )
        self.scheduler.tree_cache = SimpleNamespace(
            check_hicache_events=MagicMock(),
            ongoing_prefetch={},
            ongoing_write_through={},
            ongoing_backup={},
            ongoing_load_back={},
            cache_controller=cache_controller,
        )

        self.assertTrue(self.scheduler._has_pending_hicache_io())

    def test_chunked_request_blocks_idle_memory_check(self):
        self.scheduler.enable_hisparse = False
        self.scheduler.tree_cache = SimpleNamespace(cache_controller=None)
        self.scheduler.running_batch = SimpleNamespace(is_empty=MagicMock(return_value=True))
        self.scheduler.chunked_req = object()
        self.scheduler.disaggregation_mode = DisaggregationMode.NULL
        self.scheduler.check_memory = MagicMock()
        self.scheduler.check_tree_cache = MagicMock()
        self.scheduler.init_new_token_ratio = 0.0
        self.scheduler.maybe_sleep_on_idle = MagicMock()

        self.scheduler.self_check_during_idle()

        self.scheduler.check_memory.assert_not_called()
        self.scheduler.check_tree_cache.assert_not_called()

    @patch("sglang.srt.managers.scheduler_output_processor_mixin.release_kv_cache")
    def test_finished_failover_retry_is_inserted_into_cache(self, mock_release_kv_cache):
        req = SimpleNamespace(
            rid="rid-1",
            is_failover_retried=True,
            is_retracted=False,
            origin_input_ids=[1, 2, 3],
            output_ids=[],
            require_reasoning=False,
            multimodal_inputs=None,
            session=None,
            return_logprob=False,
            return_hidden_states=False,
            grammar=None,
            token_ids_logprob=None,
            mamba_ping_pong_track_buffer=None,
            time_stats=SimpleNamespace(
                set_last_decode_finish_time=MagicMock(),
                set_completion_time=MagicMock(),
            ),
        )
        req.finished = MagicMock(return_value=False)
        req.check_finished = MagicMock(side_effect=lambda *_: setattr(req, "done", True))
        req.finished = MagicMock(side_effect=lambda: getattr(req, "done", False))

        batch = SimpleNamespace(
            reqs=[req],
            return_logprob=False,
            spec_algorithm=SimpleNamespace(is_none=MagicMock(return_value=True)),
            is_spec_v2=False,
        )
        result = SimpleNamespace(
            copy_done=None,
            logits_output=SimpleNamespace(hidden_states=None),
            next_token_ids=torch.tensor([42]),
            can_run_cuda_graph=False,
            num_accepted_tokens=0,
        )
        self.scheduler.is_generation = True
        self.scheduler.enable_overlap = False
        self.scheduler.enable_metrics = False
        self.scheduler.num_generated_tokens = 0
        self.scheduler.server_args = SimpleNamespace(
            disaggregation_decode_enable_offload_kvcache=False
        )
        self.scheduler.enable_hisparse = False
        self.scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            free_group_begin=MagicMock(), free_group_end=MagicMock()
        )
        self.scheduler.stream_output = MagicMock()
        self.scheduler.report_decode_stats = MagicMock()
        self.scheduler.maybe_collect_routed_experts = MagicMock()
        self.scheduler.maybe_collect_customized_info = MagicMock()
        self.scheduler.forward_ct_decode = 0

        self.scheduler.process_batch_result_decode(batch, result)

        mock_release_kv_cache.assert_called_once_with(req, self.scheduler.tree_cache)

    @patch("sglang.srt.utils.failover_event_logger.append_failover_event")
    def test_gpu_failure_flushes_hicache_events_before_snapshot(
        self, mock_append_failover_event
    ):
        self.scheduler.enable_hicache_storage = True
        self.scheduler.dp_rank = 1
        self.scheduler.tree_cache.check_hicache_events = MagicMock()
        self.scheduler._collect_failover_reqs = MagicMock(return_value=[])
        self.scheduler._clear_failed_rank_scheduler_state = MagicMock()
        self.scheduler.send_to_tokenizer = MagicMock()

        self.scheduler.handle_simulate_gpu_failure(SimulateGpuFailureReqInput(dp_rank=1))

        self.scheduler.tree_cache.check_hicache_events.assert_called_once_with()
        self.scheduler._collect_failover_reqs.assert_called_once_with()
        self.scheduler.send_to_tokenizer.send_output.assert_called_once()
        self.assertEqual(mock_append_failover_event.call_count, 1)


if __name__ == "__main__":
    unittest.main()
