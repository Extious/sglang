"""Unit tests for failover abort handling in tokenizer_manager."""

import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import AbortReq, FailoverBatchReqInput, ReqSnapshot
from sglang.srt.managers.tokenizer_manager import TokenizerManager

register_cpu_ci(5, "stage-a-test-cpu")


class TestTokenizerManagerFailoverAbort(CustomTestCase):
    """Test failover-aware abort handling."""

    def setUp(self):
        self.manager = TokenizerManager.__new__(TokenizerManager)
        self.manager.pending_failover_rids = {"rid-1"}
        self.manager.failover_failed_dp_by_rid = {"rid-1": 1}
        self.manager.failover_active_dp_by_rid = {}
        self.manager.completed_failover_failed_dp_by_rid = {}
        self.manager.rid_to_state = {}
        self.manager.server_args = SimpleNamespace(
            weight_version="test-weight",
            enable_lora=False,
            skip_tokenizer_init=False,
        )
        self.manager.add_logprob_to_meta_info = MagicMock()
        self.manager.send_to_scheduler = MagicMock()

    def _make_state(self):
        return SimpleNamespace(
            finished=False,
            time_stats=MagicMock(),
            obj=SimpleNamespace(
                stream=False,
                return_logprob=False,
                lora_path=None,
            ),
            output_ids=[],
            text="partial text",
            out_list=[],
            event=MagicMock(),
        )

    @patch("sglang.srt.managers.tokenizer_manager.is_health_check_generate_req")
    def test_ignore_recoverable_abort_from_failed_rank(self, mock_health_check):
        mock_health_check.return_value = False
        state = self._make_state()
        self.manager.rid_to_state["rid-1"] = state

        self.manager._handle_abort_req(
            AbortReq(
                rid="rid-1",
                finished_reason={
                    "type": "abort",
                    "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                    "message": "recoverable failover abort",
                },
                dp_rank=1,
            )
        )

        self.assertFalse(state.finished)
        state.time_stats.set_finished_time.assert_not_called()
        self.assertEqual(state.out_list, [])
        state.event.set.assert_not_called()
        self.assertIn("rid-1", self.manager.rid_to_state)

    @patch("sglang.srt.managers.tokenizer_manager.is_health_check_generate_req")
    def test_do_not_ignore_abort_from_active_rank(self, mock_health_check):
        mock_health_check.return_value = False
        state = self._make_state()
        self.manager.rid_to_state["rid-1"] = state

        self.manager._handle_abort_req(
            AbortReq(
                rid="rid-1",
                finished_reason={
                    "type": "abort",
                    "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                    "message": "active replica abort",
                },
                dp_rank=0,
            )
        )

        self.assertTrue(state.finished)
        state.time_stats.set_finished_time.assert_called_once()
        state.event.set.assert_called_once()
        self.assertEqual(len(state.out_list), 1)
        self.assertEqual(
            state.out_list[0]["meta_info"]["finish_reason"]["message"],
            "active replica abort",
        )

    @patch("sglang.srt.utils.failover_event_logger.append_failover_event")
    def test_failover_retry_preserves_failed_dp_rank(self, mock_append_failover_event):
        sampling_params = SimpleNamespace(max_new_tokens=16)
        snapshot = ReqSnapshot(
            rid="rid-1",
            origin_input_ids=[1, 2],
            output_ids=[3, 4],
            sampling_params=sampling_params,
            original_max_new_tokens=16,
            stream=False,
            backed_up_tokens=7,
        )

        self.manager._handle_failover_batch(
            FailoverBatchReqInput(failed_dp_rank=1, snapshots=[snapshot])
        )

        self.manager.send_to_scheduler.send_pyobj.assert_called_once()
        retried_req = self.manager.send_to_scheduler.send_pyobj.call_args.args[0]
        self.assertEqual(retried_req.failed_dp_rank, 1)
        self.assertEqual(retried_req.pre_failover_backed_up_tokens, 7)
        self.assertEqual(retried_req.input_ids, [1, 2, 3, 4])
        self.assertEqual(self.manager.failover_failed_dp_by_rid["rid-1"], 1)
        self.assertEqual(mock_append_failover_event.call_count, 2)

    @patch("sglang.srt.utils.failover_event_logger.append_failover_event")
    def test_failover_retry_uses_scheduler_snapshot_tokens(
        self, mock_append_failover_event
    ):
        self.manager.rid_to_state["rid-1"] = SimpleNamespace(
            obj=SimpleNamespace(
                rid="rid-1",
                input_ids=[999, 888],
                sampling_params=SimpleNamespace(max_new_tokens=8),
                stream=False,
            )
        )
        snapshot = ReqSnapshot(
            rid="rid-1",
            origin_input_ids=[1, 2],
            output_ids=[3],
            sampling_params=SimpleNamespace(max_new_tokens=8),
            original_max_new_tokens=8,
            stream=False,
            backed_up_tokens=3,
        )

        self.manager._handle_failover_batch(
            FailoverBatchReqInput(failed_dp_rank=1, snapshots=[snapshot])
        )

        retried_req = self.manager.send_to_scheduler.send_pyobj.call_args.args[0]
        self.assertEqual(retried_req.input_ids, [1, 2, 3])
        self.assertEqual(retried_req.original_prompt_len, 2)
        self.assertEqual(mock_append_failover_event.call_count, 2)


if __name__ == "__main__":
    unittest.main()
