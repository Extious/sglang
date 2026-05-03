from __future__ import annotations

import logging
import time
import warnings
from typing import TYPE_CHECKING

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.session_aware_cache import SessionAwareCache
from sglang.srt.observability.metrics_collector import QueueCount
from sglang.srt.utils.common import ceil_align, raise_error_or_warn
from sglang.srt.utils.request_logger import disable_request_logging
from sglang.srt.utils.watchdog import WatchdogRaw

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerRuntimeCheckerMixin:
    def _has_pending_hicache_io(self: Scheduler) -> bool:
        """Skip strict idle checks while async HiCache IO is still draining."""
        tree_cache = getattr(self, "tree_cache", None)
        if tree_cache is None:
            return False

        drain_fn = getattr(tree_cache, "check_hicache_events", None)
        if callable(drain_fn):
            drain_fn()

        for dict_name in (
            "ongoing_prefetch",
            "ongoing_write_through",
            "ongoing_backup",
            "ongoing_load_back",
        ):
            d = getattr(tree_cache, dict_name, None)
            if d is not None and len(d) > 0:
                return True

        cache_controller = getattr(tree_cache, "cache_controller", None)
        if cache_controller is None:
            return False

        for queue_name in (
            "prefetch_revoke_queue",
            "ack_backup_queue",
            "host_mem_release_queue",
            "prefetch_queue",
            "prefetch_buffer",
            "backup_queue",
        ):
            q = getattr(cache_controller, queue_name, None)
            if q is None:
                continue
            qsize_fn = getattr(q, "qsize", None)
            if callable(qsize_fn) and qsize_fn() > 0:
                return True

        ack_write_queue = getattr(cache_controller, "ack_write_queue", None)
        if ack_write_queue is not None and len(ack_write_queue) > 0:
            return True

        for list_name in ("load_queue", "ack_load_queue"):
            pending = getattr(cache_controller, list_name, None)
            if pending is not None and len(pending) > 0:
                return True

        return False

    def _session_held_tokens(self: Scheduler) -> int:
        if isinstance(self.tree_cache, SessionAwareCache):
            return self.tree_cache.session_held_tokens()
        return 0

    def _session_held_full_tokens(self: Scheduler) -> int:
        if isinstance(self.tree_cache, SessionAwareCache):
            return self.tree_cache.session_held_full_tokens()
        return 0

    def _session_held_swa_tokens(self: Scheduler) -> int:
        if isinstance(self.tree_cache, SessionAwareCache):
            return self.tree_cache.session_held_swa_tokens()
        return 0

    def _session_held_req_count(self: Scheduler) -> int:
        if isinstance(self.tree_cache, SessionAwareCache):
            return self.tree_cache.session_held_req_count()
        return 0

    def _get_token_info(self: Scheduler):
        available_size = self.token_to_kv_pool_allocator.available_size()
        evictable_size = self.tree_cache.evictable_size()
        num_used = self.max_total_num_tokens - (available_size + evictable_size)
        token_usage = num_used / self.max_total_num_tokens
        return num_used, token_usage, available_size, evictable_size

    def _get_mamba_token_info(self: Scheduler):
        is_mamba_radix_cache = (
            self.tree_cache.supports_mamba() and self.tree_cache.is_tree_cache()
        )
        full_available_size = self.token_to_kv_pool_allocator.available_size()
        full_evictable_size = (
            self.tree_cache.full_evictable_size() if is_mamba_radix_cache else 0
        )
        mamba_available_size = self.req_to_token_pool.mamba_pool.available_size()
        mamba_evictable_size = (
            self.tree_cache.mamba_evictable_size() if is_mamba_radix_cache else 0
        )
        full_num_used = self.token_to_kv_pool_allocator.size - (
            full_available_size + full_evictable_size
        )
        mamba_num_used = self.req_to_token_pool.mamba_pool.size - (
            mamba_available_size + mamba_evictable_size
        )
        full_token_usage = full_num_used / self.token_to_kv_pool_allocator.size
        mamba_usage = mamba_num_used / self.req_to_token_pool.mamba_pool.size
        return (
            full_num_used,
            mamba_num_used,
            full_token_usage,
            mamba_usage,
            full_available_size,
            full_evictable_size,
            mamba_available_size,
            mamba_evictable_size,
        )

    def _get_swa_token_info(self: Scheduler):
        full_available_size = self.token_to_kv_pool_allocator.full_available_size()
        full_evictable_size = self.tree_cache.full_evictable_size()
        swa_available_size = self.token_to_kv_pool_allocator.swa_available_size()
        swa_evictable_size = self.tree_cache.swa_evictable_size()
        full_num_used = self.full_tokens_per_layer - (
            full_available_size + full_evictable_size
        )
        swa_num_used = self.swa_tokens_per_layer - (
            swa_available_size + swa_evictable_size
        )
        full_token_usage = full_num_used / self.full_tokens_per_layer
        swa_token_usage = swa_num_used / self.swa_tokens_per_layer
        return (
            full_num_used,
            swa_num_used,
            full_token_usage,
            swa_token_usage,
            full_available_size,
            full_evictable_size,
            swa_available_size,
            swa_evictable_size,
        )

    def _check_hybrid_memory(self: Scheduler):
        (
            full_num_used,
            swa_num_used,
            _,
            _,
            full_available_size,
            full_evictable_size,
            swa_available_size,
            swa_evictable_size,
        ) = self._get_swa_token_info()
        session_held_full = self._session_held_full_tokens()
        session_held_swa = self._session_held_swa_tokens()

        # Streaming sessions hold tree locks during idle, so tree-protected
        # tokens must be accounted for alongside session-held tokens.
        full_protected = self.tree_cache.full_protected_size()
        swa_protected = self.tree_cache.swa_protected_size()
        full_leaked = full_num_used - full_protected - session_held_full
        swa_leaked = swa_num_used - swa_protected - session_held_swa
        memory_leak = full_leaked != 0 or swa_leaked != 0
        token_msg = (
            f"{full_leaked=}, {swa_leaked=}\n"
            f"{self.full_tokens_per_layer=}, {full_available_size=}, {full_evictable_size=}, {full_protected=}, {session_held_full=}\n"
            f"{self.swa_tokens_per_layer=}, {swa_available_size=}, {swa_evictable_size=}, {swa_protected=}, {session_held_swa=}\n"
        )
        return memory_leak, token_msg

    def _check_mamba_memory(self: Scheduler):
        (
            full_num_used,
            mamba_num_used,
            _,
            _,
            full_available_size,
            full_evictable_size,
            mamba_available_size,
            mamba_evictable_size,
        ) = self._get_mamba_token_info()
        session_held = self._session_held_tokens()
        memory_leak = (
            full_num_used != self.tree_cache.full_protected_size() + session_held
            or mamba_num_used != self.tree_cache.mamba_protected_size()
        )
        if memory_leak:
            free_full_pages = set(
                self.token_to_kv_pool_allocator.free_pages.tolist()
                + self.token_to_kv_pool_allocator.release_pages.tolist()
            )
            cached_full_pages = set(self.tree_cache.all_values_flatten().tolist())
            expected_full_pages = set(
                range(1, self.token_to_kv_pool_allocator.size + 1)
            )
            leaked_full_pages = (
                expected_full_pages - free_full_pages - cached_full_pages
            )
            free_mamba_pages = set(
                self.req_to_token_pool.mamba_pool.free_slots.tolist()
            )
            cached_mamba_pages = set(
                self.tree_cache.all_mamba_values_flatten().tolist()
            )
            expected_mamba_pages = set(range(self.req_to_token_pool.mamba_pool.size))
            leaked_mamba_pages = (
                expected_mamba_pages - free_mamba_pages - cached_mamba_pages
            )
            token_msg = (
                f"{full_available_size=}, {full_evictable_size=}, {self.token_to_kv_pool_allocator.size=}, {self.tree_cache.full_protected_size()=}\n"
                f"{mamba_available_size=}, {mamba_evictable_size=}, {self.req_to_token_pool.mamba_pool.size=}, {self.tree_cache.mamba_protected_size()=}, leaked_full_pages={leaked_full_pages if len(leaked_full_pages) > 0 else None}, leaked_mamba_pages={leaked_mamba_pages if len(leaked_mamba_pages) > 0 else None}\n"
            )
        else:
            token_msg = (
                f"{full_available_size=}, {full_evictable_size=}, {self.token_to_kv_pool_allocator.size=}, {self.tree_cache.full_protected_size()=}\n"
                f"{mamba_available_size=}, {mamba_evictable_size=}, {self.req_to_token_pool.mamba_pool.size=}, {self.tree_cache.mamba_protected_size()=}\n"
            )
        return memory_leak, token_msg

    def _check_radix_cache_memory(self: Scheduler):
        _, _, available_size, evictable_size = self._get_token_info()
        protected_size = self.tree_cache.protected_size()
        session_held = self._session_held_tokens()
        memory_leak = (available_size + evictable_size) != (
            self.max_total_num_tokens - protected_size - session_held
        )
        token_msg = f"{self.max_total_num_tokens=}, {available_size=}, {evictable_size=}, {protected_size=}, {session_held=}\n"
        if memory_leak:
            token_msg += self._radix_memory_leak_details()
        return memory_leak, token_msg

    def _radix_memory_leak_details(self: Scheduler) -> str:
        """Return extra ownership info for leaked token pages."""
        try:
            free_pages = set(
                self.token_to_kv_pool_allocator.free_pages.detach().cpu().tolist()
                + self.token_to_kv_pool_allocator.release_pages.detach().cpu().tolist()
            )
            cached_pages = set()
            stack = [self.tree_cache.root_node]
            host_only_nodes = 0
            while stack:
                node = stack.pop()
                value = getattr(node, "value", None)
                if value is None:
                    host_only_nodes += 1
                elif len(value) > 0:
                    cached_pages.update(value.detach().cpu().tolist())
                stack.extend(getattr(node, "children", {}).values())
            expected_pages = set(range(1, self.token_to_kv_pool_allocator.size + 1))
            leaked_pages = expected_pages - free_pages - cached_pages

            if not leaked_pages:
                return f"leaked_pages=None, host_only_nodes={host_only_nodes}\n"

            # Cap leaked_pages to avoid GPU OOM in the diagnostic itself
            if len(leaked_pages) > 4096:
                return (
                    f"leaked_pages_count={len(leaked_pages)} (capped at 4096 for safety), "
                    f"host_only_nodes={host_only_nodes}\n"
                )

            leaked_tensor = torch.tensor(
                list(leaked_pages),
                dtype=self.req_to_token_pool.req_to_token.dtype,
                device="cpu",
            )
            req_to_token = self.req_to_token_pool.req_to_token.cpu()
            owner_mask = torch.isin(req_to_token, leaked_tensor)
            owner_rows = owner_mask.any(dim=1).nonzero(as_tuple=False).flatten()
            free_req_slots = set(self.req_to_token_pool.free_slots.detach().cpu().tolist())
            owners = []
            for row in owner_rows[:16].detach().cpu().tolist():
                positions = owner_mask[row].nonzero(as_tuple=False).flatten()
                owners.append(
                    {
                        "req_pool_idx": int(row),
                        "is_free_slot": row in free_req_slots,
                        "count": int(positions.numel()),
                        "first_pos": int(positions[0].item()) if positions.numel() else None,
                        "last_pos": int(positions[-1].item()) if positions.numel() else None,
                    }
                )

            leaked_sample = sorted(leaked_pages)[:64]
            return (
                f"leaked_pages_count={len(leaked_pages)}, "
                f"leaked_pages_sample={leaked_sample}, "
                f"req_pool_owners={owners}, "
                f"host_only_nodes={host_only_nodes}\n"
            )
        except Exception as e:
            return f"leaked_pages_detail_error={type(e).__name__}: {e}\n"

    def _get_batch_uncached_size(self: Scheduler, batch: ScheduleBatch) -> int:
        ret = 0
        for req in batch.reqs:
            assert req.kv_committed_freed == req.kv_overallocated_freed
            uncached_len = 0
            if not req.kv_committed_freed:
                allocated_len = req.kv_allocated_len
                if self.page_size > 1:
                    allocated_len = ceil_align(allocated_len, self.page_size)
                    assert req.cache_protected_len % self.page_size == 0
                uncached_len = allocated_len - req.cache_protected_len

            ret += uncached_len

        return ret

    def self_check_during_busy(self: Scheduler):
        current_batch: ScheduleBatch = self.last_batch

        if current_batch is None:
            return

        spec_topk = self.server_args.speculative_eagle_topk or 1
        if spec_topk > 1:
            warnings.warn(
                "Runtime memory check (busy) is not supported when speculation topk > 1."
            )
            return

        _, _, available_size, evictable_size = self._get_token_info()
        protected_size = self.tree_cache.protected_size()

        uncached_size = self._get_batch_uncached_size(current_batch)

        if (
            current_batch.forward_mode.is_extend()
            and self.running_batch is not None
            and not self.running_batch.is_empty()
        ):
            uncached_size += self._get_batch_uncached_size(self.running_batch)

        if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get() > 1:
            log_msg = f"[Mem Check (BUSY)] {available_size=}, {evictable_size=}, {protected_size=}, {uncached_size=}"
            logger.info(log_msg)

        session_held = self._session_held_tokens()
        total_tokens = (
            available_size
            + evictable_size
            + protected_size
            + uncached_size
            + session_held
        )
        assert (
            total_tokens == self.max_total_num_tokens
        ), f"Mem Leak Detected! {total_tokens=} vs {self.max_total_num_tokens=}"

    def _check_req_pool(self: Scheduler):
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            req_total_size = (
                self.req_to_token_pool.size + self.req_to_token_pool.pre_alloc_size
            )
        else:
            req_total_size = self.req_to_token_pool.size

        session_req_count = self._session_held_req_count()
        if len(self.req_to_token_pool.free_slots) + session_req_count != req_total_size:
            msg = (
                "req_to_token_pool memory leak detected!"
                f"available_size={len(self.req_to_token_pool.free_slots)}, "
                f"session_held={session_req_count}, "
                f"total_size={self.req_to_token_pool.size}\n"
            )
            raise_error_or_warn(
                self,
                envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.get(),
                "count_req_pool_leak_warnings",
                msg,
            )

    def check_memory(self: Scheduler):
        if self.is_hybrid_swa:
            memory_leak, token_msg = self._check_hybrid_memory()
        elif self.is_hybrid_ssm and self.tree_cache.supports_mamba():
            memory_leak, token_msg = self._check_mamba_memory()
        else:
            memory_leak, token_msg = self._check_radix_cache_memory()
        if memory_leak:
            msg = "token_to_kv_pool_allocator memory leak detected! " f"{token_msg}"
            raise_error_or_warn(
                self,
                envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.get(),
                "count_memory_leak_warnings",
                msg,
            )

        self._check_req_pool()

        if (
            self.current_scheduler_metrics_enabled
            and time.perf_counter() > self.metrics_collector.last_log_time + 30
        ):
            # During idle time, also collect metrics every 30 seconds.
            if self.is_hybrid_swa:
                (
                    full_num_used,
                    swa_num_used,
                    full_token_usage,
                    swa_token_usage,
                    _,
                    _,
                    _,
                    _,
                ) = self._get_swa_token_info()
                num_used = max(full_num_used, swa_num_used)
                token_usage = max(full_token_usage, swa_token_usage)
            elif self.is_hybrid_ssm:
                (
                    num_used,
                    _,
                    token_usage,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = self._get_mamba_token_info()
            else:
                num_used, token_usage, _, _ = self._get_token_info()

            priority_enabled = self.enable_priority_scheduling
            self.stats.num_running_reqs = QueueCount.from_reqs(
                self.running_batch.reqs, priority_enabled
            )
            self.stats.num_used_tokens = num_used
            self.stats.token_usage = round(token_usage, 2)
            self.stats.gen_throughput = 0
            self.stats.num_queue_reqs = QueueCount.from_reqs(
                self.waiting_queue, priority_enabled
            )
            self.stats.num_grammar_queue_reqs = len(self.grammar_manager)
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.stats.num_prefill_prealloc_queue_reqs = QueueCount.from_reqs(
                    self.disagg_prefill_bootstrap_queue.queue, priority_enabled
                )
                self.stats.num_prefill_inflight_queue_reqs = QueueCount.from_reqs(
                    self.disagg_prefill_inflight_queue, priority_enabled
                )
            if self.disaggregation_mode == DisaggregationMode.DECODE:
                self.stats.num_decode_prealloc_queue_reqs = QueueCount.from_reqs(
                    self.disagg_decode_prealloc_queue.queue, priority_enabled
                )
                self.stats.num_decode_transfer_queue_reqs = QueueCount.from_reqs(
                    self.disagg_decode_transfer_queue.queue, priority_enabled
                )
            self.metrics_collector.log_stats(self.stats)
        self._publish_kv_events()

    def check_tree_cache(self: Scheduler):
        if (
            self.tree_cache.is_tree_cache()
            and (self.is_hybrid_swa and self.tree_cache.supports_swa())
            or (self.is_hybrid_ssm and self.tree_cache.supports_mamba())
        ):
            self.tree_cache.sanity_check()

    def self_check_during_idle(self: Scheduler):
        if self.enable_hisparse and self.hisparse_coordinator.has_ongoing_staging():
            return
        # Not truly idle while failover retry requests are pending.
        if len(getattr(self, "retry_queue", [])) > 0:
            return
        if self._has_pending_hicache_io():
            return
        if getattr(self, "chunked_req", None) is not None:
            return
        # Not truly idle while decode/prefill batch is still running.
        if (
            getattr(self, "running_batch", None) is not None
            and not self.running_batch.is_empty()
        ):
            return
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            if len(self.disagg_prefill_inflight_queue) > 0:
                return
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            queue_size = (
                len(self.waiting_queue)
                + len(self.disagg_decode_transfer_queue.queue)
                + len(self.disagg_decode_prealloc_queue.queue)
            )
            if self.server_args.disaggregation_decode_enable_offload_kvcache:
                queue_size += len(self.decode_offload_manager.ongoing_offload)
            if queue_size:
                return

        self.check_memory()
        self.check_tree_cache()
        self.new_token_ratio = self.init_new_token_ratio
        self.maybe_sleep_on_idle()


def create_scheduler_watchdog(
    scheduler: Scheduler, watchdog_timeout: float, soft: bool = False
) -> WatchdogRaw:
    def dump_info() -> str:
        if scheduler.is_initializing or disable_request_logging():
            return ""
        if scheduler.is_hybrid_swa:
            _, info_msg = scheduler._check_hybrid_memory()
        elif scheduler.is_hybrid_ssm and scheduler.tree_cache.supports_mamba():
            _, info_msg = scheduler._check_mamba_memory()
        else:
            _, info_msg = scheduler._check_radix_cache_memory()
        return (
            f"{scheduler.cur_batch.batch_size()=}\n"
            f"{scheduler.cur_batch.reqs=}\n"
            f"{info_msg}"
        )

    return WatchdogRaw(
        debug_name="Scheduler",
        get_counter=lambda: getattr(scheduler, "forward_ct", 0),
        is_active=lambda: scheduler.is_initializing
        or getattr(scheduler, "cur_batch", None) is not None,
        watchdog_timeout=watchdog_timeout,
        soft=soft,
        dump_info=dump_info,
    )
