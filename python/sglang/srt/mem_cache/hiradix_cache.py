from __future__ import annotations

import atexit
import hashlib
import heapq
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from queue import Empty
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

from sglang.srt.managers.cache_controller import HiCacheController, PrefetchOperation
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InitLoadBackParams,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    NSATokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool_host import (
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
    NSATokenToKVPoolHost,
    SharedHostKVPoolConfig,
)
from sglang.srt.mem_cache.radix_cache import (
    RadixCache,
    RadixKey,
    TreeNode,
    compute_node_hash_values,
    split_node_hash_value,
)
from sglang.srt.mem_cache.utils import convert_to_bigram_key
from sglang.srt.observability.metrics_collector import StorageMetricsCollector
from sglang.srt.distributed.naive_distributed import (
    NaiveDistributed,
    get_naive_distributed,
    set_naive_distributed,
)
from sglang.srt.utils.host_shared_memory import HostSharedMemoryManager

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


@dataclass
class HostCheckpointDescriptor:
    slot_start: int
    slot_count: int


@dataclass
class FailureCheckpointMetadata:
    owner_dp_rank: int
    generation: int
    rid: str
    extra_key: Optional[str]
    checkpoint_len: int
    prefix_token_ids: List[int]
    arena_id: Optional[str] = None
    layout: Optional[str] = None
    page_size: int = 0
    descriptors: List[Dict[str, int]] = field(default_factory=list)
    pages: List[torch.Tensor] = field(default_factory=list)


class HiRadixCache(RadixCache):

    @staticmethod
    def _shared_host_descriptor_enabled(server_args: ServerArgs) -> bool:
        return (
            getattr(server_args, "kv_backup_strategy", "none") in ("host", "host_backup")
            and int(getattr(server_args, "dp_size", 1) or 1) > 1
            and int(getattr(server_args, "nnodes", 1) or 1) == 1
        )

    @staticmethod
    def _make_shared_host_pool_config(
        params: CacheInitParams, server_args: ServerArgs, page_size: int
    ) -> Optional[SharedHostKVPoolConfig]:
        if not HiRadixCache._shared_host_descriptor_enabled(server_args):
            return None

        dp_rank = int(getattr(params, "dp_rank", 0) or 0)
        rendezvous_key = "|".join(
            [
                str(getattr(server_args, "dist_init_addr", None) or ""),
                str(getattr(server_args, "host", "127.0.0.1")),
                str(getattr(server_args, "port", 0)),
                str(getattr(server_args, "model_path", "")),
                str(getattr(server_args, "hicache_mem_layout", "")),
                str(page_size),
                str(getattr(server_args, "dp_size", 1)),
            ]
        )
        arena_suffix = hashlib.sha1(rendezvous_key.encode("utf-8")).hexdigest()[:16]
        rendezvous = f"/tmp/sglang_host_backup_{arena_suffix}"
        try:
            dist = get_naive_distributed()
            if dist.get_rank() != dp_rank or dist.get_world_size() != int(
                getattr(server_args, "dp_size", 1) or 1
            ):
                logger.warning(
                    "Reusing existing naive distributed context for host descriptor failover "
                    "with mismatched rank/world_size: existing=(%s,%s) new=(%s,%s)",
                    dist.get_rank(),
                    dist.get_world_size(),
                    dp_rank,
                    getattr(server_args, "dp_size", 1),
                )
        except AssertionError:
            set_naive_distributed(
                NaiveDistributed(
                    rank=dp_rank,
                    world_size=int(getattr(server_args, "dp_size", 1) or 1),
                    rendezvous=rendezvous,
                )
            )

        return SharedHostKVPoolConfig(
            arena_id=f"host-backup-{arena_suffix}",
            manager=HostSharedMemoryManager(base_name=f"sglang_host_backup_{arena_suffix}"),
            owner_rank=dp_rank,
            world_size=int(getattr(server_args, "dp_size", 1) or 1),
        )

    def __init__(self, params: CacheInitParams, server_args: ServerArgs):
        self._enable_metrics_flag = params.enable_metrics

        self.page_size = params.page_size
        self.kv_cache = params.token_to_kv_pool_allocator.get_kvcache()
        shared_host_pool_config = self._make_shared_host_pool_config(
            params, server_args, self.page_size
        )

        if isinstance(self.kv_cache, MHATokenToKVPool):
            self.token_to_kv_pool_host = MHATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
                shared_config=shared_host_pool_config,
            )
        elif isinstance(self.kv_cache, NSATokenToKVPool):
            self.token_to_kv_pool_host = NSATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
                shared_config=shared_host_pool_config,
            )
        elif isinstance(self.kv_cache, MLATokenToKVPool):
            self.token_to_kv_pool_host = MLATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
                shared_config=shared_host_pool_config,
            )
        else:
            raise ValueError(f"HiRadixCache only supports MHA and MLA yet")

        self.tp_group = params.tp_cache_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)
        self.pp_rank = params.pp_rank
        self.pp_size = params.pp_size
        self.enable_storage = server_args.hicache_storage_backend is not None
        self.enable_storage_metrics = self.enable_storage and params.enable_metrics
        self.extra_metric_labels = server_args.extra_metric_labels

        (
            extra_config,
            prefetch_threshold,
            prefetch_timeout_base,
            prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys,
        ) = self._parse_storage_backend_extra_config(
            server_args.hicache_storage_backend_extra_config
        )
        # TODO: support more timeout check functions
        self.is_prefetch_timeout = self._prefetch_timeout_check_linear_func
        self.prefetch_stop_policy = server_args.hicache_storage_prefetch_policy

        self.load_cache_event = threading.Event()
        self.cache_controller = HiCacheController(
            params.token_to_kv_pool_allocator,
            self.token_to_kv_pool_host,
            self.page_size,
            self.tp_group,
            load_cache_event=self.load_cache_event,
            write_policy=server_args.hicache_write_policy,
            io_backend=server_args.hicache_io_backend,
            storage_backend=server_args.hicache_storage_backend,
            prefetch_threshold=prefetch_threshold,
            model_name=server_args.served_model_name,
            storage_backend_extra_config=extra_config,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            enable_storage_metrics=self.enable_storage_metrics,
            dp_rank=getattr(params, "dp_rank", None),
        )
        self._apply_storage_runtime_config(
            storage_backend=server_args.hicache_storage_backend,
            prefetch_threshold=prefetch_threshold,
            prefetch_timeout_base=prefetch_timeout_base,
            prefetch_timeout_per_ki_token=prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
            enable_storage=self.enable_storage,
            enable_storage_metrics=self.enable_storage_metrics,
            extra_metric_labels=self.extra_metric_labels,
        )

        # record the nodes with ongoing write through
        self.ongoing_write_through = {}
        # record the node segments with ongoing load back
        self.ongoing_load_back = {}
        # record the ongoing prefetch requests
        self.ongoing_prefetch = {}
        self.ongoing_backup = {}
        self.ongoing_request_namespace_backup: Dict[int, Tuple[str, int, int]] = {}
        self.request_namespace_synced_tokens_by_reqid_gen: Dict[
            Tuple[str, int], int
        ] = {}
        # track per-request tokens loaded from storage (remote backup hits)
        # key: request_id, value: number of tokens actually loaded from storage
        self.prefetch_loaded_tokens_by_reqid: dict[str, int] = {}
        # track per-request storage query hit tokens (Remote Match count)
        self.storage_query_tokens_by_reqid: dict[str, int] = {}
        # Track imported host-only checkpoint keys by (owner_dp_rank, generation)
        self.imported_host_keys_by_owner_gen: Dict[
            Tuple[int, int], List[Tuple[Optional[str], List[int]]]
        ] = {}
        # todo: dynamically adjust the threshold
        self.write_through_threshold = (
            1 if server_args.hicache_write_policy == "write_through" else 2
        )
        self.load_back_threshold = 10

        # Detach storage backend automatically on process shutdown
        atexit.register(self.shutdown)

        self.evictable_host_leaves = set()

        super().__init__(params=params)

    def shutdown(self):
        """Best-effort auto-detach of storage backend on process shutdown.

        This keeps startup and runtime behavior consistent: if a backend was attached
        (either via CLI args or via admin API), we attempt to detach it on exit.
        """
        try:
            if self.enable_storage:
                self.detach_storage_backend()
        except Exception:
            logger.exception("Failed to detach storage backend on process shutdown.")
        try:
            manager = getattr(
                getattr(self.token_to_kv_pool_host, "shared_config", None),
                "manager",
                None,
            )
            if manager is not None and hasattr(manager, "close"):
                manager.close()
        except Exception:
            logger.exception(
                "Failed to close host shared memory manager on process shutdown."
            )

    def _apply_storage_runtime_config(
        self,
        *,
        storage_backend: Optional[str],
        prefetch_threshold: int,
        prefetch_timeout_base: float,
        prefetch_timeout_per_ki_token: float,
        hicache_storage_pass_prefix_keys: bool,
        enable_storage: bool,
        enable_storage_metrics: bool,
        extra_metric_labels: Optional[Dict[str, str]],
    ) -> None:
        prefetch_timeout_per_page = (
            self.page_size / 1024 * prefetch_timeout_per_ki_token
        )

        self.enable_storage = enable_storage
        self.prefetch_threshold = prefetch_threshold
        self.prefetch_timeout_base = prefetch_timeout_base
        self.prefetch_timeout_per_page = prefetch_timeout_per_page
        self.hicache_storage_pass_prefix_keys = hicache_storage_pass_prefix_keys
        self.enable_storage_metrics = enable_storage_metrics

        if self.enable_storage_metrics:
            labels = {
                "storage_backend": storage_backend,
                "tp_rank": self.cache_controller.tp_rank,
                "dp_rank": self.cache_controller.dp_rank,
                "pp_rank": self.cache_controller.pp_rank,
                "pp_size": self.cache_controller.pp_size,
            }
            if extra_metric_labels:
                labels.update(extra_metric_labels)
            existing_collector = getattr(self, "storage_metrics_collector", None)
            if existing_collector is None:
                self.storage_metrics_collector = StorageMetricsCollector(labels=labels)
            elif set(existing_collector.labels.keys()) == set(labels.keys()):
                existing_collector.labels = labels
            else:
                logger.warning(
                    "Storage metrics labels changed (%s -> %s). Keep existing labels to "
                    "avoid duplicate metric registration.",
                    sorted(existing_collector.labels.keys()),
                    sorted(labels.keys()),
                )

    def attach_storage_backend(
        self,
        storage_backend: str,
        storage_backend_extra_config_json: Optional[str] = None,
        served_model_name: Optional[str] = None,
        hicache_storage_prefetch_policy: Optional[str] = None,
        hicache_write_policy: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Attach (enable) storage backend at runtime.

        This will start storage threads inside `HiCacheController` and enable
        prefetch/backup paths. Caller must ensure there are no running/queued
        requests to avoid races.
        """
        # Validate inputs first (no side effects).
        if hicache_storage_prefetch_policy is not None:
            allowed = ["best_effort", "wait_complete", "timeout"]
            if hicache_storage_prefetch_policy not in allowed:
                return (
                    False,
                    f"Invalid hicache_storage_prefetch_policy: {hicache_storage_prefetch_policy!r}. "
                    f"Expected one of {allowed}.",
                )

        if hicache_write_policy is not None:
            allowed = ["write_back", "write_through", "write_through_selective"]
            if hicache_write_policy not in allowed:
                return (
                    False,
                    f"Invalid hicache_write_policy: {hicache_write_policy!r}. "
                    f"Expected one of {allowed}.",
                )

        # If already enabled:
        # - backend unchanged: treat as success, update policies only.
        # - backend changed: treat as failure, do NOT update policies.
        if self.enable_storage:
            current_backend = self.cache_controller.storage_backend_type

            if current_backend == storage_backend:
                if hicache_storage_prefetch_policy is not None:
                    self.prefetch_stop_policy = hicache_storage_prefetch_policy
                    logger.info(
                        f"Set hicache_storage_prefetch_policy to {hicache_storage_prefetch_policy}"
                    )
                if hicache_write_policy is not None:
                    self.cache_controller.write_policy = hicache_write_policy
                    self.write_through_threshold = (
                        1 if hicache_write_policy == "write_through" else 2
                    )
                    logger.info(f"Set hicache_write_policy to {hicache_write_policy}")
                return (
                    True,
                    "HiCache storage backend already enabled with same backend; policies updated.",
                )

            return (
                False,
                f"HiCache storage backend is already enabled with backend '{current_backend}'. "
                f"Cannot attach different backend '{storage_backend}'. Detach first.",
            )

        # Not enabled: update policies before controller attach so storage threads observe new values.
        if hicache_storage_prefetch_policy is not None:
            self.prefetch_stop_policy = hicache_storage_prefetch_policy
            logger.info(
                f"Set hicache_storage_prefetch_policy to {hicache_storage_prefetch_policy}"
            )

        if hicache_write_policy is not None:
            self.cache_controller.write_policy = hicache_write_policy
            self.write_through_threshold = (
                1 if hicache_write_policy == "write_through" else 2
            )
            logger.info(f"Set hicache_write_policy to {hicache_write_policy}")

        logger.info(f"Attaching HiCache storage backend: {storage_backend}")
        try:
            (
                extra_config,
                prefetch_threshold,
                prefetch_timeout_base,
                prefetch_timeout_per_ki_token,
                hicache_storage_pass_prefix_keys,
            ) = self._parse_storage_backend_extra_config(
                storage_backend_extra_config_json
            )
        except Exception as e:
            logger.exception(f"Failed to parse storage_backend_extra_config_json: {e}")
            return (
                False,
                f"Failed to parse storage_backend_extra_config_json '{storage_backend_extra_config_json}': {e}",
            )

        try:
            self.cache_controller.attach_storage_backend(
                storage_backend=storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=served_model_name,
                storage_backend_extra_config=extra_config,
            )
        except Exception as e:
            logger.exception(
                f"Failed to attach storage backend '{storage_backend}': {e}"
            )
            return False, f"Failed to attach storage backend '{storage_backend}': {e}"

        self._apply_storage_runtime_config(
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            prefetch_timeout_base=prefetch_timeout_base,
            prefetch_timeout_per_ki_token=prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
            enable_storage=True,
            enable_storage_metrics=self._enable_metrics_flag,
            extra_metric_labels=self.extra_metric_labels,
        )
        return True, "Attached HiCache storage backend successfully."

    def detach_storage_backend(self) -> tuple[bool, str]:
        """Detach (disable) storage backend at runtime.

        Caller must ensure there are no running/queued requests to avoid races.
        """
        try:
            # Drain any pending control queues before tearing down storage threads/backend.
            # IMPORTANT: this must happen before we clear `ongoing_*`, otherwise acks/releases
            # cannot be matched to nodes and may leak host pages / locks.
            self._drain_storage_control_queues_local()
            # Idempotent detach: always ask controller to best-effort cleanup, even if
            # `self.enable_storage` is already False (may be leftover state from a
            # previous partial detach).
            self.cache_controller.detach_storage_backend()
        except Exception as e:
            logger.exception("Failed to detach storage backend.")
            # Do NOT crash the server for admin operations. Return failure with detail.
            return False, f"Failed to detach HiCache storage backend: {e}"

        # Best-effort cleanup of any leftover bookkeeping.
        self._drain_storage_control_queues_local()
        # After controller threads are fully stopped, it's safe to force-release any
        # leftover pending ops (e.g., async prefetch/backup that didn't get a revoke/ack).
        self._force_release_pending_storage_ops()

        self.enable_storage = False
        self.enable_storage_metrics = False
        return True, "Detached HiCache storage backend successfully."

    def _force_release_pending_storage_ops(self):
        """Force release any leftover pending prefetch/backup bookkeeping.

        This is a safety net for detach/shutdown paths. It assumes storage threads
        have been stopped already (via controller.detach), so no concurrent access
        to these structures should happen.
        """
        cc = self.cache_controller

        # Force release leftover prefetch ops: free pre-allocated host pages and
        # drop the host protection on the matched prefix node.
        try:
            for req_id, info in list(self.ongoing_prefetch.items()):
                try:
                    last_host_node, token_ids, host_indices, _operation = info
                except Exception:
                    # Unexpected shape; just drop it.
                    self.ongoing_prefetch.pop(req_id, None)
                    continue

                try:
                    if host_indices is not None:
                        cc.mem_pool_host.free(host_indices)
                except Exception:
                    logger.exception(
                        "Failed to free host indices for prefetch %s", req_id
                    )

                try:
                    last_host_node.release_host()
                except Exception:
                    logger.exception(
                        "Failed to release host protection for prefetch %s", req_id
                    )

                try:
                    cc.prefetch_tokens_occupied -= len(token_ids)
                    if cc.prefetch_tokens_occupied < 0:
                        cc.prefetch_tokens_occupied = 0
                except Exception:
                    pass

                self.ongoing_prefetch.pop(req_id, None)
        except Exception:
            logger.exception("Force release pending prefetch ops failed.")

        # Force release leftover backup ops: drop host protection on nodes.
        try:
            for ack_id, node in list(self.ongoing_backup.items()):
                try:
                    node.release_host()
                except Exception:
                    logger.exception(
                        "Failed to release host protection for backup op %s", ack_id
                    )
                self.ongoing_backup.pop(ack_id, None)
        except Exception:
            logger.exception("Force release pending backup ops failed.")

    def _drain_storage_control_queues_local(self):
        """Drain storage control queues without TP synchronization.

        This is intended for shutdown/detach paths where we want to make best-effort
        cleanup even if queue sizes temporarily differ across ranks.
        """
        self._drain_storage_control_queues_impl(
            n_revoke=None,
            n_backup=None,
            n_release=None,
            log_metrics=False,
        )

    def _drain_storage_control_queues_impl(
        self,
        n_revoke: Optional[int],
        n_backup: Optional[int],
        n_release: Optional[int],
        log_metrics: bool,
    ):
        cc = self.cache_controller

        def _drain_queue(q, limit: Optional[int]):
            drained = 0
            while limit is None or drained < limit:
                try:
                    item = q.get_nowait()
                except Empty:
                    break
                drained += 1
                yield item

        def _drain_revoke():
            for req_id in _drain_queue(cc.prefetch_revoke_queue, n_revoke):
                info = self.ongoing_prefetch.pop(req_id, None)
                if info is not None:
                    last_host_node, token_ids, _, _ = info
                    last_host_node.release_host()
                    cc.prefetch_tokens_occupied -= len(token_ids)
                    if cc.prefetch_tokens_occupied < 0:
                        cc.prefetch_tokens_occupied = 0

        def _drain_backup():
            for operation in _drain_queue(cc.ack_backup_queue, n_backup):
                ack_id = operation.id
                ns_entry = self.ongoing_request_namespace_backup.pop(ack_id, None)
                if ns_entry is not None:
                    rid, generation, synced_tokens = ns_entry
                    key = (rid, generation)
                    self.request_namespace_synced_tokens_by_reqid_gen[key] = max(
                        self.request_namespace_synced_tokens_by_reqid_gen.get(key, 0),
                        synced_tokens,
                    )
                entry = self.ongoing_backup.pop(ack_id, None)
                if entry is not None:
                    entry.storage_acked_len = max(
                        int(getattr(entry, "storage_acked_len", 0) or 0),
                        min(len(entry.key), int(operation.completed_tokens or 0)),
                    )
                    entry.release_host()
                if log_metrics and self.enable_storage_metrics:
                    self.storage_metrics_collector.log_backuped_tokens(
                        operation.completed_tokens
                    )

        def _drain_release():
            host_indices_list = []
            for host_indices in _drain_queue(cc.host_mem_release_queue, n_release):
                host_indices_list.append(host_indices)
            if host_indices_list:
                host_indices = torch.cat(host_indices_list, dim=0)
                cc.mem_pool_host.free(host_indices)

        _drain_revoke()
        _drain_backup()
        _drain_release()

    def _parse_storage_backend_extra_config(
        self, storage_backend_extra_config: Optional[str]
    ):
        """
        Parse storage backend extra config JSON and extract specific parameters.

        Args:
            storage_backend_extra_config: JSON string containing extra configuration

        Returns:
            tuple: (extra_config_dict, prefetch_threshold, prefetch_timeout_base, prefetch_timeout_per_ki_token, hicache_storage_pass_prefix_keys)
        """
        # Parse extra config if provided. Extra config can be a JSON string or a json/toml/yaml file path prefixed with "@".
        extra_config = {}
        if storage_backend_extra_config:
            try:
                if storage_backend_extra_config.startswith("@"):
                    # Read config from a json/toml/yaml file
                    path = storage_backend_extra_config[1:]
                    ext = os.path.splitext(path)[1].lower()
                    with open(path, "rb" if ext == ".toml" else "r") as f:
                        if ext == ".json":
                            extra_config = json.load(f)
                        elif ext == ".toml":
                            import tomllib

                            extra_config = tomllib.load(f)
                        elif ext in (".yaml", ".yml"):
                            import yaml

                            extra_config = yaml.safe_load(f)
                        else:
                            raise ValueError(
                                f"Unsupported config file {path} (config format: {ext})"
                            )
                else:
                    # read config from JSON string
                    extra_config = json.loads(storage_backend_extra_config)
            except Exception as e:
                logger.error(f"Invalid backend extra config JSON: {e}")
                raise e

        prefetch_threshold = extra_config.pop("prefetch_threshold", 256)  # tokens
        prefetch_timeout_base = extra_config.pop("prefetch_timeout_base", 1)  # seconds
        prefetch_timeout_per_ki_token = extra_config.pop(
            "prefetch_timeout_per_ki_token", 0.25
        )  # seconds per 1024 tokens
        hicache_storage_pass_prefix_keys = extra_config.pop(
            "hicache_storage_pass_prefix_keys", False
        )

        if not isinstance(prefetch_threshold, int):
            raise ValueError(
                f"prefetch_threshold must be int, got {type(prefetch_threshold).__name__}"
            )
        if not isinstance(prefetch_timeout_base, (int, float)):
            raise ValueError(
                f"prefetch_timeout_base must be number, got {type(prefetch_timeout_base).__name__}"
            )
        if not isinstance(prefetch_timeout_per_ki_token, (int, float)):
            raise ValueError(
                f"prefetch_timeout_per_ki_token must be number, got {type(prefetch_timeout_per_ki_token).__name__}"
            )
        if not isinstance(hicache_storage_pass_prefix_keys, bool):
            raise ValueError(
                "hicache_storage_pass_prefix_keys must be bool, got "
                f"{type(hicache_storage_pass_prefix_keys).__name__}"
            )

        return (
            extra_config,
            prefetch_threshold,
            float(prefetch_timeout_base),
            float(prefetch_timeout_per_ki_token),
            hicache_storage_pass_prefix_keys,
        )

    def reset(self):
        TreeNode.counter = 0
        self.cache_controller.reset()
        self.token_to_kv_pool_host.clear()
        self.ongoing_write_through = {}
        self.ongoing_load_back = {}
        self.ongoing_prefetch = {}
        self.ongoing_backup = {}
        # Clear per-request tracking dicts
        self.prefetch_loaded_tokens_by_reqid.clear()
        self.storage_query_tokens_by_reqid.clear()
        self.imported_host_keys_by_owner_gen.clear()
        self.evictable_host_leaves.clear()
        super().reset()

    def reset_device_state_keep_host(self):
        """Reset device-side state only, preserving host checkpoints."""
        self.cache_controller.reset()
        self.ongoing_write_through = {}
        self.ongoing_load_back = {}
        self.ongoing_prefetch = {}
        self.ongoing_backup = {}
        self.prefetch_loaded_tokens_by_reqid.clear()
        self.storage_query_tokens_by_reqid.clear()

        def _drop_device_state(node: TreeNode):
            if node is not self.root_node:
                node.value = None
                node.lock_ref = 0
            for child in list(node.children.values()):
                _drop_device_state(child)

        _drop_device_state(self.root_node)
        self.root_node.lock_ref = 1
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self.evictable_leaves.clear()

    def get_height(self, node: TreeNode):
        height = 0
        while node != self.root_node:
            node = node.parent
            height += 1
        return height

    def clear_storage_backend(self) -> bool:
        if self.enable_storage:
            try:
                # Check if the storage backend has a clear method (for nixl backends)
                if hasattr(self.cache_controller.storage_backend, "clear"):
                    self.cache_controller.storage_backend.clear()
                    logger.info(
                        "Hierarchical cache storage backend cleared successfully!"
                    )
                    return True
                else:
                    logger.warning(
                        f"Storage backend {type(self.cache_controller.storage_backend).__name__} does not support clear operation."
                    )
                    return False
            except Exception as e:
                logger.error(f"Failed to clear hierarchical cache storage backend: {e}")
                return False
        else:
            logger.warning("Hierarchical cache storage backend is not enabled.")
            return False

    def write_backup(self, node: TreeNode, write_back=False):
        host_indices = self.cache_controller.write(
            device_indices=node.value,
            node_id=node.id,
        )
        if host_indices is None:
            self.evict_host(len(node.value))
            host_indices = self.cache_controller.write(
                device_indices=node.value,
                node_id=node.id,
            )
        if host_indices is not None:
            node.host_value = host_indices.clone()
            assert len(node.host_value) > 0
            self._set_host_value_metadata(
                node,
                owner_dp_rank=int(getattr(self.cache_controller, "dp_rank", 0) or 0),
                generation=0,
                arena_id=getattr(
                    self.cache_controller.mem_pool_host, "shared_arena_id", None
                ),
                imported=False,
            )
            self.ongoing_write_through[node.id] = node
            if not write_back:
                # no need to lock nodes if write back
                self.inc_lock_ref(node)
        else:
            return 0

        return len(host_indices)

    def write_backup_storage(self, node: TreeNode):
        prefix_keys = (
            node.get_prefix_hash_values(node.parent)
            if self.hicache_storage_pass_prefix_keys
            else None
        )
        prefix_token_ids = node.get_prefix_token_ids(node.parent)
        full_token_ids = prefix_token_ids + list(node.key)
        page_start = len(prefix_token_ids) // self.page_size

        operation_id = self.cache_controller.write_storage(
            node.host_value,
            node.key,
            node.hash_value,
            prefix_keys,
            full_token_ids=full_token_ids,
            page_start=page_start,
            request_id=getattr(node, "request_id", None),
            request_generation=int(getattr(node, "request_generation", 0) or 0),
        )
        self.ongoing_backup[operation_id] = node
        node.protect_host()

    def _should_sync_request_namespace(self) -> bool:
        if not self.enable_storage:
            return False
        return getattr(self.cache_controller, "storage_backend_type", None) == (
            "remote_backup"
        )

    def _mirror_request_namespace_prefix(self, req, token_ids: List[int]) -> None:
        """Mirror the host-backed prefix into the request namespace.

        Shared radix nodes may already be backed up from prior requests, but the
        request namespace requires contiguous pages starting from page 0 for each
        individual ``(rid, generation)``. This helper incrementally mirrors the
        current host-backed prefix under the request's own namespace.
        """
        if not self._should_sync_request_namespace():
            return

        rid = getattr(req, "rid", None)
        if not rid:
            return

        generation = int(getattr(req, "remote_backup_generation", 0) or 0)
        sync_key = (rid, generation)
        host_indices = self._collect_host_backed_prefix_indices(token_ids, req.extra_key)
        checkpoint_len = (len(host_indices) // self.page_size) * self.page_size
        synced_tokens = self.request_namespace_synced_tokens_by_reqid_gen.get(
            sync_key, 0
        )
        if checkpoint_len <= synced_tokens:
            return

        host_indices = host_indices[synced_tokens:checkpoint_len]
        if host_indices.numel() == 0:
            return

        page_start = synced_tokens // self.page_size
        page_count = (checkpoint_len - synced_tokens) // self.page_size
        full_token_ids = list(token_ids[:checkpoint_len])
        page_hashes = [f"ns:{rid}:{page_start + i}" for i in range(page_count)]
        operation_id = self.cache_controller.write_storage(
            host_indices,
            full_token_ids[synced_tokens:checkpoint_len],
            hash_value=page_hashes,
            full_token_ids=full_token_ids,
            page_start=page_start,
            request_id=rid,
            request_generation=generation,
        )
        self.ongoing_request_namespace_backup[operation_id] = (
            rid,
            generation,
            checkpoint_len,
        )

    def get_request_namespace_synced_tokens(self, req) -> int:
        rid = getattr(req, "rid", None)
        if not rid:
            return 0

        generation = int(getattr(req, "remote_backup_generation", 0) or 0)
        synced = self.request_namespace_synced_tokens_by_reqid_gen.get(
            (rid, generation), 0
        )
        total_tokens = len(list(getattr(req, "origin_input_ids", [])) + list(getattr(req, "output_ids", [])))
        total_tokens = (total_tokens // self.page_size) * self.page_size
        return min(synced, total_tokens)

    def flush_remote_backup_before_failover(
        self, reqs: List[Any], timeout_s: float = 0.5
    ) -> None:
        """Best-effort bounded flush for request-namespace backups.

        This is used immediately before failover snapshot so in-flight requests
        get one last chance to mirror their current host-backed prefix into the
        request namespace before the failed rank is reset.
        """
        if not self._should_sync_request_namespace() or not reqs:
            return

        targets: Dict[Tuple[str, int], int] = {}
        for req in reqs:
            rid = getattr(req, "rid", None)
            if not rid:
                continue
            token_ids = list(getattr(req, "origin_input_ids", [])) + list(
                getattr(req, "output_ids", [])
            )
            target_tokens = (len(token_ids) // self.page_size) * self.page_size
            if target_tokens <= 0:
                continue
            generation = int(getattr(req, "remote_backup_generation", 0) or 0)
            self._mirror_request_namespace_prefix(req, token_ids)
            targets[(rid, generation)] = target_tokens

        if not targets:
            return

        self.flush_write_through_acks()
        deadline = time.monotonic() + max(0.0, float(timeout_s or 0.0))
        while True:
            self.flush_write_through_acks()
            self._drain_storage_control_queues_local()

            pending = []
            for key, target_tokens in targets.items():
                synced = self.request_namespace_synced_tokens_by_reqid_gen.get(key, 0)
                if synced < target_tokens:
                    pending.append((key, synced, target_tokens))

            if not pending:
                return
            if time.monotonic() >= deadline:
                break
            time.sleep(0.01)

        logger.warning(
            "Timed out waiting for request-namespace backups before failover: %s "
            "(backup_queue=%s ack_backup_queue=%s ongoing_backup=%s ns_ongoing=%s)",
            ", ".join(
                f"{rid[:16]}@g{generation}:{synced}/{target}"
                for (rid, generation), synced, target in pending
            ),
            self.cache_controller.backup_queue.qsize(),
            self.cache_controller.ack_backup_queue.qsize(),
            len(self.ongoing_backup),
            len(self.ongoing_request_namespace_backup),
        )

    def _inc_hit_count(self, node: TreeNode, chunked=False):
        if self.cache_controller.write_policy == "write_back" or chunked:
            return
        node.hit_count += 1

        if not node.backuped:
            if node.hit_count >= self.write_through_threshold:
                # write to host if the node is not backuped
                self.write_backup(node)

    def writing_check(self, write_back=False):
        if write_back:
            # blocking till all write back complete
            while len(self.ongoing_write_through) > 0:
                for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
                    finish_event.synchronize()
                    for ack_id in ack_list:
                        backuped_node = self.ongoing_write_through.pop(ack_id)
                        if self.enable_storage:
                            self.write_backup_storage(backuped_node)
                self.cache_controller.ack_write_queue.clear()
                assert len(self.ongoing_write_through) == 0
            return

        # NOTE: all ranks has the same ongoing_write_through, can skip sync if empty
        if len(self.ongoing_write_through) == 0:
            return

        finish_count = 0
        for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
            if not finish_event.query():
                break
            finish_count += 1
        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        if self.tp_world_size > 1:
            # synchronize TP workers to make the same update to radix cache
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )

        finish_count = int(queue_size.item())
        while finish_count > 0:
            _, finish_event, ack_list = self.cache_controller.ack_write_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                backuped_node = self.ongoing_write_through.pop(ack_id)
                self.dec_lock_ref(backuped_node)
                if self.enable_storage:
                    self.write_backup_storage(backuped_node)
            finish_count -= 1

    def loading_check(self):
        finish_count = 0
        for _, finish_event, ack_list in self.cache_controller.ack_load_queue:
            if not finish_event.query():
                # the KV cache loading is still ongoing
                break
            finish_count += 1
            # no need to sync across TP workers as batch forwarding is synced
            for ack_id in ack_list:
                end_node = self.ongoing_load_back.pop(ack_id)
                self.dec_lock_ref(end_node)

        # ACK until all events are processed
        del self.cache_controller.ack_load_queue[:finish_count]

    def evictable_size(self):
        return self.evictable_size_

    def _to_radix_key(self, token_ids: List[int]) -> RadixKey:
        """Convert raw token_ids to a RadixKey for tree walking.

        Must use list (not tuple) to match scheduler's RadixKey format,
        since _key_match_paged compares slices directly and list != tuple.
        """
        return RadixKey(token_ids=list(token_ids))

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        if self.disable:
            return IncLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)
            node = node.parent
        return IncLockRefResult(delta=delta)

    def dec_lock_ref(
        self, node: TreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if self.disable:
            return DecLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)
            if node.parent is None:
                assert (
                    node is self.root_node
                ), f"This request holds the node from another tree"
            node = node.parent
        return DecLockRefResult(delta=delta)

    def _update_host_leaf_status(self, node: TreeNode):
        if not node.evicted or node.lock_ref > 0:
            if node in self.evictable_host_leaves:
                self.evictable_host_leaves.remove(node)
            return

        for child in node.children.values():
            if child.evicted:
                if node in self.evictable_host_leaves:
                    self.evictable_host_leaves.remove(node)
                return

        if node not in self.evictable_host_leaves:
            self.evictable_host_leaves.add(node)

    def evict(self, params: EvictParams) -> EvictResult:
        start_time = time.perf_counter()
        num_tokens = params.num_tokens
        leaves = list(self.evictable_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        write_back_nodes = []
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            if x.lock_ref > 0:
                continue

            if not x.backuped:
                if self.cache_controller.write_policy == "write_back":
                    # write to host if the node is not backuped
                    num_evicted += self.write_backup(x, write_back=True)
                    write_back_nodes.append(x)
                else:
                    num_evicted += self._evict_regular(x)
            else:
                num_evicted += self._evict_backuped(x)

            for child in x.parent.children.values():
                if child in write_back_nodes:
                    continue
                if not child.evicted:
                    break
            else:
                # all children are evicted or no children
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

        if self.cache_controller.write_policy == "write_back":
            self.writing_check(write_back=True)
            for node in write_back_nodes:
                assert node.backuped
                self._evict_backuped(node)

        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

    def _evict_backuped(self, node: TreeNode):
        # GPU -> CPU demotion: no BlockRemoved since block is still reachable via load_back
        num_evicted = self.cache_controller.evict_device(node.value)
        assert num_evicted > 0
        self.evictable_size_ -= num_evicted
        node.value = None
        self._update_leaf_status(node)
        self._update_host_leaf_status(node)
        # update leaf status for the parent because the node is evicted
        self._update_leaf_status(node.parent)
        return num_evicted

    def _evict_regular(self, node: TreeNode):
        # evict a node not initiated write to host -- emit BlockRemoved
        self._record_remove_event(node)
        self.cache_controller.mem_pool_device_allocator.free(node.value)
        num_evicted = len(node.value)
        self._delete_leaf(node)
        return num_evicted

    def evict_host(self, num_tokens: int):
        leaves = list(self.evictable_host_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)
            if x == self.root_node:
                break
            # only evict the host value of evicted nodes
            if not x.evicted:
                continue

            if x.host_ref_counter > 0:
                continue

            # Block deleted entirely (GPU already evicted, now CPU freed) --
            # emit BlockRemoved so the router removes this block from its index.
            self._record_remove_event(x)
            num_evicted += self._release_host_value(x)

            key = self.get_child_key_fn(x.key)
            v = x.parent.children.pop(key, None)
            assert v == x, f"parent does not have child key, {key}"
            if x in self.evictable_host_leaves:
                self.evictable_host_leaves.remove(x)
            self._update_host_leaf_status(x.parent)

            if len(x.parent.children) == 0 and x.parent.evicted:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

    def load_back(
        self, node: TreeNode, mem_quota: Optional[int] = None
    ) -> Optional[torch.Tensor]:

        start_time = time.perf_counter()
        last_hit_node = node
        nodes_to_load = []
        while node.evicted:
            assert (
                node.backuped
            ), "No backup available on evicted nodes, should not happen"
            nodes_to_load.insert(0, node)
            node = node.parent
        else:
            ancester_node = node

        # protect the ancestor nodes from eviction
        result = self.inc_lock_ref(ancester_node)
        delta = result.delta

        # load it all or not at all
        host_indices = torch.cat([n.host_value for n in nodes_to_load])
        if len(host_indices) < self.load_back_threshold or (
            len(host_indices) > mem_quota + delta if mem_quota is not None else False
        ):
            # skip loading back if the total size is too small or exceeding the memory quota
            self.dec_lock_ref(ancester_node)
            return None

        device_indices = self.cache_controller.load(
            host_indices=host_indices, node_id=last_hit_node.id
        )
        if device_indices is None:
            self.evict(EvictParams(num_tokens=len(host_indices)))
            device_indices = self.cache_controller.load(
                host_indices=host_indices, node_id=last_hit_node.id
            )
        self.dec_lock_ref(ancester_node)
        if device_indices is None:
            # no sufficient GPU memory to load back KV caches
            logger.warning(
                "load_back: FAILED to load %d tokens for node %d "
                "even after eviction (evictable_size=%d)",
                len(host_indices),
                last_hit_node.id,
                self.evictable_size_,
            )
            return None

        self.ongoing_load_back[last_hit_node.id] = last_hit_node
        self.inc_lock_ref(last_hit_node)
        offset = 0
        for node in nodes_to_load:
            node.value = device_indices[offset : offset + len(node.host_value)].clone()
            offset += len(node.host_value)
        self.evictable_size_ += len(device_indices)

        if self.metrics_collector is not None:
            self.metrics_collector.observe_load_back_duration(
                time.perf_counter() - start_time
            )
            self.metrics_collector.increment_load_back_num_tokens(len(device_indices))

        return device_indices

    def init_load_back(
        self,
        params: InitLoadBackParams,
    ):
        last_node = params.last_host_node
        mem_quota = params.mem_quota
        if last_node.evicted:
            loading_values = self.load_back(last_node, mem_quota)
            if loading_values is not None:
                logger.debug(
                    f"loading back {len(loading_values)} tokens for node {last_node.id}"
                )
                return loading_values, last_node

            while last_node.evicted:
                last_node = last_node.parent

        return (
            torch.empty((0,), dtype=torch.int64, device=self.device),
            last_node,
        )

    def ready_to_load_host_cache(self) -> int:
        """
        Notify the cache controller to start the KV cache loading.
        Return the consumer index for the schedule batch manager to track.
        """
        return self.cache_controller.start_loading()

    def flush_write_through_acks(self) -> None:
        self.writing_check()

    def check_hicache_events(self):
        self.writing_check()
        self.loading_check()
        if self.enable_storage:
            self.drain_storage_control_queues()
        if self.enable_storage_metrics:
            self.storage_metrics_collector.log_storage_metrics(
                self.cache_controller.storage_backend.get_stats()
            )

    def drain_storage_control_queues(self):
        """
        Combine prefetch revoke, backup ack, and host mem release checks
        to minimize TP synchronization and Python overhead.
        """
        if not self.enable_storage:
            return
        cc = self.cache_controller

        qsizes = torch.tensor(
            [
                cc.prefetch_revoke_queue.qsize(),
                cc.ack_backup_queue.qsize(),
                cc.host_mem_release_queue.qsize(),
            ],
            dtype=torch.int,
        )
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                qsizes, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )

        n_revoke, n_backup, n_release = map(int, qsizes.tolist())
        self._drain_storage_control_queues_impl(
            n_revoke=n_revoke,
            n_backup=n_backup,
            n_release=n_release,
            log_metrics=True,
        )

    # Timeout is linearly increasing with the number of pages
    def _prefetch_timeout_check_linear_func(self, operation: PrefetchOperation):
        # If hash_value has not been computed in timeout_base seconds, terminate it.
        return (
            time.monotonic() - operation.start_time
            > self.prefetch_timeout_base
            + len(operation.hash_value) * self.prefetch_timeout_per_page
        )

    def can_terminate_prefetch(self, operation: PrefetchOperation):
        can_terminate = True

        if self.prefetch_stop_policy == "best_effort":
            return can_terminate

        if len(operation.hash_value) == 0:
            completed = False
        else:
            completed = (
                operation.completed_tokens == len(operation.hash_value) * self.page_size
            )

        if self.prefetch_stop_policy == "wait_complete":
            can_terminate = completed
        elif self.prefetch_stop_policy == "timeout":
            can_terminate = completed or self.is_prefetch_timeout(operation)
        else:
            # unknown prefetch stop policy, just return True
            return True

        operation_terminated = operation.is_terminated()
        if self.tp_world_size > 1:
            states = torch.tensor(
                [1 - int(can_terminate), int(operation_terminated)],
                dtype=torch.int,
            )
            torch.distributed.all_reduce(
                states,
                op=torch.distributed.ReduceOp.MAX,
                group=self.tp_group,
            )
            can_terminate = states[0].item() == 0
            operation_terminated = states[1].item() == 1
        # the operation should be terminated if it is already terminated on any TP worker
        # or it meets the termination condition on all TP workers
        can_terminate = can_terminate or operation_terminated
        return can_terminate

    def check_prefetch_progress(self, req_id: str) -> bool:
        if req_id not in self.ongoing_prefetch:
            # there is no ongoing prefetch for this request or it has been revoked
            return True

        # todo: more policies for prefetch progress such as timeout
        # the current policy is to prefetch with best effort and terminate when queuing is over
        last_host_node, token_ids, host_indices, operation = self.ongoing_prefetch[
            req_id
        ]

        if operation.host_indices is None:
            # prefetch has not been issued due to insufficient host memory
            return True

        if not self.can_terminate_prefetch(operation):
            return False

        completed_tokens, hash_value = self.cache_controller.terminate_prefetch(
            operation
        )
        logger.debug(f"Prefetch {req_id} completed with {completed_tokens} tokens")

        min_completed_tokens = completed_tokens
        if self.tp_world_size > 1:
            # synchrnoize TP workers to make the same update to hiradix cache
            completed_tokens_tensor = torch.tensor(
                min_completed_tokens, dtype=torch.int
            )
            torch.distributed.all_reduce(
                completed_tokens_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
            min_completed_tokens = completed_tokens_tensor.item()
        fetched_token_ids = token_ids[:min_completed_tokens]
        written_indices = host_indices[:min_completed_tokens]
        matched_length = self._insert_helper_host(
            last_host_node,
            RadixKey(
                token_ids=fetched_token_ids, extra_key=last_host_node.key.extra_key
            ),
            written_indices,
            hash_value[: min_completed_tokens // self.page_size],
            owner_dp_rank=int(getattr(self.cache_controller, "dp_rank", 0) or 0),
            generation=0,
            arena_id=getattr(self.cache_controller.mem_pool_host, "shared_arena_id", None),
            imported=False,
        )

        self.cache_controller.mem_pool_host.free(host_indices[:matched_length])
        self.cache_controller.append_host_mem_release(
            host_indices[min_completed_tokens:completed_tokens]
        )
        last_host_node.release_host()
        del self.ongoing_prefetch[req_id]
        self.cache_controller.prefetch_tokens_occupied -= len(token_ids)

        # Track completed storage IO for this request. Some of these tokens may
        # match existing host radix nodes; keep the completed count for response
        # accounting and compute actual reuse later from host_hit_length.
        self.prefetch_loaded_tokens_by_reqid[req_id] = min_completed_tokens
        self.storage_query_tokens_by_reqid[req_id] = getattr(
            operation, "storage_query_count", 0
        )

        if self.enable_storage_metrics:
            self.storage_metrics_collector.log_prefetched_tokens(min_completed_tokens)

        return True

    def terminate_prefetch(self, req_id: str):
        if req_id not in self.ongoing_prefetch:
            return

        _, _, _, operation = self.ongoing_prefetch[req_id]
        if operation.host_indices is None:
            return
        operation.mark_terminate()

    def pop_prefetch_loaded_tokens(self, req_id: str) -> int:
        """
        Pop and return the number of tokens loaded from storage for a request.
        Returns 0 if no prefetch was done or was revoked.
        This should be called after check_prefetch_progress() returns True.
        """
        return self.prefetch_loaded_tokens_by_reqid.pop(req_id, 0)

    def pop_storage_query_tokens(self, req_id: str) -> int:
        """Pop and return the storage query hit token count for a request."""
        return self.storage_query_tokens_by_reqid.pop(req_id, 0)

    def count_backed_up_tokens(self, token_ids: list) -> int:
        """Count tokens along the radix tree path that have been backed up to host."""
        key = RadixKey(token_ids=token_ids, extra_key=None)
        key, _ = self.maybe_bigram_convert(key)
        if len(key) == 0:
            return 0
        node = self.root_node
        child_key = self.get_child_key_fn(key)
        total = 0
        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]
            prefix_len = self.key_match_fn(child.key, key)
            if child.backuped:
                total += min(prefix_len, len(child.key))
            if prefix_len < len(child.key):
                break
            node = child
            key = key[prefix_len:]
            if len(key):
                child_key = self.get_child_key_fn(key)
        return total

    def count_remote_acked_tokens(self, token_ids: list) -> int:
        """Count tokens along the radix tree path that were acked by remote storage."""
        key = RadixKey(token_ids=token_ids, extra_key=None)
        key, _ = self.maybe_bigram_convert(key)
        if len(key) == 0:
            return 0
        node = self.root_node
        child_key = self.get_child_key_fn(key)
        total = 0
        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]
            prefix_len = self.key_match_fn(child.key, key)
            if child.storage_acked_len > 0:
                total += min(prefix_len, child.storage_acked_len)
            if prefix_len < len(child.key):
                break
            node = child
            key = key[prefix_len:]
            if len(key):
                child_key = self.get_child_key_fn(key)
        return total

    def _collect_host_backed_prefix_indices(
        self, token_ids: List[int], extra_key: Optional[str]
    ) -> torch.Tensor:
        key = RadixKey(token_ids=token_ids, extra_key=extra_key)
        key, _ = self.maybe_bigram_convert(key)
        if len(key) == 0:
            return torch.empty((0,), dtype=torch.int64)

        node = self.root_node
        child_key = self.get_child_key_fn(key)
        values: List[torch.Tensor] = []
        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]
            prefix_len = self.key_match_fn(child.key, key)
            if not child.backuped:
                break
            take_len = min(prefix_len, len(child.host_value))
            if take_len <= 0:
                break
            values.append(child.host_value[:take_len])
            if prefix_len < len(child.key):
                break
            node = child
            key = key[prefix_len:]
            if len(key):
                child_key = self.get_child_key_fn(key)
        if not values:
            return torch.empty((0,), dtype=torch.int64)
        return torch.cat(values)

    def _host_indices_to_descriptors(
        self, host_indices: torch.Tensor
    ) -> List[HostCheckpointDescriptor]:
        if host_indices.numel() == 0:
            return []
        cpu_indices = host_indices.detach().cpu().to(torch.int64)
        descriptors: List[HostCheckpointDescriptor] = []
        run_start = int(cpu_indices[0].item())
        prev = run_start
        for idx in cpu_indices[1:].tolist():
            idx = int(idx)
            if idx == prev + 1:
                prev = idx
                continue
            descriptors.append(
                HostCheckpointDescriptor(
                    slot_start=run_start,
                    slot_count=prev - run_start + 1,
                )
            )
            run_start = idx
            prev = idx
        descriptors.append(
            HostCheckpointDescriptor(
                slot_start=run_start,
                slot_count=prev - run_start + 1,
            )
        )
        return descriptors

    def _descriptor_batch_to_host_indices(
        self, descriptors: List[HostCheckpointDescriptor]
    ) -> torch.Tensor:
        if not descriptors:
            return torch.empty((0,), dtype=torch.int64)
        pieces = [
            torch.arange(
                int(descriptor.slot_start),
                int(descriptor.slot_start) + int(descriptor.slot_count),
                dtype=torch.int64,
            )
            for descriptor in descriptors
            if descriptor.slot_count > 0
        ]
        if not pieces:
            return torch.empty((0,), dtype=torch.int64)
        return torch.cat(pieces)

    def _set_host_value_metadata(
        self,
        node: TreeNode,
        *,
        owner_dp_rank: int,
        generation: int,
        arena_id: Optional[str],
        imported: bool,
    ) -> None:
        node.host_owner_dp_rank = int(owner_dp_rank)
        node.host_generation = int(generation)
        node.host_arena_id = arena_id
        node.host_imported = imported

    def _release_host_value(self, node: TreeNode) -> int:
        if not node.backuped:
            return 0
        if getattr(node, "host_imported", False):
            released = len(node.host_value)
        else:
            released = self.cache_controller.evict_host(node.host_value)
        node.host_value = None
        node.host_owner_dp_rank = None
        node.host_generation = 0
        node.host_arena_id = None
        node.host_imported = False
        return released

    def export_failure_checkpoints(self, req) -> List[Dict[str, Any]]:
        all_ids = list(req.origin_input_ids) + list(req.output_ids)
        host_indices = self._collect_host_backed_prefix_indices(all_ids, req.extra_key)
        checkpoint_len = (len(host_indices) // self.page_size) * self.page_size
        if checkpoint_len <= 0:
            return []

        host_indices = host_indices[:checkpoint_len]
        token_prefix = all_ids[:checkpoint_len]
        mem_pool_host = self.cache_controller.mem_pool_host
        descriptors = []
        if hasattr(mem_pool_host, "descriptor_context") and mem_pool_host.is_shared_across_workers():
            descriptors = [
                descriptor.__dict__
                for descriptor in self._host_indices_to_descriptors(host_indices)
            ]
            descriptor_context = mem_pool_host.descriptor_context()
            pages: List[torch.Tensor] = []
        else:
            descriptor_context = {
                "arena_id": None,
                "layout": getattr(mem_pool_host, "layout", None),
                "page_size": self.page_size,
            }
            pages = []
            for i in range(0, checkpoint_len, self.page_size):
                page_index = int(host_indices[i].item())
                pages.append(mem_pool_host.get_data_page(page_index, flat=True).clone())

        metadata = FailureCheckpointMetadata(
            owner_dp_rank=int(getattr(self.cache_controller, "dp_rank", 0) or 0),
            generation=int(getattr(req, "remote_backup_generation", 0)),
            rid=req.rid,
            extra_key=req.extra_key,
            checkpoint_len=checkpoint_len,
            prefix_token_ids=token_prefix,
            arena_id=descriptor_context["arena_id"],
            layout=descriptor_context["layout"],
            page_size=int(descriptor_context["page_size"] or self.page_size),
            descriptors=descriptors,
            pages=pages,
        )
        logger.info(
            "host-backup export rid=%s tokens=%d descriptors=%d tensor_pages=%d",
            req.rid,
            checkpoint_len,
            len(descriptors),
            len(pages),
        )
        return [metadata.__dict__]

    def cache_finished_req(self, req, is_insert: bool = True):
        super().cache_finished_req(req, is_insert=is_insert)
        self._mirror_request_namespace_prefix(
            req, list(req.origin_input_ids) + list(req.output_ids)
        )

    def cache_unfinished_req(self, req, chunked=False):
        super().cache_unfinished_req(req, chunked=chunked)
        self._mirror_request_namespace_prefix(req, list(req.fill_ids))

    def import_host_checkpoints(self, metadata_batch) -> int:
        if not metadata_batch:
            return 0
        imported_tokens = 0
        for raw_meta in metadata_batch:
            if not raw_meta:
                continue
            meta = FailureCheckpointMetadata(**raw_meta)
            if (
                meta.checkpoint_len <= 0
                or meta.checkpoint_len % self.page_size != 0
                or len(meta.prefix_token_ids) < meta.checkpoint_len
            ):
                continue

            mem_pool_host = self.cache_controller.mem_pool_host
            descriptor_mode = bool(meta.descriptors) and hasattr(
                mem_pool_host, "descriptor_is_compatible"
            )
            descriptor_compatible = False
            if descriptor_mode:
                descriptor_compatible = mem_pool_host.descriptor_is_compatible(
                    arena_id=meta.arena_id,
                    layout=meta.layout,
                    page_size=meta.page_size,
                )
                logger.info(
                    "import_host_checkpoints: rid=%s checkpoint_len=%d "
                    "descriptors=%d pages=%d descriptor_compatible=%s "
                    "meta_arena_id=%s local_arena_id=%s meta_layout=%s "
                    "local_layout=%s meta_page_size=%s local_page_size=%s",
                    meta.rid,
                    meta.checkpoint_len,
                    len(meta.descriptors),
                    len(meta.pages),
                    descriptor_compatible,
                    meta.arena_id,
                    getattr(mem_pool_host, "shared_arena_id", None),
                    meta.layout,
                    getattr(mem_pool_host, "layout", None),
                    meta.page_size,
                    getattr(mem_pool_host, "page_size", None),
                )
            if descriptor_mode and descriptor_compatible:
                descriptors = [
                    HostCheckpointDescriptor(**descriptor)
                    for descriptor in meta.descriptors
                ]
                host_indices = self._descriptor_batch_to_host_indices(descriptors)
                if len(host_indices) < meta.checkpoint_len:
                    logger.warning(
                        "import_host_checkpoints: descriptor length mismatch for rid=%s",
                        meta.rid,
                    )
                    continue
                host_indices = host_indices[: meta.checkpoint_len]
                required_pages = meta.checkpoint_len // self.page_size
            else:
                required_pages = meta.checkpoint_len // self.page_size
                if len(meta.pages) < required_pages:
                    logger.warning(
                        "import_host_checkpoints: skipping rid=%s because "
                        "descriptor_mode=%s descriptor_compatible=%s fallback_pages=%d "
                        "required_pages=%d",
                        meta.rid,
                        descriptor_mode,
                        descriptor_compatible,
                        len(meta.pages),
                        required_pages,
                    )
                    continue

                host_indices = mem_pool_host.alloc(meta.checkpoint_len)
                if host_indices is None:
                    self.evict_host(meta.checkpoint_len)
                    host_indices = mem_pool_host.alloc(meta.checkpoint_len)
                if host_indices is None:
                    logger.warning(
                        "import_host_checkpoints: insufficient host memory for %d tokens",
                        meta.checkpoint_len,
                    )
                    continue

                for page_i in range(required_pages):
                    dst_index = int(host_indices[page_i * self.page_size].item())
                    mem_pool_host.set_from_flat_data_page(dst_index, meta.pages[page_i])

            matched_length = self._insert_helper_host(
                self.root_node,
                RadixKey(
                    token_ids=meta.prefix_token_ids[: meta.checkpoint_len],
                    extra_key=meta.extra_key,
                ),
                host_indices,
                [None] * required_pages,
                owner_dp_rank=meta.owner_dp_rank,
                generation=meta.generation,
                arena_id=meta.arena_id,
                imported=descriptor_mode,
            )
            if matched_length > 0 and not descriptor_mode:
                mem_pool_host.free(host_indices[:matched_length])

            imported_tokens += max(0, meta.checkpoint_len - matched_length)
            self.imported_host_keys_by_owner_gen.setdefault(
                (meta.owner_dp_rank, meta.generation), []
            ).append((meta.extra_key, meta.prefix_token_ids[: meta.checkpoint_len]))
            logger.info(
                "host-backup import rid=%s tokens=%d matched=%d descriptors=%d fallback_pages=%d",
                meta.rid,
                meta.checkpoint_len,
                matched_length,
                len(meta.descriptors),
                len(meta.pages),
            )
        return imported_tokens

    def _delete_host_only_prefix(
        self, prefix_token_ids: List[int], extra_key: Optional[str]
    ) -> bool:
        key = RadixKey(token_ids=prefix_token_ids, extra_key=extra_key)
        key, _ = self.maybe_bigram_convert(key)
        if len(key) == 0:
            return False
        node = self.root_node
        child_key = self.get_child_key_fn(key)
        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                return False
            node = child
            key = key[prefix_len:]
            if len(key):
                child_key = self.get_child_key_fn(key)
        if len(key) != 0:
            return False

        deleted = False
        while node is not self.root_node:
            if (
                len(node.children) > 0
                or not node.evicted
                or node.host_ref_counter > 0
                or node.lock_ref > 0
            ):
                break
            parent = node.parent
            if node.backuped:
                self._release_host_value(node)
            if node in self.evictable_host_leaves:
                self.evictable_host_leaves.remove(node)
            k = self.get_child_key_fn(node.key)
            parent.children.pop(k, None)
            self._update_host_leaf_status(parent)
            node = parent
            deleted = True
        return deleted

    def invalidate_imported_generation(
        self, owner_dp_rank: int, generation: int
    ) -> None:
        keys_to_drop: List[Tuple[int, int]] = []
        for key, imported in list(self.imported_host_keys_by_owner_gen.items()):
            owner, gen = key
            if owner != owner_dp_rank:
                continue
            if generation >= 0 and gen != generation:
                continue
            for extra_key, prefix_ids in imported:
                self._delete_host_only_prefix(prefix_ids, extra_key)
            keys_to_drop.append(key)
        for key in keys_to_drop:
            self.imported_host_keys_by_owner_gen.pop(key, None)

    def match_prefix(self, params: MatchPrefixParams):
        key = params.key
        empty_value = torch.empty((0,), dtype=torch.int64, device=self.device)
        key, _ = self.maybe_bigram_convert(key)
        if self.disable or len(key) == 0:
            return MatchResult(
                device_indices=empty_value,
                last_device_node=self.root_node,
                last_host_node=self.root_node,
                host_hit_length=0,
            )

        page_aligned_len = len(key)
        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = empty_value

        host_hit_length = 0
        last_host_node = last_node
        while last_node.evicted:
            host_hit_length += len(last_node.host_value)
            last_node = last_node.parent
        while not last_host_node.backuped:
            last_host_node = last_host_node.parent

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_host_node,
            host_hit_length=host_hit_length,
        )

    def prefetch_from_storage(
        self,
        req_id: str,
        last_host_node: TreeNode,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        prefix_token_ids: Optional[List[int]] = None,
        lookup_dp_rank: Optional[int] = None,
        request_ns_restore: bool = False,
        ns_dp_rank: int = 0,
        ns_generation: int = 0,
    ) -> str:
        new_input_tokens = (
            convert_to_bigram_key(new_input_tokens)
            if self.is_eagle
            else new_input_tokens
        )
        # align the number of fetching tokens to the page size
        prefetch_length = len(new_input_tokens) - (
            len(new_input_tokens) % self.page_size
        )
        new_input_tokens = new_input_tokens[:prefetch_length]
        rate_limited = self.cache_controller.prefetch_rate_limited()
        if (
            not self.enable_storage
            or prefetch_length < self.prefetch_threshold
            or rate_limited
        ):
            return "rate_limited" if rate_limited else "skipped"

        last_host_node.protect_host()
        host_indices = self.cache_controller.mem_pool_host.alloc(prefetch_length)
        if host_indices is None:
            self.evict_host(prefetch_length)
            host_indices = self.cache_controller.mem_pool_host.alloc(prefetch_length)
        if host_indices is None:
            last_host_node.release_host()
            # no sufficient host memory for prefetch
            return "skipped"
        operation = self.cache_controller.prefetch(
            req_id,
            host_indices,
            new_input_tokens,
            last_hash,
            prefix_keys,
            prefix_token_ids=prefix_token_ids,
            lookup_dp_rank=lookup_dp_rank,
            request_ns_restore=request_ns_restore,
            ns_dp_rank=ns_dp_rank,
            ns_generation=ns_generation,
        )
        self.ongoing_prefetch[req_id] = (
            last_host_node,
            new_input_tokens,
            host_indices,
            operation,
        )
        self.cache_controller.prefetch_tokens_occupied += len(new_input_tokens)
        return "dispatched"

    def _insert_helper_host(
        self,
        node: TreeNode,
        key: RadixKey,
        host_value,
        hash_value,
        *,
        owner_dp_rank: int,
        generation: int,
        arena_id: Optional[str],
        imported: bool,
    ):
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        matched_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(node.key, key)
            key = key[prefix_len:]
            host_value = host_value[prefix_len:]
            hash_value = hash_value[prefix_len // self.page_size :]
            matched_length += prefix_len

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode(priority=node.priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = None
            new_node.host_value = host_value.clone()
            self._set_host_value_metadata(
                new_node,
                owner_dp_rank=owner_dp_rank,
                generation=generation,
                arena_id=arena_id,
                imported=imported,
            )
            new_node.storage_acked_len = len(key)
            new_node.hash_value = hash_value
            node.children[child_key] = new_node
            self._update_host_leaf_status(new_node)
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)

        return matched_length

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        node.last_access_time = time.monotonic()
        child_key = self.get_child_key_fn(key)
        value = []

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                if not new_node.evicted:
                    value.append(new_node.value)
                node = new_node
                break
            else:
                if not child.evicted:
                    value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)

        return value, node

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        # child node split into new_node -> child
        new_node = TreeNode(priority=child.priority)
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.request_id = getattr(child, "request_id", None)
        new_node.request_generation = int(
            getattr(child, "request_generation", 0) or 0
        )
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.hit_count = child.hit_count
        new_node.storage_acked_len = min(split_len, child.storage_acked_len)

        # split value and host value if exists
        if child.evicted:
            new_node.value = None
        else:
            new_node.value = child.value[:split_len].clone()
            child.value = child.value[split_len:].clone()
        if child.backuped:
            new_node.host_value = child.host_value[:split_len].clone()
            child.host_value = child.host_value[split_len:].clone()
            new_node.host_owner_dp_rank = getattr(child, "host_owner_dp_rank", None)
            new_node.host_generation = getattr(child, "host_generation", 0)
            new_node.host_arena_id = getattr(child, "host_arena_id", None)
            new_node.host_imported = getattr(child, "host_imported", False)

        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )
        child.parent = new_node
        child.key = child.key[split_len:]
        child.storage_acked_len = max(0, child.storage_acked_len - split_len)
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        return new_node

    def insert(self, params: InsertParams) -> InsertResult:
        key = params.key
        value = params.value
        chunked = params.chunked
        priority = params.priority
        request_id = params.request_id
        request_generation = int(params.request_generation or 0)

        if priority is None:
            priority = 0
        key, value = self.maybe_bigram_convert(key, value)

        if len(key) == 0:
            return InsertResult(prefix_len=0)

        if self.is_eagle and value is not None:
            # Make sure the value len equal to the EAGLE bigram key len
            value = value[: len(key)]

        node = self.root_node
        child_key = self.get_child_key_fn(key)
        total_prefix_length = 0

        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            node.priority = max(node.priority, priority)
            if request_id:
                node.request_id = request_id
                node.request_generation = request_generation
            prefix_len = self.key_match_fn(node.key, key)

            if prefix_len == len(node.key):
                if node.evicted:
                    # change the reference if the node is evicted
                    # this often happens in the case of KV cache recomputation
                    node.value = value[:prefix_len].clone()
                    self.evictable_size_ += len(node.value)
                    self._update_leaf_status(node)
                    self._update_host_leaf_status(node)
                    # update parent status as a new leaf is added into device
                    self._update_leaf_status(node.parent)
                else:
                    self._inc_hit_count(node, chunked)
                    total_prefix_length += prefix_len
            else:
                # partial match, split the node
                new_node = self._split_node(node.key, node, prefix_len)
                # shared-prefix node should also reflect max priority
                new_node.priority = max(new_node.priority, priority)
                if request_id:
                    new_node.request_id = request_id
                    new_node.request_generation = request_generation
                if new_node.evicted:
                    new_node.value = value[:prefix_len].clone()
                    self.evictable_size_ += len(new_node.value)
                    self._update_leaf_status(new_node)
                    self._update_host_leaf_status(new_node)
                    # update parent status as a new leaf is added into device
                    self._update_leaf_status(new_node.parent)
                else:
                    self._inc_hit_count(new_node, chunked)
                    total_prefix_length += prefix_len
                node = new_node

            key = key[prefix_len:]
            value = value[prefix_len:]

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            new_node.request_id = request_id
            new_node.request_generation = request_generation
            node.children[child_key] = new_node
            self.evictable_size_ += len(value)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)

            # Compute hash_value if storage or kv events are enabled
            if self.enable_storage or self.enable_kv_cache_events:
                new_node.hash_value = compute_node_hash_values(new_node, self.page_size)

            # Emit BlockStored so the router indexes this block.
            self._record_store_event(new_node)

            if self.cache_controller.write_policy != "write_back":
                self._inc_hit_count(new_node, chunked)
        return InsertResult(prefix_len=total_prefix_length)

    def release_aborted_request(self, rid: str):
        # Clean up storage hit tracking for aborted request
        self.prefetch_loaded_tokens_by_reqid.pop(rid, None)
        self.storage_query_tokens_by_reqid.pop(rid, None)
        stale_sync_keys = [
            key
            for key in self.request_namespace_synced_tokens_by_reqid_gen
            if key[0] == rid
        ]
        for key in stale_sync_keys:
            self.request_namespace_synced_tokens_by_reqid_gen.pop(key, None)

        if rid not in self.ongoing_prefetch:
            return

        last_host_node, token_ids, host_indices, operation = self.ongoing_prefetch[rid]
        if operation.host_indices is None:
            return

        completed_tokens, _ = self.cache_controller.terminate_prefetch(operation)
        if self.tp_world_size > 1:
            torch.distributed.barrier(group=self.tp_group)
        last_host_node.release_host()
        del self.ongoing_prefetch[rid]
        self.cache_controller.append_host_mem_release(host_indices[:completed_tokens])
        self.cache_controller.prefetch_tokens_occupied -= len(token_ids)

    # ------------------------------------------------------------------
    # Remote backup lifecycle hooks
    # ------------------------------------------------------------------

    def notify_remote_request_start(self, req) -> None:
        """Notify remote backup storage that a request is starting (creates lease).

        Called when a request is first enqueued so the server can protect its
        decode KV pages from premature eviction.
        """
        if not self.enable_storage:
            return
        if getattr(req, "remote_backup_lease_active", False):
            return  # Already registered (idempotent guard)
        req.remote_backup_lease_active = True
        dp_rank = getattr(self.cache_controller, "dp_rank", 0) or 0
        self.cache_controller.notify_request_start(
            req.rid, dp_rank, req.remote_backup_generation
        )

    def notify_remote_request_finish(self, req, reason: str = "normal") -> None:
        """Notify remote backup storage that a request is done (enqueues GC).

        ``reason`` must be one of: "normal", "abort", "failover_old",
        "failover_supersede".
        """
        if not self.enable_storage:
            return
        if not getattr(req, "remote_backup_lease_active", False):
            return  # Not registered; nothing to clean up
        req.remote_backup_lease_active = False
        sync_key = (req.rid, int(getattr(req, "remote_backup_generation", 0) or 0))
        self.request_namespace_synced_tokens_by_reqid_gen.pop(sync_key, None)
        dp_rank = getattr(self.cache_controller, "dp_rank", 0) or 0
        self.cache_controller.notify_request_finish(
            req.rid, dp_rank, req.remote_backup_generation, reason
        )
