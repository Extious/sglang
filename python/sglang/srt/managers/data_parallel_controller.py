# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A controller that dispatches requests to multiple data parallel workers."""

import faulthandler
import logging
import multiprocessing as mp
import signal
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

import psutil
import setproctitle
import zmq

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.managers.io_struct import (
    AbortReq,
    ActiveRanksOutput,
    BlockReqInput,
    CheckpointUpdateReq,
    ResumeGenerateReq,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    VisibleStateUpdateReq,
    WatchLoadUpdateReq,
)
from sglang.srt.managers.schedule_batch import Req, RequestStage
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.metrics.cpu_monitor import start_cpu_monitor_thread
from sglang.srt.server_args import (
    DP_ATTENTION_HANDSHAKE_PORT_DELTA,
    PortArgs,
    ServerArgs,
)
from sglang.srt.tracing.trace import (
    process_tracing_init,
    trace_get_proc_propagate_context,
    trace_set_proc_propagate_context,
    trace_set_thread_info,
    trace_slice_end,
    trace_slice_start,
)
from sglang.srt.utils import numa_utils
from sglang.srt.utils.common import (
    bind_port,
    configure_ipv6,
    configure_logger,
    get_zmq_socket,
    kill_itself_when_parent_died,
    maybe_reindex_device_id,
)
from sglang.srt.utils.failover_event_logger import append_failover_event
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils.watchdog import Watchdog
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

logger = logging.getLogger(__name__)


class LoadBalanceMethod(Enum):
    """Load balance method."""

    ROUND_ROBIN = auto()
    FOLLOW_BOOTSTRAP_ROOM = auto()
    TOTAL_REQUESTS = auto()
    TOTAL_TOKENS = auto()

    @classmethod
    def from_str(cls, method: str):
        method = method.upper()
        try:
            return cls[method]
        except KeyError as exc:
            raise ValueError(f"Invalid load balance method: {method}") from exc


class DPBudget:
    def __init__(self, dp_size: int):
        self.dp_size = dp_size
        self.total_requests = [0] * dp_size
        self.total_tokens = [0] * dp_size

    def update_budget(self, load_update: WatchLoadUpdateReq):
        """Update the budget."""
        for load in load_update.loads:
            self.total_requests[load.dp_rank] = load.num_reqs
            self.total_tokens[load.dp_rank] = load.num_tokens

    def dispatch(self, method: LoadBalanceMethod):
        if method == LoadBalanceMethod.TOTAL_REQUESTS:
            target_rank = self.total_requests.index(min(self.total_requests))
        elif method == LoadBalanceMethod.TOTAL_TOKENS:
            # Use total_requests as a tie-breaker when total_tokens are equal
            target_rank = min(
                range(self.dp_size),
                key=lambda i: (self.total_tokens[i], self.total_requests[i]),
            )
        else:
            return None

        # Increment the load of that worker by one as a heuristic
        self.total_requests[target_rank] += 1
        return target_rank


@dataclass
class TrackedGenerateRequest:
    tokenized_req: TokenizedGenerateReqInput
    owner_dp_rank: int
    backup_dp_rank: Optional[int]
    visible_output_ids: List[int] = field(default_factory=list)
    checkpointed_output_len: int = 0
    failover_epoch: int = 0
    failover_supported: bool = True
    failover_completed: bool = False


class DataParallelController:
    """A controller that dispatches requests to multiple data parallel workers."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        run_scheduler_process_func: Callable,
    ) -> None:
        # Parse args
        self.server_args = server_args
        self.port_args = port_args
        self.load_balance_method = LoadBalanceMethod.from_str(
            server_args.load_balance_method
        )
        self.run_scheduler_process_func = run_scheduler_process_func

        # For DP balance
        self.global_balance_id = 0

        # Init inter-process communication
        self.context = zmq.Context(1 + server_args.dp_size)
        self.send_to_tokenizer = None
        if server_args.node_rank == 0:
            self.recv_from_tokenizer = get_zmq_socket(
                self.context, zmq.PULL, port_args.scheduler_input_ipc_name, False
            )
            self.send_to_tokenizer = get_zmq_socket(
                self.context, zmq.PUSH, port_args.tokenizer_ipc_name, False
            )

        # Dispatch method
        self.round_robin_counter = 0
        dispatch_lookup = {
            LoadBalanceMethod.ROUND_ROBIN: self.round_robin_scheduler,
            LoadBalanceMethod.FOLLOW_BOOTSTRAP_ROOM: self.follow_bootstrap_room_scheduler,
            LoadBalanceMethod.TOTAL_REQUESTS: self.total_requests_scheduler,
            LoadBalanceMethod.TOTAL_TOKENS: self.total_tokens_scheduler,
        }
        self.dispatching = dispatch_lookup[self.load_balance_method]

        # Load balance budget
        self.dp_budget = DPBudget(server_args.dp_size)
        self.enable_internal_failover = self._should_enable_internal_failover()
        self.tracked_generate_reqs: Dict[str, TrackedGenerateRequest] = {}

        # To protect changing env vars to set CUDA_VISIBLE_DEVICES.
        self.env_lock = threading.Lock()

        # Launch data parallel workers
        self.scheduler_procs = []
        self.dp_rank_scheduler_procs: List[List[mp.Process]] = [
            [] for _ in range(server_args.dp_size)
        ]
        self.workers: List[zmq.Socket] = [None] * server_args.dp_size
        self.status: List[bool] = [True] * server_args.dp_size

        if server_args.enable_dp_attention:
            self.launch_dp_attention_schedulers(server_args, port_args)
            self.control_message_step = server_args.tp_size
        else:
            self.launch_dp_schedulers(server_args, port_args)
            self.control_message_step = 1

        self.init_dispatcher()

        self.soft_watchdog = Watchdog.create(
            debug_name="DataParallelController",
            watchdog_timeout=server_args.soft_watchdog_timeout,
            soft=True,
            test_stuck_time=envs.SGLANG_TEST_STUCK_DP_CONTROLLER.get(),
        )

        if server_args.enable_metrics:
            start_cpu_monitor_thread("data_parallel_controller")

    def _should_enable_internal_failover(self) -> bool:
        storage_backend = self.server_args.hicache_storage_backend
        if self.server_args.dp_size <= 1:
            return False
        if storage_backend not in ("peer", "PeerCacheStorage"):
            return False
        if self.server_args.pp_async_batch_depth != 0:
            logger.warning(
                "Internal DP failover is disabled because pp_async_batch_depth=%d.",
                self.server_args.pp_async_batch_depth,
            )
            return False
        return True

    def _send_request_to_worker(self, req, dp_rank: int) -> int:
        self.workers[dp_rank].send_pyobj(req)
        return dp_rank

    def _choose_backup_dp_rank(self, owner_dp_rank: int) -> Optional[int]:
        if self.server_args.dp_size <= 1:
            return None
        candidate = (owner_dp_rank + 1) % self.server_args.dp_size
        return None if candidate == owner_dp_rank else candidate

    def _supports_failover(self, req: TokenizedGenerateReqInput) -> bool:
        return (
            self.enable_internal_failover
            and req.mm_inputs in (None, {})
            and req.input_embeds is None
            and req.session_params is None
            and req.custom_logit_processor is None
            and not req.return_logprob
            and not req.return_hidden_states
            and not req.return_routed_experts
            and not req.require_reasoning
        )

    def _register_generate_request(
        self, req: TokenizedGenerateReqInput, owner_dp_rank: int
    ) -> None:
        self.tracked_generate_reqs[req.rid] = TrackedGenerateRequest(
            tokenized_req=req,
            owner_dp_rank=owner_dp_rank,
            backup_dp_rank=self._choose_backup_dp_rank(owner_dp_rank),
            failover_supported=self._supports_failover(req),
        )

    def _abort_tracked_request(self, rid: str, message: str) -> None:
        self.tracked_generate_reqs.pop(rid, None)
        if self.send_to_tokenizer is not None:
            self.send_to_tokenizer.send_pyobj(
                AbortReq(rid=rid, abort_message=message)
            )

    def _emit_failover_event(
        self,
        event: str,
        tracked: Optional[TrackedGenerateRequest],
        *,
        rid: str,
        failed_owner_dp_rank: int,
        backup_dp_rank: Optional[int],
        checkpointed_output_len: int = 0,
        visible_output_len: int = 0,
        failover_epoch: int = 0,
        reason: Optional[str] = None,
    ) -> None:
        append_failover_event(
            event,
            rid=rid,
            agent_id=getattr(tracked.tokenized_req, "agent_id", None) if tracked else None,
            failed_owner_dp_rank=failed_owner_dp_rank,
            backup_dp_rank=backup_dp_rank,
            checkpointed_output_len=checkpointed_output_len,
            visible_output_len=visible_output_len,
            failover_epoch=failover_epoch,
            reason=reason,
        )

    def _handle_failed_dp_rank(self, failed_dp_rank: int) -> None:
        affected_rids = [
            rid
            for rid, tracked in self.tracked_generate_reqs.items()
            if tracked.owner_dp_rank == failed_dp_rank
        ]
        if not self.enable_internal_failover:
            for rid in affected_rids:
                tracked = self.tracked_generate_reqs.get(rid)
                self._emit_failover_event(
                    "failover_aborted",
                    tracked,
                    rid=rid,
                    failed_owner_dp_rank=failed_dp_rank,
                    backup_dp_rank=getattr(tracked, "backup_dp_rank", None),
                    reason="internal_failover_disabled",
                )
                self._abort_tracked_request(
                    rid,
                    "The owner worker failed and internal DP failover is disabled.",
                )
            return

        for rid in affected_rids:
            tracked = self.tracked_generate_reqs.get(rid)
            if tracked is None:
                continue
            if tracked.failover_completed:
                self._emit_failover_event(
                    "failover_aborted",
                    tracked,
                    rid=rid,
                    failed_owner_dp_rank=failed_dp_rank,
                    backup_dp_rank=tracked.backup_dp_rank,
                    checkpointed_output_len=tracked.checkpointed_output_len,
                    visible_output_len=len(tracked.visible_output_ids),
                    failover_epoch=tracked.failover_epoch,
                    reason="second_owner_failure",
                )
                self._abort_tracked_request(
                    rid,
                    "The resumed owner worker failed again and no further failover is supported.",
                )
                continue

            if not tracked.failover_supported:
                self._emit_failover_event(
                    "failover_aborted",
                    tracked,
                    rid=rid,
                    failed_owner_dp_rank=failed_dp_rank,
                    backup_dp_rank=tracked.backup_dp_rank,
                    checkpointed_output_len=tracked.checkpointed_output_len,
                    visible_output_len=len(tracked.visible_output_ids),
                    failover_epoch=tracked.failover_epoch,
                    reason="unsupported_request_features",
                )
                self._abort_tracked_request(
                    rid,
                    "The request uses features that are not supported by internal DP failover.",
                )
                continue

            backup_dp_rank = tracked.backup_dp_rank
            if backup_dp_rank is None or not self.status[backup_dp_rank]:
                self._emit_failover_event(
                    "failover_aborted",
                    tracked,
                    rid=rid,
                    failed_owner_dp_rank=failed_dp_rank,
                    backup_dp_rank=backup_dp_rank,
                    checkpointed_output_len=tracked.checkpointed_output_len,
                    visible_output_len=len(tracked.visible_output_ids),
                    failover_epoch=tracked.failover_epoch,
                    reason="backup_worker_unavailable",
                )
                self._abort_tracked_request(
                    rid,
                    "No live peer backup worker is available for internal DP failover.",
                )
                continue

            checkpointed_output_len = min(
                tracked.checkpointed_output_len, len(tracked.visible_output_ids)
            )
            failover_epoch = tracked.failover_epoch + 1
            resume_req = ResumeGenerateReq(
                rid=rid,
                tokenized_req=tracked.tokenized_req,
                visible_output_ids=list(tracked.visible_output_ids),
                checkpointed_output_len=checkpointed_output_len,
                owner_dp_rank=failed_dp_rank,
                backup_dp_rank=backup_dp_rank,
                failover_epoch=failover_epoch,
            )
            self.workers[backup_dp_rank].send_pyobj(resume_req)
            self._emit_failover_event(
                "failover_dispatched",
                tracked,
                rid=rid,
                failed_owner_dp_rank=failed_dp_rank,
                backup_dp_rank=backup_dp_rank,
                checkpointed_output_len=checkpointed_output_len,
                visible_output_len=len(tracked.visible_output_ids),
                failover_epoch=failover_epoch,
            )
            tracked.owner_dp_rank = backup_dp_rank
            tracked.failover_epoch = failover_epoch
            tracked.failover_completed = True
            logger.warning(
                "Failing over rid=%s from dp_rank=%d to dp_rank=%d at checkpoint=%d visible=%d",
                rid,
                failed_dp_rank,
                backup_dp_rank,
                checkpointed_output_len,
                len(tracked.visible_output_ids),
            )

    def _poll_failed_worker_groups(self) -> None:
        for dp_rank, procs in enumerate(self.dp_rank_scheduler_procs):
            if not self.status[dp_rank]:
                continue
            if any(proc.exitcode is not None for proc in procs):
                self.status[dp_rank] = False
                logger.error("Detected failure in dp_rank=%d replica group", dp_rank)
                self._handle_failed_dp_rank(dp_rank)

    def handle_generate_request(self, req: TokenizedGenerateReqInput):
        owner_dp_rank = self.dispatching_with_trace(req)
        if owner_dp_rank is not None:
            self._register_generate_request(req, owner_dp_rank)

    def send_to_all_workers(self, obj):
        for i, worker in enumerate(self.workers):
            if self.status[i]:
                worker.send_pyobj(obj)

    def send_control_message(self, obj):
        # Send control messages to first worker of tp group
        for i in range(0, len(self.workers), self.control_message_step):
            if self.status[i]:
                self.workers[i].send_pyobj(obj)

    def handle_load_update_req(self, obj):
        self.dp_budget.update_budget(obj)

    def handle_abort_req(self, obj: AbortReq):
        if obj.abort_all:
            self.tracked_generate_reqs.clear()
        else:
            self.tracked_generate_reqs.pop(obj.rid, None)
        self.send_control_message(obj)

    def handle_checkpoint_update_req(self, obj: CheckpointUpdateReq):
        tracked = self.tracked_generate_reqs.get(obj.rid)
        if tracked is None or tracked.failover_completed:
            return
        tracked.checkpointed_output_len = max(
            tracked.checkpointed_output_len,
            obj.global_checkpointed_output_len,
        )

    def handle_visible_state_update_req(self, obj: VisibleStateUpdateReq):
        tracked = self.tracked_generate_reqs.get(obj.rid)
        if tracked is None:
            return
        tracked.visible_output_ids.extend(obj.output_ids_delta)
        if obj.finished:
            self.tracked_generate_reqs.pop(obj.rid, None)

    def update_active_ranks(self, ranks: ActiveRanksOutput):
        self.status = [cur and new for cur, new in zip(self.status, ranks.status)]

    def dispatching_with_trace(self, req: Req):
        if self.server_args.enable_trace:
            trace_set_proc_propagate_context(req.rid, req.trace_context)
            trace_slice_start(RequestStage.DC_DISPATCH, req.rid)
            req.trace_context = trace_get_proc_propagate_context(req.rid)

        target_rank = self.dispatching(req)

        if self.server_args.enable_trace:
            trace_slice_end(RequestStage.DC_DISPATCH, req.rid, thread_finish_flag=True)
        return target_rank

    def init_dispatcher(self):
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.handle_generate_request),
                (TokenizedEmbeddingReqInput, self.dispatching_with_trace),
                (BlockReqInput, self.send_to_all_workers),
                (AbortReq, self.handle_abort_req),
                (CheckpointUpdateReq, self.handle_checkpoint_update_req),
                (VisibleStateUpdateReq, self.handle_visible_state_update_req),
                (WatchLoadUpdateReq, self.handle_load_update_req),
                (ActiveRanksOutput, self.update_active_ranks),
            ]
        )
        self._request_dispatcher.add_fallback_fn(self.send_control_message)

    def launch_dp_schedulers(self, server_args, port_args):
        base_gpu_id = 0
        local_gpu_stride = self._get_local_gpu_stride(server_args)
        use_tcp_control_plane = PortArgs.use_tcp_control_plane(server_args)

        threads = []
        reserved_nccl_sockets = []
        ready_events = []
        broadcasted_dp_port_args = None
        if use_tcp_control_plane:
            if server_args.node_rank == 0:
                worker_ports = []
                dp_port_args = []
                for dp_rank in range(server_args.dp_size):
                    worker_port, worker_socket = get_zmq_socket(self.context, zmq.PUSH)
                    worker_ports.append(worker_port)
                    self.workers[dp_rank] = worker_socket
                    tmp_port_args = PortArgs.init_new(
                        server_args, dp_rank=dp_rank, worker_ports=worker_ports
                    )
                    reserved_nccl_sockets.append(bind_port(tmp_port_args.nccl_port))
                    dp_port_args.append(tmp_port_args)
                broadcasted_dp_port_args = self._broadcast_worker_ports(
                    server_args, dp_port_args
                )
            else:
                broadcasted_dp_port_args = self._broadcast_worker_ports(server_args)

        for dp_rank in range(server_args.dp_size):
            if broadcasted_dp_port_args is not None:
                tmp_port_args = broadcasted_dp_port_args[dp_rank]
            else:
                tmp_port_args = PortArgs.init_new(server_args)
                tmp_port_args.tokenizer_ipc_name = port_args.tokenizer_ipc_name
                tmp_port_args.detokenizer_ipc_name = port_args.detokenizer_ipc_name

                # This port is checked free in PortArgs.init_new.
                # We hold it first so that the next dp worker gets a different port.
                reserved_nccl_sockets.append(bind_port(tmp_port_args.nccl_port))

            ready_event = threading.Event()
            ready_events.append(ready_event)

            # Create a thread for each worker
            thread = threading.Thread(
                target=self.launch_tensor_parallel_group_thread,
                args=(server_args, tmp_port_args, base_gpu_id, dp_rank, ready_event),
            )
            threads.append(thread)
            base_gpu_id += local_gpu_stride

            if server_args.node_rank == 0 and not use_tcp_control_plane:
                self.workers[dp_rank] = get_zmq_socket(
                    self.context,
                    zmq.PUSH,
                    tmp_port_args.scheduler_input_ipc_name,
                    True,
                )

        # Free all sockets before starting the threads to launch TP workers
        for sock in reserved_nccl_sockets:
            sock.close()

        # Start all threads
        for thread in threads:
            thread.start()
        for event in ready_events:
            event.wait()

    def launch_tensor_parallel_group_thread(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: int,
        ready_event: threading.Event,
    ):
        self.launch_tensor_parallel_group(server_args, port_args, base_gpu_id, dp_rank)
        ready_event.set()

        # This thread cannot be closed because otherwise the `kill_itself_when_parent_died`
        # function in scheduler.py will kill the scheduler.
        while True:
            time.sleep(30 * 24 * 3600)

    def _get_local_gpu_stride(self, server_args: ServerArgs) -> int:
        pp_size_per_node = max(server_args.pp_size // server_args.nnodes, 1)
        nnodes_per_pp_rank = max(server_args.nnodes // server_args.pp_size, 1)
        nnodes_per_tp_group = nnodes_per_pp_rank
        tp_size_per_node = server_args.tp_size // nnodes_per_tp_group
        return pp_size_per_node * tp_size_per_node * server_args.gpu_id_step

    def _broadcast_worker_ports(
        self, server_args: ServerArgs, payload: Optional[Any] = None
    ) -> Any:
        """Broadcast a control-plane payload from node 0 to all other nodes.

        Node 0 acts as the server, waiting for all other nodes to connect and
        sending them the payload. Other nodes act as clients, connecting to
        node 0 to receive their copy.

        Args:
            server_args: Server arguments containing node configuration.
            payload: Serializable Python object to broadcast.

        Returns:
            The broadcast payload.
        """
        # Determine the endpoint for inter-node communication
        if server_args.dist_init_addr is None:
            endpoint = f"tcp://127.0.0.1:{server_args.port + DP_ATTENTION_HANDSHAKE_PORT_DELTA}"
        elif server_args.dist_init_addr.startswith("["):  # ipv6 address
            port, host = configure_ipv6(server_args.dist_init_addr)
            endpoint = f"tcp://{host}:{int(port) + DP_ATTENTION_HANDSHAKE_PORT_DELTA}"
        else:
            host, port = server_args.dist_init_addr.split(":")
            endpoint = f"tcp://{host}:{int(port) + DP_ATTENTION_HANDSHAKE_PORT_DELTA}"

        if server_args.node_rank == 0:
            # Node 0: Broadcast worker ports to all other nodes
            return self._broadcast_ports_as_server(
                endpoint, server_args.nnodes - 1, payload
            )
        else:
            # Other nodes: Receive worker ports from node 0
            return self._receive_ports_as_client(endpoint, server_args.node_rank)

    def _broadcast_ports_as_server(
        self, endpoint: str, expected_clients: int, payload: Any
    ) -> Any:
        """Broadcast a control-plane payload to all client nodes."""
        logger.debug("Broadcasting control-plane payload to %d client nodes", expected_clients)

        rep_socket = get_zmq_socket(self.context, zmq.REP, endpoint, True)

        try:
            connected_clients = 0
            while connected_clients < expected_clients:
                # Wait for client handshake
                client_rank = rep_socket.recv().decode()
                logger.debug(f"Received handshake from node {client_rank}")

                # Send payload to client
                rep_socket.send_pyobj(payload)
                connected_clients += 1
                logger.debug(
                    "Sent control-plane payload to %d/%d nodes",
                    connected_clients,
                    expected_clients,
                )

            logger.debug("Control-plane payload broadcast completed")
            return payload
        finally:
            rep_socket.close()

    def _receive_ports_as_client(self, endpoint: str, node_rank: int) -> Any:
        """Receive a control-plane payload from the server node."""
        logger.debug("Connecting to node 0 to receive control-plane payload")

        req_socket = get_zmq_socket(self.context, zmq.REQ, endpoint, False)
        req_socket.setsockopt(zmq.RCVTIMEO, 600 * 1000)  # 10 minute timeout
        req_socket.setsockopt(zmq.SNDTIMEO, 600 * 1000)

        try:
            # Send handshake with our node rank
            req_socket.send(str(node_rank).encode())

            # Receive payload
            payload = req_socket.recv_pyobj()
            logger.debug("Received control-plane payload from node 0")
            return payload
        except zmq.Again:
            logger.error("Timeout waiting for control-plane payload from node 0")
            raise RuntimeError(
                "Failed to receive control-plane payload from node 0 within timeout"
            )
        finally:
            req_socket.close()

    def launch_dp_attention_schedulers(
        self, server_args: ServerArgs, port_args: PortArgs
    ):
        # Pre-allocate worker ports on node 0 to avoid conflicts
        worker_ports = []
        if server_args.node_rank == 0:
            for dp_rank in range(server_args.dp_size):
                port_and_socket = get_zmq_socket(self.context, zmq.PUSH)
                worker_ports.append(port_and_socket[0])
                self.workers[dp_rank] = port_and_socket[1]
                logger.debug(f"Assigned port {port_and_socket[0]} to worker {dp_rank}")

        broadcasted_ports = self._broadcast_worker_ports(
            server_args, worker_ports if worker_ports else None
        )
        self.launch_tensor_parallel_group(
            server_args, port_args, 0, None, broadcasted_ports
        )

    def launch_tensor_parallel_group(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: Optional[int],
        worker_ports: Optional[List[int]] = None,
    ):
        if not server_args.enable_dp_attention:
            logger.info(f"Launch DP{dp_rank} starting at GPU #{base_gpu_id}.")

        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )

        scheduler_pipe_readers = []

        pp_size_per_node = max(server_args.pp_size // server_args.nnodes, 1)
        nnodes_per_pp_rank = max(server_args.nnodes // server_args.pp_size, 1)
        pp_rank_range = range(
            pp_size_per_node * (server_args.node_rank // nnodes_per_pp_rank),
            pp_size_per_node * (server_args.node_rank // nnodes_per_pp_rank + 1),
        )

        nnodes_per_tp_group = nnodes_per_pp_rank
        tp_size_per_node = server_args.tp_size // nnodes_per_tp_group
        tp_rank_range = range(
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group),
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group + 1),
        )

        attn_cp_rank = 0
        moe_dp_rank = 0
        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                rank_port_args = port_args

                if server_args.enable_dp_attention:
                    # dp attention has different sharding logic
                    _, _, dp_rank = compute_dp_attention_world_info(
                        server_args.enable_dp_attention,
                        tp_rank,
                        server_args.tp_size,
                        server_args.dp_size,
                        server_args.attn_cp_size,
                    )
                    # compute zmq ports for this dp rank
                    rank_port_args = PortArgs.init_new(
                        server_args, dp_rank, worker_ports
                    )
                    # Data parallelism reuses the tensor parallelism group,
                    # so all dp ranks should use the same nccl port.
                    rank_port_args.nccl_port = port_args.nccl_port

                reader, writer = mp.Pipe(duplex=False)
                gpu_id = (
                    server_args.base_gpu_id
                    + base_gpu_id
                    + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                    + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
                )
                attn_dp_size = (
                    server_args.dp_size if server_args.enable_dp_attention else 1
                )

                # Parallelism hierarchy (outermost to innermost):
                # - Attention: Global(TP) -> DP -> ATTN_CP -> ATTN_TP (innermost)
                # - MoE: Global(TP) -> MOE_DP -> EP -> MOE_TP (innermost)
                attn_tp_size = (
                    server_args.tp_size // attn_dp_size // server_args.attn_cp_size
                )
                attn_cp_rank = (tp_rank // attn_tp_size) % server_args.attn_cp_size
                moe_dp_rank = tp_rank // (
                    server_args.tp_size // server_args.moe_dp_size
                )
                moe_ep_rank = (
                    tp_rank
                    % (server_args.tp_size // server_args.moe_dp_size)
                    // (
                        server_args.tp_size
                        // server_args.moe_dp_size
                        // server_args.ep_size
                    )
                )

                with self.env_lock, maybe_reindex_device_id(gpu_id) as gpu_id:
                    proc = mp.Process(
                        target=self.run_scheduler_process_func,
                        args=(
                            server_args,
                            rank_port_args,
                            gpu_id,
                            tp_rank,
                            attn_cp_rank,
                            moe_dp_rank,
                            moe_ep_rank,
                            pp_rank,
                            dp_rank,
                            writer,
                        ),
                    )
                    with memory_saver_adapter.configure_subprocess(), numa_utils.configure_subprocess(
                        server_args, gpu_id
                    ):
                        proc.start()
                self.scheduler_procs.append(proc)
                self.dp_rank_scheduler_procs[dp_rank].append(proc)
                scheduler_pipe_readers.append(reader)

        # Wait for model to finish loading
        scheduler_info = []
        for i in range(len(scheduler_pipe_readers)):
            scheduler_info.append(scheduler_pipe_readers[i].recv())

        self.max_total_num_tokens = scheduler_info[0]["max_total_num_tokens"]
        self.max_req_input_len = scheduler_info[0]["max_req_input_len"]

    def maybe_external_dp_rank_routing(self, req: Req):
        if req.data_parallel_rank is not None:
            logger.debug(f"Direct routing to DP rank {req.data_parallel_rank}")
            return self._send_request_to_worker(req, req.data_parallel_rank)
        return None

    def round_robin_scheduler(self, req: Req):
        target_rank = self.maybe_external_dp_rank_routing(req)
        if target_rank is not None:
            return target_rank

        while True:
            if self.status[self.round_robin_counter]:
                logger.debug(f"Choose worker {self.round_robin_counter}")
                target_rank = self._send_request_to_worker(req, self.round_robin_counter)
                self.round_robin_counter = (self.round_robin_counter + 1) % len(
                    self.workers
                )
                return target_rank
            self.round_robin_counter = (self.round_robin_counter + 1) % len(
                self.workers
            )

    def follow_bootstrap_room_scheduler(self, req: Req):
        target_rank = self.maybe_external_dp_rank_routing(req)
        if target_rank is not None:
            return target_rank

        # Set default bootstrap_room if in FAKE auto mode and room is None
        if (
            req.bootstrap_room is None
            and self.server_args.disaggregation_transfer_backend == "fake"
        ):
            req.bootstrap_room = self.round_robin_counter
            self.round_robin_counter = (self.round_robin_counter + 1) % len(
                self.workers
            )

        assert req.bootstrap_room is not None, (
            "req.bootstrap_room should not be None. Do not send requests directly to "
            "prefill or decode instances; send to the router instead."
        )
        target_rank = req.bootstrap_room % len(self.workers)
        return self._send_request_to_worker(req, target_rank)

    def total_requests_scheduler(self, req: Req):
        target_rank = self.maybe_external_dp_rank_routing(req)
        if target_rank is not None:
            return target_rank
        target_worker = self.dp_budget.dispatch(LoadBalanceMethod.TOTAL_REQUESTS)
        return self._send_request_to_worker(req, target_worker)

    def total_tokens_scheduler(self, req: Req):
        target_rank = self.maybe_external_dp_rank_routing(req)
        if target_rank is not None:
            return target_rank
        target_worker = self.dp_budget.dispatch(LoadBalanceMethod.TOTAL_TOKENS)
        return self._send_request_to_worker(req, target_worker)

    def event_loop(self):
        while True:
            self._poll_failed_worker_groups()
            while True:
                self.soft_watchdog.feed()
                try:
                    recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                self._request_dispatcher(recv_req)


def run_data_parallel_controller_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
    run_scheduler_process_func: Callable = run_scheduler_process,
):
    setproctitle.setproctitle("sglang::data_parallel_controller")
    faulthandler.enable()
    kill_itself_when_parent_died()
    parent_process = psutil.Process().parent()

    configure_logger(server_args)
    if server_args.enable_trace:
        process_tracing_init(server_args.otlp_traces_endpoint, "sglang")
        thread_label = "DP Controller"
        if server_args.disaggregation_mode == "prefill":
            thread_label = "Prefill DP Controller"
        elif server_args.disaggregation_mode == "decode":
            thread_label = "Decode DP Controller"
        trace_set_thread_info(thread_label)

    try:
        controller = DataParallelController(
            server_args, port_args, run_scheduler_process_func
        )
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": controller.max_total_num_tokens,
                "max_req_input_len": controller.max_req_input_len,
            }
        )
        if server_args.node_rank == 0:
            controller.event_loop()
        for proc in controller.scheduler_procs:
            proc.join()
            logger.error(
                f"Scheduler or DataParallelController {proc.pid} terminated with {proc.exitcode}"
            )
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"DataParallelController hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
