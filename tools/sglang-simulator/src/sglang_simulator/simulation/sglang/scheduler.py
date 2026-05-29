import heapq
import importlib
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict

from sglang_simulator.hook import BaseHook
from sglang_simulator.hook.utils import get_obj_from_args
from sglang_simulator.simulation.manager import ConfigManager, Envs, FailureManager, StateManager
from sglang_simulator.simulation.sglang.utils import (
    resolve_model_info,
    resolve_scheduler_config,
)
from sglang_simulator.simulation.types import (
    RequestStats,
    SimulationMode,
)
from sglang_simulator.simulation.utils import (
    calc_metrics,
)
from sglang_simulator.time_predictor.base import InferTimePredictor
from sglang_simulator.time_predictor.base import (
    ScheduleBatch as SimulationScheduleBatch,
)
from sglang_simulator.time_predictor.base import ScheduleRequest
from sglang_simulator.utils import get_logger
from sglang_simulator.utils.json import CustomJsonEncoder

logger = get_logger("sgl_simulator")


def _reset_failure_manager_state() -> None:
    failure_config = ConfigManager.get_failure_config()
    if failure_config.enabled:
        C_SchedulerHook.FAILURE_MANAGER = FailureManager(failure_config)
        C_SchedulerHook.FAILURE_MANAGER.maybe_schedule_timeline()
    else:
        C_SchedulerHook.FAILURE_MANAGER = None


def _ensure_failure_manager() -> None:
    if C_SchedulerHook.FAILURE_MANAGER is None:
        _reset_failure_manager_state()


def _populate_request_identity(
    req_stats: RequestStats, simulation_args: dict, rid: str
) -> None:
    req_stats.job_id = str(simulation_args.get("job_id", rid))
    req_stats.worker_id = str(simulation_args.get("worker_id", ""))
    req_stats.assigned_dp_rank = int(
        simulation_args.get("assigned_dp_rank", 0) or 0
    )
    req_stats.backup_policy = str(
        simulation_args.get("backup_policy", "none") or "none"
    )
    req_stats.attempt = int(simulation_args.get("attempt", 0) or 0)


def _bool_from_simulation_arg(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return bool(value)


def _apply_failover_simulation_stats(
    req_stats: RequestStats, simulation_args: dict
) -> None:
    if not _bool_from_simulation_arg(simulation_args.get("is_failover_retried")):
        return

    req_stats.failure_impacted = True
    req_stats.is_failover_retried = True
    req_stats.failure_time_s = float(
        simulation_args.get("failure_time_s", req_stats.failure_time_s)
        or req_stats.failure_time_s
    )
    req_stats.pre_failover_output_tokens = int(
        simulation_args.get("pre_failover_output_tokens", 0) or 0
    )
    req_stats.pre_failover_backed_up_tokens = int(
        simulation_args.get("pre_failover_backed_up_tokens", 0) or 0
    )
    req_stats.retry_prefill_tokens = int(
        simulation_args.get("retry_prefill_tokens", 0) or 0
    )
    penalty = float(simulation_args.get("failover_penalty_s", 0.0) or 0.0)
    req_stats.failover_penalty_s = penalty
    req_stats.prefetch_time_s += penalty


def _computed_failover_tokens(req) -> int:
    return len(_req_origin_input_ids(req)) + len(_req_output_ids(req))


def _req_origin_input_ids(req) -> list:
    origin_input_ids = getattr(req, "origin_input_ids", None)
    if origin_input_ids is None:
        origin_input_ids = getattr(req, "input_ids", None)
    return list(origin_input_ids or [])


def _req_output_ids(req) -> list:
    return list(getattr(req, "output_ids", []) or [])


def _req_finished(req) -> bool:
    finished = getattr(req, "finished", None)
    if callable(finished):
        return bool(finished())
    return False


def _req_assigned_dp_rank(req) -> int | None:
    simulation_args = _extract_simulation_args(req)
    if simulation_args is None:
        return None
    assigned_dp_rank = simulation_args.get("assigned_dp_rank")
    if assigned_dp_rank is None:
        return None
    try:
        return int(assigned_dp_rank)
    except (TypeError, ValueError):
        return None


def _collect_and_remove_failed_pending_worker_reqs(failed_dp_rank: int) -> list:
    pending_reqs = []
    seen_rids = set()

    def add_req(req) -> None:
        rid = getattr(req, "rid", None) or f"obj:{id(req)}"
        if rid in seen_rids:
            return
        seen_rids.add(rid)
        pending_reqs.append(req)

    remaining_future_queue = []
    for item in C_SchedulerHook.FUTURE_QUEUE:
        _, _, req = item
        if _req_assigned_dp_rank(req) == failed_dp_rank:
            add_req(req)
        else:
            remaining_future_queue.append(item)
    C_SchedulerHook.FUTURE_QUEUE = remaining_future_queue
    heapq.heapify(C_SchedulerHook.FUTURE_QUEUE)

    for worker_id, entries in list(C_SchedulerHook.WORKER_PENDING.items()):
        remaining_entries = []
        for seq, req in entries:
            if _req_assigned_dp_rank(req) == failed_dp_rank:
                add_req(req)
            else:
                remaining_entries.append((seq, req))
        C_SchedulerHook.WORKER_PENDING[worker_id] = remaining_entries

    return pending_reqs


def _has_offline_pending_failover_state() -> bool:
    return bool(C_SchedulerHook.FUTURE_QUEUE or C_SchedulerHook.WORKER_PENDING)


def _ensure_request_stats_initialized_from_simulation_args(
    req_stats: RequestStats, req, simulation_args: dict | None
) -> None:
    if simulation_args is None:
        return
    if not getattr(req_stats, "rid", ""):
        req_stats.rid = req.rid
        req_stats.input_length = len(getattr(req, "input_ids", []) or [])
        sampling_params = getattr(req, "sampling_params", None)
        req_stats.output_length = int(getattr(sampling_params, "max_new_tokens", 0) or 0)
        _populate_request_identity(req_stats, simulation_args, req.rid)
        req_stats.created_time = float(simulation_args.get("created_time", 0.0) or 0.0)
        req_stats.last_event_time = req_stats.created_time
        req_stats.queue_start = req_stats.created_time


def _attribute_request_iteration_timing(
    request_stats: dict[str, RequestStats],
    batch,
    simulation_batch: SimulationScheduleBatch,
    current_inference_dur: float,
    hicache_l2_load_dur: float,
    hicache_l2_backup_dur: float,
    global_clock_s: float,
) -> dict:
    active_rids = [req.rid for req in batch.reqs]
    active_count = max(len(active_rids), 1)
    per_req_forward = current_inference_dur / active_count
    per_req_load = hicache_l2_load_dur / active_count
    per_req_backup = hicache_l2_backup_dur / active_count
    is_prefill = simulation_batch.is_prefill()

    for req in batch.reqs:
        req_stats = request_stats[req.rid]
        req_stats.prefetch_time_s += per_req_load
        req_stats.backup_time_s += per_req_backup
        req_stats.inference_time_s += per_req_forward
        if is_prefill:
            req_stats.prefill_inference_time_s += per_req_forward
        else:
            req_stats.decode_inference_time_s += per_req_forward

    return {
        "requests": simulation_batch.request_info(),
        "forward_latency": current_inference_dur,
        "l2_load_latency": hicache_l2_load_dur,
        "l2_backup_latency": hicache_l2_backup_dur,
        "rids": active_rids,
        "mode": "prefill" if is_prefill else "decode",
        "global_clock_s": global_clock_s,
    }


def _mark_completed_request_stats(
    request_stats: dict[str, RequestStats], batch
) -> None:
    for req in batch.reqs:
        if req.finished():
            req_stats = request_stats[req.rid]
            req_stats.status = "completed"
            req_stats.queue_time_s = max(req_stats.queue_end - req_stats.queue_start, 0)


def _load_failover_batch_cls():
    return getattr(
        importlib.import_module("sglang.srt.managers.io_struct"),
        "FailoverBatchReqInput",
    )


def _load_req_snapshot_cls():
    return getattr(
        importlib.import_module("sglang.srt.managers.io_struct"),
        "ReqSnapshot",
    )


def _build_failover_snapshot_and_mark_stats(
    *,
    req,
    request_stats: dict[str, RequestStats],
    failure_manager: FailureManager,
    scheduler_config,
    event,
    retry_created_time: float,
    req_snapshot_cls,
    kv_bytes_per_token: int = 0,
    memory_read_bandwidth: float | None = None,
    disk_read_bandwidth: float | None = None,
):
    generated_tokens = _req_output_ids(req)
    remaining_output = req.sampling_params.max_new_tokens - len(generated_tokens)
    if remaining_output <= 0:
        return None

    req_stats = request_stats[req.rid]
    original_input_ids = _req_origin_input_ids(req)
    failover_tokens = len(original_input_ids) + len(generated_tokens)
    backed_up_tokens = min(
        failure_manager.estimate_backed_up_tokens(
            event.backup_policy,
            failover_tokens,
        ),
        failover_tokens,
    )
    recovery_dp_rank = failure_manager.pick_recovery_dp_rank(
        event.failed_dp_rank,
        getattr(scheduler_config, "dp_size", 1),
    )

    penalty = failure_manager.restore_penalty_s(
        backup_policy=event.backup_policy,
        kv_bytes_per_token=kv_bytes_per_token,
        restored_tokens=backed_up_tokens,
        memory_read_bandwidth=memory_read_bandwidth,
        disk_read_bandwidth=disk_read_bandwidth,
    )

    req_stats.failure_impacted = True
    req_stats.failure_time_s = event.fire_time_s
    req_stats.pre_failover_output_tokens = len(generated_tokens)
    req_stats.pre_failover_backed_up_tokens = backed_up_tokens
    req_stats.retry_prefill_tokens = failover_tokens - backed_up_tokens
    req_stats.failover_penalty_s = penalty
    req_stats.last_event_time = retry_created_time
    req_stats.status = "failed_over"

    sampling_params = getattr(req, "sampling_params", None)
    custom_params = getattr(sampling_params, "custom_params", None)
    if isinstance(custom_params, dict):
        simulation_args = custom_params.get("simulation")
        if isinstance(simulation_args, dict):
            simulation_args["assigned_dp_rank"] = recovery_dp_rank
            simulation_args["backup_policy"] = event.backup_policy.value
            simulation_args["attempt"] = req_stats.attempt + 1
            simulation_args["created_time"] = retry_created_time
            simulation_args["failure_impacted"] = True
            simulation_args["is_failover_retried"] = True
            simulation_args["failure_time_s"] = event.fire_time_s
            simulation_args["pre_failover_output_tokens"] = len(generated_tokens)
            simulation_args["pre_failover_backed_up_tokens"] = backed_up_tokens
            simulation_args["retry_prefill_tokens"] = failover_tokens - backed_up_tokens
            simulation_args["failover_penalty_s"] = penalty

    return req_snapshot_cls(
        rid=req.rid,
        output_ids=generated_tokens,
        backed_up_tokens=backed_up_tokens,
        origin_input_ids=original_input_ids,
        sampling_params=req.sampling_params,
        original_max_new_tokens=getattr(req.sampling_params, "max_new_tokens", None),
        stream=getattr(req, "stream", False),
    )


def _restore_timing_kwargs_from_config() -> dict:
    platform = ConfigManager.get_platform_config()
    return {
        "kv_bytes_per_token": ConfigManager.get_kv_cache_bytes(),
        "memory_read_bandwidth": platform.memory_read_bandwidth,
        "disk_read_bandwidth": platform.disk_read_bandwidth,
    }


def _reroute_failed_dp_request(
    *,
    req,
    req_stats: RequestStats,
    failed_dp_ranks: set[int],
    failure_manager: FailureManager | None,
    scheduler_config,
) -> bool:
    if failure_manager is None:
        return False
    if int(req_stats.assigned_dp_rank) not in failed_dp_ranks:
        return False

    recovery_dp_rank = failure_manager.pick_recovery_dp_rank(
        failed_dp_rank=int(req_stats.assigned_dp_rank),
        dp_size=getattr(scheduler_config, "dp_size", 1),
    )
    req_stats.assigned_dp_rank = recovery_dp_rank
    req_stats.failure_impacted = True

    sampling_params = getattr(req, "sampling_params", None)
    custom_params = getattr(sampling_params, "custom_params", None)
    if isinstance(custom_params, dict):
        simulation_args = custom_params.get("simulation")
        if isinstance(simulation_args, dict):
            simulation_args["assigned_dp_rank"] = recovery_dp_rank
            simulation_args["failure_impacted"] = True
    return True


def _maybe_recover_failure(
    *,
    failure_manager: FailureManager | None,
    now_s: float,
    failed_dp_ranks: set[int],
    failure_events: list[dict],
) -> bool:
    if failure_manager is None or not failure_manager.should_recover(now_s):
        return False

    recovered_dp_rank = failure_manager.config.failed_dp_rank
    failed_dp_ranks.discard(recovered_dp_rank)
    failure_events.append(
        {
            "event": "sim_gpu_recovery",
            "timestamp_s": now_s,
            "recovered_dp_rank": recovered_dp_rank,
        }
    )
    return True


def _record_failure_event(
    event,
    failure_events: list[dict],
    failed_dp_ranks: set[int],
) -> None:
    failed_dp_ranks.add(event.failed_dp_rank)
    failure_events.append(
        {
            "event": "sim_gpu_failure",
            "event_id": event.event_id,
            "timestamp_s": event.fire_time_s,
            "failed_dp_rank": event.failed_dp_rank,
            "backup_policy": event.backup_policy.value,
        }
    )


def _maybe_schedule_offline_failure_on_request_start(
    *,
    failure_manager: FailureManager | None,
    simulation_args: dict,
    request_start_s: float,
    now_s: float,
    failed_dp_ranks: set[int],
    failure_events: list[dict],
) -> bool:
    if failure_manager is None:
        return False

    failure_manager.maybe_schedule_request_start(
        simulation_args,
        start_time_s=request_start_s,
    )
    event = failure_manager.should_fire(now_s)
    if event is not None:
        _record_failure_event(event, failure_events, failed_dp_ranks)
        return True

    failure_manager.maybe_schedule_recovery_request_start(
        simulation_args,
        start_time_s=request_start_s,
    )
    _maybe_recover_failure(
        failure_manager=failure_manager,
        now_s=now_s,
        failed_dp_ranks=failed_dp_ranks,
        failure_events=failure_events,
    )
    return False


def _maybe_fire_failure(
    scheduler,
    failure_manager: FailureManager | None,
    request_stats: dict[str, RequestStats],
    scheduler_config,
    now_s: float,
    failure_events: list[dict],
    failed_dp_ranks: set[int],
    restore_timing_kwargs_factory=None,
    failover_batch_cls=None,
    req_snapshot_cls=None,
) -> bool:
    if failure_manager is None:
        return False

    event = failure_manager.should_fire(now_s)
    if event is None:
        return False

    _record_failure_event(event, failure_events, failed_dp_ranks)

    if hasattr(scheduler, "_collect_failover_reqs"):
        candidate_reqs = list(scheduler._collect_failover_reqs())
    else:
        candidate_reqs = list(
            getattr(getattr(scheduler, "running_batch", None), "reqs", [])
        )
    if (
        C_SchedulerHook.SIM_MODE == SimulationMode.OFFLINE
        or _has_offline_pending_failover_state()
    ):
        candidate_reqs.extend(
            _collect_and_remove_failed_pending_worker_reqs(event.failed_dp_rank)
        )
    active_reqs = []
    seen_rids = set()
    for req in candidate_reqs:
        rid = getattr(req, "rid", None)
        if rid is None or rid in seen_rids:
            continue
        seen_rids.add(rid)
        active_reqs.append(req)
    restore_timing_kwargs = None
    snapshots = []
    first_impacted_req = None
    for req in active_reqs:
        req_stats = request_stats[req.rid]
        _ensure_request_stats_initialized_from_simulation_args(
            req_stats,
            req,
            _extract_simulation_args(req),
        )
        if req_stats.status == "completed":
            continue
        if req_stats.assigned_dp_rank != event.failed_dp_rank:
            continue
        if _req_finished(req):
            continue
        if req.sampling_params.max_new_tokens - len(_req_output_ids(req)) <= 0:
            continue
        if restore_timing_kwargs is None:
            if restore_timing_kwargs_factory is None:
                restore_timing_kwargs = {}
            else:
                restore_timing_kwargs = restore_timing_kwargs_factory()
        if req_snapshot_cls is None:
            req_snapshot_cls = _load_req_snapshot_cls()
        snapshot = _build_failover_snapshot_and_mark_stats(
            req=req,
            request_stats=request_stats,
            failure_manager=failure_manager,
            scheduler_config=scheduler_config,
            event=event,
            retry_created_time=now_s,
            req_snapshot_cls=req_snapshot_cls,
            **restore_timing_kwargs,
        )
        if snapshot is not None:
            snapshots.append(snapshot)
            if first_impacted_req is None:
                first_impacted_req = req

    if snapshots:
        if failover_batch_cls is None:
            failover_batch_cls = _load_failover_batch_cls()
        if hasattr(scheduler, "_clear_failed_rank_scheduler_state"):
            scheduler._clear_failed_rank_scheduler_state()
        failover = failover_batch_cls(
            failed_dp_rank=event.failed_dp_rank,
            snapshots=snapshots,
        )
        scheduler.send_to_tokenizer.send_output(failover, first_impacted_req)

    return True


def _is_parallel_worker_client(sim_params: dict | None) -> bool:
    return bool((sim_params or {}).get("parallel_worker_client"))


def _extract_simulation_args(req) -> dict | None:
    sampling_params = getattr(req, "sampling_params", None)
    custom_params = getattr(sampling_params, "custom_params", None)
    if not isinstance(custom_params, dict):
        return None
    simulation_args = custom_params.get("simulation")
    if not isinstance(simulation_args, dict):
        return None
    return simulation_args


class _BatchView:
    def __init__(self, reqs):
        self.reqs = reqs


def _register_parallel_worker_request(sim_params: dict, req) -> None:
    worker_id = str(sim_params.get("worker_id", "1"))
    worker_seq = int(sim_params.get("worker_seq", 0))
    C_SchedulerHook.WORKER_PENDING.setdefault(worker_id, []).append((worker_seq, req))


def _parallel_worker_received_count() -> int:
    return sum(len(queue) for queue in C_SchedulerHook.WORKER_PENDING.values())


def _seed_parallel_worker_clients() -> None:
    for worker_id in sorted(C_SchedulerHook.WORKER_PENDING.keys()):
        pending = sorted(C_SchedulerHook.WORKER_PENDING[worker_id], key=lambda x: x[0])
        C_SchedulerHook.WORKER_PENDING[worker_id] = pending
        if pending:
            _, first_req = pending[0]
            heapq.heappush(
                C_SchedulerHook.FUTURE_QUEUE,
                (0.0, time.time_ns(), first_req),
            )


def _schedule_next_parallel_worker_request(req, finish_time: float) -> None:
    custom_params = getattr(req.sampling_params, "custom_params", None) or {}
    sim_params = custom_params.get("simulation")
    if not _is_parallel_worker_client(sim_params):
        return
    worker_id = str(sim_params.get("worker_id", "1"))
    worker_seq = int(sim_params.get("worker_seq", 0))
    pending = [
        (seq, pending_req)
        for seq, pending_req in C_SchedulerHook.WORKER_PENDING.get(worker_id, [])
        if not (
            seq <= worker_seq
            and getattr(pending_req, "rid", None) == getattr(req, "rid", None)
        )
    ]
    C_SchedulerHook.WORKER_PENDING[worker_id] = pending
    for seq, next_req in pending:
        if seq != worker_seq + 1:
            continue
        next_custom_params = getattr(next_req.sampling_params, "custom_params", None)
        if isinstance(next_custom_params, dict):
            next_sim = next_custom_params.get("simulation")
            if isinstance(next_sim, dict):
                next_sim["created_time"] = finish_time
        heapq.heappush(
            C_SchedulerHook.FUTURE_QUEUE,
            (finish_time, time.time_ns(), next_req),
        )
        break


def _simulator_metadata() -> dict:
    return {
        "predictor": getattr(C_SchedulerHook.INFERENCE_PREDICTOR, "name", "unknown"),
        "simulation_mode": C_SchedulerHook.SIM_MODE.value,
        "failure_enabled": C_SchedulerHook.FAILURE_MANAGER is not None,
    }


def _write_simulator_metadata(output_dir: str) -> None:
    metadata_path = os.path.join(output_dir, "simulator_metadata.json")
    with open(metadata_path, "w") as f:
        f.write(json.dumps(_simulator_metadata()) + "\n")


class C_SchedulerHook(BaseHook):
    HOOK_CLASS_NAME = "Scheduler"
    HOOK_MODULE_NAME = "sglang.srt.managers.scheduler"

    INFERENCE_PREDICTOR: InferTimePredictor = None

    REQUEST_STATS: dict[str, RequestStats] = defaultdict(RequestStats)
    ITERATION_STATS: list[dict] = []
    LAST_CPU_TS: float = 0
    LAST_FLUSH_TS: float = 0
    SIMULATION_BATCH: SimulationScheduleBatch = None

    OVERLAP_SCHEDULE: bool = False

    SIM_MODE = SimulationMode(Envs.simulation_mode())
    OFFLINE_RECV_ALL_REQUEST: bool = False
    FUTURE_QUEUE: list[tuple[float, int, RequestStats]] = (
        []
    )  # tuple(created time, salt, request)
    WORKER_PENDING: dict[str, list[tuple[int, object]]] = {}
    FAILURE_MANAGER: FailureManager = None
    FAILED_DP_RANKS: set[int] = set()
    FAILURE_EVENTS: list[dict] = []
    IGNORED_REQUEST_RIDS: set[str] = set()

    @classmethod
    def hook(cls, target):
        original_init = target.__init__
        original_recv_requests = target.recv_requests
        original_get_new_batch_prefill = target.get_new_batch_prefill
        original_run_batch = target.run_batch
        original_process_batch_result = target.process_batch_result
        original_event_loop_normal = target.event_loop_normal
        original_init_tp_model_worker = getattr(target, "init_tp_model_worker", None)

        def override_event_loop_overlap(self, *args, **kwargs):
            # To reduce the complexity of the simulation, the overlapping schedule is not needed.
            return original_event_loop_normal(self, *args, **kwargs)

        def wrapped_init(self, *args, **kwargs):
            # Disable overlap schedule
            server_args = get_obj_from_args(
                "sglang.srt.server_args.ServerArgs", *args, **kwargs
            )
            C_SchedulerHook.OVERLAP_SCHEDULE = not getattr(
                server_args, "disable_overlap_schedule", False
            )
            setattr(server_args, "disable_overlap_schedule", True)
            logger.debug(
                f"Overlap schedule simulation mode: {C_SchedulerHook.OVERLAP_SCHEDULE}."
            )

            original_init(self, *args, **kwargs)

            try:
                if ConfigManager.get_model_info() is None:
                    model = resolve_model_info(self.model_config)
                    ConfigManager.set_model_info(model)

                model = ConfigManager.get_model_info()

                hw = ConfigManager.get_accelerator_info()

                if ConfigManager.get_scheduler_config() is None:
                    sched_config = resolve_scheduler_config(
                        server_args=self.server_args,
                    )
                    ConfigManager.set_scheduler_config(sched_config)
                sched_config = ConfigManager.get_scheduler_config()
                _reset_failure_manager_state()

                C_SchedulerHook.INFERENCE_PREDICTOR = (
                    ConfigManager.get_inference_time_predictor(model, hw, sched_config)
                )
            except Exception as e:
                logger.error(
                    f"Failed to initialize inference time predictor. Error: {e}"
                )
                raise e

        def wrapped_init_tp_model_worker(self, *args, **kwargs):
            if original_init_tp_model_worker is None:
                return None
            old_skip_tokenizer_init = getattr(
                self.server_args, "skip_tokenizer_init", False
            )
            self.server_args.skip_tokenizer_init = True
            try:
                return original_init_tp_model_worker(self, *args, **kwargs)
            finally:
                self.server_args.skip_tokenizer_init = old_skip_tokenizer_init

        def wrapped_recv_requests(self, *args, **kwargs) -> list:
            recv_reqs = []
            _ensure_failure_manager()

            if C_SchedulerHook.SIM_MODE == SimulationMode.BLOCKING:
                recv_reqs.extend(original_recv_requests(self, *args, **kwargs))
            elif C_SchedulerHook.SIM_MODE == SimulationMode.OFFLINE:
                # Initializing
                if not C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST:
                    gen_requests = []
                    extra_requests = []
                    time.sleep(0.05)  # waiting requests

                    reqs = original_recv_requests(self, *args, **kwargs)

                    for req in reqs:
                        if req.__class__.__name__ == "TokenizedGenerateReqInput":
                            gen_requests.append(req)
                        else:
                            # Such as: /profile_start, /flush_cache, etc.
                            extra_requests.append(req)

                    # Add requests to future queue
                    last_sim_params = None
                    for req in gen_requests:
                        sim_params = _extract_simulation_args(req)
                        if sim_params is None:
                            # There are some warm-up requests when starting the server without --skip-server-warmup.
                            C_SchedulerHook.IGNORED_REQUEST_RIDS.add(req.rid)
                            extra_requests.append(req)
                            logger.warning(
                                "Failed to extract the simulation parameters required for simulation from the request. Ignore this warning if the request is a warm-up request."
                            )
                            continue
                        if sim_params.get("queue_start"):
                            logger.debug(
                                "Add request to waiting queue with custom queue start timestamp."
                            )

                        if _is_parallel_worker_client(sim_params):
                            _register_parallel_worker_request(sim_params, req)
                        else:
                            C_SchedulerHook.FUTURE_QUEUE.append(
                                (
                                    sim_params.get("queue_start")
                                    or sim_params["created_time"],
                                    time.time_ns(),  # The request is not comparable, so add the salt to avoid comparison.
                                    req,
                            )
                        )
                        last_sim_params = sim_params

                    parallel_received = _parallel_worker_received_count()
                    received = (
                        parallel_received
                        if parallel_received > 0
                        else len(C_SchedulerHook.FUTURE_QUEUE)
                    )
                    if received != 0 and last_sim_params is not None:
                        total_request = last_sim_params["total_request"]

                        if received == total_request:
                            C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST = True
                            if parallel_received > 0:
                                _seed_parallel_worker_clients()
                            heapq.heapify(C_SchedulerHook.FUTURE_QUEUE)
                            logger.info(
                                "All requests received. Starting simulation now."
                            )
                        else:
                            logger.info(
                                f"Offline simulation mode enabled. {total_request} requests expected in total. Received {received} requests so far."
                            )

                    if len(extra_requests) != 0:
                        # Schedule the extra requests immediately.
                        return extra_requests
                else:
                    # Extra requests include: flush request, abort request, etc.
                    recv_reqs.extend(original_recv_requests(self, *args, **kwargs))

                # Process the arrived requests only after all requests have been added to the future queue
                current_timestamp = StateManager.get_global_clock()
                while (
                    C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST
                    and len(C_SchedulerHook.FUTURE_QUEUE) > 0
                ):
                    enqueue_time, _, req = C_SchedulerHook.FUTURE_QUEUE[0]
                    if enqueue_time > current_timestamp:
                        break
                    recv_reqs.append(req)
                    heapq.heappop(C_SchedulerHook.FUTURE_QUEUE)

            now = time.time()
            for req in recv_reqs:
                if req.__class__.__name__ in [
                    "BatchTokenizedGenerateReqInput",
                    "TokenizedGenerateReqInput",
                ]:
                    simulation_args = _extract_simulation_args(req)
                    if simulation_args is None:
                        C_SchedulerHook.IGNORED_REQUEST_RIDS.add(req.rid)
                        logger.warning(
                            "Failed to extract the simulation parameters required for "
                            "simulation from request %s. The request will run without "
                            "simulator statistics.",
                            req.rid,
                        )
                        continue
                    req_stats = C_SchedulerHook.REQUEST_STATS[req.rid]
                    req_stats.rid = req.rid
                    req_stats.input_length = len(req.input_ids)
                    req_stats.output_length = req.sampling_params.max_new_tokens
                    _populate_request_identity(req_stats, simulation_args, req.rid)
                    _apply_failover_simulation_stats(req_stats, simulation_args)
                    if C_SchedulerHook.SIM_MODE == SimulationMode.BLOCKING:
                        if "server_created_time" not in simulation_args:
                            logger.warning(
                                "The request's creation time is missing, which may cause the TTFT to be inaccurate."
                            )
                        req_stats.created_time = simulation_args.get(
                            "server_created_time", now
                        )
                        req_stats.last_event_time = req_stats.created_time
                        req_stats.queue_start = now
                    elif C_SchedulerHook.SIM_MODE == SimulationMode.OFFLINE:
                        req_stats.created_time = simulation_args["created_time"]
                        req_stats.last_event_time = req_stats.created_time
                        # Align with the real queue start timestamp if queue_start is not None. For debugging only.
                        queue_start = simulation_args.get("queue_start")
                        if queue_start is not None:
                            StateManager.set_global_clock(queue_start)
                        req_stats.queue_start = StateManager.get_global_clock()
                        if C_SchedulerHook.FAILURE_MANAGER is not None:
                            C_SchedulerHook.FAILURE_MANAGER.maybe_schedule_recovery_request_start(
                                simulation_args,
                                start_time_s=req_stats.queue_start,
                            )
                            _maybe_recover_failure(
                                failure_manager=C_SchedulerHook.FAILURE_MANAGER,
                                now_s=StateManager.get_global_clock(),
                                failed_dp_ranks=C_SchedulerHook.FAILED_DP_RANKS,
                                failure_events=C_SchedulerHook.FAILURE_EVENTS,
                            )
                    _reroute_failed_dp_request(
                        req=req,
                        req_stats=req_stats,
                        failed_dp_ranks=C_SchedulerHook.FAILED_DP_RANKS,
                        failure_manager=C_SchedulerHook.FAILURE_MANAGER,
                        scheduler_config=ConfigManager.get_scheduler_config(),
                    )

            if recv_reqs and C_SchedulerHook.LAST_CPU_TS == 0:
                C_SchedulerHook.LAST_CPU_TS = time.time()
                StateManager.set_global_clock(0)

            if C_SchedulerHook.FAILURE_MANAGER is not None:
                fired = _maybe_fire_failure(
                    scheduler=self,
                    failure_manager=C_SchedulerHook.FAILURE_MANAGER,
                    request_stats=C_SchedulerHook.REQUEST_STATS,
                    scheduler_config=ConfigManager.get_scheduler_config(),
                    now_s=StateManager.get_global_clock(),
                    failure_events=C_SchedulerHook.FAILURE_EVENTS,
                    failed_dp_ranks=C_SchedulerHook.FAILED_DP_RANKS,
                    restore_timing_kwargs_factory=_restore_timing_kwargs_from_config,
                )
                _maybe_recover_failure(
                    failure_manager=C_SchedulerHook.FAILURE_MANAGER,
                    now_s=StateManager.get_global_clock(),
                    failed_dp_ranks=C_SchedulerHook.FAILED_DP_RANKS,
                    failure_events=C_SchedulerHook.FAILURE_EVENTS,
                )

            return recv_reqs

        def wrapped_get_new_batch_prefill(self, *args, **kwargs):
            new_batch = original_get_new_batch_prefill(self, *args, **kwargs)
            now = time.time()
            if new_batch is not None:
                for req in new_batch.reqs:
                    if req.rid in C_SchedulerHook.IGNORED_REQUEST_RIDS:
                        continue
                    req_stats = C_SchedulerHook.REQUEST_STATS[req.rid]
                    req_stats.final_reused_tokens = req.cached_tokens
                    if req_stats.queue_end == -1:
                        if C_SchedulerHook.SIM_MODE == SimulationMode.BLOCKING:
                            req_stats.queue_end = now
                        else:
                            req_stats.queue_end = StateManager.get_global_clock()
                            if C_SchedulerHook.FAILURE_MANAGER is not None:
                                simulation_args = _extract_simulation_args(req)
                                if simulation_args is not None:
                                    C_SchedulerHook.FAILURE_MANAGER.maybe_schedule_request_start(
                                        simulation_args,
                                        start_time_s=req_stats.queue_end,
                                    )
                    else:
                        # Chunked request
                        pass
            elif len(self.running_batch.reqs) == 0 and len(self.waiting_queue) > 0:
                # Prefetching
                StateManager.step_global_clock(0.005)
                StateManager.set_current_inference_dur(0.005)
            else:
                if C_SchedulerHook.SIM_MODE == SimulationMode.OFFLINE and (
                    len(C_SchedulerHook.FUTURE_QUEUE) != 0
                    and len(self.running_batch.reqs) == 0
                ):
                    next_created_time, _, req = C_SchedulerHook.FUTURE_QUEUE[0]
                    StateManager.set_global_clock(next_created_time + 1e-6)
                    fired = _maybe_fire_failure(
                        scheduler=self,
                        failure_manager=C_SchedulerHook.FAILURE_MANAGER,
                        request_stats=C_SchedulerHook.REQUEST_STATS,
                        scheduler_config=ConfigManager.get_scheduler_config(),
                        now_s=StateManager.get_global_clock(),
                        failure_events=C_SchedulerHook.FAILURE_EVENTS,
                        failed_dp_ranks=C_SchedulerHook.FAILED_DP_RANKS,
                        restore_timing_kwargs_factory=_restore_timing_kwargs_from_config,
                    )
                    _maybe_recover_failure(
                        failure_manager=C_SchedulerHook.FAILURE_MANAGER,
                        now_s=StateManager.get_global_clock(),
                        failed_dp_ranks=C_SchedulerHook.FAILED_DP_RANKS,
                        failure_events=C_SchedulerHook.FAILURE_EVENTS,
                    )
            logger.debug(
                f"Get new batch prefill: global iteration={StateManager.get_iteration()}, "
                f"new batch={new_batch.batch_size() if new_batch is not None else 0}, "
                f"waiting queue={len(self.waiting_queue)}"
            )

            return new_batch

        def wrapped_run_batch(self, *args, **kwargs):
            ret = original_run_batch(self, *args, **kwargs)

            batch = get_obj_from_args(
                "sglang.srt.managers.schedule_batch.ScheduleBatch", *args, **kwargs
            )

            if ret.__class__.__name__ == "GenerationBatchResult":
                simulation_batch = SimulationScheduleBatch(reqs=[])
                if batch.forward_mode.is_extend():
                    for req in batch.reqs:
                        if req.rid in C_SchedulerHook.IGNORED_REQUEST_RIDS:
                            continue
                        simulation_batch.reqs.append(
                            ScheduleRequest(
                                extend_length=req.extend_input_len,
                                past_kv_length=len(req.prefix_indices)
                                + len(req.output_ids),
                            )
                        )
                elif batch.forward_mode.is_decode():
                    for req in batch.reqs:
                        if req.rid in C_SchedulerHook.IGNORED_REQUEST_RIDS:
                            continue
                        simulation_batch.reqs.append(
                            ScheduleRequest(
                                extend_length=1,
                                past_kv_length=len(req.prefix_indices)
                                + len(req.output_ids),
                            )
                        )

                if not simulation_batch.is_empty():
                    StateManager.inc_iteration()
                    predicted_latency = (
                        C_SchedulerHook.INFERENCE_PREDICTOR.predict_infer_time(
                            simulation_batch
                        )
                    )
                    predicted_latency = float(predicted_latency)

                    forward_latency = 0
                    if C_SchedulerHook.SIM_MODE == SimulationMode.BLOCKING:
                        time.sleep(abs(predicted_latency))
                        now = time.time()
                        forward_latency = now - C_SchedulerHook.LAST_CPU_TS
                        C_SchedulerHook.LAST_CPU_TS = now
                    else:
                        forward_latency = predicted_latency

                    StateManager.set_current_inference_dur(forward_latency)

                C_SchedulerHook.SIMULATION_BATCH = simulation_batch

            return ret

        def wrapped_process_batch_result(self, *args, **kwargs):
            ret = original_process_batch_result(self, *args, **kwargs)

            batch = get_obj_from_args(
                "sglang.srt.managers.schedule_batch.ScheduleBatch", *args, **kwargs
            )
            if batch is not None:
                tracked_reqs = [
                    req
                    for req in batch.reqs
                    if req.rid not in C_SchedulerHook.IGNORED_REQUEST_RIDS
                ]
                if len(tracked_reqs) == 0:
                    return ret
                tracked_batch = _BatchView(tracked_reqs)

                hicache_l2_load_dur = StateManager.pop_hicache_l2_load_dur()
                hicache_l2_backup_dur = StateManager.pop_hicache_l2_backup_dur()
                current_inference_dur = StateManager.get_current_inference_dur()

                if C_SchedulerHook.OVERLAP_SCHEDULE:
                    StateManager.step_global_clock(
                        max(
                            hicache_l2_load_dur - StateManager.get_last_inference_dur(),
                            0,
                        )
                    )
                    StateManager.step_global_clock(current_inference_dur)
                    request_response_time = (
                        StateManager.get_global_clock() + hicache_l2_backup_dur
                    )
                else:
                    StateManager.step_global_clock(
                        hicache_l2_load_dur
                        + current_inference_dur
                        + hicache_l2_backup_dur
                    )
                    request_response_time = StateManager.get_global_clock()
                iteration_stats = _attribute_request_iteration_timing(
                    request_stats=C_SchedulerHook.REQUEST_STATS,
                    batch=tracked_batch,
                    simulation_batch=C_SchedulerHook.SIMULATION_BATCH,
                    current_inference_dur=current_inference_dur,
                    hicache_l2_load_dur=hicache_l2_load_dur,
                    hicache_l2_backup_dur=hicache_l2_backup_dur,
                    global_clock_s=StateManager.get_global_clock(),
                )
                # Request statistics
                for req in tracked_reqs:
                    if req.is_chunked == 0:
                        req_stats = C_SchedulerHook.REQUEST_STATS[req.rid]
                        req_stats.gen_token_latencies.append(
                            request_response_time
                            - req_stats.last_event_time  # queue duration
                        )
                        req_stats.last_event_time = request_response_time
                    else:
                        # Chunked request: nothing to do
                        pass
                # Iteration statistics
                C_SchedulerHook.ITERATION_STATS.append(iteration_stats)
                _mark_completed_request_stats(
                    C_SchedulerHook.REQUEST_STATS,
                    tracked_batch,
                )
                for req in tracked_reqs:
                    if req.finished():
                        req_stats = C_SchedulerHook.REQUEST_STATS[req.rid]
                        if req_stats.status == "completed":
                            _schedule_next_parallel_worker_request(
                                req, request_response_time
                            )
                if C_SchedulerHook.FAILURE_MANAGER is not None:
                    if C_SchedulerHook.SIM_MODE == SimulationMode.OFFLINE:
                        fired = _maybe_fire_failure(
                            scheduler=self,
                            failure_manager=C_SchedulerHook.FAILURE_MANAGER,
                            request_stats=C_SchedulerHook.REQUEST_STATS,
                            scheduler_config=ConfigManager.get_scheduler_config(),
                            now_s=StateManager.get_global_clock(),
                            failure_events=C_SchedulerHook.FAILURE_EVENTS,
                            failed_dp_ranks=C_SchedulerHook.FAILED_DP_RANKS,
                            restore_timing_kwargs_factory=_restore_timing_kwargs_from_config,
                        )
                    for req in tracked_reqs:
                        if req.finished():
                            if C_SchedulerHook.SIM_MODE != SimulationMode.OFFLINE:
                                C_SchedulerHook.FAILURE_MANAGER.on_request_completed(
                                    req.rid,
                                    completion_time_s=StateManager.get_global_clock(),
                                )
                    if C_SchedulerHook.SIM_MODE != SimulationMode.OFFLINE:
                        fired = _maybe_fire_failure(
                            scheduler=self,
                            failure_manager=C_SchedulerHook.FAILURE_MANAGER,
                            request_stats=C_SchedulerHook.REQUEST_STATS,
                            scheduler_config=ConfigManager.get_scheduler_config(),
                            now_s=StateManager.get_global_clock(),
                            failure_events=C_SchedulerHook.FAILURE_EVENTS,
                            failed_dp_ranks=C_SchedulerHook.FAILED_DP_RANKS,
                            restore_timing_kwargs_factory=_restore_timing_kwargs_from_config,
                        )
                    if C_SchedulerHook.SIM_MODE == SimulationMode.OFFLINE:
                        for req in tracked_reqs:
                            simulation_args = _extract_simulation_args(req)
                            if simulation_args is None:
                                continue
                            C_SchedulerHook.FAILURE_MANAGER.maybe_schedule_recovery_request_start(
                                simulation_args,
                                start_time_s=request_response_time,
                            )
                    _maybe_recover_failure(
                        failure_manager=C_SchedulerHook.FAILURE_MANAGER,
                        now_s=StateManager.get_global_clock(),
                        failed_dp_ranks=C_SchedulerHook.FAILED_DP_RANKS,
                        failure_events=C_SchedulerHook.FAILURE_EVENTS,
                    )
            C_SchedulerHook.LAST_CPU_TS = time.time()
            return ret

        def wrapped_profile(self, req, *args, **kwargs):
            stats: list[RequestStats] = []
            for item in C_SchedulerHook.REQUEST_STATS.values():
                if item.rid is not None and item.input_length > 0:
                    stats.append(item)

            stats = sorted(stats, key=lambda req: req.created_time)

            dp_rank = getattr(self, "dp_rank", 0) or 0
            base_output_dir = Envs.output_dir()
            output_dir = os.path.join(base_output_dir, f"dp_{dp_rank}")
            os.makedirs(output_dir, exist_ok=True)

            try:
                with open(f"{output_dir}/failure_events.jsonl", "w") as f:
                    for item in C_SchedulerHook.FAILURE_EVENTS:
                        f.write(json.dumps(item) + "\n")
            except Exception as e:
                logger.error(f"Failed to dump failure events. Error: {e}")

            try:
                _write_simulator_metadata(base_output_dir)
            except Exception as e:
                logger.error(f"Failed to dump simulator metadata. Error: {e}")

            if len(stats) > 0:
                # Remove warmup requests.
                if len(stats) > Envs.num_warmup():
                    metrics_stats = stats[Envs.num_warmup() :]
                else:
                    metrics_stats = stats

                min_created_time = metrics_stats[0].created_time
                # Align timestamps
                for item in stats:
                    item.created_time -= min_created_time
                    item.queue_start -= min_created_time
                    item.queue_end -= min_created_time
                    item.last_event_time -= min_created_time

                metrics = calc_metrics(metrics_stats)
                metrics["time_cost"] = time.time() - C_SchedulerHook.LAST_FLUSH_TS

                try:
                    metrics_path = os.path.join(base_output_dir, "metrics.json")
                    with open(metrics_path, "w") as f:
                        f.write(json.dumps(metrics, cls=CustomJsonEncoder) + "\n")

                    with open(f"{output_dir}/iteration.jsonl", "w") as f:
                        for item in C_SchedulerHook.ITERATION_STATS:
                            f.write(json.dumps(item) + "\n")

                    with open(f"{output_dir}/request.jsonl", "w") as f:
                        for item in stats:
                            f.write(json.dumps(asdict(item)) + "\n")

                    logger.info(f"Simulation results saved to {output_dir}.")

                except Exception as e:
                    logger.error(f"Failed to dump results. Error: {e}")
            else:
                logger.warning("No request statistics available.")

            StateManager.reset()
            C_SchedulerHook.REQUEST_STATS.clear()
            C_SchedulerHook.ITERATION_STATS.clear()
            C_SchedulerHook.LAST_CPU_TS = 0
            C_SchedulerHook.LAST_FLUSH_TS = time.time()
            C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST = False
            C_SchedulerHook.FUTURE_QUEUE.clear()
            C_SchedulerHook.WORKER_PENDING.clear()
            C_SchedulerHook.FAILURE_EVENTS.clear()
            C_SchedulerHook.FAILED_DP_RANKS.clear()
            C_SchedulerHook.IGNORED_REQUEST_RIDS.clear()
            _reset_failure_manager_state()

            ProfileReqOutput = getattr(
                importlib.import_module("sglang.srt.managers.io_struct"),
                "ProfileReqOutput",
            )
            result = {
                "total_request": len(stats),
                "output_directory": output_dir,
            }

            return ProfileReqOutput(True, json.dumps(result))

        target.event_loop_overlap = override_event_loop_overlap
        target.__init__ = wrapped_init
        if original_init_tp_model_worker is not None:
            target.init_tp_model_worker = wrapped_init_tp_model_worker
        target.recv_requests = wrapped_recv_requests
        target.get_new_batch_prefill = wrapped_get_new_batch_prefill
        target.run_batch = wrapped_run_batch
        target.process_batch_result = wrapped_process_batch_result
        target.profile = wrapped_profile
