from collections import defaultdict
from types import SimpleNamespace

from sglang_simulator.simulation.types import RequestStats
from sglang_simulator.simulation.sglang.scheduler import (
    _attribute_request_iteration_timing,
    _mark_completed_request_stats,
    _populate_request_identity,
)
from sglang_simulator.time_predictor.base import (
    ScheduleBatch as SimulationScheduleBatch,
    ScheduleRequest,
)


def test_populate_request_identity_uses_simulation_args_with_defaults():
    req_stats = RequestStats()

    _populate_request_identity(
        req_stats,
        {
            "job_id": 7,
            "worker_id": 2,
            "assigned_dp_rank": None,
            "backup_policy": "",
            "attempt": None,
        },
        rid="rid-1",
    )

    assert req_stats.job_id == "7"
    assert req_stats.worker_id == "2"
    assert req_stats.assigned_dp_rank == 0
    assert req_stats.backup_policy == "none"
    assert req_stats.attempt == 0


def test_populate_request_identity_falls_back_to_request_rid():
    req_stats = RequestStats()

    _populate_request_identity(req_stats, {}, rid="rid-2")

    assert req_stats.job_id == "rid-2"
    assert req_stats.worker_id == ""
    assert req_stats.assigned_dp_rank == 0
    assert req_stats.backup_policy == "none"
    assert req_stats.attempt == 0


def test_attribute_request_iteration_timing_splits_latency_by_active_request():
    request_stats = defaultdict(RequestStats)
    batch = SimpleNamespace(
        reqs=[SimpleNamespace(rid="r1"), SimpleNamespace(rid="r2")]
    )
    simulation_batch = SimulationScheduleBatch(
        reqs=[
            ScheduleRequest(extend_length=8, past_kv_length=0),
            ScheduleRequest(extend_length=1, past_kv_length=4),
        ]
    )

    iteration_stats = _attribute_request_iteration_timing(
        request_stats=request_stats,
        batch=batch,
        simulation_batch=simulation_batch,
        current_inference_dur=0.6,
        hicache_l2_load_dur=0.2,
        hicache_l2_backup_dur=0.4,
        global_clock_s=12.5,
    )

    assert request_stats["r1"].prefetch_time_s == 0.1
    assert request_stats["r1"].backup_time_s == 0.2
    assert request_stats["r1"].inference_time_s == 0.3
    assert request_stats["r1"].prefill_inference_time_s == 0.3
    assert request_stats["r1"].decode_inference_time_s == 0
    assert request_stats["r2"].prefetch_time_s == 0.1
    assert request_stats["r2"].backup_time_s == 0.2
    assert request_stats["r2"].inference_time_s == 0.3
    assert request_stats["r2"].prefill_inference_time_s == 0.3
    assert request_stats["r2"].decode_inference_time_s == 0
    assert iteration_stats["rids"] == ["r1", "r2"]
    assert iteration_stats["mode"] == "prefill"
    assert iteration_stats["global_clock_s"] == 12.5


def test_attribute_request_iteration_timing_records_decode_mode():
    request_stats = defaultdict(RequestStats)
    batch = SimpleNamespace(reqs=[SimpleNamespace(rid="r1")])
    simulation_batch = SimulationScheduleBatch(
        reqs=[ScheduleRequest(extend_length=1, past_kv_length=4)]
    )

    _attribute_request_iteration_timing(
        request_stats=request_stats,
        batch=batch,
        simulation_batch=simulation_batch,
        current_inference_dur=0.6,
        hicache_l2_load_dur=0.2,
        hicache_l2_backup_dur=0.4,
        global_clock_s=12.5,
    )

    assert request_stats["r1"].prefill_inference_time_s == 0
    assert request_stats["r1"].decode_inference_time_s == 0.6


def test_mark_completed_request_stats_updates_status_and_queue_time():
    request_stats = defaultdict(RequestStats)
    request_stats["done"].queue_start = 1.5
    request_stats["done"].queue_end = 3.0
    request_stats["running"].queue_start = 1.5
    request_stats["running"].queue_end = 3.0
    batch = SimpleNamespace(
        reqs=[
            SimpleNamespace(rid="done", finished=lambda: True),
            SimpleNamespace(rid="running", finished=lambda: False),
        ]
    )

    _mark_completed_request_stats(request_stats, batch)

    assert request_stats["done"].status == "completed"
    assert request_stats["done"].queue_time_s == 1.5
    assert request_stats["running"].status == "running"
    assert request_stats["running"].queue_time_s == 0


def test_parallel_worker_client_seeds_and_schedules_next_request():
    from sglang_simulator.simulation.sglang.scheduler import (
        C_SchedulerHook,
        _register_parallel_worker_request,
        _schedule_next_parallel_worker_request,
        _seed_parallel_worker_clients,
    )

    C_SchedulerHook.FUTURE_QUEUE.clear()
    C_SchedulerHook.WORKER_PENDING.clear()

    def make_req(worker_seq: int):
        return SimpleNamespace(
            sampling_params=SimpleNamespace(
                custom_params={
                    "simulation": {
                        "worker_id": "1",
                        "worker_seq": worker_seq,
                        "parallel_worker_client": True,
                        "created_time": 0,
                    }
                }
            )
        )

    req0 = make_req(0)
    req1 = make_req(1)
    _register_parallel_worker_request(
        req0.sampling_params.custom_params["simulation"], req0
    )
    _register_parallel_worker_request(
        req1.sampling_params.custom_params["simulation"], req1
    )
    _seed_parallel_worker_clients()

    assert len(C_SchedulerHook.FUTURE_QUEUE) == 1
    assert C_SchedulerHook.FUTURE_QUEUE[0][2] is req0

    _schedule_next_parallel_worker_request(req0, 12.5)

    assert len(C_SchedulerHook.FUTURE_QUEUE) == 2
    assert C_SchedulerHook.FUTURE_QUEUE[1][0] == 12.5
    assert C_SchedulerHook.FUTURE_QUEUE[1][2] is req1
    assert (
        req1.sampling_params.custom_params["simulation"]["created_time"] == 12.5
    )

    C_SchedulerHook.FUTURE_QUEUE.clear()
    C_SchedulerHook.WORKER_PENDING.clear()
