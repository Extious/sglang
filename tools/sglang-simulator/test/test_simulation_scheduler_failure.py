from collections import defaultdict
from dataclasses import dataclass, field
from types import SimpleNamespace

from sglang_simulator.simulation.sglang import bench_runner as bench_runner_module
from sglang_simulator.simulation.sglang import scheduler as scheduler_module
from sglang_simulator.simulation.manager.failure import (
    BackupPolicy,
    FailureConfig,
    FailureManager,
)
from sglang_simulator.simulation.sglang.scheduler import (
    C_SchedulerHook,
    _apply_failover_simulation_stats,
    _build_failover_snapshot_and_mark_stats,
    _mark_completed_request_stats,
    _maybe_fire_failure,
    _maybe_schedule_offline_failure_on_request_start,
    _maybe_recover_failure,
    _reroute_failed_dp_request,
)
from sglang_simulator.simulation.types import RequestStats, SchedulerConfig


@dataclass
class FakeSamplingParams:
    max_new_tokens: int
    custom_params: dict | None = None


@dataclass
class FakeReqSnapshot:
    rid: str
    output_ids: list[int]
    backed_up_tokens: int = 0
    origin_input_ids: list[int] = field(default_factory=list)
    sampling_params: object | None = None
    original_max_new_tokens: int | None = None
    stream: bool = False


@dataclass
class FakeFailoverBatchInput:
    failed_dp_rank: int = 0
    snapshots: list[FakeReqSnapshot] = field(default_factory=list)


@dataclass
class FakeReq:
    rid: str
    origin_input_ids: list[int]
    output_ids: list[int]
    sampling_params: FakeSamplingParams
    mm_inputs: object = None
    return_logprob: bool = False
    logprob_start_len: int = 0
    top_logprobs_num: int = 0
    token_ids_logprob: list[int] = field(default_factory=list)
    stream: bool = False
    return_hidden_states: bool = False
    return_routed_experts: bool = False
    routed_experts_start_len: int = 0
    input_embeds: object = None
    session_params: object = None
    lora_id: str | None = None
    custom_logit_processor: str | None = None
    bootstrap_host: str | None = None
    bootstrap_port: int | None = None
    bootstrap_room: int | None = None
    decode_tp_size: int | None = None
    require_reasoning: bool = False
    routed_dp_rank: int | None = None
    disagg_prefill_dp_rank: int | None = None
    priority: int | None = None
    extra_key: str | None = None
    routing_key: str | None = None
    no_logs: bool = False
    return_bytes: bool = False
    return_entropy: bool = False
    token_type_ids: list[int] | None = None
    aborted_with: str | None = None

    def set_finish_with_abort(self, error_msg: str):
        self.aborted_with = error_msg


@dataclass
class FakeTokenizedReq:
    rid: str
    input_ids: list[int]
    sampling_params: FakeSamplingParams
    stream: bool = False


@dataclass
class FakeFinishedReq:
    rid: str

    def finished(self):
        return True


class FakeScheduler:
    def __init__(self, reqs):
        self.running_batch = SimpleNamespace(reqs=reqs)
        self.retried = []
        self.failover_outputs = []
        self.cleared = False
        self.send_to_tokenizer = SimpleNamespace(send_output=self._send_output)

    def handle_generate_request(self, req):
        self.retried.append(req)

    def _send_output(self, output, recv_obj=None):
        self.failover_outputs.append((output, recv_obj))

    def _clear_failed_rank_scheduler_state(self):
        self.cleared = True
        self.running_batch.reqs = []


def _make_failure_manager(policy=BackupPolicy.REMOTE_BACKUP) -> FailureManager:
    return FailureManager(
        FailureConfig(
            enabled=True,
            backup_policy=policy,
            failed_dp_rank=1,
            remote_backup_ratio=0.5,
            host_backup_ratio=0.25,
        )
    )


def test_failure_manager_estimates_backed_up_tokens_by_policy():
    mgr = _make_failure_manager()

    assert mgr.estimate_backed_up_tokens(BackupPolicy.NONE, 10) == 0
    assert mgr.estimate_backed_up_tokens(BackupPolicy.HOST, 10) == 2
    assert mgr.estimate_backed_up_tokens(BackupPolicy.REMOTE_BACKUP, 10) == 5


def test_failure_manager_picks_next_dp_rank_for_recovery():
    mgr = _make_failure_manager()

    assert mgr.pick_recovery_dp_rank(failed_dp_rank=1, dp_size=1) == 1
    assert mgr.pick_recovery_dp_rank(failed_dp_rank=1, dp_size=4) == 2
    assert mgr.pick_recovery_dp_rank(failed_dp_rank=3, dp_size=4) == 0


def test_build_failover_snapshot_marks_stats_and_preserves_original_rid():
    request_stats = defaultdict(RequestStats)
    req_stats = request_stats["req-1"]
    req_stats.rid = "req-1"
    req_stats.job_id = "job-7"
    req_stats.worker_id = "worker-3"
    req_stats.assigned_dp_rank = 1
    req_stats.backup_policy = "remote_backup"
    req_stats.attempt = 0

    req = FakeReq(
        rid="req-1",
        origin_input_ids=[11, 12, 13],
        output_ids=[21, 22],
        mm_inputs={"image": 1},
        sampling_params=FakeSamplingParams(
            max_new_tokens=5,
            custom_params={
                "simulation": {
                    "job_id": "job-7",
                    "worker_id": "worker-3",
                    "assigned_dp_rank": 1,
                    "backup_policy": "remote_backup",
                    "attempt": 0,
                    "created_time": 4.5,
                    "total_request": 9,
                }
            },
        ),
        return_logprob=True,
        logprob_start_len=2,
        top_logprobs_num=4,
        token_ids_logprob=[99],
        stream=True,
    )

    snapshot = _build_failover_snapshot_and_mark_stats(
        req=req,
        request_stats=request_stats,
        failure_manager=_make_failure_manager(),
        scheduler_config=SchedulerConfig(dp_size=4),
        event=SimpleNamespace(
            event_id=7,
            fire_time_s=8.0,
            failed_dp_rank=1,
            backup_policy=BackupPolicy.REMOTE_BACKUP,
        ),
        retry_created_time=8.25,
        req_snapshot_cls=FakeReqSnapshot,
    )

    assert snapshot is not None
    assert snapshot.rid == "req-1"
    assert snapshot.origin_input_ids == [11, 12, 13]
    assert snapshot.output_ids == [21, 22]
    assert snapshot.backed_up_tokens == 2
    assert snapshot.sampling_params is req.sampling_params
    assert snapshot.original_max_new_tokens == 5
    assert snapshot.stream is True

    assert req_stats.job_id == "job-7"
    assert req_stats.worker_id == "worker-3"
    assert req_stats.assigned_dp_rank == 1
    assert req_stats.backup_policy == "remote_backup"
    assert req_stats.attempt == 0
    assert req_stats.is_failover_retried is False
    assert req_stats.failure_impacted is True
    assert req_stats.failure_time_s == 8.0
    assert req_stats.pre_failover_output_tokens == 2
    assert req_stats.pre_failover_backed_up_tokens == 2
    assert req_stats.retry_prefill_tokens == 3
    assert req_stats.last_event_time == 8.25
    assert req_stats.queue_start == -1
    assert req_stats.failover_penalty_s == 0
    assert req_stats.prefetch_time_s == 0
    assert req_stats.status == "failed_over"

    sim_params = req.sampling_params.custom_params["simulation"]
    assert sim_params["job_id"] == "job-7"
    assert sim_params["worker_id"] == "worker-3"
    assert sim_params["assigned_dp_rank"] == 2
    assert sim_params["backup_policy"] == "remote_backup"
    assert sim_params["attempt"] == 1
    assert sim_params["created_time"] == 8.25
    assert sim_params["failure_impacted"] is True
    assert sim_params["is_failover_retried"] is True
    assert sim_params["failure_time_s"] == 8.0
    assert sim_params["pre_failover_output_tokens"] == 2
    assert sim_params["pre_failover_backed_up_tokens"] == 2
    assert sim_params["retry_prefill_tokens"] == 3
    assert sim_params["failover_penalty_s"] == 0
    assert sim_params["total_request"] == 9


def test_build_failover_snapshot_adds_restore_penalty_to_stats():
    request_stats = defaultdict(RequestStats)
    req_stats = request_stats["req-penalty"]
    req_stats.rid = "req-penalty"
    req_stats.assigned_dp_rank = 1

    req = FakeReq(
        rid="req-penalty",
        origin_input_ids=[11, 12],
        output_ids=[21, 22, 23, 24],
        sampling_params=FakeSamplingParams(
            max_new_tokens=6,
            custom_params={"simulation": {"created_time": 1.0, "total_request": 1}},
        ),
    )

    snapshot = _build_failover_snapshot_and_mark_stats(
        req=req,
        request_stats=request_stats,
        failure_manager=_make_failure_manager(),
        scheduler_config=SchedulerConfig(dp_size=2),
        event=SimpleNamespace(
            event_id=1,
            fire_time_s=3.0,
            failed_dp_rank=1,
            backup_policy=BackupPolicy.REMOTE_BACKUP,
        ),
        retry_created_time=3.25,
        req_snapshot_cls=FakeReqSnapshot,
        kv_bytes_per_token=100,
        memory_read_bandwidth=1000,
        disk_read_bandwidth=100,
    )

    assert snapshot is not None
    assert req_stats.pre_failover_backed_up_tokens == 3
    assert req_stats.failover_penalty_s == 3.0
    assert req_stats.prefetch_time_s == 0
    assert (
        req.sampling_params.custom_params["simulation"]["failover_penalty_s"]
        == 3.0
    )


def test_build_failover_snapshot_reuses_computed_prefix_tokens_for_backup():
    request_stats = defaultdict(RequestStats)
    req_stats = request_stats["req-prefix"]
    req_stats.rid = "req-prefix"
    req_stats.assigned_dp_rank = 1

    req = FakeReq(
        rid="req-prefix",
        origin_input_ids=[11, 12, 13, 14],
        output_ids=[21, 22],
        sampling_params=FakeSamplingParams(
            max_new_tokens=5,
            custom_params={"simulation": {"created_time": 1.0, "total_request": 1}},
        ),
    )

    snapshot = _build_failover_snapshot_and_mark_stats(
        req=req,
        request_stats=request_stats,
        failure_manager=_make_failure_manager(),
        scheduler_config=SchedulerConfig(dp_size=2),
        event=SimpleNamespace(
            event_id=1,
            fire_time_s=3.0,
            failed_dp_rank=1,
            backup_policy=BackupPolicy.REMOTE_BACKUP,
        ),
        retry_created_time=3.25,
        req_snapshot_cls=FakeReqSnapshot,
    )

    assert snapshot is not None
    assert req_stats.pre_failover_output_tokens == 2
    assert req_stats.pre_failover_backed_up_tokens == 3
    assert req_stats.retry_prefill_tokens == 3


def test_build_failover_snapshot_returns_none_when_no_tokens_remaining():
    request_stats = defaultdict(RequestStats)
    req = FakeReq(
        rid="req-2",
        origin_input_ids=[1],
        output_ids=[7, 8],
        sampling_params=FakeSamplingParams(
            max_new_tokens=2,
            custom_params={"simulation": {"created_time": 1.0, "total_request": 1}},
        ),
    )

    snapshot = _build_failover_snapshot_and_mark_stats(
        req=req,
        request_stats=request_stats,
        failure_manager=_make_failure_manager(),
        scheduler_config=SchedulerConfig(dp_size=2),
        event=SimpleNamespace(
            event_id=1,
            fire_time_s=3.0,
            failed_dp_rank=1,
            backup_policy=BackupPolicy.REMOTE_BACKUP,
        ),
        retry_created_time=3.25,
        req_snapshot_cls=FakeReqSnapshot,
    )

    assert snapshot is None
    assert request_stats["req-2"].status == "running"


def test_apply_failover_simulation_stats_restores_retry_metadata():
    req_stats = RequestStats(rid="same")

    _apply_failover_simulation_stats(
        req_stats,
        {
            "is_failover_retried": True,
            "failure_time_s": 8.0,
            "pre_failover_output_tokens": 2,
            "pre_failover_backed_up_tokens": 2,
            "retry_prefill_tokens": 3,
            "failover_penalty_s": 0.2,
        },
    )

    assert req_stats.failure_impacted is True
    assert req_stats.is_failover_retried is True
    assert req_stats.failure_time_s == 8.0
    assert req_stats.pre_failover_output_tokens == 2
    assert req_stats.pre_failover_backed_up_tokens == 2
    assert req_stats.retry_prefill_tokens == 3
    assert req_stats.failover_penalty_s == 0.2
    assert req_stats.prefetch_time_s == 0.2


def test_maybe_fire_failure_does_not_call_restore_timing_factory_when_no_failure():
    restore_timing_calls = []

    def restore_timing_kwargs_factory():
        restore_timing_calls.append("called")
        return {
            "kv_bytes_per_token": 100,
            "memory_read_bandwidth": 1000,
            "disk_read_bandwidth": 100,
        }

    fired = _maybe_fire_failure(
        scheduler=FakeScheduler([]),
        failure_manager=FailureManager(
            FailureConfig(
                enabled=True,
                backup_policy=BackupPolicy.HOST,
                failed_dp_rank=1,
                timeline_after_start_s=5.0,
            )
        ),
        request_stats=defaultdict(RequestStats),
        scheduler_config=SchedulerConfig(dp_size=2),
        now_s=4.0,
        failure_events=[],
        failed_dp_ranks=set(),
        restore_timing_kwargs_factory=restore_timing_kwargs_factory,
    )

    assert fired is False
    assert restore_timing_calls == []


def test_maybe_fire_failure_does_not_call_restore_timing_factory_without_retry():
    mgr = FailureManager(
        FailureConfig(
            enabled=True,
            backup_policy=BackupPolicy.HOST,
            failed_dp_rank=1,
            timeline_after_start_s=5.0,
        )
    )
    mgr.maybe_schedule_timeline()

    request_stats = defaultdict(RequestStats)
    request_stats["same"].assigned_dp_rank = 1
    request_stats["other"].assigned_dp_rank = 0

    finished_impacted = FakeReq(
        rid="same",
        origin_input_ids=[1, 2],
        output_ids=[3, 4],
        sampling_params=FakeSamplingParams(
            max_new_tokens=2,
            custom_params={"simulation": {"created_time": 1.0, "total_request": 1}},
        ),
    )
    unaffected = FakeReq(
        rid="other",
        origin_input_ids=[5],
        output_ids=[6],
        sampling_params=FakeSamplingParams(
            max_new_tokens=4,
            custom_params={"simulation": {"created_time": 2.0, "total_request": 1}},
        ),
    )
    calls = []

    def restore_timing_kwargs_factory():
        calls.append("called")
        return {
            "kv_bytes_per_token": 100,
            "memory_read_bandwidth": 1000,
            "disk_read_bandwidth": 100,
        }

    fired = _maybe_fire_failure(
        scheduler=FakeScheduler([finished_impacted, unaffected]),
        failure_manager=mgr,
        request_stats=request_stats,
        scheduler_config=SchedulerConfig(dp_size=2),
        now_s=5.5,
        failure_events=[],
        failed_dp_ranks=set(),
        restore_timing_kwargs_factory=restore_timing_kwargs_factory,
    )

    assert fired is True
    assert finished_impacted.aborted_with is None
    assert unaffected.aborted_with is None
    assert calls == []


def test_maybe_fire_failure_uses_tokenizer_failover_with_original_rid():
    mgr = FailureManager(
        FailureConfig(
            enabled=True,
            backup_policy=BackupPolicy.HOST,
            failed_dp_rank=1,
            timeline_after_start_s=5.0,
            host_backup_ratio=0.5,
        )
    )
    mgr.maybe_schedule_timeline()

    request_stats = defaultdict(RequestStats)
    request_stats["same"].assigned_dp_rank = 1
    request_stats["same"].job_id = "job-a"
    request_stats["same"].worker_id = "worker-a"
    request_stats["same"].backup_policy = "host"
    request_stats["same"].created_time = 1.0
    request_stats["other"].assigned_dp_rank = 0
    request_stats["other"].job_id = "job-b"
    request_stats["other"].worker_id = "worker-b"
    request_stats["other"].backup_policy = "host"
    request_stats["other"].created_time = 2.0

    impacted = FakeReq(
        rid="same",
        origin_input_ids=[1, 2],
        output_ids=[3, 4],
        sampling_params=FakeSamplingParams(
            max_new_tokens=4,
            custom_params={
                "simulation": {
                    "job_id": "job-a",
                    "worker_id": "worker-a",
                    "assigned_dp_rank": 1,
                    "backup_policy": "host",
                    "attempt": 0,
                    "created_time": 1.0,
                    "total_request": 2,
                }
            },
        ),
    )
    unaffected = FakeReq(
        rid="other",
        origin_input_ids=[5],
        output_ids=[6],
        sampling_params=FakeSamplingParams(
            max_new_tokens=4,
            custom_params={
                "simulation": {
                    "job_id": "job-b",
                    "worker_id": "worker-b",
                    "assigned_dp_rank": 0,
                    "backup_policy": "host",
                    "attempt": 0,
                    "created_time": 2.0,
                    "total_request": 2,
                }
            },
        ),
    )
    scheduler = FakeScheduler([impacted, unaffected])
    restore_timing_calls = []

    def restore_timing_kwargs_factory():
        restore_timing_calls.append("called")
        return {
            "kv_bytes_per_token": 100,
            "memory_read_bandwidth": 1000,
            "disk_read_bandwidth": 100,
        }

    fired = _maybe_fire_failure(
        scheduler=scheduler,
        failure_manager=mgr,
        request_stats=request_stats,
        failover_batch_cls=FakeFailoverBatchInput,
        req_snapshot_cls=FakeReqSnapshot,
        scheduler_config=SchedulerConfig(dp_size=2),
        now_s=5.5,
        failure_events=[],
        failed_dp_ranks=set(),
        restore_timing_kwargs_factory=restore_timing_kwargs_factory,
    )

    assert fired is True
    assert impacted.aborted_with is None
    assert unaffected.aborted_with is None
    assert restore_timing_calls == ["called"]
    assert scheduler.retried == []
    assert scheduler.cleared is True
    assert len(scheduler.failover_outputs) == 1
    failover, recv_obj = scheduler.failover_outputs[0]
    assert failover.failed_dp_rank == 1
    assert recv_obj is impacted
    assert len(failover.snapshots) == 1
    snapshot = failover.snapshots[0]
    assert snapshot.rid == "same"
    assert snapshot.origin_input_ids == [1, 2]
    assert snapshot.output_ids == [3, 4]
    assert snapshot.backed_up_tokens == 2
    assert snapshot.original_max_new_tokens == 4
    assert request_stats["same"].status == "failed_over"
    assert request_stats["same"].is_failover_retried is False
    assert request_stats["same"].retry_prefill_tokens == 2
    assert request_stats["same"].failover_penalty_s == 0.2
    assert request_stats["same"].prefetch_time_s == 0
    assert request_stats["other"].status == "running"


def test_maybe_fire_failure_snapshots_waiting_requests_before_clearing_failed_rank():
    mgr = FailureManager(
        FailureConfig(
            enabled=True,
            backup_policy=BackupPolicy.REMOTE_BACKUP,
            failed_dp_rank=1,
            timeline_after_start_s=5.0,
            remote_backup_ratio=0.5,
        )
    )
    mgr.maybe_schedule_timeline()

    request_stats = defaultdict(RequestStats)
    for rid in ("active", "waiting"):
        request_stats[rid].assigned_dp_rank = 1
        request_stats[rid].job_id = rid
        request_stats[rid].backup_policy = "remote_backup"

    active = FakeReq(
        rid="active",
        origin_input_ids=[1, 2],
        output_ids=[3],
        sampling_params=FakeSamplingParams(
            max_new_tokens=4,
            custom_params={
                "simulation": {
                    "job_id": "active",
                    "assigned_dp_rank": 1,
                    "backup_policy": "remote_backup",
                    "created_time": 1.0,
                    "total_request": 2,
                }
            },
        ),
    )
    waiting = FakeReq(
        rid="waiting",
        origin_input_ids=[10, 11],
        output_ids=[],
        sampling_params=FakeSamplingParams(
            max_new_tokens=4,
            custom_params={
                "simulation": {
                    "job_id": "waiting",
                    "assigned_dp_rank": 1,
                    "backup_policy": "remote_backup",
                    "created_time": 2.0,
                    "total_request": 2,
                }
            },
        ),
    )
    scheduler = FakeScheduler([active])
    scheduler.waiting_queue = [waiting]
    scheduler._collect_failover_reqs = lambda: [active, waiting]

    fired = _maybe_fire_failure(
        scheduler=scheduler,
        failure_manager=mgr,
        request_stats=request_stats,
        failover_batch_cls=FakeFailoverBatchInput,
        req_snapshot_cls=FakeReqSnapshot,
        scheduler_config=SchedulerConfig(dp_size=2),
        now_s=5.5,
        failure_events=[],
        failed_dp_ranks=set(),
    )

    assert fired is True
    assert scheduler.cleared is True
    assert len(scheduler.failover_outputs) == 1
    failover, _ = scheduler.failover_outputs[0]
    assert [snapshot.rid for snapshot in failover.snapshots] == [
        "active",
        "waiting",
    ]
    assert request_stats["active"].status == "failed_over"
    assert request_stats["waiting"].status == "failed_over"


def test_maybe_fire_failure_snapshots_parallel_worker_pending_requests():
    mgr = FailureManager(
        FailureConfig(
            enabled=True,
            backup_policy=BackupPolicy.REMOTE_BACKUP,
            failed_dp_rank=1,
            timeline_after_start_s=5.0,
            remote_backup_ratio=0.5,
        )
    )
    mgr.maybe_schedule_timeline()

    C_SchedulerHook.FUTURE_QUEUE.clear()
    C_SchedulerHook.WORKER_PENDING.clear()

    request_stats = defaultdict(RequestStats)
    request_stats["active"].assigned_dp_rank = 1
    request_stats["active"].job_id = "active"
    request_stats["active"].backup_policy = "remote_backup"

    active = FakeReq(
        rid="active",
        origin_input_ids=[1, 2],
        output_ids=[3],
        sampling_params=FakeSamplingParams(
            max_new_tokens=4,
            custom_params={
                "simulation": {
                    "job_id": "active",
                    "worker_id": "worker-a",
                    "worker_seq": 0,
                    "parallel_worker_client": True,
                    "assigned_dp_rank": 1,
                    "backup_policy": "remote_backup",
                    "created_time": 1.0,
                    "total_request": 2,
                }
            },
        ),
    )
    future = FakeReq(
        rid="future",
        origin_input_ids=[10, 11],
        output_ids=[],
        sampling_params=FakeSamplingParams(
            max_new_tokens=4,
            custom_params={
                "simulation": {
                    "job_id": "future",
                    "worker_id": "worker-a",
                    "worker_seq": 1,
                    "parallel_worker_client": True,
                    "assigned_dp_rank": 1,
                    "backup_policy": "remote_backup",
                    "created_time": 2.0,
                    "total_request": 2,
                }
            },
        ),
    )
    C_SchedulerHook.WORKER_PENDING["worker-a"] = [(0, active), (1, future)]
    scheduler = FakeScheduler([active])
    scheduler._collect_failover_reqs = lambda: [active]

    try:
        fired = _maybe_fire_failure(
            scheduler=scheduler,
            failure_manager=mgr,
            request_stats=request_stats,
            failover_batch_cls=FakeFailoverBatchInput,
            req_snapshot_cls=FakeReqSnapshot,
            scheduler_config=SchedulerConfig(dp_size=2),
            now_s=5.5,
            failure_events=[],
            failed_dp_ranks=set(),
        )

        assert fired is True
        failover, _ = scheduler.failover_outputs[0]
        assert [snapshot.rid for snapshot in failover.snapshots] == [
            "active",
            "future",
        ]
        assert request_stats["future"].assigned_dp_rank == 1
        assert request_stats["future"].status == "failed_over"
        assert C_SchedulerHook.WORKER_PENDING["worker-a"] == []
    finally:
        C_SchedulerHook.FUTURE_QUEUE.clear()
        C_SchedulerHook.WORKER_PENDING.clear()


def test_maybe_fire_failure_snapshots_tokenized_pending_worker_request():
    mgr = FailureManager(
        FailureConfig(
            enabled=True,
            backup_policy=BackupPolicy.REMOTE_BACKUP,
            failed_dp_rank=1,
            timeline_after_start_s=5.0,
            remote_backup_ratio=0.5,
        )
    )
    mgr.maybe_schedule_timeline()

    C_SchedulerHook.FUTURE_QUEUE.clear()
    C_SchedulerHook.WORKER_PENDING.clear()

    pending = FakeTokenizedReq(
        rid="tokenized-pending",
        input_ids=[10, 11],
        sampling_params=FakeSamplingParams(
            max_new_tokens=4,
            custom_params={
                "simulation": {
                    "job_id": "tokenized-pending",
                    "worker_id": "worker-a",
                    "worker_seq": 1,
                    "parallel_worker_client": True,
                    "assigned_dp_rank": 1,
                    "backup_policy": "remote_backup",
                    "created_time": 2.0,
                    "total_request": 2,
                }
            },
        ),
    )
    C_SchedulerHook.WORKER_PENDING["worker-a"] = [(1, pending)]
    scheduler = FakeScheduler([])
    scheduler._collect_failover_reqs = lambda: []

    try:
        fired = _maybe_fire_failure(
            scheduler=scheduler,
            failure_manager=mgr,
            request_stats=defaultdict(RequestStats),
            failover_batch_cls=FakeFailoverBatchInput,
            req_snapshot_cls=FakeReqSnapshot,
            scheduler_config=SchedulerConfig(dp_size=2),
            now_s=5.5,
            failure_events=[],
            failed_dp_ranks=set(),
        )

        assert fired is True
        failover, _ = scheduler.failover_outputs[0]
        assert len(failover.snapshots) == 1
        snapshot = failover.snapshots[0]
        assert snapshot.rid == "tokenized-pending"
        assert snapshot.origin_input_ids == [10, 11]
        assert snapshot.output_ids == []
        assert C_SchedulerHook.WORKER_PENDING["worker-a"] == []
    finally:
        C_SchedulerHook.FUTURE_QUEUE.clear()
        C_SchedulerHook.WORKER_PENDING.clear()


def test_maybe_fire_failure_skips_completed_stale_worker_pending_request():
    mgr = FailureManager(
        FailureConfig(
            enabled=True,
            backup_policy=BackupPolicy.REMOTE_BACKUP,
            failed_dp_rank=1,
            timeline_after_start_s=5.0,
            remote_backup_ratio=0.5,
        )
    )
    mgr.maybe_schedule_timeline()

    C_SchedulerHook.FUTURE_QUEUE.clear()
    C_SchedulerHook.WORKER_PENDING.clear()

    request_stats = defaultdict(RequestStats)
    request_stats["done"].rid = "done"
    request_stats["done"].assigned_dp_rank = 1
    request_stats["done"].status = "completed"

    done = FakeTokenizedReq(
        rid="done",
        input_ids=[1],
        sampling_params=FakeSamplingParams(
            max_new_tokens=4,
            custom_params={
                "simulation": {
                    "job_id": "done",
                    "worker_id": "worker-a",
                    "worker_seq": 0,
                    "parallel_worker_client": True,
                    "assigned_dp_rank": 1,
                    "backup_policy": "remote_backup",
                    "created_time": 1.0,
                    "total_request": 2,
                }
            },
        ),
    )
    future = FakeTokenizedReq(
        rid="future",
        input_ids=[2],
        sampling_params=FakeSamplingParams(
            max_new_tokens=4,
            custom_params={
                "simulation": {
                    "job_id": "future",
                    "worker_id": "worker-a",
                    "worker_seq": 1,
                    "parallel_worker_client": True,
                    "assigned_dp_rank": 1,
                    "backup_policy": "remote_backup",
                    "created_time": 2.0,
                    "total_request": 2,
                }
            },
        ),
    )
    C_SchedulerHook.WORKER_PENDING["worker-a"] = [(0, done), (1, future)]
    scheduler = FakeScheduler([])
    scheduler._collect_failover_reqs = lambda: []

    try:
        fired = _maybe_fire_failure(
            scheduler=scheduler,
            failure_manager=mgr,
            request_stats=request_stats,
            failover_batch_cls=FakeFailoverBatchInput,
            req_snapshot_cls=FakeReqSnapshot,
            scheduler_config=SchedulerConfig(dp_size=2),
            now_s=5.5,
            failure_events=[],
            failed_dp_ranks=set(),
        )

        assert fired is True
        failover, _ = scheduler.failover_outputs[0]
        assert [snapshot.rid for snapshot in failover.snapshots] == ["future"]
        assert request_stats["done"].status == "completed"
        assert request_stats["future"].status == "failed_over"
    finally:
        C_SchedulerHook.FUTURE_QUEUE.clear()
        C_SchedulerHook.WORKER_PENDING.clear()


def test_mark_completed_request_stats_marks_failover_retry_completed():
    request_stats = defaultdict(RequestStats)
    request_stats["failed"].rid = "failed"
    request_stats["failed"].status = "running"
    request_stats["failed"].is_failover_retried = True
    request_stats["done"].rid = "done"
    request_stats["done"].status = "running"

    _mark_completed_request_stats(
        request_stats,
        SimpleNamespace(
            reqs=[
                FakeFinishedReq("failed"),
                FakeFinishedReq("done"),
            ]
        ),
    )

    assert request_stats["failed"].status == "completed"
    assert request_stats["failed"].is_failover_retried is True
    assert request_stats["done"].status == "completed"


def test_maybe_fire_failure_records_event_and_mutates_state():
    mgr = FailureManager(
        FailureConfig(
            enabled=True,
            backup_policy=BackupPolicy.REMOTE_BACKUP,
            failed_dp_rank=2,
            timeline_after_start_s=1.0,
        )
    )
    mgr.maybe_schedule_timeline()
    failure_events = []
    failed_dp_ranks = set()
    scheduler = FakeScheduler([])

    fired = _maybe_fire_failure(
        scheduler=scheduler,
        failure_manager=mgr,
        request_stats=defaultdict(RequestStats),
        scheduler_config=SchedulerConfig(dp_size=4),
        now_s=1.0,
        failure_events=failure_events,
        failed_dp_ranks=failed_dp_ranks,
    )

    assert fired is True
    assert failure_events == [
        {
            "event": "sim_gpu_failure",
            "event_id": 1,
            "timestamp_s": 1.0,
            "failed_dp_rank": 2,
            "backup_policy": "remote_backup",
        }
    ]
    assert failed_dp_ranks == {2}


def test_reroute_failed_dp_request_updates_metadata_and_custom_params():
    manager = _make_failure_manager()
    request_stats = RequestStats(rid="future", assigned_dp_rank=1)
    req = FakeReq(
        rid="future",
        origin_input_ids=[1],
        output_ids=[],
        sampling_params=FakeSamplingParams(
            max_new_tokens=2,
            custom_params={
                "simulation": {
                    "assigned_dp_rank": 1,
                    "job_id": "future-job",
                }
            },
        ),
    )

    rerouted = _reroute_failed_dp_request(
        req=req,
        req_stats=request_stats,
        failed_dp_ranks={1},
        failure_manager=manager,
        scheduler_config=SchedulerConfig(dp_size=4),
    )

    assert rerouted is True
    assert request_stats.assigned_dp_rank == 2
    assert request_stats.failure_impacted is True
    sim_params = req.sampling_params.custom_params["simulation"]
    assert sim_params["assigned_dp_rank"] == 2


def test_maybe_recover_failure_clears_failed_rank_and_records_event():
    manager = FailureManager(
        FailureConfig(
            enabled=True,
            backup_policy=BackupPolicy.REMOTE_BACKUP,
            failed_dp_rank=1,
            inject_after_request=1,
            recover_after_request=2,
        )
    )
    manager.on_request_completed("r1", completion_time_s=1.0)
    manager.should_fire(1.0)
    manager.on_request_completed("r2", completion_time_s=2.0)
    failed_dp_ranks = {1}
    failure_events = []

    recovered = _maybe_recover_failure(
        failure_manager=manager,
        now_s=2.0,
        failed_dp_ranks=failed_dp_ranks,
        failure_events=failure_events,
    )

    assert recovered is True
    assert failed_dp_ranks == set()
    assert failure_events == [
        {
            "event": "sim_gpu_recovery",
            "timestamp_s": 2.0,
            "recovered_dp_rank": 1,
        }
    ]


def test_offline_failure_schedules_from_request_start_and_fires_mid_request():
    manager = FailureManager(
        FailureConfig(
            enabled=True,
            backup_policy=BackupPolicy.HOST,
            failed_dp_rank=0,
            inject_after_job=10,
            inject_delay_s=5.0,
        )
    )
    failed_dp_ranks = set()
    failure_events = []

    fired = _maybe_schedule_offline_failure_on_request_start(
        failure_manager=manager,
        simulation_args={
            "job_id": "11",
            "assigned_dp_rank": 0,
        },
        request_start_s=100.0,
        now_s=104.0,
        failed_dp_ranks=failed_dp_ranks,
        failure_events=failure_events,
    )

    assert fired is False
    assert failed_dp_ranks == set()

    fired = _maybe_schedule_offline_failure_on_request_start(
        failure_manager=manager,
        simulation_args={
            "job_id": "11",
            "assigned_dp_rank": 0,
        },
        request_start_s=100.0,
        now_s=105.0,
        failed_dp_ranks=failed_dp_ranks,
        failure_events=failure_events,
    )

    assert fired is True
    assert failed_dp_ranks == {0}
    assert failure_events == [
        {
            "event": "sim_gpu_failure",
            "event_id": 1,
            "timestamp_s": 105.0,
            "failed_dp_rank": 0,
            "backup_policy": "host",
        }
    ]


def test_offline_failure_recovers_from_request_start_threshold():
    manager = FailureManager(
        FailureConfig(
            enabled=True,
            backup_policy=BackupPolicy.REMOTE_BACKUP,
            failed_dp_rank=0,
            inject_after_job=10,
            recover_after_job=20,
        )
    )
    failed_dp_ranks = set()
    failure_events = []
    _maybe_schedule_offline_failure_on_request_start(
        failure_manager=manager,
        simulation_args={
            "job_id": "11",
            "assigned_dp_rank": 0,
        },
        request_start_s=100.0,
        now_s=100.0,
        failed_dp_ranks=failed_dp_ranks,
        failure_events=failure_events,
    )

    assert failed_dp_ranks == {0}

    recovered = _maybe_schedule_offline_failure_on_request_start(
        failure_manager=manager,
        simulation_args={
            "job_id": "21",
            "assigned_dp_rank": 0,
        },
        request_start_s=140.0,
        now_s=140.0,
        failed_dp_ranks=failed_dp_ranks,
        failure_events=failure_events,
    )

    assert recovered is False
    assert failed_dp_ranks == set()
    assert failure_events[-1] == {
        "event": "sim_gpu_recovery",
        "timestamp_s": 140.0,
        "recovered_dp_rank": 0,
    }


def test_profile_persists_failure_events_and_resets_hook_state(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_SIMULATOR_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(
        scheduler_module,
        "_reset_failure_manager_state",
        lambda: setattr(
            C_SchedulerHook,
            "FAILURE_MANAGER",
            FailureManager(FailureConfig(enabled=True)),
        ),
    )

    class FakeProfileReqOutput:
        def __init__(self, success, message):
            self.success = success
            self.message = message

    def fake_import_module(name, package=None):
        if name == "sglang.srt.managers.io_struct":
            return SimpleNamespace(ProfileReqOutput=FakeProfileReqOutput)
        return original_import_module(name, package)

    class FakeTarget:
        def __init__(self):
            pass

        def recv_requests(self):
            return []

        def get_new_batch_prefill(self):
            return None

        def run_batch(self):
            return None

        def process_batch_result(self):
            return None

        def event_loop_normal(self):
            return None

        def profile(self, req, *args, **kwargs):
            return None

    original_import_module = scheduler_module.importlib.import_module
    monkeypatch.setattr(
        scheduler_module.importlib, "import_module", fake_import_module
    )
    C_SchedulerHook.hook(FakeTarget)

    C_SchedulerHook.REQUEST_STATS.clear()
    C_SchedulerHook.ITERATION_STATS.clear()
    C_SchedulerHook.FAILURE_EVENTS[:] = [
        {
            "event": "sim_gpu_failure",
            "event_id": 1,
            "timestamp_s": 3.5,
            "failed_dp_rank": 0,
            "backup_policy": "remote_backup",
        }
    ]
    C_SchedulerHook.FAILED_DP_RANKS.add(0)
    C_SchedulerHook.FAILURE_MANAGER = object()

    object.__new__(FakeTarget).profile(None)

    runner = object.__new__(bench_runner_module.SGLangBenchmarkRunner)
    failure_events = [
        row
        for row in runner.get_failure_events()
        if row["event"] == "sim_gpu_failure"
    ]

    assert failure_events
    assert failure_events[0]["failed_dp_rank"] == 0
    assert C_SchedulerHook.FAILURE_EVENTS == []
    assert C_SchedulerHook.FAILED_DP_RANKS == set()
    assert C_SchedulerHook.FAILURE_MANAGER is not None
