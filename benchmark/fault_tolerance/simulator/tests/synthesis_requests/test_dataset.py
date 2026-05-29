import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT)]

from benchmark.fault_tolerance.simulator.synthesis_requests.common.dataset import (
    build_synthetic_requests,
    split_requests_by_worker,
)
from benchmark.fault_tolerance.simulator.synthesis_requests.common.schema import (
    BackupPolicy,
    SyntheticRequest,
    WorkloadConfig,
)


class DummyTokenizer:
    vocab_size = 100


def test_build_synthetic_requests_sets_lengths_and_metadata():
    workload = WorkloadConfig(
        input_len=0,
        output_len=0,
        num_requests=4,
        app_workers=2,
        request_rate=float("inf"),
        seed=7,
    )

    requests = build_synthetic_requests(
        DummyTokenizer(),
        workload,
        dp_size=2,
        backup_policy=BackupPolicy.HOST,
    )

    assert len(requests) == 4
    assert [request.job_id for request in requests] == ["1", "2", "3", "4"]
    assert [request.worker_id for request in requests] == ["1", "2", "1", "2"]
    assert [request.worker_seq for request in requests] == [0, 0, 1, 1]
    assert [request.assigned_dp_rank for request in requests] == [0, 1, 0, 1]
    assert [request.output_length for request in requests] == [1, 1, 1, 1]
    assert [request.created_time for request in requests] == [0.0, 0.0, 0.0, 0.0]
    assert [request.backup_policy for request in requests] == [BackupPolicy.HOST] * 4
    assert all(len(request.token_ids) == 1 for request in requests)
    assert all(isinstance(request.token_ids, tuple) for request in requests)
    assert all(25 <= request.token_ids[0] < 75 for request in requests)


def test_build_synthetic_requests_is_deterministic():
    workload = WorkloadConfig(
        input_len=3,
        output_len=2,
        num_requests=3,
        app_workers=1,
        request_rate=float("inf"),
        seed=11,
    )

    first = build_synthetic_requests(
        DummyTokenizer(),
        workload,
        dp_size=0,
        backup_policy=BackupPolicy.REMOTE_BACKUP,
    )
    second = build_synthetic_requests(
        DummyTokenizer(),
        workload,
        dp_size=0,
        backup_policy=BackupPolicy.REMOTE_BACKUP,
    )

    assert [request.token_ids for request in first] == [
        request.token_ids for request in second
    ]
    assert [request.assigned_dp_rank for request in first] == [0, 0, 0]
    assert [request.output_length for request in first] == [2, 2, 2]
    assert all(len(request.token_ids) == 3 for request in first)


def test_build_synthetic_requests_uses_deterministic_finite_request_rate():
    workload = WorkloadConfig(
        input_len=3,
        output_len=2,
        num_requests=4,
        app_workers=1,
        request_rate=2.0,
        seed=9,
    )

    first = build_synthetic_requests(
        DummyTokenizer(),
        workload,
        dp_size=1,
        backup_policy=BackupPolicy.HOST,
    )
    second = build_synthetic_requests(
        DummyTokenizer(),
        workload,
        dp_size=1,
        backup_policy=BackupPolicy.HOST,
    )

    first_created_times = [request.created_time for request in first]
    second_created_times = [request.created_time for request in second]
    assert first_created_times == second_created_times
    assert first_created_times[0] == 0.0
    assert any(created_time > 0.0 for created_time in first_created_times[1:])


def test_build_synthetic_requests_keeps_tokens_independent_from_request_rate():
    base_workload = dict(
        input_len=3,
        output_len=2,
        num_requests=4,
        app_workers=1,
        seed=9,
    )
    inf_requests = build_synthetic_requests(
        DummyTokenizer(),
        WorkloadConfig(request_rate=float("inf"), **base_workload),
        dp_size=1,
        backup_policy=BackupPolicy.HOST,
    )
    finite_requests = build_synthetic_requests(
        DummyTokenizer(),
        WorkloadConfig(request_rate=2.0, **base_workload),
        dp_size=1,
        backup_policy=BackupPolicy.HOST,
    )

    assert [request.token_ids for request in finite_requests] == [
        request.token_ids for request in inf_requests
    ]
    assert finite_requests[0].created_time == 0.0
    assert any(request.created_time > 0.0 for request in finite_requests[1:])


def test_build_synthetic_requests_reuses_configured_shared_prefix():
    workload = WorkloadConfig(
        input_len=6,
        output_len=2,
        num_requests=4,
        app_workers=2,
        request_rate=float("inf"),
        seed=13,
        shared_prefix_len=3,
    )

    requests = build_synthetic_requests(
        DummyTokenizer(),
        workload,
        dp_size=2,
        backup_policy=BackupPolicy.REMOTE_BACKUP,
    )

    shared_prefixes = [request.token_ids[:3] for request in requests]
    suffixes = [request.token_ids[3:] for request in requests]
    assert len({prefix for prefix in shared_prefixes}) == 1
    assert len({suffix for suffix in suffixes}) > 1
    assert all(len(request.token_ids) == 6 for request in requests)


def test_build_synthetic_requests_clamps_shared_prefix_to_input_length():
    workload = WorkloadConfig(
        input_len=2,
        output_len=2,
        num_requests=3,
        app_workers=1,
        request_rate=float("inf"),
        seed=17,
        shared_prefix_len=99,
    )

    requests = build_synthetic_requests(
        DummyTokenizer(),
        workload,
        dp_size=1,
        backup_policy=BackupPolicy.HOST,
    )

    assert len({request.token_ids for request in requests}) == 1
    assert all(len(request.token_ids) == 2 for request in requests)


def test_build_synthetic_requests_keeps_each_worker_on_one_dp_rank():
    workload = WorkloadConfig(
        input_len=1,
        output_len=1,
        num_requests=6,
        app_workers=1,
        request_rate=float("inf"),
        seed=5,
    )

    requests = build_synthetic_requests(
        DummyTokenizer(),
        workload,
        dp_size=2,
        backup_policy=BackupPolicy.NONE,
    )

    assert [request.worker_seq for request in requests] == [0, 1, 2, 3, 4, 5]
    assert [request.assigned_dp_rank for request in requests] == [0, 0, 0, 0, 0, 0]


@pytest.mark.parametrize("request_rate", [0.0, -1.0, float("nan"), float("-inf")])
def test_build_synthetic_requests_rejects_invalid_request_rate(request_rate):
    workload = WorkloadConfig(
        input_len=3,
        output_len=2,
        num_requests=4,
        app_workers=1,
        request_rate=request_rate,
        seed=9,
    )

    with pytest.raises(ValueError, match="request_rate must be positive or infinity"):
        build_synthetic_requests(
            DummyTokenizer(),
            workload,
            dp_size=1,
            backup_policy=BackupPolicy.HOST,
        )


def test_split_requests_by_worker():
    requests = [
        SyntheticRequest(
            job_id="1",
            worker_id="1",
            worker_seq=0,
            assigned_dp_rank=0,
            token_ids=(1,),
            output_length=1,
            created_time=0.0,
            backup_policy=BackupPolicy.NONE,
        ),
        SyntheticRequest(
            job_id="2",
            worker_id="3",
            worker_seq=0,
            assigned_dp_rank=0,
            token_ids=(2,),
            output_length=1,
            created_time=0.0,
            backup_policy=BackupPolicy.NONE,
        ),
    ]

    by_worker = split_requests_by_worker(requests, app_workers=2)

    assert list(by_worker) == ["1", "2", "3"]
    assert by_worker["1"] == [requests[0]]
    assert by_worker["2"] == []
    assert by_worker["3"] == [requests[1]]
