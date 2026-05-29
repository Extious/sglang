import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT)]

from benchmark.fault_tolerance.simulator.synthesis_requests.common.payload import (
    build_generate_kwargs,
    build_http_generate_payload,
    with_total_request,
)
from benchmark.fault_tolerance.simulator.synthesis_requests.common.schema import (
    BackupPolicy,
    SyntheticRequest,
)


def _request() -> SyntheticRequest:
    return SyntheticRequest(
        job_id="7",
        worker_id="2",
        worker_seq=3,
        assigned_dp_rank=1,
        token_ids=[11, 12, 13],
        output_length=5,
        created_time=4.5,
        backup_policy=BackupPolicy.HOST,
    )


def test_with_total_request_adds_total_request_without_mutating_request():
    request = _request()
    original_custom_params = request.custom_params

    payload = with_total_request(request, total_request=9)

    assert payload == {
        "job_id": "7",
        "worker_id": "2",
        "worker_seq": 3,
        "parallel_worker_client": True,
        "assigned_dp_rank": 1,
        "backup_policy": "host",
        "attempt": 0,
        "created_time": 4.5,
        "total_request": 9,
    }
    assert type(payload["total_request"]) is int

    payload["attempt"] = 8
    payload["total_request"] = 10

    assert original_custom_params == {
        "job_id": "7",
        "worker_id": "2",
        "worker_seq": 3,
        "parallel_worker_client": True,
        "assigned_dp_rank": 1,
        "backup_policy": "host",
        "attempt": 0,
        "created_time": 4.5,
    }
    assert request.custom_params == original_custom_params


def test_build_generate_kwargs_builds_offline_payload_and_copies_input_ids():
    request = _request()

    payload = build_generate_kwargs(request, total_request=9)

    assert payload["prompt"] == ""
    assert payload["input_ids"] == [11, 12, 13]
    assert isinstance(payload["input_ids"], list)
    assert payload["sampling_params"]["ignore_eos"] is True
    assert payload["sampling_params"]["max_new_tokens"] == 5
    assert type(payload["sampling_params"]["max_new_tokens"]) is int
    assert payload["sampling_params"]["custom_params"]["simulation"] == {
        "job_id": "7",
        "worker_id": "2",
        "worker_seq": 3,
        "parallel_worker_client": True,
        "assigned_dp_rank": 1,
        "backup_policy": "host",
        "attempt": 0,
        "created_time": 4.5,
        "total_request": 9,
    }
    assert payload["routed_dp_rank"] == 1
    assert type(payload["routed_dp_rank"]) is int

    payload["input_ids"].append(99)

    assert request.token_ids == (11, 12, 13)


def test_build_http_generate_payload_matches_generate_kwargs():
    request = _request()

    payload = build_http_generate_payload(request, total_request=9)

    assert payload == build_generate_kwargs(request, total_request=9)
    assert payload["sampling_params"]["custom_params"]["simulation"]["worker_id"] == "2"
    assert payload["sampling_params"]["custom_params"]["simulation"]["worker_seq"] == 3
    assert (
        payload["sampling_params"]["custom_params"]["simulation"][
            "parallel_worker_client"
        ]
        is True
    )
