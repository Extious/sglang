from __future__ import annotations

import random
from collections.abc import Sequence
from math import isinf, isnan

from .schema import BackupPolicy, SyntheticRequest, WorkloadConfig


def build_synthetic_requests(
    tokenizer: object,
    workload: WorkloadConfig,
    dp_size: int,
    backup_policy: BackupPolicy,
) -> list[SyntheticRequest]:
    token_rng = random.Random(workload.seed)
    arrival_rng = random.Random(workload.seed + 1)
    vocab = getattr(tokenizer, "vocab_size", None) or 32000
    low = max(1, vocab // 4)
    high = max(low + 1, vocab * 3 // 4)
    request_count = max(workload.num_requests, 0)
    worker_count = max(workload.app_workers, 1)
    dp_count = max(dp_size, 1)
    input_len = max(workload.input_len, 1)
    output_len = max(workload.output_len, 1)
    shared_prefix_len = min(max(workload.shared_prefix_len, 0), input_len)
    suffix_len = input_len - shared_prefix_len
    shared_prefix = tuple(
        token_rng.randrange(low, high) for _ in range(shared_prefix_len)
    )
    request_rate = workload.request_rate

    if isnan(request_rate) or request_rate <= 0:
        raise ValueError("request_rate must be positive or infinity")
    finite_request_rate = not isinf(request_rate)

    requests = []
    created_time = 0.0
    for idx in range(request_count):
        worker_index = idx % worker_count
        unique_suffix = tuple(token_rng.randrange(low, high) for _ in range(suffix_len))
        token_ids = shared_prefix + unique_suffix
        requests.append(
            SyntheticRequest(
                job_id=str(idx + 1),
                worker_id=str(worker_index + 1),
                worker_seq=idx // worker_count,
                assigned_dp_rank=worker_index % dp_count,
                token_ids=token_ids,
                output_length=output_len,
                created_time=created_time,
                backup_policy=backup_policy,
            )
        )
        if finite_request_rate:
            created_time += arrival_rng.expovariate(request_rate)
    return requests


def split_requests_by_worker(
    requests: Sequence[SyntheticRequest],
    app_workers: int,
) -> dict[str, list[SyntheticRequest]]:
    by_worker = {str(idx): [] for idx in range(1, max(app_workers, 0) + 1)}
    for request in requests:
        by_worker.setdefault(request.worker_id, []).append(request)
    return by_worker
