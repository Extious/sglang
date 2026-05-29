from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from math import isinf
from pathlib import Path
from typing import Any

import requests as requests_module

from benchmark.fault_tolerance.simulator.synthesis_requests.common.payload import (
    build_http_generate_payload,
)
from benchmark.fault_tolerance.simulator.synthesis_requests.common.dataset import (
    split_requests_by_worker,
)
from benchmark.fault_tolerance.simulator.synthesis_requests.common.schema import (
    SyntheticRequest,
)


def build_generate_url(host, port) -> str:
    return f"http://{host}:{int(port)}/generate"


def build_profile_url(host, port) -> str:
    return f"http://{host}:{int(port)}/start_profile"


def wait_for_server_ready(
    host,
    port,
    timeout_s: float = 300.0,
    *,
    process: Any | None = None,
    log_path: Path | None = None,
    http_client=requests_module,
) -> None:
    url = f"http://{host}:{int(port)}/health"
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        _raise_if_process_exited(process, log_path)
        try:
            response = http_client.get(url, timeout=5)
            if response.status_code < 500:
                return
        except requests_module.RequestException as exc:
            last_error = exc
        _raise_if_process_exited(process, log_path)
        time.sleep(1)

    message = f"Timed out waiting for server readiness at {url} after {timeout_s}s"
    if last_error is not None:
        message = f"{message}; last error: {last_error}"
    log_tail = _read_log_tail(log_path)
    if log_tail:
        message = f"{message}\nLast server log lines:\n{log_tail}"
    raise TimeoutError(message)


def _raise_if_process_exited(process: Any | None, log_path: Path | None) -> None:
    if process is None:
        return
    exit_code = process.poll()
    if exit_code is None:
        return

    message = f"Online simulator server exited before readiness with exit code {exit_code}."
    log_tail = _read_log_tail(log_path)
    if log_tail:
        message = f"{message}\nLast server log lines:\n{log_tail}"
    raise RuntimeError(message)


def _read_log_tail(path: Path | None, max_lines: int = 80) -> str:
    if path is None or not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def run_online_client(
    host,
    port,
    requests: list[SyntheticRequest],
    total_request,
    app_workers: int = 1,
    request_rate: float = float("inf"),
    http_client=requests_module,
) -> list[dict]:
    url = build_generate_url(host, port)
    request_list = list(requests)
    if not request_list:
        return []

    start_time = time.monotonic()
    by_worker = split_requests_by_worker(request_list, app_workers)
    active_workers = [
        worker_requests
        for _, worker_requests in sorted(by_worker.items(), key=lambda item: item[0])
        if worker_requests
    ]
    max_workers = max(int(app_workers), 1)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _run_worker_requests,
                url,
                worker_requests,
                total_request,
                http_client,
                start_time,
                not isinf(float(request_rate)),
            )
            for worker_requests in active_workers
        ]

    results = []
    for future in futures:
        results.extend(future.result())
    return results


def _run_worker_requests(
    url: str,
    requests: list[SyntheticRequest],
    total_request: int,
    http_client,
    start_time: float,
    respect_arrival_times: bool,
) -> list[dict]:
    results = []
    for request in sorted(requests, key=lambda item: (item.worker_seq, item.created_time)):
        if respect_arrival_times:
            delay = start_time + float(request.created_time) - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        response = http_client.post(
            url,
            json=build_http_generate_payload(
                request,
                total_request=total_request,
            ),
            timeout=3600,
        )
        response.raise_for_status()
        results.append(response.json())
    return results


def trigger_profile(host, port, http_client=requests_module) -> None:
    response = http_client.post(build_profile_url(host, port), json={}, timeout=300)
    response.raise_for_status()
