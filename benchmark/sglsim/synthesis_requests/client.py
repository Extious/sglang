from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .cache_store import (
    load_prepared,
    load_workload,
    save_prepared,
    save_workload,
    warn_legacy_cache,
)
from .utils import (
    ROUTING_KEY_HEADER,
    ClientConfig,
    RequestDetailRow,
    ServerConfig,
    WorkloadRequest,
    generate_shared_prefix_workload,
    read_log_tail,
    response_to_detail_row,
)

logger = logging.getLogger("sglang.benchmark.sglsim.synthesis_requests.client")


@dataclass(frozen=True)
class PreparedRequest:
    rid: str
    prompt: str
    prompt_len: int
    output_len: int
    body: bytes
    routing_key: str | None = None


def build_prepared_requests_from_workload(
    workload: list[WorkloadRequest],
    client: ClientConfig,
    server: ServerConfig,
) -> list[PreparedRequest]:
    if client.gsp_num_turns > 1:
        logger.warning(
            "gsp_num_turns=%d; HTTP client sends the first turn only",
            client.gsp_num_turns,
        )

    model_path = server.model_path
    prepared: list[PreparedRequest] = []
    for req in workload:
        turn_prompt = req.first_turn_prompt
        payload: dict[str, Any] = {
            "model": model_path,
            "temperature": 0.0,
            "max_tokens": int(req.output_len),
            "stream": False,
            "ignore_eos": True,
            "user": req.rid,
        }
        if client.gsp_fast_prepare:
            payload["input_len"] = int(req.prompt_len)
            actual_prompt_len = int(req.prompt_len)
        else:
            payload["prompt"] = turn_prompt
            actual_prompt_len = int(req.prompt_len)
        prepared.append(
            PreparedRequest(
                rid=req.rid,
                prompt=turn_prompt,
                prompt_len=actual_prompt_len,
                output_len=req.output_len,
                body=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
                routing_key=req.routing_key,
            )
        )
    return prepared


def build_prepared_requests(
    client: ClientConfig,
    server: ServerConfig,
) -> list[PreparedRequest]:
    warn_legacy_cache(logger)

    cached_prepared = load_prepared(client, server, logger=logger)
    if cached_prepared is not None:
        return cached_prepared

    workload = load_workload(client, server, logger=logger)
    generated_workload = workload is None
    if workload is None:
        workload = generate_shared_prefix_workload(client, server, logger=logger)
        save_workload(client, server, workload, logger=logger)

    prepared = build_prepared_requests_from_workload(workload, client, server)
    save_prepared(client, server, prepared, logger=logger)
    if generated_workload:
        logger.info(
            "Built %d prepared requests from generated workload", len(prepared)
        )
    else:
        logger.info(
            "Built %d prepared requests from cached workload", len(prepared)
        )
    return prepared


def _base_url(host: str, port: int) -> str:
    return f"http://{host}:{int(port)}"


def _raise_if_process_exited(process: Any | None, log_path) -> None:
    if process is None:
        return
    exit_code = process.poll()
    if exit_code is None:
        return
    message = f"SGLang server exited before readiness with exit code {exit_code}."
    log_tail = read_log_tail(log_path)
    if log_tail:
        message = f"{message}\nLast server log lines:\n{log_tail}"
    raise RuntimeError(message)


def wait_for_server_ready(
    host: str,
    port: int,
    timeout_s: float = 1800.0,
    *,
    process: Any | None = None,
    log_path=None,
) -> None:
    url = f"{_base_url(host, port)}/health"
    logger.info("Waiting for server readiness at %s", url)
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        _raise_if_process_exited(process, log_path)
        try:
            req = urllib.request.Request(url, method="GET")
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(req, timeout=5) as resp:
                if resp.status < 500:
                    logger.info("Server is ready at %s", url)
                    return
        except (urllib.error.URLError, OSError) as exc:
            last_error = exc
        _raise_if_process_exited(process, log_path)
        time.sleep(1)
    message = f"Timed out waiting for server readiness at {url} after {timeout_s}s"
    if last_error is not None:
        message = f"{message}; last error: {last_error}"
    log_tail = read_log_tail(log_path)
    if log_tail:
        message = f"{message}\nLast server log lines:\n{log_tail}"
    raise TimeoutError(message)


def post_completions_sync(
    host: str,
    port: int,
    prepared: PreparedRequest,
    *,
    timeout_s: float = 7200.0,
) -> tuple[dict[str, Any], float, str]:
    url = f"{_base_url(host, port)}/v1/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer EMPTY_API_KEY",
    }
    if prepared.routing_key:
        headers[ROUTING_KEY_HEADER] = prepared.routing_key
    start = time.perf_counter()
    try:
        req = urllib.request.Request(
            url,
            data=prepared.body,
            method="POST",
            headers=headers,
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=timeout_s) as resp:
            if resp.status >= 400:
                raise urllib.error.HTTPError(
                    url, resp.status, resp.reason, resp.headers, resp.read()
                )
            raw = json.loads(resp.read().decode("utf-8"))
        e2e_latency_s = time.perf_counter() - start
        return (raw if isinstance(raw, dict) else {}), e2e_latency_s, ""
    except Exception as exc:
        e2e_latency_s = time.perf_counter() - start
        return {}, e2e_latency_s, str(exc)[:500]


async def _bounded_post(
    host: str,
    port: int,
    prepared: PreparedRequest,
    semaphore: asyncio.Semaphore | None,
) -> RequestDetailRow:
    if semaphore is not None:
        async with semaphore:
            response, e2e_latency_s, error = await asyncio.to_thread(
                post_completions_sync, host, port, prepared
            )
    else:
        response, e2e_latency_s, error = await asyncio.to_thread(
            post_completions_sync, host, port, prepared
        )
    return response_to_detail_row(
        prepared.rid,
        response,
        e2e_latency_s,
        prepared_prompt_len=prepared.prompt_len,
        prepared_output_len=prepared.output_len,
        error=error,
    )


async def _run_client_async(
    host: str,
    port: int,
    prepared_requests: list[PreparedRequest],
    *,
    max_concurrency: int | None,
) -> list[RequestDetailRow]:
    semaphore = (
        asyncio.Semaphore(max_concurrency) if max_concurrency is not None else None
    )
    tasks: list[asyncio.Task[RequestDetailRow]] = []
    for prepared in prepared_requests:
        tasks.append(
            asyncio.create_task(
                _bounded_post(host, port, prepared, semaphore),
                name=f"gsp-rid-{prepared.rid}",
            )
        )
    if not tasks:
        return []
    rows = await asyncio.gather(*tasks)
    rows.sort(key=lambda row: int(row.rid) if row.rid.isdigit() else row.rid)
    return list(rows)


def run_client(
    host: str,
    port: int,
    prepared_requests: list[PreparedRequest],
    client: ClientConfig,
) -> list[RequestDetailRow]:
    logger.info(
        "Running bench_serving-style client requests=%d request_rate=inf "
        "max_concurrency=%s",
        len(prepared_requests),
        client.max_concurrency,
    )
    rows = asyncio.run(
        _run_client_async(
            host,
            port,
            prepared_requests,
            max_concurrency=client.max_concurrency,
        )
    )
    logger.info("Completed %d requests", len(rows))
    return rows
