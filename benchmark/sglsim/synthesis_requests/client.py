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
    turn_prompts: tuple[str, ...] | None = None
    model_path: str = ""

    @property
    def is_multi_turn(self) -> bool:
        return self.turn_prompts is not None and len(self.turn_prompts) > 1


def _turn_prompts_from_workload(req: WorkloadRequest) -> tuple[str, ...] | None:
    if isinstance(req.prompt, list) and len(req.prompt) > 1:
        return tuple(req.prompt)
    return None


def build_chat_payload(
    *,
    model_path: str,
    messages: list[dict[str, str]],
    output_len: int,
    rid: str,
) -> dict[str, Any]:
    return {
        "model": model_path,
        "messages": messages,
        "temperature": 0.0,
        "max_completion_tokens": int(output_len),
        "stream": False,
        "ignore_eos": True,
        "user": rid,
    }


def build_prepared_requests_from_workload(
    workload: list[WorkloadRequest],
    client: ClientConfig,
    server: ServerConfig,
) -> list[PreparedRequest]:
    if client.gsp_num_turns > 1 and client.gsp_fast_prepare:
        logger.warning(
            "gsp_num_turns=%d with gsp_fast_prepare; multi-turn uses chat messages "
            "instead of input_len",
            client.gsp_num_turns,
        )

    model_path = server.model_path
    prepared: list[PreparedRequest] = []
    for req in workload:
        turn_prompts = _turn_prompts_from_workload(req)
        if turn_prompts is not None:
            prepared.append(
                PreparedRequest(
                    rid=req.rid,
                    prompt=turn_prompts[0],
                    prompt_len=int(req.prompt_len),
                    output_len=req.output_len,
                    body=b"",
                    routing_key=req.routing_key,
                    turn_prompts=turn_prompts,
                    model_path=model_path,
                )
            )
            continue

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
                model_path=model_path,
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


def _post_json_sync(
    url: str,
    body: bytes,
    *,
    routing_key: str | None,
    timeout_s: float = 7200.0,
) -> tuple[dict[str, Any], float, str]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer EMPTY_API_KEY",
    }
    if routing_key:
        headers[ROUTING_KEY_HEADER] = routing_key
    start = time.perf_counter()
    try:
        req = urllib.request.Request(
            url,
            data=body,
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


def post_completions_sync(
    host: str,
    port: int,
    prepared: PreparedRequest,
    *,
    timeout_s: float = 7200.0,
) -> tuple[dict[str, Any], float, str]:
    url = f"{_base_url(host, port)}/v1/completions"
    return _post_json_sync(
        url,
        prepared.body,
        routing_key=prepared.routing_key,
        timeout_s=timeout_s,
    )


def post_chat_completions_sync(
    host: str,
    port: int,
    *,
    body: bytes,
    routing_key: str | None,
    timeout_s: float = 7200.0,
) -> tuple[dict[str, Any], float, str]:
    url = f"{_base_url(host, port)}/v1/chat/completions"
    return _post_json_sync(
        url,
        body,
        routing_key=routing_key,
        timeout_s=timeout_s,
    )


def _assistant_content(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    return content if isinstance(content, str) else ""


def run_multi_turn_session_sync(
    host: str,
    port: int,
    prepared: PreparedRequest,
    *,
    timeout_s: float = 7200.0,
) -> list[tuple[str, dict[str, Any], float, str]]:
    if prepared.turn_prompts is None:
        raise ValueError("run_multi_turn_session_sync requires turn_prompts")
    messages: list[dict[str, str]] = []
    outputs: list[tuple[str, dict[str, Any], float, str]] = []
    for turn_idx, turn_prompt in enumerate(prepared.turn_prompts):
        turn_rid = f"{prepared.rid}-t{turn_idx}"
        messages.append({"role": "user", "content": turn_prompt})
        payload = build_chat_payload(
            model_path=prepared.model_path,
            messages=messages,
            output_len=prepared.output_len,
            rid=turn_rid,
        )
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        response, e2e_latency_s, error = post_chat_completions_sync(
            host,
            port,
            body=body,
            routing_key=prepared.routing_key,
            timeout_s=timeout_s,
        )
        outputs.append((turn_rid, response, e2e_latency_s, error))
        if error:
            break
        messages.append(
            {"role": "assistant", "content": _assistant_content(response)}
        )
    return outputs


async def _bounded_post(
    host: str,
    port: int,
    prepared: PreparedRequest,
    semaphore: asyncio.Semaphore | None,
    progress: dict[str, int] | None = None,
) -> list[RequestDetailRow]:
    async def _run() -> list[RequestDetailRow]:
        if prepared.is_multi_turn:
            turn_outputs = await asyncio.to_thread(
                run_multi_turn_session_sync, host, port, prepared
            )
            rows = []
            for turn_rid, response, e2e_latency_s, error in turn_outputs:
                rows.append(
                    response_to_detail_row(
                        turn_rid,
                        response,
                        e2e_latency_s,
                        prepared_prompt_len=prepared.prompt_len,
                        prepared_output_len=prepared.output_len,
                        error=error,
                    )
                )
            return rows
        response, e2e_latency_s, error = await asyncio.to_thread(
            post_completions_sync, host, port, prepared
        )
        return [
            response_to_detail_row(
                prepared.rid,
                response,
                e2e_latency_s,
                prepared_prompt_len=prepared.prompt_len,
                prepared_output_len=prepared.output_len,
                error=error,
            )
        ]

    if semaphore is not None:
        async with semaphore:
            rows = await _run()
    else:
        rows = await _run()

    if progress is not None:
        progress["done"] += 1
        logger.info(
            "Completed %d/%d sessions (rid=%s turns=%d)",
            progress["done"],
            progress["total"],
            prepared.rid,
            len(rows),
        )
    return rows


def _rid_sort_key(rid: str) -> tuple[Any, ...]:
    if "-t" in rid:
        base, turn = rid.rsplit("-t", 1)
        turn_key: Any = int(turn) if turn.isdigit() else turn
        base_key: Any = int(base) if base.isdigit() else base
        return (base_key, turn_key)
    return (int(rid) if rid.isdigit() else rid, 0)


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
    progress = {"done": 0, "total": len(prepared_requests)}
    tasks: list[asyncio.Task[list[RequestDetailRow]]] = []
    for prepared in prepared_requests:
        tasks.append(
            asyncio.create_task(
                _bounded_post(host, port, prepared, semaphore, progress),
                name=f"gsp-rid-{prepared.rid}",
            )
        )
    if not tasks:
        return []
    nested = await asyncio.gather(*tasks)
    rows = [row for session_rows in nested for row in session_rows]
    rows.sort(key=lambda row: _rid_sort_key(row.rid))
    return list(rows)


def run_client(
    host: str,
    port: int,
    prepared_requests: list[PreparedRequest],
    client: ClientConfig,
) -> list[RequestDetailRow]:
    multi_turn = sum(1 for p in prepared_requests if p.is_multi_turn)
    logger.info(
        "Running bench_serving-style client sessions=%d multi_turn_sessions=%d "
        "request_rate=inf max_concurrency=%s",
        len(prepared_requests),
        multi_turn,
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
    logger.info("Completed %d detail rows (%d sessions)", len(rows), len(prepared_requests))
    return rows
