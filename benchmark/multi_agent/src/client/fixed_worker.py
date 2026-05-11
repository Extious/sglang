from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import queue
import random
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from client.metrics_utils import (
    write_cache_hit_records,
    write_job_summary_records,
    write_task_summary_records,
)


@dataclass(frozen=True)
class PreparedRequest:
    req_idx: int
    job_id: str
    prompt: str
    body: bytes


@dataclass(frozen=True)
class CompletedRequest:
    job_id: str
    worker_id: str
    status: str
    error: str
    duration_s: float
    rid: str
    usage: dict[str, Any]
    response_body: dict[str, Any]


def _log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _extract_cache_details(body: dict[str, Any]) -> dict[str, Any]:
    usage = body.get("usage", {}) if isinstance(body, dict) else {}
    if not isinstance(usage, dict):
        usage = {}
    details = usage.get("cached_tokens_details", {})
    if isinstance(details, dict) and details:
        return details
    sglext = body.get("sglext", {}) if isinstance(body, dict) else {}
    if isinstance(sglext, dict):
        details = sglext.get("cached_tokens_details", {})
        if isinstance(details, dict):
            return details
    return {}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _extract_failover_details(body: dict[str, Any]) -> dict[str, Any]:
    usage = body.get("usage", {}) if isinstance(body, dict) else {}
    if not isinstance(usage, dict):
        usage = {}
    details = usage.get("failover", {})
    if isinstance(details, dict) and details:
        return details
    sglext = body.get("sglext", {}) if isinstance(body, dict) else {}
    if isinstance(sglext, dict):
        details = sglext.get("failover", {})
        if isinstance(details, dict):
            return details
    return {}


def _extract_usage_int(usage: dict[str, Any], key: str) -> int:
    val = usage.get(key, 0)
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def _build_completion_payload(
    *,
    model_path: str,
    prompt: str,
    output_len: int,
    ignore_eos: bool,
) -> dict[str, Any]:
    return {
        "model": model_path,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": int(output_len),
        "stream": False,
        "ignore_eos": bool(ignore_eos),
        "return_cached_tokens_details": True,
    }


def _cache_row_from_response(
    *,
    job_id: str,
    rid: str,
    response_body: dict[str, Any],
    usage: dict[str, Any],
) -> dict[str, Any]:
    # Parse cache details from the full response first (includes sglext),
    # then fallback to usage-only fields.
    cache_details = _extract_cache_details(response_body)
    if not cache_details:
        cache_details = _extract_cache_details({"usage": usage})

    failover = _extract_failover_details(response_body)
    if not failover:
        failover = _extract_failover_details({"usage": usage})

    pre_output = _extract_usage_int(failover, "pre_failover_output_tokens")
    pre_backed_up = _extract_usage_int(failover, "pre_failover_backed_up_tokens")
    is_retried = (
        _truthy(failover.get("is_failover_retried"))
        or _truthy(failover.get("is_retried"))
        or pre_output > 0
        or pre_backed_up > 0
    )

    return {
        "job_id": job_id,
        "task_label": "fixed_request",
        "agent_role": "fixed_client",
        "rid": rid,
        "l1_match": int(cache_details.get("device", 0) or 0),
        "l2_match": int(cache_details.get("host", 0) or 0),
        "remote_match": int(cache_details.get("storage_query", 0) or 0),
        "remote_prefetch": int(cache_details.get("storage", 0) or 0),
        "reused_device": int(cache_details.get("reused_device", 0) or 0),
        "reused_host": int(cache_details.get("reused_host", 0) or 0),
        "reused_storage": int(cache_details.get("reused_storage", 0) or 0),
        "is_failover_retried": "True" if is_retried else "",
        "pre_failover_output_tokens": str(pre_output) if pre_output > 0 else "",
        "pre_failover_backed_up_tokens": (
            str(pre_backed_up) if pre_backed_up > 0 else ""
        ),
    }


def _prepare_request(
    *,
    req_idx: int,
    prompt_builder: "PromptBuilder",
    model_path: str,
    output_len: int,
    ignore_eos: bool,
) -> PreparedRequest:
    # Build prompt/payload ahead of time so request construction can overlap
    # with server-side inference from earlier requests.
    prompt = prompt_builder.build(req_idx)
    payload = _build_completion_payload(
        model_path=model_path,
        prompt=prompt,
        output_len=output_len,
        ignore_eos=ignore_eos,
    )
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    return PreparedRequest(
        req_idx=req_idx,
        job_id=str(req_idx + 1),
        prompt=prompt,
        body=body,
    )


def _prepare_requests(
    *,
    pending_queue: "queue.Queue[int]",
    prepared_requests: list[PreparedRequest | None],
    prompt_builder: "PromptBuilder",
    model_path: str,
    output_len: int,
    ignore_eos: bool,
    progress: "ProgressTracker | None" = None,
) -> None:
    while True:
        try:
            req_idx = pending_queue.get_nowait()
        except queue.Empty:
            return
        prepared_requests[req_idx] = _prepare_request(
            req_idx=req_idx,
            prompt_builder=prompt_builder,
            model_path=model_path,
            output_len=output_len,
            ignore_eos=ignore_eos,
        )
        if progress is not None:
            progress.note_prepared(req_idx)


def _fill_ready_queue(
    *,
    ready_queue: "queue.Queue[PreparedRequest | None]",
    prepared_requests: list[PreparedRequest | None],
) -> None:
    for req_idx, prepared in enumerate(prepared_requests):
        if prepared is None:
            raise RuntimeError(f"request {req_idx} was not prepared")
        ready_queue.put(prepared)


class ProgressTracker:
    def __init__(self, total_requests: int) -> None:
        self.total_requests = max(0, int(total_requests))
        self.prepared = 0
        self.dispatched = 0
        self.completed = 0
        self.failed = 0
        self._lock = threading.Lock()

    def note_prepared(self, req_idx: int) -> None:
        with self._lock:
            self.prepared += 1
            prepared = self.prepared
            total = self.total_requests
        _log(f"prepare progress: built request {req_idx + 1}/{total} ({prepared}/{total} ready)")

    def note_prepared_done(self) -> None:
        _log(f"prepare progress: all {self.total_requests} requests built and queued")

    def note_dispatched(self, *, job_id: str, worker_id: str) -> None:
        with self._lock:
            self.dispatched += 1
            dispatched = self.dispatched
            total = self.total_requests
        _log(
            f"dispatch progress: worker={worker_id} sent request {job_id} "
            f"({dispatched}/{total} dispatched)"
        )

    def note_completed(
        self, *, job_id: str, worker_id: str, status: str, duration_s: float, error: str
    ) -> None:
        with self._lock:
            self.completed += 1
            if status != "completed":
                self.failed += 1
            completed = self.completed
            failed = self.failed
            total = self.total_requests
        summary = (
            f"complete progress: worker={worker_id} finished request {job_id} "
            f"status={status} duration={duration_s:.3f}s ({completed}/{total} done"
        )
        if failed > 0:
            summary += f", {failed} failed"
        summary += ")"
        if error:
            summary += f" error={error}"
        _log(summary)


class PromptBuilder:
    """Build deterministic prompts with exact tokenizer token length."""

    def __init__(self, model_path: str, target_input_len: int, seed: int) -> None:
        from transformers import AutoTokenizer

        self.target_input_len = max(int(target_input_len), 1)
        self.seed = int(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
        )
        self._lock = threading.Lock()
        self._stable_filler_token_ids = self._build_stable_filler_token_ids()

    def _token_ids(self, text: str) -> list[int]:
        with self._lock:
            return self.tokenizer.encode(text, add_special_tokens=False)

    def _decode_token_ids(self, token_ids: list[int]) -> str:
        with self._lock:
            return self.tokenizer.decode(
                token_ids,
                clean_up_tokenization_spaces=False,
                skip_special_tokens=False,
            )

    def _build_stable_filler_token_ids(self) -> tuple[int, ...]:
        special_ids = set(getattr(self.tokenizer, "all_special_ids", []))
        filler_token_ids: list[int] = []
        vocab_size = len(self.tokenizer)
        target_pool_size = 256

        for token_id in range(vocab_size):
            if token_id in special_ids:
                continue
            text = self._decode_token_ids([token_id])
            if not text or not text.startswith(" ") or not text.isascii():
                continue
            if any(ch in text for ch in "\r\n\t"):
                continue
            roundtrip_ids = self._token_ids(text)
            if roundtrip_ids == [token_id]:
                filler_token_ids.append(token_id)
                if len(filler_token_ids) >= target_pool_size:
                    break

        if not filler_token_ids:
            raise RuntimeError("Failed to find stable filler tokens for fixed prompt generation")

        return tuple(filler_token_ids)

    def _normalize_token_ids(
        self, token_ids: list[int], rng: random.Random, request_index: int
    ) -> str:
        current_ids = list(token_ids)

        for _ in range(3):
            text = self._decode_token_ids(current_ids)
            roundtrip_ids = self._token_ids(text)
            if len(roundtrip_ids) == self.target_input_len:
                return text

            if len(roundtrip_ids) > self.target_input_len:
                current_ids = roundtrip_ids[: self.target_input_len]
                continue

            missing = self.target_input_len - len(roundtrip_ids)
            current_ids = roundtrip_ids + [
                self._stable_filler_token_ids[
                    rng.randrange(len(self._stable_filler_token_ids))
                ]
                for _ in range(missing)
            ]

        raise RuntimeError(
            "Failed to build exact-length prompt after normalization: "
            f"req={request_index}, got {len(roundtrip_ids)}, want {self.target_input_len}"
        )

    def build(self, request_index: int) -> str:
        mixed_seed = (self.seed * 1000003 + int(request_index)) & 0xFFFFFFFF
        rng = random.Random(mixed_seed)
        prefix = f"req{int(request_index):05d}"
        prefix_ids = self._token_ids(prefix)

        if len(prefix_ids) >= self.target_input_len:
            return self._normalize_token_ids(
                prefix_ids[: self.target_input_len], rng, request_index
            )

        filler_count = self.target_input_len - len(prefix_ids)
        prompt_token_ids = prefix_ids + [
            self._stable_filler_token_ids[
                rng.randrange(len(self._stable_filler_token_ids))
            ]
            for _ in range(filler_count)
        ]

        return self._normalize_token_ids(prompt_token_ids, rng, request_index)


def _post_control_event(control_url: str, event_type: str, payload: dict[str, Any]) -> None:
    if not control_url:
        return
    body = json.dumps(
        {"event": event_type, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"), **payload},
        ensure_ascii=True,
    ).encode("utf-8")
    req = urllib.request.Request(
        control_url.rstrip("/") + "/event",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=10):
            pass
    except (urllib.error.URLError, OSError):
        return


def _build_completed_request(
    *,
    job_id: str,
    worker_id: str,
    status: str,
    error: str,
    duration_s: float,
    rid: str,
    usage: dict[str, Any],
    response_body: dict[str, Any],
) -> CompletedRequest:
    return CompletedRequest(
        job_id=job_id,
        worker_id=worker_id,
        status=status,
        error=error,
        duration_s=duration_s,
        rid=rid,
        usage=usage,
        response_body=response_body,
    )


def _build_job_record(result: CompletedRequest) -> dict[str, Any]:
    prompt_tokens = _extract_usage_int(result.usage, "prompt_tokens")
    completion_tokens = _extract_usage_int(result.usage, "completion_tokens")
    prompt_details = result.usage.get("prompt_tokens_details", {})
    if not isinstance(prompt_details, dict):
        prompt_details = {}
    cached_prompt_tokens = _extract_usage_int(prompt_details, "cached_tokens")
    total_tokens = prompt_tokens + completion_tokens
    hit_rate = round(cached_prompt_tokens / prompt_tokens, 4) if prompt_tokens > 0 else ""
    task_record = {
        "agent": "fixed_client",
        "task_label": "fixed_request",
        "duration_s": result.duration_s,
        "llm_calls": 1 if result.status == "completed" else 0,
        "prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": cached_prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "prefill_cache_hit_rate": hit_rate,
        "rid": result.rid,
        "error": result.error,
    }
    return {
        "topic_id": result.job_id,
        "topic": f"request_{result.job_id}",
        "year": "",
        "worker_id": result.worker_id,
        "status": result.status,
        "error": result.error,
        "agent_timings": [task_record],
        "summary": {
            "wall_time_s": result.duration_s,
            "llm_calls": task_record["llm_calls"],
            "prompt_tokens": prompt_tokens,
            "prefill_cached_tokens": cached_prompt_tokens,
            "total_tokens": total_tokens,
        },
    }


def _drain_completed_requests(
    *,
    completed_queue: "queue.Queue[CompletedRequest | None]",
    job_records: list[dict[str, Any]],
    cache_rows: list[dict[str, Any]],
    control_url: str,
) -> None:
    while True:
        completed = completed_queue.get()
        if completed is None:
            return

        record = _build_job_record(completed)
        cache_rows.append(
            _cache_row_from_response(
                job_id=completed.job_id,
                rid=completed.rid,
                response_body=completed.response_body,
                usage=completed.usage,
            )
        )
        _post_control_event(
            control_url,
            "task_completed" if completed.status == "completed" else "task_failed",
            {
                "job_id": completed.job_id,
                "job": record["topic"],
                "year": "",
                "worker_id": completed.worker_id,
                "task_label": "fixed_request",
                "agent_role": "fixed_client",
                "duration_s": completed.duration_s,
                **({"error": completed.error} if completed.error else {}),
            },
        )
        _post_control_event(
            control_url,
            "job_completed" if completed.status == "completed" else "job_failed",
            {
                "job_id": completed.job_id,
                "job": record["topic"],
                "year": "",
                "worker_id": completed.worker_id,
                "status": completed.status,
                "wall_time_s": completed.duration_s,
                **({"error": completed.error} if completed.error else {}),
            },
        )
        job_records.append(record)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fixed-length workload worker process")
    parser.add_argument("--server-url", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--app-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ignore-eos", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--control-url", type=str, default="")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    endpoint = args.server_url.rstrip("/")
    if not endpoint.endswith("/v1"):
        endpoint += "/v1"
    endpoint += "/completions"

    num_requests = max(args.num_requests, 0)
    pending_queue: queue.Queue[int] = queue.Queue()
    for req_idx in range(num_requests):
        pending_queue.put(req_idx)

    job_records: list[dict[str, Any]] = []
    cache_rows: list[dict[str, Any]] = []
    worker_count = max(1, int(args.app_workers))
    ready_queue: queue.Queue[PreparedRequest | None] = queue.Queue()
    completed_queue: queue.Queue[CompletedRequest | None] = queue.Queue()
    producer_errors: list[Exception] = []
    producer_count = min(worker_count, num_requests)
    prepared_requests: list[PreparedRequest | None] = [None] * num_requests
    producer_error_lock = threading.Lock()
    prompt_builder = PromptBuilder(args.model_path, args.input_len, args.seed)
    progress = ProgressTracker(num_requests)
    start_gate = threading.Event()

    def producer() -> None:
        try:
            _prepare_requests(
                pending_queue=pending_queue,
                prepared_requests=prepared_requests,
                prompt_builder=prompt_builder,
                model_path=args.model_path,
                output_len=args.output_len,
                ignore_eos=bool(args.ignore_eos),
                progress=progress,
            )
        except Exception as exc:
            with producer_error_lock:
                producer_errors.append(exc)

    def worker(worker_idx: int) -> None:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        start_gate.wait()
        while True:
            prepared = ready_queue.get()
            if prepared is None:
                return

            job_id = prepared.job_id
            worker_id = str(worker_idx)
            progress.note_dispatched(job_id=job_id, worker_id=worker_id)
            wall_start = time.time()
            error = ""
            status = "completed"
            usage: dict[str, Any] = {}
            response_body: dict[str, Any] = {}
            rid = ""
            req = urllib.request.Request(
                endpoint,
                data=prepared.body,
                method="POST",
                headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY_API_KEY"},
            )
            try:
                with opener.open(req, timeout=240) as resp:
                    raw_body = json.loads(resp.read().decode("utf-8"))
                response_body = raw_body if isinstance(raw_body, dict) else {}
                if not isinstance(response_body, dict):
                    raise ValueError("invalid response body")
                usage = response_body.get("usage", {})
                if not isinstance(usage, dict):
                    usage = {}
                rid = str(response_body.get("id", "") or "")
            except Exception as exc:
                status = "failed"
                error = str(exc)[:200]

            wall_end = time.time()
            duration_s = round(wall_end - wall_start, 3)
            progress.note_completed(
                job_id=job_id,
                worker_id=worker_id,
                status=status,
                duration_s=duration_s,
                error=error,
            )
            completed_queue.put(
                _build_completed_request(
                    job_id=job_id,
                    worker_id=worker_id,
                    status=status,
                    error=error,
                    duration_s=duration_s,
                    rid=rid,
                    usage=usage,
                    response_body=response_body,
                )
            )

    producer_threads: list[threading.Thread] = []
    _log(
        f"fixed workload start: requests={num_requests}, app_workers={worker_count}, "
        f"input_len={args.input_len}, output_len={args.output_len}"
    )
    for _ in range(producer_count):
        th = threading.Thread(target=producer, daemon=True)
        producer_threads.append(th)
        th.start()

    for th in producer_threads:
        th.join()

    if producer_errors:
        raise RuntimeError(f"Failed to prepare fixed workload requests: {producer_errors[0]}")

    _fill_ready_queue(ready_queue=ready_queue, prepared_requests=prepared_requests)
    progress.note_prepared_done()
    for _ in range(worker_count):
        ready_queue.put(None)

    recorder_thread = threading.Thread(
        target=_drain_completed_requests,
        kwargs={
            "completed_queue": completed_queue,
            "job_records": job_records,
            "cache_rows": cache_rows,
            "control_url": args.control_url,
        },
        daemon=True,
    )
    recorder_thread.start()

    threads: list[threading.Thread] = []
    for idx in range(worker_count):
        th = threading.Thread(target=worker, args=(idx + 1,), daemon=True)
        threads.append(th)
        th.start()
    _log(f"dispatch progress: releasing {worker_count} app_workers")
    start_gate.set()

    for th in threads:
        th.join()
    completed_queue.put(None)
    recorder_thread.join()

    job_records.sort(key=lambda item: int(item.get("topic_id", "0") or 0))
    write_cache_hit_records(cache_rows, output_dir / "cache_hits.csv")
    write_job_summary_records(job_records, output_dir / "job_summary.csv")
    write_task_summary_records(job_records, output_dir / "task_summary.csv")
    return 0 if all(item.get("status") == "completed" for item in job_records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
