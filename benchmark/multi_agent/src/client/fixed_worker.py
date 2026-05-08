from __future__ import annotations

import argparse
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


def _extract_usage_int(usage: dict[str, Any], key: str) -> int:
    val = usage.get(key, 0)
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


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

    def _token_ids(self, text: str) -> list[int]:
        with self._lock:
            return self.tokenizer.encode(text, add_special_tokens=False)

    def build(self, request_index: int) -> str:
        mixed_seed = (self.seed * 1000003 + int(request_index)) & 0xFFFFFFFF
        rng = random.Random(mixed_seed)
        words = [f"req{int(request_index):05d}"]

        # Grow text until tokenized length reaches target.
        text = " ".join(words)
        token_ids = self._token_ids(text)
        while len(token_ids) < self.target_input_len:
            words.append(f"tok{rng.randrange(0, 100000):05d}")
            text = " ".join(words)
            token_ids = self._token_ids(text)

        # Trim to exact token length and decode back to prompt text.
        if len(token_ids) > self.target_input_len:
            token_ids = token_ids[: self.target_input_len]
            with self._lock:
                text = self.tokenizer.decode(
                    token_ids,
                    clean_up_tokenization_spaces=False,
                    skip_special_tokens=False,
                )
            token_ids = self._token_ids(text)

        # Ensure exact size after decode/encode round-trip.
        while len(token_ids) < self.target_input_len:
            text = text + " x"
            token_ids = self._token_ids(text)
        if len(token_ids) > self.target_input_len:
            token_ids = token_ids[: self.target_input_len]
            with self._lock:
                text = self.tokenizer.decode(
                    token_ids,
                    clean_up_tokenization_spaces=False,
                    skip_special_tokens=False,
                )
            token_ids = self._token_ids(text)

        if len(token_ids) != self.target_input_len:
            raise RuntimeError(
                f"Failed to build exact-length prompt: got {len(token_ids)}, want {self.target_input_len}"
            )
        return text


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
    prompt_builder = PromptBuilder(args.model_path, args.input_len, args.seed)
    endpoint = args.server_url.rstrip("/")
    if not endpoint.endswith("/v1"):
        endpoint += "/v1"
    endpoint += "/completions"

    req_queue: queue.Queue[int] = queue.Queue()
    for req_idx in range(max(args.num_requests, 0)):
        req_queue.put(req_idx)

    job_records: list[dict[str, Any]] = []
    cache_rows: list[dict[str, Any]] = []
    results_lock = threading.Lock()

    def worker(worker_idx: int) -> None:
        while True:
            try:
                req_idx = req_queue.get_nowait()
            except queue.Empty:
                return

            job_id = str(req_idx + 1)
            worker_id = str(worker_idx)
            wall_start = time.time()
            error = ""
            status = "completed"
            usage: dict[str, Any] = {}
            response_body: dict[str, Any] = {}
            rid = ""
            # Keep request lengths fixed while varying prompt content per request.
            # Deterministic seed progression preserves reproducibility across runs.
            prompt = prompt_builder.build(req_idx)

            payload = {
                "model": args.model_path,
                "prompt": prompt,
                "temperature": 0.0,
                "max_tokens": int(args.output_len),
                "stream": False,
                "extra_body": {
                    "ignore_eos": bool(args.ignore_eos),
                    "return_cached_tokens_details": True,
                },
            }
            body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            req = urllib.request.Request(
                endpoint,
                data=body,
                method="POST",
                headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY_API_KEY"},
            )
            try:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(req, timeout=120) as resp:
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
            prompt_tokens = _extract_usage_int(usage, "prompt_tokens")
            completion_tokens = _extract_usage_int(usage, "completion_tokens")
            cached_prompt_tokens = _extract_usage_int(
                usage.get("prompt_tokens_details", {}) if isinstance(usage.get("prompt_tokens_details", {}), dict) else {},
                "cached_tokens",
            )
            total_tokens = prompt_tokens + completion_tokens
            hit_rate = round(cached_prompt_tokens / prompt_tokens, 4) if prompt_tokens > 0 else ""

            task_record = {
                "agent": "fixed_client",
                "task_label": "fixed_request",
                "duration_s": duration_s,
                "llm_calls": 1 if status == "completed" else 0,
                "prompt_tokens": prompt_tokens,
                "cached_prompt_tokens": cached_prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "prefill_cache_hit_rate": hit_rate,
                "rid": rid,
                "error": error,
            }
            record = {
                "topic_id": job_id,
                "topic": f"request_{job_id}",
                "year": "",
                "worker_id": worker_id,
                "status": status,
                "error": error,
                "agent_timings": [task_record],
                "summary": {
                    "wall_time_s": duration_s,
                    "llm_calls": task_record["llm_calls"],
                    "prompt_tokens": prompt_tokens,
                    "prefill_cached_tokens": cached_prompt_tokens,
                    "total_tokens": total_tokens,
                },
            }

            # Parse cache details from the full response first (includes sglext),
            # then fallback to usage-only fields.
            cache_details = _extract_cache_details(response_body)
            if not cache_details:
                cache_details = _extract_cache_details({"usage": usage})
            cache_rows.append(
                {
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
                    "is_failover_retried": "",
                    "pre_failover_output_tokens": "",
                    "pre_failover_backed_up_tokens": "",
                }
            )

            _post_control_event(
                args.control_url,
                "task_completed" if status == "completed" else "task_failed",
                {
                    "job_id": job_id,
                    "job": record["topic"],
                    "year": "",
                    "worker_id": worker_id,
                    "task_label": "fixed_request",
                    "agent_role": "fixed_client",
                    "duration_s": duration_s,
                    **({"error": error} if error else {}),
                },
            )
            _post_control_event(
                args.control_url,
                "job_completed" if status == "completed" else "job_failed",
                {
                    "job_id": job_id,
                    "job": record["topic"],
                    "year": "",
                    "worker_id": worker_id,
                    "status": status,
                    "wall_time_s": duration_s,
                    **({"error": error} if error else {}),
                },
            )
            with results_lock:
                job_records.append(record)

    worker_count = max(1, int(args.app_workers))
    threads: list[threading.Thread] = []
    for idx in range(worker_count):
        th = threading.Thread(target=worker, args=(idx + 1,), daemon=True)
        threads.append(th)
        th.start()

    for th in threads:
        th.join()

    job_records.sort(key=lambda item: int(item.get("topic_id", "0") or 0))
    write_cache_hit_records(cache_rows, output_dir / "cache_hits.csv")
    write_job_summary_records(job_records, output_dir / "job_summary.csv")
    write_task_summary_records(job_records, output_dir / "task_summary.csv")
    return 0 if all(item.get("status") == "completed" for item in job_records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
