import argparse
import csv
import fcntl
import json
import multiprocessing as mp
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "true"

from collections import defaultdict
import litellm
from crewai import Agent, Crew, Task, Process as CrewProcess, LLM
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.agent_events import (
    AgentExecutionStartedEvent, AgentExecutionCompletedEvent, AgentExecutionErrorEvent,
)
from crewai.events.types.llm_events import LLMCallCompletedEvent, LLMCallType
from crewai.events.types.task_events import TaskCompletedEvent, TaskFailedEvent, TaskStartedEvent
from crewai.llms.base_llm import get_current_call_id

ROLE_TO_TASK_LABEL = {}
TOPIC_CSV_FILE = Path(__file__).with_name("topics.csv")
TRACE_LOG_FILE = Path(__file__).with_name("trace_log.json")
EVENTS_FILE_ENV = os.environ.get("CREWAI_EVENTS_FILE", "").strip()
EVENTS_FILE = Path(EVENTS_FILE_ENV).expanduser() if EVENTS_FILE_ENV else None
TOPIC_LIMIT = 10
WORKER_COUNT = 2
WORKER_START_DELAY_SECS = float(
    os.environ.get("CREWAI_WORKER_START_DELAY_SECS", "10").strip() or "10"
)
DEFAULT_YEAR = "2025"
SERVER_BASE_URL = os.environ.get("CREWAI_SERVER_BASE_URL", "http://gpu10:28000").rstrip("/")
MODEL_ID = os.environ.get("CREWAI_MODEL_PATH", "Qwen/Qwen3-8B")
ENABLE_STREAM = os.environ.get("CREWAI_ENABLE_STREAM", "0").strip().lower() in {
    "1", "true", "yes",
}
PRINT_COMPLETED_ONLY = os.environ.get("CREWAI_PRINT_COMPLETED_ONLY", "1").strip().lower() not in {
    "0", "false", "no",
}


def _env_flag(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    return int(raw)


def _env_json_dict(name):
    raw = os.environ.get(name)
    if raw is None:
        return {}
    raw = raw.strip()
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object")
    return parsed


def _append_no_proxy_host(host):
    if not host:
        return

    for key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(key, "")
        items = [item.strip() for item in current.split(",") if item.strip()]
        if host not in items:
            items.append(host)
            os.environ[key] = ",".join(items)


_append_no_proxy_host(urlparse(SERVER_BASE_URL).hostname)

# ── Live progress & prefill cache stats ──
_progress_lock = threading.Lock()
_progress_state = {
    "total_tasks": 0,
    "completed_tasks": 0,
    "failed_tasks": 0,
    "running_tasks": set(),
    "wall_start": None,
    "wall_end": None,
}
_prefill_stats = {
    "llm_calls": 0,
    "prompt_tokens": 0,
    "cached_tokens": 0,
    "backup_cache_tokens": 0,
}
_run_context_lock = threading.Lock()
_run_context = {
    "job_id": None,
    "job": None,
    "year": None,
    "worker_id": None,
}
_request_context_lock = threading.Lock()
_request_context = {
    "task_label": None,
    "agent_role": None,
}


def _format_elapsed(seconds):
    total_seconds = max(int(seconds), 0)
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_hit_rate(cached_tokens, prompt_tokens):
    return f"{cached_tokens / prompt_tokens:.1%}" if prompt_tokens else "N/A"


def _set_run_context(topic_id=None, topic=None, year=None, worker_id=None):
    with _run_context_lock:
        _run_context["job_id"] = topic_id
        _run_context["job"] = topic
        _run_context["year"] = year
        _run_context["worker_id"] = worker_id


def _get_run_context():
    with _run_context_lock:
        return dict(_run_context)


def _set_request_context(task_label=None, agent_role=None):
    with _request_context_lock:
        _request_context["task_label"] = task_label
        _request_context["agent_role"] = agent_role


def _get_request_context():
    with _request_context_lock:
        return dict(_request_context)


def _build_request_context_payload():
    run_context = _get_run_context()
    request_context = _get_request_context()
    payload = {
        "job_id": run_context.get("job_id"),
        "job": run_context.get("job"),
        "year": run_context.get("year"),
        "task_label": request_context.get("task_label"),
        "agent_role": request_context.get("agent_role"),
        "worker_id": run_context.get("worker_id"),
    }
    return {key: value for key, value in payload.items() if value is not None}


def _build_request_user_tag():
    payload = _build_request_context_payload()
    if not payload:
        return ""
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def _append_event_record(event_type, **payload):
    if EVENTS_FILE is None:
        return

    EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "event": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with open(EVENTS_FILE, "a", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        fh.write(json.dumps(record, ensure_ascii=True) + "\n")
        fh.flush()
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _task_label(task):
    agent = getattr(task, "agent", None)
    role = getattr(agent, "role", None)
    if role and role in ROLE_TO_TASK_LABEL:
        return ROLE_TO_TASK_LABEL[role]

    raw = getattr(task, "name", None) or getattr(task, "description", "Unknown task")
    first_line = raw.strip().splitlines()[0].strip()
    return first_line[:60] + ("..." if len(first_line) > 60 else "")


def _snapshot_progress():
    with _progress_lock:
        total = _progress_state["total_tasks"]
        completed = _progress_state["completed_tasks"]
        failed = _progress_state["failed_tasks"]
        running = sorted(_progress_state["running_tasks"])
        wall_start = _progress_state["wall_start"]
        wall_end = _progress_state["wall_end"]
        prefill_stats = dict(_prefill_stats)

    now = wall_end if wall_end is not None else time.monotonic()
    elapsed = now - wall_start if wall_start is not None else 0.0
    return {
        "total": total,
        "completed": completed,
        "failed": failed,
        "done": completed + failed,
        "running": running,
        "elapsed": elapsed,
        "prefill": prefill_stats,
    }


def _print_progress(kind, detail):
    if PRINT_COMPLETED_ONLY:
        return

    snapshot = _snapshot_progress()
    total = snapshot["total"]
    done = snapshot["done"]
    percent = (done / total * 100) if total else 0.0
    prefill = snapshot["prefill"]
    hit_rate = _format_hit_rate(prefill["cached_tokens"], prefill["prompt_tokens"])
    print(
        f"[+{_format_elapsed(snapshot['elapsed'])}] {kind:<10} "
        f"{done:>2}/{total:<2} ({percent:5.1f}%) | "
        f"running={len(snapshot['running'])} | prefill={hit_rate} | {detail}",
        flush=True,
    )

def _start_progress_monitor(total_tasks):
    with _progress_lock:
        _progress_state["total_tasks"] = total_tasks
        _progress_state["completed_tasks"] = 0
        _progress_state["failed_tasks"] = 0
        _progress_state["running_tasks"] = set()
        _progress_state["wall_start"] = time.monotonic()
        _progress_state["wall_end"] = None
        _prefill_stats["llm_calls"] = 0
        _prefill_stats["prompt_tokens"] = 0
        _prefill_stats["cached_tokens"] = 0
        _prefill_stats["backup_cache_tokens"] = 0

def _stop_progress_monitor():
    crewai_event_bus.flush(timeout=10.0)
    with _progress_lock:
        if _progress_state["wall_start"] is not None and _progress_state["wall_end"] is None:
            _progress_state["wall_end"] = time.monotonic()

# ── Monkey-patch litellm completion calls to capture raw usage ──
_pending_usage = {}
_usage_lock = threading.Lock()
_original_completion = litellm.completion
_original_acompletion = litellm.acompletion

def _usage_get(obj, key, default=None):
    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(key, default)

    return getattr(obj, key, default)


def _extract_cached_tokens_details_from_obj(obj):
    if obj is None:
        return None

    details = _usage_get(_usage_get(obj, "sglext"), "cached_tokens_details")
    if details:
        return details

    provider_fields = _usage_get(obj, "provider_specific_fields")
    details = _usage_get(provider_fields, "cached_tokens_details")
    if details:
        return details

    hidden_params = _usage_get(obj, "_hidden_params")
    provider_fields = _usage_get(hidden_params, "provider_specific_fields")
    details = _usage_get(provider_fields, "cached_tokens_details")
    if details:
        return details

    return None


def _extract_cached_tokens_details(payload):
    if payload is None:
        return None

    details = _extract_cached_tokens_details_from_obj(payload)
    if details:
        return details

    choices = _usage_get(payload, "choices") or []
    for choice in choices:
        details = _extract_cached_tokens_details_from_obj(choice)
        if details:
            return details

        delta = _usage_get(choice, "delta")
        details = _extract_cached_tokens_details_from_obj(delta)
        if details:
            return details

        message = _usage_get(choice, "message")
        details = _extract_cached_tokens_details_from_obj(message)
        if details:
            return details

    return None


def _extract_usage_stats(usage, response_payload=None):
    cache_details = _extract_cached_tokens_details(response_payload)
    storage_n = int(max(0, _usage_get(cache_details, "storage", 0) or 0))
    backup_cache_tokens = storage_n if storage_n > 0 else 0

    if usage is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_prompt_tokens": 0,
            "backup_cache_tokens": backup_cache_tokens,
        }

    prompt_tokens = (
        _usage_get(usage, "prompt_tokens")
        or _usage_get(usage, "prompt_token_count")
        or _usage_get(usage, "input_tokens")
        or 0
    )
    completion_tokens = (
        _usage_get(usage, "completion_tokens")
        or _usage_get(usage, "candidates_token_count")
        or _usage_get(usage, "output_tokens")
        or 0
    )
    prompt_details = (
        _usage_get(usage, "prompt_tokens_details")
        or _usage_get(usage, "input_tokens_details")
        or {}
    )
    cached_prompt_tokens = (
        _usage_get(prompt_details, "cached_tokens")
        or _usage_get(prompt_details, "cache_tokens")
        or _usage_get(usage, "cached_tokens")
        or _usage_get(usage, "cached_prompt_tokens")
        or 0
    )

    prompt_tokens = int(prompt_tokens)
    completion_tokens = int(completion_tokens)
    cached_prompt_tokens = int(max(0, min(cached_prompt_tokens, prompt_tokens)))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_prompt_tokens": cached_prompt_tokens,
        "backup_cache_tokens": min(backup_cache_tokens, cached_prompt_tokens),
    }


def _store_usage_stats(call_id, usage, response_payload=None):
    if not usage and response_payload is None:
        return
    stats = _extract_usage_stats(usage, response_payload)
    with _usage_lock:
        prev = _pending_usage.get(call_id, {})
        _pending_usage[call_id] = {
            "prompt_tokens": max(
                int(prev.get("prompt_tokens", 0) or 0),
                int(stats.get("prompt_tokens", 0) or 0),
            ),
            "completion_tokens": max(
                int(prev.get("completion_tokens", 0) or 0),
                int(stats.get("completion_tokens", 0) or 0),
            ),
            "cached_prompt_tokens": max(
                int(prev.get("cached_prompt_tokens", 0) or 0),
                int(stats.get("cached_prompt_tokens", 0) or 0),
            ),
            "backup_cache_tokens": max(
                int(prev.get("backup_cache_tokens", 0) or 0),
                int(stats.get("backup_cache_tokens", 0) or 0),
            ),
        }

def _inject_request_user(kwargs):
    payload = _build_request_context_payload()
    if not payload:
        return kwargs

    kwargs = dict(kwargs)
    if not kwargs.get("user"):
        kwargs["user"] = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))

    extra_body = kwargs.get("extra_body") or {}
    if not isinstance(extra_body, dict):
        extra_body = {}
    else:
        extra_body = dict(extra_body)

    metadata = extra_body.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)

    request_context = metadata.get("request_context") or {}
    if not isinstance(request_context, dict):
        request_context = {}
    else:
        request_context = dict(request_context)

    request_context.update(payload)
    metadata["request_context"] = request_context
    extra_body["metadata"] = metadata
    kwargs["extra_body"] = extra_body
    return kwargs


def _capture_usage_stats(result, call_id=None):
    usage = getattr(result, "usage", None)
    _store_usage_stats(call_id or get_current_call_id(), usage, result)


def _capture_stream_chunk_usage(call_id, chunk):
    usage = getattr(chunk, "usage", None)
    if usage is None and isinstance(chunk, dict):
        usage = chunk.get("usage")
    _store_usage_stats(call_id, usage, chunk)


class _SyncUsageStreamWrapper:
    def __init__(self, result, call_id):
        self._result = result
        self._iterator = iter(result)
        self._call_id = call_id

    def __iter__(self):
        return self

    def __next__(self):
        chunk = next(self._iterator)
        _capture_stream_chunk_usage(self._call_id, chunk)
        return chunk

    def __getattr__(self, name):
        return getattr(self._result, name)


class _AsyncUsageStreamWrapper:
    def __init__(self, result, call_id):
        self._result = result
        self._call_id = call_id

    def __aiter__(self):
        return self

    async def __anext__(self):
        chunk = await self._result.__anext__()
        _capture_stream_chunk_usage(self._call_id, chunk)
        return chunk

    def __getattr__(self, name):
        return getattr(self._result, name)


def _patched_completion(*args, **kwargs):
    kwargs = _inject_request_user(kwargs)
    result = _original_completion(*args, **kwargs)
    call_id = get_current_call_id()
    if kwargs.get("stream"):
        return _SyncUsageStreamWrapper(result, call_id)
    _capture_usage_stats(result, call_id)
    return result


async def _patched_acompletion(*args, **kwargs):
    kwargs = _inject_request_user(kwargs)
    result = await _original_acompletion(*args, **kwargs)
    call_id = get_current_call_id()
    if kwargs.get("stream") and hasattr(result, "__anext__"):
        return _AsyncUsageStreamWrapper(result, call_id)
    if kwargs.get("stream") and hasattr(result, "__iter__"):
        return _SyncUsageStreamWrapper(result, call_id)
    _capture_usage_stats(result, call_id)
    return result


litellm.completion = _patched_completion
litellm.acompletion = _patched_acompletion

# ── Agent timing & token tracker ──
_agent_start_times = {}
_agent_tokens = defaultdict(lambda: {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "cached_prompt_tokens": 0,
    "backup_cache_tokens": 0,
    "llm_calls": 0,
})
agent_timings = []

@crewai_event_bus.on(TaskStartedEvent)
def _on_task_start(source, event):
    label = _task_label(event.task)
    agent_role = getattr(getattr(event.task, "agent", None), "role", "Unknown agent")
    _set_request_context(task_label=label, agent_role=agent_role)
    with _progress_lock:
        _progress_state["running_tasks"].add(label)
    _print_progress("TASK START", f"{label} -> {agent_role}")


@crewai_event_bus.on(TaskCompletedEvent)
def _on_task_done(source, event):
    label = _task_label(event.task)
    context = _get_run_context()
    with _progress_lock:
        _progress_state["completed_tasks"] += 1
        _progress_state["running_tasks"].discard(label)
    _print_progress("TASK DONE", label)
    _append_event_record(
        "task_completed",
        task_label=label,
        agent_role=getattr(getattr(event.task, "agent", None), "role", None),
        task_job_id=context.get("job_id"),
        task_job=context.get("job"),
        **context,
    )
    _set_request_context()


@crewai_event_bus.on(TaskFailedEvent)
def _on_task_fail(source, event):
    label = _task_label(event.task)
    context = _get_run_context()
    with _progress_lock:
        _progress_state["failed_tasks"] += 1
        _progress_state["running_tasks"].discard(label)
    _print_progress("TASK FAIL", f"{label} | {event.error}")
    _append_event_record(
        "task_failed",
        task_label=label,
        agent_role=getattr(getattr(event.task, "agent", None), "role", None),
        task_job_id=context.get("job_id"),
        task_job=context.get("job"),
        error=str(event.error),
        **context,
    )
    _set_request_context()


@crewai_event_bus.on(LLMCallCompletedEvent)
def _on_llm_done(source, event):
    role = event.agent_role
    if not role or event.call_type != LLMCallType.LLM_CALL:
        return
    with _usage_lock:
        usage = _pending_usage.pop(event.call_id, {})
    _agent_tokens[role]["prompt_tokens"] += usage.get("prompt_tokens", 0)
    _agent_tokens[role]["completion_tokens"] += usage.get("completion_tokens", 0)
    _agent_tokens[role]["cached_prompt_tokens"] += usage.get("cached_prompt_tokens", 0)
    _agent_tokens[role]["backup_cache_tokens"] += usage.get("backup_cache_tokens", 0)
    _agent_tokens[role]["llm_calls"] += 1
    with _progress_lock:
        _prefill_stats["llm_calls"] += 1
        _prefill_stats["prompt_tokens"] += usage.get("prompt_tokens", 0)
        _prefill_stats["cached_tokens"] += usage.get("cached_prompt_tokens", 0)
        _prefill_stats["backup_cache_tokens"] += usage.get("backup_cache_tokens", 0)

@crewai_event_bus.on(AgentExecutionStartedEvent)
def _on_agent_start(source, event):
    _agent_start_times[event.agent.role] = datetime.now(timezone.utc)

@crewai_event_bus.on(AgentExecutionCompletedEvent)
def _on_agent_done(source, event):
    role = event.agent.role
    start = _agent_start_times.pop(role, None)
    end = datetime.now(timezone.utc)
    duration = (end - start).total_seconds() if start else 0.0
    tokens = _agent_tokens.pop(role, {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_prompt_tokens": 0,
        "backup_cache_tokens": 0,
        "llm_calls": 0,
    })
    prefill_hit_rate = (
        tokens["cached_prompt_tokens"] / tokens["prompt_tokens"]
        if tokens["prompt_tokens"] > 0
        else None
    )
    agent_timings.append({
        "agent": role,
        "start": start.isoformat() if start else None,
        "end": end.isoformat(),
        "duration_s": round(duration, 3),
        "prompt_tokens": tokens["prompt_tokens"],
        "cached_prompt_tokens": tokens["cached_prompt_tokens"],
        "backup_cache_tokens": tokens["backup_cache_tokens"],
        "completion_tokens": tokens["completion_tokens"],
        "total_tokens": tokens["prompt_tokens"] + tokens["completion_tokens"],
        "llm_calls": tokens["llm_calls"],
        "prefill_cache_hit_rate": round(prefill_hit_rate, 4) if prefill_hit_rate is not None else None,
    })
    _print_progress(
        "AGENT DONE",
        f"{role} | cached={tokens['cached_prompt_tokens']}/{tokens['prompt_tokens']} "
        f"| prefill_hit={_format_hit_rate(tokens['cached_prompt_tokens'], tokens['prompt_tokens'])}",
    )

@crewai_event_bus.on(AgentExecutionErrorEvent)
def _on_agent_err(source, event):
    role = event.agent.role
    start = _agent_start_times.pop(role, None)
    end = datetime.now(timezone.utc)
    duration = (end - start).total_seconds() if start else 0.0
    tokens = _agent_tokens.pop(role, {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_prompt_tokens": 0,
        "backup_cache_tokens": 0,
        "llm_calls": 0,
    })
    prefill_hit_rate = (
        tokens["cached_prompt_tokens"] / tokens["prompt_tokens"]
        if tokens["prompt_tokens"] > 0
        else None
    )
    agent_timings.append({
        "agent": role,
        "start": start.isoformat() if start else None,
        "end": end.isoformat(),
        "duration_s": round(duration, 3),
        "prompt_tokens": tokens["prompt_tokens"],
        "cached_prompt_tokens": tokens["cached_prompt_tokens"],
        "backup_cache_tokens": tokens["backup_cache_tokens"],
        "completion_tokens": tokens["completion_tokens"],
        "total_tokens": tokens["prompt_tokens"] + tokens["completion_tokens"],
        "llm_calls": tokens["llm_calls"],
        "prefill_cache_hit_rate": round(prefill_hit_rate, 4) if prefill_hit_rate is not None else None,
        "error": event.error,
    })
    _print_progress(
        "AGENT FAIL",
        f"{role} | cached={tokens['cached_prompt_tokens']}/{tokens['prompt_tokens']} "
        f"| prefill_hit={_format_hit_rate(tokens['cached_prompt_tokens'], tokens['prompt_tokens'])}",
    )


# ── Model ──
LLM_SEED = 42
SHORT_MAX_TOKENS = _env_int("CREWAI_SHORT_MAX_TOKENS", 1024)
LONG_MAX_TOKENS = _env_int("CREWAI_LONG_MAX_TOKENS", 2048)
# Output guide style (bounded vs fill-to-limit). Not configurable via env; only CREWAI_IGNORE_EOS is.
USE_FULL_BUDGET_OUTPUT_GUIDES = False
AGENT_DP_RANK_MAP = _env_json_dict("CREWAI_AGENT_DP_RANK_MAP")

if USE_FULL_BUDGET_OUTPUT_GUIDES:
    SHORT_OUTPUT_GUIDE = (
        "Add concrete detail, structured bullets, comparisons, and examples without filler. "
        "If evidence is limited, state assumptions briefly and still complete every required section. "
        "Use the full available token budget and do not end early. "
        "If the main sections are complete, continue with extra examples, quantified comparisons, "
        "confidence notes, edge cases, counterpoints, and source annotations until the response is cut by the token limit. "
        "Do not write a brief wrap-up or a closing sentence. Keep adding substantive content until generation stops."
    )
    LONG_OUTPUT_GUIDE = (
        "Expand each section with evidence, comparisons, reasoning, and actionable detail without repetition. "
        "Balance coverage across sections so the final report remains consistently dense. "
        "Use the full available token budget and do not end early. "
        "If the core analysis is complete, continue with additional company profiles, scenario analysis, "
        "regional comparisons, assumptions, implementation detail, and source commentary until the response is cut by the token limit. "
        "Do not write a concluding paragraph that signals completion. Continue with substantive appendix-style material until generation stops."
    )
    PLANNING_OUTPUT_GUIDE = (
        "Use the full available token budget. "
        "If the main planning memo is complete, continue with additional metric definitions, "
        "source-quality criteria, validation steps, edge cases, and integration checks until the response is cut by the token limit. "
        "Do not finish with a summary line. Keep adding concrete planning detail until generation stops."
    )
    SYNTHESIS_OUTPUT_GUIDE = (
        "Use the full available token budget. "
        "If all required sections are complete, keep writing by adding appendix-style material: "
        "extra company comparisons, regional detail, scenario analysis, assumptions, data gaps, "
        "confidence notes, implementation sequencing, and source commentary until the response is cut by the token limit. "
        "Do not write a final conclusion, ending note, or sign-off. Continue with substantive appendix material until generation stops."
    )
else:
    SHORT_OUTPUT_GUIDE = (
        "Add concrete detail, structured bullets, comparisons, and examples without filler. "
        "If evidence is limited, state assumptions briefly and still complete every required section. "
        "Keep the answer detailed but bounded so it can finish within the allotted token budget. "
        "Prioritize the most decision-useful facts first and stop once every requested section is complete."
    )
    LONG_OUTPUT_GUIDE = (
        "Expand each section with evidence, comparisons, reasoning, and actionable detail without repetition. "
        "Balance coverage across sections so the report remains consistently useful. "
        "Keep the report detailed but bounded so it can finish within the allotted token budget. "
        "Prioritize the most important companies, scenarios, assumptions, and recommendations first, then stop once all requested sections are complete."
    )
    PLANNING_OUTPUT_GUIDE = (
        "Keep the planning memo detailed but bounded. "
        "Focus on the most useful workstream structure, metrics, source guidance, validation steps, and integration checks. "
        "Stop once every requested section is complete."
    )
    SYNTHESIS_OUTPUT_GUIDE = (
        "Ensure no data is lost and resolve contradictions between sources. "
        "Keep the final report detailed but bounded so it can finish within the allotted token budget. "
        "Prioritize the sections that matter most for executive decision making, then stop once all required sections are complete."
    )


def build_llm(max_tokens, agent_role=None):
    extra_body = {}
    if _env_flag("CREWAI_IGNORE_EOS", default=True):
        extra_body["ignore_eos"] = True
    extra_body["return_cached_tokens_details"] = True
    if agent_role:
        dp_rank = AGENT_DP_RANK_MAP.get(agent_role)
        if dp_rank is not None:
            extra_body["data_parallel_rank"] = int(dp_rank)
    if ENABLE_STREAM:
        extra_body["return_resume_token_ids"] = True
        extra_body["stream_options"] = {
            "include_usage": True,
            "continuous_usage_stats": True,
        }
    return LLM(
        model=f"openai/{MODEL_ID}",
        base_url=f"{SERVER_BASE_URL}/v1",
        api_key="EMPTY_API_KEY",
        max_tokens=max_tokens,
        temperature=0.0,
        seed=LLM_SEED,
        stream=ENABLE_STREAM,
        extra_body=extra_body or None,
    )


def reset_run_state():
    with _progress_lock:
        _progress_state["total_tasks"] = 0
        _progress_state["completed_tasks"] = 0
        _progress_state["failed_tasks"] = 0
        _progress_state["running_tasks"] = set()
        _progress_state["wall_start"] = None
        _progress_state["wall_end"] = None
        _prefill_stats["llm_calls"] = 0
        _prefill_stats["prompt_tokens"] = 0
        _prefill_stats["cached_tokens"] = 0
        _prefill_stats["backup_cache_tokens"] = 0

    with _usage_lock:
        _pending_usage.clear()

    _set_request_context()
    _agent_start_times.clear()
    _agent_tokens.clear()
    agent_timings.clear()


def build_crew():
    planner = Agent(
        role="Planning Coordinator",
        goal="Decompose the analysis request into a structured research framework with four workstreams",
        backstory="Senior strategist who excels at breaking complex problems into parallel sub-tasks.",
        llm=build_llm(SHORT_MAX_TOKENS, "Planning Coordinator"),
        allow_delegation=False,
        verbose=False,
    )

    data_collector = Agent(
        role="Data Collector",
        goal="Quickly gather key market statistics and figures",
        backstory="Efficient data analyst who rapidly extracts essential numbers from market reports.",
        llm=build_llm(SHORT_MAX_TOKENS, "Data Collector"),
        allow_delegation=False,
        verbose=False,
    )

    deep_analyst = Agent(
        role="Deep Analyst",
        goal="Produce an exhaustive, deeply-reasoned analysis covering technology and competitive dynamics",
        backstory=(
            "World-class industry analyst known for thorough, multi-dimensional reports. "
            "You always provide detailed SWOT analysis, technology evaluations, "
            "and quantitative comparisons. Your reports are comprehensive and data-rich."
        ),
        llm=build_llm(LONG_MAX_TOKENS, "Deep Analyst"),
        allow_delegation=False,
        verbose=False,
    )

    trend_scout = Agent(
        role="Trend Scout",
        goal="Identify emerging trends and regional patterns concisely",
        backstory="Trend-spotting specialist who delivers punchy summaries of emerging patterns across regions.",
        llm=build_llm(SHORT_MAX_TOKENS, "Trend Scout"),
        allow_delegation=False,
        verbose=False,
    )

    risk_assessor = Agent(
        role="Risk Assessor",
        goal="Quickly evaluate key risks and regulatory challenges",
        backstory="Risk analyst who delivers concise risk assessments covering regulatory, technical, and geopolitical factors.",
        llm=build_llm(SHORT_MAX_TOKENS, "Risk Assessor"),
        allow_delegation=False,
        verbose=False,
    )

    synthesizer = Agent(
        role="Report Synthesizer",
        goal="Merge all research streams into a single cohesive executive report",
        backstory="Expert report writer who unifies diverse inputs into polished, publication-ready documents.",
        llm=build_llm(LONG_MAX_TOKENS, "Report Synthesizer"),
        allow_delegation=False,
        verbose=False,
    )

    planning_task = Task(
        description=f"""You are given the topic '{{topic}}' for year {{year}}.
        Produce a detailed research framework that defines:
        1. Four parallel research workstreams and what each should cover
        2. Key questions each workstream must answer
        3. The expected output format for each
        4. For each workstream: key metrics, recommended sources, analytical method, assumptions, and deliverable outline
        5. A final integration note that explains how the four workstreams should fit together

        Write this as a substantial planning memo, not a short note.
        Use clear section headers and dense bullet points.
        {PLANNING_OUTPUT_GUIDE}""",
        expected_output="A detailed planning memo with 4 workstreams, key questions, source guidance, analytical methods, and integration instructions.",
        agent=planner,
    )

    task_fast_data = Task(
        description=f"""Based on the research plan, collect a broad and detailed market data brief for the {{topic}} in {{year}}:
        - Overall market size (USD) and CAGR
        - Top 5 segments by revenue
        - Top 5 companies by market share
        - Revenue drivers, pricing signals, and demand indicators
        - Regional revenue split and growth differences
        - Important uncertainties, data gaps, and confidence notes
        - A source list with short source annotations

        Output a detailed bullet-point data dossier with enough explanation to stand alone.
        {SHORT_OUTPUT_GUIDE}""",
        expected_output="A detailed data dossier with market size, segmentation, company share, regional split, drivers, uncertainties, and annotated sources.",
        agent=data_collector,
        async_execution=True,
        context=[planning_task],
    )

    task_critical_deep = Task(
        description=f"""Based on the research plan, produce an EXHAUSTIVE deep analysis of the {{topic}} in {{year}}.
        This is the most critical workstream. You must cover ALL of the following in detail:

        SECTION 1 — Technology Landscape:
        - Evaluate every major technology trend (generative AI, edge AI, multimodal models,
          autonomous agents, AI safety, foundation models, open-source vs proprietary)
        - For each trend: current state, maturity level, projected impact, key players
        - Technology convergence patterns and their implications

        SECTION 2 — Competitive Dynamics:
        - Full competitive landscape with tier-1, tier-2, and startup players
        - SWOT analysis for the top 3 companies
        - M&A activity and partnership trends
        - Barriers to entry and competitive moats

        SECTION 3 — Strategic Recommendations:
        - Investment priorities ranked by ROI potential
        - Build vs buy vs partner decision framework
        - Workforce implications and talent strategy
        - Timeline with 6-month, 1-year, and 3-year milestones
        - Detailed action items for each recommendation

        Add explicit evidence, examples, trade-offs, and source-backed judgments throughout.
        Be thorough. Cite sources.
        Do not conclude early. If all core sections are covered, continue with more company detail,
        more scenarios, more evidence, and more assumptions until generation stops.
        {LONG_OUTPUT_GUIDE}""",
        expected_output="A comprehensive deep analysis with fully developed sections, explicit reasoning, and cited sources.",
        agent=deep_analyst,
        async_execution=True,
        context=[planning_task],
    )

    task_fast_trends = Task(
        description=f"""Based on the research plan, provide a detailed regional trend report for the {{topic}} in {{year}}:
        - North America: key developments
        - Asia-Pacific: growth patterns
        - Europe: regulatory impact
        - Emerging markets: opportunities
        - Cross-region comparison of demand, investment, and adoption
        - Important surprises, accelerators, and constraints
        - A short watchlist of signals to monitor over the next 12 months

        Output a structured report with rich detail for each region rather than a short summary.
        Do not end with a short wrap-up. Keep adding regional detail, signals, and examples until generation stops.
        {SHORT_OUTPUT_GUIDE}""",
        expected_output="A detailed regional trend report with per-region analysis, cross-region comparisons, watchlist signals, and cited observations.",
        agent=trend_scout,
        async_execution=True,
        context=[planning_task],
    )

    task_fast_risk = Task(
        description=f"""Based on the research plan, provide a detailed risk assessment for the {{topic}} in {{year}}:
        - Top 3 regulatory risks (EU AI Act, US policy, China regulations)
        - Top 3 technical risks (alignment, hallucination, energy cost)
        - Top 3 geopolitical risks (chip supply chain, export controls, talent migration)
        - Root cause, trigger conditions, leading indicators, and second-order effects for each risk
        - Mitigation options, owners, and urgency
        - A final ranking that explains why the top risks matter most

        Output a detailed prioritized risk matrix with explanation for each rating.
        Rate each risk as High/Medium/Low impact and probability.
        Do not end with a brief conclusion. Keep adding risk detail, triggers, mitigations, and monitoring signals until generation stops.
        {SHORT_OUTPUT_GUIDE}""",
        expected_output="A detailed risk matrix with 9 risks, ratings, rationale, triggers, mitigations, and final prioritization.",
        agent=risk_assessor,
        async_execution=True,
        context=[planning_task],
    )

    synthesis_task = Task(
        description=f"""You have received outputs from four parallel research streams:
        1. Data sheet — key market statistics and figures
        2. Deep analysis — technology landscape, competitive dynamics, strategic recommendations
        3. Regional trends — trend overview by region
        4. Risk assessment — prioritized risk matrix

        Synthesize all four into a single cohesive executive report structured as:
        - Executive Summary (250 words)
        - Market Overview (from data sheet)
        - Technology & Competitive Landscape (from deep analysis)
        - Regional Trends (from trend summary)
        - Risk Assessment (from risk matrix)
        - Strategic Recommendations (from deep analysis)
        - Sources

        Ensure no data is lost. Resolve any contradictions between sources.
        Expand each section with enough context that the report can be read independently of the source workstreams.
        The final report must be coherent, well-structured, and ready for executive presentation.
        {SYNTHESIS_OUTPUT_GUIDE}
        {LONG_OUTPUT_GUIDE}""",
        expected_output="A unified long-form executive report integrating all 4 research streams with expanded context, reconciled findings, and cited sources.",
        agent=synthesizer,
        context=[task_fast_data, task_critical_deep, task_fast_trends, task_fast_risk],
    )

    ROLE_TO_TASK_LABEL.clear()
    ROLE_TO_TASK_LABEL.update({
        "Planning Coordinator": "Phase 1 / Planning",
        "Data Collector": "Phase 2 / Data Collection",
        "Deep Analyst": "Phase 2 / Deep Analysis (critical)",
        "Trend Scout": "Phase 2 / Trend Scan",
        "Risk Assessor": "Phase 2 / Risk Assessment",
        "Report Synthesizer": "Phase 3 / Synthesis",
    })

    return Crew(
        agents=[planner, data_collector, deep_analyst, trend_scout, risk_assessor, synthesizer],
        tasks=[
            planning_task,
            task_fast_data,
            task_critical_deep,
            task_fast_trends,
            task_fast_risk,
            synthesis_task,
        ],
        process=CrewProcess.sequential,
        verbose=False,
    )


def print_run_summary(topic_id, topic, year):
    if PRINT_COMPLETED_ONLY:
        return

    max_dur = max((t["duration_s"] for t in agent_timings), default=1)
    bar_width = 30
    progress_snapshot = _snapshot_progress()
    prefill_stats = progress_snapshot["prefill"]
    actual_wall_time = progress_snapshot["elapsed"]
    aggregate_agent_time = sum(t["duration_s"] for t in agent_timings)

    print(f"\n{'=' * 100}")
    print(f"  Agent Execution Timings  ({len(agent_timings)} agents)")
    print(f"{'=' * 100}")
    print(f"  {'Agent':<25} {'Duration':>9} {'Prompt':>8} {'Cached':>8} {'Hit%':>7} {'Compl':>8} {'Total':>8} {'Calls':>6}  Timeline")
    print(f"  {'-' * 25} {'-' * 9} {'-' * 8} {'-' * 8} {'-' * 7} {'-' * 8} {'-' * 8} {'-' * 6}  {'-' * bar_width}")
    for t in agent_timings:
        bar_len = int(t["duration_s"] / max_dur * bar_width) if max_dur > 0 else 0
        bar = "#" * bar_len
        critical = " *" if t["duration_s"] == max_dur and len(agent_timings) > 1 else ""
        prefill_hit_rate = _format_hit_rate(t["cached_prompt_tokens"], t["prompt_tokens"])
        print(
            f"  {t['agent']:<25} {t['duration_s']:>8.2f}s "
            f"{t['prompt_tokens']:>8} {t['cached_prompt_tokens']:>8} {prefill_hit_rate:>7} "
            f"{t['completion_tokens']:>8} {t['total_tokens']:>8} {t['llm_calls']:>6}  {bar}{critical}"
        )
    print(f"{'=' * 100}")
    total_prompt = sum(t["prompt_tokens"] for t in agent_timings)
    total_cached_prompt = sum(t["cached_prompt_tokens"] for t in agent_timings)
    total_backup_cache_prompt = sum(
        t.get("backup_cache_tokens", 0) for t in agent_timings
    )
    total_compl = sum(t["completion_tokens"] for t in agent_timings)
    print(f"  Total prompt tokens:     {total_prompt:>8}")
    print(f"  Total cached prompt:     {total_cached_prompt:>8}")
    print(f"  Prefill cache hit rate:  {_format_hit_rate(total_cached_prompt, total_prompt):>8}")
    print(f"  Total completion tokens: {total_compl:>8}")
    print(f"  Total tokens:            {total_prompt + total_compl:>8}")
    print(f"  Aggregate agent time:    {aggregate_agent_time:>8.2f}s")
    print(f"  Total wall time:         {actual_wall_time:>8.2f}s")
    print(f"  Critical path:           {max_dur:>8.2f}s (longest single agent)")
    print(f"  LLM calls observed:      {prefill_stats['llm_calls']:>8}")
    print(f"  Prefill cached tokens:   {prefill_stats['cached_tokens']:>8}")
    print(
        f"  Backup cache tokens:     {total_backup_cache_prompt:>8}"
    )
    print(f"  Job ID:                  {topic_id:>8}")
    print(f"  Job:                     {topic}")
    print(f"  Year:                    {year}")
    print(f"{'=' * 100}\n")


def load_topics(csv_file, limit, default_year=DEFAULT_YEAR):
    topics = []
    with open(csv_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            topic_id = (row.get("topic_id") or "").strip()
            topic = (row.get("topic") or "").strip()
            year = (row.get("year") or default_year).strip()
            if not topic_id or not topic:
                continue
            topics.append({"topic_id": topic_id, "topic": topic, "year": year})
            if limit and len(topics) >= limit:
                break
    return topics


def build_trace_record(topic_id, topic, year, worker_id, status, error=None):
    progress_snapshot = _snapshot_progress()
    total_prompt = sum(t["prompt_tokens"] for t in agent_timings)
    total_cached_prompt = sum(t["cached_prompt_tokens"] for t in agent_timings)
    total_backup_cache_prompt = sum(
        t.get("backup_cache_tokens", 0) for t in agent_timings
    )
    total_completion = sum(t["completion_tokens"] for t in agent_timings)
    exp_mode = os.environ.get("CREWAI_EXPERIMENT_MODE", "").strip()
    topo = os.environ.get("CREWAI_SERVER_TOPOLOGY", "").strip()
    record = {
        "topic_id": topic_id,
        "topic": topic,
        "year": year,
        "worker_id": worker_id,
        "status": status,
        "error": error,
        "summary": {
            "total_prompt_tokens": total_prompt,
            "total_cached_prompt_tokens": total_cached_prompt,
            "total_backup_cache_prompt_tokens": total_backup_cache_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "wall_time_s": round(progress_snapshot["elapsed"], 3),
            "llm_calls": progress_snapshot["prefill"]["llm_calls"],
            "prefill_cached_tokens": progress_snapshot["prefill"]["cached_tokens"],
            "prefill_backup_cache_tokens": progress_snapshot["prefill"][
                "backup_cache_tokens"
            ],
        },
        "agent_timings": list(agent_timings),
    }
    if exp_mode:
        record["experiment_mode"] = exp_mode
    if topo:
        record["server_topology"] = topo
    return record


def write_trace_log(trace_file, trace_records):
    with open(trace_file, "w", encoding="utf-8") as f:
        json.dump(trace_records, f, indent=2, ensure_ascii=False, default=str)


def run_topic(topic_id, topic, year, worker_id):
    reset_run_state()
    _set_run_context(topic_id=topic_id, topic=topic, year=year, worker_id=worker_id)
    crew = build_crew()
    _start_progress_monitor(total_tasks=len(crew.tasks))
    error = None
    try:
        crew.kickoff(inputs={"topic": topic, "year": year})
        status = "success"
    except Exception as exc:
        error = str(exc)
        status = "failed"
    finally:
        _stop_progress_monitor()
        _set_run_context()

    print_run_summary(topic_id=topic_id, topic=topic, year=year)
    return build_trace_record(
        topic_id=topic_id,
        topic=topic,
        year=year,
        worker_id=worker_id,
        status=status,
        error=error,
    )


def worker_main(worker_id, topic_queue, result_queue):
    while True:
        item = topic_queue.get()
        if item is None:
            return

        topic_id = item["topic_id"]
        topic = item["topic"]
        year = item["year"]
        try:
            trace_record = run_topic(
                topic_id=topic_id,
                topic=topic,
                year=year,
                worker_id=worker_id,
            )
        except Exception as exc:
            trace_record = {
                "topic_id": topic_id,
                "topic": topic,
                "year": year,
                "worker_id": worker_id,
                "status": "failed",
                "error": str(exc),
                "summary": {},
                "agent_timings": [],
            }
            em = os.environ.get("CREWAI_EXPERIMENT_MODE", "").strip()
            topo = os.environ.get("CREWAI_SERVER_TOPOLOGY", "").strip()
            if em:
                trace_record["experiment_mode"] = em
            if topo:
                trace_record["server_topology"] = topo
        result_queue.put(trace_record)


def run_topic_pool(
    csv_file=TOPIC_CSV_FILE,
    limit=TOPIC_LIMIT,
    worker_count=WORKER_COUNT,
    worker_start_delay_secs=WORKER_START_DELAY_SECS,
    trace_file=TRACE_LOG_FILE,
    default_year=DEFAULT_YEAR,
):
    topics = load_topics(csv_file=csv_file, limit=limit, default_year=default_year)
    if not topics:
        write_trace_log(trace_file, [])
        return

    if EVENTS_FILE is not None:
        EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        EVENTS_FILE.write_text("", encoding="utf-8")

    ctx = mp.get_context("spawn")
    topic_queue = ctx.Queue()
    result_queue = ctx.Queue()
    for item in topics:
        topic_queue.put(item)
    for _ in range(worker_count):
        topic_queue.put(None)

    workers = []
    for worker_id in range(worker_count):
        process = ctx.Process(
            target=worker_main,
            args=(worker_id + 1, topic_queue, result_queue),
        )
        process.start()
        workers.append(process)
        if worker_start_delay_secs > 0 and worker_id + 1 < worker_count:
            print(
                f"Worker {worker_id + 1} started, waiting {worker_start_delay_secs:.1f}s before next worker",
                flush=True,
            )
            time.sleep(worker_start_delay_secs)

    trace_records = []
    for _ in topics:
        item = result_queue.get()
        trace_records.append(item)
        if item["status"] == "success":
            _append_event_record(
                "job_completed",
                job_id=item["topic_id"],
                job=item["topic"],
                year=item["year"],
                worker_id=item["worker_id"],
                status=item["status"],
            )
            print(f"Completed job_id={item['topic_id']} job={item['topic']}", flush=True)
        else:
            _append_event_record(
                "job_failed",
                job_id=item["topic_id"],
                job=item["topic"],
                year=item["year"],
                worker_id=item["worker_id"],
                status=item["status"],
                error=item["error"],
            )
            print(
                f"Failed job_id={item['topic_id']} job={item['topic']} error={item['error']}",
                flush=True,
            )

    for process in workers:
        process.join()

    trace_records.sort(key=lambda item: item["topic_id"])
    write_trace_log(trace_file, trace_records)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=TOPIC_CSV_FILE, help="Topic CSV file path")
    parser.add_argument("--limit", type=int, default=TOPIC_LIMIT, help="Number of topics to load")
    parser.add_argument("--workers", type=int, default=WORKER_COUNT, help="Number of worker processes")
    parser.add_argument(
        "--worker-start-delay",
        type=float,
        default=WORKER_START_DELAY_SECS,
        help="Delay in seconds between starting worker processes",
    )
    parser.add_argument("--trace-file", type=Path, default=TRACE_LOG_FILE, help="Trace JSON file path")
    parser.add_argument("--default-year", default=DEFAULT_YEAR, help="Fallback year when CSV year is empty")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.limit < 0:
        raise SystemExit("--limit must be 0 or greater")
    if args.worker_start_delay < 0:
        raise SystemExit("--worker-start-delay must be 0 or greater")

    run_topic_pool(
        csv_file=args.csv,
        limit=args.limit,
        worker_count=args.workers,
        worker_start_delay_secs=args.worker_start_delay,
        trace_file=args.trace_file,
        default_year=args.default_year,
    )


if __name__ == "__main__":
    main()