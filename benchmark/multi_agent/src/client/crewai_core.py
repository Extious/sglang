# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.

"""
CrewAI workload runner.

Wraps a CrewAI multi-agent pipeline against a pre-deployed SGLang server,
collects per-job / per-task metrics (tokens, timing, cache hits),
and writes CSV summaries.
"""

from __future__ import annotations

import csv
import contextvars
import fcntl
import json
import os
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "true"

import httpx
import litellm
from crewai import Agent, Crew, Process as CrewProcess, Task, LLM
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionErrorEvent,
    AgentExecutionStartedEvent,
)
from crewai.events.types.llm_events import LLMCallCompletedEvent, LLMCallType
from crewai.llms.hooks.base import BaseInterceptor

from client.config import CrewAIRunnerConfig

litellm.suppress_debug_info = True
litellm.drop_params = True

_response_stats_var: contextvars.ContextVar[Optional[list[dict]]] = contextvars.ContextVar(
    "response_stats_queue",
    default=None,
)

_CONFIG_PATH = Path(__file__).resolve().parent / "crewai_config.json"


@dataclass
class TaskMetrics:
    agent_role: str
    task_label: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0
    backup_cache_tokens: int = 0
    llm_calls: int = 0
    cache_l1_match: int = 0
    cache_l2_match: int = 0
    cache_remote_match: int = 0
    cache_remote_prefetch: int = 0
    cache_reused_device: int = 0
    cache_reused_host: int = 0
    cache_reused_storage: int = 0
    is_failover_retried: bool = False
    pre_failover_output_tokens: int = 0
    pre_failover_backed_up_tokens: int = 0
    request_rids: list[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def duration_s(self) -> float:
        if self.start_time and self.end_time:
            return round((self.end_time - self.start_time).total_seconds(), 3)
        return 0.0

    @property
    def cache_hit_rate(self) -> Optional[float]:
        if self.prompt_tokens > 0:
            return round(self.cached_prompt_tokens / self.prompt_tokens, 4)
        return None
# PLACEHOLDER: JobMetrics


@dataclass
class JobMetrics:
    job_id: str
    topic: str
    year: str
    worker_id: str
    status: str = "running"
    error: Optional[str] = None
    wall_start: Optional[datetime] = None
    wall_end: Optional[datetime] = None
    tasks: list[TaskMetrics] = field(default_factory=list)

    @property
    def wall_time_s(self) -> Optional[float]:
        if self.wall_start and self.wall_end:
            return round((self.wall_end - self.wall_start).total_seconds(), 3)
        return None

    def to_trace_record(self) -> dict:
        total_prompt = sum(t.prompt_tokens for t in self.tasks)
        total_completion = sum(t.completion_tokens for t in self.tasks)
        total_cached = sum(t.cached_prompt_tokens for t in self.tasks)
        total_backup = sum(t.backup_cache_tokens for t in self.tasks)
        total_llm_calls = sum(t.llm_calls for t in self.tasks)
        return {
            "topic_id": self.job_id,
            "topic": self.topic,
            "year": self.year,
            "worker_id": self.worker_id,
            "status": self.status,
            "error": self.error,
            "agent_timings": [
                {
                    "agent": t.agent_role,
                    "task_label": t.task_label,
                    "start": t.start_time.isoformat() if t.start_time else None,
                    "end": t.end_time.isoformat() if t.end_time else None,
                    "duration_s": t.duration_s,
                    "prompt_tokens": t.prompt_tokens,
                    "cached_prompt_tokens": t.cached_prompt_tokens,
                    "backup_cache_tokens": t.backup_cache_tokens,
                    "completion_tokens": t.completion_tokens,
                    "total_tokens": t.prompt_tokens + t.completion_tokens,
                    "llm_calls": t.llm_calls,
                    "prefill_cache_hit_rate": t.cache_hit_rate,
                    "rid": _format_request_rids(t.request_rids),
                    "request_rids": list(t.request_rids),
                    **({"error": t.error} if t.error else {}),
                }
                for t in self.tasks
            ],
            "summary": {
                "wall_time_s": self.wall_time_s,
                "llm_calls": total_llm_calls,
                "prompt_tokens": total_prompt,
                "prefill_cached_tokens": total_cached,
                "backup_cache_tokens": total_backup,
                "total_tokens": total_prompt + total_completion,
            },
        }
# PLACEHOLDER: CrewAIRunner


def _load_config() -> dict:
    return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))


def _task_label(task) -> str:
    agent = getattr(task, "agent", None)
    role = getattr(agent, "role", None)
    if role:
        mapping = _load_config().get("role_to_task_label", {})
        if role in mapping:
            return mapping[role]
    return getattr(task, "name", None) or getattr(task, "description", "")[:60]


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
            return dumped if isinstance(dumped, dict) else {}
        except Exception:
            return {}
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if not k.startswith("_")}
    return {}


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y")
    return bool(value)


def _extract_cache_details(container: Any) -> dict:
    data = _as_dict(container)
    usage = _as_dict(data.get("usage")) if "usage" in data else data
    details = _as_dict(usage.get("cached_tokens_details"))
    if details:
        return details
    sglext = _as_dict(data.get("sglext"))
    return _as_dict(sglext.get("cached_tokens_details"))


def _extract_failover_details(container: Any) -> dict:
    data = _as_dict(container)
    usage = _as_dict(data.get("usage")) if "usage" in data else data
    sglext = _as_dict(data.get("sglext"))
    for raw in (
        usage.get("failover"),
        usage.get("failover_details"),
        sglext.get("failover"),
        sglext.get("failover_details"),
        data.get("failover"),
        data.get("failover_details"),
    ):
        failover = _as_dict(raw)
        if not failover:
            continue
        pre_output = int(failover.get("pre_failover_output_tokens", 0) or 0)
        backed_up = int(failover.get("pre_failover_backed_up_tokens", 0) or 0)
        is_retried = (
            _truthy(failover.get("is_failover_retried"))
            or _truthy(failover.get("is_retried"))
            or pre_output > 0
            or backed_up > 0
        )
        if not is_retried:
            continue
        return {
            "is_failover_retried": True,
            "is_retried": True,
            "pre_failover_output_tokens": pre_output,
            "pre_failover_backed_up_tokens": backed_up,
        }
    return {}


def _format_request_rids(rids: list[str]) -> str:
    return ";".join(str(rid) for rid in rids if str(rid))


_CACHE_CSV_HEADER = [
    "job_id", "task_label", "agent_role", "rid",
    "l1_match", "l2_match", "remote_match", "remote_prefetch",
    "reused_device", "reused_host", "reused_storage",
    "is_failover_retried", "pre_failover_output_tokens",
    "pre_failover_backed_up_tokens",
]


class CrewAIRunner:
    def __init__(self, cfg: CrewAIRunnerConfig) -> None:
        self.cfg = cfg
        self._crewai_cfg = _load_config()
        self._extra_instructions_text = self._load_extra_instructions(cfg)
        self._cache_csv_initialized = False
        self._cache_csv_lock = __import__("threading").Lock()
        self._cache_rows: list[dict] = []
        self._setup_proxy_bypass()
        self._interceptor = _CacheInterceptor(self)

    @staticmethod
    def _load_extra_instructions(cfg: CrewAIRunnerConfig) -> str:
        path = cfg.extra_instructions_path
        if not path:
            return ""
        p = Path(path)
        if not p.is_file():
            return ""
        return p.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all_jobs(self) -> list[JobMetrics]:
        records = self._load_topics()
        job_queue: queue.Queue[dict] = queue.Queue()
        for rec in records:
            job_queue.put(rec)

        results: list[JobMetrics] = []
        results_lock = threading.Lock()

        def _worker(worker_id: str) -> None:
            while True:
                try:
                    rec = job_queue.get_nowait()
                except queue.Empty:
                    break
                metric = self._run_single_job(
                    rec["job_id"], rec["topic"], rec["year"], worker_id,
                    _skip_stagger=True,
                )
                with results_lock:
                    results.append(metric)

        num_workers = min(self.cfg.app_workers, len(records))
        threads: list[threading.Thread] = []
        for i in range(num_workers):
            t = threading.Thread(
                target=_worker, args=(str(i + 1),), daemon=True,
            )
            threads.append(t)
            t.start()
            # Stagger each worker launch so jobs are offset in time.
            if self.cfg.worker_start_stagger_s > 0 and i < num_workers - 1:
                time.sleep(self.cfg.worker_start_stagger_s)

        for t in threads:
            t.join()

        return results

    def cache_hit_rows(self) -> list[dict]:
        return list(self._cache_rows)

    # ------------------------------------------------------------------
    # Topic loading
    # ------------------------------------------------------------------

    def _load_topics(self) -> list[dict]:
        records = []
        with open(self.cfg.jobs_csv, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                records.append(row)
        jobs = []
        for i, row in enumerate(records[: self.cfg.job_limit]):
            jobs.append({
                "job_id": row.get("id", str(i + 1)),
                "topic": row.get("topic", ""),
                "year": row.get("year", self.cfg.default_year),
            })
        return jobs

    # ------------------------------------------------------------------
    # Single job execution
    # ------------------------------------------------------------------

    def _run_single_job(
        self, job_id: str, topic: str, year: str, worker_id: str,
        *, _skip_stagger: bool = False,
    ) -> JobMetrics:
        if not _skip_stagger and self.cfg.worker_start_stagger_s > 0:
            try:
                idx = int(worker_id.strip())
            except ValueError:
                idx = 0
            time.sleep(float(idx) * self.cfg.worker_start_stagger_s)

        metrics = JobMetrics(
            job_id=job_id, topic=topic, year=year, worker_id=worker_id,
            wall_start=datetime.now(timezone.utc),
        )
        current_tasks: dict[str, TaskMetrics] = {}
        own_task_ids: set[str] = set()
        completed_task_ids: set[str] = set()
        handlers: list[tuple] = []

        def _event_task_id(event) -> str:
            task_id = str(getattr(event, "task_id", "") or "")
            if task_id:
                return task_id
            task = getattr(event, "task", None)
            return str(getattr(task, "id", "") or "")

        def _is_own_event(event) -> bool:
            task_id = _event_task_id(event)
            return bool(task_id and task_id in own_task_ids)

        def _on_agent_start(source, event):
            if not _is_own_event(event):
                return
            task_id = _event_task_id(event)
            role = event.agent.role
            tm = TaskMetrics(agent_role=role, task_label=_task_label(event))
            tm.start_time = datetime.now(timezone.utc)
            current_tasks[task_id] = tm

        def _on_llm_done(source, event):
            role = getattr(event, "agent_role", None)
            if not role or event.call_type != LLMCallType.LLM_CALL:
                return
            if not _is_own_event(event):
                return
            tm = current_tasks.get(_event_task_id(event))
            if not tm:
                return
            response_stats = self._pop_response_stats()
            usage = event.usage or {}
            tm.prompt_tokens += int(usage.get("prompt_tokens", 0))
            tm.completion_tokens += int(usage.get("completion_tokens", 0))
            details = usage.get("prompt_tokens_details") or {}
            tm.cached_prompt_tokens += int(
                details.get("cached_tokens", usage.get("cached_prompt_tokens", 0)) or 0
            )
            tm.llm_calls += 1
            failover = response_stats.get("failover") or _extract_failover_details(usage)
            response_obj = getattr(event, "response", None)
            rid = str(response_stats.get("rid", "") or "")
            if isinstance(response_obj, dict):
                rid = rid or str(response_obj.get("id", "") or "")
            elif response_obj is not None:
                try:
                    rid = rid or str(getattr(response_obj, "id", "") or "")
                except Exception:
                    rid = ""
            if not rid:
                rid = str(getattr(event, "call_id", "") or "")
            if rid and rid not in tm.request_rids:
                tm.request_rids.append(rid)
            if isinstance(failover, dict):
                tm.is_failover_retried = tm.is_failover_retried or bool(
                    failover.get("is_failover_retried", False)
                    or failover.get("is_retried", False)
                )
                tm.pre_failover_output_tokens += int(
                    failover.get("pre_failover_output_tokens", 0) or 0
                )
                tm.pre_failover_backed_up_tokens += int(
                    failover.get("pre_failover_backed_up_tokens", 0) or 0
                )
            self._append_cache_csv_from_llm_event(
                job_id=job_id,
                agent_role=role,
                llm_call_index=tm.llm_calls,
                usage=usage,
                response_stats=response_stats,
                task_metrics=tm,
            )

        def _on_agent_done(source, event):
            if not _is_own_event(event):
                return
            task_id = _event_task_id(event)
            tm = current_tasks.get(task_id)
            if tm:
                tm.end_time = datetime.now(timezone.utc)
                if task_id not in completed_task_ids:
                    metrics.tasks.append(tm)
                    completed_task_ids.add(task_id)
                self._append_event(
                    "task_completed",
                    job_id=job_id, job=topic, year=year, worker_id=worker_id,
                    task_label=tm.task_label, agent_role=tm.agent_role,
                    duration_s=tm.duration_s,
                )

        def _on_agent_err(source, event):
            if not _is_own_event(event):
                return
            task_id = _event_task_id(event)
            tm = current_tasks.get(task_id)
            if tm:
                tm.end_time = datetime.now(timezone.utc)
                tm.error = str(event.error)[:200]
                if task_id not in completed_task_ids:
                    metrics.tasks.append(tm)
                    completed_task_ids.add(task_id)
                self._append_event(
                    "task_failed",
                    job_id=job_id, job=topic, year=year, worker_id=worker_id,
                    task_label=tm.task_label, agent_role=tm.agent_role,
                    error=tm.error,
                )

        handlers.append((AgentExecutionStartedEvent, crewai_event_bus.on(AgentExecutionStartedEvent)(_on_agent_start)))
        handlers.append((LLMCallCompletedEvent, crewai_event_bus.on(LLMCallCompletedEvent)(_on_llm_done)))
        handlers.append((AgentExecutionCompletedEvent, crewai_event_bus.on(AgentExecutionCompletedEvent)(_on_agent_done)))
        handlers.append((AgentExecutionErrorEvent, crewai_event_bus.on(AgentExecutionErrorEvent)(_on_agent_err)))

        error_msg = None
        try:
            crew = self._build_crew(topic, year)
            own_task_ids.update(
                str(getattr(task, "id", "") or "")
                for task in getattr(crew, "tasks", [])
            )
            crew.kickoff()
            metrics.status = "completed"
        except Exception as exc:
            error_msg = str(exc)[:500]
            metrics.status = "failed"
            metrics.error = error_msg
            print(f"[Job {job_id}] FAILED: {error_msg}", file=sys.stderr)
        finally:
            metrics.wall_end = datetime.now(timezone.utc)
            try:
                crewai_event_bus.flush(timeout=60.0)
            except Exception:
                pass
            for tm in metrics.tasks:
                self._append_cache_csv_task_final(job_id=job_id, tm=tm)
            for evt_type, h in handlers:
                crewai_event_bus.off(evt_type, h)

        evt = {
            "job_id": job_id, "job": topic, "year": year,
            "worker_id": worker_id, "status": metrics.status,
            "wall_time_s": metrics.wall_time_s,
        }
        if error_msg:
            evt["error"] = error_msg
        self._append_event("job_failed" if error_msg else "job_completed", **evt)
        return metrics

    # ------------------------------------------------------------------
    # Crew builder
    # ------------------------------------------------------------------

    def _build_crew(self, topic: str, year: str) -> Crew:
        cfg = self._crewai_cfg
        agents_cfg = cfg.get("agents", [])
        tasks_cfg = cfg.get("tasks", [])
        output_guides = cfg.get("output_guides", {})

        agents_by_role: dict[str, Agent] = {}
        for ac in agents_cfg:
            role = ac["role"]
            max_tok = (
                self.cfg.long_max_tokens
                if ac.get("max_tokens_tier") == "long"
                else self.cfg.short_max_tokens
            )
            llm = self._build_llm(cfg, max_tok, role)
            agents_by_role[role] = Agent(
                role=role,
                goal=ac["goal"].format(topic=topic, year=year),
                backstory=ac["backstory"].format(topic=topic, year=year),
                llm=llm,
                verbose=False,
            )

        tasks_by_id: dict[str, Task] = {}
        task_list: list[Task] = []
        for tc in tasks_cfg:
            context_ids = tc.get("context_task_ids") or tc.get("context") or []
            context_tasks = [tasks_by_id[cid] for cid in context_ids if cid in tasks_by_id]
            guide_key = tc.get("output_guide_key", "short")
            desc = tc["description"].format(
                topic=topic, year=year,
                output_guide=output_guides.get(guide_key, output_guides.get("short", "")),
                long_output_guide=output_guides.get("long", ""),
                planning_guide=output_guides.get("planning", ""),
            )
            if self._extra_instructions_text:
                desc = (
                    f"{topic}\n\n"
                    "## Additional instructions (reference)\n"
                    f"{self._extra_instructions_text}\n\n"
                    "## Task specification\n"
                    f"{desc}"
                )
            task = Task(
                description=desc,
                expected_output=tc["expected_output"],
                agent=agents_by_role[tc["agent_role"]],
                async_execution=tc.get("async_execution", False),
                context=context_tasks or None,
            )
            tasks_by_id[tc["id"]] = task
            task_list.append(task)

        return Crew(
            agents=list(agents_by_role.values()),
            tasks=task_list,
            process=CrewProcess.sequential,
            verbose=False,
        )

    def _build_llm(self, cfg: dict, max_tokens: int, agent_role: Optional[str] = None) -> LLM:
        llm_cfg = cfg.get("llm", {})
        extra_body: dict = {"ignore_eos": bool(self.cfg.ignore_eos), "return_cached_tokens_details": True}
        if agent_role:
            dp_rank = self.cfg.agent_dp_rank_map.get(agent_role)
            if dp_rank is not None:
                extra_body["data_parallel_rank"] = int(dp_rank)
        if self.cfg.enable_stream:
            extra_body["stream_options"] = {"include_usage": True}
        base = self.cfg.server_url.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        llm_kwargs: dict = {
            "model": self.cfg.model_path,
            "provider": "openai",
            "base_url": base,
            "api_key": "EMPTY_API_KEY",
            "max_tokens": max_tokens,
            "seed": llm_cfg.get("seed", 42),
            "temperature": 0.0,
            "stream": self.cfg.enable_stream,
            "additional_params": {"extra_body": extra_body},
            "interceptor": self._interceptor,
        }
        return LLM(**llm_kwargs)

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _append_event(self, event_type: str, **payload) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            **payload,
        }
        if self.cfg.control_url:
            body = json.dumps(record, ensure_ascii=True).encode("utf-8")
            url = self.cfg.control_url.rstrip("/") + "/event"
            req = urllib.request.Request(
                url,
                data=body,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            try:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(req, timeout=10) as resp:
                    resp.read()
            except (urllib.error.URLError, OSError) as exc:
                print(f"[control] failed to POST {event_type}: {exc}", file=sys.stderr)
            return
        if not self.cfg.events_file:
            return
        self.cfg.events_file.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=True) + "\n"
        with open(self.cfg.events_file, "a", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            fh.write(line)
            fh.flush()
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def _setup_proxy_bypass(self) -> None:
        from urllib.parse import urlparse
        parsed = urlparse(self.cfg.server_url)
        host = parsed.hostname
        if not host:
            return
        for key in ("NO_PROXY", "no_proxy"):
            items = [i.strip() for i in os.environ.get(key, "").split(",") if i.strip()]
            if host not in items:
                items.append(host)
                os.environ[key] = ",".join(items)

    def _on_response_body(self, body: dict) -> None:
        """Called by _CacheInterceptor with the parsed JSON response body."""
        stats = {
            "rid": str(body.get("id", "") or ""),
            "cache_details": _extract_cache_details(body),
            "failover": _extract_failover_details(body),
        }
        queue_ = _response_stats_var.get()
        if queue_ is None:
            queue_ = []
            _response_stats_var.set(queue_)
        queue_.append(stats)

    def _pop_response_stats(self) -> dict:
        queue_ = _response_stats_var.get()
        if queue_:
            return queue_.pop(0)
        return {}

    def _append_cache_csv_row(self, csv_path: Path, row: dict) -> None:
        with self._cache_csv_lock:
            self._cache_rows.append(row)

    def _append_cache_csv_from_llm_event(
        self,
        *,
        job_id: str,
        agent_role: str,
        llm_call_index: int,
        usage: dict,
        response_stats: dict,
        task_metrics: TaskMetrics,
    ) -> None:
        if not isinstance(usage, dict):
            return
        details = response_stats.get("cache_details") or _extract_cache_details(usage)
        failover = response_stats.get("failover") or _extract_failover_details(usage)

        def ig(key: str, default: int = 0) -> int:
            return int(details.get(key, default) or 0)

        l1_match = ig("device")
        l2_match = ig("host")
        remote_match = ig("storage_query")
        remote_prefetch = ig("storage")
        reused_device = ig("reused_device")
        reused_host = ig("reused_host")
        reused_storage = ig("reused_storage")

        task_metrics.cache_l1_match += l1_match
        task_metrics.cache_l2_match += l2_match
        task_metrics.cache_remote_match += remote_match
        task_metrics.cache_remote_prefetch += remote_prefetch
        task_metrics.cache_reused_device += reused_device
        task_metrics.cache_reused_host += reused_host
        task_metrics.cache_reused_storage += reused_storage
        task_metrics.backup_cache_tokens += remote_prefetch

    def _append_cache_csv_task_final(self, *, job_id: str, tm: TaskMetrics) -> None:
        row = {
            "job_id": str(job_id),
            "task_label": str(tm.task_label),
            "agent_role": str(tm.agent_role),
            "rid": _format_request_rids(tm.request_rids),
            "l1_match": int(tm.cache_l1_match),
            "l2_match": int(tm.cache_l2_match),
            "remote_match": int(tm.cache_remote_match),
            "remote_prefetch": int(tm.cache_remote_prefetch),
            "reused_device": int(tm.cache_reused_device),
            "reused_host": int(tm.cache_reused_host),
            "reused_storage": int(tm.cache_reused_storage),
            "is_failover_retried": "True" if tm.is_failover_retried else "",
            "pre_failover_output_tokens": (
                str(int(tm.pre_failover_output_tokens))
                if int(tm.pre_failover_output_tokens) > 0 else ""
            ),
            "pre_failover_backed_up_tokens": (
                str(int(tm.pre_failover_backed_up_tokens))
                if int(tm.pre_failover_backed_up_tokens) > 0 else ""
            ),
        }
        self._append_cache_csv_row(self.cfg.output_dir / "cache_hits.csv", row)


class _CacheInterceptor(BaseInterceptor[httpx.Request, httpx.Response]):
    """Intercepts raw HTTP responses to extract SGLang cache hit details."""

    def __init__(self, runner: CrewAIRunner) -> None:
        self._runner = runner

    def on_outbound(self, message: httpx.Request) -> httpx.Request:
        return message

    def on_inbound(self, message: httpx.Response) -> httpx.Response:
        try:
            # Read the raw body bytes — this consumes the stream.
            raw_bytes = message.read()
            body = json.loads(raw_bytes)
            if isinstance(body, dict) and "usage" in body:
                self._runner._on_response_body(body)
        except Exception:
            pass
        return message
