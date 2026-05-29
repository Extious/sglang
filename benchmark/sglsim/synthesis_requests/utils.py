from __future__ import annotations

import csv
import json
import logging
import random
import statistics
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_DIR = BENCHMARK_DIR / "config" / "synthesis-a100-smoke"
DEFAULT_RESULTS_ROOT = BENCHMARK_DIR / "results"
LOGS_DIR = BENCHMARK_DIR / "logs"
CLIENT_LOG_PATH = LOGS_DIR / "client.log"
SERVER_LOG_PATH = LOGS_DIR / "server.log"
ROUTING_KEY_HEADER = "X-SMG-Routing-Key"


@dataclass(frozen=True)
class ClientConfig:
    dataset_name: str
    seed: int
    request_rate: float
    max_concurrency: int | None
    gsp_num_groups: int
    gsp_prompts_per_group: int
    gsp_system_prompt_len: int
    gsp_question_len: int
    gsp_output_len: int
    gsp_range_ratio: float
    gsp_fast_prepare: bool
    gsp_send_routing_key: bool
    gsp_num_turns: int
    gsp_ordered: bool
    num_prompts: int | None = None

    @property
    def num_requests(self) -> int:
        total = self.gsp_num_groups * self.gsp_prompts_per_group
        if self.num_prompts is None:
            return total
        return min(self.num_prompts, total)


@dataclass(frozen=True)
class ServerConfig:
    model_path: str
    host: str
    port: int
    dp_size: int
    tp_size: int
    pp_size: int
    raw: dict[str, Any]
    experiment_name: str = "synthesis"
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkloadRequest:
    rid: str
    prompt: str | list[str]
    prompt_len: int
    output_len: int
    group_id: int
    routing_key: str | None = None
    input_ids: list[int] | None = None

    @property
    def first_turn_prompt(self) -> str:
        if isinstance(self.prompt, list):
            return self.prompt[0]
        return self.prompt


@dataclass
class RequestDetailRow:
    rid: str
    status: str
    created_time_s: float
    queue_start_s: float
    queue_end_s: float
    finish_time_s: float
    waiting_time_s: float
    inference_time_s: float
    total_latency_s: float
    input_length: int
    output_length: int
    cache_hit_tokens: int
    load_back_tokens: int
    prefetch_complete_tokens: int
    e2e_latency_s: float
    ttft_ms: float
    mean_tbt_ms: float
    error: str = ""


def setup_file_logger(name: str, log_path: Path, log_level: str = "info") -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    handler = logging.FileHandler(log_path, encoding="utf-8", mode="w")
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def read_log_tail(path: Path | None, max_lines: int = 80) -> str:
    if path is None or not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def parse_request_rate(value: Any) -> float:
    if value is None:
        return float("inf")
    if isinstance(value, str):
        if value.lower() in ("inf", "infinity"):
            return float("inf")
        return float(value)
    return float(value)


def require_request_rate_inf(rate: float) -> None:
    if rate != float("inf"):
        raise ValueError(
            "request_rate must be inf; use max_concurrency to control load"
        )


def _parse_extra_args(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("server.json extra_args must be a list of strings")
    return tuple(str(item) for item in raw)


def load_config_dir(config_dir: Path) -> tuple[ClientConfig, ServerConfig]:
    client = load_client_config(config_dir / "client.json")
    server = load_server_config(config_dir / "server.json")
    return client, server


def load_client_config(path: Path) -> ClientConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    request_rate = parse_request_rate(raw.get("request_rate", float("inf")))
    require_request_rate_inf(request_rate)
    return ClientConfig(
        dataset_name=str(raw.get("dataset_name", "generated-shared-prefix")),
        seed=int(raw.get("seed", 1)),
        request_rate=request_rate,
        max_concurrency=(
            int(raw["max_concurrency"]) if raw.get("max_concurrency") is not None else None
        ),
        gsp_num_groups=int(raw["gsp_num_groups"]),
        gsp_prompts_per_group=int(raw["gsp_prompts_per_group"]),
        gsp_system_prompt_len=int(raw["gsp_system_prompt_len"]),
        gsp_question_len=int(raw["gsp_question_len"]),
        gsp_output_len=int(raw["gsp_output_len"]),
        gsp_range_ratio=float(raw.get("gsp_range_ratio", 1.0)),
        gsp_fast_prepare=bool(raw.get("gsp_fast_prepare", False)),
        gsp_send_routing_key=bool(raw.get("gsp_send_routing_key", False)),
        gsp_num_turns=int(raw.get("gsp_num_turns", 1)),
        gsp_ordered=bool(raw.get("gsp_ordered", False)),
        num_prompts=(
            int(raw["num_prompts"]) if raw.get("num_prompts") is not None else None
        ),
    )


def load_server_config(path: Path) -> ServerConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    sched = raw.get("scheduler", {})
    if not isinstance(sched, dict):
        sched = {}
    return ServerConfig(
        model_path=str(raw["model_path"]),
        host=str(raw.get("host", "127.0.0.1")),
        port=int(raw.get("port", 30000)),
        dp_size=int(sched.get("dp_size", raw.get("dp_size", 1))),
        tp_size=int(sched.get("tp_size", raw.get("tp_size", 1))),
        pp_size=int(sched.get("pp_size", raw.get("pp_size", 1))),
        raw=dict(raw),
        experiment_name=str(
            raw.get("experiment_name", raw.get("name", "synthesis"))
        ),
        extra_args=_parse_extra_args(raw.get("extra_args")),
    )


def new_run_dir(results_root: Path, *, suite_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(results_root) / suite_name / "online" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_run_config(
    run_dir: Path,
    *,
    client: ClientConfig,
    server: ServerConfig,
    timestamp: str,
) -> Path:
    path = run_dir / "server_config.json"
    workload = asdict(client)
    workload["num_requests"] = client.num_requests
    workload["request_rate"] = (
        "inf" if client.request_rate == float("inf") else client.request_rate
    )
    payload = {
        "backend": "sglang",
        "execution_mode": "online",
        "benchmark": {
            "suite": server.experiment_name,
            "workload": "synthesis_requests",
            "execution_mode": "online",
            "compare_target": "sglsim",
        },
        "experiment": {
            "timestamp": timestamp,
            "name": server.experiment_name,
        },
        "server": server.raw,
        "deploy": {
            "model_path": server.model_path,
            "dp_size": server.dp_size,
            "tp_size": server.tp_size,
            "pp_size": server.pp_size,
            "host": server.host,
            "port": server.port,
            "extra_args": list(server.extra_args),
        },
        "workload": workload,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def compute_random_lens(full_len: int, range_ratio: float, num: int) -> list[int]:
    if full_len <= 0:
        return [0] * num
    return np.random.randint(
        max(int(full_len * range_ratio), 1),
        full_len + 1,
        size=num,
    ).tolist()


def gen_prompt(tokenizer: Any, token_num: int) -> str:
    if token_num <= 0:
        return ""
    vocab = tokenizer.get_vocab()
    available = [tid for tid in vocab.values() if isinstance(tid, int)]
    selected = random.choices(available, k=token_num)
    return tokenizer.decode(selected)


@lru_cache(maxsize=4)
def get_tokenizer(model_path: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
    )


def _encode_prompt_len(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def generate_shared_prefix_workload(
    client: ClientConfig,
    server: ServerConfig,
    *,
    logger: logging.Logger | None = None,
) -> list[WorkloadRequest]:
    from .progress import iter_with_progress, tqdm_logging

    if client.dataset_name != "generated-shared-prefix":
        raise ValueError(
            f"Unsupported dataset_name={client.dataset_name!r}; "
            "only generated-shared-prefix is supported"
        )

    model_path = server.model_path
    if logger:
        logger.info(
            "Generating GSP workload gsp_num_groups=%d gsp_prompts_per_group=%d "
            "gsp_system_prompt_len=%d gsp_question_len=%d gsp_output_len=%d "
            "gsp_range_ratio=%.3f gsp_num_turns=%d gsp_fast_prepare=%s",
            client.gsp_num_groups,
            client.gsp_prompts_per_group,
            client.gsp_system_prompt_len,
            client.gsp_question_len,
            client.gsp_output_len,
            client.gsp_range_ratio,
            client.gsp_num_turns,
            client.gsp_fast_prepare,
        )

    random.seed(client.seed)
    np.random.seed(client.seed)
    tokenizer = get_tokenizer(model_path)

    system_prompt_lens = compute_random_lens(
        client.gsp_system_prompt_len, client.gsp_range_ratio, client.gsp_num_groups
    )
    question_lens = np.array(
        compute_random_lens(
            client.gsp_question_len,
            client.gsp_range_ratio,
            client.gsp_num_groups
            * client.gsp_prompts_per_group
            * client.gsp_num_turns,
        )
    ).reshape(client.gsp_num_groups, client.gsp_prompts_per_group, client.gsp_num_turns)
    output_lens = np.array(
        compute_random_lens(
            client.gsp_output_len,
            client.gsp_range_ratio,
            client.gsp_num_groups * client.gsp_prompts_per_group,
        )
    ).reshape(client.gsp_num_groups, client.gsp_prompts_per_group)

    system_prompts: list[str] = []
    if not client.gsp_fast_prepare:
        with tqdm_logging(logger):
            system_prompts = [
                gen_prompt(tokenizer, int(system_prompt_lens[i]))
                for i in iter_with_progress(
                    range(client.gsp_num_groups),
                    total=client.gsp_num_groups,
                    desc="Generating system prompts",
                    logger=logger,
                )
            ]

    questions: list[list[list[str]]] = [
        [[""] * client.gsp_num_turns for _ in range(client.gsp_prompts_per_group)]
        for _ in range(client.gsp_num_groups)
    ]
    with tqdm_logging(logger):
        for group_idx in iter_with_progress(
            range(client.gsp_num_groups),
            total=client.gsp_num_groups,
            desc="Generating questions",
            logger=logger,
        ):
            for prompt_idx in range(client.gsp_prompts_per_group):
                for turn_idx in range(client.gsp_num_turns):
                    questions[group_idx][prompt_idx][turn_idx] = gen_prompt(
                        tokenizer,
                        int(question_lens[group_idx, prompt_idx, turn_idx]),
                    )

    run_random_str = uuid.uuid4().hex[:8]
    run_start_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    draft_rows: list[tuple[int, int, str | list[str], int, int, str | None]] = []
    rid_counter = 0
    for group_idx in range(client.gsp_num_groups):
        routing_key = (
            f"{run_random_str}_{run_start_timestamp}_{group_idx}"
            if client.gsp_send_routing_key
            else None
        )
        system_prompt = system_prompts[group_idx] if system_prompts else ""
        for prompt_idx in range(client.gsp_prompts_per_group):
            rid_counter += 1
            turn_questions = questions[group_idx][prompt_idx]
            turn_prompts = [f"{system_prompt}\n\n{turn_questions[0]}"] + turn_questions[1:]
            full_prompt = (
                turn_prompts[0] if client.gsp_num_turns == 1 else turn_prompts
            )
            draft_rows.append(
                (
                    rid_counter,
                    group_idx,
                    full_prompt,
                    int(output_lens[group_idx, prompt_idx]),
                    1 if client.gsp_fast_prepare else -1,
                    routing_key,
                )
            )

    requests: list[WorkloadRequest] = []
    if client.gsp_fast_prepare:
        for rid, group_id, full_prompt, output_len, _, routing_key in draft_rows:
            requests.append(
                WorkloadRequest(
                    rid=str(rid),
                    prompt=full_prompt,
                    prompt_len=1,
                    output_len=output_len,
                    group_id=group_id,
                    routing_key=routing_key,
                )
            )
    else:
        with tqdm_logging(logger):
            for rid, group_id, full_prompt, output_len, _, routing_key in iter_with_progress(
                draft_rows,
                total=len(draft_rows),
                desc="Computing prompt lengths",
                logger=logger,
            ):
                first_turn = (
                    full_prompt if isinstance(full_prompt, str) else full_prompt[0]
                )
                prompt_len = _encode_prompt_len(tokenizer, first_turn)
                requests.append(
                    WorkloadRequest(
                        rid=str(rid),
                        prompt=full_prompt,
                        prompt_len=prompt_len,
                        output_len=output_len,
                        group_id=group_id,
                        routing_key=routing_key,
                    )
                )

    if client.num_prompts is not None:
        requests = requests[: client.num_prompts]

    if not client.gsp_ordered:
        random.shuffle(requests)

    if logger:
        logger.info("Generated %d GSP requests", len(requests))
    return requests


def response_to_detail_row(
    rid: str,
    response: dict[str, Any],
    e2e_latency_s: float,
    *,
    prepared_prompt_len: int,
    prepared_output_len: int,
    error: str = "",
) -> RequestDetailRow:
    status = "failed" if error else "completed"
    usage = response.get("usage", {})
    if not isinstance(usage, dict):
        usage = {}
    prompt_tokens = int(usage.get("prompt_tokens", prepared_prompt_len) or 0)
    completion_tokens = int(
        usage.get("completion_tokens", prepared_output_len) or 0
    )
    return RequestDetailRow(
        rid=rid,
        status=status,
        created_time_s=-1.0,
        queue_start_s=-1.0,
        queue_end_s=-1.0,
        finish_time_s=e2e_latency_s,
        waiting_time_s=0.0,
        inference_time_s=e2e_latency_s,
        total_latency_s=e2e_latency_s,
        input_length=prompt_tokens or prepared_prompt_len,
        output_length=completion_tokens or prepared_output_len,
        cache_hit_tokens=0,
        load_back_tokens=0,
        prefetch_complete_tokens=0,
        e2e_latency_s=e2e_latency_s,
        ttft_ms=0.0,
        mean_tbt_ms=0.0,
        error=error,
    )


def write_request_details_csv(run_dir: Path, rows: list[RequestDetailRow]) -> Path:
    path = run_dir / "request_details.csv"
    fieldnames = [
        "rid",
        "status",
        "created_time_s",
        "queue_start_s",
        "queue_end_s",
        "finish_time_s",
        "waiting_time_s",
        "inference_time_s",
        "total_latency_s",
        "input_length",
        "output_length",
        "cache_hit_tokens",
        "load_back_tokens",
        "prefetch_complete_tokens",
        "e2e_latency_s",
        "ttft_ms",
        "mean_tbt_ms",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return path


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def aggregate_metrics(
    rows: list[RequestDetailRow],
    *,
    workload_start_s: float,
    workload_end_s: float,
) -> dict[str, Any]:
    completed = [row for row in rows if row.status == "completed"]
    failed = [row for row in rows if row.status != "completed"]
    duration = max(0.0, workload_end_s - workload_start_s)
    e2e_latencies = [row.e2e_latency_s for row in completed]
    total_input = sum(row.input_length for row in completed)
    total_output = sum(row.output_length for row in completed)
    completed_count = len(completed)

    def _mean_ms(values: list[float]) -> float:
        if not values:
            return 0.0
        return statistics.mean(values) * 1000.0

    return {
        "completed": completed_count,
        "failed": len(failed),
        "duration": duration,
        "total_input": total_input,
        "total_output": total_output,
        "request_throughput": completed_count / duration if duration > 0 else 0.0,
        "input_throughput": total_input / duration if duration > 0 else 0.0,
        "output_throughput": total_output / duration if duration > 0 else 0.0,
        "total_throughput": (total_input + total_output) / duration if duration > 0 else 0.0,
        "mean_e2e_latency_ms": _mean_ms(e2e_latencies),
        "median_e2e_latency_ms": statistics.median(e2e_latencies) * 1000.0
        if e2e_latencies
        else 0.0,
        "p90_e2e_latency_ms": _percentile(e2e_latencies, 90.0) * 1000.0,
        "p99_e2e_latency_ms": _percentile(e2e_latencies, 99.0) * 1000.0,
        "max_e2e_latency_ms": max(e2e_latencies) * 1000.0 if e2e_latencies else 0.0,
        "backend": "sglang",
    }


def write_metrics_json(run_dir: Path, metrics: dict[str, Any]) -> Path:
    path = run_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path
