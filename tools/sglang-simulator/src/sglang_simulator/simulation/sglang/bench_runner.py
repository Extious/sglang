from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from sglang_simulator.simulation.benchmark import BaseBenchmarkRunner, BenchmarkConfig
from sglang_simulator.simulation.types import RequestStats
from sglang_simulator.simulation.utils import calc_metrics
from sglang_simulator.utils.logger import get_logger


def _simulator_output_dir() -> str:
    return os.environ.get(
        "SGLANG_SIMULATOR_OUTPUT_DIR", "/tmp/sglang_simulator/output"
    )


def _simulator_metrics_path() -> str:
    return f"{_simulator_output_dir()}/metrics.json"

if os.getenv("HISIM_SIMULATION_MODE") is None:
    os.environ["HISIM_SIMULATION_MODE"] = "OFFLINE"

logger = get_logger("sglang_simulator")


def _is_internal_simulated_failure(result: Any) -> bool:
    return isinstance(result, ValueError) and str(result) == "simulated gpu failure"


def _resolve_routed_dp_rank(simulation_params: dict) -> int | None:
    if "assigned_dp_rank" not in simulation_params:
        return None
    return int(simulation_params.get("assigned_dp_rank", 0) or 0)


def _collect_jsonl_records(base_dir: Path, filename: str) -> list[dict]:
    data: list[dict] = []
    dp_paths = sorted(base_dir.glob(f"dp_*/{filename}"))
    paths = dp_paths if dp_paths else [base_dir / filename]
    for file_path in paths:
        if not file_path.is_file():
            continue
        with open(file_path) as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data


def _request_stats_from_records(records: list[dict]) -> list[RequestStats]:
    field_names = {field.name for field in fields(RequestStats)}
    return [
        RequestStats(**{key: value for key, value in record.items() if key in field_names})
        for record in records
    ]


def _load_metrics_from_output_dir() -> dict | None:
    metrics_path = Path(_simulator_metrics_path())
    if not metrics_path.exists():
        logger.error(
            f"Failed to load metrics from serving backend. The metrics file should be loaded from {metrics_path}."
        )
        return None

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    request_records = _collect_jsonl_records(metrics_path.parent, "request.jsonl")
    if not request_records:
        return metrics

    aggregated_metrics = calc_metrics(_request_stats_from_records(request_records))
    if "time_cost" in metrics:
        aggregated_metrics["time_cost"] = metrics["time_cost"]
    with open(metrics_path, "w") as f:
        f.write(json.dumps(aggregated_metrics) + "\n")
    return aggregated_metrics


def _count_requests_by_dp_rank(dataset: Any) -> dict[int, int] | None:
    counts: dict[int, int] = {}
    has_dp_assignment = False
    for req in dataset:
        custom_params = getattr(req, "custom_params", None) or {}
        if "assigned_dp_rank" not in custom_params:
            continue
        has_dp_assignment = True
        dp_rank = int(custom_params.get("assigned_dp_rank", 0) or 0)
        counts[dp_rank] = counts.get(dp_rank, 0) + 1
    return counts if has_dp_assignment else None


def _resolve_total_request(
    dataset: Any,
    simulation_params: dict,
    dp_request_counts: dict[int, int] | None,
) -> int:
    if dp_request_counts is not None:
        dp_rank = int(simulation_params.get("assigned_dp_rank", 0) or 0)
        return dp_request_counts.get(dp_rank, 0)
    return len(dataset)


def _create_simulator_engine(server_args: Any):
    from sglang_simulator.simulation.sglang.hook_installer import install_sglang_hooks

    install_sglang_hooks()

    from sglang.srt.entrypoints.engine import Engine
    from sglang_simulator.simulation.sglang.subprocess_entry import (
        sim_run_scheduler_process,
    )

    class SimulatorEngine(Engine):
        run_scheduler_process_func = staticmethod(sim_run_scheduler_process)

    server_args.disable_cuda_graph = True
    return SimulatorEngine(**asdict(server_args))


class SGLangBenchmarkRunner(BaseBenchmarkRunner):
    def __init__(self, server_args: Any):
        self.server_args = server_args
        self.engine = _create_simulator_engine(server_args)

    def flush_cache(self):
        self.engine.flush_cache()

    def clear_hicache_storage(self):
        self.engine.tokenizer_manager.clear_hicache_storage()

    def get_request(
        self,
        dataset: Any,
        ignore_timestamp: bool = False,
        request_rate: float = float("inf"),
    ) -> Iterator[tuple[Any, dict]]:
        dp_request_counts = _count_requests_by_dp_rank(dataset)
        yield_delay = 0
        for req in dataset:
            if ignore_timestamp:
                created_time = yield_delay
                yield_delay += np.random.exponential(1.0 / request_rate)
            else:
                created_time = req.custom_params.get("created_time", 0)

            simulation_params = dict(req.custom_params or {})
            simulation_params["total_request"] = _resolve_total_request(
                dataset,
                simulation_params,
                dp_request_counts,
            )
            simulation_params["created_time"] = simulation_params.get(
                "created_time", created_time
            )
            routed_dp_rank = _resolve_routed_dp_rank(simulation_params)

            yield (req, simulation_params, routed_dp_rank)

    async def async_benchmark(
        self,
        benchmark_config: BenchmarkConfig,
        dataset: Any,
    ):
        await self.engine.tokenizer_manager.start_profile(profile_prefix="reset")

        if os.path.exists(_simulator_metrics_path()):
            with open(_simulator_metrics_path(), "w") as f:
                # clear data
                pass

        tasks = []
        logger.info(f"Created {len(dataset)} request tasks.")
        for req, simulation_params, routed_dp_rank in self.get_request(
            dataset,
            ignore_timestamp=benchmark_config.ignore_request_timestamp,
            request_rate=benchmark_config.request_rate,
        ):
            generate_kwargs = {
                "prompt": req.prompt,
                "input_ids": req.token_ids,
                "sampling_params": {
                    "ignore_eos": True,
                    "max_new_tokens": req.output_length,
                    "custom_params": {
                        # (tmp) Transfer simulation arguments to the scheduler through the custom_params in sampling_params
                        "simulation": simulation_params
                    },
                },
            }
            if routed_dp_rank is not None:
                generate_kwargs["routed_dp_rank"] = routed_dp_rank
            task = asyncio.create_task(self.engine.async_generate(**generate_kwargs))
            tasks.append(task)

        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        unexpected_errors = [
            result
            for result in task_results
            if isinstance(result, Exception)
            and not _is_internal_simulated_failure(result)
        ]
        if unexpected_errors:
            raise unexpected_errors[0]

        # dump result
        await self.engine.tokenizer_manager.start_profile()

        return _load_metrics_from_output_dir()

    async def async_benchmark_parallel_clients(
        self,
        dataset: Any,
        *,
        app_workers: int,
    ):
        await self.engine.tokenizer_manager.start_profile(profile_prefix="reset")

        if os.path.exists(_simulator_metrics_path()):
            with open(_simulator_metrics_path(), "w"):
                pass

        from collections import defaultdict

        worker_queues: dict[str, list[Any]] = defaultdict(list)
        for req in dataset:
            worker_id = str((req.custom_params or {}).get("worker_id", "1"))
            worker_queues[worker_id].append(req)
        for idx in range(max(int(app_workers), 1)):
            worker_queues.setdefault(str(idx + 1), [])
        total_requests = len(dataset)
        active_workers = sum(1 for reqs in worker_queues.values() if reqs)
        logger.info(
            f"Starting {active_workers} parallel clients for {total_requests} requests."
        )

        async def run_client(worker_id: str, requests: list[Any]):
            for req in requests:
                simulation_params = dict(req.custom_params or {})
                simulation_params["total_request"] = _resolve_total_request(
                    dataset,
                    simulation_params,
                    _count_requests_by_dp_rank(dataset),
                )
                simulation_params["worker_id"] = worker_id
                simulation_params["server_created_time"] = time.time()
                generate_kwargs = {
                    "prompt": req.prompt,
                    "input_ids": req.token_ids,
                    "sampling_params": {
                        "ignore_eos": True,
                        "max_new_tokens": req.output_length,
                        "custom_params": {"simulation": simulation_params},
                    },
                }
                routed_dp_rank = _resolve_routed_dp_rank(simulation_params)
                if routed_dp_rank is not None:
                    generate_kwargs["routed_dp_rank"] = routed_dp_rank
                await self.engine.async_generate(**generate_kwargs)

        tasks = [
            asyncio.create_task(run_client(worker_id, requests))
            for worker_id, requests in sorted(worker_queues.items())
            if requests
        ]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        unexpected_errors = [
            result
            for result in task_results
            if isinstance(result, Exception)
            and not _is_internal_simulated_failure(result)
        ]
        if unexpected_errors:
            raise unexpected_errors[0]

        await self.engine.tokenizer_manager.start_profile()

        return _load_metrics_from_output_dir()

    def benchmark_parallel_clients(self, dataset: Any, *, app_workers: int):
        return self.engine.loop.run_until_complete(
            self.async_benchmark_parallel_clients(
                dataset,
                app_workers=app_workers,
            )
        )

    def benchmark(self, benchmark_config: BenchmarkConfig, dataset: Any):
        return self.engine.loop.run_until_complete(
            self.async_benchmark(benchmark_config, dataset)
        )

    def get_iteration_stats(self) -> list[dict]:
        base_dir = Path(_simulator_output_dir())
        data = _collect_jsonl_records(base_dir, "iteration.jsonl")
        if not data:
            logger.error(
                f"The iteration statistics data under {base_dir} does not exist."
            )
        return data

    def get_request_stats(self) -> list[dict]:
        base_dir = Path(_simulator_output_dir())
        data = _collect_jsonl_records(base_dir, "request.jsonl")
        if not data:
            logger.error(
                f"The request statistics data under {base_dir} does not exist."
            )
        return data

    def get_failure_events(self) -> list[dict]:
        return _collect_jsonl_records(
            Path(_simulator_output_dir()), "failure_events.jsonl"
        )

    def shutdown(self):
        logger.info("Attempting to shut down the SGLang backend engine.")
        return self.engine.shutdown()
