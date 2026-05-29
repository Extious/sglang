import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "synthesis_requests"
ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT)]

from benchmark.fault_tolerance.simulator.synthesis_requests.common.config import (
    build_simulator_config,
    load_experiment_suite,
)
from benchmark.fault_tolerance.simulator.synthesis_requests.common.schema import (
    BackupPolicy,
    BackupStrategy,
    ExecutionMode,
    SyntheticRequest,
)


def test_load_experiment_suite_from_json_files(tmp_path):
    cfg_dir = tmp_path / "synthesis-a100"
    cfg_dir.mkdir()
    (cfg_dir / "base.json").write_text(
        json.dumps(
            {
                "name": "synthesis-a100",
                "server": {
                    "model_path": "Qwen/Qwen3-8B",
                    "dp_size": 2,
                    "tp_size": 1,
                    "pp_size": 1,
                    "accelerator_name": "a100_sxm",
                    "port": 30000,
                },
                "workload": {
                    "input_len": 16,
                    "output_len": 4,
                    "num_requests": 8,
                    "app_workers": 2,
                    "request_rate": "inf",
                    "seed": 7,
                },
                "failure": {
                    "enabled": "false",
                    "inject_after_job": 2,
                    "recover_after_job": 6,
                    "delay": 0,
                    "dp_rank": 0,
                },
                "predictor": {
                    "name": "aiconfigurator",
                    "database_mode": "SILICON",
                },
                "platform": {
                    "disk_read_bandwidth_gb": 8,
                    "disk_write_bandwidth_gb": 8,
                    "memory_read_bandwidth_gb": 64,
                    "memory_write_bandwidth_gb": 64,
                    "num_device_per_node": 8,
                },
            }
        ),
        encoding="utf-8",
    )
    (cfg_dir / "strategies.json").write_text(
        json.dumps(
            {
                "strategies": {
                    "baseline": {"backup_policy": "none"},
                    "host_backup": {"backup_policy": "host"},
                    "remote_backup": {"backup_policy": "remote_backup"},
                }
            }
        ),
        encoding="utf-8",
    )

    suite = load_experiment_suite(cfg_dir)

    assert suite.name == "synthesis-a100"
    assert suite.server.model_path == "Qwen/Qwen3-8B"
    assert suite.workload.input_len == 16
    assert suite.failure.enabled is False
    assert suite.failure.inject_after_job == 2
    assert suite.predictor.name == "aiconfigurator"
    assert suite.platform.disk_read_bandwidth_gb == 8
    assert suite.platform.disk_write_bandwidth_gb == 8
    assert suite.platform.memory_read_bandwidth_gb == 64
    assert suite.platform.memory_write_bandwidth_gb == 64
    assert suite.platform.num_device_per_node == 8
    assert {s.strategy: s.backup_policy.value for s in suite.strategies} == {
        BackupStrategy.BASELINE: "none",
        BackupStrategy.HOST_BACKUP: "host",
        BackupStrategy.REMOTE_BACKUP: "remote_backup",
    }


def test_synthetic_request_normalizes_token_ids_to_tuple():
    request = SyntheticRequest(
        job_id="1",
        worker_id="1",
        worker_seq=0,
        assigned_dp_rank=0,
        token_ids=[1, 2, 3],
        output_length=4,
        created_time=0.0,
        backup_policy=BackupPolicy.HOST,
    )

    assert request.token_ids == (1, 2, 3)


def test_build_simulator_config_sets_strategy_and_mode(tmp_path):
    suite = load_experiment_suite(
        Path("benchmark/fault_tolerance/simulator/synthesis_requests/configs/synthesis-a100")
    )
    experiment = next(
        s for s in suite.strategies if s.strategy == BackupStrategy.HOST_BACKUP
    )

    payload = build_simulator_config(
        suite=suite,
        experiment=experiment,
        mode=ExecutionMode.OFFLINE,
    )

    assert payload["predictor"]["name"] == "aiconfigurator"
    assert payload["scheduler"]["dp_size"] == suite.server.dp_size
    assert payload["scheduler"]["tp_size"] == suite.server.tp_size
    assert payload["failure"]["enabled"] is True
    assert payload["failure"]["backup_policy"] == "host"
    assert payload["failure"]["failed_dp_rank"] == suite.failure.dp_rank
    assert payload["benchmark"]["workload"] == "synthesis_requests"
    assert payload["benchmark"]["execution_mode"] == "offline"
    assert payload["benchmark"]["strategy"] == "host_backup"
