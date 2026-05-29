import json

from sglang_simulator.simulation.manager import ConfigManager
from sglang_simulator.simulation.manager.failure import (
    BackupPolicy,
    FailureConfig,
    FailureManager,
)


def test_failure_config_defaults_disabled():
    cfg = FailureConfig.from_dict({})
    assert cfg.enabled is False
    assert cfg.backup_policy == BackupPolicy.NONE


def test_failure_after_request_schedules_at_completion_plus_delay():
    cfg = FailureConfig.from_dict(
        {
            "enabled": True,
            "backup_policy": "remote_backup",
            "failed_dp_rank": 0,
            "inject_after_request": 2,
            "inject_delay_s": 3.0,
            "recover_after_request": 4,
        }
    )
    mgr = FailureManager(cfg)
    assert mgr.on_request_completed("r1", completion_time_s=10.0) is None
    event = mgr.on_request_completed("r2", completion_time_s=12.0)
    assert event is not None
    assert event.fire_time_s == 15.0
    assert event.failed_dp_rank == 0
    assert event.backup_policy == BackupPolicy.REMOTE_BACKUP


def test_failure_config_parses_aliases_numeric_strings_and_false_strings():
    cfg = FailureConfig.from_dict(
        {
            "enabled": "false",
            "backup_policy": "host",
            "dp_rank": "3",
            "inject_after_request": "2",
            "inject_after_task": "5",
            "inject_after_job": "7",
            "timeline_after_start_s": "11.5",
            "delay": "1.25",
            "recover_after_request": "13",
            "recover_after_task": "17",
            "recover_after_job": "19",
            "recovery_delay": "2.5",
            "host_backup_ratio": None,
            "remote_backup_ratio": None,
        }
    )

    assert cfg.enabled is False
    assert cfg.backup_policy == BackupPolicy.HOST
    assert cfg.failed_dp_rank == 3
    assert cfg.inject_after_request == 2
    assert cfg.inject_after_task == 5
    assert cfg.inject_after_job == 7
    assert cfg.timeline_after_start_s == 11.5
    assert cfg.inject_delay_s == 1.25
    assert cfg.recover_after_request == 13
    assert cfg.recover_after_task == 17
    assert cfg.recover_after_job == 19
    assert cfg.recovery_delay_s == 2.5
    assert cfg.host_backup_ratio == 1.0
    assert cfg.remote_backup_ratio == 1.0


def test_failure_config_parses_common_false_boolean_values():
    for value in [False, 0, 0.0, "false", "False", "0", "no", "off", ""]:
        assert FailureConfig.from_dict({"enabled": value}).enabled is False


def test_request_completion_maps_task_and_job_thresholds_to_fixed_requests():
    cfg = FailureConfig.from_dict(
        {
            "enabled": True,
            "backup_policy": "remote_backup",
            "inject_after_task": 2,
            "inject_after_job": 3,
        }
    )
    mgr = FailureManager(cfg)

    assert mgr.on_request_completed("r1", completion_time_s=10.0) is None
    event = mgr.on_request_completed("r2", completion_time_s=11.0)
    assert event is not None
    assert event.fire_time_s == 11.0
    assert event.backup_policy == BackupPolicy.REMOTE_BACKUP


def test_recovery_schedules_from_fixed_request_threshold():
    cfg = FailureConfig.from_dict(
        {
            "enabled": True,
            "backup_policy": "remote_backup",
            "inject_after_request": 1,
            "recover_after_job": 2,
            "recovery_delay_s": 0.5,
        }
    )
    mgr = FailureManager(cfg)

    mgr.on_request_completed("r1", completion_time_s=10.0)
    assert mgr.should_fire(10.0) is not None
    mgr.on_request_completed("r2", completion_time_s=11.0)

    assert mgr.should_recover(11.4) is False
    assert mgr.should_recover(11.5) is True
    assert mgr.should_recover(12.0) is False


def test_maybe_schedule_timeline_adds_delay_once():
    cfg = FailureConfig.from_dict(
        {
            "enabled": True,
            "backup_policy": "host",
            "failed_dp_rank": 2,
            "timeline_after_start_s": 4.0,
            "inject_delay_s": 1.5,
        }
    )
    mgr = FailureManager(cfg)

    event = mgr.maybe_schedule_timeline()

    assert event is not None
    assert event.fire_time_s == 5.5
    assert event.failed_dp_rank == 2
    assert event.backup_policy == BackupPolicy.HOST
    assert mgr.maybe_schedule_timeline() is None


def test_should_fire_returns_event_at_fire_time_once():
    cfg = FailureConfig.from_dict(
        {
            "enabled": True,
            "backup_policy": "remote_backup",
            "inject_after_request": 1,
            "inject_delay_s": 2.0,
        }
    )
    mgr = FailureManager(cfg)
    scheduled = mgr.on_request_completed("r1", completion_time_s=10.0)

    assert scheduled is not None
    assert mgr.should_fire(11.9) is None
    assert mgr.should_fire(12.0) == scheduled
    assert mgr.should_fire(13.0) is None


def test_offline_request_start_schedules_failure_inside_post_threshold_request():
    cfg = FailureConfig.from_dict(
        {
            "enabled": True,
            "backup_policy": "host",
            "failed_dp_rank": 0,
            "inject_after_job": 10,
            "inject_delay_s": 5.0,
        }
    )
    mgr = FailureManager(cfg)

    assert (
        mgr.maybe_schedule_request_start(
            {
                "job_id": "10",
                "assigned_dp_rank": 0,
            },
            start_time_s=100.0,
        )
        is None
    )
    assert (
        mgr.maybe_schedule_request_start(
            {
                "job_id": "11",
                "assigned_dp_rank": 1,
            },
            start_time_s=101.0,
        )
        is None
    )

    event = mgr.maybe_schedule_request_start(
        {
            "job_id": "11",
            "assigned_dp_rank": 0,
        },
        start_time_s=102.0,
    )

    assert event is not None
    assert event.fire_time_s == 107.0
    assert event.failed_dp_rank == 0
    assert event.backup_policy == BackupPolicy.HOST
    assert mgr.maybe_schedule_request_start(
        {
            "job_id": "13",
            "assigned_dp_rank": 0,
        },
        start_time_s=110.0,
    ) is None


def test_offline_request_start_schedules_recovery_before_post_threshold_request():
    cfg = FailureConfig.from_dict(
        {
            "enabled": True,
            "backup_policy": "remote_backup",
            "failed_dp_rank": 0,
            "inject_after_job": 10,
            "recover_after_job": 20,
            "recovery_delay_s": 0.5,
        }
    )
    mgr = FailureManager(cfg)
    mgr.maybe_schedule_request_start(
        {
            "job_id": "11",
            "assigned_dp_rank": 0,
        },
        start_time_s=100.0,
    )
    assert mgr.should_fire(100.0) is not None

    assert (
        mgr.maybe_schedule_recovery_request_start(
            {
                "job_id": "20",
                "assigned_dp_rank": 0,
            },
            start_time_s=120.0,
        )
        is False
    )
    assert mgr.maybe_schedule_recovery_request_start(
        {
            "job_id": "21",
            "assigned_dp_rank": 1,
        },
        start_time_s=121.0,
    ) is False

    assert mgr.maybe_schedule_recovery_request_start(
        {
            "job_id": "21",
            "assigned_dp_rank": 0,
        },
        start_time_s=122.0,
    ) is True
    assert mgr.should_recover(122.4) is False
    assert mgr.should_recover(122.5) is True


def test_restore_penalty_uses_memory_for_host_and_disk_for_remote():
    cfg = FailureConfig.from_dict({"enabled": True, "backup_policy": "host"})
    mgr = FailureManager(cfg)
    assert (
        mgr.restore_penalty_s(
            backup_policy=BackupPolicy.NONE,
            kv_bytes_per_token=100,
            restored_tokens=10,
            memory_read_bandwidth=1000,
            disk_read_bandwidth=100,
        )
        == 0
    )
    assert (
        mgr.restore_penalty_s(
            backup_policy=BackupPolicy.HOST,
            kv_bytes_per_token=100,
            restored_tokens=10,
            memory_read_bandwidth=1000,
            disk_read_bandwidth=100,
        )
        == 1.0
    )
    assert (
        mgr.restore_penalty_s(
            backup_policy=BackupPolicy.REMOTE_BACKUP,
            kv_bytes_per_token=100,
            restored_tokens=10,
            memory_read_bandwidth=1000,
            disk_read_bandwidth=100,
        )
        == 10.0
    )


def test_manager_package_dir_lists_failure_exports():
    import sglang_simulator.simulation.manager as manager

    names = dir(manager)

    assert "FailureConfig" in names
    assert "FailureEvent" in names
    assert "FailureManager" in names


def test_config_manager_loads_failure_config(tmp_path, monkeypatch):
    config_path = tmp_path / "sim.json"
    config_path.write_text(
        json.dumps(
            {
                "platform": {
                    "accelerator": {"name": "a100_sxm", "hbm_capacity_gb": 80}
                },
                "predictor": {"name": "aiconfigurator"},
                "scheduler": {"tp_size": 1, "dp_size": 2, "backend_version": "0.5.9"},
                "failure": {
                    "enabled": True,
                    "backup_policy": "host",
                    "failed_dp_rank": 1,
                    "inject_after_request": 3,
                },
                "benchmark": {"request_rate": 4.0},
            }
        )
    )
    monkeypatch.setenv("SGLANG_SIMULATOR_CONFIG_PATH", str(config_path))
    ConfigManager.reset_config_cache()

    failure = ConfigManager.get_failure_config()

    assert failure.enabled is True
    assert failure.backup_policy.value == "host"
    assert failure.failed_dp_rank == 1
    assert ConfigManager.get_benchmark_config()["request_rate"] == 4.0
