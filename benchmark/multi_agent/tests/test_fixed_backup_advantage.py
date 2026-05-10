from __future__ import annotations

import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _write_experiment_dir(base: Path, *, job2_remote_reuse: int, job2_cached_tokens: int) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    (base / "job_summary.csv").write_text(
        "job_id,job,year,worker_id,status,job_completion_time_s,llm_calls,prefill_cached_tokens,total_tokens,error\n"
        "1,request_1,,1,completed,10.0,1,5,11000,\n"
        f"2,request_2,,2,completed,20.0,1,{job2_cached_tokens},11000,\n",
        encoding="utf-8",
    )
    (base / "task_summary.csv").write_text(
        "job_id,job,year,worker_id,job_status,agent,task_label,task_completion_time_s,llm_calls,prompt_tokens,cached_prompt_tokens,completion_tokens,total_tokens,prefill_cache_hit_rate,rid,error\n"
        "1,request_1,,1,completed,fixed_client,fixed_request,10.0,1,1000,5,100,1100,0.005,rid-1,\n"
        f"2,request_2,,2,completed,fixed_client,fixed_request,20.0,1,1000,{job2_cached_tokens},100,1100,0.8,rid-2,\n",
        encoding="utf-8",
    )
    (base / "cache_hits.csv").write_text(
        "job_id,task_label,agent_role,rid,l1_match,l2_match,remote_match,remote_prefetch,reused_device,reused_host,reused_storage,is_failover_retried,pre_failover_output_tokens,pre_failover_backed_up_tokens\n"
        "1,fixed_request,fixed_client,rid-1,5,0,0,0,5,0,0,,,\n"
        f"2,fixed_request,fixed_client,rid-2,5,0,{job2_remote_reuse},{job2_remote_reuse},5,0,{job2_remote_reuse},{'True' if job2_remote_reuse else ''},300,800\n",
        encoding="utf-8",
    )
    return base


def test_build_comparison_highlights_failover_advantage(tmp_path):
    baseline_dir = _write_experiment_dir(
        tmp_path / "baseline", job2_remote_reuse=0, job2_cached_tokens=5
    )
    remote_dir = _write_experiment_dir(
        tmp_path / "remote_backup", job2_remote_reuse=800, job2_cached_tokens=805
    )

    from plot.fixed_backup_advantage import build_comparison, load_experiment

    comparison = build_comparison(
        load_experiment(baseline_dir, label="Baseline"),
        load_experiment(remote_dir, label="Remote Backup"),
    )

    assert comparison.affected_job_ids == ["2"]
    assert comparison.by_job_id["2"].baseline.remote_reuse_tokens == 0
    assert comparison.by_job_id["2"].backup.remote_reuse_tokens == 800
    assert comparison.by_job_id["2"].backup.recomputed_tokens == 195


def test_main_writes_backup_advantage_figure(tmp_path, monkeypatch):
    baseline_dir = _write_experiment_dir(
        tmp_path / "baseline", job2_remote_reuse=0, job2_cached_tokens=5
    )
    remote_dir = _write_experiment_dir(
        tmp_path / "remote_backup", job2_remote_reuse=800, job2_cached_tokens=805
    )
    output = tmp_path / "figures" / "fixed_backup_advantage.png"

    from plot import fixed_backup_advantage

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fixed_backup_advantage.py",
            "--baseline-dir",
            str(baseline_dir),
            "--backup-dir",
            str(remote_dir),
            "--output",
            str(output),
        ],
    )

    fixed_backup_advantage.main()

    assert output.is_file()
