from __future__ import annotations

import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _write_experiment_dir(
    base: Path,
    *,
    job2_remote_reuse: int,
    job2_cached_tokens: int,
    job2_time_s: float,
) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    (base / "job_summary.csv").write_text(
        "job_id,job,year,worker_id,status,job_completion_time_s,llm_calls,prefill_cached_tokens,total_tokens,error\n"
        "1,request_1,,1,completed,10.0,1,5,11000,\n"
        f"2,request_2,,2,completed,{job2_time_s},1,{job2_cached_tokens},11000,\n",
        encoding="utf-8",
    )
    (base / "task_summary.csv").write_text(
        "job_id,job,year,worker_id,job_status,agent,task_label,task_completion_time_s,llm_calls,prompt_tokens,cached_prompt_tokens,completion_tokens,total_tokens,prefill_cache_hit_rate,rid,error\n"
        "1,request_1,,1,completed,fixed_client,fixed_request,10.0,1,10000,5,1000,11000,0.0005,rid-1,\n"
        f"2,request_2,,2,completed,fixed_client,fixed_request,{job2_time_s},1,10000,{job2_cached_tokens},1000,11000,0.8,rid-2,\n",
        encoding="utf-8",
    )
    (base / "cache_hits.csv").write_text(
        "job_id,task_label,agent_role,rid,l1_match,l2_match,remote_match,remote_prefetch,reused_device,reused_host,reused_storage,is_failover_retried,pre_failover_output_tokens,pre_failover_backed_up_tokens\n"
        "1,fixed_request,fixed_client,rid-1,5,0,0,0,5,0,0,,,\n"
        f"2,fixed_request,fixed_client,rid-2,5,0,{job2_remote_reuse},{job2_remote_reuse},5,0,{job2_remote_reuse},{'True' if job2_remote_reuse else ''},300,800\n",
        encoding="utf-8",
    )
    return base


def test_load_records_for_fixed_mode_groups_affected_jobs(tmp_path):
    baseline_dir = _write_experiment_dir(
        tmp_path / "baseline", job2_remote_reuse=0, job2_cached_tokens=5, job2_time_s=20.0
    )
    backup_dir = _write_experiment_dir(
        tmp_path / "remote_backup",
        job2_remote_reuse=8000,
        job2_cached_tokens=8005,
        job2_time_s=16.0,
    )

    from plot.fixed_backup_advantage_paper import load_records

    request_records = load_records(
        baseline_dir=baseline_dir,
        backup_dir=backup_dir,
        context_label="4090",
        baseline_label="Baseline",
        backup_label="Remote Backup",
    )

    assert len(request_records) == 4
    affected_requests = [
        record for record in request_records if record.exposure == "Failure-affected"
    ]
    assert len(affected_requests) == 1
    assert affected_requests[0].strategy == "Remote Backup"
    assert affected_requests[0].remote_reuse_tokens == 8000
    assert affected_requests[0].recomputed_tokens == 1995


def test_main_writes_all_fixed_paper_figures(tmp_path, monkeypatch):
    baseline_dir = _write_experiment_dir(
        tmp_path / "baseline", job2_remote_reuse=0, job2_cached_tokens=5, job2_time_s=20.0
    )
    backup_dir = _write_experiment_dir(
        tmp_path / "remote_backup",
        job2_remote_reuse=8000,
        job2_cached_tokens=8005,
        job2_time_s=16.0,
    )
    output_prefix = tmp_path / "figures" / "fixed_paper"

    from plot import fixed_backup_advantage_paper

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fixed_backup_advantage_paper.py",
            "--baseline-dir",
            str(baseline_dir),
            "--backup-dir",
            str(backup_dir),
            "--context-label",
            "4090",
            "--output-prefix",
            str(output_prefix),
        ],
    )

    fixed_backup_advantage_paper.main()

    assert output_prefix.parent.joinpath("fixed_paper_request_latency.png").is_file()
    assert output_prefix.parent.joinpath("fixed_paper_prefill.png").is_file()
    assert not output_prefix.parent.joinpath("fixed_paper_job_latency.png").exists()
