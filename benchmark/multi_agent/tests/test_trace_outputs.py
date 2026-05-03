from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _trace_record(job_id: str = "1") -> dict:
    return {
        "topic_id": job_id,
        "topic": "test topic",
        "year": "2026",
        "worker_id": "1",
        "status": "completed",
        "error": None,
        "agent_timings": [],
        "summary": {
            "wall_time_s": 1.0,
            "llm_calls": 0,
            "prompt_tokens": 0,
            "prefill_cached_tokens": 0,
            "total_tokens": 0,
        },
    }


class _FakeMetric:
    def __init__(self, record: dict) -> None:
        self._record = record

    def to_trace_record(self) -> dict:
        return self._record


def test_crewai_worker_writes_only_final_summary_csvs(tmp_path, monkeypatch):
    jobs_csv = tmp_path / "jobs.csv"
    jobs_csv.write_text("id,topic,year\n1,test topic,2026\n", encoding="utf-8")
    out_dir = tmp_path / "out"

    fake_core = types.ModuleType("client.crewai_core")

    class FakeRunner:
        def __init__(self, cfg) -> None:
            self.cfg = cfg

        def run_all_jobs(self) -> list[_FakeMetric]:
            return [_FakeMetric(_trace_record())]

    fake_core.CrewAIRunner = FakeRunner
    monkeypatch.setitem(sys.modules, "client.crewai_core", fake_core)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "crewai_worker.py",
            "--server-url",
            "http://127.0.0.1:28000",
            "--model-path",
            "model",
            "--jobs-csv",
            str(jobs_csv),
            "--output-dir",
            str(out_dir),
        ],
    )

    from client import crewai_worker

    assert crewai_worker.main() == 0

    assert not (out_dir / "trace_log.json").exists()
    assert not (out_dir / "events.jsonl").exists()
    assert (out_dir / "cache_hits.csv").is_file()
    assert (out_dir / "job_summary.csv").is_file()
    assert (out_dir / "task_summary.csv").is_file()


def test_pipeline_uses_config_named_result_dir_and_clears_existing(tmp_path):
    from main import ExperimentPipeline

    stale = tmp_path / "results" / "remote_backup" / "stale.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("old data", encoding="utf-8")

    pipeline = ExperimentPipeline(
        server_cfg=SimpleNamespace(),
        fi_cfg=SimpleNamespace(),
        crewai_cfg=SimpleNamespace(),
        kv_backup="remote_backup",
        log_dir=tmp_path / "logs",
        output_dir=tmp_path / "results",
        timestamp="20260101_000000",
        venv=tmp_path / ".venv",
        deploy_config_name="remote_backup",
    )

    assert pipeline.output_dir == tmp_path / "results" / "remote_backup"
    assert not stale.exists()
    assert (pipeline.output_dir / "figures").is_dir()


def test_pipeline_generates_profiles_from_summary_csvs(tmp_path):
    from main import ExperimentPipeline

    out_dir = tmp_path / "results" / "baseline"
    figures = out_dir / "figures"
    figures.mkdir(parents=True)
    job_summary = out_dir / "job_summary.csv"
    job_summary.write_text(
        "job_id,job,year,worker_id,status,job_completion_time_s,llm_calls,"
        "prefill_cached_tokens,total_tokens,error\n"
        "1,test topic,2026,1,completed,1.0,0,0,0,\n",
        encoding="utf-8",
    )
    task_summary = out_dir / "task_summary.csv"
    task_summary.write_text(
        "job_id,job,year,worker_id,job_status,agent,task_label,"
        "task_completion_time_s,llm_calls,prompt_tokens,cached_prompt_tokens,"
        "completion_tokens,total_tokens,prefill_cache_hit_rate,error\n"
        "1,test topic,2026,1,completed,Planning Coordinator,Phase 1 / Planning,"
        "1.0,1,10,0,5,15,0.0,\n",
        encoding="utf-8",
    )
    (out_dir / "cache_hits.csv").write_text(
        "job_id,task_label,agent_role,rid,l1_match,l2_match,remote_match,"
        "remote_prefetch,reused_device,reused_host,reused_storage,"
        "is_failover_retried,pre_failover_output_tokens,"
        "pre_failover_backed_up_tokens\n",
        encoding="utf-8",
    )
    calls: list[tuple[str, Path, tuple[str, ...]]] = []

    class PipelineForTest(ExperimentPipeline):
        def _run_plot(self, script_name: str, output: Path, *extra_args: str) -> None:
            calls.append((script_name, output, extra_args))

    pipeline = PipelineForTest(
        server_cfg=SimpleNamespace(),
        fi_cfg=SimpleNamespace(),
        crewai_cfg=SimpleNamespace(),
        kv_backup="remote",
        log_dir=tmp_path / "logs",
        output_dir=tmp_path / "results",
        timestamp="20260101_000000",
        venv=tmp_path / ".venv",
        deploy_config_name="baseline",
    )

    # ExperimentPipeline clears the run directory during initialization.
    figures.mkdir(parents=True, exist_ok=True)
    job_summary.write_text(
        "job_id,job,year,worker_id,status,job_completion_time_s,llm_calls,"
        "prefill_cached_tokens,total_tokens,error\n"
        "1,test topic,2026,1,completed,1.0,0,0,0,\n",
        encoding="utf-8",
    )
    task_summary.write_text(
        "job_id,job,year,worker_id,job_status,agent,task_label,"
        "task_completion_time_s,llm_calls,prompt_tokens,cached_prompt_tokens,"
        "completion_tokens,total_tokens,prefill_cache_hit_rate,error\n"
        "1,test topic,2026,1,completed,Planning Coordinator,Phase 1 / Planning,"
        "1.0,1,10,0,5,15,0.0,\n",
        encoding="utf-8",
    )
    (out_dir / "cache_hits.csv").write_text(
        "job_id,task_label,agent_role,rid,l1_match,l2_match,remote_match,"
        "remote_prefetch,reused_device,reused_host,reused_storage,"
        "is_failover_retried,pre_failover_output_tokens,"
        "pre_failover_backed_up_tokens\n",
        encoding="utf-8",
    )

    pipeline._generate_plots(out_dir)

    assert (
        "trace_log_profile.py",
        figures / "trace_log_profile.png",
        ("--job-summary", str(job_summary), "--task-summary", str(task_summary)),
    ) in calls
    assert (
        "backup_cache_hits.py",
        figures / "backup_cache_hit_profile.png",
        ("--cache-csv", str(out_dir / "cache_hits.csv")),
    ) in calls


def test_pipeline_copies_logs_after_child_processes_exit(tmp_path, monkeypatch):
    import main as main_mod
    from main import ExperimentPipeline

    class FakeServerConfig(SimpleNamespace):
        def topology_tag(self) -> str:
            return "DP2_PP1_TP1_N1"

        def build_deploy_flags(self, **_kwargs) -> list[str]:
            return []

    class FakeProc:
        def __init__(self, name: str, pid: int) -> None:
            self.name = name
            self.pid = pid
            self.returncode = None
            self.stdout = []

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

        def terminate(self) -> None:
            pass

        def kill(self) -> None:
            self.returncode = -9

    jobs_csv = tmp_path / "jobs.csv"
    jobs_csv.write_text("id,topic,year\n1,test,2026\n", encoding="utf-8")

    procs = [
        FakeProc("server", 101),
        FakeProc("injector", 102),
        FakeProc("client", 103),
    ]
    proc_iter = iter(procs)
    copy_states: list[dict[str, object]] = []

    def fake_popen(*_args, **_kwargs):
        return next(proc_iter)

    def fake_copy_process_logs(_log_dir, _dest_dir):
        copy_states.append({proc.name: proc.poll() for proc in procs})
        return []

    monkeypatch.setattr(main_mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(main_mod, "copy_process_logs", fake_copy_process_logs)

    pipeline = ExperimentPipeline(
        server_cfg=FakeServerConfig(model_path="model"),
        fi_cfg=SimpleNamespace(
            after_job="",
            after_task="",
            timeline_after_start_s="",
            delay=0,
            dp_rank="1",
            pp_rank="0",
            tp_rank="0",
            method="api",
            recovery_delay=0,
            recover_after_job="",
            recover_after_task="",
            inject_match_task_label="",
            inject_match_agent_role="",
            inject_match_worker_id="",
        ),
        crewai_cfg=SimpleNamespace(
            jobs_csv=str(jobs_csv),
            job_limit=1,
            app_workers=1,
            default_year=2026,
            short_max_tokens=16,
            long_max_tokens=16,
            enable_stream=False,
            ignore_eos=True,
            worker_start_stagger_s=0,
            agent_dp_rank_map={},
            extra_instructions_path=None,
        ),
        kv_backup="remote_backup",
        log_dir=tmp_path / "logs",
        output_dir=tmp_path / "results",
        timestamp="20260101_000000",
        venv=tmp_path / ".venv",
        deploy_config_name="remote_backup",
    )

    def fake_wait_for_event(_proc, _events, event_name: str, timeout_s: int):
        if event_name == "server_ready":
            return {"manifest": {"server_url": "http://127.0.0.1:28000"}}
        if event_name == "injector_ready":
            return {"control_url": "http://127.0.0.1:40000"}
        raise AssertionError(event_name)

    monkeypatch.setattr(pipeline, "_start_log_reader", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "_wait_for_event", fake_wait_for_event)

    assert pipeline.run() == 0
    assert copy_states == [{"server": 0, "injector": 0, "client": 0}]


def test_task_summary_writes_task_rid_column(tmp_path):
    from client.metrics_utils import write_task_summary_records

    csv_path = tmp_path / "task_summary.csv"
    records = [
        {
            **_trace_record(),
            "agent_timings": [
                {
                    "agent": "Planning Coordinator",
                    "task_label": "Phase 1 / Planning",
                    "duration_s": 1.0,
                    "llm_calls": 1,
                    "prompt_tokens": 10,
                    "cached_prompt_tokens": 2,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "prefill_cache_hit_rate": 0.2,
                    "rid": "rid-plan-1",
                }
            ],
        }
    ]

    write_task_summary_records(records, csv_path)

    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert (
        lines[0]
        == "job_id,job,year,worker_id,job_status,agent,task_label,"
        "task_completion_time_s,llm_calls,prompt_tokens,cached_prompt_tokens,"
        "completion_tokens,total_tokens,prefill_cache_hit_rate,rid,error"
    )
    assert lines[1].endswith(",0.2,rid-plan-1,")


def test_cache_hit_final_row_uses_real_task_rid():
    from client.config import CrewAIRunnerConfig
    from client.crewai_core import CrewAIRunner, TaskMetrics

    runner = CrewAIRunner.__new__(CrewAIRunner)
    runner.cfg = CrewAIRunnerConfig(
        server_url="http://127.0.0.1:28000",
        model_path="model",
        jobs_csv=Path("/tmp/jobs.csv"),
        output_dir=Path("/tmp/out"),
    )
    runner._cache_rows = []
    runner._cache_csv_lock = __import__("threading").Lock()

    task_metrics = TaskMetrics(
        agent_role="Report Synthesizer",
        task_label="Phase 3 / Synthesis",
    )
    task_metrics.request_rids.append("rid-synth-1")
    task_metrics.cache_l1_match = 49

    runner._append_cache_csv_task_final(job_id="1", tm=task_metrics)

    assert runner._cache_rows == [
        {
            "job_id": "1",
            "task_label": "Phase 3 / Synthesis",
            "agent_role": "Report Synthesizer",
            "rid": "rid-synth-1",
            "l1_match": 49,
            "l2_match": 0,
            "remote_match": 0,
            "remote_prefetch": 0,
            "reused_device": 0,
            "reused_host": 0,
            "reused_storage": 0,
            "is_failover_retried": "",
            "pre_failover_output_tokens": "",
            "pre_failover_backed_up_tokens": "",
        }
    ]
