from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "benchmark" / "multi_agent" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def test_daai_slurm_uses_fixed_log_root_without_creating_subdirs():
    slurm = (
        REPO_ROOT
        / "benchmark"
        / "multi_agent"
        / "scripts"
        / "daai"
        / "run_experiment_qwen3_8b.slurm"
    )
    text = slurm.read_text(encoding="utf-8")

    assert "#SBATCH --output=/dev/null" in text
    assert "#SBATCH --error=/dev/null" in text
    assert 'LOG_DIR="${REPO_ROOT}/benchmark/multi_agent/logs"' in text
    assert 'mkdir -p "${LOG_DIR}"' in text
    assert '--log-dir "${LOG_DIR}"' in text


def test_process_log_paths_are_job_scoped_under_one_log_root(tmp_path, monkeypatch):
    from process_logs import PROCESS_LOG_NAMES, get_process_log_paths

    monkeypatch.setenv("SLURM_JOB_ID", "12345")

    paths = get_process_log_paths(tmp_path)

    assert set(paths) == set(PROCESS_LOG_NAMES)
    assert {path.parent for path in paths.values()} == {tmp_path}
    assert {path.name for path in paths.values()} == {
        "main_12345.log",
        "server_12345.log",
        "backup_server_12345.log",
        "injector_12345.log",
        "client_12345.log",
    }


def test_process_log_path_allows_explicit_env_override(tmp_path, monkeypatch):
    from process_logs import get_process_log_path

    override = tmp_path / "custom-server.log"
    monkeypatch.setenv("MULTI_AGENT_SERVER_LOG", str(override))
    monkeypatch.setenv("SLURM_JOB_ID", "12345")

    assert get_process_log_path(tmp_path, "server") == override
