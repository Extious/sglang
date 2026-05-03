from __future__ import annotations

import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def test_crewai_requests_force_minimum_temperature(tmp_path):
    from client.config import CrewAIRunnerConfig
    from client.crewai_core import CrewAIRunner

    cfg = CrewAIRunnerConfig(
        server_url="http://127.0.0.1:28000",
        model_path="Qwen/Qwen3-8B",
        jobs_csv=tmp_path / "topics.csv",
        output_dir=tmp_path,
    )
    runner = CrewAIRunner(cfg)
    runner._crewai_cfg["llm"]["temperature"] = 0.7

    llm = runner._build_llm(
        runner._crewai_cfg,
        max_tokens=128,
        agent_role="Report Synthesizer",
    )

    assert llm.temperature == 0.0
