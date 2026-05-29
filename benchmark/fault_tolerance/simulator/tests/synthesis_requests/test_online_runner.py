import os
import sys
import subprocess
from pathlib import Path
import json
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT)]
SIMULATOR_SRC = ROOT / "tools" / "sglang-simulator" / "src"
SGLANG_PYTHON = ROOT / "python"

CONFIG_DIR = (
    ROOT
    / "benchmark/fault_tolerance/simulator/synthesis_requests/configs/synthesis-a100"
)


class DummyTokenizer:
    vocab_size = 100


class FakeAutoTokenizer:
    calls = []

    @classmethod
    def from_pretrained(cls, model_path):
        cls.calls.append(model_path)
        return DummyTokenizer()


def _synthetic_request():
    from benchmark.fault_tolerance.simulator.synthesis_requests.common.schema import (
        BackupPolicy,
        SyntheticRequest,
    )

    return SyntheticRequest(
        job_id="7",
        worker_id="2",
        worker_seq=3,
        assigned_dp_rank=1,
        token_ids=[11, 12, 13],
        output_length=5,
        created_time=4.5,
        backup_policy=BackupPolicy.HOST,
    )


def test_build_online_server_command_and_env(tmp_path, monkeypatch):
    from benchmark.fault_tolerance.simulator.synthesis_requests.common.config import (
        load_experiment_suite,
    )
    from benchmark.fault_tolerance.simulator.synthesis_requests.online.server import (
        build_server_command,
        build_server_env,
    )

    suite = load_experiment_suite(CONFIG_DIR)
    sim_config = tmp_path / "simulator_config.json"
    raw_dir = tmp_path / "simulator_raw"
    monkeypatch.setenv("SGLANG_USE_CPU_ENGINE", "1")
    monkeypatch.delenv("FLASHINFER_DISABLE_VERSION_CHECK", raising=False)

    command = build_server_command(suite=suite, sim_config_path=sim_config)
    env = build_server_env(sim_config_path=sim_config, simulator_raw_dir=raw_dir)

    assert command[:3] == [
        sys.executable,
        "-u",
        "-m",
    ]
    assert command[:4] == [
        sys.executable,
        "-u",
        "-m",
        "sglang_simulator.simulation.sglang.launch_server",
    ]
    assert command == [
        sys.executable,
        "-u",
        "-m",
        "sglang_simulator.simulation.sglang.launch_server",
        "--model-path",
        suite.server.model_path,
        "--load-format",
        "dummy",
        "--host",
        suite.server.host,
        "--port",
        str(suite.server.port),
        "--tp-size",
        str(suite.server.tp_size),
        "--pp-size",
        str(suite.server.pp_size),
        "--dp-size",
        str(suite.server.dp_size),
        "--enable-hierarchical-cache",
        "--hicache-storage-backend",
        "file",
        "--skip-server-warmup",
        "--skip-tokenizer-init",
        "--sim-config-path",
        str(sim_config),
    ]
    assert env["SGLANG_SIMULATOR_CONFIG_PATH"] == str(sim_config)
    assert env["SGLANG_SIMULATOR_OUTPUT_DIR"] == str(raw_dir)
    assert env["SGLANG_SIMULATOR_OUTPUT_MODE"] == "BLOCKING"
    assert env["SGLANG_SIMULATOR_SKIP_REMOTE_QUANT_CONFIG"] == "1"
    assert env["SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION"] == "0"
    assert env["SGLANG_SIMULATOR_STACK_DUMP_INTERVAL"] == "0"
    assert env["FLASHINFER_DISABLE_VERSION_CHECK"] == "1"
    pythonpath_entries = env["PYTHONPATH"].split(os.pathsep)
    assert pythonpath_entries[:3] == [
        str(SIMULATOR_SRC),
        str(SGLANG_PYTHON),
        str(ROOT),
    ]
    assert "SGLANG_USE_CPU_ENGINE" not in env
    assert os.environ["SGLANG_USE_CPU_ENGINE"] == "1"


def test_start_server_uses_command_and_env(tmp_path, monkeypatch):
    from benchmark.fault_tolerance.simulator.synthesis_requests.common.config import (
        load_experiment_suite,
    )
    from benchmark.fault_tolerance.simulator.synthesis_requests.online import server

    suite = load_experiment_suite(CONFIG_DIR)
    sim_config = tmp_path / "simulator_config.json"
    raw_dir = tmp_path / "simulator_raw"
    calls = []

    class FakeProcess:
        pass

    def fake_popen(command, env, stdout=None, stderr=None, text=None):
        calls.append((command, env, stdout, stderr, text))
        return FakeProcess()

    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)

    process = server.start_server(
        suite=suite,
        sim_config_path=sim_config,
        simulator_raw_dir=raw_dir,
    )

    assert isinstance(process, FakeProcess)
    command, env, stdout, stderr, text = calls[0]
    assert command == server.build_server_command(
        suite=suite,
        sim_config_path=sim_config,
    )
    assert env == server.build_server_env(
        sim_config_path=sim_config,
        simulator_raw_dir=raw_dir,
    )
    assert stdout is not None
    assert stderr is subprocess.STDOUT
    assert text is True
    assert (raw_dir / "server.log").is_file()


def test_raise_if_server_exited_reports_log_tail(tmp_path):
    from benchmark.fault_tolerance.simulator.synthesis_requests.online import run_one

    log_path = tmp_path / "server.log"
    log_path.write_text("line1\nline2\nline3\n", encoding="utf-8")

    class FakeProcess:
        def poll(self):
            return 17

    try:
        run_one.raise_if_server_exited(FakeProcess(), log_path=log_path)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected RuntimeError")

    assert "exited before readiness" in message
    assert "exit code 17" in message
    assert "line1" in message
    assert "line3" in message


def test_run_online_experiment_checks_server_exit_while_waiting(
    tmp_path, monkeypatch
):
    from benchmark.fault_tolerance.simulator.synthesis_requests.online import run_one

    FakeAutoTokenizer.calls = []
    calls = []

    class FakeProcess:
        def __init__(self):
            self.terminated = False

        def poll(self):
            return 9

        def terminate(self):
            self.terminated = True
            calls.append(("terminate",))

        def wait(self, timeout=None):
            calls.append(("wait", timeout))

    fake_process = FakeProcess()

    def fake_start_server(*, suite, sim_config_path, simulator_raw_dir):
        calls.append(("start_server",))
        simulator_raw_dir.mkdir(parents=True, exist_ok=True)
        (simulator_raw_dir / "server.log").write_text(
            "scheduler failed\n",
            encoding="utf-8",
        )
        return fake_process

    def fake_wait_for_server_ready(*, host, port, timeout_s, process, log_path):
        calls.append(("wait_ready", process is fake_process, log_path.name, timeout_s))
        run_one.raise_if_server_exited(process, log_path=log_path)

    monkeypatch.setattr(run_one, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(run_one, "start_server", fake_start_server)
    monkeypatch.setattr(run_one, "wait_for_server_ready", fake_wait_for_server_ready)

    try:
        run_one.run_online_experiment(CONFIG_DIR, "baseline", tmp_path / "out")
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected RuntimeError")

    assert ("wait_ready", True, "server.log", 900.0) in calls
    assert "scheduler failed" in message
    assert fake_process.terminated is False


def test_wait_for_server_ready_stops_when_process_exits(tmp_path, monkeypatch):
    from benchmark.fault_tolerance.simulator.synthesis_requests.online import client

    log_path = tmp_path / "server.log"
    log_path.write_text("server traceback\n", encoding="utf-8")
    sleeps = []

    class FakeProcess:
        def poll(self):
            return 12

    class FakeHttpClient:
        @staticmethod
        def get(url, timeout):
            raise client.requests_module.ConnectionError("not listening")

    def fail_if_sleep(delay):
        sleeps.append(delay)
        raise AssertionError("should not sleep after process exit")

    monkeypatch.setattr(client.time, "sleep", fail_if_sleep)

    try:
        client.wait_for_server_ready(
            "127.0.0.1",
            30000,
            timeout_s=300,
            process=FakeProcess(),
            log_path=log_path,
            http_client=FakeHttpClient,
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected RuntimeError")

    assert "exited before readiness" in message
    assert "exit code 12" in message
    assert "server traceback" in message
    assert sleeps == []


def test_build_generate_url():
    from benchmark.fault_tolerance.simulator.synthesis_requests.online.client import (
        build_generate_url,
        build_profile_url,
    )

    assert build_generate_url("127.0.0.1", "30000") == "http://127.0.0.1:30000/generate"
    assert build_profile_url("localhost", 30001) == "http://localhost:30001/start_profile"


def test_run_online_client_sends_all_requests():
    from benchmark.fault_tolerance.simulator.synthesis_requests.online.client import (
        run_online_client,
    )
    from benchmark.fault_tolerance.simulator.synthesis_requests.common.payload import (
        build_http_generate_payload,
    )

    requests = [_synthetic_request(), _synthetic_request()]
    calls = []

    class FakeResponse:
        def __init__(self, index):
            self.index = index
            self.raised = False

        def raise_for_status(self):
            self.raised = True

        def json(self):
            assert self.raised is True
            return {"index": self.index}

    class FakeHttpClient:
        @staticmethod
        def post(url, json, timeout):
            calls.append((url, json, timeout))
            return FakeResponse(len(calls))

    results = run_online_client(
        "127.0.0.1",
        "30000",
        requests,
        total_request=9,
        http_client=FakeHttpClient,
    )

    assert results == [{"index": 1}, {"index": 2}]
    assert calls == [
        (
            "http://127.0.0.1:30000/generate",
            build_http_generate_payload(requests[0], total_request=9),
            3600,
        ),
        (
            "http://127.0.0.1:30000/generate",
            build_http_generate_payload(requests[1], total_request=9),
            3600,
        ),
    ]


def test_run_online_client_uses_workers_and_arrival_times(monkeypatch):
    from benchmark.fault_tolerance.simulator.synthesis_requests.common.schema import (
        BackupPolicy,
        SyntheticRequest,
    )
    from benchmark.fault_tolerance.simulator.synthesis_requests.online import client

    requests = [
        SyntheticRequest(
            job_id="1",
            worker_id="1",
            worker_seq=0,
            assigned_dp_rank=0,
            token_ids=[1],
            output_length=1,
            created_time=0.0,
            backup_policy=BackupPolicy.NONE,
        ),
        SyntheticRequest(
            job_id="2",
            worker_id="2",
            worker_seq=0,
            assigned_dp_rank=1,
            token_ids=[2],
            output_length=1,
            created_time=0.2,
            backup_policy=BackupPolicy.NONE,
        ),
        SyntheticRequest(
            job_id="3",
            worker_id="1",
            worker_seq=1,
            assigned_dp_rank=0,
            token_ids=[3],
            output_length=1,
            created_time=0.5,
            backup_policy=BackupPolicy.NONE,
        ),
    ]
    sleeps = []
    posts = []
    now = {"value": 100.0}
    executors = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True}

    class FakeExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers
            self.tasks = []
            executors.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args):
            future = SimpleNamespace(result=lambda: fn(*args))
            self.tasks.append((fn, args, future))
            return future

    def fake_sleep(delay):
        sleeps.append(round(delay, 3))
        now["value"] += delay

    def fake_post(url, json, timeout):
        posts.append(
            (
                json["sampling_params"]["custom_params"]["simulation"]["worker_id"],
                json["sampling_params"]["custom_params"]["simulation"]["job_id"],
                timeout,
            )
        )
        return FakeResponse()

    monkeypatch.setattr(client.time, "monotonic", lambda: now["value"])
    monkeypatch.setattr(client.time, "sleep", fake_sleep)
    monkeypatch.setattr(client, "ThreadPoolExecutor", FakeExecutor)

    results = client.run_online_client(
        "127.0.0.1",
        30000,
        requests,
        total_request=3,
        app_workers=4,
        request_rate=2.0,
        http_client=SimpleNamespace(post=fake_post),
    )

    assert results == [{"ok": True}, {"ok": True}, {"ok": True}]
    assert executors[0].max_workers == 4
    assert posts == [("1", "1", 3600), ("1", "3", 3600), ("2", "2", 3600)]
    assert sleeps == [0.5]


def test_run_online_experiment_writes_config_and_stops_server(
    tmp_path, monkeypatch
):
    from benchmark.fault_tolerance.simulator.synthesis_requests.online import run_one

    FakeAutoTokenizer.calls = []
    calls = []

    class FakeProcess:
        def __init__(self):
            self.terminated = False
            self.killed = False
            self.wait_timeouts = []

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True
            calls.append(("terminate",))

        def wait(self, timeout=None):
            self.wait_timeouts.append(timeout)
            calls.append(("wait", timeout))

        def kill(self):
            self.killed = True
            calls.append(("kill",))

    fake_process = FakeProcess()

    def fake_start_server(*, suite, sim_config_path, simulator_raw_dir):
        calls.append(("start_server", suite.server.host, suite.server.port, sim_config_path))
        raw_path = simulator_raw_dir / "request.jsonl"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(
            '{"rid":"r1","job_id":"1","worker_id":"0","assigned_dp_rank":0,'
            '"backup_policy":"host","status":"completed","created_time":1,'
            '"finish_time":2}\n',
            encoding="utf-8",
        )
        return fake_process

    def fake_wait_for_server_ready(*, host, port, timeout_s, process, log_path):
        calls.append(
            ("wait_ready", host, port, process is fake_process, log_path.name, timeout_s)
        )

    def fake_run_online_client(
        host, port, requests, total_request, app_workers, request_rate
    ):
        calls.append(
            (
                "run_client",
                host,
                port,
                len(requests),
                total_request,
                app_workers,
                request_rate,
            )
        )
        assert len(requests) == 32
        assert total_request == 32
        assert app_workers == 4
        assert request_rate == float("inf")
        return [{"ok": True} for _ in requests]

    def fake_trigger_profile(host, port):
        calls.append(("profile", host, port))

    monkeypatch.delenv("SGLANG_USE_CPU_ENGINE", raising=False)
    monkeypatch.setattr(run_one, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(run_one, "start_server", fake_start_server)
    monkeypatch.setattr(run_one, "wait_for_server_ready", fake_wait_for_server_ready)
    monkeypatch.setattr(run_one, "run_online_client", fake_run_online_client)
    monkeypatch.setattr(run_one, "trigger_profile", fake_trigger_profile)

    result = run_one.run_online_experiment(
        CONFIG_DIR,
        "host_backup",
        tmp_path / "out",
    )

    simulator_config = json.loads(
        (tmp_path / "out" / "simulator_config.json").read_text(encoding="utf-8")
    )
    assert simulator_config["benchmark"]["execution_mode"] == "online"
    assert simulator_config["failure"]["backup_policy"] == "host"
    assert result.strategy.value == "host_backup"
    assert result.output_dir == tmp_path / "out"
    assert result.metrics == {"client_completed": 32}
    assert (tmp_path / "out" / "request_detail.csv").is_file()
    assert (tmp_path / "out" / "simulator_request.jsonl").read_text(
        encoding="utf-8"
    ).strip()
    assert FakeAutoTokenizer.calls == ["Qwen/Qwen3-8B"]
    assert ("wait_ready", "127.0.0.1", 30000, True, "server.log", 900.0) in calls
    assert ("run_client", "127.0.0.1", 30000, 32, 32, 4, float("inf")) in calls
    assert ("profile", "127.0.0.1", 30000) in calls
    assert fake_process.terminated is True
    assert fake_process.killed is False
    assert fake_process.wait_timeouts == [30]
    assert "SGLANG_USE_CPU_ENGINE" not in os.environ
