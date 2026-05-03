from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


PYTHON_ROOT = Path(__file__).resolve().parents[3] / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

for package_name, package_path in (
    ("sglang", PYTHON_ROOT / "sglang"),
    ("sglang.srt", PYTHON_ROOT / "sglang" / "srt"),
):
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_path)]
    sys.modules.setdefault(package_name, package)

if "tqdm" not in sys.modules:
    tqdm_stub = types.ModuleType("tqdm")
    tqdm_stub.tqdm = lambda iterable=None, *args, **kwargs: iterable
    sys.modules["tqdm"] = tqdm_stub


def _stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _install_dpc_dependency_stubs() -> None:
    if "sglang.srt.managers.data_parallel_controller" in sys.modules:
        return

    _stub_module(
        "psutil",
        Process=lambda *args, **kwargs: SimpleNamespace(
            parent=lambda: SimpleNamespace(send_signal=lambda signal: None)
        ),
    )
    _stub_module("setproctitle", setproctitle=lambda *args, **kwargs: None)
    _stub_module("zmq", ZMQError=Exception, NOBLOCK=1)
    _stub_module("sglang.srt.environ", envs=SimpleNamespace())
    _stub_module(
        "sglang.srt.layers.dp_attention",
        compute_dp_attention_world_info=lambda *args, **kwargs: None,
    )
    _stub_module(
        "sglang.srt.managers.io_struct",
        ActiveRanksOutput=type("ActiveRanksOutput", (), {}),
        BatchTokenizedEmbeddingReqInput=type("BatchTokenizedEmbeddingReqInput", (), {}),
        BatchTokenizedGenerateReqInput=type("BatchTokenizedGenerateReqInput", (), {}),
        BlockReqInput=type("BlockReqInput", (), {}),
        ProfileReq=type("ProfileReq", (), {}),
        SimulateGpuFailureReqInput=type("SimulateGpuFailureReqInput", (), {}),
        SimulateGpuRecoveryReqInput=type("SimulateGpuRecoveryReqInput", (), {}),
        TokenizedEmbeddingReqInput=type("TokenizedEmbeddingReqInput", (), {}),
        TokenizedGenerateReqInput=type("TokenizedGenerateReqInput", (), {}),
        WatchLoadUpdateReq=type("WatchLoadUpdateReq", (), {}),
    )
    _stub_module("sglang.srt.managers.schedule_batch", Req=type("Req", (), {}))
    _stub_module(
        "sglang.srt.managers.scheduler",
        run_scheduler_process=lambda *args, **kwargs: None,
    )
    _stub_module(
        "sglang.srt.observability.cpu_monitor",
        start_cpu_monitor_thread=lambda *args, **kwargs: None,
    )

    class _DPControllerReqTimeStats:
        @classmethod
        def new_from_obj(cls, obj):
            return obj

    _stub_module(
        "sglang.srt.observability.req_time_stats",
        DPControllerReqTimeStats=_DPControllerReqTimeStats,
    )
    _stub_module(
        "sglang.srt.observability.trace",
        process_tracing_init=lambda *args, **kwargs: None,
        trace_set_thread_info=lambda *args, **kwargs: None,
    )
    _stub_module(
        "sglang.srt.server_args",
        DP_ATTENTION_HANDSHAKE_PORT_DELTA=0,
        PortArgs=type("PortArgs", (), {}),
        ServerArgs=type("ServerArgs", (), {}),
    )
    _stub_module("sglang.srt.utils.numa_utils")
    _stub_module(
        "sglang.srt.utils.common",
        configure_logger=lambda *args, **kwargs: None,
        kill_itself_when_parent_died=lambda *args, **kwargs: None,
        maybe_reindex_device_id=lambda *args, **kwargs: None,
    )
    _stub_module(
        "sglang.srt.utils.network",
        NetworkAddress=type("NetworkAddress", (), {}),
        bind_port=lambda *args, **kwargs: None,
        get_zmq_socket=lambda *args, **kwargs: None,
        get_zmq_socket_on_host=lambda *args, **kwargs: None,
    )
    _stub_module(
        "sglang.srt.utils.torch_memory_saver_adapter",
        TorchMemorySaverAdapter=type("TorchMemorySaverAdapter", (), {}),
    )
    _stub_module(
        "sglang.srt.utils.watchdog",
        Watchdog=type("Watchdog", (), {"__init__": lambda self, *args, **kwargs: None}),
    )

    class _TypeBasedDispatcher:
        def __init__(self, handlers):
            self.handlers = handlers

    _stub_module(
        "sglang.utils",
        TypeBasedDispatcher=_TypeBasedDispatcher,
        get_exception_traceback=lambda: "",
    )


class _Worker:
    def __init__(self) -> None:
        self.sent = []

    def send_pyobj(self, obj) -> None:
        self.sent.append(obj)


def _controller(*, failed=(), total_tokens=(100, 0), total_requests=(0, 0)):
    _install_dpc_dependency_stubs()
    from sglang.srt.managers.data_parallel_controller import (
        DPBudget,
        DataParallelController,
    )

    controller = DataParallelController.__new__(DataParallelController)
    controller.failed_dp_ranks = set(failed)
    controller.workers = [_Worker(), _Worker()]
    controller.status = [True, True]
    controller.round_robin_counter = 0
    controller.dp_budget = DPBudget(2)
    controller.dp_budget.total_tokens = list(total_tokens)
    controller.dp_budget.total_requests = list(total_requests)
    return controller


class FailoverRoutingTest(unittest.TestCase):
    def test_total_tokens_dispatch_avoids_failover_source_rank(self):
        controller = _controller(failed=(), total_tokens=(100, 0))
        req = SimpleNamespace(routed_dp_rank=None, failed_dp_rank=1)

        controller.total_tokens_scheduler(req)

        self.assertEqual(controller.workers[0].sent, [req])
        self.assertEqual(controller.workers[1].sent, [])

    def test_direct_route_to_failed_rank_is_rebalanced(self):
        controller = _controller(failed={1}, total_tokens=(100, 0))
        req = SimpleNamespace(routed_dp_rank=1, failed_dp_rank=None)

        controller.total_tokens_scheduler(req)

        self.assertIsNone(req.routed_dp_rank)
        self.assertEqual(controller.workers[0].sent, [req])
        self.assertEqual(controller.workers[1].sent, [])

    def test_dispatch_does_not_fall_back_to_excluded_rank_when_all_excluded(self):
        _install_dpc_dependency_stubs()
        from sglang.srt.managers.data_parallel_controller import (
            DPBudget,
            LoadBalanceMethod,
        )

        budget = DPBudget(2)

        self.assertIsNone(
            budget.dispatch(LoadBalanceMethod.TOTAL_TOKENS, exclude={0, 1})
        )


if __name__ == "__main__":
    unittest.main()
