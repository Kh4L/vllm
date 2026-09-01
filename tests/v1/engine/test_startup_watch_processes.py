# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from multiprocessing import connection
from threading import Event
from types import SimpleNamespace

import pytest
import zmq

import vllm.platforms as platforms
from vllm.v1.engine import core as core_module
from vllm.v1.engine import utils as engine_utils
from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState
from vllm.v1.engine.utils import (
    CoreEngine,
    CoreEngineLaunch,
    CoreEngineProcManager,
    EngineZmqAddresses,
    wait_for_engine_startup,
)

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    ("is_rocm", "request_timeout", "manager_timeout", "process_timeout"),
    [
        (True, 0, 0, 15.0),
        (True, 0, 7, 7),
        (True, 0, None, None),
        (False, 0, 0, 0),
        (True, 7, 0, 0),
    ],
)
def test_engine_core_process_shutdown_timeout(
    monkeypatch: pytest.MonkeyPatch,
    is_rocm: bool,
    request_timeout: float | None,
    manager_timeout: float | None,
    process_timeout: float | None,
):
    manager = object.__new__(CoreEngineProcManager)
    manager._request_shutdown_timeout = request_timeout
    manager.manager_stopped = Event()
    manager.processes = [object()]
    detach_results = iter((object(), None))
    manager._finalizer = SimpleNamespace(detach=lambda: next(detach_results))

    shutdown_calls = []
    monkeypatch.setattr(
        engine_utils,
        "current_platform",
        SimpleNamespace(is_rocm=lambda: is_rocm),
    )
    monkeypatch.setattr(
        engine_utils,
        "shutdown",
        lambda processes, timeout: shutdown_calls.append((processes, timeout)),
    )

    manager.shutdown(timeout=manager_timeout)
    manager.shutdown(timeout=manager_timeout)

    assert manager.manager_stopped.is_set()
    assert shutdown_calls == [(manager.processes, process_timeout)]


@pytest.mark.parametrize(
    (
        "is_rocm",
        "shutdown_state",
        "has_work",
        "shutdown_timeout",
        "exit_code",
        "expected_calls",
    ),
    [
        (
            True,
            EngineShutdownState.SHUTTING_DOWN,
            False,
            0,
            None,
            ["shutdown", "freeze"],
        ),
        (False, EngineShutdownState.SHUTTING_DOWN, False, 0, None, ["shutdown"]),
        (True, EngineShutdownState.RUNNING, False, 0, None, ["shutdown"]),
        (True, EngineShutdownState.SHUTTING_DOWN, True, 0, None, ["shutdown"]),
        (True, EngineShutdownState.SHUTTING_DOWN, False, 7, None, ["shutdown"]),
        (True, EngineShutdownState.SHUTTING_DOWN, False, 0, 1, ["shutdown"]),
    ],
)
def test_freeze_gc_after_clean_rocm_engine_core_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    is_rocm: bool,
    shutdown_state: EngineShutdownState,
    has_work: bool,
    shutdown_timeout: int,
    exit_code: int | None,
    expected_calls: list[str],
):
    calls: list[str] = []
    vllm_config = SimpleNamespace(shutdown_timeout=shutdown_timeout)
    proc = SimpleNamespace(
        shutdown_state=EngineShutdownState.RUNNING,
        has_work=lambda: has_work,
        vllm_config=vllm_config,
    )

    def run_busy_loop():
        proc.shutdown_state = shutdown_state
        raise SystemExit(exit_code)

    proc.run_busy_loop = run_busy_loop
    proc.shutdown = lambda: calls.append("shutdown")
    parallel_config = SimpleNamespace(
        data_parallel_size=1,
        numa_bind=False,
        reconfigure_for_independent_dp_rank=lambda: None,
    )
    vllm_config.parallel_config = parallel_config

    for name in (
        "maybe_register_config_serialize_by_value",
        "set_process_title",
        "maybe_init_worker_tracer",
        "decorate_logs",
    ):
        monkeypatch.setattr(core_module, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(core_module, "EngineCoreProc", lambda *args, **kwargs: proc)
    monkeypatch.setattr(
        core_module,
        "SignalCallback",
        lambda callback: SimpleNamespace(trigger=lambda: None, stop=lambda: None),
    )
    monkeypatch.setattr(core_module.signal, "signal", lambda *args: None)
    monkeypatch.setattr(
        platforms, "current_platform", SimpleNamespace(is_rocm=lambda: is_rocm)
    )
    monkeypatch.setattr(core_module.gc, "freeze", lambda: calls.append("freeze"))

    with pytest.raises(SystemExit):
        EngineCoreProc.run_engine_core(vllm_config=vllm_config)

    assert calls == expected_calls


class _FinishedProcess:
    name = "RustFrontend"

    def __init__(self, sentinel):
        self.sentinel = sentinel

    @property
    def exitcode(self):
        return 1


def test_wait_for_engine_startup_reports_watched_process_exit():
    ctx = zmq.Context()
    handshake_socket = ctx.socket(zmq.ROUTER)
    recv, send = connection.Pipe(duplex=False)
    send.close()

    parallel_config = SimpleNamespace(
        data_parallel_size_local=1,
        data_parallel_hybrid_lb=False,
        data_parallel_external_lb=False,
    )

    try:
        launch = CoreEngineLaunch(
            engine_manager=None,
            coordinator=None,
            addresses=EngineZmqAddresses(inputs=[], outputs=[]),
            tensor_queue=None,
        )
        launch.watched_frontend_processes = [_FinishedProcess(recv)]
        with pytest.raises(RuntimeError) as exc_info:
            wait_for_engine_startup(
                handshake_socket,
                [CoreEngine()],
                parallel_config,  # type: ignore[arg-type]
                coordinated_dp=False,
                cache_config=None,  # type: ignore[arg-type]
                launch=launch,
            )
    finally:
        recv.close()
        handshake_socket.close(linger=0)
        ctx.term()

    assert "Frontend process failed during engine core initialization" in str(
        exc_info.value
    )
    assert "Failed frontend proc(s): {'RustFrontend': 1}" in str(exc_info.value)


def _replicate_handshake_parallel_config(replicate: bool) -> SimpleNamespace:
    return SimpleNamespace(
        data_parallel_size_local=1,
        data_parallel_hybrid_lb=False,
        data_parallel_external_lb=False,
        data_parallel_replicate_moe=replicate,
        data_parallel_master_ip="127.0.0.1",
        data_parallel_master_port=0,
        _data_parallel_master_port_list=[],
        data_parallel_size=2,
        compute_hash=lambda: "confighash",
    )


def _drive_handshake(
    *,
    coordinated_dp: bool,
    replicated_moe_dp: bool,
    hello_extra: dict,
    ready_extra: dict,
):
    """Run wait_for_engine_startup against a scripted fake engine and return
    the parallel_config dict the front-end sent in the init message."""
    import threading
    import uuid

    import msgspec

    ctx = zmq.Context()
    address = f"inproc://handshake-{uuid.uuid4()}"
    handshake_socket = ctx.socket(zmq.ROUTER)
    handshake_socket.bind(address)
    engine = CoreEngine()
    init_holder: dict = {}

    def fake_engine():
        sock = ctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.IDENTITY, engine.identity)
        sock.connect(address)
        base = {"status": "HELLO", "local": True, "headless": False}
        sock.send(msgspec.msgpack.encode(base | hello_extra))
        # The front-end only replies if the HELLO passed validation.
        if sock.poll(timeout=5000):
            init_holder["init"] = msgspec.msgpack.decode(sock.recv())
            base["status"] = "READY"
            sock.send(msgspec.msgpack.encode(base | ready_extra))
        sock.close(linger=0)

    thread = threading.Thread(target=fake_engine)
    thread.start()
    launch = CoreEngineLaunch(
        engine_manager=None,
        coordinator=None,
        addresses=EngineZmqAddresses(inputs=[], outputs=[]),
        tensor_queue=None,
    )
    try:
        wait_for_engine_startup(
            handshake_socket,
            [engine],
            _replicate_handshake_parallel_config(replicated_moe_dp),  # type: ignore[arg-type]
            coordinated_dp=coordinated_dp,
            cache_config=None,  # type: ignore[arg-type]
            launch=launch,
            replicated_moe_dp=replicated_moe_dp,
        )
    finally:
        thread.join(timeout=10)
        handshake_socket.close(linger=0)
        ctx.term()
    return init_holder.get("init", {}).get("parallel_config")


def test_wait_for_engine_startup_replicated_moe_success():
    """A replicated-MoE engine that reports the flag and the pre-localization
    config hash completes the handshake; the init payload carries only the
    replicate flag (no DP master info that would re-widen the engine)."""
    sent = _drive_handshake(
        coordinated_dp=False,
        replicated_moe_dp=True,
        hello_extra={"data_parallel_replicate_moe": True},
        ready_extra={"parallel_config_hash": "confighash"},
    )
    assert sent == {"data_parallel_replicate_moe": True}


def test_wait_for_engine_startup_replicated_rejects_lockstep_engine():
    """An engine without the replicate flag (e.g. a node launched without
    --data-parallel-replicate-moe) fails fast at HELLO time."""
    with pytest.raises(RuntimeError, match="data-parallel-replicate-moe"):
        _drive_handshake(
            coordinated_dp=False,
            replicated_moe_dp=True,
            hello_extra={},
            ready_extra={},
        )


def test_wait_for_engine_startup_coordinated_rejects_replicated_engine():
    with pytest.raises(RuntimeError, match="data-parallel-replicate-moe"):
        _drive_handshake(
            coordinated_dp=True,
            replicated_moe_dp=False,
            hello_extra={"data_parallel_replicate_moe": True},
            ready_extra={},
        )


def test_wait_for_engine_startup_coordinated_tolerates_missing_flag():
    """Colocated-front-end handshakes (external/hybrid LB) and engines of
    older vLLM versions send no flag; coordinated mode must accept that."""
    sent = _drive_handshake(
        coordinated_dp=True,
        replicated_moe_dp=False,
        hello_extra={},
        ready_extra={"parallel_config_hash": "confighash"},
    )
    assert sent is not None
    assert sent["data_parallel_size"] == 2
    assert sent["data_parallel_replicate_moe"] is False


def test_wait_for_engine_startup_replicated_hash_mismatch():
    """Cross-rank config divergence is caught via the pre-localization
    config hash in the READY message."""
    with pytest.raises(RuntimeError, match="Configuration mismatch"):
        _drive_handshake(
            coordinated_dp=False,
            replicated_moe_dp=True,
            hello_extra={"data_parallel_replicate_moe": True},
            ready_extra={"parallel_config_hash": "wronghash"},
        )


def _drive_child_handshake(child_parallel_config, init_parallel_config: dict):
    """Run EngineCoreProc.startup_handshake against a scripted front-end and
    return (hello_msg, result_addresses, exception)."""
    import threading
    import uuid

    import msgspec

    from vllm.v1.engine.utils import EngineHandshakeMetadata

    ctx = zmq.Context()
    address = f"inproc://child-handshake-{uuid.uuid4()}"
    frontend_socket = ctx.socket(zmq.ROUTER)
    frontend_socket.bind(address)
    holder: dict = {}

    def child():
        sock = ctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.IDENTITY, b"\x00\x00")
        sock.connect(address)
        try:
            holder["addresses"] = EngineCoreProc.startup_handshake(
                sock, True, False, child_parallel_config
            )
        except Exception as e:  # noqa: BLE001
            holder["exception"] = e
        finally:
            sock.close(linger=0)

    thread = threading.Thread(target=child)
    thread.start()
    try:
        identity, hello_bytes = frontend_socket.recv_multipart()
        holder["hello"] = msgspec.msgpack.decode(hello_bytes)
        init_message = msgspec.msgpack.encode(
            EngineHandshakeMetadata(
                addresses=EngineZmqAddresses(inputs=[], outputs=[]),
                parallel_config=init_parallel_config,
            )
        )
        frontend_socket.send_multipart((identity, init_message))
        thread.join(timeout=10)
    finally:
        frontend_socket.close(linger=0)
        ctx.term()
    return holder.get("hello"), holder.get("addresses"), holder.get("exception")


def test_startup_handshake_replicated_engine_echoes_flag_and_validates():
    """A localized replicated-MoE engine (already DP=1) echoes the replicate
    flag in HELLO and accepts a matching init message."""
    from vllm.config import ParallelConfig

    parallel_config = ParallelConfig(
        data_parallel_size=2,
        is_moe_model=True,
        data_parallel_replicate_moe=True,
    )
    parallel_config.reconfigure_for_independent_dp_rank()

    hello, addresses, exception = _drive_child_handshake(
        parallel_config, {"data_parallel_replicate_moe": True}
    )
    assert exception is None
    assert addresses is not None
    assert hello["data_parallel_replicate_moe"] is True
    assert parallel_config.data_parallel_size == 1


def test_startup_handshake_replicated_engine_rejects_lockstep_frontend():
    """A replicated engine under a lockstep front-end must fail fast instead
    of being silently re-widened to DP>1 by the init message."""
    from vllm.config import ParallelConfig

    parallel_config = ParallelConfig(
        data_parallel_size=2,
        is_moe_model=True,
        data_parallel_replicate_moe=True,
    )
    parallel_config.reconfigure_for_independent_dp_rank()

    hello, addresses, exception = _drive_child_handshake(
        parallel_config,
        {"data_parallel_replicate_moe": False, "data_parallel_size": 2},
    )
    assert isinstance(exception, RuntimeError)
    assert "data-parallel-replicate-moe" in str(exception)
    # The mismatched init message must not have been applied.
    assert parallel_config.data_parallel_size == 1


def test_startup_handshake_lockstep_engine_hello_reports_flag_off():
    """A coordinated MoE-DP engine reports the flag as False so the
    front-end can detect a replicated/lockstep mismatch in either
    direction."""
    from vllm.config import ParallelConfig

    parallel_config = ParallelConfig(data_parallel_size=2, is_moe_model=True)

    hello, addresses, exception = _drive_child_handshake(parallel_config, {})
    assert exception is None
    assert hello["data_parallel_replicate_moe"] is False


def test_startup_handshake_dense_engine_hello_unchanged():
    """Dense engines keep the pre-existing HELLO wire format (no flag)."""
    from vllm.config import ParallelConfig

    parallel_config = ParallelConfig(data_parallel_size=2, is_moe_model=False)
    parallel_config.reconfigure_for_independent_dp_rank()

    hello, addresses, exception = _drive_child_handshake(parallel_config, {})
    assert exception is None
    assert "data_parallel_replicate_moe" not in hello


@pytest.mark.parametrize(
    ("moe_spans_dp", "replicate_moe", "expected_class", "expect_pre_hash"),
    [
        # Lockstep MoE DP keeps the coordinated engine class.
        (True, False, "dp_moe", False),
        # Replicated MoE localizes to DP=1 and runs the independent engine,
        # shipping its pre-localization config hash for validation.
        (False, True, "independent", True),
        # Dense DP (pre-existing behavior).
        (False, False, "independent", False),
    ],
)
def test_run_engine_core_class_selection(
    monkeypatch: pytest.MonkeyPatch,
    moe_spans_dp: bool,
    replicate_moe: bool,
    expected_class: str,
    expect_pre_hash: bool,
):
    """run_engine_core selects DPEngineCoreProc only when the expert layers
    span the DP ranks; replicated-MoE ranks take the independent path with
    preserved global identity."""
    constructed: dict = {}
    reconfigure_calls: list[str] = []

    vllm_config = SimpleNamespace(shutdown_timeout=0, kv_transfer_config=None)

    def fake_reconfigure():
        reconfigure_calls.append("reconfigured")

    parallel_config = SimpleNamespace(
        data_parallel_size=2,
        numa_bind=False,
        moe_spans_dp=moe_spans_dp,
        data_parallel_replicate_moe=replicate_moe,
        # Order-sensitive: the hash shipped to the front-end must be computed
        # BEFORE the config is localized to DP=1.
        compute_hash=lambda: "posthash" if reconfigure_calls else "prehash",
        reconfigure_for_independent_dp_rank=fake_reconfigure,
    )
    vllm_config.parallel_config = parallel_config
    vllm_config.model_config = SimpleNamespace(is_moe=moe_spans_dp or replicate_moe)

    def make_proc(kind):
        def ctor(*args, **kwargs):
            constructed["kind"] = kind
            constructed["kwargs"] = kwargs
            proc = SimpleNamespace(
                shutdown_state=EngineShutdownState.RUNNING,
                has_work=lambda: False,
                vllm_config=vllm_config,
                shutdown=lambda: None,
            )

            def run_busy_loop():
                raise SystemExit(0)

            proc.run_busy_loop = run_busy_loop
            return proc

        return ctor

    for name in (
        "maybe_register_config_serialize_by_value",
        "set_process_title",
        "maybe_init_worker_tracer",
        "decorate_logs",
    ):
        monkeypatch.setattr(core_module, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(core_module, "EngineCoreProc", make_proc("independent"))
    monkeypatch.setattr(core_module, "DPEngineCoreProc", make_proc("dp_moe"))
    monkeypatch.setattr(
        core_module,
        "SignalCallback",
        lambda callback: SimpleNamespace(trigger=lambda: None, stop=lambda: None),
    )
    monkeypatch.setattr(core_module.signal, "signal", lambda *args: None)
    monkeypatch.setattr(
        platforms, "current_platform", SimpleNamespace(is_rocm=lambda: False)
    )

    with pytest.raises(SystemExit):
        EngineCoreProc.run_engine_core(
            vllm_config=vllm_config, dp_rank=1, local_dp_rank=1
        )

    assert constructed["kind"] == expected_class
    # Global identity is preserved for routing and metrics in all modes.
    assert parallel_config.data_parallel_index == 1
    if expected_class == "independent":
        assert reconfigure_calls == ["reconfigured"]
        assert constructed["kwargs"]["engine_index"] == 1
    else:
        assert not reconfigure_calls
        assert parallel_config.data_parallel_rank == 1
    if expect_pre_hash:
        assert constructed["kwargs"]["pre_localized_config_hash"] == "prehash"
    else:
        assert "pre_localized_config_hash" not in constructed["kwargs"]


@pytest.mark.parametrize(
    ("moe_spans_dp", "expected_wave_coordination"),
    [
        # Lockstep MoE DP: coordinator handles wave coordination.
        (True, True),
        # Independent DP ranks (dense or replicated MoE): stats-only.
        (False, False),
    ],
)
def test_launch_core_engines_wave_coordination_gate(
    monkeypatch: pytest.MonkeyPatch,
    moe_spans_dp: bool,
    expected_wave_coordination: bool,
):
    """launch_core_engines must construct the DPCoordinator with wave
    coordination gated on whether the expert layers span the DP ranks (the
    Ray branch is used to exercise the shared coordinator-construction code
    without spawning engine processes)."""
    captured: dict = {}

    class _StubCoordinator:
        def __init__(self, parallel_config, enable_wave_coordination=True):
            captured["enable_wave_coordination"] = enable_wave_coordination
            self.proc = SimpleNamespace(pid=1234)

        def get_engine_socket_addresses(self):
            return ("ipc://coord-in", "ipc://coord-out")

        def get_stats_publish_address(self):
            return "ipc://coord-stats"

    monkeypatch.setattr(engine_utils, "DPCoordinator", _StubCoordinator)
    monkeypatch.setattr(
        engine_utils,
        "CoreEngineActorManager",
        lambda **kwargs: SimpleNamespace(),
    )

    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=2,
            data_parallel_size_local=1,
            data_parallel_rank_local=None,
            data_parallel_rank=0,
            data_parallel_master_ip="127.0.0.1",
            local_engines_only=False,
            data_parallel_backend="ray",
            moe_spans_dp=moe_spans_dp,
            data_parallel_replicate_moe=False,
        ),
        model_config=SimpleNamespace(multimodal_config=None),
        needs_dp_coordinator=True,
    )

    with engine_utils.launch_core_engines(
        vllm_config,  # type: ignore[arg-type]
        executor_class=object,  # type: ignore[arg-type]
        log_stats=False,
        addresses=EngineZmqAddresses(inputs=[], outputs=[]),
    ) as launch:
        assert launch.coordinator is not None

    assert captured["enable_wave_coordination"] == expected_wave_coordination


def test_startup_handshake_replicated_engine_rejects_missing_flag():
    """A replicated engine must fail closed when the init message lacks the
    replicate flag (e.g. a front-end from an older vLLM version): applying
    its DP topology fields would silently re-widen the engine to DP>1."""
    from vllm.config import ParallelConfig

    parallel_config = ParallelConfig(
        data_parallel_size=2,
        is_moe_model=True,
        data_parallel_replicate_moe=True,
    )
    parallel_config.reconfigure_for_independent_dp_rank()

    hello, addresses, exception = _drive_child_handshake(
        parallel_config,
        {"data_parallel_size": 2, "data_parallel_master_ip": "127.0.0.1"},
    )
    assert isinstance(exception, RuntimeError)
    assert "data-parallel-replicate-moe" in str(exception)
    assert parallel_config.data_parallel_size == 1


def test_startup_handshake_replicated_engine_rejects_topology_fields():
    """Even with the flag set, a replicated engine rejects any DP topology
    field in the init message and stays localized to DP=1."""
    from vllm.config import ParallelConfig

    parallel_config = ParallelConfig(
        data_parallel_size=2,
        is_moe_model=True,
        data_parallel_replicate_moe=True,
    )
    parallel_config.reconfigure_for_independent_dp_rank()

    hello, addresses, exception = _drive_child_handshake(
        parallel_config,
        {"data_parallel_replicate_moe": True, "data_parallel_size": 2},
    )
    assert isinstance(exception, RuntimeError)
    assert "data_parallel_size" in str(exception)
    assert parallel_config.data_parallel_size == 1


def test_wait_for_engine_startup_uncoordinated_frontend_rejects_replicated_engine():
    """A front-end that is neither coordinated nor replicated (e.g. dense DP)
    must still reject an engine reporting the replicate flag."""
    with pytest.raises(RuntimeError, match="data-parallel-replicate-moe"):
        _drive_handshake(
            coordinated_dp=False,
            replicated_moe_dp=False,
            hello_extra={"data_parallel_replicate_moe": True},
            ready_extra={},
        )
