"""Focused real-thread startup faults, isolated and observed with psutil.

The trace injects at the qualified post-launch line; it does not replace
Thread.start, bootstrap, or the low-level launcher. Native arithmetic actually
runs before the worker is held until the dispatcher's cancellation event.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import psutil
import pytest


def test_supported_cpython_startup_capability():
    from dwd import _residual_startup_cleanup as cleanup

    if sys.implementation.name == "cpython" and (3, 11) <= sys.version_info[:2] <= (3, 14):
        assert cleanup._qualified_wait_line() is not None


@pytest.mark.parametrize("mode", ["per-call", "reused", "registered", "repeated"])
def test_interrupted_native_start_drains_owned_worker_process(mode):
    from dwd import _compiled_residual as compiled
    from dwd import _residual_startup_cleanup as cleanup

    if cleanup._qualified_wait_line() is None:
        pytest.skip("Runtime does not expose a qualified CPython startup capability")
    if compiled._ACCEL is None:
        pytest.skip("Optional native residual extension is unavailable")
    # Windows venv redirectors spawn another process. Use the same interpreter's
    # base executable so Popen owns the process whose native threads we inspect;
    # the explicit module paths below preserve this test environment's packages.
    executable = getattr(sys, "_base_executable", None) or sys.executable
    command = [executable, "-B", str(Path(__file__).resolve()), "--startup-control", mode]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([
        str(Path(compiled.__file__).resolve().parent.parent),
        *[str(Path(p).resolve()) for p in sys.path if p and Path(p).is_dir()]
    ])
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, env=environment,
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    observed = psutil.Process(child.pid)
    creation = observed.create_time()
    try:
        stdout, stderr = child.communicate("go\n", timeout=30)
    except BaseException:
        # This timeout belongs only to this isolated fault-control process.
        # Production numerical cleanup never terminates an active worker.
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=10)
        raise
    assert child.wait() == child.returncode == 0, (stdout, stderr)
    started, result = [json.loads(line) for line in stdout.splitlines()]
    assert started == {"pid": child.pid, "creation_time": creation}
    assert not observed.is_running()
    assert result["primary_preserved"] is True
    assert result["cancelled_before_return"] is True
    assert result["native_calls"] >= (2 if mode == "registered" else 1)
    assert result["omitted_registered"] is False
    assert result["owned_native_threads_remaining"] == []
    assert result["foreign_thread_survived"] is True
    assert result["scope_retired"] is True
    assert result["deferred_interrupts"] == (2 if mode == "repeated" else 0)
    assert "'qualified': True" in result["cleanup_note"]
    assert "'drained': True" in result["cleanup_note"]


def test_nonstartup_error_declines_without_claiming_drain():
    from concurrent.futures import ThreadPoolExecutor
    from dwd import _residual_startup_cleanup as cleanup

    with ThreadPoolExecutor(max_workers=1) as pool:
        error = KeyboardInterrupt("outside Thread.start")
        result = cleanup.drain_started_before_registration(pool, error)
        assert result["qualified"] is False
        assert result["drained"] is False
        assert pool.submit(lambda: 42).result(timeout=5) == 42


@pytest.mark.parametrize("unsupported", ["runtime", "source", "bootstrap", "launcher"])
def test_unqualified_runtime_declines(monkeypatch, unsupported):
    import types
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from dwd import _residual_startup_cleanup as cleanup

    if unsupported == "runtime":
        monkeypatch.setattr(cleanup, "sys", types.SimpleNamespace(
            implementation=types.SimpleNamespace(name="pypy")))
    elif unsupported == "source":
        def missing_source(*args):
            raise OSError("runtime source unavailable")
        monkeypatch.setattr(cleanup.inspect, "getsourcelines", missing_source)
    elif unsupported == "bootstrap":
        monkeypatch.setattr(threading.Thread, "_bootstrap", lambda self: None)
    else:
        alias = "_start_joinable_thread" if cleanup._HANDLE is not None else "_start_new_thread"
        monkeypatch.setattr(threading, alias, lambda *args, **kwargs: None)
    assert cleanup._qualified_wait_line() is None
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = cleanup.drain_started_before_registration(pool, KeyboardInterrupt("primary"))
        assert result["qualified"] is False
        assert result["drained"] is False
        assert not pool._shutdown


@pytest.mark.parametrize("failure_stage", ["cleanup", "context"])
def test_cleanup_failure_keeps_primary_exception(monkeypatch, failure_stage):
    from contextlib import contextmanager
    import types
    import numpy as np
    from dwd import _compiled_residual as compiled

    primary = KeyboardInterrupt("primary startup failure")
    secondary = RuntimeError(failure_stage + " failure")

    class Pool:
        def submit(self, *args):
            raise primary

    @contextmanager
    def executor(*args):
        try:
            yield Pool()
        finally:
            if failure_stage == "context":
                raise secondary

    def cleanup(pool, error):
        assert error is primary
        frame = error.__traceback__
        while frame.tb_frame.f_code is not compiled._compiled_values_or_refusal.__code__:
            frame = frame.tb_next
        assert frame.tb_frame.f_locals["cancelled"].is_set()
        if failure_stage == "cleanup":
            raise secondary
        return {"qualified": False, "drained": False}

    monkeypatch.setattr(compiled, "_ACCEL", types.SimpleNamespace(evaluate=lambda *args: 0))
    monkeypatch.setattr(compiled, "_worker_count", lambda n: 2)
    monkeypatch.setattr(compiled, "executor_for_call", executor)
    monkeypatch.setattr(compiled, "drain_started_before_registration", cleanup)
    with pytest.raises(KeyboardInterrupt) as caught:
        compiled.compiled_values(np.eye(8), np.ones(8), np.zeros(8), .125, .3)
    assert caught.value is primary
    assert primary.__cause__ is secondary
    assert any("drain is not established" in note for note in primary.__notes__)


def _startup_control(mode):
    import concurrent.futures.thread as impl
    from contextlib import nullcontext
    import threading
    import types
    import numpy as np
    from dwd import _compiled_residual as compiled
    from dwd import _residual_startup_cleanup as cleanup
    from dwd._residual_worker_scope import ResidualWorkerScope

    print(json.dumps({"pid": os.getpid(), "creation_time": psutil.Process().create_time()}), flush=True)
    assert sys.stdin.readline() == "go\n"
    native = compiled._ACCEL.evaluate
    cancelled = threading.Event()
    entered = [threading.Event(), threading.Event()]
    finished = []
    foreign_release = threading.Event()
    foreign = threading.Thread(target=foreign_release.wait, name="unrelated-startup-control")
    foreign.start()
    foreign_id = foreign.native_id
    primary = KeyboardInterrupt("exact post-launch interruption")
    omitted = []
    owned_ids = []
    starts = []
    native_calls = []
    wait_line = cleanup._qualified_wait_line()
    assert wait_line is not None

    def evaluate(*args):
        result = native(*args)
        part = 0 if args[5] == 0 else 1
        native_calls.append(part)
        owned_ids.append(threading.get_native_id())
        entered[part].set()
        assert cancelled.wait(10), "dispatcher did not publish cancellation"
        finished.append(part)
        return result

    # Hold an actual native evaluation at a deterministic cancellation boundary.
    # The trusted wrapper also exercises the solver-local reusable-pool route.
    owner = types.SimpleNamespace(evaluate=evaluate)
    compiled._ACCEL = compiled._TRUSTED_ACCEL = owner
    compiled._TRUSTED_EVALUATE = evaluate
    compiled._worker_count = lambda n: 2
    compiled.Event = lambda: cancelled

    def trace(frame, event, arg):
        if frame.f_code is cleanup._START.__code__ and event == "line" and frame.f_lineno == wait_line:
            worker = frame.f_locals["self"]
            args = getattr(worker, "_args", ())
            if getattr(worker, "_target", None) is impl._worker and len(args) in (3, 4):
                pool = args[0]()
                starts.append(worker)
                number = len(starts) - 1
                assert entered[number].wait(5), "native worker did not enter"
                if mode != "registered" or number == 1:
                    assert worker not in pool._threads
                    assert worker not in impl._threads_queues
                    omitted.append((worker, pool))
                    raise primary
        return trace

    original_wait = cleanup._wait_completions
    repetitions = []
    if mode == "repeated":
        def repeated_wait(mechanism, tokens):
            if len(repetitions) < 2:
                repetitions.append(True)
                raise KeyboardInterrupt("additional cleanup interruption")
            return original_wait(mechanism, tokens)
        cleanup._wait_completions = repeated_wait
    scope = ResidualWorkerScope() if mode == "reused" else None
    previous_trace = sys.gettrace()
    caught = None
    try:
        with scope if scope is not None else nullcontext():
            sys.settrace(trace)
            try:
                compiled.compiled_values(np.eye(65), np.ones(65), np.zeros(65), .125, .3)
            except KeyboardInterrupt as error:
                caught = error
            finally:
                sys.settrace(previous_trace)
        assert caught is primary
        assert len(omitted) == 1
        worker, pool = omitted[0]
        assert pool._shutdown
        assert sorted(finished) == sorted(native_calls)
        current_ids = {row.id for row in psutil.Process().threads()}
        result = {"primary_preserved": caught is primary,
                  "cancelled_before_return": cancelled.is_set(),
                  "native_calls": len(native_calls),
                  "omitted_registered": worker in pool._threads or worker in impl._threads_queues,
                  "owned_native_threads_remaining": sorted(set(owned_ids) & current_ids),
                  "foreign_thread_survived": foreign.is_alive() and foreign_id in current_ids,
                  "scope_retired": scope is None or scope.executor is None,
                  "deferred_interrupts": len(repetitions),
                  "cleanup_note": next(n for n in primary.__notes__ if n.startswith("Interrupted-start recovery:"))}
    finally:
        sys.settrace(previous_trace)
        cancelled.set()
        foreign_release.set()
        foreign.join(5)
        cleanup._wait_completions = original_wait
        for worker, pool in omitted:
            pool.shutdown(wait=True)
            worker.join(5)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    assert sys.argv[1] == "--startup-control"
    _startup_control(sys.argv[2])
