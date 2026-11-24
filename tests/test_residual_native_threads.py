"""Independent controls for the thread identities used by the cleanup probes."""
from concurrent.futures import ThreadPoolExecutor
import ctypes
import errno
import os
from queue import Queue
import threading

import pytest

from _residual_native_threads import (
    _darwin_live_ids, live_native_ids, require_native_exit, wait_native_gone, worker_completion_probe,
)


def test_observes_worker_after_nonwaiting_shutdown():
    entered = Queue()
    release = threading.Event()

    def hold_worker():
        entered.put((threading.get_native_id(), worker_completion_probe(threading.current_thread())))
        return release.wait(15)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(hold_worker)
        try:
            worker, completed = entered.get(timeout=5)
            observer = threading.get_native_id()
            pool.shutdown(wait=False)
            assert worker != observer and not future.done()
            assert live_native_ids([worker, observer]) == {worker, observer}
            assert not completed()
            with pytest.raises(AssertionError, match="without completion proof"):
                require_native_exit([worker], [])
            with pytest.raises(RuntimeError, match="still running") as error:
                wait_native_gone([worker], timeout=0)
            assert str(worker) in str(error.value)
        finally:
            release.set()
    assert future.result(timeout=0) and completed()
    require_native_exit(live_native_ids([worker]), [worker])
    assert live_native_ids([worker, observer]) == {observer}


def test_completed_future_does_not_prove_its_worker_exited():
    def identify():
        return threading.get_native_id(), worker_completion_probe(threading.current_thread())

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(identify)
        worker, completed = future.result(timeout=5)
        assert future.done() and not completed()
        assert live_native_ids([worker]) == {worker}
        with pytest.raises(AssertionError, match="without completion proof"):
            require_native_exit([worker], [])
    assert completed()
    require_native_exit(live_native_ids([worker]), [worker])


def test_darwin_queries_64_bit_ids_and_recognizes_missing_threads():
    observer = threading.get_native_id()
    worker = 2**40 + 17
    missing = worker + 1
    queried = set()

    def read(pid, flavor, tid, buffer, size):
        assert pid == os.getpid() and flavor == 15
        assert size == ctypes.sizeof(buffer) == 112
        queried.add(tid)
        if tid == missing:
            ctypes.set_errno(errno.ESRCH)
            return 0
        return size

    assert _darwin_live_ids({worker, missing}, read) == {worker}
    assert queried == {observer, worker, missing}


@pytest.mark.parametrize(("count", "error", "exception"), [
    (0, errno.EPERM, OSError),
    (0, errno.ESRCH, OSError),  # The calling thread must be observable.
    (0, 0, RuntimeError),
    (8, 0, RuntimeError),
])
def test_darwin_refuses_unverified_observations(count, error, exception):
    def read(pid, flavor, tid, buffer, size):
        ctypes.set_errno(error)
        return count

    with pytest.raises(exception):
        _darwin_live_ids({threading.get_native_id()}, read)
