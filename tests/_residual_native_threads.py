"""Observe the caller's native thread IDs independently of executor bookkeeping."""
import ctypes
import errno
import os
import sys
import threading
import time

import psutil


if sys.platform == "darwin":
    # libproc.h: int proc_pidinfo(int, int, uint64_t, void *, int).
    _proc_pidinfo = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True).proc_pidinfo
    _proc_pidinfo.argtypes = (
        ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int,
    )
    _proc_pidinfo.restype = ctypes.c_int


def _darwin_live_ids(wanted, proc_pidinfo):
    # <sys/proc_info.h>: PROC_PIDTHREADID64INFO looks up the same uint64_t ID
    # returned by pthread_threadid_np (CPython's get_native_id on macOS).
    thread_id_64_info = 15
    # struct proc_threadinfo: two uint64_t, eight int32_t, and a 64-byte name.
    info = ctypes.create_string_buffer(112)
    size = ctypes.sizeof(info)
    current = threading.get_native_id()
    pid = os.getpid()
    live = set()
    for tid in wanted | {current}:
        ctypes.set_errno(0)
        count = proc_pidinfo(pid, thread_id_64_info, tid, info, size)
        if count == size:
            live.add(tid)
        elif count == 0:
            error = ctypes.get_errno()
            if error == 0:
                raise RuntimeError(f"proc_pidinfo returned zero without errno for thread {tid}")
            if error != errno.ESRCH or tid == current:
                raise OSError(error, f"Could not verify native thread {tid}: {os.strerror(error)}")
        else:
            raise RuntimeError(f"Unexpected proc_threadinfo size: {count} (expected {size})")
    return live & wanted


def live_native_ids(thread_ids):
    """Return the requested native IDs currently present in this process."""
    wanted = set(thread_ids)
    if sys.platform == "darwin":
        return _darwin_live_ids(wanted, _proc_pidinfo)
    return wanted.intersection(thread.id for thread in psutil.Process().threads())


def wait_native_gone(thread_ids, timeout=5):
    """Wait for the given native workers to leave the process."""
    wanted = set(thread_ids)
    deadline = time.monotonic() + timeout
    while remaining := live_native_ids(wanted):
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Owned native workers still running: {sorted(remaining)}")
        time.sleep(0.001)


def worker_completion_probe(worker):
    """Capture CPython's completion token while this exact worker is running."""
    sentinel = getattr(worker, "_tstate_lock", None)
    if sentinel is not None:
        assert sentinel.locked(), "Expected a running worker's thread-state lock"
        return lambda: not sentinel.locked()
    for name in ("_os_thread_handle", "_handle"):
        handle = getattr(worker, name, None)
        if handle is not None:
            assert handle.ident == worker.ident and not handle.is_done()
            return handle.is_done
    raise AssertionError("Cannot identify the running worker's completion token")


def require_native_exit(observed_ids, completed_ids):
    """Allow time for OS retirement only for workers proved complete at the snapshot."""
    observed = set(observed_ids)
    unproved = observed.difference(completed_ids)
    assert not unproved, f"Native workers still present without completion proof: {sorted(unproved)}"
    if observed:
        wait_native_gone(observed)
