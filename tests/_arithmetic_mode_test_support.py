"""Existing audited arithmetic controls for native correctness tests."""
from contextlib import contextmanager
import ctypes
import platform
import sys
import pytest


@contextmanager
def changed_arithmetic(mode):
    # Exact existing platform controls from tests/test_ordered_reduction_core.py.
    if sys.platform == 'win32':
        control = ctypes.CDLL('ucrtbase')._controlfp_s
        control.argtypes = [ctypes.POINTER(ctypes.c_uint), ctypes.c_uint, ctypes.c_uint]
        control.restype = ctypes.c_int
        state = ctypes.c_uint()
        assert control(ctypes.byref(state), 0, 0) == 0
        saved = state.value
        value, mask = ((0x100, 0x300) if mode == 'down' else (0x01000000, 0x03000000))
        enter = lambda: control(ctypes.byref(state), value, mask)
        leave = lambda: control(ctypes.byref(state), saved, mask)
        def unchanged():
            assert control(ctypes.byref(state), 0, 0) == 0
            return state.value & mask == value
    elif sys.platform == 'linux' and platform.machine().lower() in ('x86_64', 'amd64') and mode == 'down':
        library = ctypes.CDLL(None)
        get, set_mode = library.fegetround, library.fesetround
        get.argtypes, get.restype = [], ctypes.c_int
        set_mode.argtypes, set_mode.restype = [ctypes.c_int], ctypes.c_int
        saved = get()
        assert saved >= 0
        enter, leave = lambda: set_mode(0x400), lambda: set_mode(saved)
        unchanged = lambda: get() == 0x400
    else:
        pytest.skip('No existing audited control for this platform/mode')
    try:
        assert enter() == 0
        yield unchanged
    finally:
        assert leave() == 0
