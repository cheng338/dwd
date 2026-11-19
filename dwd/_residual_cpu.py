"""Conservative CPU caps for residual threads; never change CPU placement."""
from functools import lru_cache
import os


def _positive_count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 1


@lru_cache(maxsize=1)
def _windows_affinity_functions():
    """Retain API bindings, not an affinity mask or a process CPU count."""
    import ctypes

    library = ctypes.WinDLL("kernel32.dll", use_last_error=True, winmode=0x00000800)
    current_process = library.GetCurrentProcess
    current_process.argtypes = ()
    current_process.restype = ctypes.c_void_p
    affinity_mask = library.GetProcessAffinityMask
    affinity_mask.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                             ctypes.POINTER(ctypes.c_size_t))
    affinity_mask.restype = ctypes.c_int
    return library, current_process, affinity_mask


def _windows_affinity_count():
    """Return a fresh primary-group mask count, or a conservative serial cap."""
    import ctypes

    _, current_process, affinity_mask = _windows_affinity_functions()
    process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
    if not affinity_mask(current_process(), ctypes.byref(process_mask), ctypes.byref(system_mask)):
        return 1
    process, system = process_mask.value, system_mask.value
    if not process or not system or process & system != process:
        return 1
    # On multi-group Windows systems this describes the calling thread's
    # primary group; it is not a count of every CPU available across groups.
    return process.bit_count()


def available_cpu_count():
    """Bound residual parallelism without claiming complete scheduler capacity.

    A failed supported affinity query selects one worker conservatively; it
    does not establish that the actual affinity mask contains one CPU. On
    platforms with no supported affinity API, retain the system-count cap.
    """
    try:
        system = _positive_count(os.cpu_count())
    except (OSError, NotImplementedError):
        return 1
    if system == 1:
        return 1
    try:
        affinity = getattr(os, "sched_getaffinity", None)
        if affinity is not None:
            mask = affinity(0)
            if (type(mask) not in (set, frozenset) or not mask
                    or any(type(cpu) is not int or cpu < 0 for cpu in mask)):
                return 1
            count = len(mask)
        elif os.name == "nt":
            count = _windows_affinity_count()
        else:
            return system
    except (OSError, NotImplementedError, AttributeError, TypeError, ValueError, ImportError):
        return 1
    return min(system, _positive_count(count))
