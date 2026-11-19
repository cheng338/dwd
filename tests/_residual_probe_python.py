"""Select Python for test probes whose process identity the parent must verify."""
import os
import sys


def probe_python():
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    executable = sys.executable
    if os.name == "nt":
        base = getattr(sys, "_base_executable", None)
        if base and os.path.normcase(base) != os.path.normcase(executable):
            # CPython's multiprocessing launcher preserves the venv this way
            # while starting the interpreter directly, with no redirector PID.
            environment["__PYVENV_LAUNCHER__"] = executable
            executable = base
    return executable, environment
