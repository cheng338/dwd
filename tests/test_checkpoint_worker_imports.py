"""Parallel checkpoint checks with optional solvers already loaded by the caller."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('solver_module', ['cvxpy', 'scs'])
def test_preloaded_solver_parallel_cv_matches_serial_and_replays(solver_module):
    if importlib.util.find_spec(solver_module) is None:
        pytest.skip(f'{solver_module} is an optional solver dependency')
    # Start with a fresh runtime fingerprint for each import order. The existing
    # end-to-end case compares fold scores, fitted coefficients and receipt replay.
    child = """
import importlib, json, sys, unittest
sys.path[:] = json.loads(sys.argv[1])
importlib.import_module(sys.argv[2])
suite = unittest.defaultTestLoader.loadTestsFromName(sys.argv[3])
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() and result.testsRun == 1 and not result.skipped else 1)
"""
    tests = str(Path(__file__).resolve().parent)
    case = ('test_resumable_kernel_cv.ResumableKernelCVTests.'
            'test_real_two_worker_execution_matches_serial_and_replays')
    run = subprocess.run(
        [sys.executable, '-B', '-c', child, json.dumps([tests, *sys.path]), solver_module, case],
        capture_output=True, text=True, timeout=300,
    )
    assert run.returncode == 0, run.stdout + run.stderr
