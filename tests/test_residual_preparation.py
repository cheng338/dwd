"""Prepared coefficients preserve arithmetic and clean up allocation failures.

The ordinary compiled arithmetic tests run without these test-only fixtures.
For allocator instrumentation set DWD_NATIVE_BASELINE to the frozen earlier
helper, DWD_TEST_NATIVE_ALLOCATOR to a counted-allocation helper, and
DWD_TEST_NATIVE_ALLOCFAIL to the same helper with forced malloc refusal.
These variables select test libraries; no production fault-injection API exists.
"""
import ctypes
import importlib.util
import os
import sys
import unittest

import numpy as np

from test_compiled_residual import _cases


_FIXTURES = ('DWD_NATIVE_BASELINE', 'DWD_TEST_NATIVE_ALLOCATOR',
             'DWD_TEST_NATIVE_ALLOCFAIL')


def _load(path, label):
    spec = importlib.util.spec_from_file_location(label + '._residual_accel', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    library = ctypes.CDLL(path)
    for name in ('test_alloc_calls', 'test_alloc_live'):
        if hasattr(library, name):
            getattr(library, name).restype = ctypes.c_size_t
    return module, library


def _evaluate(module, args, start=0, stop=None):
    K, x, rhs, shift, intercept = args
    stop = len(x) if stop is None else stop
    outputs = [np.full(stop-start, 42.) for _ in range(3)]
    status = module.evaluate(K, x, rhs, shift, intercept, start, stop, *outputs)
    return status, [values.tobytes() for values in outputs]


@unittest.skipUnless(all(os.environ.get(name) for name in _FIXTURES),
                     'Optional native allocation-test libraries unavailable.')
class PreparedCoefficientAllocationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline, _ = _load(os.environ[_FIXTURES[0]], '_prepared_reference')
        cls.prepared, cls.counter = _load(os.environ[_FIXTURES[1]], '_prepared_counted')
        cls.refused, cls.refusal_counter = _load(os.environ[_FIXTURES[2]], '_prepared_refused')

    def test_preparation_and_malloc_refusal_preserve_all_guarded_cases(self):
        for n, kind, *args in _cases():
            with self.subTest(n=n, case=kind):
                expected = _evaluate(self.baseline, args)
                for module, counter in ((self.prepared, self.counter),
                                        (self.refused, self.refusal_counter)):
                    before = counter.test_alloc_calls()
                    self.assertEqual(_evaluate(module, args), expected)
                    self.assertEqual(counter.test_alloc_calls(), before + 1)
                    self.assertEqual(counter.test_alloc_live(), 0)

    def test_domain_refusal_before_and_after_allocation_releases_storage(self):
        n = 9
        args = [np.ones((n, n)), np.ones(n), np.zeros(n), .125, .25]
        for position in range(6):
            with self.subTest(position=position):
                bad = [value.copy() if isinstance(value, np.ndarray) else value for value in args]
                if position == 0: bad[1][-1] = np.inf
                if position == 1: bad[2][0] = np.inf
                if position == 2: bad[2][-1] = np.inf
                if position == 3: bad[0][0, 0] = np.inf
                if position == 4: bad[0][-1, -1] = np.inf
                if position == 5: bad[3] = np.inf
                before = self.counter.test_alloc_calls()
                actual = _evaluate(self.prepared, bad)
                self.assertEqual(actual, _evaluate(self.baseline, bad))
                self.assertEqual(actual[0], 2)
                self.assertEqual(self.counter.test_alloc_calls() - before,
                                 0 if position in (0, 5) else 1)
                self.assertEqual(self.counter.test_alloc_live(), 0)

    def test_empty_interval_needs_no_preparation(self):
        args = [np.ones((9, 9)), np.ones(9), np.zeros(9), .125, .25]
        before = self.counter.test_alloc_calls()
        actual = _evaluate(self.prepared, args, start=3, stop=3)
        self.assertEqual(actual, _evaluate(self.baseline, args, start=3, stop=3))
        self.assertEqual(self.counter.test_alloc_calls(), before)
        self.assertEqual(self.counter.test_alloc_live(), 0)

    @unittest.skipUnless(sys.platform == 'win32', 'Windows CRT floating-state control required.')
    def test_unsupported_rounding_and_underflow_decline_without_allocation(self):
        control = ctypes.CDLL('ucrtbase')._controlfp_s
        control.argtypes = [ctypes.POINTER(ctypes.c_uint), ctypes.c_uint, ctypes.c_uint]
        control.restype = ctypes.c_int
        current = ctypes.c_uint()
        self.assertEqual(control(ctypes.byref(current), 0, 0), 0)
        saved = current.value
        args = [np.ones((9, 9)), np.ones(9), np.zeros(9), .125, .25]
        for label, value, mask in [('down', 0x100, 0x300), ('up', 0x200, 0x300),
                                   ('zero', 0x300, 0x300), ('FTZ', 0x01000000, 0x03000000)]:
            with self.subTest(mode=label):
                try:
                    self.assertEqual(control(ctypes.byref(current), value, mask), 0)
                    before = self.counter.test_alloc_calls()
                    actual = _evaluate(self.prepared, args)
                    self.assertEqual(actual, _evaluate(self.baseline, args))
                    self.assertEqual(actual[0], 4)
                    self.assertEqual(self.counter.test_alloc_calls(), before)
                    self.assertEqual(self.counter.test_alloc_live(), 0)
                finally:
                    self.assertEqual(control(ctypes.byref(current), saved, mask), 0)


if __name__ == '__main__':
    unittest.main()
