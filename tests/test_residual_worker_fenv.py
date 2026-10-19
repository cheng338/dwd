"""Caller floating-state admission remains per call with retained workers."""
import ctypes
import sys
import unittest
from unittest.mock import patch

import numpy as np

from dwd import _compiled_residual as compiled
from dwd import _native_residual as native
from dwd import _residual_worker_scope as workers


def _bits(values):
    return None if values is None else [value.tobytes() for value in values]


@unittest.skipUnless(sys.platform == 'win32', 'Windows CRT floating-state control required.')
@unittest.skipIf(compiled._ACCEL is None, 'Optional compiled residual extension unavailable.')
class ParentFloatingStateTests(unittest.TestCase):
    def test_admission_rechecks_parent_state_between_reused_worker_calls(self):
        if not native._native_supported():
            self.skipTest('Native residual runtime is not qualified.')
        control = ctypes.CDLL('ucrtbase')._controlfp_s
        control.argtypes = [ctypes.POINTER(ctypes.c_uint), ctypes.c_uint, ctypes.c_uint]
        control.restype = ctypes.c_int
        current = ctypes.c_uint()
        self.assertEqual(control(ctypes.byref(current), 0, 0), 0)
        saved = current.value
        mask = 0x03000300
        rng = np.random.default_rng(336)
        K = rng.normal(size=(65, 65))
        x, rhs = rng.normal(size=65), rng.normal(size=65)
        args = (K, x, rhs, .125, .25)
        native_args = (K, .125, rhs, x, .25, 0.)
        try:
            self.assertEqual(control(ctypes.byref(current), 0, mask), 0)
            with patch.object(compiled, '_worker_count', return_value=2), workers.ResidualWorkerScope() as scope:
                expected = _bits(compiled.compiled_values(*args))
                pool = scope.executor
                self.assertIsNotNone(expected)
                self.assertIsNotNone(pool)
                for label, value in [('nearest', 0), ('downward', 0x100),
                                     ('upward', 0x200), ('toward-zero', 0x300),
                                     ('flush-to-zero', 0x01000000)]:
                    with self.subTest(mode=label):
                        self.assertEqual(control(ctypes.byref(current), value, mask), 0)
                        if value:
                            with patch.object(native, '_dispatch_compiled_values',
                                              side_effect=AssertionError('Caller must decline before dispatch')) as dispatch:
                                self.assertIsNone(native.native_compensated_residual(*native_args))
                                dispatch.assert_not_called()
                        else:
                            with patch.object(native, '_dispatch_compiled_values',
                                              wraps=native._dispatch_compiled_values) as dispatch:
                                self.assertIsNotNone(native.native_compensated_residual(*native_args))
                                self.assertEqual(dispatch.call_count, 1)
                        self.assertEqual(control(ctypes.byref(current), 0, 0), 0)
                        self.assertEqual(current.value & mask, value)
                        self.assertIs(scope.executor, pool)
                        self.assertEqual(control(ctypes.byref(current), 0, mask), 0)
                self.assertEqual(_bits(compiled.compiled_values(*args)), expected)
            self.assertTrue(pool._shutdown)
            self.assertTrue(all(not thread.is_alive() for thread in pool._threads))
        finally:
            self.assertEqual(control(ctypes.byref(current), saved, mask), 0)


if __name__ == '__main__':
    unittest.main()
