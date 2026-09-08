"""Optional compiled arithmetic, guards, fallback, and thread allocation.

Core tests run independently of the Python native-screen version guard. On
runtimes without the audited math.sumprod, exact rational bounds provide the
reference instead; the whole numerical module is never skipped for Python 3.11.
"""
from concurrent.futures import ThreadPoolExecutor
import ctypes
from fractions import Fraction
import math
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

import dwd._compiled_residual as compiled
import dwd._native_residual as native


UNAVAILABLE = 'Optional compiled residual extension unavailable.'


def _cases():
    rng = np.random.default_rng(909183)
    for n in (1, 2, 3, 7, 17, 64):
        for kind in ('random', 'near_constant_cancelled', 'tiny', 'huge',
                     'broad_exponents', 'signed_zero', 'strided', 'fortran'):
            K, x, rhs = rng.normal(size=(n, n)), rng.normal(size=n), rng.normal(size=n)
            shift, intercept = .125, .3
            if kind == 'near_constant_cancelled':
                K = 1. + rng.normal(scale=1e-7, size=(n, n))
                x *= 1e8
                x[-1] = -math.fsum(x[:-1].tolist())
                shift = 1e-9
            elif kind == 'tiny':
                K.fill(2.**-200)
                x.fill(2.**-200)
            elif kind == 'huge':
                K.fill(2.**200)
                x.fill(2.**200)
            elif kind == 'broad_exponents':
                K = np.ldexp(rng.choice([-.5, .5], size=(n, n)),
                             rng.integers(-199, 201, size=(n, n)))
                x = np.ldexp(rng.choice([-.5, .5], size=n),
                             rng.integers(-199, 201, size=n))
            elif kind == 'signed_zero':
                K.fill(-0.)
                x.fill(-0.)
                rhs.fill(-0.)
                shift, intercept = -0., -0.
            elif kind == 'strided':
                K = rng.normal(size=(2*n, 2*n))[::-2, ::2]
                x = rng.normal(size=2*n)[::-2]
                rhs = rng.normal(size=2*n)[::2]
            elif kind == 'fortran':
                K = np.asfortranarray(K)
            yield n, kind, K, x, rhs, shift, intercept


def _rows(K, x, rhs, shift, intercept, start=0, stop=None):
    stop = len(x) if stop is None else stop
    outputs = [np.empty(stop-start) for _ in range(3)]
    status = compiled._ACCEL.evaluate(K, x, rhs, shift, intercept, start, stop, *outputs)
    return (status, *outputs)


def _exact(value):
    return Fraction.from_float(float(value))


class CompiledAllocationTests(unittest.TestCase):
    """These integration decisions remain testable without the optional binary."""
    def test_worker_count_uses_smallest_blas_budget_and_cpu_and_rows(self):
        examples = [
            (8192, 16, [{'user_api': 'blas', 'num_threads': 8},
                        {'user_api': 'blas', 'num_threads': 3},
                        {'user_api': 'openmp', 'num_threads': 1}], 3),
            (8192, 2, [{'user_api': 'blas', 'num_threads': 8}], 2),
            (256, 16, [{'user_api': 'blas', 'num_threads': 8}], 1),
            (8192, None, [{'user_api': 'blas', 'num_threads': 8}], 1),
            (8192, 16, [], 1),
            (8192, 16, [{'user_api': 'blas', 'num_threads': 0},
                        {'user_api': 'openmp', 'num_threads': 32}], 1),
        ]
        for n, cpus, pools, expected in examples:
            with self.subTest(n=n, cpus=cpus, pools=pools), patch(
                    'threadpoolctl.threadpool_info', return_value=pools), patch.object(
                    compiled.os, 'cpu_count', return_value=cpus):
                self.assertEqual(compiled._worker_count(n), expected)

    def test_compiled_values_partitions_rows_with_budget_without_dense_copy(self):
        n = 4097
        K = np.broadcast_to(np.array(1.), (n, n))
        x, rhs = np.ones(n), np.zeros(n)
        intervals = []

        def evaluate(matrix, vector, right, shift, s, start, stop, scores, residual, maxima):
            self.assertIs(matrix, K)
            self.assertIs(vector, x)
            self.assertIs(right, rhs)
            self.assertEqual(len(scores), stop-start)
            intervals.append((start, stop))
            scores[:] = np.arange(start, stop)
            residual[:] = -scores
            maxima[:] = 1.
            return 0

        pools = [{'user_api': 'blas', 'num_threads': 8},
                 {'user_api': 'blas', 'num_threads': 3}]
        with patch.object(compiled, '_ACCEL', types.SimpleNamespace(evaluate=evaluate)), patch(
                'threadpoolctl.threadpool_info', return_value=pools), patch.object(
                compiled.os, 'cpu_count', return_value=16):
            result = compiled.compiled_values(K, x, rhs, .5, .25)
        self.assertIsNotNone(result)
        self.assertEqual(sorted(intervals), [(0, n//3), (n//3, 2*n//3), (2*n//3, n)])
        np.testing.assert_array_equal(result[0], np.arange(n))
        np.testing.assert_array_equal(result[1], -np.arange(n))
        np.testing.assert_array_equal(result[2], np.ones(n))

    def test_missing_extension_small_input_and_status_failure_decline(self):
        K, x, rhs = np.ones((256, 256)), np.ones(256), np.zeros(256)
        with patch.object(compiled, '_ACCEL', None):
            self.assertIsNone(compiled.compiled_values(K, x, rhs, .5, .25))
        accelerator = types.SimpleNamespace(evaluate=lambda *args: 2)
        with patch.object(compiled, '_ACCEL', accelerator):
            self.assertIsNone(compiled.compiled_values(K, x, rhs, .5, .25))
        with patch.object(compiled, '_ACCEL') as accelerator:
            self.assertIsNone(compiled.compiled_values(K[:2, :2], x[:2], rhs[:2], .5, .25))
            accelerator.evaluate.assert_not_called()


@unittest.skipUnless(compiled._ACCEL is not None, UNAVAILABLE)
class CompiledArithmeticTests(unittest.TestCase):
    def assert_bits_equal(self, actual, expected):
        np.testing.assert_array_equal(np.asarray(actual).view(np.uint64),
                                      np.asarray(expected).view(np.uint64))

    def assert_core_reference(self, K, x, rhs, shift, s, outputs):
        scores, residual, maxima = outputs
        self.assert_bits_equal(maxima, np.max(abs(K), axis=1))
        if native._native_supported():
            expected_scores = np.array([math.sumprod(row.tolist(), x.tolist()) for row in K])
            expected_residual = np.array([
                math.sumprod(row.tolist() + [float(rhs[i]), -s, -shift],
                             (-x).tolist() + [1., 1., float(x[i])])
                for i, row in enumerate(K)
            ])
            self.assert_bits_equal(scores, expected_scores)
            self.assert_bits_equal(residual, expected_residual)
        else:
            # CPython 3.11 has no sumprod. Test the compiled arithmetic against
            # exact rational values and the same mathematical forward bounds.
            n = len(x)
            eta = native._gradual_underflow()
            absolute_x = native._up(native._up(math.fsum(abs(x).tolist())))
            sc = native._bound_constants(n, eta)
            rc = native._bound_constants(n+3, eta)
            for i, row in enumerate(K):
                dot = sum((_exact(a)*_exact(b) for a, b in zip(row, x)), Fraction())
                wanted = _exact(rhs[i]) - _exact(s) - _exact(shift)*_exact(x[i]) - dot
                upper = native._mul(float(maxima[i]), absolute_x)
                score_bound = native._error_bound(float(scores[i]), upper, sc)
                total = native._add(native._add(native._add(upper, abs(float(rhs[i]))), abs(s)),
                                    native._mul(abs(shift), abs(float(x[i]))))
                residual_bound = native._error_bound(float(residual[i]), total, rc)
                self.assertLessEqual(abs(_exact(scores[i])-dot), _exact(score_bound))
                self.assertLessEqual(abs(_exact(residual[i])-wanted), _exact(residual_bound))

    def test_48_small_cases_and_input_immutability(self):
        count = 0
        for n, kind, K, x, rhs, shift, s in _cases():
            with self.subTest(n=n, kind=kind):
                snapshots = [value.copy() for value in (K, x, rhs)]
                for value in (K, x, rhs):
                    value.flags.writeable = False
                status, *outputs = _rows(K, x, rhs, shift, s)
                self.assertEqual(status, 0)
                self.assert_core_reference(K, x, rhs, shift, s, outputs)
                for value, before in zip((K, x, rhs), snapshots):
                    self.assert_bits_equal(value, before)
                    self.assertFalse(value.flags.writeable)
                count += 1
        self.assertEqual(count, 48)

    def test_disjoint_thread_chunks_match_single_call(self):
        with ThreadPoolExecutor(max_workers=4) as executor:
            for n, kind, K, x, rhs, shift, s in _cases():
                with self.subTest(n=n, kind=kind):
                    expected = _rows(K, x, rhs, shift, s)
                    endpoints = sorted(set([0, n//4, n//2, 3*n//4, n]))
                    futures = [executor.submit(_rows, K, x, rhs, shift, s, start, stop)
                               for start, stop in zip(endpoints[:-1], endpoints[1:])]
                    result = [f.result() for f in futures]
                    self.assertTrue(all(value[0] == 0 for value in result))
                    for index in (1, 2, 3):
                        self.assert_bits_equal(np.concatenate([value[index] for value in result]),
                                               expected[index])

    def test_expanded_residual_keeps_lost_score_information(self):
        K = np.array([[float(2**53), float(2**53+2)], [0., 0.]])
        x, rhs = np.array([1., .5]), np.array([float(3*2**52), 0.])
        status, scores, residual, _ = _rows(K, x, rhs, 0., 0.)
        self.assertEqual(status, 0)
        self.assertEqual(rhs[0]-scores[0], 0.)
        self.assertEqual(residual[0], -1.)

    def test_guarded_boundaries_and_each_unsupported_operand_position(self):
        for value in (0., -0., 2.**-200, -2.**-200, 2.**200, -2.**200):
            with self.subTest(boundary=value):
                self.assertEqual(_rows(np.full((3, 3), value), np.full(3, value),
                                       np.full(3, value), value, value)[0], 0)
        invalid = [math.nan, math.inf, -math.inf, float(np.nextafter(0., 1.)),
                   float(np.nextafter(2.**-200, 0.)), float(np.nextafter(2.**200, np.inf))]
        for location in ('K', 'x', 'rhs', 'shift', 's'):
            for value in invalid:
                with self.subTest(location=location, value=value):
                    K, x, rhs = np.ones((3, 3)), np.ones(3), np.zeros(3)
                    shift, s = .125, .3
                    if location == 'K': K[1, 1] = value
                    elif location == 'x': x[1] = value
                    elif location == 'rhs': rhs[1] = value
                    elif location == 'shift': shift = value
                    else: s = value
                    self.assertEqual(_rows(K, x, rhs, shift, s)[0], 2)

    def test_zero_stride_and_unaligned_readonly_inputs(self):
        n = 7
        for layout in ('broadcast', 'unaligned'):
            with self.subTest(layout=layout):
                if layout == 'broadcast':
                    K = np.broadcast_to(np.arange(n, dtype=float), (n, n))
                    x = np.broadcast_to(np.array(.75), (n,))
                    rhs = np.broadcast_to(np.array(.25), (n,))
                else:
                    K = np.ndarray((n, n), dtype=float,
                                   buffer=np.empty(n*n*8+1, dtype=np.uint8), offset=1)
                    x = np.ndarray((n,), dtype=float,
                                   buffer=np.empty(n*8+3, dtype=np.uint8), offset=3)
                    rhs = np.ndarray((n,), dtype=float,
                                     buffer=np.empty(n*8+5, dtype=np.uint8), offset=5)
                    K[:] = np.arange(n*n).reshape(n, n)/4
                    x[:] = np.linspace(-1, 1, n)
                    rhs[:] = np.arange(n)/3
                for value in (K, x, rhs):
                    value.flags.writeable = False
                status, *outputs = _rows(K, x, rhs, .5, -.25)
                self.assertEqual(status, 0)
                self.assert_core_reference(K, x, rhs, .5, -.25, outputs)

    def test_extension_buffer_shapes_aliasing_and_output_writability(self):
        K, x, rhs = np.ones((3, 3)), np.ones(3), np.zeros(3)
        outputs = [np.empty(3) for _ in range(3)]
        for matrix in (K.astype(np.float32), K.astype(np.dtype(float).newbyteorder('S'))):
            with self.assertRaises(TypeError):
                compiled._ACCEL.evaluate(matrix, x, rhs, .5, .25, 0, 3, *outputs)
        for matrix, vector, right in ((K[:, :2], x, rhs), (K, x[:2], rhs),
                                       (K, x, rhs.reshape(3, 1))):
            with self.assertRaises(ValueError):
                compiled._ACCEL.evaluate(matrix, vector, right, .5, .25, 0, 3, *outputs)
        for start, stop in ((-1, 2), (1, 4), (2, 1)):
            with self.assertRaises(ValueError):
                compiled._ACCEL.evaluate(K, x, rhs, .5, .25, start, stop, *outputs)
        bad_outputs = [x, K[0], outputs[1], np.empty(6)[::2], np.empty(2),
                       np.ndarray((3,), dtype=float, buffer=np.empty(25, dtype=np.uint8), offset=1)]
        for bad in bad_outputs:
            with self.subTest(shape=bad.shape, strides=bad.strides), self.assertRaises(ValueError):
                compiled._ACCEL.evaluate(K, x, rhs, .5, .25, 0, 3, bad, outputs[1], outputs[2])
        readonly = np.empty(3)
        readonly.flags.writeable = False
        with self.assertRaises((ValueError, BufferError)):
            compiled._ACCEL.evaluate(K, x, rhs, .5, .25, 0, 3, readonly, outputs[1], outputs[2])
        status, *empty = _rows(K, x, rhs, .5, .25, 1, 1)
        self.assertEqual(status, 0)
        self.assertTrue(all(value.shape == (0,) for value in empty))

    @unittest.skipUnless(sys.platform == 'win32', 'Windows floating control API')
    def test_non_nearest_and_flush_to_zero_decline_and_restore(self):
        # https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/controlfp-s
        control = ctypes.CDLL('ucrtbase')._controlfp_s
        control.argtypes = [ctypes.POINTER(ctypes.c_uint), ctypes.c_uint, ctypes.c_uint]
        control.restype = ctypes.c_int
        current = ctypes.c_uint()
        self.assertEqual(control(ctypes.byref(current), 0, 0), 0)
        saved = current.value
        K, x, rhs = np.ones((3, 3)), np.ones(3), np.zeros(3)
        outputs = [np.empty(3) for _ in range(3)]
        args = (K, x, rhs, .5, .25, 0, 3, *outputs)
        for name, value, mask in (('down', 0x100, 0x300), ('up', 0x200, 0x300),
                                  ('toward_zero', 0x300, 0x300),
                                  ('flush_to_zero', 0x01000000, 0x03000000)):
            with self.subTest(mode=name):
                try:
                    set_status = control(ctypes.byref(current), value, mask)
                    status = compiled._ACCEL.evaluate(*args)
                finally:
                    restore_status = control(ctypes.byref(current), saved, mask)
                self.assertEqual(set_status, 0)
                self.assertEqual(restore_status, 0)
                self.assertEqual(status, 4)
                self.assertEqual(_rows(K, x, rhs, .5, .25)[0], 0)

    def test_enabled_disabled_native_contract_and_readonly_inputs(self):
        n = 256
        rng = np.random.default_rng(923178)
        K, x = rng.uniform(.25, 1., size=(n, n)), rng.uniform(.5, 1., size=n)
        rhs = rng.normal(size=n)
        snapshots = [value.copy() for value in (K, x, rhs)]
        for value in (K, x, rhs):
            value.flags.writeable = False
        with patch('threadpoolctl.threadpool_info', return_value=[
                {'user_api': 'blas', 'num_threads': 1}]):
            raw = compiled.compiled_values(K, x, rhs, .125, .3)
            self.assertIsNotNone(raw)
            if native._native_supported():
                args = (K, .125, rhs, x, .3, -.25)
                with patch.object(compiled, '_ACCEL', None):
                    disabled = native.native_compensated_residual(*args)
                enabled = native.native_compensated_residual(*args)
                self.assertIsNotNone(disabled)
                self.assertIsNotNone(enabled)
                for first, second in zip(disabled, enabled):
                    self.assert_bits_equal(first, second)
            else:
                # The optional arithmetic still runs on Python 3.11; its
                # caller's audited native screen correctly remains unavailable.
                self.assertIsNone(native.native_compensated_residual(K, .125, rhs, x, .3, -.25))
                self.assert_core_reference(K[:7, :7], x[:7], rhs[:7], .125, .3,
                                           _rows(K[:7, :7], x[:7], rhs[:7], .125, .3)[1:])
        for value, before in zip((K, x, rhs), snapshots):
            self.assert_bits_equal(value, before)
            self.assertFalse(value.flags.writeable)


if __name__ == '__main__':
    unittest.main(verbosity=2)
