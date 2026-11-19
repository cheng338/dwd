"""Shared-product arithmetic: frozen binary differential and exact bounds.

Set DWD_NATIVE_BASELINE to an unchanged older accelerator for the differential
tests. The independent Fraction oracle runs on every available native build.
No fit or dataset is needed; both native modules read the same immutable inputs.
"""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import importlib.util
import math
import os
from pathlib import Path
import struct
import unittest

import numpy as np

from dwd import _compiled_residual as compiled
from dwd import _native_residual as native


def load_baseline():
    path = os.environ.get('DWD_NATIVE_BASELINE')
    if not path:
        return None
    path = Path(path).resolve()
    if compiled._ACCEL is not None and path == Path(compiled._ACCEL.__file__).resolve():
        raise RuntimeError('Differential baseline must be a distinct frozen binary.')
    spec = importlib.util.spec_from_file_location('_frozen_dwd._residual_accel', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASELINE = load_baseline()


def evaluate(module, K, x, rhs, shift=.125, intercept=.3, start=0, stop=None):
    stop = len(x) if stop is None else stop
    outputs = [np.full(stop - start, np.nan) for _ in range(3)]
    status = module.evaluate(K, x, rhs, shift, intercept, start, stop, *outputs)
    return status, outputs


def cases():
    rng = np.random.default_rng(6100117)
    # Products with zero components require the conservative old-product path.
    boundary = np.array([0., -0., 1., -1., 2.**-200, -(2.**-200),
                         2.**200, -(2.**200), np.nextafter(1., 2.),
                         np.nextafter(1., 0.), np.nextafter(2.**-200, 1.),
                         np.nextafter(2.**200, 0.)])
    yield np.resize(boundary, (24, 24)), np.resize(boundary[::-1], 24), np.resize(boundary, 24), 0., -0.
    for n in (1, 2, 3, 7, 8, 17, 64, 129):
        for mode in ('normal', 'cancellation', 'broad', 'powers', 'fortran', 'negative_stride', 'unaligned'):
            K = rng.normal(size=(n, n))
            x, rhs = rng.normal(size=n), rng.normal(size=n)
            shift, intercept = .125, -.3
            if mode == 'cancellation':
                K = 1. + rng.normal(scale=1e-12, size=(n, n))
                x *= 1e9
                x[-1] = -math.fsum(x[:-1].tolist())
                shift = 1e-9
            elif mode == 'broad':
                K = np.ldexp(rng.uniform(.5, 1., size=(n, n)), rng.integers(-199, 201, size=(n, n)))
                x = np.ldexp(rng.uniform(.5, 1., size=n), rng.integers(-199, 201, size=n))
                K *= rng.choice([-1., 1.], size=(n, n))
                x *= rng.choice([-1., 1.], size=n)
            elif mode == 'powers':
                K = np.ldexp(rng.choice([-1., 1.], size=(n, n)), rng.integers(-200, 201, size=(n, n)))
                x = np.ldexp(rng.choice([-1., 1.], size=n), rng.integers(-200, 201, size=n))
            elif mode == 'fortran':
                K = np.asfortranarray(K)
            elif mode == 'negative_stride':
                K, x, rhs = K[::-1, ::-1], x[::-1], rhs[::-1]
            elif mode == 'unaligned':
                raw = bytearray(K.nbytes + 1)
                view = np.ndarray(K.shape, dtype=np.float64, buffer=raw, offset=1)
                view[:] = K
                K = view
            yield K, x, rhs, shift, intercept


def exact(x):
    return Fraction.from_float(float(x))


class ProductSignArgumentTests(unittest.TestCase):
    def test_nonzero_components_sign_reverse_and_zeros_use_fallback(self):
        # This mirrors only the scalar sign argument, not the independent
        # residual oracle below. Exact zero components deliberately do not
        # assert a negated sign: the native optimization recomputes those.
        def split(x):
            t = x * 134217729.
            hi = t - (t - x)
            return hi, x - hi
        def product(x, y):
            xh, xl = split(x)
            yh, yl = split(y)
            p = xh * yh
            q = xh * yl + xl * yh
            z = p + q
            return z, p - z + q + xl * yl
        rng = np.random.default_rng(68215)
        reuse = fallback = 0
        for i in range(20000):
            x, y = np.ldexp(rng.uniform(.5, 1., size=2), rng.integers(-199, 201, size=2))
            if i % 4 == 0:
                y = 1. if i % 8 else -0.
            p = product(float(x), float(y))
            old_negative = product(float(x), -float(y))
            if all(value != 0. for value in p):
                reuse += 1
                self.assertEqual(struct.pack('dd', *old_negative), struct.pack('dd', -p[0], -p[1]))
            else:
                fallback += 1
        self.assertGreater(reuse, 10000)
        self.assertGreater(fallback, 4000)


@unittest.skipUnless(compiled._ACCEL is not None, 'Optional native extension is absent.')
class SharedProductOracleTests(unittest.TestCase):
    def test_independent_exact_rational_error_allowances(self):
        # Always use exact rational arithmetic, even when math.sumprod exists.
        eta = native._gradual_underflow()
        count = 0
        for K, x, rhs, shift, intercept in cases():
            if len(x) > 24:
                continue
            status, outputs = evaluate(compiled._ACCEL, K, x, rhs, shift, intercept)
            self.assertEqual(status, 0)
            n = len(x)
            ax = native._up(native._up(math.fsum(np.abs(x).tolist())))
            sc, rc = native._bound_constants(n, eta), native._bound_constants(n + 3, eta)
            for i, row in enumerate(K):
                value = sum((exact(a) * exact(b) for a, b in zip(row, x)), Fraction())
                residual = exact(rhs[i]) - exact(intercept) - exact(shift) * exact(x[i]) - value
                upper = native._mul(float(np.max(np.abs(row))), ax)
                sb = native._error_bound(outputs[0][i], upper, sc)
                upper = native._add(native._add(native._add(upper, abs(float(rhs[i]))), abs(intercept)),
                                    native._mul(abs(shift), abs(float(x[i]))))
                rb = native._error_bound(outputs[1][i], upper, rc)
                self.assertLessEqual(abs(exact(outputs[0][i]) - value), exact(sb))
                self.assertLessEqual(abs(exact(outputs[1][i]) - residual), exact(rb))
                count += 1
        self.assertGreater(count, 250)

    def test_rounding_mode_rejection_is_preserved(self):
        if os.name != 'nt':
            self.skipTest('This fenv control probe is specific to the audited Windows build.')
        import ctypes
        crt = ctypes.CDLL('ucrtbase')
        control = crt._controlfp_s
        control.argtypes = [ctypes.POINTER(ctypes.c_uint), ctypes.c_uint, ctypes.c_uint]
        control.restype = ctypes.c_int
        old = ctypes.c_uint()
        self.assertEqual(control(ctypes.byref(old), 0, 0), 0)
        try:
            current = ctypes.c_uint()
            # _MCW_RC, _RC_DOWN. No Python numerical arithmetic is required
            # between changing the mode and the native supported-mode probe.
            self.assertEqual(control(ctypes.byref(current), 0x100, 0x300), 0)
            K, x, rhs = np.ones((2, 2)), np.ones(2), np.zeros(2)
            status, _ = evaluate(compiled._ACCEL, K, x, rhs)
            self.assertEqual(status, 4)
        finally:
            restored = ctypes.c_uint()
            self.assertEqual(control(ctypes.byref(restored), old.value, 0x300), 0)


@unittest.skipUnless(compiled._ACCEL is not None and BASELINE is not None,
                     'Set DWD_NATIVE_BASELINE to a frozen native binary.')
class SharedProductDifferentialTests(unittest.TestCase):
    def bits_equal(self, a, b):
        self.assertEqual(np.asarray(a).tobytes(), np.asarray(b).tobytes())

    def test_all_outputs_bitwise_match_frozen_binary_and_inputs_are_untouched(self):
        count = 0
        for K, x, rhs, shift, intercept in cases():
            before = [value.tobytes() for value in (K, x, rhs)]
            for value in (K, x, rhs):
                value.flags.writeable = False
            old_status, old = evaluate(BASELINE, K, x, rhs, shift, intercept)
            status, new = evaluate(compiled._ACCEL, K, x, rhs, shift, intercept)
            self.assertEqual(status, old_status)
            self.assertEqual(status, 0)
            for a, b in zip(new, old):
                self.bits_equal(a, b)
            self.assertEqual(before, [value.tobytes() for value in (K, x, rhs)])
            count += 1
        self.assertEqual(count, 57)

    def test_domain_declines_match_without_using_partial_outputs(self):
        bad = [float('nan'), float('inf'), -float('inf'), np.nextafter(2.**-200, 0.),
               np.nextafter(2.**200, math.inf), np.nextafter(0., 1.)]
        for value in bad:
            for where in ('matrix', 'vector', 'rhs', 'shift', 'intercept'):
                K, x, rhs = np.ones((3, 3)), np.ones(3), np.ones(3)
                shift, intercept = .125, .3
                if where == 'matrix': K[1, 2] = value
                elif where == 'vector': x[1] = value
                elif where == 'rhs': rhs[1] = value
                elif where == 'shift': shift = value
                else: intercept = value
                with self.subTest(value=value, where=where):
                    old, _ = evaluate(BASELINE, K, x, rhs, shift, intercept)
                    new, _ = evaluate(compiled._ACCEL, K, x, rhs, shift, intercept)
                    self.assertEqual(old, 2)
                    self.assertEqual(new, old)

    def test_thread_partition_order_and_baseline_are_bitwise_identical(self):
        rng = np.random.default_rng(117)
        n = 257
        K = rng.uniform(.25, 1., size=(n, n))
        x, rhs = rng.normal(size=n), rng.normal(size=n)
        old_status, expected = evaluate(BASELINE, K, x, rhs)
        self.assertEqual(old_status, 0)
        for workers in (1, 2, 4):
            with ThreadPoolExecutor(max_workers=workers) as pool:
                chunks = list(pool.map(lambda part: evaluate(compiled._ACCEL, K, x, rhs,
                                   start=n * part // workers, stop=n * (part + 1) // workers), range(workers)))
            self.assertTrue(all(status == 0 for status, _ in chunks))
            for i in range(3):
                self.bits_equal(np.concatenate([values[i] for _, values in chunks]), expected[i])


if __name__ == '__main__':
    unittest.main(verbosity=2)
