"""Exact-rational and fallback checks for the guarded native residual screen."""
from fractions import Fraction
import math
import types
import unittest
from unittest.mock import patch

import numpy as np

import dwd._native_residual as native


def exact(value):
    return Fraction.from_float(float(value))


def exact_sumprod(a, b):
    return float(sum((exact(x) * exact(y) for x, y in zip(a, b)), Fraction()))


class NativeResidualTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[2., .3], [.3, 3.]])
        self.x = np.array([.7, -.4])
        self.args = (self.K, .01, np.array([.3, -.2]), self.x, .1, .2)

    def assert_exact_bounds(self, args, result):
        K, shift, rhs, x, s, target = args
        residual, constraint, scores, bounds, cb, score_bounds = result
        for i, row in enumerate(K):
            score = sum((exact(a) * exact(b) for a, b in zip(row, x)), Fraction())
            wanted = exact(rhs[i]) - exact(s) - exact(shift) * exact(x[i]) - score
            self.assertLessEqual(abs(exact(scores[i]) - score), exact(score_bounds[i]))
            self.assertLessEqual(abs(exact(residual[i]) - wanted), exact(bounds[i]))
            faithful = math.nextafter(4 * np.finfo(float).eps * abs(scores[i])
                                      + (2 * len(x) + 6) * np.nextafter(0., 1.), math.inf)
            self.assertLessEqual(score_bounds[i], faithful)
        wanted_constraint = exact(target) - sum(map(exact, x), Fraction())
        self.assertLessEqual(abs(exact(constraint) - wanted_constraint), exact(cb))

    @unittest.skipUnless(native._native_supported(), 'Requires audited CPython native sumprod.')
    def test_native_bounds_cover_exact_rationals_at_many_scales(self):
        rng = np.random.default_rng(674892)
        accepted = 0
        for n in (1, 2, 7, 31):
            for kexp in (-180, -20, 0, 20, 180):
                for xexp in (-180, -20, 0, 20, 180):
                    K = np.ldexp(rng.uniform(-1, 1, (n, n)), kexp)
                    x = np.ldexp(rng.uniform(-1, 1, n), xexp)
                    rhs = rng.normal(size=n)
                    args = (K, .03125, rhs, x, .37, -.17)
                    result = native.native_compensated_residual(*args)
                    if result is not None:
                        accepted += 1
                        self.assert_exact_bounds(args, result)
        self.assertEqual(accepted, 100)

    @unittest.skipUnless(native._native_supported(), 'Requires audited CPython native sumprod.')
    def test_cancelled_original_equations_and_strides_have_valid_bounds(self):
        rng = np.random.default_rng(90216)
        n = 23
        K = 1e7 + rng.normal(size=(n, n)) * 1e-5
        x = rng.normal(size=n) * 1e7
        x[-1] = -math.fsum(x[:-1])
        rhs = K @ x + .003 * x + .13
        for args in ((K, .003, rhs, x, .13, .23),
                     (K[::-1, ::-1], .003, rhs[::-1], x[::-1], .13, .23)):
            result = native.native_compensated_residual(*args)
            self.assertIsNotNone(result)
            self.assert_exact_bounds(args, result)

    @unittest.skipUnless(native._native_supported(), 'Requires audited CPython native sumprod.')
    def test_operand_boundaries_are_accepted_with_strict_numpy_errors(self):
        for value in (native._MIN_OPERAND, native._MAX_OPERAND):
            args = (np.array([[value]]), value, np.array([value]),
                    np.array([value]), value, value)
            with np.errstate(all='raise'):
                result = native.native_compensated_residual(*args)
            self.assertIsNotNone(result)
            self.assert_exact_bounds(args, result)

    def test_unsupported_runtime_or_missing_builtin_falls_back(self):
        for version in ((3, 11), (3, 13), (3, 14)):
            with patch.object(native.sys, 'version_info', version):
                self.assertFalse(native._native_supported())
                self.assertIsNone(native.native_compensated_residual(*self.args))
        with patch.object(native.sys, 'implementation', types.SimpleNamespace(name='pypy')):
            self.assertFalse(native._native_supported())
        with patch.object(native.math, 'sumprod', None, create=True):
            self.assertFalse(native._native_supported())
            self.assertIsNone(native.native_compensated_residual(*self.args))

    def test_nonstandard_rounding_and_unavailable_subnormals_fall_back(self):
        with patch.object(native, '_native_supported', return_value=True):
            with patch.object(native, '_round_to_nearest', return_value=False):
                self.assertIsNone(native.native_compensated_residual(*self.args))
            with patch.object(native, '_gradual_underflow', side_effect=FloatingPointError):
                self.assertIsNone(native.native_compensated_residual(*self.args))

    def test_invalid_shapes_nonfinite_values_and_extreme_operands_fall_back(self):
        cases = []
        for value in (math.nan, math.inf, -math.inf, np.nextafter(0., 1.),
                      np.finfo(float).tiny, 2.**-201, 2.**201, 1e308):
            for index in (0, 1, 2, 3, 4, 5):
                args = list(self.args)
                if index in (0, 2, 3):
                    args[index] = np.full_like(args[index], value)
                else:
                    args[index] = value
                cases.append(args)
        cases.extend([(np.ones((2, 3)),) + self.args[1:],
                      (self.K.astype(np.float32),) + self.args[1:],
                      (self.K.tolist(),) + self.args[1:],
                      (self.K, .1, np.ones(3), self.x, 0., 0.),
                      (self.K, .1, np.ones((2, 1)), self.x[:, None], 0., 0.),
                      (np.empty((0, 0)), .1, np.empty(0), np.empty(0), 0., 0.)])
        with patch.object(native, '_native_supported', return_value=True):
            for args in cases:
                with self.subTest(args=args):
                    self.assertIsNone(native.native_compensated_residual(*args))
            with patch.object(native, '_MAX_TERMS', 4):
                self.assertIsNone(native.native_compensated_residual(*self.args))

    def test_native_sumprod_failures_and_nonfinite_returns_fall_back(self):
        with patch.object(native, '_native_supported', return_value=True):
            for failure in (OverflowError, ValueError, FloatingPointError, TypeError):
                with patch.object(native.math, 'sumprod', side_effect=failure, create=True):
                    self.assertIsNone(native.native_compensated_residual(*self.args))
            for value in (math.nan, math.inf):
                with patch.object(native.math, 'sumprod', return_value=value, create=True):
                    self.assertIsNone(native.native_compensated_residual(*self.args))

    def test_terms_are_python_floats_and_residual_is_one_original_dot(self):
        calls = []

        def record_sumprod(a, b):
            self.assertIsInstance(a, list)
            self.assertIsInstance(b, list)
            self.assertTrue(all(type(value) is float for value in a + b))
            calls.append((a.copy(), b.copy()))
            return exact_sumprod(a, b)

        with patch.object(native, '_native_supported', return_value=True), patch.object(
                native.math, 'sumprod', side_effect=record_sumprod, create=True):
            result = native.native_compensated_residual(*self.args)
        self.assertIsNotNone(result)
        self.assert_exact_bounds(self.args, result)
        K, shift, rhs, x, s, _ = self.args
        self.assertEqual(len(calls), 2 * len(x))
        for i in range(len(x)):
            self.assertEqual(calls[2*i], (K[i].tolist(), x.tolist()))
            self.assertEqual(calls[2*i+1],
                             (K[i].tolist() + [rhs[i], -s, -shift],
                              (-x).tolist() + [1., 1., x[i]]))

    def test_uncertain_score_precision_uses_portable_fallback(self):
        # The exact kernel score is zero under severe cancellation. A native
        # forward bound is useful but cannot meet the faithful-zero budget.
        args = (np.ones((2, 2)), .01, np.ones(2), np.array([1e7, -1e7]), 0., 0.)
        with patch.object(native, '_native_supported', return_value=True), patch.object(
                native.math, 'sumprod', side_effect=exact_sumprod, create=True):
            self.assertIsNone(native.native_compensated_residual(*args))

    def test_error_bound_constants_enclose_exact_formula(self):
        eps, eta = exact(np.finfo(float).eps), exact(np.nextafter(0., 1.))
        for m in (1, 2, 31, 9674, native._MAX_TERMS):
            gamma = (4*m-2) * eps / (1 - (4*m-2) * eps)
            a = eps + 2 * gamma**2
            for result, sum_upper in ((0., 1.), (1e-100, 1e100), (1e50, 1e100)):
                wanted = (a * exact(abs(result)) + gamma**3 * exact(sum_upper) + 5*m*eta) / (1-a)
                bound = native._error_bound(result, sum_upper,
                                            native._bound_constants(m, float(eta)))
                self.assertGreaterEqual(exact(bound), wanted)


if __name__ == '__main__':
    unittest.main(verbosity=2)
