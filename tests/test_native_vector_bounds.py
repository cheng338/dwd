"""Vectorized outward bounds and compiled-screen acceptance guards."""
from fractions import Fraction
import math
import unittest
from unittest.mock import patch

import numpy as np

import dwd._native_residual as native


class NativeVectorBoundTests(unittest.TestCase):
    def test_array_bounds_match_scalar_bits_and_preserve_inputs(self):
        eta = math.nextafter(0., 1.)
        computed = np.array([0., -0., eta, -eta, 2.**-200, -2.**-200,
                             1., -1., math.nextafter(1., 0.),
                             math.nextafter(1., math.inf), 2.**200, -2.**200])
        upper = np.array([0., eta, 2.**-200, 1., 2.**200, 2.**420])
        values, sums = np.broadcast_arrays(computed[:, None], upper[None, :])
        before = [v.copy() for v in (values, sums)]
        for terms in (1, 2, 7, 31, 255, 256, 2048, native._MAX_TERMS):
            with self.subTest(terms=terms):
                constants = native._bound_constants(terms, eta)
                # Scalar nextafter does not raise NumPy underflow exceptions.
                with np.errstate(all='ignore'):
                    actual = native._array_error_bound(values, sums, constants)
                expected = np.array([
                    native._error_bound(float(value), float(total), constants)
                    for value, total in zip(values.flat, sums.flat)
                ]).reshape(values.shape)
                self.assertEqual(actual.dtype, np.dtype('float64'))
                np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
        for actual, expected in zip((values, sums), before):
            np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))

    def test_array_bounds_enclose_independent_exact_formula(self):
        exact = Fraction.from_float
        eta = math.nextafter(0., 1.)
        values = np.array([0., 1e-100, 1e50, -1e50])
        upper = np.array([1., 1e100, 1e100, 1e100])
        eps = exact(float(np.finfo(float).eps))
        for terms in (1, 31, 2048, native._MAX_TERMS):
            gamma = (4 * terms - 2) * eps / (1 - (4 * terms - 2) * eps)
            a = eps + 2 * gamma**2
            with np.errstate(all='ignore'):
                actual = native._array_error_bound(values, upper, native._bound_constants(terms, eta))
            for value, total, bound in zip(values, upper, actual):
                with self.subTest(terms=terms, value=value, total=total):
                    wanted = (a * abs(exact(float(value))) + gamma**3 * exact(float(total))
                              + 5 * terms * exact(eta)) / (1 - a)
                    self.assertGreaterEqual(exact(float(bound)), wanted)

    def test_compiled_screen_rejects_nonfinite_and_uncertain_outputs(self):
        # Test NumPy guards without requiring an optional extension or sumprod;
        # separate public runtime tests retain the unsupported-runtime rejection.
        args = (np.ones((2, 2)), .125, np.ones(2), np.ones(2), .25, 0.)
        good = (np.full(2, 2.), np.full(2, -1.375), np.ones(2))
        cases = [('valid', good, True)]
        for index, name in enumerate(('score', 'residual', 'maximum')):
            # Successful core maxima are nonnegative by construction; the
            # independent core tests verify that absolute-value invariant.
            bad_values = (math.nan, math.inf) if name == 'maximum' else (math.nan, math.inf, -math.inf)
            for bad in bad_values:
                values = [a.copy() for a in good]
                values[index][0] = bad
                cases.append((name + ':' + str(bad), tuple(values), False))
        values = [a.copy() for a in good]
        values[0][0] = 1e-300
        cases.append(('insufficient_score_precision', tuple(values), False))
        for name, values, accepted in cases:
            with self.subTest(case=name), patch.object(native, '_native_supported', return_value=True), patch.object(
                    native, 'compiled_values', return_value=values), patch.object(
                    native.math, 'sumprod', side_effect=AssertionError('compiled branch used scalar sumprod'), create=True):
                with np.errstate(all='raise'):
                    expected_errors = np.geterr().copy()
                    result = native.native_compensated_residual(*args)
                    self.assertEqual(np.geterr(), expected_errors)
                if accepted:
                    self.assertIsNotNone(result)
                    self.assertIs(result[0], values[1])
                    self.assertIs(result[2], values[0])
                else:
                    self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
