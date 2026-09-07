"""Original-equation residuals checked independently with exact rational math.

These are arithmetic and supported-input regressions, not MNIST or tuning tests.
"""
from fractions import Fraction
import math
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from scipy.linalg import eigh

import dwd._kernel_linear_system as linear
from dwd._compensated_residual import _products, _gradual_underflow, compensated_residual
from dwd._kernel_linear_system import KernelLinearSystem
from dwd._spectral_linear_system import SpectralLinearSystem


def exact(value):
    return Fraction.from_float(float(value))


def rational_residual(K, shift, rhs, x, s):
    return [exact(r) - exact(s) - exact(shift) * exact(xi)
            - sum((exact(k) * exact(xj) for k, xj in zip(row, x)), Fraction())
            for row, r, xi in zip(K, rhs, x)]


class CompensatedResidualTests(unittest.TestCase):
    def test_product_expansions_agree_with_exact_products_at_wide_scales(self):
        rng = np.random.RandomState(8703)
        a = np.ldexp(rng.uniform(-1, 1, 300), rng.randint(-1000, 1000, 300))
        # Include mixed large/small operands, and gradual product underflow.
        b = np.ldexp(rng.uniform(-1, 1, 300), -np.frexp(a)[1])
        # Independently vary the final product scale as well as operand scales.
        a = np.r_[a, np.ldexp(rng.uniform(-1, 1, 300), rng.randint(-500, 500, 300))]
        b = np.r_[b, np.ldexp(rng.uniform(-1, 1, 300), rng.randint(-500, 500, 300))]
        eta = np.nextafter(0., 1.)
        a = np.r_[a, eta, np.finfo(float).tiny, 0., -eta, 2.**-600, np.finfo(float).max]
        b = np.r_[b, .5, 2.**-53, 1e300, 1.5, 2.**-500, .75]
        high, low = _products(a, b)
        for ai, bi, hi, lo in zip(a, b, high, low):
            self.assertLessEqual(abs(exact(ai)*exact(bi)-exact(hi)-exact(lo)), 2*exact(eta))

    def test_cancellation_residual_and_constraint_bounds_cover_exact_rational_values(self):
        rng = np.random.RandomState(209)
        features = rng.normal(size=(15, 4))
        K = features @ features.T + 1e7
        x = rng.normal(size=15) * 1e7
        x[-1] = -math.fsum(x[:-1])
        shift, s, target = 1e-7, -.13, .23
        rhs = K @ x + shift*x + s
        r, c, scores, bound, cb = compensated_residual(K, shift, rhs, x, s, target)
        expected = rational_residual(K, shift, rhs, x, s)
        for actual, wanted, allowance in zip(r, expected, bound):
            self.assertLessEqual(abs(exact(actual)-wanted), exact(allowance))
        self.assertLessEqual(abs(exact(c)-(exact(target)-sum(map(exact, x), Fraction()))), exact(cb))
        # Scores are separately rounded once from the expanded dot products.
        wanted_scores = np.array([float(sum((exact(k)*exact(v) for k, v in zip(row, x)), Fraction())) for row in K])
        assert_array_equal(scores, wanted_scores)
        self.assertGreater(np.max(abs(rhs-K@x-shift*x-s-r)), 1e-3)

    def test_subnormal_products_and_strict_numpy_underflow_setting(self):
        tiny = np.finfo(float).tiny
        K = np.array([[tiny, tiny/2], [tiny/2, tiny]])
        x = np.array([.75, -.5])
        with np.errstate(under='raise'):
            r, c, scores, bounds, cb = compensated_residual(K, tiny, np.zeros(2), x, 0., 0.)
        for actual, wanted, allowance in zip(r, rational_residual(K, tiny, np.zeros(2), x, 0.), bounds):
            self.assertLessEqual(abs(exact(actual)-wanted), exact(allowance))
        self.assertTrue(np.isfinite(scores).all())

    def test_invalid_and_overflowing_product_state_fails_closed(self):
        for a, b in (([np.inf], [0.]), ([np.nan], [1.]), ([1e308], [1e308])):
            with self.subTest(a=a, b=b), self.assertRaises(FloatingPointError):
                _products(a, b)
        with patch('dwd._compensated_residual._gradual_underflow', side_effect=FloatingPointError('unsupported')):
            with self.assertRaisesRegex(FloatingPointError, 'unsupported'):
                compensated_residual(np.eye(2), .1, np.ones(2), np.ones(2), 0., 0.)
        # Emulate unavailable minimum subnormal without modifying process-wide
        # floating-point modes or the user's numerical runtime.
        with patch('dwd._compensated_residual.np.nextafter', return_value=0.):
            with self.assertRaisesRegex(FloatingPointError, 'gradual'):
                _gradual_underflow()

    def test_good_ordinary_solve_does_not_invoke_compensated_fallback(self):
        K = np.diag([.2, .7, 1.3, 2.])
        rhs = np.array([-.1, .4, .2, -.3])
        for system in (KernelLinearSystem(K, .13), SpectralLinearSystem(K, .13, np.eye(4), K.diagonal())):
            with self.subTest(backend=type(system).__name__), patch.object(
                    linear, 'compensated_residual', side_effect=AssertionError('Unnecessary compensation')):
                x, s = system.solve_constrained(rhs)
                assert_array_equal(system.last_product, K@x)
                self.assertNotIn('compensated_residual_checks', system.info)
                self.assertLess(np.max(abs(rhs-K@x-.13*x-s)), 1e-14)

    def test_recheck_does_not_relax_residual_tolerance(self):
        system = KernelLinearSystem(np.eye(4), .1)
        x, rhs = np.zeros(4), np.array([1.01e-10, -1.01e-10, 0., 0.])
        good, residual, _, _, maximum, _ = system._measure(rhs, 0., x, 0.)
        self.assertFalse(good)
        self.assertGreater(maximum, 1e-10)
        assert_array_equal(residual, rhs)
        self.assertEqual(system.info['compensated_residual_checks'], 1)

    def test_singular_kernel_compensation_refines_original_equation_with_nonzero_target(self):
        rng = np.random.RandomState(76)
        F = rng.normal(size=(12, 3))
        K, shift, target = F@F.T, .01, -.23
        rhs = rng.normal(size=12)
        values, vectors = eigh(K)
        values = np.maximum(values, 0.)
        for system in (KernelLinearSystem(K, shift), SpectralLinearSystem(K, shift, vectors, values)):
            original = system._candidate
            calls = []
            def corrupted_once(r, t):
                xx, ss = original(r, t)
                calls.append(1)
                if len(calls) == 1:
                    xx = xx + np.linspace(-.01, .02, len(xx))
                elif len(calls) == 2:
                    # Make the cheap preflight ineffective so this test still
                    # exercises compensated refinement of the original state.
                    return np.zeros_like(xx), 0.
                return xx, ss
            with self.subTest(backend=type(system).__name__), patch.object(system, '_candidate', side_effect=corrupted_once):
                x, s = system.solve_constrained(rhs, target)
            residual = rational_residual(K, shift, rhs, x, s)
            self.assertLess(max(map(abs, residual)), exact(1e-10))
            self.assertLess(abs(math.fsum(x)-target), 1e-11)
            self.assertGreater(system.info['compensated_residual_checks'], 0)
            self.assertGreater(system.info['refinement_steps'], 0)
            self.assertEqual(system.info['positive_eigenvalues_discarded'], 0)
            self.assertEqual(system.info['added_objective_regularization'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
