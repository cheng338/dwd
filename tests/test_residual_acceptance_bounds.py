"""Independent original-equation tests for bounded ordinary acceptance."""
from decimal import Decimal, localcontext
import math
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import eigh
from sklearn.metrics.pairwise import rbf_kernel
from threadpoolctl import threadpool_limits

from dwd._kernel_linear_system import KernelLinearSystem
from dwd._spectral_linear_system import SpectralLinearSystem
from dwd._ordinary_residual_bounds import OrdinaryResidualBounds
from dwd.gen_kern_dwd import KernGDWD


def dec(value):
    return Decimal.from_float(float(value))


def exact_residual(K, shift, rhs, x, scalar, target):
    """Direct Decimal arithmetic, independent of production dot/error code."""
    with localcontext() as context:
        context.prec = 100
        coefficients = list(map(dec, x))
        residual = [dec(rhs[i]) - sum((dec(a)*b for a, b in zip(row, coefficients)), Decimal(0))
                    - dec(shift)*coefficients[i] - dec(scalar) for i, row in enumerate(K)]
        constraint = dec(target)-sum(coefficients, Decimal(0))
        return max(map(abs, residual)), abs(constraint)


def rounded_left_product(K, x):
    """A legal binary64 ordinary reduction, independent of the BLAS provider.

    Separate multiply/add ufuncs force the tested summation order. A provider
    that happens to return a more accurate dot must not make this regression
    fail: the acceptance bound must cover this legal ordinary result too.
    """
    product = np.zeros(len(K))
    for column in range(len(x)):
        term = K[:, column]*x[column]
        product = product + term
    return product


class ResidualAcceptanceBoundsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limits = threadpool_limits(2)

    @classmethod
    def tearDownClass(cls):
        cls.limits.restore_original_limits()

    def systems(self, K, shift):
        values, vectors = eigh(K, driver='evd')
        return (KernelLinearSystem(K, shift),
                SpectralLinearSystem(K, shift, vectors, np.maximum(values, 0.)))

    def hidden_roundoff(self, n=8):
        K = np.ones((n, n))
        x = np.zeros(n)
        x[:3] = [2.**53, 1., -2.**53]
        shift = 2.**-60
        rhs = rounded_left_product(K, x) + shift*x
        return K, shift, rhs, x

    def test_zero_ordinary_residual_does_not_authorize_inaccurate_state(self):
        for n in (4, 8, 16, 32):
            K, shift, rhs, x = self.hidden_roundoff(n)
            for system in self.systems(K, shift):
                with self.subTest(n=n, system=type(system).__name__):
                    product = rounded_left_product(K, x)
                    residual = rhs-product-shift*x
                    # A zero measured residual alone can be wrong by one.
                    self.assertTrue(system._assess(rhs, 1., x, product, residual, 0.)[0])
                    maximum, constraint = exact_residual(K, shift, rhs, x, 0., 1.)
                    self.assertEqual(maximum, Decimal(1))
                    self.assertEqual(constraint, Decimal(0))
                    self.assertFalse(system._ordinary_measure(rhs, 1., x, 0., product=product)[0])

    def test_hidden_roundoff_with_zero_sum_and_rank_one_psd_kernel(self):
        # The optimized MM gauge is zero, so necessity is not limited to the
        # nonzero target used by coefficient-reference inverse actions.
        for n in (4, 8, 16, 32):
            v = np.ones(n)
            v[3] = 2.
            K = np.outer(v, v)
            x = np.zeros(n)
            x[:4] = [2.**53, 1., -2.**53, -1.]
            shift = 2.**-60
            product = rounded_left_product(K, x)
            rhs = product + shift*x
            maximum, constraint = exact_residual(K, shift, rhs, x, 0., 0.)
            self.assertGreater(maximum, Decimal('1.9'))
            self.assertEqual(constraint, Decimal(0))
            for system in self.systems(K, shift):
                self.assertTrue(system._assess(rhs, 0., x, product,
                                              rhs-product-shift*x, 0.)[0])
                self.assertFalse(system._ordinary_measure(rhs, 0., x, 0., product=product)[0])

    def test_native_trial_cannot_bypass_bounded_acceptance(self):
        K, shift, rhs, bad = self.hidden_roundoff()
        initial = np.zeros(len(K))
        for system in self.systems(K, shift):
            original = initial.copy()
            sentinel = (False, rhs.copy(), 1., np.zeros(len(K)), 1., 1.)
            with self.subTest(system=type(system).__name__), patch.object(
                    system, '_candidate', return_value=(bad.copy(), 0.)), patch.object(
                    system, '_compensated_measure', return_value=sentinel) as accurate:
                result_x, result_s, measurement = system._measure_with_native_trial(
                    rhs, 1., initial, 0.)
            np.testing.assert_array_equal(result_x, original)
            np.testing.assert_array_equal(initial, original)
            self.assertEqual(result_s, 0.)
            self.assertIs(measurement, sentinel)
            self.assertEqual(system.info['native_refinement_discarded'], 1)
            self.assertNotIn('native_refinement_acceptances', system.info)
            self.assertEqual(accurate.call_count, 1)
            np.testing.assert_array_equal(accurate.call_args.args[2], original)

    def test_well_conditioned_scaled_solves_still_use_fast_screen(self):
        rng = np.random.RandomState(613)
        X = rng.normal(size=(12, 5))
        base = X@X.T + 2.*np.eye(12)
        counts = 0
        for exponent in (-120, 0, 120):
            scale = 2.**exponent
            K, shift = scale*base, .3*scale
            rhs = scale*rng.normal(size=12)
            for system in self.systems(K, shift):
                with self.subTest(exponent=exponent, system=type(system).__name__), patch.object(
                        system, '_compensated_measure', side_effect=AssertionError('Unnecessary compensated pass')):
                    x, scalar = system.solve_constrained(rhs, .125)
                maximum, constraint = exact_residual(K, shift, rhs, x, scalar, .125)
                tolerance = Decimal('1e-10')*max(Decimal(1), max(map(abs, map(dec, rhs))))
                self.assertLessEqual(maximum, tolerance)
                self.assertLessEqual(maximum, dec(system.info['max_linear_residual']))
                self.assertLessEqual(constraint, dec(64*np.finfo(float).eps*max(1., math.fsum(abs(x)))))
                self.assertGreater(system.info['ordinary_residual_acceptances'], 0)
                counts += 1
        self.assertEqual(counts, 6)

    def test_cache_is_lazy_shared_and_linear_storage(self):
        K = np.diag(np.arange(1., 10.))
        for system in self.systems(K, .2):
            self.assertFalse(hasattr(system, '_ordinary_bounds'))
            system.solve_constrained(np.linspace(-1., 1., len(K)))
            cache = system._ordinary_bounds
            self.assertIsInstance(cache, OrdinaryResidualBounds)
            for value in vars(cache).values():
                if isinstance(value, np.ndarray):
                    self.assertEqual(value.shape, (len(K),))
            system.solve_constrained(np.linspace(1., -1., len(K)))
            self.assertIs(system._ordinary_bounds, cache)

    def test_unsupported_rounding_cannot_populate_or_use_cache(self):
        K = np.eye(4)
        with patch('dwd._ordinary_residual_bounds._round_to_nearest', return_value=False):
            with self.assertRaisesRegex(FloatingPointError, 'round-to-nearest'):
                OrdinaryResidualBounds(K, K.mean(axis=0))
        system = KernelLinearSystem(K, .2)
        x, scalar = system._candidate(np.array([1., -1., 2., -2.]), 0.)
        self.assertTrue(system._ordinary_measure(np.array([1., -1., 2., -2.]), 0., x, scalar)[0])
        with patch('dwd._ordinary_residual_bounds._round_to_nearest', return_value=False):
            self.assertFalse(system._ordinary_measure(np.array([1., -1., 2., -2.]), 0., x, scalar)[0])

    def test_product_underflow_declines_fast_screen(self):
        K = np.eye(4)*2.**-600
        helper = OrdinaryResidualBounds(K, K.mean(axis=0))
        x = np.array([1., -1., 1., -1.])*2.**-600
        with self.assertRaisesRegex(FloatingPointError, 'underflow'):
            helper.measure(np.zeros(4), 0., x, 2.**-600, 0., K@x,
                           np.zeros(4), 0., 0.)

    def test_unrepresentable_positive_row_bound_declines(self):
        K = np.full((4, 4), 1e308)
        with np.errstate(over='ignore'):
            mean = K.mean(axis=0)
        with self.assertRaises(FloatingPointError):
            OrdinaryResidualBounds(K, mean)

    def test_block_products_enclose_independent_decimal_scores(self):
        rng = np.random.RandomState(5093)
        for n in (33, 96, 257):
            X = rng.normal(size=(n, 7))
            base = rbf_kernel(X, gamma=.001)
            x = rng.normal(size=n)*2.**30
            for exponent in (-120, 0, 120):
                K = np.ldexp(base, exponent)
                helper = OrdinaryResidualBounds(K, K.mean(axis=0))
                for layout in ('C', 'F', 'reverse'):
                    matrix = np.array(K, order=layout) if layout != 'reverse' else K[::-1, ::-1]
                    coefficients = x if layout != 'reverse' else x[::-1]
                    cache = helper if layout != 'reverse' else OrdinaryResidualBounds(matrix, matrix.mean(axis=0))
                    scores, bounds = cache.blocked_product(matrix, coefficients)
                    with self.subTest(n=n, exponent=exponent, layout=layout), localcontext() as context:
                        context.prec = 120
                        exact_x = list(map(dec, coefficients))
                        for i, row in enumerate(matrix):
                            expected = sum((dec(a)*b for a, b in zip(row, exact_x)), Decimal(0))
                            self.assertLessEqual(abs(dec(scores[i])-expected), dec(bounds[i]))

    def test_public_near_constant_rbf_accepted_equations(self):
        X = np.random.RandomState(83).normal(size=(96, 6))
        K = rbf_kernel(X, gamma=2.**-28)
        y = np.where(np.arange(96)%3 == 0, 1., -1.)
        for implementation, backend, stopping in (
                ('optimized', 'cholesky', 'objective'),
                ('optimized', 'cholesky', 'fixed'),
                ('reference', 'spectral', 'fixed')):
            checked = []
            original_cholesky = KernelLinearSystem.solve_constrained
            original_spectral = SpectralLinearSystem.solve_constrained
            original_refine = SpectralLinearSystem.refine_candidate
            def wrap(action):
                def checked_action(system, rhs, target_sum=0.):
                    result = action(system, rhs, target_sum)
                    maximum, constraint = exact_residual(system.K, system.shift, rhs,
                                                         result[0], result[1], target_sum)
                    tolerance = Decimal('1e-10')*max(Decimal(1), max(map(abs, map(dec, rhs))))
                    self.assertLessEqual(maximum, tolerance)
                    self.assertLessEqual(constraint, dec(64*np.finfo(float).eps*max(
                        1., math.fsum(abs(result[0])), abs(target_sum))))
                    checked.append(float(maximum))
                    return result
                return checked_action
            def checked_refine(system, rhs, x, scalar, scores):
                result = original_refine(system, rhs, x, scalar, scores)
                maximum, constraint = exact_residual(system.K, system.shift, rhs,
                                                     result[0], result[1], 0.)
                tolerance = Decimal('1e-10')*max(Decimal(1), max(map(abs, map(dec, rhs))))
                self.assertLessEqual(maximum, tolerance)
                self.assertLessEqual(constraint, dec(64*np.finfo(float).eps*max(
                    1., math.fsum(abs(result[0])))))
                checked.append(float(maximum))
                return result
            with self.subTest(implementation=implementation, backend=backend, stopping=stopping), patch.object(
                    KernelLinearSystem, 'solve_constrained', wrap(original_cholesky)), patch.object(
                    SpectralLinearSystem, 'solve_constrained', wrap(original_spectral)), patch.object(
                    SpectralLinearSystem, 'refine_candidate', checked_refine):
                model = KernGDWD(kernel='precomputed', lambd=2.**-18/48.,
                                 initialization='zero', random_state=314159,
                                 implementation=implementation, backend=backend,
                                 stopping=stopping, max_iter=100).fit(K, y)
                self.assertTrue(np.isfinite(model.decision_function(K)).all())
            self.assertGreaterEqual(len(checked), 1)
            if stopping == 'fixed':
                self.assertGreaterEqual(len(checked), 100)


if __name__ == '__main__':
    unittest.main(verbosity=2)
