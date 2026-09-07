"""Independent structured-PSD oracles and fault tests for the auxiliary bound."""
from decimal import Decimal, localcontext
import math
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

import dwd._kernel_linear_system as linear
from dwd._kernel_linear_system import KernelLinearSystem
from dwd._spectral_linear_system import SpectralLinearSystem
from dwd._compensated_residual import compensated_residual
from dwd.gen_kern_dwd import KernGDWD


def decimal(value):
    return Decimal.from_float(float(value))


def structured_solution(K, shift, rhs, target=0.):
    """Independent Decimal100 closed form for diag(d)+constant*11.T."""
    with localcontext() as context:
        context.prec = 100
        constant = decimal(K[0, 1])
        d = [decimal(K[i, i])-constant for i in range(len(K))]
        inverse = [1/(value+decimal(shift)) for value in d]
        h = (sum((decimal(r)*v for r, v in zip(rhs, inverse)), Decimal(0))-decimal(target))/sum(inverse, Decimal(0))
        x = [(decimal(r)-h)*v for r, v in zip(rhs, inverse)]
        return x, h-constant*decimal(target)


def structured_product(K, x):
    with localcontext() as context:
        context.prec = 100
        constant = decimal(K[0, 1])
        dx = list(map(decimal, x))
        total = sum(dx, Decimal(0))
        return np.array([float((decimal(K[i,i])-constant)*v+constant*total) for i, v in enumerate(dx)])


def actual_error(K, x, ideal):
    with localcontext() as context:
        context.prec = 100
        constant = decimal(K[0, 1])
        error = [decimal(v)-w for v, w in zip(x, ideal)]
        squared = sum(((decimal(K[i,i])-constant)*e*e for i, e in enumerate(error)), Decimal(0))
        squared += constant*sum(error, Decimal(0))**2
        return float(squared.sqrt())


class AposterioriRKHSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limit = threadpool_limits(4)
        rng = np.random.RandomState(91330)
        rng.randint(-3, 4, size=(30, 5))
        rng.normal(size=(30, 4))
        cls.K = np.diag(np.exp2(np.linspace(-30, 30, 30)))+1024*np.ones((30, 30))
        cls.rhs = rng.normal(size=30)+.23

    @classmethod
    def tearDownClass(cls):
        cls.limit.restore_original_limits()

    def accurate_state(self, shift):
        ideal, ideal_s = structured_solution(self.K, shift, self.rhs)
        x, s = np.array(list(map(float, ideal))), float(ideal_s)
        residual = compensated_residual(self.K, shift, self.rhs, x, s, 0.)[0]
        s += math.fsum(residual)/len(x)
        return x, s, ideal

    def test_accurate_single_action_bounds_states_without_overwriting_original_scores(self):
        values, vectors = eigh(self.K, driver='evd')
        for shift in (1e-14, 1e-6):
            x, s, ideal = self.accurate_state(shift)
            for system in (KernelLinearSystem(self.K, shift),
                           SpectralLinearSystem(self.K, shift, vectors, np.maximum(values, 0.))):
                with self.subTest(shift=shift, action=type(system).__name__):
                    original_x = x.copy()
                    sentinel = np.arange(len(x), dtype=float)
                    system.last_product = sentinel
                    # This test isolates the one-action bound algebra. A raw
                    # normwise-accurate EVD need not give an accurate inverse
                    # for tiny modes of this ill-scaled matrix on every BLAS.
                    # Native inverse/recovery behavior is tested separately.
                    def accurate_action(rhs, target):
                        correction, scalar = structured_solution(self.K, shift, rhs, target)
                        return np.array(list(map(float, correction))), float(scalar)
                    with patch.object(system, 'solve_constrained', side_effect=AssertionError('Recursive solve')), \
                            patch.object(system, '_candidate', side_effect=accurate_action) as action:
                        good, _, _, scores, _, estimate = system._measure(self.rhs, 0., x, s)
                    self.assertTrue(good)
                    self.assertEqual(action.call_count, 1)
                    self.assertIs(system.last_product, sentinel)
                    assert_array_equal(x, original_x)
                    assert_array_equal(scores, structured_product(self.K, x))
                    actual = actual_error(self.K, x, ideal)
                    self.assertLessEqual(actual, estimate)
                    self.assertEqual(system.info['aposteriori_rkhs_acceptances'], 1)
                    details = system.info['last_aposteriori_rkhs_check']
                    self.assertGreater(details['global_bound'], details['tolerance'])
                    self.assertLessEqual(estimate, details['tolerance'])

    def test_actual_production_solver_returns_both_original_ill_scaled_cases(self):
        values, vectors = eigh(self.K, driver='evd')
        for shift in (1e-14, 1e-6):
            for system in (KernelLinearSystem(self.K, shift),
                           SpectralLinearSystem(self.K, shift, vectors, np.maximum(values, 0.))):
                with self.subTest(shift=shift, action=type(system).__name__):
                    original_rhs, original_K = self.rhs.copy(), self.K.copy()
                    x, s = system.solve_constrained(self.rhs)
                    ideal, _ = structured_solution(self.K, shift, self.rhs)
                    residual, constraint, scores, _, _ = compensated_residual(self.K, shift, self.rhs, x, s, 0.)
                    detail = {'shift': shift, 'action': type(system).__name__,
                              'representation': system.info['factor_representation'],
                              'linear_residual': float(np.max(abs(residual))),
                              'last_bound': system.info.get('last_aposteriori_rkhs_check')}
                    self.assertLessEqual(np.max(abs(residual)), 1e-10*max(1., np.max(abs(self.rhs))), msg=detail)
                    self.assertLessEqual(abs(constraint), 64*np.finfo(float).eps*max(1., math.fsum(abs(x))), msg=detail)
                    tolerance = 5e-7*max(1., np.sqrt(abs(float(x@scores))))
                    self.assertLessEqual(actual_error(self.K, x, ideal), tolerance, msg=detail)
                    # Required behavior is an accurate native solve, regardless
                    # of whether its provider needs scalar/bound refinements.
                    self.assertEqual(system.info['linear_solves'], 1)
                    self.assertEqual(system.info['added_objective_regularization'], 0.)
                    self.assertEqual(system.info['positive_eigenvalues_discarded'], 0)
                    assert_array_equal(system.last_product, scores)
                    assert_array_equal(self.rhs, original_rhs)
                    assert_array_equal(self.K, original_K)

    def test_normal_public_dwd_fit_uses_bound_and_matches_independent_mm_path(self):
        n, shift = 30, 1e-14
        K = np.diag(np.exp2(np.linspace(-15, 15, n)))+1024*np.ones((n, n))
        y = np.tile([-1., 1.], n//2)
        lambd = shift/(n/2.)
        states = []
        model = KernGDWD(kernel='precomputed', implementation='optimized', initialization='zero',
                        lambd=lambd, max_iter=5, stopping='fixed', callback=lambda s: states.append(dict(s)))
        model.fit(K, y)
        self.assertEqual(model.n_iter_, 5)
        self.assertGreater(model.diagnostics_['linear_system_diagnostics']['aposteriori_rkhs_acceptances'], 0)
        alpha, b = np.zeros(n), 0.
        expected_decisions, expected_objectives = [], []
        for iteration in range(6):
            g = structured_product(K, alpha)
            margin = y*(g+b)
            loss = 1-margin
            tail = margin > .5
            loss[tail] = .25/margin[tail]
            expected_decisions.append(g+b)
            expected_objectives.append(float(np.mean(loss)+lambd*math.fsum(alpha*g)))
            if iteration == 5:
                break
            derivative = -np.ones(n)
            derivative[tail] = -.25/margin[tail]**2
            exact_x, exact_s = structured_solution(K, shift, g-y*derivative/4.)
            alpha = np.array(list(map(float, exact_x)))
            b += float(exact_s)
        assert_allclose([s['decision_values'] for s in states], expected_decisions, rtol=0., atol=2e-7)
        assert_allclose([s['objective'] for s in states], expected_objectives, rtol=0., atol=5e-8)
        self.assertAlmostEqual(model.final_objective_, expected_objectives[-1], delta=5e-8)
        assert_array_equal(states[0]['alpha'], np.zeros(n))
        self.assertEqual(states[0]['offset'], 0.)

    def test_inaccurate_or_nonfinite_auxiliary_action_cannot_certify(self):
        x, s, _ = self.accurate_state(1e-6)
        for value in (0., np.nan, 1e300):
            with self.subTest(value=value):
                system = KernelLinearSystem(self.K, 1e-6)
                with patch.object(system, '_candidate', return_value=(np.full(30, value), 0.)) as action:
                    measured = system._measure(self.rhs, 0., x, s)
                self.assertFalse(measured[0])
                self.assertEqual(action.call_count, 1)
                self.assertFalse(system.info['last_aposteriori_rkhs_check']['accepted'])
                self.assertIsNone(system.last_product)

    def test_true_feature_error_cannot_be_hidden_by_passing_equation_and_sum_gates(self):
        K, shift = np.ones((4, 4)), 1e-12
        x = np.array([2.**40, -2.**40, .001, 0.])
        scores = np.full(4, .001)
        rhs = scores+shift*x
        system = KernelLinearSystem(K, shift)
        measured = system._measure(rhs, 0., x, 0.)
        self.assertFalse(measured[0])
        self.assertTrue(system._equations_acceptable(rhs, 0., x, measured[1], measured[2]))
        self.assertGreater(system.info['last_aposteriori_rkhs_check']['estimate'], 5e-7)
        self.assertFalse(system.info['last_aposteriori_rkhs_check']['accepted'])

    def test_both_residual_and_constraint_uncertainties_are_propagated(self):
        system = KernelLinearSystem(np.diag([1., 2., 3., 4.]), .1)
        first = np.array([1e-8, 2e-8, 3e-8, 4e-8])
        second = first[::-1].copy()
        c1, c2 = 2e-9, 3e-9
        zero = np.zeros(4)
        with patch.object(system, '_candidate', return_value=(zero, 0.)), \
                patch.object(linear, 'compensated_residual', return_value=(zero, 0., zero, second, c2)):
            estimate, details = system._aposteriori_rkhs_bound(zero, 0., first, c1)
        expected = (np.linalg.norm(first+second)+(c1+c2)*np.linalg.norm(system.mean-system.mean.mean()))/(2*np.sqrt(.1))
        expected += (c1+c2)*np.sqrt(system.mean.mean())
        self.assertGreaterEqual(estimate, expected)
        self.assertGreaterEqual(details['combined_residual_allowance_max'], np.max(first+second))
        self.assertGreaterEqual(details['combined_constraint_allowance'], c1+c2)

    def test_invalid_quadratic_upper_bound_fails_closed(self):
        x, s, _ = self.accurate_state(1e-6)
        for q, bound in ((-2., 1.), (np.inf, 0.), (0., np.nan)):
            with self.subTest(q=q, bound=bound):
                system = KernelLinearSystem(self.K, 1e-6)
                with patch.object(linear, '_compensated_quadratic', return_value=(np.zeros(30), q, bound)):
                    measured = system._measure(self.rhs, 0., x, s)
                self.assertFalse(measured[0])
                self.assertIn('error', system.info['last_aposteriori_rkhs_check'])

    def test_fast_path_avoids_both_auxiliary_action_and_compensation(self):
        K = np.diag([.1, .3, .7, 1.3])
        system = KernelLinearSystem(K, .2)
        with patch.object(system, '_aposteriori_rkhs_bound', side_effect=AssertionError('Unneeded auxiliary check')), \
                patch.object(linear, 'compensated_residual', side_effect=AssertionError('Unneeded compensation')):
            x, s = system.solve_constrained(np.array([.3, -.1, .7, -.4]))
        assert_array_equal(system.last_product, K@x)
        self.assertNotIn('aposteriori_rkhs_checks', system.info)


if __name__ == '__main__':
    unittest.main(verbosity=2)
