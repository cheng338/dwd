"""Independent regressions for feasible kernel-DWD dual diagnostics."""
import unittest
from unittest.mock import patch
from decimal import Decimal, localcontext
from fractions import Fraction
import numpy as np
from dwd._dual_certificate import (
    feasible_dual_slopes, kernel_dual_lower_bound, _conjugate_mean_lower,
    _quadratic_upper,
)
from dwd._kernel_solver import solve_kernel
from dwd.gen_dwd import V_grad


def fraction(value):
    return Fraction.from_float(float(value))


def exact_quadratic(K, x):
    values = [fraction(v) for v in x]
    return sum((left * fraction(entry) * right
                for left, row in zip(values, K) for entry, right in zip(row, values)), Fraction(0))


def constant_optimum(q, positive=13, negative=19):
    # Minimize over the free constant prediction; alpha=0 realizes the optimum.
    # The majority margin is t*(negative/positive)**(1/(q+1)).
    with localcontext() as ctx:
        ctx.prec = 100
        q = Decimal.from_float(float(q))
        ratio = Decimal(negative) / Decimal(positive)
        power = (ratio.ln() / (q + 1)).exp()
        objective = Decimal(positive) * (1 + power) / Decimal(positive + negative)
        margin = q / (q + 1) * power
        return objective, margin


class FeasibleDualCertificateTests(unittest.TestCase):
    def test_class_balance_is_exact_not_a_rounded_dot(self):
        rng = np.random.RandomState(7492)
        for n in (2, 33, 1024):
            y = np.ones(n); y[:max(1, n * 3 // 5)] = -1
            for raw in (rng.uniform(size=n), np.ones(n), np.full(n, 1e-300)):
                rho = feasible_dual_slopes(raw, y)
                self.assertTrue(np.all((rho >= 0) & (rho <= 1)))
                self.assertEqual(sum((fraction(a) * fraction(b) for a, b in zip(y, rho)), Fraction(0)), 0)

    def test_conjugate_bound_general_q_and_subnormals(self):
        values = np.array([0., np.nextafter(0., 1.), 2.**-52, .13, .5, 1.])
        for q in (1e-300, .1, 1., 2., 10., 1e300):
            with self.subTest(q=q), localcontext() as ctx:
                ctx.prec = 180
                exponent = Decimal.from_float(q) / (Decimal.from_float(q) + 1)
                expected = sum((Decimal.from_float(float(v)) ** exponent for v in values if v), Decimal(0)) / len(values)
                lower = _conjugate_mean_lower(values, q)
                computed = Decimal(lower.numerator) / Decimal(lower.denominator)
                self.assertLessEqual(computed, expected)
                self.assertGreaterEqual(computed, 0)

    def test_quadratic_upper_on_cancellation_and_layouts(self):
        rng = np.random.RandomState(8233)
        for n in (9, 63, 65, 128):
            features = rng.randint(-3, 4, size=(n, 4)).astype(float) / 8
            K = 2.**20 * np.ones((n, n)) + features @ features.T
            y = np.where(np.arange(n) % 2, 1., -1.)
            signed = y * feasible_dual_slopes(rng.uniform(size=n), y)
            expected = exact_quadratic(K, signed)
            for order in ('C', 'F'):
                for target in (Fraction(10**20), Fraction(1, 10**100)):
                    upper, method = _quadratic_upper(np.array(K, order=order), signed, target)
                    self.assertGreaterEqual(upper, expected)
                    if n <= 64 and target < 1:
                        self.assertEqual(method, 'exact_small_escalation')
                        self.assertEqual(upper, expected)
                    else:
                        self.assertEqual(method, 'bounded_binary64')

    def test_constant_kernel_bound_below_independent_optimum(self):
        n = 32
        y = np.where(np.arange(n) % 2, 1., -1.); y[:6] = -1
        for q in (.1, 1., 10.):
            optimum, margin = constant_optimum(q)
            raw = -V_grad(y * -float(margin), q=q)
            for scale in (0., 1., 2.**54):
                for penalty in (1e-300, 1e-30, .1, 1e100):
                    for order in ('C', 'F'):
                        lower, info = kernel_dual_lower_bound(np.full((n, n), scale, order=order), y, raw, penalty, q)
                        self.assertLessEqual(Decimal.from_float(lower), optimum)
                        self.assertLess(float(optimum) - lower, 2e-13)
                        self.assertEqual(info['dual_quadratic_method'], 'exact_constant')
                        self.assertIsNone(info['fallback_reason'])

    def test_zero_slopes_fallback_is_explicit_not_a_negative_gap_clip(self):
        n = 12
        y = np.where(np.arange(n) % 2, 1., -1.)
        lower, info = kernel_dual_lower_bound(np.eye(n), y, np.ones(n), 1e-100, .3)
        self.assertEqual(lower, 0.)
        self.assertEqual(info['fallback_reason'], 'zero_slopes_give_a_stronger_certified_bound')
        self.assertTrue(info['primal_objective_and_gap_are_estimates'])

    def test_invalid_negative_quadratic_is_not_hidden_by_fallback(self):
        y = np.r_[-np.ones(6), np.ones(6)]
        with self.assertRaisesRegex(ValueError, 'negative dual quadratic'):
            kernel_dual_lower_bound(-np.eye(12), y, np.ones(12), .1, 1.)
        with self.assertRaises(ValueError):
            solve_kernel(-np.eye(12), y, .1, max_iter=1)

    def test_asymmetric_kernel_uses_explicit_zero_certificate_without_rejection(self):
        n = 12
        y = np.where(np.arange(n) % 2, 1., -1.)
        K = np.eye(n); K[0, 1] = np.finfo(float).eps
        lower, info = kernel_dual_lower_bound(K, y, np.ones(n), .1, 1.)
        self.assertEqual(lower, 0.)
        self.assertEqual(info['fallback_reason'], 'asymmetric_kernel_no_nonzero_dual_certificate')
        self.assertEqual(info['kernel_assumption'], 'nonnegative represented RKHS penalty')
        result = solve_kernel(K, y, .1, max_iter=5, stopping='fixed')
        self.assertEqual(result['n_iter'], 5)
        self.assertEqual(result['diagnostics']['dual_certificate']['fallback_reason'],
                         'asymmetric_kernel_no_nonzero_dual_certificate')

    def test_unsupported_rounding_declines_nonzero_bound(self):
        y = np.r_[-np.ones(6), np.ones(6)]
        with patch('dwd._dual_certificate._round_to_nearest', return_value=False):
            lower, info = kernel_dual_lower_bound(np.eye(12), y, np.ones(12), .1, 1.)
        self.assertEqual(lower, 0.)
        self.assertEqual(info['fallback_reason'], 'nonzero_dual_bound_unavailable')

    def test_historical_constant_fits_are_not_rejected(self):
        n = 32
        y = np.where(np.arange(n) % 2, 1., -1.); y[:6] = -1
        for q in (.1, 1., 10.):
            optimum, _ = constant_optimum(q)
            for delta in (1e-30, 1e-100):
                penalty = delta / (2 * n * q / (q + 1)**2)
                for order in ('C', 'F'):
                    for implementation in ('optimized', 'reference'):
                        with self.subTest(q=q, delta=delta, order=order, implementation=implementation):
                            result = solve_kernel(np.ones((n, n), order=order), y, penalty, q=q,
                                implementation=implementation, stopping='fixed', max_iter=5)
                            self.assertEqual(result['n_iter'], 5)
                            self.assertLessEqual(Decimal.from_float(result['dual_objective']), optimum)
                            self.assertGreaterEqual(result['dual_gap'], -1e-12)
                            self.assertEqual(result['dual_equality_residual'], 0.)

    def test_nearly_optimal_constant_fit_retains_accurate_certificate(self):
        n = 32
        y = np.where(np.arange(n) % 2, 1., -1.); y[:6] = -1
        penalty = 1e-30 / (2 * n / 4)
        optimum, _ = constant_optimum(1.)
        for order in ('C', 'F'):
            result = solve_kernel(np.ones((n, n), order=order), y, penalty, q=1.,
                stopping='optimality', max_iter=100)
            self.assertEqual(result['n_iter'], 39)
            self.assertLess(abs(result['final_objective'] - float(optimum)), 1e-11)
            self.assertLessEqual(Decimal.from_float(result['dual_objective']), optimum)
            self.assertLess(result['dual_gap'], 1e-9)
            self.assertEqual(result['dual_equality_residual'], 0.)


if __name__ == '__main__':
    unittest.main()
