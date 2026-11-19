"""Release-arithmetic compatibility when accelerating compensated residuals.

The frozen arithmetic below retains the pre-optimization product construction
and NumPy-scalar accumulation. Exact rational correctness is tested separately
in test_compensated_residual; these tests ensure performance changes also keep
the established binary64 trajectory, including signs of zero and error paths.
"""
import unittest
from unittest.mock import patch

import numpy as np

import dwd._kernel_linear_system as linear
from dwd._compensated_residual import (
    _fsum, _gradual_underflow, _products, compensated_residual,
)
from dwd._kernel_solver import solve_kernel
from dwd._spectral_linear_system import SpectralLinearSystem


def _release_products(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise FloatingPointError('Invalid compensated product inputs.')
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        am, ae = np.frexp(a)
        bm, be = np.frexp(b)
        splitter = 134217729.
        ca, cb = splitter * am, splitter * bm
        ah, bh = ca - (ca - am), cb - (cb - bm)
        al, bl = am - ah, bm - bh
        high = am * bm
        low = ((ah * bh - high) + ah * bl + al * bh) + al * bl
        exponent = ae + be
        high, low = np.ldexp(high, exponent), np.ldexp(low, exponent)
    if not np.isfinite(high).all() or not np.isfinite(low).all():
        raise FloatingPointError('Nonfinite or overflowing compensated products.')
    return high, low


def _release_residual(K, shift, rhs, x, s, target_sum):
    eta = _gradual_underflow()
    n = len(x)
    shifted_hi, shifted_lo = _release_products(np.full(n, shift), x)
    residual, scores = np.empty(n), np.empty(n)
    for i, row in enumerate(K):
        hi, lo = _release_products(row, x)
        scores[i] = _fsum([*hi, *lo])
        residual[i] = _fsum([float(rhs[i]), -float(s), -float(shifted_hi[i]),
                             -float(shifted_lo[i]), *(-hi), *(-lo)])
    constraint = _fsum([float(target_sum), *(-x)])
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        bounds = np.nextafter(4 * np.finfo(float).eps * np.abs(residual)
                             + (2 * n + 6) * eta, np.inf)
        constraint_bound = np.nextafter(
            4 * np.finfo(float).eps * abs(constraint) + 2 * eta, np.inf)
    if not np.isfinite(bounds).all() or not np.isfinite(constraint_bound):
        raise FloatingPointError('Unrepresentable compensated residual error allowance.')
    return residual, constraint, scores, bounds, float(constraint_bound)


class CompensatedResidualCompatibilityTests(unittest.TestCase):
    def assert_same_bits(self, actual, expected):
        actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.tobytes(), expected.tobytes())

    def assert_same_residual(self, args):
        for actual, expected in zip(compensated_residual(*args), _release_residual(*args)):
            self.assert_same_bits(actual, expected)

    def test_products_preserve_wide_scale_and_signed_zero_bits(self):
        rng = np.random.default_rng(894123)
        a = np.ldexp(rng.uniform(-1, 1, 1000), rng.integers(-1000, 1001, 1000))
        b = np.ldexp(rng.uniform(-1, 1, 1000), -np.frexp(a)[1])
        a = np.r_[a, 0., -0., 0., -0., np.finfo(float).tiny]
        b = np.r_[b, 1., 1., -1., -1., 2.**-53]
        for actual, expected in zip(_products(a, b), _release_products(a, b)):
            self.assert_same_bits(actual, expected)

    def test_residuals_preserve_bits_under_cancellation_and_strided_inputs(self):
        rng = np.random.default_rng(56121)
        for n in (1, 2, 19, 128):
            for scale in (0., 1., 1e7, 1e150):
                K = scale + rng.normal(size=(n, n)) * max(1., scale * 1e-7)
                x = rng.normal(size=n) * 1e5
                x[-1] = -sum(x[:-1])
                rhs = rng.normal(size=n)
                args = (K, 4.5e-6, rhs, x, -.27, .31)
                with self.subTest(n=n, scale=scale):
                    self.assert_same_residual(args)
                    self.assert_same_residual((K[::-1, ::-1], args[1], rhs[::-1],
                                               x[::-1], args[4], args[5]))

    def test_subnormal_products_and_signed_zero_residuals_preserve_bits(self):
        tiny = np.finfo(float).tiny
        with np.errstate(under='raise'):
            for K, x in ((np.array([[tiny, tiny / 2], [tiny / 2, tiny]]),
                          np.array([.75, -.5])),
                         (np.array([[0., -0.], [-0., 0.]]), np.array([-0., 0.]))):
                self.assert_same_residual((K, tiny, np.array([-0., 0.]), x, -0., -0.))

    def test_operand_cache_is_refreshed_between_calls(self):
        K = np.array([[2., .3, .2], [.3, 3., .4], [.2, .4, 4.]])
        x = np.array([1e6, -1e6, 1.])
        args = (K, .01, np.ones(3), x, 0., .5)
        self.assert_same_residual(args)
        x *= np.array([-1., .7, -1000.])
        self.assert_same_residual(args)

    def test_invalid_and_overflow_errors_are_unchanged(self):
        cases = [
            (np.array([[np.nan]]), 1., np.ones(1), np.ones(1), 0., 0.),
            (np.ones((1, 1)), 1., np.ones(1), np.array([np.inf]), 0., 0.),
            (np.array([[1e308]]), 1., np.ones(1), np.array([2.]), 0., 0.),
            (np.ones((2, 2)), 1., np.ones(2), np.ones(1), 0., 0.),
            (np.array([[1e308, 1e308], [1., 1.]]), 1., np.ones(2), np.ones(2), 0., 0.),
            (np.ones((1, 1)), 1., np.array([np.nan]), np.ones(1), 0., 0.),
        ]
        for args in cases:
            errors = []
            for function in (_release_residual, compensated_residual):
                with self.subTest(function=function.__name__, args=args):
                    try:
                        with np.errstate(all='raise'):
                            function(*args)
                    except FloatingPointError as exc:
                        errors.append((type(exc), str(exc)))
                    else:
                        self.fail('Invalid residual input did not raise.')
            self.assertEqual(errors[0], errors[1])

    def test_kernel_mm_trajectory_is_identical_when_compensation_is_required(self):
        rng = np.random.default_rng(58213)
        features = rng.normal(size=(16, 5))
        K = features @ features.T + np.eye(16) * .05
        y = np.r_[np.ones(9), -np.ones(7)]
        ordinary = linear.KernelLinearSystem._ordinary_measure

        def require_compensation(system, *args, **kwargs):
            result = ordinary(system, *args, **kwargs)
            return (False,) + result[1:]

        for q in (.3, 1., 3.5):
            for backend in ('cholesky',):
                fits = []
                for function in (_release_residual, compensated_residual):
                    with patch.object(linear, 'native_compensated_residual', return_value=None), patch.object(
                            linear, 'compensated_residual', function), patch.object(
                            linear.KernelLinearSystem, '_ordinary_measure', require_compensation):
                        fits.append(solve_kernel(K, y, .1, q=q, backend=backend,
                                                 max_iter=12, stopping='fixed'))
                with self.subTest(q=q, backend=backend):
                    for key in ('alpha', 'offset', 'objective_history', 'final_objective',
                                'rkhs_norm_squared', 'dual_gap'):
                        self.assert_same_bits(fits[0][key], fits[1][key])
                    for fit in fits:
                        self.assertGreater(fit['diagnostics']['linear_system_diagnostics'][
                            'compensated_residual_checks'], 0)

    def test_spectral_and_cholesky_constrained_solutions_preserve_bits(self):
        rng = np.random.default_rng(19017)
        features = rng.normal(size=(16, 5))
        K = features @ features.T + .25 * np.eye(16)
        rhs = rng.normal(size=16)
        values, vectors = np.linalg.eigh(K)
        for kind in ('cholesky', 'spectral'):
            results = []
            for function in (_release_residual, compensated_residual):
                system = (linear.KernelLinearSystem(K, .01) if kind == 'cholesky'
                          else SpectralLinearSystem(K, .01, vectors, values))
                ordinary = system._ordinary_measure

                def require_compensation(*args, **kwargs):
                    measured = ordinary(*args, **kwargs)
                    return (False,) + measured[1:]

                with patch.object(linear, 'native_compensated_residual', return_value=None), patch.object(
                        linear, 'compensated_residual', function), patch.object(
                        system, '_ordinary_measure', require_compensation):
                    results.append(system.solve_constrained(rhs, .23))
                self.assertGreater(system.info['compensated_residual_checks'], 0)
            for old, new in zip(*results):
                self.assert_same_bits(old, new)


if __name__ == '__main__':
    unittest.main(verbosity=2)
