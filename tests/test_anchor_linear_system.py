"""Independent original-equation controls for anchor LU candidates."""
from decimal import Decimal, localcontext
import math
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import solve

import dwd._anchor_linear_system as anchor


def decimal_residual(K, shift, rhs, x, scalar, target):
    with localcontext() as context:
        context.prec = 80
        number = lambda value: Decimal.from_float(float(value))
        coefficients = list(map(number, x))
        residual = [number(rhs[i])-sum((number(k)*v for k, v in zip(row, coefficients)), Decimal(0))
                    -number(shift)*coefficients[i]-number(scalar) for i, row in enumerate(K)]
        return max(map(abs, residual)), abs(number(target)-sum(coefficients, Decimal(0)))


class AnchorLinearSystemTests(unittest.TestCase):
    def test_original_bordered_solve_and_decimal_residuals(self):
        rng = np.random.default_rng(52)
        F = rng.normal(size=(8, 3))
        matrices = {'spd': F@F.T+np.eye(8), 'singular': F@F.T,
                    'zero': np.zeros((8, 8)), 'constant': np.ones((8, 8)),
                    'nearly_constant': np.ones((8, 8))+2.**-20*(F@F.T)}
        rounded = matrices['nearly_constant'].copy()
        rounded[1, 2] = np.nextafter(rounded[1, 2], np.inf)
        matrices['rounding_asymmetric'] = rounded
        rhs, shift = rng.normal(size=8), .125
        for name, K in matrices.items():
            before = K.copy(), rhs.copy()
            K.flags.writeable = False
            system = anchor.AnchorLinearSystem(K, shift)
            self.assertEqual(system.info['representation'], 'anchor_lu')
            self.assertEqual(system.info['anchor_index'], 0)
            self.assertEqual(system.info['reduced_dimension'], 7)
            for target in (0., .37, -2.):
                with self.subTest(matrix=name, target=target):
                    x, scalar = system.candidate(rhs, target)
                    bordered = np.block([[K+shift*np.eye(8), np.ones((8, 1))],
                                         [np.ones((1, 8)), np.zeros((1, 1))]])
                    expected = solve(bordered, np.r_[rhs, target], assume_a='gen')
                    assert_allclose(np.r_[x, scalar], expected, rtol=2e-12, atol=2e-12)
                    residual, constraint = decimal_residual(K, shift, rhs, x, scalar, target)
                    self.assertLess(float(residual), 2e-12)
                    self.assertLess(float(constraint), 2e-13)
                    self.assertEqual(x.dtype, np.dtype('float64'))
            assert_array_equal(K, before[0]); assert_array_equal(rhs, before[1])

    def test_grouped_reduction_preserves_stored_asymmetry(self):
        K = np.full((4, 4), 1.-2.**-20)
        np.fill_diagonal(K, 1.)
        K[1, 2] = np.nextafter(K[1, 2], np.inf)
        shift = 2.**-24
        saved = K.copy()
        original_factor = anchor.lu_factor
        captured = []
        def factor(matrix, **kwargs):
            captured.append(matrix.copy())
            return original_factor(matrix, **kwargs)
        with patch.object(anchor, 'lu_factor', side_effect=factor):
            system = anchor.AnchorLinearSystem(K, shift)
        with localcontext() as context:
            context.prec = 80
            number = lambda value: Decimal.from_float(float(value))
            expected = np.array([[float(number(K[i, j])-number(K[i, 0])-number(K[0, j])
                          +number(K[0, 0])+number(shift)*(1+(i == j))) for j in range(1, 4)]
                          for i in range(1, 4)])
        assert_array_equal(captured[0], expected)
        self.assertNotEqual(captured[0][0, 1], captured[0][1, 0])
        self.assertGreater(system.info['reduced_symmetry_error'], 0.)
        assert_array_equal(K, saved)

    def test_single_observation_needs_no_factor(self):
        K, rhs, shift = np.array([[3.5]]), np.array([-2.]), .25
        with patch.object(anchor, 'lu_factor', side_effect=AssertionError('Unneeded factor')):
            system = anchor.AnchorLinearSystem(K, shift)
            for target in (0., .37, -3.):
                x, scalar = system.candidate(rhs, target)
                self.assertEqual(x[0], target)
                residual, constraint = decimal_residual(K, shift, rhs, x, scalar, target)
                self.assertLess(float(residual), 1e-15)
                self.assertEqual(constraint, 0)
        self.assertEqual(system.info['reduced_dimension'], 0)
        self.assertIsNone(system.info['rcond'])

    def test_singular_factor_warning_becomes_clean_failure(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with self.assertRaisesRegex(FloatingPointError, 'singular'):
                anchor.AnchorLinearSystem(-.5*np.eye(3), .5)
        self.assertEqual(caught, [])

    def test_zero_condition_estimate_does_not_reject_accurate_candidates(self):
        with patch.object(anchor, 'dgecon', return_value=(0., 0)):
            system = anchor.AnchorLinearSystem(np.eye(3), .5)
        x, scalar = system.candidate(np.array([1., 2., 3.]), .37)
        self.assertEqual(system.info['rcond'], 0.)
        self.assertLess(abs(math.fsum(x)-.37), 1e-14)
        assert_allclose(1.5*x+scalar, [1., 2., 3.], rtol=0, atol=1e-14)

    def test_invalid_inputs_and_unrepresentable_arithmetic_fail(self):
        for K in (np.empty((0, 0)), np.zeros((2, 3)), np.eye(2).astype(complex)):
            with self.subTest(K=K), self.assertRaises(ValueError):
                anchor.AnchorLinearSystem(K, .5)
        for shift in (0., -1., np.nan, np.inf, True, 1j, [.1]):
            with self.subTest(shift=shift), self.assertRaises(ValueError):
                anchor.AnchorLinearSystem(np.eye(2), shift)
        with self.assertRaises(FloatingPointError):
            anchor.AnchorLinearSystem(np.full((2, 2), np.nan), .5)
        huge = np.finfo(float).max
        with self.assertRaises(FloatingPointError):
            anchor.AnchorLinearSystem(np.array([[huge, -huge], [-huge, huge]]), .5)
        system = anchor.AnchorLinearSystem(np.eye(3), .5)
        for rhs, target in (([1., 2.], 0.), ([1., np.nan, 3.], 0.), ([1., 2., 3.], np.inf)):
            with self.assertRaises(FloatingPointError):
                system.candidate(rhs, target)
        with self.assertRaises(ValueError):
            system.candidate(np.ones(3, complex), 0.)
        with patch.object(anchor, 'lu_solve', return_value=np.full(2, np.inf)):
            with self.assertRaisesRegex(FloatingPointError, 'Nonfinite'):
                system.candidate(np.ones(3), 0.)


if __name__ == '__main__':
    unittest.main()
