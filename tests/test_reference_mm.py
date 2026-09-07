"""Independent bounded checks of the coefficient-coordinate reference update.

The main oracle solves the augmented MM equations, not the implementation's
constrained coefficient-direction system. A second oracle solves directly for
the next model, providing a different right-hand side and update formulation.
No data download, eigendecomposition, or optional CVXPY dependency is needed.
"""
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import solve

from dwd._reference_mm import reference_update


def slope(margin, q):
    return -1. if margin <= q / (q + 1.) else -(q / ((q + 1.) * margin)) ** (q + 1.)


def mean_derivative(K, y, alpha, offset, q):
    margins = y * (K @ alpha + offset)
    return y * np.array([slope(float(m), q) for m in margins]) / len(y)


def augmented_mm_step(K, z, alpha, lambd, step_scale, singular=False):
    n = len(alpha)
    ones = np.ones(n)
    shift = 2. * lambd * step_scale
    P = np.block([[np.array([[n]]), (ones @ K)[None, :]],
                  [(K @ ones)[:, None], K @ K + shift * K]])
    rhs = step_scale * np.r_[z.sum(), K @ (z + 2. * lambd * alpha)]
    delta = np.linalg.pinv(P, rcond=1e-12) @ rhs if singular else solve(P, rhs)
    return delta, P, rhs


class DenseConstrainedSystem:
    """Test-only full saddle-system solve; no Schur complement implementation."""
    def __init__(self, K, shift):
        n = len(K)
        self.matrix = np.block([[K + shift * np.eye(n), np.ones((n, 1))],
                                [np.ones((1, n)), np.zeros((1, 1))]])
        self.calls = []

    def solve_constrained(self, rhs, target_sum=0.):
        self.calls.append((np.array(rhs, copy=True), target_sum))
        result = solve(self.matrix, np.r_[rhs, target_sum], assume_a='sym')
        return result[:-1], float(result[-1])


def direct_next_model(K, z, alpha, offset, lambd, step_scale):
    """Independent direct next-state solve, with a zero coefficient-sum target."""
    n = len(alpha)
    shifted = K + (2. * lambd * step_scale) * np.eye(n)
    system = np.block([[shifted, np.ones((n, 1))],
                       [np.ones((1, n)), np.zeros((1, 1))]])
    result = solve(system, np.r_[K @ alpha - step_scale * z, 0.], assume_a='sym')
    return result[:-1], float(offset + result[-1])


class CoefficientReferenceMMTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(60612)
        self.features = rng.normal(size=(11, 3))
        self.singular_K = self.features @ self.features.T / 3.
        self.K = self.singular_K + .7 * np.eye(11)
        self.y = np.r_[np.ones(7), -np.ones(4)]
        self.alpha = rng.normal(size=11) / 8. + .04
        self.offset = -.37
        self.assertGreater(abs(self.alpha.sum()), .01)

    def test_augmented_mm_equations_match_general_q_and_nonzero_initial_state(self):
        for q in (.2, 1., 3.5):
            for lambd in (.005, .13, 2.):
                with self.subTest(q=q, lambd=lambd):
                    t = len(self.y) * q / (q + 1.) ** 2
                    z = mean_derivative(self.K, self.y, self.alpha, self.offset, q)
                    system = DenseConstrainedSystem(self.K, 2. * lambd * t)
                    alpha, offset, scores = reference_update(
                        self.K, self.alpha, self.offset, z, lambd, t, system)
                    expected, P, rhs = augmented_mm_step(self.K, z, self.alpha, lambd, t)
                    actual = np.r_[self.offset - offset, self.alpha - alpha]
                    assert_allclose(actual, expected, rtol=2e-10, atol=2e-11)
                    assert_allclose(P @ actual, rhs, rtol=2e-10, atol=2e-11)
                    assert_allclose(scores, self.K @ alpha, rtol=0, atol=0)
                    self.assertAlmostEqual(system.calls[0][1], self.alpha.sum() / t)
                    self.assertAlmostEqual(alpha.sum(), 0., delta=2e-13)

    def test_singular_and_zero_kernel_match_augmented_prediction_step(self):
        for q in (.2, 1., 3.5):
            for K in (self.singular_K, np.zeros_like(self.K)):
                with self.subTest(q=q, zero=not K.any()):
                    t, lambd = len(self.y) * q / (q + 1.) ** 2, .13
                    z = mean_derivative(K, self.y, self.alpha, self.offset, q)
                    system = DenseConstrainedSystem(K, 2. * lambd * t)
                    alpha, offset, scores = reference_update(
                        K, self.alpha, self.offset, z, lambd, t, system)
                    expected, P, rhs = augmented_mm_step(K, z, self.alpha, lambd, t, singular=True)
                    actual = np.r_[self.offset - offset, self.alpha - alpha]
                    assert_allclose(P @ actual, rhs, rtol=2e-10, atol=3e-11)
                    assert_allclose(actual[0] + K @ actual[1:],
                                    expected[0] + K @ expected[1:], rtol=2e-10, atol=3e-11)
                    assert_allclose(scores, K @ alpha, rtol=0, atol=0)

    def test_matched_trajectories_against_direct_next_model_at_1_2_10_updates(self):
        for q in (.2, 1., 3.5):
            for K in (self.K, self.singular_K, np.zeros_like(self.K)):
                for zero_init in (False, True):
                    with self.subTest(q=q, zero_kernel=not K.any(), zero_init=zero_init):
                        t, lambd = len(self.y) * q / (q + 1.) ** 2, .13
                        alpha = np.zeros_like(self.alpha) if zero_init else self.alpha.copy()
                        offset = 0. if zero_init else self.offset
                        direct_alpha, direct_offset = alpha.copy(), offset
                        system = DenseConstrainedSystem(K, 2. * lambd * t)
                        for iteration in range(1, 11):
                            z = mean_derivative(K, self.y, alpha, offset, q)
                            direct_z = mean_derivative(K, self.y, direct_alpha, direct_offset, q)
                            alpha, offset, scores = reference_update(K, alpha, offset, z, lambd, t, system)
                            direct_alpha, direct_offset = direct_next_model(
                                K, direct_z, direct_alpha, direct_offset, lambd, t)
                            if iteration in (1, 2, 10):
                                assert_allclose(alpha, direct_alpha, rtol=5e-11, atol=3e-12)
                                assert_allclose(offset, direct_offset, rtol=5e-11, atol=3e-12)
                                assert_allclose(scores + offset, K @ direct_alpha + direct_offset,
                                                rtol=5e-11, atol=3e-12)

    def test_nonzero_constraint_target_and_intercept_sign_are_required(self):
        K = np.zeros((3, 3))
        alpha = np.array([.3, -.1, .4])
        offset, lambd, t = .2, .5, .75
        z = np.array([-.2, .1, -.2])
        system = DenseConstrainedSystem(K, 2. * lambd * t)
        updated_alpha, updated_offset, scores = reference_update(K, alpha, offset, z, lambd, t, system)
        self.assertAlmostEqual(system.calls[0][1], .8)
        self.assertAlmostEqual(updated_offset, offset - t * z.mean())
        assert_allclose(updated_alpha, -(z - z.mean()) / (2. * lambd), atol=1e-15)
        assert_array_equal(scores, np.zeros(3))

    def test_inputs_are_not_mutated(self):
        K, alpha = self.K.copy(), self.alpha.copy()
        z = mean_derivative(K, self.y, alpha, self.offset, 1.)
        copies = [a.copy() for a in (K, alpha, z)]
        for a in (K, alpha, z):
            a.flags.writeable = False
        t = len(alpha) / 4.
        result = reference_update(K, alpha, self.offset, z, .13, t, DenseConstrainedSystem(K, .26 * t))
        for original, copy in zip((K, alpha, z), copies):
            assert_array_equal(original, copy)
        self.assertFalse(np.shares_memory(alpha, result[0]))

    def test_nonfinite_or_overflowing_direction_cannot_return_a_model(self):
        class InvalidDirection:
            def __init__(self, value):
                self.value = value

            def solve_constrained(self, rhs, target_sum=0.):
                return np.full_like(rhs, self.value), 0.

        for value in (np.nan, np.inf, np.finfo(float).max):
            with self.subTest(value=value):
                initial = self.alpha.copy()
                with self.assertRaises(FloatingPointError):
                    reference_update(self.K, initial, self.offset, np.zeros_like(initial),
                                     .13, 4., InvalidDirection(value))
                assert_array_equal(initial, self.alpha)


if __name__ == '__main__':
    unittest.main(verbosity=2)
