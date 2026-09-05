"""Independent small-matrix checks of the corrected kernel DWD MM solver.

Every matrix has <= 24 rows; no external datasets are required.
The references solve an augmented system, use its pseudoinverse, or differentiate
the scalar objective; they do not duplicate the implementation's Schur formula.
"""
import unittest

import numpy as np
from numpy.testing import assert_allclose

from dwd.gen_dwd import V_, V_grad_
from dwd.gen_kern_dwd import get_K_eig, solve_gen_kern_dwd


def objective(K, y, q, penalty, alpha, offset):
    margins = y * (K @ alpha + offset)
    return np.mean([V_(u, q=q) for u in margins]) + penalty * alpha @ K @ alpha


def augmented_step(K, y, q, penalty, alpha, offset, singular=False):
    n = len(y)
    factor = n * q / (q + 1) ** 2
    shift = 2 * penalty * factor
    ones = np.ones(n)
    z = y * np.array([V_grad_(u, q=q) for u in y * (K @ alpha + offset)]) / n
    rhs = factor * np.r_[z.sum(), K @ z + 2 * penalty * K @ alpha]
    P = np.block([[np.array([[n]]), (ones @ K)[None, :]],
                  [(K @ ones)[:, None], K @ K + shift * K]])
    step = np.linalg.pinv(P, rcond=1e-12) @ rhs if singular else np.linalg.solve(P, rhs)
    return step, P, rhs


class KernelMathExtendedTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(6095)
        self.n = 12
        features = self.rng.normal(size=(self.n, self.n))
        self.K = features @ features.T / self.n + 0.4 * np.eye(self.n)
        self.y = np.resize([-1, 1], self.n)
        self.alpha = self.rng.normal(size=self.n) / self.n
        self.offset = 0.23

    def one_step(self, K, q, penalty, alpha=None, offset=None, K_eig=None):
        alpha = self.alpha if alpha is None else alpha
        offset = self.offset if offset is None else offset
        out = solve_gen_kern_dwd(
            K, self.y, penalty, q=q, alpha_init=alpha, offset_init=offset,
            max_iter=1, obj_tol=0, K_eig=K_eig, solver_mode="schur")
        return np.r_[offset - out[1], alpha - out[0]], out

    def test_augmented_system_identity_across_q_and_regularization(self):
        for q in (0.2, 1.0, 3.5):
            for penalty in (1e-5, 0.03, 10.0):
                with self.subTest(q=q, penalty=penalty):
                    got, _ = self.one_step(self.K, q, penalty)
                    expected, P, rhs = augmented_step(
                        self.K, self.y, q, penalty, self.alpha, self.offset)
                    assert_allclose(got, expected, rtol=2e-8, atol=1e-9)
                    assert_allclose(P @ got, rhs, rtol=2e-8, atol=1e-10)

    def test_step_rhs_matches_finite_difference_objective_gradient(self):
        q, penalty = 1.7, 0.11
        got, _ = self.one_step(self.K, q, penalty)
        _, P, _ = augmented_step(self.K, self.y, q, penalty, self.alpha, self.offset)
        theta = np.r_[self.offset, self.alpha]
        eps = 1e-6
        finite_gradient = []
        for j in range(len(theta)):
            upper, lower = theta.copy(), theta.copy()
            upper[j] += eps
            lower[j] -= eps
            finite_gradient.append((
                objective(self.K, self.y, q, penalty, upper[1:], upper[0]) -
                objective(self.K, self.y, q, penalty, lower[1:], lower[0])) / (2 * eps))
        factor = self.n * q / (q + 1) ** 2
        assert_allclose(P @ got / factor, finite_gradient, rtol=1e-7, atol=2e-10)

    def test_eigenvector_sign_and_permutation_invariance(self):
        U, d = get_K_eig(self.K)
        signs = self.rng.choice([-1, 1], self.n)
        order = self.rng.permutation(self.n)
        base = self.one_step(self.K, 1.0, 0.03, K_eig=(U, d))[0]
        changed = self.one_step(self.K, 1.0, 0.03,
                                K_eig=((U * signs)[:, order], d[order]))[0]
        assert_allclose(changed, base, rtol=2e-13, atol=2e-13)

    def test_rotation_within_repeated_eigenvalue_space(self):
        U, _ = np.linalg.qr(self.rng.normal(size=(self.n, self.n)))
        d = np.r_[np.full(5, 0.4), np.linspace(0.8, 2.0, self.n - 5)]
        K = (U * d) @ U.T
        rotation, _ = np.linalg.qr(self.rng.normal(size=(5, 5)))
        rotated = U.copy()
        rotated[:, :5] = U[:, :5] @ rotation
        a = self.one_step(K, 2.0, 0.05, K_eig=(U, d))[0]
        b = self.one_step(K, 2.0, 0.05, K_eig=(rotated, d))[0]
        assert_allclose(a, b, rtol=3e-13, atol=3e-13)

    def test_singular_psd_step_matches_pseudoinverse_prediction_and_residual(self):
        X = self.rng.normal(size=(self.n, 3))
        K = X @ X.T
        got, _ = self.one_step(K, 1.0, 0.04)
        expected, P, rhs = augmented_step(K, self.y, 1.0, 0.04,
                                          self.alpha, self.offset, singular=True)
        assert_allclose(P @ got, rhs, rtol=1e-9, atol=1e-9)
        assert_allclose(got[0] + K @ got[1:], expected[0] + K @ expected[1:],
                        rtol=1e-9, atol=1e-9)
        # Different nullspace coefficients represent exactly the same function.
        null = np.linalg.svd(X.T, full_matrices=True)[2][-1]
        alternate = self.one_step(K, 1.0, 0.04,
                                  alpha=self.alpha + 7.0 * null)[1]
        original = self.one_step(K, 1.0, 0.04)[1]
        assert_allclose(K @ alternate[0] + alternate[1],
                        K @ original[0] + original[1], rtol=1e-10, atol=1e-10)

    def test_singular_kernel_objective_descent_and_zero_kernel(self):
        X = self.rng.normal(size=(self.n, 3))
        for K in (X @ X.T, np.zeros((self.n, self.n))):
            with self.subTest(zero=not K.any()):
                out = solve_gen_kern_dwd(K, self.y, 0.05, q=0.5,
                    alpha_init=self.alpha, offset_init=self.offset,
                    max_iter=35, obj_tol=0, solver_mode="schur")
                self.assertTrue(np.isfinite(out[2]).all())
                self.assertLessEqual(float(np.max(np.diff(out[2]))), 2e-12)

    def test_tiny_penalty_singular_kernel_is_rejected_instead_of_wrong_norm(self):
        rng = np.random.RandomState(32)
        X = rng.normal(size=(24, 3))
        K = X @ X.T
        y = np.resize([-1, 1], 24)
        initial = rng.normal(size=24) / 24
        # Without an explicit numerical support boundary, nullspace coefficients
        # grow to ~2e7 and the original output C is ~6.98, even though the
        # independently evaluable feature norm gives 4*||X.T@alpha||^2 ~1.71.
        # Reject this case rather than silently projecting or adding a ridge.
        with self.assertRaises((ValueError, FloatingPointError)):
            solve_gen_kern_dwd(K, y, 1e-9, q=1, alpha_init=initial,
                              max_iter=100, obj_tol=0, solver_mode="schur")

    def test_genuinely_indefinite_kernel_is_rejected(self):
        K = np.eye(self.n)
        K[0, 0] = -0.1
        with self.assertRaises(ValueError):
            self.one_step(K, 1.0, 0.1)

    def test_asymmetric_kernel_is_rejected(self):
        K = self.K.copy()
        K[0, 1] += 0.2
        with self.assertRaises(ValueError):
            self.one_step(K, 1.0, 0.1)

    def test_nonfinite_initial_state_is_rejected(self):
        for alpha, offset in ((np.full(self.n, np.nan), 0.0),
                              (self.alpha, np.inf)):
            with self.subTest(offset=offset):
                with self.assertRaises((ValueError, FloatingPointError)):
                    self.one_step(self.K, 1.0, 0.1, alpha=alpha, offset=offset)

    def test_vector_intercept_is_rejected(self):
        with self.assertRaises(ValueError):
            self.one_step(self.K, 1.0, 0.1, offset=np.zeros(self.n))

    def test_nonfinite_precomputed_eigenvectors_are_rejected(self):
        U, d = get_K_eig(self.K)
        U[0, 0] = np.nan
        with self.assertRaises((ValueError, FloatingPointError)):
            self.one_step(self.K, 1.0, 0.1, K_eig=(U, d))


if __name__ == "__main__":
    unittest.main(verbosity=2)
