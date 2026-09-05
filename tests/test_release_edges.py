"""Small release regressions for corrected numerical/API boundaries."""
import unittest


import numpy as np
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix

from dwd.gen_dwd import GenDWD, solve_gen_dwd
from dwd.gen_kern_dwd import KernGDWD, solve_gen_kern_dwd


class FinalMathChecks(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(80)
        self.X32 = rng.normal(size=(20, 3)).astype(np.float32)
        self.X64 = self.X32.astype(np.float64)
        self.y = np.resize([-1, 1], 20)

    def kernel(self, **kwargs):
        parameters = dict(kernel="linear", solver_mode="schur", lambd=.1,
                          max_iter=8, obj_tol=0, random_state=42)
        parameters.update(kwargs)
        return KernGDWD(**parameters)

    def linear(self, **kwargs):
        parameters = dict(solver_mode="schur", lambd=.1, max_iter=8,
                          obj_tol=0, random_state=42)
        parameters.update(kwargs)
        return GenDWD(**parameters)

    def test_explicit_arraylike_kernel_normalized_before_diagnostics(self):
        K = self.X64 @ self.X64.T + np.eye(len(self.y))
        array_model = self.kernel().fit(self.X64, self.y, K=K)
        list_model = self.kernel().fit(self.X64, self.y, K=K.tolist())
        assert_allclose(list_model.dual_coef_, array_model.dual_coef_, rtol=0, atol=0)
        self.assertTrue(np.isfinite(list_model.gradient_inf_norm_))

    def test_corrected_float32_features_promoted_before_kernel(self):
        low = self.kernel().fit(self.X32, self.y)
        high = self.kernel().fit(self.X64, self.y)
        assert_allclose(low.decision_function(self.X64), high.decision_function(self.X64),
                        rtol=1e-12, atol=1e-12)

    def test_corrected_float32_features_promoted_before_linear_gram(self):
        low = self.linear().fit(self.X32, self.y)
        high = self.linear().fit(self.X64, self.y)
        assert_allclose(low.coef_, high.coef_, rtol=1e-12, atol=1e-12)
        assert_allclose(low.intercept_, high.intercept_, rtol=1e-12, atol=1e-12)

    def test_corrected_low_level_linear_float32_promotion(self):
        kwargs = dict(lambd=.1, random_state=9, max_iter=8, obj_tol=0,
                      solver_mode="schur")
        low = solve_gen_dwd(self.X32, self.y, **kwargs)
        high = solve_gen_dwd(self.X64, self.y, **kwargs)
        assert_allclose(low[0], high[0], rtol=1e-12, atol=1e-12)

    def test_already_rounded_indefinite_float32_kernel_is_rejected(self):
        K = self.X32 @ self.X32.T
        self.assertLess(np.linalg.eigvalsh(K)[0], -1e-8)
        # Casting after Gram formation cannot undo prior float32 roundoff.
        with self.assertRaises(ValueError):
            self.kernel().fit(self.X32, self.y, K=K)

    def test_kernel_hidden_cache_handles_dense_to_csr(self):
        cached = self.kernel(kernel="rbf").cv_init(self.X64)
        cached.fit(csr_matrix(self.X64), self.y)
        fresh = self.kernel(kernel="rbf").fit(csr_matrix(self.X64), self.y)
        assert_allclose(cached.decision_function(self.X64),
                        fresh.decision_function(self.X64), rtol=1e-11, atol=1e-11)

    def test_kernel_mode_change_does_not_reuse_float32_legacy_gram(self):
        cached = self.kernel(kernel="rbf", solver_mode="legacy").cv_init(self.X32)
        cached.set_params(solver_mode="schur").fit(self.X32, self.y)
        fresh = self.kernel(kernel="rbf").fit(self.X64, self.y)
        assert_allclose(cached.decision_function(self.X64),
                        fresh.decision_function(self.X64), rtol=1e-11, atol=1e-11)

    def test_linear_mode_change_does_not_reuse_float32_legacy_gram(self):
        cached = self.linear(solver_mode="legacy").cv_init(self.X32)
        cached.set_params(solver_mode="schur").fit(self.X32, self.y)
        fresh = self.linear().fit(self.X64, self.y)
        assert_allclose(cached.coef_, fresh.coef_, rtol=1e-12, atol=1e-12)
        assert_allclose(cached.intercept_, fresh.intercept_, rtol=1e-12, atol=1e-12)

    def test_overflowing_finite_initial_coefficients_do_not_return_fitted_model(self):
        for iterations in (0, 1):
            for kind in ("kernel", "linear"):
                with self.subTest(kind=kind, iterations=iterations), np.errstate(all="ignore"):
                    with self.assertRaises((ValueError, FloatingPointError)):
                        if kind == "kernel":
                            self.kernel(max_iter=iterations).fit(self.X64, self.y,
                                alpha_init=np.full(len(self.y), 1e308))
                        else:
                            self.linear(max_iter=iterations).fit(self.X64, self.y,
                                beta_init=np.full(self.X64.shape[1], 1e308))

    def test_zero_kernel_has_finite_zero_norm_conversion(self):
        K = np.zeros((len(self.y), len(self.y)))
        for q in (.5, 1., 3.5):
            with self.subTest(q=q):
                result = solve_gen_kern_dwd(K, self.y, .1, q=q,
                    random_state=42, max_iter=5, solver_mode="schur")
                self.assertEqual(result[3], 0.)
                self.assertTrue(np.isfinite(result[2]).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
