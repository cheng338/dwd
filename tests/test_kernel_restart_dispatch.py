"""Mock-only recovery/estimator dispatch tests. No numerical solver is executed."""
from pathlib import Path
import unittest
import pickle
import weakref
from unittest.mock import Mock, patch

import numpy as np
import dwd
import dwd._kernel_recovery as recovery_module
from dwd.gen_kern_dwd import KernGDWD
from dwd._kernel_recovery import (ConstrainedSolveFailure, MMRecoveryExhausted,
                                  solve_with_spectral_restart)


def failure(message="bounded constrained/MM recovery exhausted", **updates):
    details = dict(failed_backend="cholesky", failed_iteration=3,
                   completed_iterations=2, setup_seconds=1., optimization_seconds=2.,
                   constrained_failure="all constrained representations failed",
                   linear_mode="anchor", linear_solves=3, linear_recoveries=2,
                   mm_function_recovery_attempted=True,
                   mm_function_recovery_accepted_actions=0,
                   mm_function_certificate_status="entry_budget_exhausted")
    details.update(updates)
    return MMRecoveryExhausted(message, details)


def result(n=4):
    return dict(alpha=np.zeros(n), offset=.125,
                objective_history=np.array([1., .7, .4]), n_iter=2,
                returned_iteration=2, termination_reason="max_iter", converged=False,
                backend="spectral", prediction_precision="adaptive",
                final_objective=.4, C=1., gradient_inf_norm=.2,
                rkhs_gradient_norm=.2, dual_gap=.01,
                validation_history=[{"iteration": 2, "score": .5}],
                diagnostics=dict(requested_backend="spectral", initial_backend="spectral",
                    attempted_backends=["spectral"], effective_backend="spectral",
                    auto_fallback=False, fallback_reason=None,
                    setup_seconds=2., optimization_seconds=3., total_seconds=6.,
                    positive_eigenvalues_discarded=0, kernel_approximation_used=False))


class TestRetryDispatch(unittest.TestCase):
    def setUp(self):
        self.K, self.y, self.alpha, self.validation = object(), object(), object(), object()
        self.options = dict(backend="auto", implementation="optimized", acceleration=None,
                            callback=None, K_eig=None, alpha_init=self.alpha,
                            offset_init=-2., q=2., max_iter=7, obj_tol=2e-7, tol=4e-6,
                            stopping="validation", patience=4, min_delta=.02,
                            check_interval=2, validation=self.validation, psd_known=True)

    def call(self, solve, eligible=True, **updates):
        return solve_with_spectral_restart(solve, self.K, self.y, 1e-9,
                eligible=eligible, **dict(self.options, **updates))

    def test_typed_failure_survives_worker_serialization(self):
        original = failure()
        restored = pickle.loads(pickle.dumps(original))
        self.assertIsInstance(restored, MMRecoveryExhausted)
        self.assertEqual(str(restored), str(original))
        self.assertEqual(restored.details, original.details)
        self.assertIsNot(restored.details, original.details)

    def test_healthy_result_is_returned_unchanged(self):
        expected = result()
        solve = Mock(return_value=expected)
        self.assertIs(self.call(solve), expected)
        self.assertEqual(solve.call_count, 1)
        self.assertNotIn("spectral_restart", expected["diagnostics"])

    def test_retry_preserves_all_inputs_and_stopping_options(self):
        expected = result()
        solve = Mock(side_effect=[failure(), expected])
        actual = self.call(solve)
        self.assertIs(actual, expected)
        first, second = solve.call_args_list
        self.assertEqual(first.args, second.args)
        self.assertIs(first.args[0], self.K)
        self.assertIs(first.args[1], self.y)
        self.assertEqual(second.kwargs, dict(first.kwargs, backend="spectral"))
        self.assertIs(second.kwargs["alpha_init"], self.alpha)
        self.assertIs(second.kwargs["validation"], self.validation)
        self.assertEqual(actual["n_iter"], 2)
        self.assertEqual(actual["validation_history"], [{"iteration": 2, "score": .5}])
        np.testing.assert_array_equal(actual["objective_history"], [1., .7, .4])

    def test_retry_counters_and_total_timing_include_discarded_work(self):
        solve = Mock(side_effect=[failure(), result()])
        with patch("dwd._kernel_recovery.perf_counter", side_effect=[10., 14., 18.]):
            actual = self.call(solve)
        d = actual["diagnostics"]
        self.assertEqual(d["attempted_backends"], ["cholesky", "spectral"])
        self.assertEqual(d["requested_backend"], "auto")
        self.assertEqual(d["effective_backend"], "spectral")
        self.assertTrue(d["auto_fallback"])
        r = d["spectral_restart"]
        self.assertEqual(r["restarts"], 1)
        self.assertEqual(r["discarded_completed_updates"], 2)
        self.assertEqual(r["failed_attempted_iteration"], 3)
        self.assertEqual(r["total_completed_updates"], 4)
        self.assertEqual(r["total_attempted_updates"], 5)
        self.assertEqual(r["max_iter_per_attempt"], 7)
        self.assertEqual(r["failed_attempt"]["elapsed_seconds"], 4.)
        self.assertEqual(d["setup_seconds"], 3.)
        self.assertEqual(d["optimization_seconds"], 5.)
        self.assertEqual(d["total_seconds"], 8.)
        self.assertFalse(actual["converged"])

    def test_only_specific_exhaustion_is_retried(self):
        for error in (FloatingPointError("readout"),
                      ConstrainedSolveFailure("MM recovery was not yet exhausted"),
                      ValueError("invalid eigenpairs"), MemoryError("allocation"),
                      RuntimeError("callback error"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                solve = Mock(side_effect=error)
                with self.assertRaises(type(error)) as caught:
                    self.call(solve)
                self.assertIs(caught.exception, error)
                self.assertEqual(solve.call_count, 1)

    def test_exhaustion_after_spectral_retry_is_not_retried_again(self):
        second = failure("second failure")
        solve = Mock(side_effect=[failure(), second])
        try:
            self.call(solve)
        except MMRecoveryExhausted as caught:
            self.assertIs(caught, second)
            self.assertIsNotNone(caught.__traceback__)
        else:
            self.fail("A second failure must propagate")
        self.assertEqual(solve.call_count, 2)

    def test_spectral_acceptance_error_propagates(self):
        second = FloatingPointError("returned coefficients lost objective accuracy")
        solve = Mock(side_effect=[failure(), second])
        with self.assertRaises(FloatingPointError) as caught:
            self.call(solve)
        self.assertIs(caught.exception, second)
        self.assertEqual(solve.call_count, 2)

    def test_explicit_backends_and_other_modes_do_not_restart(self):
        variants = [dict(backend=b) for b in ("cholesky", "spectral", "lbfgs")]
        variants += [dict(implementation="reference"), dict(acceleration="restart"),
                     dict(callback=lambda _: None), dict(K_eig=object())]
        for options in variants:
            with self.subTest(options=options):
                error = failure()
                solve = Mock(side_effect=error)
                try:
                    self.call(solve, **options)
                except MMRecoveryExhausted as caught:
                    self.assertIs(caught, error)
                    self.assertIsNotNone(caught.__traceback__)
                else:
                    self.fail("An ineligible failure must propagate")
                self.assertEqual(solve.call_count, 1)

    def test_untrusted_kernel_does_not_restart(self):
        solve = Mock(side_effect=failure())
        with self.assertRaises(MMRecoveryExhausted):
            self.call(solve, eligible=False)
        self.assertEqual(solve.call_count, 1)

    def test_failed_chained_frames_are_released_before_retry(self):
        references, held_errors = [], []
        class Payload:
            pass
        def inner():
            payload = Payload()
            references.append(weakref.ref(payload))
            raise ConstrainedSolveFailure("inner constrained failure")
        def solve(*args, **kwargs):
            if kwargs["backend"] == "auto":
                payload = Payload()
                references.append(weakref.ref(payload))
                try:
                    inner()
                except ConstrainedSolveFailure as first:
                    error = failure()
                    held_errors.extend([first, error])
                    raise error from first
            self.assertTrue(references)
            self.assertTrue(all(ref() is None for ref in references))
            for error in held_errors:
                self.assertIsNone(error.__traceback__)
                self.assertIsNone(error.__cause__)
                self.assertIsNone(error.__context__)
            return result()
        self.call(solve)


class TestMockedPublicFit(unittest.TestCase):
    """All kernels are canned arrays; solve_kernel is replaced in every test."""
    def setUp(self):
        self.X = np.array([[0., 1.], [1., 2.], [2., 3.], [3., 4.]])
        self.y = np.array(["a", "b", "a", "b"])
        self.K = np.eye(4)
        self.kernel_patch = patch("dwd.kernel_utils.pairwise_kernels", return_value=self.K)
        self.kernel = self.kernel_patch.start()
        self.addCleanup(self.kernel_patch.stop)
        self.solve_patch = patch("dwd._kernel_solver.solve_kernel")
        self.solve = self.solve_patch.start()
        self.addCleanup(self.solve_patch.stop)

    def model(self, cls=KernGDWD, **options):
        return cls(kernel="rbf", kernel_kws={"gamma": .5},
                   initialization="zero", **options)

    def test_recovery_module_is_coinstalled(self):
        self.assertEqual(Path(dwd.__file__).resolve().parent,
                         Path(recovery_module.__file__).resolve().parent)

    def test_internal_rbf_retries_and_keeps_parameters_and_fitted_history(self):
        model = self.model(max_iter=7)
        self.solve.side_effect = [failure(), result()]
        model.fit(self.X, self.y)
        self.assertEqual(self.solve.call_count, 2)
        self.assertEqual(self.kernel.call_count, 1)
        self.assertEqual(model.backend, "auto")
        self.assertEqual(model.backend_, "spectral")
        self.assertEqual(model.n_iter_, 2)
        self.assertFalse(model.converged_)
        np.testing.assert_array_equal(model.classes_, ["a", "b"])
        np.testing.assert_array_equal(model.objective_history_, [1., .7, .4])
        self.assertEqual(model.diagnostics_["spectral_restart"]["max_iter_per_attempt"], 7)
        first, second = self.solve.call_args_list
        self.assertIs(first.args[0], second.args[0])

    def test_realized_random_initialization_is_drawn_once(self):
        model = KernGDWD(kernel="rbf", initialization="random", random_state=42)
        rng = Mock()
        initial = np.array([1., 2., 3., 4.])
        rng.normal.return_value = initial.copy()
        self.solve.side_effect = [failure(), result()]
        with patch("dwd.gen_kern_dwd.check_random_state", return_value=rng) as get_rng:
            model.fit(self.X, self.y, offset_init=.75)
        get_rng.assert_called_once_with(42)
        rng.normal.assert_called_once_with(size=4)
        first, second = self.solve.call_args_list
        self.assertIs(first.kwargs["alpha_init"], second.kwargs["alpha_init"])
        np.testing.assert_array_equal(first.kwargs["alpha_init"], initial/np.linalg.norm(initial))
        self.assertEqual(second.kwargs["offset_init"], .75)

    def test_explicit_initialization_and_validation_are_reused(self):
        model = self.model(stopping="validation", patience=5, min_delta=.01, check_interval=3)
        initial = np.array([.1, -.1, .2, -.2])
        self.solve.side_effect = [failure(), result()]
        model.fit(self.X, self.y, alpha_init=initial, offset_init=-.25,
                  validation_data=(self.X.copy(), self.y.copy()))
        first, second = self.solve.call_args_list
        self.assertIs(first.kwargs["alpha_init"], initial)
        self.assertIs(second.kwargs["alpha_init"], initial)
        self.assertIs(first.kwargs["validation"], second.kwargs["validation"])
        self.assertEqual(second.kwargs["patience"], 5)
        self.assertEqual(second.kwargs["check_interval"], 3)
        self.assertEqual(second.kwargs["min_delta"], .01)
        np.testing.assert_array_equal(initial, [.1, -.1, .2, -.2])

    def test_explicit_K_or_eigenpairs_never_restart(self):
        for fit_kwargs in (dict(K=self.K), dict(K_eig=(np.ones(4), np.eye(4)))):
            with self.subTest(fit_kwargs=list(fit_kwargs)):
                self.solve.reset_mock()
                self.solve.side_effect = failure()
                with self.assertRaises(MMRecoveryExhausted):
                    self.model().fit(self.X, self.y, **fit_kwargs)
                self.assertEqual(self.solve.call_count, 1)

    def test_named_non_rbf_and_callable_kernels_never_restart(self):
        for kernel in ("linear", "poly", "precomputed",
                       lambda train, query: np.eye(len(train), len(query))):
            with self.subTest(kernel=kernel):
                self.solve.reset_mock()
                self.solve.side_effect = failure()
                model = KernGDWD(kernel=kernel, initialization="zero")
                with self.assertRaises(MMRecoveryExhausted):
                    model.fit(self.K if kernel == "precomputed" else self.X, self.y)
                self.assertEqual(self.solve.call_count, 1)

    def test_construction_or_psd_override_never_authorizes_restart(self):
        class CustomKernel(KernGDWD):
            def _compute_kernel(self, X):
                return super()._compute_kernel(X)
        class CustomTraining(KernGDWD):
            def _compute_training_kernel(self, X):
                return super()._compute_training_kernel(X)
        class CustomKnown(KernGDWD):
            def _known_psd_kernel(self):
                return True
        for cls in (CustomKernel, CustomTraining, CustomKnown):
            with self.subTest(cls=cls.__name__):
                self.solve.reset_mock()
                self.solve.side_effect = failure()
                with self.assertRaises(MMRecoveryExhausted):
                    self.model(cls).fit(self.X, self.y)
                self.assertEqual(self.solve.call_count, 1)

    def test_cache_from_builtin_construction_is_preserved_and_reused(self):
        model = self.model()
        model.cv_init(self.X)
        cached = model._cv_K
        self.kernel.reset_mock()
        self.solve.side_effect = [failure(), result()]
        model.fit(self.X, self.y)
        self.assertEqual(self.solve.call_count, 2)
        self.assertEqual(self.kernel.call_count, 0)
        self.assertIs(model._cv_K, cached)
        self.assertIs(self.solve.call_args_list[0].args[0], cached)

    def test_overridden_cache_provenance_never_authorizes_restart(self):
        class CustomCacheInit(KernGDWD):
            def cv_init(self, X):
                return super().cv_init(X)
        class CustomCacheMatch(KernGDWD):
            def _cv_cache_matches(self, X):
                return super()._cv_cache_matches(X)
        for cls in (CustomCacheInit, CustomCacheMatch):
            with self.subTest(cls=cls.__name__):
                model = self.model(cls)
                model.cv_init(self.X)
                self.solve.reset_mock()
                self.solve.side_effect = failure()
                with self.assertRaises(MMRecoveryExhausted):
                    model.fit(self.X, self.y)
                self.assertEqual(self.solve.call_count, 1)

    def test_temporary_override_when_cache_was_built_is_not_trusted(self):
        model = self.model()
        with patch.object(model, "_compute_kernel", return_value=self.K):
            model.cv_init(self.X)
        self.assertFalse(model._cv_trusted_rbf_construction)
        self.solve.side_effect = failure()
        with self.assertRaises(MMRecoveryExhausted):
            model.fit(self.X, self.y)
        self.assertEqual(self.solve.call_count, 1)

    def test_older_cache_without_provenance_is_not_trusted(self):
        model = self.model()
        model.cv_init(self.X)
        del model._cv_trusted_rbf_construction
        self.solve.side_effect = failure()
        with self.assertRaises(MMRecoveryExhausted):
            model.fit(self.X, self.y)
        self.assertEqual(self.solve.call_count, 1)

    def test_failed_retry_clears_old_and_partial_fit_but_keeps_checked_cache(self):
        model = self.model()
        model.cv_init(self.X)
        cached = model._cv_K
        self.solve.return_value = result()
        model.fit(self.X, self.y)
        self.assertTrue(hasattr(model, "dual_coef_"))
        self.solve.side_effect = [failure(), FloatingPointError("spectral check rejected")]
        with self.assertRaisesRegex(FloatingPointError, "spectral check rejected"):
            model.fit(self.X, self.y)
        for key in vars(model):
            self.assertFalse(key.endswith("_") and not key.startswith("_"), key)
        self.assertFalse(hasattr(model, "_Xfit"))
        self.assertIs(model._cv_K, cached)
        self.assertEqual(model.backend, "auto")
        self.solve.side_effect = None
        self.solve.return_value = result()
        model.fit(self.X, self.y)
        self.assertTrue(hasattr(model, "dual_coef_"))

    def test_callback_mode_propagates_original_exception_without_replay(self):
        callback = Mock()
        model = self.model(callback=callback)
        error = failure()
        self.solve.side_effect = error
        with self.assertRaises(MMRecoveryExhausted) as caught:
            model.fit(self.X, self.y)
        self.assertIs(caught.exception, error)
        self.assertEqual(self.solve.call_count, 1)
        callback.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
