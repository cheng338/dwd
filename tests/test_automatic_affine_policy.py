"""Automatic scoring, historical policies and fitted-state ownership."""
import pickle
import tempfile
import unittest
from fractions import Fraction
from inspect import signature
from unittest.mock import patch
import numpy as np
from scipy.linalg import eigh
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV
from threadpoolctl import threadpool_limits
from dwd.gen_kern_dwd import KernGDWD, KernGDWDCV
from dwd.cv import run_cv

class AutomaticAffinePolicyTests(unittest.TestCase):

    def setUp(self):
        self.limits = threadpool_limits(1)
        self.addCleanup(self.limits.restore_original_limits)
        self.X = np.array([[0.0], [1.0]])
        self.y = np.array([-3, 7])
        self.query = np.array([[np.nextafter(0.5, np.inf)]])
        self.Xn = np.column_stack((np.linspace(-2, 2, 24), np.sin(np.linspace(-2, 2, 24))))
        self.yn = np.tile([-1, 1], 12)

    def model(self, **kw):
        options = dict(kernel='linear', lambd=0.1, max_iter=5, stopping='fixed', initialization='zero', random_state=7)
        options.update(kw)
        return KernGDWD(**options)

    def test_constructor_clone_parameter_defaults(self):
        for cls in (KernGDWD, KernGDWDCV):
            self.assertEqual(signature(cls).parameters['affine_computation'].default, 'auto')
            self.assertEqual(cls.affine_computation, 'standard')
            obj = cls()
            self.assertEqual(obj.get_params()['affine_computation'], 'auto')
            copied = clone(obj).get_params()
            self.assertEqual(set(copied), set(obj.get_params()))
            for key, value in obj.get_params().items():
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(copied[key], value)
                else:
                    self.assertEqual(copied[key], value)
            self.assertFalse(hasattr(obj, 'affine_computation_'))
            obj.set_params(affine_computation='standard')
            self.assertEqual(clone(obj).affine_computation, 'standard')

    def test_prefit_does_not_create_resolved_policy(self):
        for obj in (self.model(),):
            with self.assertRaises(NotFittedError):
                obj.predict(self.X)
            self.assertFalse(hasattr(obj, 'affine_computation_'))

    def test_new_default_recovers_natural_boundary(self):
        # The affine-policy fixture depends on five ordinary MM updates.
        auto = self.model(acceleration=None).fit(self.X, self.y)
        standard = self.model(affine_computation='standard', acceleration=None).fit(self.X, self.y)
        joint = self.model(affine_computation='joint', acceleration=None).fit(self.X, self.y)
        self.assertEqual(auto.affine_computation, 'auto')
        self.assertEqual(auto.affine_computation_, 'joint')
        self.assertEqual(auto.predict(self.query)[0], 7)
        self.assertEqual(standard.predict(self.query)[0], -3)
        for name in ('dual_coef_', 'intercept_', 'objective_history_'):
            np.testing.assert_array_equal(getattr(auto, name), getattr(standard, name))
            np.testing.assert_array_equal(getattr(auto, name), getattr(joint, name))
        np.testing.assert_array_equal(auto.decision_function(self.query), joint.decision_function(self.query))

    def test_fitted_policy_ownership_mutation_and_refit(self):
        obj = self.model(acceleration=None).fit(self.X, self.y)
        obj.set_params(affine_computation='standard')
        self.assertEqual(obj.predict(self.query)[0], 7)
        restored = pickle.loads(pickle.dumps(obj))
        self.assertEqual(restored.affine_computation, 'standard')
        self.assertEqual(restored.affine_computation_, 'joint')
        self.assertEqual(restored.predict(self.query)[0], 7)
        restored.fit(self.X, self.y)
        self.assertEqual(restored.affine_computation_, 'standard')
        self.assertEqual(restored.predict(self.query)[0], -3)

    def test_corrected_backends_and_supplied_eigensystems(self):
        K = self.Xn @ self.Xn.T
        vals, vecs = eigh(K)
        for opts, fitopts in [({}, {}), ({'backend': 'cholesky'}, {}), ({'backend': 'spectral'}, {}), ({'backend': 'lbfgs'}, {}), ({'implementation': 'reference'}, {}), ({}, {'K': K, 'K_eig': (vecs, vals)})]:
            with self.subTest(opts=opts, fitopts=bool(fitopts)):
                auto = self.model(max_iter=2, **opts).fit(self.Xn, self.yn, **fitopts)
                joint = self.model(max_iter=2, affine_computation='joint', **opts).fit(self.Xn, self.yn, **fitopts)
                self.assertEqual(auto.affine_computation_, 'joint')
                for name in ('dual_coef_', 'intercept_', 'objective_history_'):
                    np.testing.assert_array_equal(getattr(auto, name), getattr(joint, name))
                np.testing.assert_array_equal(auto.decision_function(self.Xn.copy()), joint.decision_function(self.Xn.copy()))

    def test_legacy_resolves_standard_and_preserves_explicit_rejection(self):
        auto = self.model(solver_mode='legacy', kernel='rbf', kernel_kws={'gamma': 1.0}).fit(self.X, self.y)
        standard = self.model(solver_mode='legacy', kernel='rbf', kernel_kws={'gamma': 1.0}, affine_computation='standard').fit(self.X, self.y)
        self.assertEqual(auto.affine_computation_, 'standard')
        np.testing.assert_array_equal(auto.decision_function(self.query), standard.decision_function(self.query))
        with self.assertRaises(ValueError):
            self.model(solver_mode='legacy', affine_computation='joint').fit(self.X, self.y)

    def test_auto_validation_and_zero_iteration_use_concrete_policy(self):
        for opts in ({}, {'backend': 'spectral'}, {'backend': 'lbfgs'}, {'implementation': 'reference'}):
            with self.subTest(opts=opts):
                auto = self.model(stopping='validation', **opts).fit(self.X, self.y, validation_data=(self.query, np.array([7])))
                joint = self.model(stopping='validation', affine_computation='joint', **opts).fit(self.X, self.y, validation_data=(self.query, np.array([7])))
                self.assertEqual(auto.validation_history_, joint.validation_history_)
                self.assertEqual(auto.returned_iteration_, joint.returned_iteration_)
        seed = self.model(acceleration=None).fit(self.X, self.y)
        zero = self.model(max_iter=0, stopping='validation').fit(self.X, self.y, alpha_init=seed.dual_coef_[0], offset_init=float(seed.intercept_[0]), validation_data=(self.query, np.array([7])))
        self.assertEqual(zero.validation_history_[0]['score'], 1.0)

    def test_cv_cache_identity_and_actual_use(self):
        obj = self.model().cv_init(self.X)
        self.assertEqual(obj._cv_affine_computation, 'auto')
        self.assertTrue(obj._cv_cache_matches(self.X))
        with patch.object(obj, '_compute_training_kernel', side_effect=AssertionError('cache expected')):
            obj.fit(self.X, self.y)
        self.assertEqual(obj.affine_computation_, 'joint')
        obj.set_params(affine_computation='joint')
        self.assertFalse(obj._cv_cache_matches(self.X))
        obj.set_params(affine_computation='standard')
        self.assertFalse(obj._cv_cache_matches(self.X))
        obj.cv_init(self.X).fit(self.X, self.y)
        self.assertEqual(obj.affine_computation_, 'standard')

    def test_failed_fit_removes_resolved_policy(self):
        for obj in (self.model(),):
            obj.fit(self.X, self.y)
            obj.set_params(affine_computation='invalid')
            with self.assertRaises(ValueError):
                obj.fit(self.X, self.y)
            self.assertFalse(hasattr(obj, 'affine_computation_'))
            with self.assertRaises(NotFittedError):
                obj.predict(self.X)

    def test_cv_estimator_default_and_legacy(self):
        for mode, expected in [('schur', 'joint'), ('legacy', 'standard')]:
            cv = KernGDWDCV(lambd_vals=[0.1], q_vals=[1.0], cv=2, max_iter=2, stopping='fixed', random_state=7, solver_mode=mode).fit(self.Xn, self.yn)
            self.assertEqual(cv.affine_computation, 'auto')
            self.assertEqual(cv.affine_computation_, expected)
            self.assertEqual(cv.best_estimator_.affine_computation_, expected)
            scores = cv.decision_function(self.Xn.copy())
            cv.set_params(affine_computation='standard')
            np.testing.assert_array_equal(cv.decision_function(self.Xn.copy()), scores)
            np.testing.assert_array_equal(pickle.loads(pickle.dumps(cv)).decision_function(self.Xn.copy()), scores)

    def test_run_cv_and_sklearn_grid_mixed_policies(self):
        grid = {'affine_computation': ['auto', 'standard', 'joint'], 'lambd': [0.1]}
        results = run_cv(self.model(max_iter=2), self.Xn, self.yn, grid, cv=2, refit_best=False)[3]
        self.assertEqual(len(results['params']), 3)
        self.assertEqual(results['mean_test_score'][0], results['mean_test_score'][2])
        search = GridSearchCV(self.model(max_iter=2), grid, cv=2, n_jobs=1).fit(self.Xn, self.yn)
        expected = 'joint' if search.best_params_['affine_computation'] == 'auto' else search.best_params_['affine_computation']
        self.assertEqual(search.best_estimator_.affine_computation_, expected)

    def test_guarded_affine_exact_fallback(self):
        import dwd._affine_scores as affine
        obj = self.model().fit(self.X, self.y)
        expected = obj.decision_function(self.query)
        with patch.object(affine, '_products', side_effect=FloatingPointError('forced expansion refusal')):
            np.testing.assert_array_equal(obj.decision_function(self.query), expected)

    def test_auto_spectral_restart_keeps_resolved_policy(self):
        from dwd import _kernel_solver
        from dwd._kernel_recovery import MMRecoveryExhausted
        original = _kernel_solver.solve_kernel
        calls = []

        def solve(*args, **kwargs):
            calls.append((kwargs['backend'], kwargs['affine_computation']))
            if len(calls) == 1:
                raise MMRecoveryExhausted('controlled exhausted attempt', dict(completed_iterations=0, failed_iteration=1, setup_seconds=0.0, optimization_seconds=0.0))
            return original(*args, **kwargs)
        with patch.object(_kernel_solver, 'solve_kernel', new=solve):
            model = self.model(kernel='rbf', kernel_kws={'gamma': 0.2}, max_iter=2).fit(self.Xn, self.yn)
        self.assertEqual(calls, [('auto', 'joint'), ('spectral', 'joint')])
        self.assertEqual(model.affine_computation_, 'joint')
        self.assertEqual(model.backend_, 'spectral')
        self.assertEqual(model.diagnostics_['spectral_restart']['restarts'], 1)

    def test_auto_fit_with_optional_native_helper_disabled(self):
        from dwd import _compiled_residual, _native_residual
        with patch.object(_compiled_residual, '_ACCEL', None), patch.object(_native_residual, '_native_supported', return_value=False):
            model = self.model(acceleration=None).fit(self.X, self.y)
            self.assertEqual(model.affine_computation_, 'joint')
            self.assertEqual(model.predict(self.query)[0], 7)

    def test_restart_default_affine_scores_match_exact_returned_state(self):
        models = {}
        for acceleration in ('auto', 'restart'):
            for affine in ('auto', 'standard', 'joint'):
                model = self.model(acceleration=acceleration, affine_computation=affine).fit(self.X, self.y)
                models[acceleration, affine] = model
                self.assertEqual(model.acceleration_, 'restart')
                if affine != 'standard':
                    row = model._compute_kernel(self.query).T[0]
                    exact = sum((Fraction(float(k)) * Fraction(float(a))
                                 for k, a in zip(row, model.dual_coef_[0])),
                                Fraction(float(model.intercept_[0])))
                    self.assertEqual(model.decision_function(self.query)[0], float(exact))
                    self.assertEqual(model.predict(self.query)[0], model.classes_[int(exact > 0)])
                for name in ('dual_coef_', 'intercept_', 'objective_history_'):
                    np.testing.assert_array_equal(getattr(model, name),
                                                  getattr(models[acceleration, 'auto'], name))
        for affine in ('auto', 'standard', 'joint'):
            a, b = models['auto', affine], models['restart', affine]
            for name in ('dual_coef_', 'intercept_', 'objective_history_'):
                np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
            np.testing.assert_array_equal(a.decision_function(self.query), b.decision_function(self.query))
