"""Tiny API-transition controls; numerical kernels and gates are unchanged."""
import inspect
import pickle
import unittest
import warnings
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal
from scipy.linalg import eigh
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from dwd._acceleration_policy import resolve_acceleration
from dwd._kernel_solver import solve_kernel
from dwd._kernel_recovery import MMRecoveryExhausted
from dwd.gen_kern_dwd import KernGDWD, KernGDWDCV, solve_gen_kern_dwd

X = np.random.RandomState(21620).normal(size=(16, 3))
y = np.tile([-3, 7], 8)
Q = X[:5] + .03
K = X @ X.T + np.eye(len(X))
signed = np.where(y == 7, 1., -1.)
COMMON = dict(kernel='rbf', kernel_kws={'gamma': .3}, lambd=.1,
              stopping='fixed', max_iter=4, random_state=44)


def same_science(a, b):
    for name in ('dual_coef_', 'intercept_', 'objective_history_', 'n_iter_',
                 'returned_iteration_', 'termination_reason_', 'converged_',
                 'final_objective_', 'rkhs_gradient_norm_', 'backend_',
                 'prediction_precision_'):
        assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
    assert_array_equal(a.decision_function(Q), b.decision_function(Q))
    assert_array_equal(a.predict(Q), b.predict(Q))


class RestartDefaultTests(unittest.TestCase):
    def test_01_policy_is_only_compatibility(self):
        self.assertEqual(tuple(inspect.signature(resolve_acceleration).parameters),
                         ('requested', 'implementation', 'solver_mode', 'backend'))
        for implementation in ('optimized', 'reference'):
            for mode in ('schur', 'legacy'):
                for backend in ('auto', 'cholesky', 'spectral', 'lbfgs'):
                    options = dict(implementation=implementation, solver_mode=mode, backend=backend)
                    eligible = implementation == 'optimized' and mode == 'schur' and backend != 'lbfgs'
                    self.assertEqual(resolve_acceleration('auto', **options), 'restart' if eligible else None)
                    self.assertIsNone(resolve_acceleration(None, **options))
                    if eligible:
                        self.assertEqual(resolve_acceleration('restart', **options), 'restart')
                    else:
                        with self.assertRaises(ValueError): resolve_acceleration('restart', **options)
        for bad in (False, True, 1, [], {}, 'fista'):
            with self.assertRaises(ValueError): resolve_acceleration(bad)

    def test_02_declared_defaults_and_explicit_none(self):
        for factory in (KernGDWD, KernGDWDCV, solve_kernel, solve_gen_kern_dwd):
            self.assertEqual(inspect.signature(factory).parameters['acceleration'].default, 'auto')
        self.assertEqual(inspect.signature(solve_gen_kern_dwd).parameters['acceleration'].kind,
                         inspect.Parameter.KEYWORD_ONLY)
        for factory in (KernGDWD, KernGDWDCV):
            self.assertIsNone(factory(acceleration=None).get_params()['acceleration'])
            self.assertEqual(clone(factory()).get_params()['acceleration'], 'auto')

    def test_03_public_route_matches_explicit_effective_policy(self):
        routes = ({}, {'backend': 'cholesky'}, {'backend': 'spectral'},
                  {'implementation': 'reference'}, {'backend': 'lbfgs'},
                  {'solver_mode': 'legacy'})
        for route in routes:
            policy = resolve_acceleration('auto', **route)
            with self.subTest(route=route), warnings.catch_warnings():
                warnings.simplefilter('ignore', FutureWarning)
                a = KernGDWD(**COMMON, **route).fit(X, y)
                b = KernGDWD(**COMMON, **route, acceleration=policy).fit(X, y)
                same_science(a, b)
                self.assertEqual(a.acceleration, 'auto')
                self.assertEqual(a.requested_acceleration_, 'auto')
                self.assertEqual(a.acceleration_, policy)

    def test_04_stopping_never_selects_acceleration(self):
        for stopping in ('fixed', 'objective', 'optimality', 'validation'):
            options = dict(COMMON, stopping=stopping)
            fit_options = {'validation_data': (X[:8], y[:8])} if stopping == 'validation' else {}
            a = KernGDWD(**options).fit(X, y, **fit_options)
            b = KernGDWD(**options, acceleration='restart').fit(X, y, **fit_options)
            same_science(a, b)
            self.assertEqual(a.acceleration_, 'restart')

    def test_05_functional_route_and_historical_optout(self):
        for route in ({}, {'backend': 'spectral'}, {'implementation': 'reference'}, {'backend': 'lbfgs'}):
            policy = resolve_acceleration('auto', **route)
            a = solve_kernel(K, signed, .1, max_iter=4, stopping='fixed', **route)
            b = solve_kernel(K, signed, .1, max_iter=4, stopping='fixed', acceleration=policy, **route)
            for name in ('alpha', 'offset', 'objective_history', 'converged', 'termination_reason'):
                assert_array_equal(a[name], b[name])
            self.assertEqual(a['diagnostics']['requested_acceleration'], 'auto')
            self.assertEqual(a['diagnostics']['acceleration'], policy)
        for policy in (None, 'restart'):
            a = solve_gen_kern_dwd(K, y, .1, max_iter=4, obj_tol=0., acceleration=policy)
            b = solve_kernel(K, signed, .1, max_iter=4, obj_tol=0., acceleration=policy)
            for got, expected in zip(a[:3], (b['alpha'], b['offset'], b['objective_history'])):
                assert_array_equal(got, expected)
        with self.assertRaises(ValueError):
            solve_gen_kern_dwd(K, y, .1, solver_mode='legacy', acceleration='restart')

    def test_06_supplied_and_cached_eigenbasis(self):
        values, vectors = eigh(K)
        for backend in ('auto', 'spectral', 'cholesky'):
            a = KernGDWD(**COMMON, backend=backend).fit(X, y, K=K, K_eig=(vectors, values))
            b = KernGDWD(**COMMON, backend=backend, acceleration='restart').fit(X, y, K=K, K_eig=(vectors, values))
            same_science(a, b)
        a = KernGDWD(**COMMON, backend='spectral').cv_init(X)
        prepared_K, prepared_eigen = a._cv_K, a._K_eig
        a.set_params(acceleration=None).fit(X, y)
        a.set_params(acceleration='auto').fit(X, y)
        self.assertIs(a._cv_K, prepared_K)
        self.assertIs(a._K_eig, prepared_eigen)
        self.assertEqual(a.acceleration_, 'restart')

    def test_07_old_missing_field_clone_pickle_and_refit(self):
        for factory in (KernGDWD, KernGDWDCV):
            old = factory()
            del old.acceleration
            restored = pickle.loads(pickle.dumps(old))
            self.assertIsNone(restored.get_params()['acceleration'])
            self.assertIsNone(clone(restored).acceleration)
        old = KernGDWD(**COMMON)
        del old.acceleration
        old.fit(X, y)
        plain = KernGDWD(**COMMON, acceleration=None).fit(X, y)
        same_science(old, plain)
        self.assertIsNone(old.acceleration_)

    def test_08_fitted_policy_survives_constructor_mutation(self):
        fitted = KernGDWD(**COMMON).fit(X, y)
        restored = pickle.loads(pickle.dumps(fitted))
        same_science(fitted, restored)
        old_scores = fitted.decision_function(Q)
        fitted.set_params(acceleration=None)
        assert_array_equal(fitted.decision_function(Q), old_scores)
        self.assertEqual(fitted.acceleration_, 'restart')
        fitted.fit(X, y)
        self.assertIsNone(fitted.acceleration_)
        self.assertIsNone(fitted.requested_acceleration_)

    def test_09_failure_clears_new_fitted_metadata(self):
        model = KernGDWD(**COMMON).fit(X, y)
        model.set_params(acceleration='bad')
        with self.assertRaises(ValueError): model.fit(X, y)
        for name in ('acceleration_', 'requested_acceleration_', 'dual_coef_'):
            self.assertFalse(hasattr(model, name))
        for route in ({'implementation': 'reference'}, {'solver_mode': 'legacy'}, {'backend': 'lbfgs'}):
            with self.assertRaises(ValueError):
                KernGDWD(**COMMON, acceleration='restart', **route).fit(X, y)

    def test_10_automatic_spectral_recovery_keeps_concrete_restart(self):
        original = solve_kernel
        calls = []
        def injected(*args, **kwargs):
            calls.append(kwargs.copy())
            if len(calls) == 1:
                raise MMRecoveryExhausted('controlled exhausted attempt', dict(
                    completed_iterations=0, failed_iteration=1,
                    setup_seconds=0., optimization_seconds=0.))
            return original(*args, **kwargs)
        with patch('dwd._kernel_solver.solve_kernel', injected):
            model = KernGDWD(**COMMON).fit(X, y)
        self.assertEqual(len(calls), 2)
        self.assertEqual([c['acceleration'] for c in calls], ['restart', 'restart'])
        self.assertEqual(model.requested_acceleration_, 'auto')
        self.assertEqual(model.acceleration_, 'restart')
        same_science(model, KernGDWD(**COMMON, backend='spectral', acceleration='restart',
                                  residual_check_order='refinement_first').fit(X, y))

    def test_11_cv_and_grid_keep_requests(self):
        cv = KernGDWDCV(lambd_vals=[.1], q_vals=[1.], cv=2,
                       max_iter=3, stopping='fixed', random_state=44).fit(X, y)
        self.assertEqual(cv.requested_acceleration_, 'auto')
        self.assertEqual(cv.acceleration_, 'restart')
        grid = GridSearchCV(KernGDWD(**COMMON), {'acceleration': [None, 'auto']},
                            cv=2, n_jobs=1, error_score='raise').fit(X, y)
        self.assertEqual(list(grid.cv_results_['param_acceleration']), [None, 'auto'])



    def test_14_callback_and_zero_update_preserved(self):
        seen = []
        model = KernGDWD(**dict(COMMON, max_iter=0), callback=lambda s: seen.append(s['iteration'])).fit(X, y)
        self.assertEqual(seen, [0])
        self.assertEqual(model.n_iter_, 0)
        self.assertEqual(model.diagnostics_['acceleration_proposals'], 0)
        def fail(_): raise KeyboardInterrupt('controlled')
        model = KernGDWD(**COMMON, callback=fail)
        with self.assertRaises(KeyboardInterrupt): model.fit(X, y)
        self.assertFalse(hasattr(model, 'acceleration_'))


if __name__ == '__main__': unittest.main()
