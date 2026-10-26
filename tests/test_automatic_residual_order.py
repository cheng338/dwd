"""Eligible automatic residual ordering and preserved recovery contracts."""
import copy
import pickle
import tempfile
import unittest
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
import inspect
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import Pipeline
from dwd._kernel_solver import solve_kernel
from dwd._kernel_recovery import MMRecoveryExhausted
import dwd._kernel_recovery as recovery
from numerical_test_support import AcceptedNativeReplay, objective_oracle

rng = np.random.RandomState(8201)
X = rng.normal(size=(40, 4))
y = np.tile([-1, 1], 20)
Q = rng.normal(size=(7, 4))
COMMON = dict(lambd=.1, kernel='rbf', kernel_kws={'gamma': .3},
              max_iter=4, stopping='fixed', random_state=44)
ENS = dict(M=2, theta=.25, sv_selection_ratio=.6, lambd=.1, gamma=.3,
           max_iter=4, stopping='fixed', random_state=44)

def error_details():
    return dict(completed_iterations=0, failed_iteration=1,
                setup_seconds=0., optimization_seconds=0.)

def supplied_eig(K):
    values, vectors = eigh(K)
    # Package public K_eig uses vectors first; SciPy returns values first.
    return vectors, values

def scientific(model):
    result = dict(coef=model.dual_coef_.copy(), intercept=model.intercept_.copy(),
                  history=model.objective_history_.copy(), scores=model.decision_function(Q),
                  labels=model.predict(Q))
    for key in ('n_iter_', 'returned_iteration_', 'termination_reason_', 'converged_',
                'criterion_reached_', 'final_objective_', 'gradient_inf_norm_',
                'rkhs_gradient_norm_', 'dual_gap_', 'backend_', 'prediction_precision_'):
        result[key] = getattr(model, key)
    return result

def same(first, second):
    assert first.keys() == second.keys()
    for key in first:
        assert_array_equal(first[key], second[key], err_msg=key)

ROUTES = {
    'optimized_auto': {}, 'cholesky': {'backend': 'cholesky'},
    'restart': {'acceleration': 'restart'},
    'reference': {'implementation': 'reference'},
    'spectral': {'backend': 'spectral'}, 'lbfgs': {'backend': 'lbfgs'},
    'legacy': {'solver_mode': 'legacy'},
}


class DispatchTests(unittest.TestCase):

    def test_02_automatic_dispatch_and_concrete_parity(self):
        for name, route in ROUTES.items():
            with self.subTest(route=name):
                policy = 'adaptive' if name in ('optimized_auto', 'cholesky', 'restart') else 'refinement_first'
                a = KernGDWD(**COMMON, **route).fit(X, y)
                b = KernGDWD(**COMMON, **route, residual_check_order=policy).fit(X, y)
                same(scientific(a), scientific(b))
                self.assertEqual(a.get_params()['residual_check_order'], 'auto')
                self.assertEqual(a.requested_residual_check_order_, 'auto')
                self.assertEqual(a.residual_check_order_, policy)
                self.assertEqual(a.diagnostics_['initial_residual_check_order'], policy)

    def test_03_supplied_eigensystems_resolve_after_fit_arguments(self):
        eig = supplied_eig(rbf_kernel(X, gamma=0.3))
        for backend in ('auto', 'cholesky'):
            a = KernGDWD(**COMMON, backend=backend).fit(X, y, K_eig=eig)
            b = KernGDWD(**COMMON, backend=backend, residual_check_order='refinement_first').fit(X, y, K_eig=eig)
            same(scientific(a), scientific(b))
            self.assertEqual(a.residual_check_order_, 'refinement_first')
        with self.assertRaises(ValueError):
            KernGDWD(**COMMON, residual_check_order='adaptive').fit(X, y, K_eig=eig)
        explicit = KernGDWD(**COMMON, backend='cholesky', residual_check_order='adaptive').fit(X, y, K_eig=eig)
        self.assertEqual(explicit.residual_check_order_, 'adaptive')

    def test_04_cached_eigensystem_and_cv_cache_invalidation(self):
        a = KernGDWD(**COMMON).cv_init(X)
        self.assertIsNone(a._K_eig)
        a.fit(X, y)
        self.assertEqual(a.residual_check_order_, 'adaptive')
        a.cv_init(X)
        a._K_eig = supplied_eig(a._cv_K)
        a.fit(X, y)
        self.assertEqual(a.residual_check_order_, 'refinement_first')
        self.assertEqual(a.backend_, 'spectral')
        a.set_params(backend='cholesky').fit(X, y)
        self.assertEqual(a.residual_check_order_, 'adaptive')
        self.assertEqual(a.backend_, 'cholesky')

    def test_05_cv_and_sklearn_pipeline_forwarding(self):
        seen = []
        original = KernGDWD.fit

        def recorded(model, *a, **k):
            result = original(model, *a, **k)
            seen.append((model.residual_check_order, model.residual_check_order_))
            return result
        cv = KernGDWDCV(lambd_vals=[0.1, 0.2], q_vals=[1.0], cv=2, max_iter=2, stopping='fixed', random_state=44)
        with patch.object(KernGDWD, 'fit', recorded):
            cv.fit(X, y)
        self.assertEqual(seen, [('auto', 'adaptive')] * 5)
        self.assertEqual(cv.residual_check_order_, 'adaptive')
        self.assertEqual(cv.requested_residual_check_order_, 'auto')
        grid = GridSearchCV(Pipeline([('dwd', KernGDWD(**COMMON))]), {'dwd__lambd': [0.1, 0.2]}, cv=2, n_jobs=1, error_score='raise')
        grid.fit(X, y)
        self.assertEqual(grid.best_estimator_['dwd'].residual_check_order_, 'adaptive')

    def _assert_retry_arguments(self, calls):
        self.assertEqual(len(calls), 2)
        first, second = calls
        self.assertIs(first['args'][0], second['args'][0])
        self.assertIs(first['args'][1], second['args'][1])
        self.assertEqual(first['args'][2], second['args'][2])
        expected = dict(first['kwargs'], backend='spectral',
                        residual_check_order='refinement_first')
        self.assertEqual(set(expected), set(second['kwargs']))
        for name, value in expected.items():
            if isinstance(value, np.ndarray):
                assert_array_equal(value, second['kwargs'][name],
                                   err_msg='retry argument ' + name)
            else:
                self.assertEqual(value, second['kwargs'][name],
                                 'retry argument ' + name)
        assert_array_equal(first['kernel_snapshot'], second['kernel_snapshot'])
        # Publicly supplied K_eig would make this recovery route ineligible.
        self.assertIsNone(first['kwargs']['K_eig'])
        self.assertIsNone(second['kwargs']['K_eig'])

    def _assert_recovery_pair(self, requested, exact):
        calls = []

        def injected(*args, **kwargs):
            calls.append(dict(args=args, kwargs=copy.deepcopy(kwargs),
                              kernel_snapshot=args[0].copy()))
            if len(calls) == 1:
                raise MMRecoveryExhausted('controlled first-attempt exhaustion',
                                         error_details())
            return solve_kernel(*args, **kwargs)

        with patch('dwd._kernel_solver.solve_kernel', injected):
            model = KernGDWD(**COMMON, residual_check_order=requested).fit(X, y)
        self._assert_retry_arguments(calls)
        self.assertEqual(model.requested_residual_check_order_, requested)
        self.assertEqual(model.residual_check_order_, 'refinement_first')
        self.assertEqual(model.diagnostics_['initial_residual_check_order'],
                         'adaptive' if requested in ('auto', 'adaptive')
                         else 'refinement_first')
        self.assertEqual(model.diagnostics_['attempted_backends'],
                         ['cholesky', 'spectral'])
        self.assertEqual(model.diagnostics_['spectral_restart']['history_scope'],
                         'successful_spectral_attempt')
        reference = KernGDWD(**COMMON, backend='spectral',
                            residual_check_order='refinement_first').fit(X, y)
        left, right = scientific(model), scientific(reference)
        if exact:
            same(left, right)
        else:
            continuous = {'coef', 'intercept', 'history', 'scores',
                          'final_objective_', 'gradient_inf_norm_',
                          'rkhs_gradient_norm_', 'dual_gap_'}
            self.assertEqual(set(left), set(right))
            for name in left:
                if name in continuous:
                    # Fixed-fixture bounds; labels and discrete state stay exact.
                    assert_allclose(left[name], right[name], rtol=1e-10,
                                    atol=1e-12, err_msg=name)
                else:
                    assert_array_equal(left[name], right[name], err_msg=name)
        K = calls[0]['kernel_snapshot']
        signed = calls[0]['args'][1]
        for fitted in (model, reference):
            oracle = objective_oracle(K, signed, fitted.dual_coef_.ravel(),
                                      fitted.intercept_[0], .1)
            self.assertAlmostEqual(fitted.final_objective_, oracle,
                                   delta=5e-13 * max(1., abs(oracle)))

    def test_09_spectral_failure_recovery_preserves_three_policy_diagnostics(self):
        for requested in ('auto', 'adaptive', 'refinement_first'):
            with self.subTest(requested=requested), AcceptedNativeReplay() as replay:
                self._assert_recovery_pair(requested, exact=True)
                self.assertGreater(replay.replays, 0)

    def test_recovery_policy_with_independent_native_preparations(self):
        for requested in ('auto', 'adaptive', 'refinement_first'):
            with self.subTest(requested=requested):
                self._assert_recovery_pair(requested, exact=False)

    def test_wrong_retry_settings_are_detected(self):
        original = recovery.solve_with_spectral_restart
        for changed in ({'residual_check_order': 'auto'}, {'max_iter': 5}):
            def mutate(solve, *args, **kwargs):
                def changed_retry(*inner_args, **options):
                    if options.get('backend') == 'spectral':
                        options = dict(options, **changed)
                    return solve(*inner_args, **options)
                return original(changed_retry, *args, **kwargs)

            with self.subTest(changed=changed), AcceptedNativeReplay(), \
                    patch.object(recovery, 'solve_with_spectral_restart', mutate):
                with self.assertRaisesRegex(AssertionError, 'retry argument'):
                    self._assert_recovery_pair('auto', exact=True)

    def test_10_ineligible_recovery_and_interrupts_not_swallowed(self):
        for extra in ({'callback': lambda state: False}, {'backend': 'cholesky'}):
            with patch('dwd._kernel_solver.solve_kernel', side_effect=MMRecoveryExhausted('controlled', error_details())) as solver:
                model = KernGDWD(**COMMON, **extra)
                with self.assertRaises(MMRecoveryExhausted):
                    model.fit(X, y)
                self.assertEqual(solver.call_count, 1)
                self.assertFalse(hasattr(model, 'residual_check_order_'))
        with patch('dwd._kernel_solver.solve_kernel', side_effect=MMRecoveryExhausted('controlled', error_details())) as solver:
            with self.assertRaises(MMRecoveryExhausted):
                KernGDWD(**COMMON).fit(X, y, K_eig=supplied_eig(rbf_kernel(X, gamma=0.3)))
            self.assertEqual(solver.call_count, 1)
        for failure in (KeyboardInterrupt, SystemExit, RuntimeError):
            with patch('dwd._kernel_solver.solve_kernel', side_effect=failure) as solver:
                with self.assertRaises(failure):
                    KernGDWD(**COMMON).fit(X, y)
                self.assertEqual(solver.call_count, 1)

    def test_11_functional_api_historical_default_and_explicit_auto(self):
        K = X @ X.T + np.eye(len(X))
        self.assertEqual(inspect.signature(solve_kernel).parameters['residual_check_order'].default, 'refinement_first')
        self.assertEqual(solve_kernel(K, y, 0.1, max_iter=2)['diagnostics']['residual_check_order'], 'refinement_first')
        self.assertEqual(solve_kernel(K, y, 0.1, max_iter=2, residual_check_order='auto')['diagnostics']['residual_check_order'], 'adaptive')
        for backend in ('spectral', 'lbfgs'):
            got = solve_kernel(K, y, 0.1, max_iter=2, backend=backend, residual_check_order='auto')
            self.assertEqual(got['diagnostics']['residual_check_order'], 'refinement_first')

    def test_12_explicit_validation_and_invalid_eigenbasis_remain_strict(self):
        for route in ({'implementation': 'reference'}, {'solver_mode': 'legacy'}, {'backend': 'spectral'}, {'backend': 'lbfgs'}):
            for factory in (KernGDWD, KernGDWDCV):
                with self.assertRaises(ValueError):
                    factory(residual_check_order='adaptive', **route).fit(X, y)
        for bad in (None, False, 1, 'wrong'):
            for factory in (KernGDWD, KernGDWDCV):
                with self.assertRaises(ValueError):
                    factory(residual_check_order=bad).fit(X, y)
        eig = supplied_eig(rbf_kernel(X, gamma=0.3))
        with self.assertRaises((ValueError, FloatingPointError)):
            KernGDWD(**COMMON).fit(X, y, K_eig=(eig[0], eig[1] + 3.0))
