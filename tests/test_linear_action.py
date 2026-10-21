"""Explicit inverse integration keeps the current acceptance and ownership contract."""
import gc
import inspect
import pickle
import unittest
import weakref
from unittest.mock import patch
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from scipy.linalg import eigh
from sklearn.base import clone
import dwd._kernel_linear_system as linear
import dwd._kernel_solver as solver
from dwd._kernel_recovery import MMRecoveryExhausted, ConstrainedSolveFailure
from dwd._linear_action import validate_linear_action
from dwd.gen_kern_dwd import KernGDWD, KernGDWDCV, solve_gen_kern_dwd
X = np.array([[0., 1.], [1., 0.], [1., 1.], [2., 0.], [0., 2.], [2., 2.]])
Y = np.array([-1., 1., -1., 1., -1., 1.])
K = X @ X.T + np.eye(len(X))
COMMON = dict(kernel='rbf', kernel_kws={'gamma': .3}, lambd=.1,
              max_iter=4, stopping='fixed', random_state=42)
def same_results(a, b):
    for key in ('alpha', 'offset', 'objective_history', 'final_objective',
                'rkhs_gradient_norm', 'n_iter', 'termination_reason', 'converged'):
        assert_array_equal(a[key], b[key], err_msg=key)
class KernelLinearActionTests(unittest.TestCase):
    def test_04_bordered_reference_and_owned_readonly_action(self):
        system = linear.KernelLinearSystem(K, .125, linear_action='inverse')
        B, v, denominator = system._full_inverse_action
        self.assertFalse(B.flags.writeable)
        self.assertFalse(v.flags.writeable)
        self.assertFalse(np.shares_memory(B, K))
        self.assertIsNot(v, system.v)
        rhs = np.arange(6) / 8.
        x, offset = system.solve_constrained(rhs, .25)
        bordered = np.block([[K + .125 * np.eye(6), np.ones((6, 1))],
                             [np.ones((1, 6)), np.zeros((1, 1))]])
        expected = np.linalg.solve(bordered, np.r_[rhs, .25])
        assert_allclose(np.r_[x, offset], expected, rtol=1e-12, atol=1e-13)
        self.assertGreater(denominator, 0.)

    def test_05_numerical_build_refusal_keeps_factor_schur_state(self):
        expected = linear.KernelLinearSystem(K, .125)
        original = linear.cho_solve
        def fail_matrix(factor, rhs, **kwargs):
            if rhs.ndim == 2:
                raise FloatingPointError('controlled inverse construction refusal')
            return original(factor, rhs, **kwargs)
        with patch.object(linear, 'cho_solve', fail_matrix):
            actual = linear.KernelLinearSystem(K, .125, linear_action='inverse')
        self.assertIsNone(actual._full_inverse_action)
        assert_array_equal(actual.v, expected.v)
        self.assertEqual(actual.denominator, expected.denominator)
        self.assertEqual(actual.info['effective_linear_action'], 'factor')
        self.assertIn('controlled', actual.info['inverse_build_failure'])
        for a, b in zip(actual.solve_constrained(Y, 0.), expected.solve_constrained(Y, 0.)):
            assert_array_equal(a, b)

    def test_06_memoryerror_propagates_and_failed_fit_cleans(self):
        model = KernGDWD(**COMMON).fit(X, Y)
        model.set_params(linear_action='inverse')
        original = linear.cho_solve
        def fail_matrix(factor, rhs, **kwargs):
            if rhs.ndim == 2:
                raise MemoryError('controlled full inverse allocation')
            return original(factor, rhs, **kwargs)
        with patch.object(linear, 'cho_solve', fail_matrix), self.assertRaises(MemoryError):
            model.fit(X, Y)
        for name in ('dual_coef_', 'linear_action_', '_Xfit'):
            self.assertFalse(hasattr(model, name))

    def test_07_actual_recovery_releases_B_before_factor_allocation(self):
        system = linear.KernelLinearSystem(K, .125, linear_action='inverse')
        reference = weakref.ref(system._full_inverse_action[0])
        original_factor, original_candidate = linear.cho_factor, system._candidate
        calls = []
        def factor(*args, **kwargs):
            self.assertIsNone(reference())
            self.assertIsNone(system._full_inverse_action)
            calls.append('reprepared')
            return original_factor(*args, **kwargs)
        def candidate(*args):
            if system.mode == 'original':
                raise FloatingPointError('controlled original action refusal')
            return original_candidate(*args)
        with patch.object(linear, 'cho_factor', factor), patch.object(system, '_candidate', candidate):
            system.solve_constrained(Y, 0.)
        self.assertEqual(calls, ['reprepared'])
        self.assertEqual(system.mode, 'centered')
        self.assertEqual(system.info['effective_linear_action'], 'factor')
        self.assertGreater(system.info['linear_recoveries'], 0)

    def test_08_original_equation_checks_still_reject_bad_actions(self):
        system = linear.KernelLinearSystem(K, .125, linear_action='inverse')
        with patch.object(system, '_candidate', return_value=(np.zeros(6), 0.)), self.assertRaises(ConstrainedSolveFailure):
            system.solve_constrained(Y, 0.)
        self.assertIsNone(system._full_inverse_action)

    def test_09_policy_routes_and_supplied_eigen_validation(self):
        for bad in (None, True, [], 'auto', 'eigen'):
            with self.assertRaises(ValueError): validate_linear_action(bad)
        for options in ({'backend': 'spectral'}, {'backend': 'lbfgs'},
                        {'implementation': 'reference'}, {'solver_mode': 'legacy'}):
            with self.assertRaises(ValueError): KernGDWD(**COMMON, linear_action='inverse', **options).fit(X, Y)
        values, vectors = eigh(K)
        with self.assertRaises(ValueError):
            solver.solve_kernel(K, Y, .1, linear_action='inverse', K_eig=(vectors, values))
        a = solver.solve_kernel(K, Y, .1, linear_action='inverse', backend='cholesky',
                                K_eig=(vectors, values), max_iter=3, stopping='fixed')
        b = solver.solve_kernel(K, Y, .1, linear_action='inverse', backend='cholesky',
                                max_iter=3, stopping='fixed')
        same_results(a, b)
        with self.assertRaises((ValueError, FloatingPointError)):
            solver.solve_kernel(K, Y, .1, linear_action='inverse', backend='cholesky',
                                K_eig=(vectors, values + 1.))

    def test_10_existing_spectral_restart_resets_action_and_preserves_request(self):
        original = solver.solve_kernel
        calls = []
        def injected(*args, **kwargs):
            calls.append(kwargs.copy())
            if len(calls) == 1:
                raise MMRecoveryExhausted('controlled', dict(completed_iterations=0,
                    failed_iteration=1, setup_seconds=0., optimization_seconds=0.))
            return original(*args, **kwargs)
        with patch.object(solver, 'solve_kernel', injected):
            model = KernGDWD(**COMMON, linear_action='inverse').fit(X, Y)
        self.assertEqual([c['linear_action'] for c in calls], ['inverse', 'factor'])
        self.assertEqual(model.requested_linear_action_, 'inverse')
        self.assertEqual(model.linear_action_, 'factor')
        self.assertEqual(model.backend_, 'spectral')

    def test_11_api_clone_pickle_old_objects_and_refit(self):
        for cls in (KernGDWD, KernGDWDCV):
            self.assertEqual(inspect.signature(cls).parameters['linear_action'].default, 'factor')
            self.assertEqual(clone(cls(linear_action='inverse')).linear_action, 'inverse')
            old = cls()
            del old.linear_action
            self.assertEqual(clone(pickle.loads(pickle.dumps(old))).linear_action, 'factor')
        model = KernGDWD(**COMMON, linear_action='inverse').fit(X, Y)
        self.assertEqual(model.linear_action_, 'inverse')
        restored = pickle.loads(pickle.dumps(model))
        assert_array_equal(restored.decision_function(X), model.decision_function(X))
        self.assertNotIn('_full_inverse_action', vars(model))
        self.assertNotIn('factor', vars(model))
        model.set_params(linear_action='factor')
        self.assertEqual(model.linear_action_, 'inverse')
        model.fit(X, Y)
        self.assertEqual(model.linear_action_, 'factor')

    def test_12_workspace_dies_on_fit_return_and_callback_exception(self):
        references = []
        original = solver.KernelLinearSystem
        def factory(*args, **kwargs):
            obj = original(*args, **kwargs)
            references.append((weakref.ref(obj), weakref.ref(obj._full_inverse_action[0])))
            return obj
        with patch.object(solver, 'KernelLinearSystem', factory):
            KernGDWD(**COMMON, linear_action='inverse').fit(X, Y)
            def stop(_): raise KeyboardInterrupt('controlled')
            with self.assertRaises(KeyboardInterrupt):
                KernGDWD(**COMMON, linear_action='inverse', callback=stop).fit(X, Y)
        gc.collect()
        self.assertEqual(len(references), 2)
        self.assertTrue(all(ref() is None for pair in references for ref in pair))

    def test_13_cv_kernel_reuse_but_new_workspace_for_each_shift(self):
        model = KernGDWD(**COMMON, linear_action='inverse').cv_init(X)
        gram = model._cv_K
        self.assertIsNone(model._K_eig)
        model.fit(X, Y)
        first = model.dual_coef_.copy()
        model.set_params(lambd=.2).fit(X, Y)
        self.assertIs(model._cv_K, gram)
        self.assertFalse(np.array_equal(first, model.dual_coef_))
        cv = KernGDWDCV(lambd_vals=[.1], q_vals=[1.], cv=2, kernel='rbf',
                        kernel_kws_vals=[{'gamma': .3}], max_iter=3,
                        stopping='fixed', linear_action='inverse').fit(X, Y)
        self.assertEqual(cv.best_estimator_.linear_action_, 'inverse')
        self.assertEqual(cv.requested_linear_action_, 'inverse')

    def test_14_historical_functional_wrapper_not_implicit_P_redefinition(self):
        self.assertEqual(inspect.signature(solve_gen_kern_dwd).parameters['linear_action'].kind,
                         inspect.Parameter.KEYWORD_ONLY)
        a = solve_gen_kern_dwd(K, Y, .1, max_iter=3, obj_tol=0., linear_action='inverse')
        b = solver.solve_kernel(K, Y, .1, max_iter=3, obj_tol=0., linear_action='inverse')
        for got, want in zip(a[:3], (b['alpha'], b['offset'], b['objective_history'])):
            assert_array_equal(got, want)
        with self.assertRaises(NotImplementedError):
            solve_gen_kern_dwd(K, Y, .1, implicit_P=False, linear_action='inverse')

    def test_17_spectral_subclass_owns_its_action_without_new_base_attributes(self):
        from dwd._spectral_linear_system import SpectralLinearSystem
        values, vectors = eigh(K)
        system = SpectralLinearSystem(K, .125, vectors, values)
        self.assertFalse(hasattr(system, '_full_inverse_action'))
        x, offset = system.solve_constrained(Y, 0.)
        assert_allclose((K + .125 * np.eye(6)) @ x + offset, Y, rtol=1e-12, atol=1e-12)
        result = solver.solve_kernel(K, Y, .1, backend='spectral', implementation='reference',
                                     max_iter=3, stopping='fixed')
        self.assertEqual(result['diagnostics']['linear_action'], 'factor')
