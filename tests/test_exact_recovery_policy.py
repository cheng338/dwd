"""Optional exact recovery changes resource capacity, never acceptance gates."""

import hashlib
import os
from pathlib import Path
import pickle
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

import dwd._certified_kernel_mm as actions
import dwd._exact_kernel_factor as factors
import dwd._kernel_solver as solver
from dwd._kernel_linear_system import KernelLinearSystem
from dwd._kernel_mm_recovery import KernelMMRecovery
from dwd._kernel_recovery import MMRecoveryExhausted
from dwd._spectral_linear_system import SpectralLinearSystem
from dwd.cv import run_cv
from dwd.gen_kern_dwd import KernGDWD, KernGDWDCV


EXTENDED = dict(max_entries=65536, max_rank=32, max_bits=8192,
                max_operations=1000000)


class ExactRecoveryPolicyTests(unittest.TestCase):
    def setUp(self):
        limits = threadpool_limits(1)
        self.addCleanup(limits.restore_original_limits)
        values = np.linspace(-2., 2., 12)
        self.X = np.column_stack((values, np.sin(values)))
        self.y = np.where(values > 0., 7, -3)

    def model(self, **overrides):
        options = dict(kernel='rbf', kernel_kws={'gamma': .3}, max_iter=2,
                       stopping='fixed', initialization='zero', random_state=7)
        options.update(overrides)
        return KernGDWD(**options)

    def test_default_helper_calls_and_extended_caps_apply_to_every_action(self):
        K = np.eye(5) + .25 * np.ones((5, 5))
        rhs = np.arange(5, dtype=float) / 8.
        for policy in ('standard', 'extended'):
            with self.subTest(policy=policy), \
                    patch.object(factors, 'exact_kernel_factor', wraps=factors.exact_kernel_factor) as factor, \
                    patch.object(actions, 'CertifiedKernelMM', wraps=actions.CertifiedKernelMM) as action, \
                    patch.object(actions, 'ExactArithmetic', wraps=factors.ExactArithmetic) as arithmetic:
                recovery = KernelMMRecovery(K, .5, exact_recovery=policy)
                self.assertEqual(factor.call_count, 0)
                self.assertEqual(action.call_count, 0)
                recovery.step(rhs, .125, 'injected numerical failure')
                recovery.step(rhs / 2., -.25)
            kwargs = EXTENDED if policy == 'extended' else {}
            self.assertEqual(factor.call_args.kwargs, kwargs)
            self.assertEqual(action.call_args.kwargs, kwargs)
            self.assertEqual(factor.call_count, 1)
            self.assertEqual(action.call_count, 1)
            # Preparation and each step create independently capped arithmetic.
            self.assertEqual(arithmetic.call_count, 3)
            for call in arithmetic.call_args_list:
                self.assertEqual(call.kwargs, dict(max_bits=8192 if policy == 'extended' else 4096,
                                                  max_operations=1000000))
            self.assertEqual(recovery.action.max_rank, 32 if policy == 'extended' else 16)
            self.assertEqual(recovery.action.max_entries, 65536)
            self.assertEqual(recovery.info['accepted_actions'], 2)
            self.assertTrue(recovery.info['true_mm_function_checked'])
            self.assertFalse(recovery.info['kernel_approximation_used'])
            self.assertEqual(recovery.info['positive_directions_discarded'], 0)

    def test_public_naturally_failed_rank30_fit_recovers_only_when_opted_in(self):
        # Exact stored historical RBF matrix, preserved rather than regenerated
        # through a potentially different kernel backend. Its full exact rank is
        # 30; the extended action also needs more than 4096 rational bits.
        with np.load(Path(__file__).parent / 'fixtures/nearly_constant_rbf_rank30.npz',
                     allow_pickle=False) as data:
            K = data['K'].copy()
        self.assertEqual(hashlib.sha256(K.tobytes()).hexdigest(),
                         '8218ead0da0babcbcc97f58170134b77228925f58cc479c131e05e3b8ec4c9c0')
        before = K.tobytes()
        y = np.where(np.arange(len(K)) % 2, 1, -1)
        for implementation in ('reference', 'optimized'):
            with self.subTest(implementation=implementation):
                model = self.model(kernel='precomputed', kernel_kws=None,
                                   implementation=implementation, lambd=2e-14 / len(K),
                                   max_iter=1, random_state=314159)
                with self.assertRaisesRegex(FloatingPointError, 'rank_budget_exhausted'):
                    model.fit(K, y)
                self.assertFalse(hasattr(model, 'dual_coef_'))
                self.assertFalse(hasattr(model, 'exact_recovery_'))
                model.set_params(exact_recovery='extended').fit(K, y)
                self.assertEqual(model.exact_recovery_, 'extended')
                self.assertEqual(model.n_iter_, 1)
                info = model.diagnostics_['mm_function_recovery']
                self.assertEqual(info['certificate']['rank'], 30)
                self.assertEqual(info['resource_caps'], EXTENDED)
                self.assertEqual(info['accepted_actions'], 1)
                self.assertTrue(info['true_mm_function_checked'])
                self.assertFalse(info['kernel_approximation_used'])
                self.assertTrue(np.isfinite(model.decision_function(K)).all())
                self.assertEqual(K.tobytes(), before)

    def test_healthy_fits_and_default_call_shape_are_unchanged(self):
        native = KernelMMRecovery
        for options in ({}, {'implementation': 'reference'}, {'backend': 'spectral'},
                        {'backend': 'lbfgs'}, {'acceleration': 'restart'}):
            with self.subTest(options=options), \
                    patch.object(KernelMMRecovery, 'step', side_effect=AssertionError('Unneeded recovery')), \
                    patch.object(solver, 'KernelMMRecovery', wraps=native) as recovery:
                ordinary = self.model(**options).fit(self.X, self.y)
                self.assertEqual(recovery.call_args.kwargs, {})
                explicit = self.model(exact_recovery='standard', **options).fit(self.X, self.y)
                self.assertEqual(recovery.call_args.kwargs, {})
                extended = self.model(exact_recovery='extended', **options).fit(self.X, self.y)
                self.assertEqual(recovery.call_args.kwargs, {'exact_recovery': 'extended'})
            for candidate in (explicit, extended):
                for name in ('dual_coef_', 'intercept_', 'objective_history_'):
                    np.testing.assert_array_equal(getattr(candidate, name), getattr(ordinary, name))
                np.testing.assert_array_equal(candidate.decision_function(self.X),
                                              ordinary.decision_function(self.X))
                self.assertNotIn('mm_function_recovery', candidate.diagnostics_)

    def test_healthy_extended_path_does_not_import_exact_modules(self):
        script = '''import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from dwd.gen_kern_dwd import KernGDWD
from threadpoolctl import threadpool_limits
names = ('dwd._exact_kernel_factor', 'dwd._certified_kernel_mm')
assert all(name not in sys.modules for name in names)
with threadpool_limits(1):
 for policy in ('standard', 'extended'):
  for impl in ('reference', 'optimized'):
   KernGDWD(kernel='precomputed', implementation=impl, max_iter=2,
            stopping='fixed', exact_recovery=policy).fit(np.eye(6), [-1, 1, -1, 1, -1, 1])
assert all(name not in sys.modules for name in names)
'''
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', BLIS_NUM_THREADS='1')
        result = subprocess.run([sys.executable, '-I', '-B', '-c', script,
                                 str(Path(__file__).resolve().parents[1])],
                                capture_output=True, text=True, env=env, timeout=30,
                                creationflags=0x08000000 if os.name == 'nt' else 0)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_extended_is_forwarded_through_automatic_spectral_restart(self):
        native = solver.solve_kernel
        calls = []

        def once(K, y, lambd, **options):
            calls.append(dict(options))
            if len(calls) == 1:
                raise MMRecoveryExhausted('injected exhausted MM recovery', dict(
                    failed_iteration=1, completed_iterations=0,
                    setup_seconds=0., optimization_seconds=0.))
            return native(K, y, lambd, **options)

        with patch.object(solver, 'solve_kernel', once):
            model = self.model(exact_recovery='extended').fit(self.X, self.y)
        self.assertEqual([c['backend'] for c in calls], ['auto', 'spectral'])
        self.assertEqual([c['exact_recovery'] for c in calls], ['extended', 'extended'])
        self.assertEqual(model.exact_recovery_, 'extended')
        self.assertTrue(model.diagnostics_['auto_fallback'])

    def test_extended_boundary_and_indefinite_refusal_remain_complete(self):
        for rank in (0, 1, 5, 16, 17, 32, 33):
            K = np.diag(np.r_[np.ones(rank), np.zeros(2)])
            recovery = KernelMMRecovery(K, .5, exact_recovery='extended')
            if rank == 33:
                with self.assertRaisesRegex(FloatingPointError, 'rank_budget_exhausted'):
                    recovery.step(np.zeros(len(K)), 0., 'injected failure')
                self.assertIsNone(recovery.action)
            else:
                recovery.step(np.zeros(len(K)), 0., 'injected failure')
                self.assertEqual(recovery.action.rank, rank)
                self.assertEqual(recovery.info['positive_directions_discarded'], 0)
        for K, expected in ((np.eye(257), 'entry_budget_exhausted'),
                            (np.diag([1., -1.]), 'exact_negative_schur_diagonal')):
            recovery = KernelMMRecovery(K, .5, exact_recovery='extended')
            with self.assertRaisesRegex(FloatingPointError, expected):
                recovery.step(np.zeros(len(K)), 0., 'injected failure')
            self.assertIsNone(recovery.action)
            self.assertEqual(recovery.info['accepted_actions'], 0)

    def test_interruption_and_failed_preparation_clear_refitted_state(self):
        K, y = np.eye(6), np.array([-1, 1, -1, 1, -1, 1])
        native = factors.ExactArithmetic._tick

        def interrupt(ar, name):
            if ar.operations >= 32:
                raise KeyboardInterrupt('interrupted exact preparation')
            return native(ar, name)

        for implementation in ('reference', 'optimized'):
            model = self.model(kernel='precomputed', kernel_kws=None,
                               implementation=implementation, exact_recovery='extended').fit(K, y)
            target = SpectralLinearSystem if implementation == 'reference' else KernelLinearSystem
            method = 'refine_candidate' if implementation == 'reference' else 'solve_constrained'
            with patch.object(target, method, side_effect=FloatingPointError('Injected failure')), \
                    patch.object(factors.ExactArithmetic, '_tick', interrupt):
                with self.assertRaisesRegex(KeyboardInterrupt, 'interrupted exact preparation'):
                    model.fit(K, y)
            self.assertFalse(hasattr(model, 'dual_coef_'))
            self.assertFalse(hasattr(model, 'exact_recovery_'))
            with self.assertRaises(NotFittedError):
                model.predict(K)
        model = self.model(kernel='precomputed', kernel_kws=None,
                           exact_recovery='extended').fit(K, y)
        with patch.object(KernelLinearSystem, 'solve_constrained', side_effect=FloatingPointError('Injected')), \
                patch.object(actions, 'CertifiedKernelMM', side_effect=FloatingPointError('Rejected preparation')):
            with self.assertRaisesRegex(FloatingPointError, 'Rejected preparation'):
                model.fit(K, y)
        self.assertFalse(hasattr(model, 'dual_coef_'))
        self.assertFalse(hasattr(model, 'exact_recovery_'))

    def test_old_pickle_clone_and_refit_only_provenance(self):
        model = self.model().cv_init(self.X).fit(self.X, self.y)
        expected = model.decision_function(self.X)
        for name in ('exact_recovery', 'exact_recovery_', '_cv_exact_recovery'):
            del model.__dict__[name]
        old = pickle.loads(pickle.dumps(model))
        self.assertEqual(old.get_params()['exact_recovery'], 'standard')
        self.assertEqual(clone(old).exact_recovery, 'standard')
        self.assertTrue(old._cv_cache_matches(self.X))
        old.set_params(exact_recovery='extended')
        np.testing.assert_array_equal(old.decision_function(self.X), expected)
        self.assertFalse(old._cv_cache_matches(self.X))
        for policy, changed in (('extended', 'standard'), ('standard', 'extended')):
            fitted = self.model(exact_recovery=policy).fit(self.X, self.y)
            expected = fitted.decision_function(self.X)
            fitted.set_params(exact_recovery=changed)
            restored = pickle.loads(pickle.dumps(fitted))
            self.assertEqual(restored.exact_recovery_, policy)
            np.testing.assert_array_equal(restored.decision_function(self.X), expected)
            restored.fit(self.X, self.y)
            self.assertEqual(restored.exact_recovery_, changed)
        for estimator in (self.model(exact_recovery='extended'),
                          KernGDWDCV(exact_recovery='extended')):
            self.assertEqual(clone(estimator).exact_recovery, 'extended')
            del estimator.__dict__['exact_recovery']
            self.assertEqual(clone(pickle.loads(pickle.dumps(estimator))).exact_recovery, 'standard')

    def test_cache_identity_public_cv_and_mixed_grid(self):
        model = self.model().cv_init(self.X)
        self.assertTrue(model._cv_cache_matches(self.X))
        model.set_params(exact_recovery='extended')
        self.assertFalse(model._cv_cache_matches(self.X))
        model.cv_init(self.X)
        self.assertTrue(model._cv_cache_matches(self.X))
        model.set_params(exact_recovery='standard')
        self.assertFalse(model._cv_cache_matches(self.X))
        cv = KernGDWDCV(lambd_vals=[.1], q_vals=[1.], cv=2, max_iter=2,
                        stopping='fixed', random_state=7, exact_recovery='extended').fit(self.X, self.y)
        self.assertEqual(cv.exact_recovery_, 'extended')
        self.assertEqual(cv.best_estimator_.exact_recovery_, 'extended')
        expected = cv.decision_function(self.X)
        cv.set_params(exact_recovery='standard')
        np.testing.assert_array_equal(pickle.loads(pickle.dumps(cv)).decision_function(self.X), expected)
        self.assertEqual(cv.exact_recovery_, 'extended')
        folds = list(StratifiedKFold(2, shuffle=True, random_state=19).split(self.X, self.y))
        mixed = run_cv(self.model(), self.X, self.y,
                       {'exact_recovery': ['standard', 'extended'], 'lambd': [.1, .2]},
                       cv=folds, refit_best=False)[3]
        for policy in ('standard', 'extended'):
            separate = run_cv(self.model(exact_recovery=policy), self.X, self.y,
                              {'lambd': [.1, .2]}, cv=folds, refit_best=False)[3]
            for params, score in zip(separate['params'], separate['mean_test_score']):
                index = mixed['params'].index(dict(params, exact_recovery=policy))
                self.assertEqual(mixed['mean_test_score'][index], score)

    def test_extended_combines_with_direct_rbf_and_joint_affine(self):
        model = self.model(exact_recovery='extended', rbf_computation='direct',
                           affine_computation='joint').fit(self.X, self.y)
        self.assertEqual(model.exact_recovery_, 'extended')
        self.assertEqual(model.kernel_computation_, 'direct_v1')
        self.assertEqual(model.affine_computation_, 'joint')
        self.assertTrue(np.isfinite(model.decision_function(self.X)).all())

    def test_invalid_options_and_legacy_refuse_before_numerical_work(self):
        for value in (None, True, 0, [], np.array(['standard']), 'invalid'):
            for method in ('fit', 'cv_init'):
                with self.subTest(value=value, method=method), self.assertRaisesRegex(ValueError, 'exact_recovery'):
                    model = self.model(exact_recovery=value)
                    model.fit(self.X, self.y) if method == 'fit' else model.cv_init(self.X)
            with self.assertRaisesRegex(ValueError, 'exact_recovery'):
                KernGDWDCV(exact_recovery=value).fit(self.X, self.y)
            with self.assertRaisesRegex(ValueError, 'exact_recovery'):
                solver.solve_kernel(np.eye(2), np.array([-1., 1.]), .1, exact_recovery=value)
        with self.assertRaisesRegex(ValueError, 'requires solver_mode'):
            self.model(solver_mode='legacy', exact_recovery='extended').fit(self.X, self.y)
        with self.assertRaisesRegex(ValueError, 'requires solver_mode'):
            KernGDWDCV(solver_mode='legacy', exact_recovery='extended').fit(self.X, self.y)
        legacy = self.model(kernel='linear', kernel_kws=None, solver_mode='legacy').fit(self.X, self.y)
        self.assertEqual(legacy.exact_recovery_, 'standard')
        legacy.set_params(exact_recovery='invalid')
        with self.assertRaises(ValueError):
            legacy.fit(self.X, self.y)
        self.assertFalse(hasattr(legacy, 'exact_recovery_'))


if __name__ == '__main__':
    unittest.main()
