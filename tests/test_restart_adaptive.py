"""Qualification of opt-in residual scheduling with restarted kernel MM."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from sklearn.base import clone
from sklearn.metrics.pairwise import rbf_kernel

from dwd._kernel_solver import solve_kernel
from dwd._kernel_linear_system import KernelLinearSystem
from dwd.gen_kern_dwd import KernGDWD, KernGDWDCV


class RestartAdaptiveTests(unittest.TestCase):
    def setUp(self):
        self.X = np.random.default_rng(872).normal(size=(32, 5))
        self.y = np.tile([-1., 1.], 16)
        self.K = rbf_kernel(self.X, gamma=.13)

    def test_public_estimator_and_internal_cv_forward_both_options(self):
        model = KernGDWD(kernel='rbf', kernel_kws={'gamma': .13}, lambd=.1,
            acceleration='restart', residual_check_order='adaptive',
            max_iter=4, stopping='fixed')
        fitted = clone(model).fit(self.X, self.y)
        self.assertEqual(fitted.diagnostics_['acceleration'], 'restart')
        self.assertEqual(fitted.diagnostics_['residual_check_order'], 'adaptive')
        seen = []
        original = KernGDWD.fit
        def watch(model, *args, **kwargs):
            seen.append((model.acceleration, model.residual_check_order))
            return original(model, *args, **kwargs)
        with patch.object(KernGDWD, 'fit', watch):
            cv = KernGDWDCV(kernel='rbf', lambd_vals=[.1, .2], q_vals=[1.], cv=2,
                acceleration='restart', residual_check_order='adaptive',
                max_iter=4, stopping='fixed').fit(self.X, self.y)
        self.assertEqual(seen, [('restart', 'adaptive')] * 5)
        self.assertEqual(cv.best_estimator_.residual_check_order, 'adaptive')

    def test_ineligible_routes_still_fail_before_work(self):
        for options in ({'backend':'spectral'}, {'backend':'lbfgs'},
                        {'implementation':'reference'}, {'solver_mode':'legacy'}):
            with self.subTest(options=options), patch.object(KernGDWD,
                    '_compute_training_kernel', side_effect=AssertionError('premature work')):
                with self.assertRaises(ValueError):
                    KernGDWD(acceleration='restart', residual_check_order='adaptive',
                             **options).fit(self.X, self.y)
        values, vectors = np.linalg.eigh(self.K)
        with self.assertRaises(ValueError):
            solve_kernel(self.K, self.y, .1, K_eig=(vectors, values),
                         acceleration='restart', residual_check_order='adaptive')

    def test_healthy_default_and_adaptive_states_are_identical(self):
        args = dict(acceleration='restart', max_iter=25, stopping='fixed')
        default = solve_kernel(self.K, self.y, .1, **args)
        explicit = solve_kernel(self.K, self.y, .1, residual_check_order='refinement_first', **args)
        adaptive = solve_kernel(self.K, self.y, .1, residual_check_order='adaptive', **args)
        for result in (explicit, adaptive):
            assert_array_equal(default['alpha'], result['alpha'])
            assert_array_equal(default['objective_history'], result['objective_history'])
            self.assertEqual(default['offset'], result['offset'])
            self.assertEqual(default['diagnostics']['acceleration_proposals'],
                             result['diagnostics']['acceleration_proposals'])

    def test_real_compensated_checks_skip_only_discarded_trials(self):
        ordinary = KernelLinearSystem._ordinary_measure
        def inconclusive(system, *args, **kwargs):
            measured = ordinary(system, *args, **kwargs)
            return (False, *measured[1:])
        results = []
        with patch.object(KernelLinearSystem, '_ordinary_measure', inconclusive):
            for order in ('refinement_first', 'adaptive'):
                results.append(solve_kernel(self.K, self.y, .1, acceleration='restart',
                    residual_check_order=order, max_iter=25, stopping='fixed'))
        default, adaptive = results
        assert_array_equal(default['alpha'], adaptive['alpha'])
        assert_array_equal(default['objective_history'], adaptive['objective_history'])
        self.assertEqual(default['offset'], adaptive['offset'])
        old = default['diagnostics']['linear_system_diagnostics']
        new = adaptive['diagnostics']['linear_system_diagnostics']
        self.assertGreater(old['native_refinement_discarded'], 2)
        self.assertEqual(new['native_refinement_trials'], 2)
        self.assertEqual(new['adaptive_native_trials_skipped'], old['native_refinement_trials']-2)
        self.assertEqual(new['compensated_residual_checks'], old['compensated_residual_checks'])


# Replay the existing independent bordered-system, generalized-q epigraph,
# objective restart, numerical retry, callbacks and validation-restoration
# contracts with adaptive ordering enabled on eligible Cholesky calls only.
_path = Path(__file__).with_name('test_accelerated_mm.py')
_spec = importlib.util.spec_from_file_location('restart_contracts', _path)
_existing = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_existing)
_plain_solve = _existing.solve_kernel
def _adaptive_solve(*args, **kwargs):
    if (kwargs.get('implementation', 'optimized') == 'optimized'
            and kwargs.get('backend', 'auto') in ('auto', 'cholesky')
            and kwargs.get('K_eig') is None):
        kwargs['residual_check_order'] = 'adaptive'
    return _plain_solve(*args, **kwargs)
_existing.solve_kernel = _adaptive_solve
class AdaptiveRestartIndependentContracts(_existing.AccelerationMathTests):
    pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
