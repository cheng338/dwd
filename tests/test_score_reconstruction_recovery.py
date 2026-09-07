"""Deterministic evaluation faults exercise score/J recovery and state policy.

These injected 2e-7 perturbations are control-flow regressions, not claims that
the identity kernel naturally suffers that floating-point error pattern.
"""
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal
from sklearn.exceptions import NotFittedError
from threadpoolctl import threadpool_limits

import dwd._kernel_solver as solver
from dwd.gen_kern_dwd import KernGDWD


class ScoreReconstructionRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limit = threadpool_limits(limits=2)

    @classmethod
    def tearDownClass(cls):
        cls.limit.restore_original_limits()

    def setUp(self):
        self.K = np.eye(4)
        self.y = np.array([-1., -1., 1., 1.])

    def model(self, **kwargs):
        options = dict(kernel='precomputed', implementation='optimized',
                       initialization='zero', lambd=.1, max_iter=0, stopping='fixed')
        options.update(kwargs)
        return KernGDWD(**options)

    def test_objective_only_fault_retries_once_without_changing_optimizer_state(self):
        original = solver.adaptive_kernel_matvec
        perturbation = 2e-7*self.y
        # At alpha=0 the injected score error passes 5e-7, while the artificial
        # q=1 loss/objective error is 2e-7 and must fail the separate 5e-8 gate.
        self.assertLess(np.max(np.abs(perturbation)), 5e-7)
        self.assertGreater(abs(np.mean(1-self.y*perturbation)-1), 5e-8)
        for validation in (False, True):
            with self.subTest(validation=validation):
                observed = []
                calls = []
                def injected(K, alpha):
                    calls.append(np.array(alpha, copy=True))
                    return original(K, alpha)+perturbation
                model = self.model(stopping='validation' if validation else 'fixed',
                                   callback=lambda state: observed.append(dict(state)))
                with patch.object(solver, 'adaptive_kernel_matvec', side_effect=injected):
                    model.fit(self.K, self.y, **({'validation_data': (self.K, self.y)} if validation else {}))
                self.assertEqual(len(calls), 1)
                self.assertEqual(model.diagnostics_['objective_evaluation_retries'], 1)
                self.assertEqual(model.prediction_precision_, 'compensated')
                self.assertEqual(model.n_iter_, 0)
                self.assertEqual(model.final_objective_, 1.)
                assert_array_equal(model.dual_coef_, np.zeros((1, 4)))
                assert_array_equal(model.intercept_, [0.])
                assert_array_equal(model.objective_history_, [1.])
                assert_array_equal(model.decision_function(self.K), np.zeros(4))
                self.assertEqual(len(observed), 1)
                assert_array_equal(observed[0]['training_scores'], np.zeros(4))
                assert_array_equal(observed[0]['alpha'], np.zeros(4))
                self.assertEqual(observed[0]['objective'], 1.)
                if validation:
                    # Faulty signs would give accuracy 1.; corrected zero scores
                    # give .5, so unchecked values never influenced selection.
                    self.assertEqual(model.validation_history_[0]['score'], .5)
                    self.assertEqual(model.validation_history_[0]['prediction_precision'], 'compensated')

    def test_best_validation_state_restores_its_own_prediction_precision(self):
        baseline_states = []
        self.model(max_iter=3, callback=lambda state: baseline_states.append(dict(state))).fit(self.K, self.y)
        second_alpha = baseline_states[2]['alpha'].tobytes()
        first_alpha = baseline_states[1]['alpha']
        original = solver.adaptive_kernel_matvec
        fault_hits = []
        def injected(K, alpha):
            values = original(K, alpha)
            if np.asarray(alpha).tobytes() == second_alpha:
                fault_hits.append(np.array(alpha, copy=True))
                return values+2e-7*self.y
            return values
        observed = []
        model = self.model(max_iter=3, stopping='validation', patience=2,
                           callback=lambda state: observed.append(dict(state)))
        with patch.object(solver, 'adaptive_kernel_matvec', side_effect=injected):
            model.fit(self.K, self.y, validation_data=(self.K, np.array([-1., 1., -1., 1.])))
        self.assertEqual(len(fault_hits), 1)
        self.assertEqual(model.diagnostics_['objective_evaluation_retries'], 1)
        self.assertEqual(model.returned_iteration_, 1)
        self.assertEqual(model.n_iter_, 3)
        self.assertEqual(model.termination_reason_, 'validation_patience')
        self.assertEqual(model.prediction_precision_, 'adaptive')
        self.assertEqual(model.diagnostics_['prediction_precision'], 'adaptive')
        assert_array_equal(model.dual_coef_[0], first_alpha)
        self.assertEqual(model.intercept_[0], baseline_states[1]['offset'])
        self.assertEqual(model.final_objective_, baseline_states[1]['objective'])
        self.assertEqual([s['iteration'] for s in model.validation_history_], [1, 2, 3])
        self.assertEqual([s['score'] for s in model.validation_history_], [.5, .5, .5])
        self.assertEqual([s['prediction_precision'] for s in model.validation_history_],
                         ['adaptive', 'compensated', 'compensated'])
        # Retry checked evaluation only; no alpha/score/J optimizer state was
        # replaced by the injected, recomputed or best-state values mid-path.
        self.assertEqual(len(observed), len(baseline_states))
        for actual, expected in zip(observed, baseline_states):
            assert_array_equal(actual['alpha'], expected['alpha'])
            assert_array_equal(actual['training_scores'], expected['training_scores'])
            self.assertEqual(actual['offset'], expected['offset'])
            self.assertEqual(actual['objective'], expected['objective'])

    def test_joint_bad_evaluation_still_rejects_before_callback_or_selection(self):
        # Both evaluation paths are deliberately wrong. One violates only the
        # objective gate, the other the score gate. Neither can be waived.
        original = solver.adaptive_kernel_matvec
        for amount, message in ((2e-7, 'original RKHS objective accuracy'),
                                (2e-3, 'original-kernel score accuracy')):
            with self.subTest(amount=amount):
                observed = []
                def injected(K, alpha):
                    return original(K, alpha)+amount*self.y
                model = self.model(stopping='validation', callback=lambda state: observed.append(dict(state)))
                with patch.object(solver, 'adaptive_kernel_matvec', side_effect=injected), \
                     patch.object(solver, 'compensated_kernel_matvec', side_effect=injected) as compensated:
                    with self.assertRaisesRegex(FloatingPointError, message):
                        model.fit(self.K, self.y, validation_data=(self.K, self.y))
                self.assertEqual(compensated.call_count, 1)
                self.assertEqual(observed, [])
                self.assertFalse(hasattr(model, 'dual_coef_'))
                self.assertFalse(hasattr(model, 'prediction_precision_'))
                self.assertFalse(hasattr(model, 'validation_history_'))
                with self.assertRaises(NotFittedError):
                    model.decision_function(self.K)


if __name__ == '__main__':
    unittest.main(verbosity=2)
