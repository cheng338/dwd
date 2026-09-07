"""Bounded extension is conditional on original-equation improvement."""
import math
import unittest
from collections import Counter
from unittest.mock import patch

import numpy as np
from scipy.linalg import eigh
from sklearn.metrics.pairwise import rbf_kernel
from threadpoolctl import threadpool_limits

from dwd._spectral_linear_system import SpectralLinearSystem
import dwd._spectral_linear_system as spectral
from test_toy_residual_regression import DecimalOracle

RECOVERY_MODES = ('power_of_two_equilibrated_eigenbasis', 'original_kernel_evd_eigenbasis',
                  'original_kernel_evr_eigenbasis', 'original_kernel_evx_eigenbasis')


class SpectralAdaptiveRefinementTests(unittest.TestCase):
    def test_attainable_nearly_constant_cases_pass_original_decimal80_equations(self):
        with threadpool_limits(2):
            for n in (30, 120):
                with self.subTest(n=n):
                    rng = np.random.RandomState(91300+n)
                    rng.randint(-3, 4, size=(n, 5))
                    K = rbf_kernel(rng.normal(size=(n, 4)), gamma=1e-5)
                    rhs = rng.normal(size=n)+.23
                    shift, target = 1e-10, .37
                    D, U = eigh(K, driver='evd')
                    system = SpectralLinearSystem(K, shift, U, np.maximum(D, 0.))
                    x, s = system.solve_constrained(rhs, target)
                    residual, constraint = DecimalOracle(K, shift).residual(rhs, x, s, target)
                    self.assertLessEqual(np.max(abs(residual)), 1e-10*max(1., np.max(abs(rhs))))
                    self.assertLessEqual(abs(constraint), 64*np.finfo(float).eps*max(1., math.fsum(abs(x)), abs(target)))
                    phases = 1 + system.info['linear_recoveries']
                    self.assertLessEqual(phases, 5)
                    self.assertLessEqual(system.info['refinement_steps'], 8 * phases)
                    self.assertLessEqual(system.info.get('refinement_budget_extensions', 0), phases)
                    if system.info['refinement_steps'] > 3 * phases:
                        self.assertGreater(system.info['refinement_budget_extensions'], 0)
                        self.assertGreater(system.info['extended_refinement_steps'], 0)
                    self.assertEqual(system.info['added_objective_regularization'], 0)
                    self.assertEqual(system.info['positive_eigenvalues_discarded'], 0)

    def test_noncontracting_wrong_correction_retains_three_step_bound(self):
        for include_recovery in (False, True):
            with self.subTest(recovery=include_recovery):
                system = SpectralLinearSystem(np.eye(4), .1, np.eye(4), np.ones(4))
                rhs = np.array([1., -1., 1., -1.])
                modes = []
                def wrong_action(r, target):
                    modes.append(system.info['factor_representation'])
                    return np.zeros(4), 0.
                with patch.object(system, '_candidate', side_effect=wrong_action), \
                        patch.object(spectral, 'validated_eigh', wraps=spectral.validated_eigh) as prepare:
                    with self.assertRaisesRegex(FloatingPointError, 'failed residual accuracy'):
                        if include_recovery:
                            system.solve_constrained(rhs)
                        else:
                            system._solve_constrained(rhs, 0.)
                # The direct inner control retains the original exact bound;
                # the outer control permits the fixed four additional phases.
                attempts = 1 + 4 * int(include_recovery)
                expected = {'validated_eigenbasis': 5}
                if include_recovery:
                    expected.update({mode: 5 for mode in RECOVERY_MODES})
                self.assertEqual(Counter(modes), expected)
                self.assertEqual(prepare.call_count, 4 * int(include_recovery))
                self.assertEqual(system.info['native_refinement_discarded'], attempts)
                self.assertEqual(system.info['refinement_steps'], 3 * attempts)
                self.assertEqual(system.info['linear_solves'], int(include_recovery))
                self.assertNotIn('refinement_budget_extensions', system.info)
                self.assertIsNone(system.last_product)

    def test_contraction_allows_extension_then_representation_stagnation_stops(self):
        for include_recovery in (False, True):
            with self.subTest(recovery=include_recovery):
                system = SpectralLinearSystem(np.eye(4), .1, np.eye(4), np.ones(4))
                rhs = np.array([1., -1., 1., -1.])
                calls = []
                def limited(r, target):
                    calls.append(system.info['factor_representation'])
                    # The second call is the discarded preflight; one committed
                    # correction contracts, then the original phase stagnates.
                    return (.9*rhs/1.1, 0.) if len(calls) == 3 else (np.zeros(4), 0.)
                with patch.object(system, '_candidate', side_effect=limited), \
                        patch.object(spectral, 'validated_eigh', wraps=spectral.validated_eigh) as prepare:
                    message = 'failed residual accuracy' if include_recovery else 'stagnated'
                    with self.assertRaisesRegex(FloatingPointError, message):
                        if include_recovery:
                            system.solve_constrained(rhs)
                        else:
                            system._solve_constrained(rhs, 0.)
                expected = {'validated_eigenbasis': 6}
                if include_recovery:
                    expected.update({mode: 5 for mode in RECOVERY_MODES})
                    self.assertIn('stagnated', system.info['spectral_recovery_attempt']['trigger'])
                self.assertEqual(Counter(calls), expected)
                self.assertEqual(prepare.call_count, 4 * int(include_recovery))
                self.assertEqual(system.info['native_refinement_discarded'], 1 + 4 * int(include_recovery))
                self.assertEqual(system.info['refinement_steps'], 4 + 12 * int(include_recovery))
                self.assertEqual(system.info['refinement_budget_extensions'], 1)
                self.assertEqual(system.info['extended_refinement_steps'], 1)
                self.assertEqual(system.info['extended_refinement_stagnation_count'], 1)
                self.assertIsNone(system.last_product)

    def test_contracting_but_persistent_error_never_exceeds_eight_corrections(self):
        for use_candidate, include_recovery in ((False, False), (False, True), (True, False), (True, True)):
            with self.subTest(candidate=use_candidate, recovery=include_recovery):
                system = SpectralLinearSystem(np.eye(4), .1, np.eye(4), np.ones(4))
                rhs = np.array([1., -1., 1., -1.])
                # Slow continued contraction, never near the accuracy target.
                original = system._candidate
                calls = []
                def slow(r, target):
                    calls.append(system.info['factor_representation'])
                    x, s = original(r, target)
                    return .5*x, .5*s
                with patch.object(system, '_candidate', side_effect=slow), \
                        patch.object(spectral, 'validated_eigh', wraps=spectral.validated_eigh) as prepare:
                    with self.assertRaisesRegex(FloatingPointError, 'failed.*accuracy'):
                        if use_candidate:
                            method = system.refine_candidate if include_recovery else system._refine_candidate
                            method(rhs, np.zeros(4), 0., np.zeros(4))
                        else:
                            method = system.solve_constrained if include_recovery else system._solve_constrained
                            method(rhs, 0.)
                per_phase = 9 if use_candidate else 10  # Preflight + 8 corrections (+ initial solve).
                expected = {'validated_eigenbasis': per_phase}
                if include_recovery:
                    expected.update({mode: per_phase for mode in RECOVERY_MODES})
                attempts = 1 + 4 * int(include_recovery)
                self.assertEqual(Counter(calls), expected)
                self.assertEqual(prepare.call_count, 4 * int(include_recovery))
                self.assertEqual(system.info['refinement_steps'], 8 * attempts)
                self.assertEqual(system.info['refinement_budget_extensions'], attempts)
                self.assertEqual(system.info['extended_refinement_steps'], 5 * attempts)
                self.assertEqual(system.info['native_refinement_trials'], attempts)
                self.assertEqual(system.info['native_refinement_discarded'], attempts)
                self.assertEqual(system.info['linear_solves'], int(include_recovery and not use_candidate))
                self.assertIsNone(system.last_product)


if __name__ == '__main__':
    unittest.main(verbosity=2)
