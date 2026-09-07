"""Finite inverse-basis recovery, independently checked in original units."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import weakref

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

import dwd._spectral_linear_system as spectral
import dwd._kernel_linear_system as linear
import dwd._kernel_solver as kernel
from dwd._eigen import validated_eigh
from dwd._exact_kernel_factor import FactorResult
from test_toy_residual_regression import DecimalOracle

MODES = ['validated_eigenbasis', 'power_of_two_equilibrated_eigenbasis',
         'original_kernel_evd_eigenbasis', 'original_kernel_evr_eigenbasis',
         'original_kernel_evx_eigenbasis']


class NativeSpectralRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limits = threadpool_limits(2)

    @classmethod
    def tearDownClass(cls):
        cls.limits.restore_original_limits()

    def benign(self):
        K = np.diag([.1, .3, .8, 1.2])
        return spectral.SpectralLinearSystem(K, .2, np.eye(4), K.diagonal())

    def test_fixed_two_thread_basis_can_recover_under_two_or_four_solve_threads(self):
        directory = Path(__file__).parent / 'fixtures'
        metadata = json.loads((directory / 'spectral_n120_cached_evd.json').read_text())
        file = directory / 'spectral_n120_cached_evd.npz'
        self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(), metadata['fixture_sha256'])
        with np.load(file) as saved:
            K, rhs, values, vectors = [saved[name].copy() for name in ('K', 'rhs', 'values', 'vectors')]
        for name, array in [('K', K), ('rhs', rhs), ('values', values), ('vectors', vectors)]:
            self.assertEqual(hashlib.sha256(array.tobytes()).hexdigest(), metadata[name + '_sha256'])
        self.assertEqual(metadata['preparation_threads'], 2)
        for threads in (2, 4):
            with self.subTest(arithmetic_threads=threads), threadpool_limits(threads):
                validated_eigh(K, supplied=(vectors, values))
                system = spectral.SpectralLinearSystem(K, 1e-10, vectors, np.maximum(values, 0.))
                with patch.object(linear, 'cho_factor', side_effect=AssertionError('Reference changed backend')):
                    x, s = system.solve_constrained(rhs, .37)
                r, c = DecimalOracle(K, 1e-10).residual(rhs, x, s, .37)
                self.assertLessEqual(max(abs(r)), 1e-10 * max(1., max(abs(rhs))))
                self.assertLessEqual(abs(c), 64 * np.finfo(float).eps * max(1., sum(abs(x)), .37))
                self.assertEqual(system.info['linear_solves'], 1)
                attempts = system.info.get('spectral_recovery_attempts', [])
                self.assertEqual([a['kind'] for a in attempts], ['equilibrated', 'evd', 'evr', 'evx'][:len(attempts)])
                self.assertLessEqual(len(attempts), 4)
                if attempts:
                    self.assertTrue(attempts[-1]['accepted'])
                self.assertEqual(system.info['added_objective_regularization'], 0.)
                self.assertEqual(system.info['positive_eigenvalues_discarded'], 0)

    def test_preparation_success_does_not_authorize_an_inaccurate_inverse(self):
        system = self.benign()
        rhs, target = np.array([1., -1., .2, .5]), .37
        original = system._solve_constrained
        modes, prepares = [], []
        def selective(r, c):
            mode = system.info['factor_representation']
            modes.append(mode)
            if mode != MODES[-1]:
                raise FloatingPointError('Manufactured inaccurate action in ' + mode)
            return original(r, c)
        def prepared(A, **kwargs):
            prepares.append(kwargs.get('drivers'))
            return validated_eigh(A, **kwargs)
        with patch.object(system, '_solve_constrained', side_effect=selective), \
                patch.object(spectral, 'validated_eigh', side_effect=prepared):
            x, s = system.solve_constrained(rhs, target)
            again_x, again_s = system.solve_constrained(rhs, target)
        self.assertEqual(modes, MODES + [MODES[-1]])
        self.assertEqual(prepares, [None, ('evd',), ('evr',), ('evx',)])
        self.assertEqual(system.info['linear_solves'], 2)
        attempts = system.info['spectral_recovery_attempts']
        self.assertTrue(all(a['preparation_succeeded'] for a in attempts))
        self.assertEqual([a['accepted'] for a in attempts], [False, False, False, True])
        self.assertEqual(attempts[-1]['accurate_actions'], 2)
        self.assertTrue(all('solve_error' in a for a in attempts[:-1]))
        matrix = np.block([[system.K + system.shift * np.eye(4), np.ones((4, 1))],
                           [np.ones((1, 4)), np.zeros((1, 1))]])
        assert_allclose(np.r_[x, s], solve(matrix, np.r_[rhs, target]), rtol=1e-12, atol=1e-12)
        assert_array_equal(again_x, x)
        self.assertEqual(again_s, s)

    def test_failed_preparations_continue_once_and_exhausted_plan_never_restarts(self):
        system = self.benign()
        with patch.object(system, '_solve_constrained', side_effect=FloatingPointError('original action error')), \
                patch.object(spectral, 'validated_eigh', side_effect=FloatingPointError('bad provider')) as prepare:
            for _ in range(2):
                with self.assertRaisesRegex(FloatingPointError, 'at most five representations'):
                    system.solve_constrained(np.ones(4))
        self.assertEqual(prepare.call_count, 4)
        self.assertEqual(system._recovery_next, 4)
        self.assertEqual(system.info['linear_solves'], 2)
        self.assertEqual(len(system.info['spectral_inverse_failures']), 1)
        self.assertEqual([a['kind'] for a in system.info['spectral_recovery_attempts']],
                         ['equilibrated', 'evd', 'evr', 'evx'])
        self.assertTrue(all(not a['preparation_succeeded'] and not a['accepted']
                            for a in system.info['spectral_recovery_attempts']))
        self.assertIsNone(system.vectors)

    def test_only_psd_roundoff_negative_inverse_weights_are_clipped(self):
        system = self.benign()
        system.K = np.diag([-np.finfo(float).eps, 1., 2., 3.])
        prepared = system._build_original('evx')
        vectors, inverse, scale, _, _, info = prepared
        self.assertIsNone(scale)
        self.assertTrue(info['inverse_preconditioner_only'])
        self.assertLess(info['minimum_eigenvalue'], 0.)
        self.assertEqual(info['negative_roundoff_eigenvalues_clipped'], 1)
        self.assertEqual(info['positive_eigenvalues_retained'], 3)
        limit = kernel._check_spectrum(system.K.diagonal(), 4, system.shift)
        self.assertEqual(info['psd_tolerance'], limit)
        assert_array_equal(inverse, 1. / (np.array([0., 1., 2., 3.]) + .2))
        assert_array_equal(system.K.diagonal(), [-np.finfo(float).eps, 1., 2., 3.])
        rhs = np.array([1., 2., 3., 4.])
        assert_allclose(vectors @ (inverse * (vectors.T @ rhs)),
                        solve(system.K + system.shift * np.eye(4), rhs), rtol=1e-14)
        # Even a positive shifted denominator cannot excuse materially
        # indefinite K. Conversely the public roundoff threshold is shared.
        for negative in (-2 * limit, -.1, -.3):
            system.K = np.diag([negative, 1., 2., 3.])
            with self.assertRaisesRegex(ValueError, 'positive semidefinite'):
                kernel._check_spectrum(system.K.diagonal(), 4, system.shift)
            with self.assertRaisesRegex(FloatingPointError, 'PSD tolerance'):
                system._build_original('evx')

    def test_nonfinite_and_unrepresentable_fresh_weights_fail_closed(self):
        for values, shift in ((np.array([np.nan, 1., 2., 3.]), .2),
                              (np.array([np.inf, 1., 2., 3.]), .2),
                              (np.ones(3), .2),
                              (np.ones(4) * 1e308, 1e308),
                              (np.zeros(4), np.finfo(float).smallest_subnormal),
                              (np.zeros(4), 0.)):
            with self.subTest(values=values, shift=shift):
                system = self.benign()
                system.shift = shift
                with patch.object(spectral, 'validated_eigh', return_value=((values, np.eye(4)), {})):
                    with self.assertRaises(FloatingPointError):
                        system._build_original('evd')

    def test_invalid_fresh_computed_basis_is_rejected_before_any_inverse_action(self):
        system = self.benign()
        with patch('dwd._eigen.eigh', return_value=(np.ones(4), np.zeros((4, 4)))):
            with self.assertRaises(FloatingPointError):
                system._build_original('evx')

    def test_failed_recovery_bases_are_released_before_next_allocation(self):
        system = self.benign()
        refs = []
        original_equilibrated, original_native = system._build_equilibrated, system._build_original
        def capture(build, *args):
            self.assertTrue(all(r() is None for r in refs))
            prepared = build(*args)
            refs.append(weakref.ref(prepared[0]))
            return prepared
        with patch.object(system, '_build_equilibrated', side_effect=lambda: capture(original_equilibrated)), \
                patch.object(system, '_build_original', side_effect=lambda d: capture(original_native, d)), \
                patch.object(system, '_solve_constrained', side_effect=FloatingPointError('force next representation')):
            with self.assertRaises(FloatingPointError):
                system.solve_constrained(np.arange(4.))
        self.assertEqual(len(refs), 4)
        self.assertTrue(all(r() is None for r in refs))
        self.assertIsNone(system.vectors)

    def test_exhausted_all_recoveries_does_not_repeat_mm_iterations_or_callbacks(self):
        K, y = np.eye(4), np.array([-1., 1., -1., 1.])
        system = spectral.SpectralLinearSystem(K, .2, np.eye(4), np.ones(4))
        modes, seen = [], []
        def zero(r, target):
            modes.append(system.info['factor_representation'])
            return np.zeros(4), 0.
        unavailable = FactorResult('injected_certificate_unavailable', None, {})
        with patch.object(kernel, 'SpectralLinearSystem', return_value=system), \
                patch.object(system, '_candidate', side_effect=zero), \
                patch('dwd._exact_kernel_factor.exact_kernel_factor', return_value=unavailable):
            with self.assertRaisesRegex(FloatingPointError, 'at most five representations'):
                kernel.solve_kernel(K, y, .1, implementation='reference',
                                    max_iter=3, stopping='fixed',
                                    callback=lambda state: seen.append(state['iteration']))
        self.assertEqual(seen, [0])
        self.assertEqual(Counter(modes), {mode: 5 for mode in MODES})
        self.assertEqual(system.info['linear_solves'], 1)
        self.assertEqual(system.info['refinement_steps'], 15)
        self.assertEqual(system.info['native_refinement_discarded'], 5)
        self.assertIsNone(system.last_product)


if __name__ == '__main__':
    unittest.main()
