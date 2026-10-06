"""Represented-input oracles for joint affine cancellation and recovery."""

from fractions import Fraction
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix

from dwd import _affine_scores as module


def oracle(K, alpha, b):
    answers = []
    for index in range(K.shape[0]):
        if hasattr(K, 'indptr'):
            start, stop = K.indptr[index:index + 2]
            terms = zip(K.data[start:stop], alpha[K.indices[start:stop]])
        else:
            terms = zip(K[index], alpha)
        answers.append(float(sum((Fraction(float(k)) * Fraction(float(a)) for k, a in terms),
                                 Fraction(float(b)))))
    return np.asarray(answers)


class AffineScoreTests(unittest.TestCase):
    def test_exact_oracle_for_cancellation_range_and_subnormal_cases(self):
        A, eps, eta, maximum = 2.**54, np.finfo(float).eps, np.finfo(float).smallest_subnormal, np.finfo(float).max
        cases = [
            ([1., 1., 0., 0.], [A, 1., -A, -1.], -A),
            ([0., 0., 1., 1.], [A, 1., -A, -1.], A),
            ([1., 0.], [A, -A], -A),
            ([1., 1., 1., 0.], [A, 1., -A, -1.], -.5),
            ([1., 1., 1., 0.], [-A, -1., A, 1.], .5),
            ([1.+eps, 0.], [1.+eps, -(1.+eps)], -(1.+2*eps)),
            ([1.+eps, 0.], [1.-eps, -(1.-eps)], -1.),
            ([eta, eta, 0.], [.5, .5, -1.], 0.),
            ([eta, eta, 0.], [-.5, -.5, 1.], 0.),
            ([eta, 0.], [.5, -.5], 0.),
            ([eta, 0.], [-.5, .5], 0.),
            ([eta, 0.], [1., -1.], 0.),
            ([eta, 0.], [-1., 1.], 0.),
            ([-0., 0.], [1., -1.], -0.),
            ([1., 1., 0., 0.], [maximum, maximum, -maximum, -maximum], -maximum),
            ([2., 0.], [1e308, -1e308], -1e308),
            ([2., 2.], [maximum, -maximum], 1.),
            ([2., 2.], [maximum, -maximum], 0.),
            ([.25, .5, .75, 1.], [1., -1., 2., -2.], .125),
        ]
        for index, (row, alpha, b) in enumerate(cases):
            dense, alpha = np.asarray([row] * 3), np.asarray(alpha)
            for K in (dense, csr_matrix(dense)):
                expected = oracle(K, alpha, b)
                for policy in ('adaptive', 'compensated'):
                    with self.subTest(case=index, sparse=hasattr(K, 'indptr'), policy=policy):
                        observed, counts = module.affine_scores(K, alpha, b, policy=policy, return_info=True)
                        np.testing.assert_array_equal(observed, expected)
                        self.assertEqual(counts['ordinary_rows'] + counts['joint_rows'], len(K.indptr)-1
                                         if hasattr(K, 'indptr') else len(K))
                        self.assertLessEqual(counts['exact_rows'], counts['joint_rows'])

    def test_stored_csr_duplicates_are_not_prerounded(self):
        K = csr_matrix((np.array([2.**54, 1., -2.**54, 0.]),
                        np.array([0, 0, 0, 1]), np.array([0, 4])), shape=(1, 2))
        alpha, b = np.array([1., -1.]), -.5
        before = (K.data.copy(), K.indices.copy(), K.indptr.copy())
        for policy in ('adaptive', 'compensated'):
            np.testing.assert_array_equal(module.affine_scores(K, alpha, b, policy=policy), [.5])
        for current, original in zip((K.data, K.indices, K.indptr), before):
            np.testing.assert_array_equal(current, original)

    def test_refinement_keeps_affine_magnitude_not_only_sign(self):
        # A sign-only screen would accept a million-unit absolute error here.
        from dwd._kernel_scores import adaptive_kernel_matvec
        K = np.array([[1., 1., 1., 1., 0., 0.]])
        A, L, t, R = 2.**54, 2.**80, 2.**20, 2.**34
        alpha = np.array([L, t, A + R, -L, -t, -(A + R)])
        b = -A
        expected = oracle(K, alpha, b)
        self.assertNotEqual((adaptive_kernel_matvec(K, alpha) + b)[0], expected[0])
        np.testing.assert_array_equal(module.affine_scores(K, alpha, b), expected)

    def test_final_unrepresentable_result_raises(self):
        maximum = np.finfo(float).max
        for policy in ('adaptive', 'compensated'):
            with self.assertRaises(FloatingPointError):
                module.affine_scores(np.ones((1, 2)), np.array([maximum, maximum]), 0., policy=policy)

    def test_tile_failure_retries_rows_and_unrecoverable_memory_propagates(self):
        K = np.array([[1., 1., 0.], [1., -1., 0.], [1., 0., 1.], [1., .5, .5]])
        alpha, b = np.array([2.**54, 1., -1.]), -2.**54
        expected = oracle(K, alpha, b)
        original = module._products
        for failure in (MemoryError, FloatingPointError):
            calls = {'tile_failures': 0, 'rows': 0}

            def products(a, c, **kwargs):
                if a.ndim == 2 and calls['tile_failures'] == 0:
                    calls['tile_failures'] += 1
                    raise failure('Simulated tile failure')
                calls['rows'] += int(a.ndim == 1)
                return original(a, c, **kwargs)

            with patch.object(module, '_products', side_effect=products):
                observed = module.affine_scores(K, alpha, b, policy='compensated')
            np.testing.assert_array_equal(observed, expected)
            self.assertEqual(calls['tile_failures'], 1)
            self.assertGreaterEqual(calls['rows'], len(K))
        for target in ('_products', '_exact_row'):
            with patch.object(module, target, side_effect=MemoryError('Simulated irrecoverable allocation')):
                with self.assertRaises(MemoryError):
                    module.affine_scores(np.array([[1.]]), np.array([1.]), -1., policy='compensated')

    def test_invalid_input_and_strict_warning_modes(self):
        cases = [
            (np.ones((1, 2), complex), np.ones(2), 0., ValueError),
            (np.ones((1, 2)), np.ones(2, complex), 0., ValueError),
            (np.ones((1, 2)), np.ones(2), 1j, ValueError),
            (np.ones((1, 2)), np.ones(2), [0.], ValueError),
            (np.ones((1, 2)), np.ones(2), np.nan, FloatingPointError),
            (np.ones((1, 2)), np.array([np.inf, 0.]), 0., FloatingPointError),
            (np.array([[np.nan, 0.]]), np.ones(2), 0., FloatingPointError),
            (np.ones((1, 3)), np.ones(2), 0., ValueError),
        ]
        for policy in ('adaptive', 'compensated'):
            for K, alpha, b, error in cases:
                with self.subTest(policy=policy, error=error), self.assertRaises(error):
                    module.affine_scores(K, alpha, b, policy=policy)
            with np.errstate(all='raise'):
                np.testing.assert_array_equal(module.affine_scores(np.ones((1, 2)), np.ones(2), 0.,
                                                                  policy=policy), [2.])
            for K in (np.empty((0, 2)), csr_matrix((0, 2))):
                self.assertEqual(module.affine_scores(K, np.ones(2), 0., policy=policy).shape, (0,))


if __name__ == '__main__':
    unittest.main()
