"""Numerical and input-ownership contracts for direct RBF construction.

The direct float64 reduction is not a universally correctly rounded distance.
These small represented-input oracles cover cancellation and range boundaries;
positive tails are checked relatively rather than hidden by an absolute floor.
"""

from decimal import Decimal, localcontext
from fractions import Fraction
import math
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import make_classification

from dwd._direct_rbf import accurate_rbf


def represented_rbf(X, Y, gamma):
    """Independent exact squared differences, then high-precision exponential."""
    result = np.empty((len(X), len(Y)))
    gamma = Fraction(float(gamma))
    with localcontext() as context:
        context.prec = 100
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                squared = sum(((Fraction(float(a)) - Fraction(float(b))) ** 2
                               for a, b in zip(x, y)), Fraction(0))
                exponent = gamma * squared
                value = Decimal(exponent.numerator) / Decimal(exponent.denominator)
                result[i, j] = 0. if value > 1000 else float((-value).exp())
    return result


class DirectRBFTests(unittest.TestCase):
    def setUp(self):
        offsets = np.array([[0., 0., 0.], [.125, .25, -.5], [1., -1., 2.],
                            [0., 0., 0.], [-2., 4., -8.]])
        self.X = 2. ** 40 + offsets

    def test_large_common_offset_agrees_with_represented_input_oracle(self):
        expected = represented_rbf(self.X, self.X, .125)
        actual = accurate_rbf(self.X, gamma=.125)
        np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=0)
        # These dyadic translations preserve the represented differences.
        centered = self.X - 2. ** 40
        np.testing.assert_array_equal(actual, accurate_rbf(centered, gamma=.125))

    def test_offdiagonal_duplicate_and_small_positive_tail(self):
        X, _ = make_classification(n_samples=36, n_features=5, n_informative=2,
                                  n_redundant=0, random_state=118, class_sep=.5)
        X[:, 2] = X[:, 0] + 1e-12 * X[:, 1]
        X[:, 3] = 1e5 * X[:, 0]
        X[1::4] = X[::4][:len(X[1::4])]
        # Include the identical off-diagonal pair and a tiny nonzero kernel.
        X = X[[20, 21, 11, 24]]
        np.testing.assert_array_equal(X[0], X[1])
        actual = accurate_rbf(X, gamma=.12)
        self.assertEqual(actual[0, 1], 1.)
        self.assertEqual(actual[1, 0], 1.)
        expected = represented_rbf(X, X, .12)
        self.assertGreater(expected[2, 3], 0.)
        self.assertLess(expected[2, 3], 1e-100)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)
        np.testing.assert_array_equal(actual, accurate_rbf(X, X.copy(), .12))

    def test_self_copy_order_subset_batch_and_layout_invariance(self):
        storage = np.empty((len(self.X), 2 * self.X.shape[1]))
        storage[:, ::2] = self.X
        order = [2, 0, 3, 1, 4]
        subset = [3, 0, 3]
        for X in (self.X, np.asfortranarray(self.X), storage[:, ::2], csr_matrix(self.X)):
            with self.subTest(layout=type(X).__name__):
                expected = accurate_rbf(X, gamma=.3, block_size=1)
                np.testing.assert_array_equal(expected, expected.T)
                np.testing.assert_array_equal(np.diag(expected), np.ones(len(self.X)))
                np.testing.assert_array_equal(expected, accurate_rbf(X, X.copy(), .3))
                for block in (1, 2, 3, 7):
                    np.testing.assert_array_equal(expected, accurate_rbf(X, gamma=.3, block_size=block))
                np.testing.assert_array_equal(expected[:, order], accurate_rbf(X, X[order], .3))
                np.testing.assert_array_equal(expected[order], accurate_rbf(X[order], X, .3))
                np.testing.assert_array_equal(expected[:, subset], accurate_rbf(X, X[subset], .3))
                np.testing.assert_array_equal(expected, np.column_stack([
                    accurate_rbf(X, X[start:start + 2], .3) for start in range(0, len(self.X), 2)]))

    def test_dense_csr_and_mixed_representations(self):
        S = csr_matrix(self.X)
        expected = accurate_rbf(self.X, gamma=.3)
        for X, Y in ((S, S), (S, self.X), (self.X, S)):
            np.testing.assert_array_equal(expected, accurate_rbf(X, Y, .3))

    def test_exceptional_distance_and_gamma_ranges(self):
        cases = [
            (np.array([[0.], [1e160], [1e160]]), 1e-320),
            (np.array([[0.], [1e160], [1e160]]), np.nextafter(0., 1.)),
            (np.array([[0.], [1e-160], [1e-160]]), 1e308),
            (np.array([[0.], [1e-162], [1.6e-162]]), np.finfo(float).max),
            (np.array([[1e308], [-1e308], [1e308]]), np.nextafter(0., 1.)),
            (np.array([[0.], [1.], [1.]]), np.finfo(float).max),
        ]
        for X, gamma in cases:
            with self.subTest(gamma=gamma, largest=float(np.max(np.abs(X)))):
                expected = represented_rbf(X, X, gamma)
                for features in (X, csr_matrix(X)):
                    actual = accurate_rbf(features, gamma=gamma)
                    self.assertTrue(np.isfinite(actual).all())
                    for value, reference in zip(actual.flat, expected.flat):
                        self.assertLessEqual(abs(value - reference), 2 * math.ulp(float(reference)))

    def test_exponential_underflow_boundary(self):
        X = np.array([[0.], [np.sqrt(744.)], [np.sqrt(745.)], [np.sqrt(746.)]])
        expected = represented_rbf(X, X, 1.)
        actual = accurate_rbf(X, gamma=1.)
        self.assertTrue(np.any((expected > 0) & (expected < np.finfo(float).tiny)))
        self.assertEqual(expected[0, 3], 0.)
        for value, reference in zip(actual.flat, expected.flat):
            self.assertLessEqual(abs(value - reference), 2 * math.ulp(float(reference)))

    def test_gamma_zero_and_default(self):
        huge = np.array([[1e308], [-1e308]])
        np.testing.assert_array_equal(accurate_rbf(huge, gamma=0), np.ones((2, 2)))
        np.testing.assert_array_equal(accurate_rbf(self.X), accurate_rbf(self.X, gamma=1. / 3))

    def test_noncanonical_csr_is_not_mutated_or_densified(self):
        # Row zero has an unsorted duplicate column and an explicit zero.
        S = csr_matrix((np.array([.5, 0., .5, 2., -1.]),
                        np.array([1, 0, 1, 2, 0]), np.array([0, 4, 5])), shape=(2, 3))
        expected = accurate_rbf(np.array([[0., 1., 2.], [-1., 0., 0.]]))
        before = [value.copy() for value in (S.data, S.indices, S.indptr)]
        with patch.object(csr_matrix, 'toarray', side_effect=AssertionError('CSR was densified')):
            actual = accurate_rbf(S)
        np.testing.assert_array_equal(actual, expected)
        for original, current in zip(before, (S.data, S.indices, S.indptr)):
            np.testing.assert_array_equal(original, current)

    def test_dense_read_only_inputs_and_empty_queries(self):
        before = self.X.copy()
        self.X.flags.writeable = False
        accurate_rbf(self.X, gamma=.3)
        np.testing.assert_array_equal(self.X, before)
        self.assertEqual(accurate_rbf(self.X, np.empty((0, 3))).shape, (5, 0))

    def test_rejects_invalid_inputs(self):
        for gamma in (-1, np.nan, np.inf, True, 1j, 'x'):
            with self.subTest(gamma=gamma), self.assertRaises(ValueError):
                accurate_rbf(self.X, gamma=gamma)
        for block in (0, -1, True, 1.5):
            with self.subTest(block=block), self.assertRaises(ValueError):
                accurate_rbf(self.X, block_size=block)
        with self.assertRaises(ValueError):
            accurate_rbf(self.X, np.empty((2, 0), dtype=float))
        with self.assertRaises(ValueError):
            accurate_rbf(self.X, np.array([[np.inf] * 3]))
        with self.assertRaises(TypeError):
            accurate_rbf(self.X.astype(np.float32))
        with self.assertRaises(TypeError):
            accurate_rbf(csr_matrix(self.X).tocsc())


if __name__ == '__main__':
    unittest.main()
