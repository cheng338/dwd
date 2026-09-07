"""Computed helper eigenpairs preserve order, full spectrum and reconstruction."""
import unittest
from unittest.mock import patch
import numpy as np
from numpy.testing import assert_allclose
from scipy import sparse
import dwd.gen_dwd as linear
import dwd.gen_kern_dwd as kernel
import dwd._eigen as eigen


class EigenHelperReleaseTests(unittest.TestCase):
    def test_kernel_helper_uses_evd_and_preserves_complete_eigenpairs(self):
        rng = np.random.default_rng(913)
        rotation, _ = np.linalg.qr(rng.normal(size=(8, 8)))
        for values in (np.geomspace(1e-10, 1., 8),
                       np.array([0., 0., 0., .2, .2, 1., 1., 1.]), np.zeros(8)):
            K = (rotation * values) @ rotation.T
            with patch.object(eigen, 'eigh', wraps=eigen.eigh) as call:
                U, D = kernel.get_K_eig(K)
            self.assertEqual(call.call_args.kwargs['driver'], 'evd')
            self.assertEqual(U.shape, (8, 8))
            self.assertEqual(D.shape, (8,))
            self.assertTrue(U.flags.f_contiguous)
            self.assertTrue(np.all(np.diff(D) <= 0))
            assert_allclose(D, values[::-1], atol=3e-14, rtol=3e-14)
            assert_allclose((U * D) @ U.T, K, atol=3e-14, rtol=3e-14)
            assert_allclose(U.T @ U, np.eye(8), atol=3e-14)

    def test_linear_helper_preserves_augmented_free_intercept_matrix(self):
        rng = np.random.default_rng(8172)
        X = rng.normal(size=(15, 4))
        X[:, 3] = X[:, 0]
        augmented = np.column_stack([np.ones(len(X)), X])
        P0 = augmented.T @ augmented
        for data in (X, sparse.csr_matrix(X)):
            with patch.object(eigen, 'eigh', wraps=eigen.eigh) as call:
                U, D = linear.get_P0_eig(data)
            self.assertEqual(call.call_args.kwargs['driver'], 'evd')
            self.assertEqual(U.shape, (5, 5))
            self.assertEqual(D.shape, (5,))
            self.assertTrue(U.flags.f_contiguous)
            self.assertTrue(np.all(np.diff(D) <= 0))
            assert_allclose((U * D) @ U.T, P0, atol=2e-13, rtol=2e-14)
            assert_allclose(U.T @ U, np.eye(5), atol=2e-14)
            self.assertAlmostEqual(float(np.sum(D)), float(np.trace(P0)), places=12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
