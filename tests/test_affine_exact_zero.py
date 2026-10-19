"""Independent rational oracles for guarded affine exact-zero handling."""
import math
import pickle
import sys
import unittest
from fractions import Fraction
from unittest.mock import patch
import numpy as np
from scipy.sparse import csr_matrix
from dwd import _affine_scores as m
from dwd.gen_kern_dwd import KernGDWD

def oracle(row, a, b):
    return float(sum((Fraction(float(x)) * Fraction(float(y)) for x, y in zip(row, a)), Fraction(float(b))))

class CurrentZeroCertificate(unittest.TestCase):

    @unittest.skipUnless(sys.platform == "win32", "Windows floating-point control fixture")
    def test_rounding_modes_preserve_exact_fallback_and_caller_state(self):
        import ctypes
        control = ctypes.CDLL('ucrtbase')._controlfp_s
        control.argtypes = [ctypes.POINTER(ctypes.c_uint), ctypes.c_uint, ctypes.c_uint]
        control.restype = ctypes.c_int
        saved = ctypes.c_uint()
        self.assertEqual(control(ctypes.byref(saved), 0, 0), 0)
        mask = 768
        records = []
        try:
            for mode in (0, 256, 512, 768):
                observed = ctypes.c_uint()
                self.assertEqual(control(ctypes.byref(observed), mode, mask), 0)
                self.assertEqual(m._round_to_nearest_now(), mode == 0)
                K = np.array([[1.0, 1.0]])
                a = np.array([1.0, -1.0])
                score, info = m.affine_scores(K, a, 0.0, policy='compensated', return_info=True)
                self.assertEqual(score[0], 0.0)
                self.assertEqual(info['exact_rows'], int(mode != 0))
                after = ctypes.c_uint()
                self.assertEqual(control(ctypes.byref(after), 0, 0), 0)
                self.assertEqual(after.value & mask, mode)
                records.append({'mode': mode, 'exact_rows': info['exact_rows'], 'caller_mode_preserved': True})
        finally:
            restored = ctypes.c_uint()
            control(ctypes.byref(restored), saved.value, mask)

    def test_natural_fit_boundary(self):
        X = np.array([[0.0], [1.0]])
        y = np.array([-3, 7])
        Q = np.array([[np.nextafter(0.5, np.inf)]])
        fits = []
        for policy in ['standard', 'joint']:
            fit = KernGDWD(kernel='linear', lambd=0.1, q=1.0, implementation='optimized', max_iter=5, stopping='fixed', initialization='zero', random_state=7, affine_computation=policy).fit(X, y)
            fits.append(fit)
        for key in ['dual_coef_', 'intercept_', 'objective_history_']:
            np.testing.assert_array_equal(getattr(fits[0], key), getattr(fits[1], key))
        self.assertEqual(fits[0].predict(Q)[0], -3)
        self.assertEqual(fits[1].predict(Q)[0], 7)
        expected = oracle(fits[1]._compute_kernel(Q).T[0], fits[1].dual_coef_[0], fits[1].intercept_[0])
        self.assertEqual(fits[1].decision_function(Q)[0], expected)
        self.assertEqual(pickle.loads(pickle.dumps(fits[1])).decision_function(csr_matrix(Q))[0], expected)

    def test_exact_ties_avoid_rational_only_when_certified(self):
        K = np.ones((128, 256))
        a = np.tile([1.0, -1.0], 128)
        for policy in ['adaptive', 'compensated']:
            got, info = m.affine_scores(K, a, 0.0, policy=policy, return_info=True)
            np.testing.assert_array_equal(got, np.zeros(128))
            self.assertEqual(info['exact_rows'], 0)
            self.assertEqual(info['joint_rows'], 128)

    def test_underflow_can_conceal_positive_margin(self):
        v = 0.5 + 2.0 ** (-53)
        x = math.ldexp(v, -484)
        a = math.ldexp(v, -485)
        K = np.array([[x, x]])
        alpha = np.array([a, a])
        hi, lo = m._products(K[0], alpha)
        b = -2.0 * float(hi[0])
        expected = oracle(K[0], alpha, b)
        self.assertEqual(expected, np.finfo(float).smallest_subnormal)
        self.assertFalse(m._zero_has_exact_products(K, alpha, False, 0))
        for sparse in [False, True]:
            got, info = m.affine_scores(csr_matrix(K) if sparse else K, alpha, b, policy='compensated', return_info=True)
            self.assertEqual(got[0], expected)
            self.assertEqual(info['exact_rows'], 1)

    def test_exponent_bound_and_random_cancellation(self):
        rng = np.random.default_rng(450018)
        count = 0
        certified = 0
        for i in range(400):
            width = int(rng.integers(1, 17))
            powers = rng.integers(-540, 490, size=(2, width))
            left = np.ldexp(rng.uniform(0.5, 1.0, width), powers[0])
            alpha = np.ldexp(rng.uniform(0.5, 1.0, width), powers[1])
            row = np.repeat(left, 2)
            coef = np.column_stack((alpha, -alpha)).ravel()
            b = 0.0 if i % 3 else float(rng.choice([-1.0, 1.0])) * np.finfo(float).smallest_subnormal
            expected = oracle(row, coef, b)
            for sparse in [False, True]:
                K = csr_matrix(row.reshape(1, -1)) if sparse else row.reshape(1, -1)
                got, info = m.affine_scores(K, coef, b, policy='compensated', return_info=True)
                self.assertEqual(float(got[0]).hex(), float(expected).hex())
                count += 1
                certified += int(info['exact_rows'] == 0 and expected == 0)
        row = np.zeros((1, 4))
        a = np.full(4, np.finfo(float).max)
        self.assertEqual(m.affine_scores(row, a, -0.0, policy='compensated')[0], 0.0)

    def test_guards_and_scratch_failure(self):
        K = np.ones((1, 2))
        a = np.array([1.0, -1.0])
        with patch.object(m, '_exact_row', wraps=m._exact_row) as exact:
            got, info = m.affine_scores(K, a, 0.0, policy='compensated', return_info=True)
            self.assertEqual(exact.call_count, 1)
            self.assertEqual(info['exact_rows'], 1)
        with patch.object(m, '_zero_has_exact_products', side_effect=MemoryError('certificate scratch')):
            got, info = m.affine_scores(K, a, 0.0, policy='compensated', return_info=True)
            self.assertEqual(got[0], 0.0)
            self.assertEqual(info['exact_rows'], 1)
        original = m._products
        with patch.object(m, '_products', wraps=original):
            got, info = m.affine_scores(K, a, 0.0, policy='compensated', return_info=True)
            self.assertEqual(info['exact_rows'], 1)
        with patch.object(m, '_fsum', wraps=m._fsum):
            got, info = m.affine_scores(K, a, 0.0, policy='compensated', return_info=True)
            self.assertEqual(info['exact_rows'], 1)

    def test_duplicate_csr_ties_and_finite_recovery(self):
        K = csr_matrix((np.array([2.0 ** 54, 1.0, -2.0 ** 54, -1.0]), np.array([0, 0, 0, 0]), np.array([0, 4])), shape=(1, 1))
        a = np.array([1.0])
        before = [K.data.copy(), K.indices.copy(), K.indptr.copy()]
        got, info = m.affine_scores(K, a, 0.0, policy='compensated', return_info=True)
        self.assertEqual(got[0], 0.0)
        self.assertEqual(info['exact_rows'], 0)
        for x, y in zip(before, [K.data, K.indices, K.indptr]):
            np.testing.assert_array_equal(x, y)
        maximum = np.finfo(float).max
        got, info = m.affine_scores(np.array([[2.0, 2.0]]), np.array([maximum, -maximum]), 1.0, policy='compensated', return_info=True)
        self.assertEqual(got[0], 1.0)
        self.assertEqual(info['exact_rows'], 1)
