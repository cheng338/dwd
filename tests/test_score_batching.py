"""Bounded score batching preserves retained row-wise arithmetic and guards.

The frozen score and product helpers load independently of the live package.
No external checkout, environment variable or saved study data is required.
"""
from fractions import Fraction
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings
import weakref

import numpy as np
from scipy.sparse import csr_array, csr_matrix

import dwd._kernel_scores as scores


FUNCTIONS = ('compensated_kernel_matvec', 'adaptive_kernel_matvec')


def load_baseline():
    root = Path(__file__).resolve().parent / 'fixtures' / 'score_batching_before'
    provenance = json.loads((root / 'provenance.json').read_text(encoding='utf-8'))
    if set(provenance['files']) != {'_kernel_scores.py', '_compensated_residual.py'}:
        raise AssertionError('The retained score fixture inventory changed')
    for name, expected in provenance['files'].items():
        data = (root / name).read_bytes().replace(b'\r\n', b'\n')
        if hashlib.sha256(data).hexdigest() != expected['sha256_lf']:
            raise AssertionError('The retained score fixture changed: ' + name)
    package = types.ModuleType('_dwd_score_batching_before')
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package
    name = package.__name__ + '._kernel_scores'
    spec = importlib.util.spec_from_file_location(name, root / '_kernel_scores.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BEFORE = load_baseline()


def snapshot(value):
    if hasattr(value, 'indptr'):
        return value.shape, value.format, snapshot(value.data), snapshot(value.indices), snapshot(value.indptr)
    if isinstance(value, np.ndarray):
        return value.dtype.str, value.shape, value.strides, value.flags.writeable, value.tobytes()
    return repr(value)


def capture(function, K, alpha, policy=None):
    with np.errstate(**({'all': 'ignore'} if policy is None else policy)), warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        old_policy = np.geterr()
        try:
            value = np.asarray(function(K, alpha))
            outcome = ('value', value.dtype.str, value.shape, value.tobytes())
        except Exception as error:
            outcome = ('exception', type(error).__name__, str(error))
        if np.geterr() != old_policy:
            raise AssertionError('Caller NumPy error policy changed')
        return outcome, [(item.category.__name__, str(item.message)) for item in seen]


def dense_cases():
    alpha = np.array([1e16, 1., -1e16])
    K = np.tile([[1., 1., 1.], [2., -3., 2.], [0., 4., 0.], [.5, 3., .5]], (9, 1))
    yield 'cancellation', K, alpha
    yield 'fortran', np.asfortranarray(K), alpha
    yield 'negative_strides', K[::-1, ::-1], alpha[::-1]
    backing = np.zeros(6); backing[::2] = alpha
    yield 'strided_coefficients', K, backing[::2]
    unaligned = np.ndarray(K.shape, dtype='float64', buffer=bytearray(K.nbytes+1), offset=1)
    unaligned[:] = K
    yield 'unaligned', unaligned, alpha
    yield 'broadcast', np.broadcast_to(K[0], (33, 3)), alpha
    yield 'float32', K.astype(np.float32), alpha.astype(np.float32)
    yield 'integer', np.tile([[1, 2, 3], [0, -2, 1]], (17, 1)), np.array([1, 2, 3])
    yield 'boolean', np.tile([[True, False], [True, True]], (17, 1)), np.array([True, False])
    yield 'zero_signs', np.tile([[0., -0., 0., -0.], [-0., 0., -0., 0.]], (17, 1)), np.array([0., -0., 1., -1.])
    eta = np.nextafter(0., 1.)
    yield 'subnormal', np.tile([[eta, -eta], [2*eta, eta], [-eta, -2*eta]], (11, 1)), np.array([.5, 1.])
    yield 'cancelling_extreme', np.tile([[1e308, 1e308], [-1e308, -1e308]], (17, 1)), np.array([1., -1.])
    yield 'extreme_coefficients', np.tile([[.5, .5], [2.**-900, 2.**-900]], (17, 1)), np.array([1e308, -1e308])
    yield 'extreme_scales', np.tile([[2.**900, 2.**-900], [-2.**900, 2.**-900]], (17, 1)), np.array([2.**-900, 2.**900])
    yield 'zero_coefficients', np.tile([[2., -3.], [4., 5.]], (17, 1)), np.array([0., -0.])
    yield 'empty_rows', np.empty((0, 3)), alpha
    mixed = np.tile([1., 0., 0.], (260, 1))
    mixed[[0, 127, 128, 129, 255, 256, 259]] = [1., 1., 1.]
    yield 'mixed_three_blocks', mixed, alpha
    late = np.tile([1., 0., 0.], (260, 1)); late[[128, 259]] = [1., 1., 1.]
    yield 'late_uncertain', late, alpha
    rng = np.random.default_rng(61005011)
    for n in (1, 2, 7, 32):
        K = np.ldexp(rng.uniform(-1., 1., size=(17, n)), rng.integers(-400, 401, size=(17, n)))
        alpha = np.ldexp(rng.uniform(-1., 1., size=n), rng.integers(-400, 401, size=n))
        yield 'random_exponents_' + str(n), K, alpha


class ScoreBatchDifferentialTests(unittest.TestCase):
    def assert_same(self, K, alpha, policy=None):
        before_inputs = snapshot(K), snapshot(alpha)
        for name in FUNCTIONS:
            with self.subTest(function=name):
                self.assertEqual(capture(getattr(scores, name), K, alpha, policy),
                                 capture(getattr(BEFORE, name), K, alpha, policy))
                self.assertEqual((snapshot(K), snapshot(alpha)), before_inputs)

    def test_dense_layouts_cancellation_extremes_and_mixed_rows_are_bitwise_equal(self):
        for label, K, alpha in dense_cases():
            with self.subTest(case=label):
                K.flags.writeable = alpha.flags.writeable = False
                self.assert_same(K, alpha)

    def test_error_policy_warnings_and_overflow_exceptions_match(self):
        cases = list(dense_cases())
        cases += [('product_overflow', np.tile([1e308, 1.], (17, 1)), np.array([2., 1.])),
                  ('sum_overflow', np.tile([1e308, 1e308], (17, 1)), np.array([1., 1.]))]
        for policy in ({'all': 'raise'}, {'all': 'warn'},
                       {'over': 'raise', 'invalid': 'raise', 'under': 'ignore', 'divide': 'raise'}):
            for label, K, alpha in cases:
                with self.subTest(case=label, policy=policy):
                    self.assert_same(K, alpha, policy)

    def test_sparse_duplicate_storage_is_unchanged(self):
        data = np.array([1e16, 1., -1e16, 2., -2., 3.])
        indices, indptr = np.array([0, 0, 0, 1, 1, 0]), np.array([0, 3, 3, 6])
        for cls in (csr_matrix, csr_array):
            K = cls((data.copy(), indices.copy(), indptr.copy()), shape=(3, 2))
            with self.subTest(kind=cls.__name__), patch.object(
                    scores, '_expanded_dense_rows', side_effect=AssertionError('Sparse data batched')):
                self.assert_same(K, np.array([1., 1.]))
                self.assertFalse(K.has_canonical_format)

    def test_changed_coefficients_between_calls_are_not_cached(self):
        K = np.tile([1., 1., 1.], (33, 1))
        for name in FUNCTIONS:
            alpha = np.array([1e16, 1., -1e16])
            with self.subTest(function=name), patch.object(scores, '_split_operand', wraps=scores._split_operand) as prepare:
                first = getattr(scores, name)(K, alpha)
                alpha[1] = 3.
                second = getattr(scores, name)(K, alpha)
                self.assertEqual(prepare.call_count, 2)
                self.assertEqual(first.tobytes(), np.ones(33).tobytes())
                self.assertEqual(second.tobytes(), np.full(33, 3.).tobytes())

    def test_independent_rational_cancellation_reference(self):
        epsilon = 2.**-27
        K = np.tile([[1.+epsilon, -1.], [-1.-epsilon, 1.]], (17, 1))
        alpha = np.array([1.-epsilon, 1.])
        expected = np.array([float(sum((Fraction.from_float(float(k))*Fraction.from_float(float(a))
                                      for k, a in zip(row, alpha)), Fraction())) for row in K])
        actual = scores.compensated_kernel_matvec(K, alpha)
        self.assertEqual(actual.tobytes(), expected.tobytes())


class ScoreBatchBehaviorTests(unittest.TestCase):
    def test_public_paths_actually_batch_and_keep_per_row_sum_order(self):
        K = np.tile([1., 1., 1.], (33, 1))
        K[:, 1] = np.arange(1., 34.)
        alpha = np.array([1e16, 1., -1e16])
        for name in FUNCTIONS:
            original_product = scores._products
            shapes, sums, old_sums = [], [], []
            def products(a, b, **kwargs):
                shapes.append(np.shape(a))
                return original_product(a, b, **kwargs)
            def observe_sum(terms):
                sums.append(list(terms))
                return original_sum(terms)
            def observe_before(terms):
                old_sums.append(list(terms))
                return before_sum(terms)
            original_sum, before_sum = scores._fsum, BEFORE._fsum
            with self.subTest(function=name), patch.object(scores, '_products', side_effect=products), patch.object(
                    scores, '_fsum', side_effect=observe_sum), patch.object(BEFORE, '_fsum', side_effect=observe_before):
                actual = getattr(scores, name)(K, alpha)
                expected = getattr(BEFORE, name)(K, alpha)
            self.assertEqual(actual.tobytes(), expected.tobytes())
            self.assertEqual(sums, old_sums)
            self.assertTrue(any(len(shape) == 2 and shape[0] > 1 for shape in shapes))
            self.assertEqual(sum(shape[0] if len(shape) == 2 else 1 for shape in shapes), len(K))

    def test_tile_rows_and_element_budget_are_bounded(self):
        for n in (1, 3, 64, 1024, 4096, 4097, 10000):
            K = np.ones((25, n)); alpha = np.ones(n)
            original, shapes = scores._products, []
            def observed(a, b, **kwargs):
                shapes.append(np.shape(a)); return original(a, b, **kwargs)
            with self.subTest(columns=n), patch.object(scores, '_products', side_effect=observed):
                actual = scores.compensated_kernel_matvec(K, alpha)
            self.assertEqual(actual.tobytes(), BEFORE.compensated_kernel_matvec(K, alpha).tobytes())
            batch_shapes = [s for s in shapes if len(s) == 2]
            for shape in batch_shapes:
                self.assertLessEqual(shape[0], 8)
                self.assertLessEqual(shape[0]*shape[1]*128, 1024*1024)
            if n > 4096:
                self.assertFalse(batch_shapes)

    def test_helper_changes_only_requested_rows_in_given_order(self):
        rng = np.random.default_rng(61005031)
        K, alpha = rng.normal(size=(13, 17)), rng.normal(size=17)
        indices = np.array([7, 2, 11, 0, 8, 9])
        result = np.full(len(K), -712.5)
        expected = result.copy()
        for i in indices: expected[i] = BEFORE._expanded_row(K, alpha, False, int(i))
        scores._expanded_dense_rows(K, alpha, indices, scores._split_operand(alpha), result)
        self.assertEqual(result.tobytes(), expected.tobytes())

    def test_single_expanded_row_and_reliable_rows_do_not_batch(self):
        alpha = np.array([1e16, 1., -1e16])
        original, shapes = scores._products, []
        def observed(a, b, **kwargs):
            shapes.append(np.shape(a)); return original(a, b, **kwargs)
        with patch.object(scores, '_products', side_effect=observed):
            K = np.array([[1., 0., 0.], [1., 1., 1.], [0., 2., 0.]])
            actual = scores.adaptive_kernel_matvec(K, alpha)
        self.assertEqual(actual.tobytes(), BEFORE.adaptive_kernel_matvec(K, alpha).tobytes())
        self.assertEqual(shapes, [(3,)])
        with patch.object(scores, '_products', side_effect=AssertionError('Reliable row expanded')):
            scores.adaptive_kernel_matvec(np.eye(3), alpha)

    def test_natural_later_product_failure_preserves_earlier_sum_exception(self):
        K, alpha = np.array([[1e308, 1e308], [np.nan, 1.]]), np.array([1., 1.])
        expected = capture(BEFORE.compensated_kernel_matvec, K, alpha)
        actual = capture(scores.compensated_kernel_matvec, K, alpha)
        self.assertEqual(actual, expected)
        self.assertEqual(actual[0][1:], ('FloatingPointError', 'Nonfinite or overflowing compensated residual sum.'))

    def test_injected_batch_failure_preserves_earlier_sum_exception(self):
        K, alpha = np.array([[1e308, 1e308], [1., 1.]]), np.array([1., 1.])
        original, batches = scores._products, []
        def observed(a, b, **kwargs):
            if np.ndim(a) == 2:
                batches.append(1)
                raise FloatingPointError('Controlled later-row batch failure')
            return original(a, b, **kwargs)
        with patch.object(scores, '_products', side_effect=observed):
            actual = capture(scores.compensated_kernel_matvec, K, alpha)
        self.assertEqual(actual, capture(BEFORE.compensated_kernel_matvec, K, alpha))
        self.assertEqual(len(batches), 1)

    def test_product_memory_error_disables_batching_for_whole_call(self):
        K, alpha = np.tile([1., 1., 1.], (260, 1)), np.array([1e16, 1., -1e16])
        for name in FUNCTIONS:
            original, batches = scores._products, []
            def observed(a, b, **kwargs):
                if np.ndim(a) == 2:
                    batches.append(1)
                    raise MemoryError('Controlled tile product allocation failure')
                return original(a, b, **kwargs)
            with self.subTest(function=name), patch.object(scores, '_products', side_effect=observed):
                actual = getattr(scores, name)(K, alpha)
            self.assertEqual(actual.tobytes(), getattr(BEFORE, name)(K, alpha).tobytes())
            self.assertEqual(len(batches), 1)

    def test_gather_memory_error_before_tile_exists_falls_back_without_retry(self):
        class FailGather(np.ndarray):
            def __new__(cls, values):
                result = np.asarray(values).view(cls); result.attempts = 0; return result
            def __array_finalize__(self, obj): self.attempts = getattr(obj, 'attempts', 0)
            def __getitem__(self, item):
                if isinstance(item, (list, np.ndarray)):
                    self.attempts += 1
                    raise MemoryError('Controlled gather allocation failure')
                return super().__getitem__(item)
        K = FailGather(np.tile([1., 1., 1.], (33, 1)))
        alpha = np.array([1e16, 1., -1e16]); result = np.empty(len(K))
        scores._expanded_dense_rows(K, alpha, np.arange(len(K)), scores._split_operand(alpha), result)
        self.assertEqual(K.attempts, 1)
        self.assertEqual(result.tobytes(), BEFORE.compensated_kernel_matvec(np.asarray(K), alpha).tobytes())

    def test_sum_numerical_error_is_propagated_without_replaying_arithmetic(self):
        K, alpha = np.tile([1., 1., 1.], (17, 1)), np.array([1e16, 1., -1e16])
        with patch.object(scores, '_fsum', side_effect=FloatingPointError('Controlled sum failure')) as summed:
            with self.assertRaisesRegex(FloatingPointError, 'Controlled sum'):
                scores.compensated_kernel_matvec(K, alpha)
        self.assertEqual(summed.call_count, 1)

    def test_sum_memory_failure_falls_back_and_disables_subsequent_batches(self):
        K, alpha = np.tile([1., 1., 1.], (260, 1)), np.array([1e16, 1., -1e16])
        for name in FUNCTIONS:
            original, batch_calls = scores._products, []
            original_sum, sum_calls = scores._fsum, []
            def products(a, b, **kwargs):
                if np.ndim(a) == 2: batch_calls.append(1)
                return original(a, b, **kwargs)
            def summed(terms):
                sum_calls.append(1)
                if len(sum_calls) == 1: raise MemoryError('Controlled sum memory failure')
                return original_sum(terms)
            with self.subTest(function=name), patch.object(scores, '_products', products), patch.object(scores, '_fsum', summed):
                actual = getattr(scores, name)(K, alpha)
            self.assertEqual(actual.tobytes(), getattr(BEFORE, name)(K, alpha).tobytes())
            self.assertEqual(len(batch_calls), 1)

    def test_batch_traceback_arrays_are_released_before_row_fallback(self):
        K, alpha = np.tile([1., 1., 1.], (17, 1)), np.array([1e16, 1., -1e16])
        original, references, checked = scores._products, [], []
        def products(a, b, **kwargs):
            if np.ndim(a) == 2:
                scratch = np.empty_like(a)
                references.extend([weakref.ref(scratch), weakref.ref(a)])
                raise MemoryError('Controlled failure retaining temporary in traceback')
            if references:
                self.assertTrue(all(ref() is None for ref in references))
                checked.append(1)
            return original(a, b, **kwargs)
        with patch.object(scores, '_products', products):
            actual = scores.compensated_kernel_matvec(K, alpha)
        self.assertTrue(checked)
        self.assertEqual(actual.tobytes(), BEFORE.compensated_kernel_matvec(K, alpha).tobytes())



HAS_WIDE_FLOAT = np.dtype(np.longdouble).itemsize > 8


class ScoreBatchRevisionTests(unittest.TestCase):
    def test_signed_and_unsigned_integer_extremes_match_original(self):
        cases = [
            (np.tile(np.array([[-(2**63), 7, -(2**63)], [2**63-1, -3, 2**63-1]], dtype=np.int64), (9, 1)),
             np.array([1e6, 1., -1e6])),
            (np.tile(np.array([[2**64-1, 7, 2**64-1], [2**63+1, 3, 2**63+1]], dtype=np.uint64), (9, 1)),
             np.array([1e6, 1., -1e6])),
        ]
        for K, alpha in cases:
            before = snapshot(K), snapshot(alpha)
            for name in FUNCTIONS:
                with self.subTest(dtype=K.dtype.str, function=name):
                    self.assertEqual(capture(getattr(scores, name), K, alpha),
                                     capture(getattr(BEFORE, name), K, alpha))
                    self.assertEqual((snapshot(K), snapshot(alpha)), before)

    def test_supported_float_widths_and_non_native_endian_match_original(self):
        for dtype in (np.float16, np.float32, np.float64, np.dtype('float64').newbyteorder('S')):
            K = np.tile([[1., 1., 1.], [2., -3., 2.], [-0., 0., -0.]], (11, 1)).astype(dtype)
            alpha = np.array([1e16, 1., -1e16])
            for name in FUNCTIONS:
                with self.subTest(dtype=K.dtype.str, function=name):
                    self.assertEqual(capture(getattr(scores, name), K, alpha),
                                     capture(getattr(BEFORE, name), K, alpha))

    @unittest.skipUnless(HAS_WIDE_FLOAT, 'This NumPy platform has no floating dtype wider than float64.')
    def test_actual_wide_float_values_use_original_rows_and_match_bits(self):
        K = np.tile([[1., 1., 1.], [2., -3., 2.], [-0., 0., -0.]], (11, 1)).astype(np.longdouble)
        alpha = np.array([1e16, 1., -1e16])
        for name in FUNCTIONS:
            shapes, original = [], scores._products
            def products(a, b, **kwargs):
                shapes.append(np.shape(a)); return original(a, b, **kwargs)
            with self.subTest(function=name), patch.object(scores, '_products', products):
                self.assertEqual(capture(getattr(scores, name), K, alpha),
                                 capture(getattr(BEFORE, name), K, alpha))
            self.assertTrue(shapes)
            self.assertTrue(all(len(shape) == 1 for shape in shapes))

    @unittest.skipUnless(HAS_WIDE_FLOAT, 'This NumPy platform has no floating dtype wider than float64.')
    def test_earlier_sum_error_precedes_later_wide_cast_error(self):
        large = np.longdouble(np.finfo(np.float64).max) * np.longdouble(2)
        K = np.array([[np.longdouble(1e308), np.longdouble(1e308)],
                      [large, np.longdouble(1)]], dtype=np.longdouble)
        alpha = np.array([1., 1.])
        for policy in ({'all': 'raise'}, {'all': 'warn'}, {'all': 'ignore'}):
            with self.subTest(policy=policy):
                expected = capture(BEFORE.compensated_kernel_matvec, K, alpha, policy)
                actual = capture(scores.compensated_kernel_matvec, K, alpha, policy)
                self.assertEqual(actual, expected)
                self.assertEqual(actual[0][1:], ('FloatingPointError', 'Nonfinite or overflowing compensated residual sum.'))

    def test_wide_dtype_guard_routing_surrogate_is_not_arithmetic_evidence(self):
        class RoutingSurrogate(np.ndarray):
            @property
            def dtype(self):
                return SimpleNamespace(kind='f', itemsize=16)
        ordinary = np.tile([1., 1., 1.], (17, 1))
        K = ordinary.view(RoutingSurrogate)
        alpha, result = np.array([1e16, 1., -1e16]), np.empty(len(K))
        shapes, original = [], scores._products
        def products(a, b, **kwargs):
            shapes.append(np.shape(a)); return original(a, b, **kwargs)
        with patch.object(scores, '_products', products):
            scores._expanded_dense_rows(K, alpha, range(len(K)), scores._split_operand(alpha), result)
        self.assertEqual(shapes, [(3,)]*len(K))
        self.assertEqual(result.tobytes(), BEFORE.compensated_kernel_matvec(ordinary, alpha).tobytes())


if __name__ == '__main__':
    unittest.main(verbosity=2)
