"""Dense preparation reuse preserves the previous score arithmetic and guards.

The portable fixture is the byte-identical helper before preparation reuse.
It imports the unchanged product/summation primitives, but executes its own
original row loop and screening decisions independently of the current helper.
"""
from fractions import Fraction
import hashlib
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from scipy.sparse import csr_array, csr_matrix

import dwd._kernel_scores as scores


# Ignore checkout line endings while retaining the exact source text.
BEFORE_SHA256 = 'e6cd72be63a0c565d9b74c29d6097097d308c6202cfd8c575aac7059e913ed15'
FUNCTIONS = ('compensated_kernel_matvec', 'adaptive_kernel_matvec')


def before_module():
    path = Path(__file__).parent / 'fixtures' / 'score_preparation_before.py'
    if hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() != BEFORE_SHA256:
        raise AssertionError('The independent before-reuse fixture changed')
    spec = importlib.util.spec_from_file_location('dwd._score_preparation_before', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture(function, K, alpha, policy):
    with np.errstate(**policy), warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        before = np.geterr()
        try:
            values = np.asarray(function(K, alpha))
            outcome = ('values', values.dtype.str, values.shape, values.tobytes())
        except Exception as error:
            outcome = ('exception', type(error).__name__, str(error))
        if np.geterr() != before:
            raise AssertionError('Score evaluation changed the caller error policy')
        messages = [(item.category.__name__, str(item.message)) for item in seen]
    return outcome, messages


def input_snapshot(value):
    if hasattr(value, 'indptr'):
        return (value.shape, value.format, input_snapshot(value.data),
                input_snapshot(value.indices), input_snapshot(value.indptr))
    if isinstance(value, np.ndarray):
        return value.dtype.str, value.shape, value.strides, value.flags.writeable, value.tobytes()
    return repr(value)


def dense_cases():
    alpha = np.array([1e16, 1., -1e16])
    query = np.array([[1., 1., 1.], [2., -3., 2.], [0., 4., 0.],
                      [1., 1e-10, 1.], [-1., -2., -1.], [.5, 3., .5]])
    yield 'cancellation', query, alpha
    yield 'fortran', np.asfortranarray(query), alpha
    yield 'reverse_strides', query[::-1, ::-1], alpha[::-1]
    storage = np.zeros(6)
    storage[::2] = alpha
    yield 'strided_coefficients', query, storage[::2]
    yield 'float32', query.astype(np.float32), alpha.astype(np.float32)
    yield 'integer_input', np.array([[1, 2, 3], [0, -2, 1]], dtype=np.int64), np.array([1, 2, 3])
    yield 'boolean_input', np.array([[True, False], [True, True]]), np.array([True, False])
    yield 'signed_zero', np.array([[0., -0., 0., -0.], [-0., 0., -0., 0.]]), np.array([0., -0., 1., -1.])
    eta = np.nextafter(0., 1.)
    yield 'subnormal', np.array([[eta, -eta], [2*eta, eta], [-eta, -2*eta]]), np.array([.5, 1.])
    yield 'cancelling_extreme', np.array([[1e308, 1e308], [-1e308, -1e308]]), np.array([1., -1.])
    yield 'extreme_coefficients', np.array([[.5, .5], [2.**-900, 2.**-900]]), np.array([1e308, -1e308])
    yield 'extreme_scales', np.array([[2.**900, 2.**-900], [-2.**900, 2.**-900]]), np.array([2.**-900, 2.**900])
    yield 'empty_rows', np.empty((0, 3)), alpha
    yield 'zero_coefficients', np.array([[2., -3.], [4., 5.]]), np.array([0., -0.])
    mixed = np.tile([1., 0., 0.], (260, 1))
    mixed[[0, 127, 128, 129, 255, 256, 259]] = [1., 1., 1.]
    yield 'mixed_three_blocks', mixed, alpha
    late = np.tile([1., 0., 0.], (260, 1))
    late[[128, 259]] = [1., 1., 1.]
    yield 'first_block_all_reliable', late, alpha
    rng = np.random.default_rng(36192)
    for columns in (1, 2, 7, 32):
        query = np.ldexp(rng.uniform(-1., 1., size=(9, columns)), rng.integers(-400, 401, size=(9, columns)))
        coefficients = np.ldexp(rng.uniform(-1., 1., size=columns), rng.integers(-400, 401, size=columns))
        yield 'exponent_mix_' + str(columns), query, coefficients


class ScorePreparationDifferentialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = before_module()

    def assert_same(self, K, alpha, policy=None):
        policy = {'all': 'ignore'} if policy is None else policy
        snapshots = input_snapshot(K), input_snapshot(alpha)
        for name in FUNCTIONS:
            with self.subTest(helper=name, policy=policy):
                expected = capture(getattr(self.before, name), K, alpha, policy)
                actual = capture(getattr(scores, name), K, alpha, policy)
                self.assertEqual(actual, expected)
                self.assertEqual((input_snapshot(K), input_snapshot(alpha)), snapshots)

    def test_dense_values_signed_zero_layouts_and_cancellation_match_bytes(self):
        for label, query, alpha in dense_cases():
            with self.subTest(case=label):
                query.flags.writeable = alpha.flags.writeable = False
                self.assert_same(query, alpha)

    def test_external_error_policies_and_warning_behavior_match(self):
        cases = list(dense_cases())
        cases += [('product_overflow', np.array([[1e308]]), np.array([2.])),
                  ('sum_overflow', np.array([[1e308, 1e308]]), np.array([1., 1.]))]
        for policy in ({'all': 'raise'}, {'all': 'warn'},
                       {'over': 'raise', 'invalid': 'raise', 'under': 'ignore', 'divide': 'raise'}):
            for label, query, alpha in cases:
                with self.subTest(case=label, policy=policy):
                    self.assert_same(query, alpha, policy)

    def test_validation_and_exception_types_messages_are_unchanged(self):
        cases = [([1., 2.], [1., 2.]), ([[1., 2.]], [[1., 2.]]),
                 ([[1., 2.]], [1.]), ([[1j]], [1.]), ([[1.]], [1j]),
                 ([['1']], [1.]), ([[1.]], []), (np.empty((0, 0)), []),
                 (np.empty((0, 3)), [1., np.inf, 2.]),
                 ([[np.nan]], [1.]), ([[np.inf]], [1.]), ([[1.]], [np.inf]),
                 ([[1e308]], [2.]), ([[1e308, 1e308]], [1., 1.]),
                 (csr_matrix([[1.]]).tocsc(), [1.]), (csr_matrix([[np.inf]]), [1.])]
        for index, (query, alpha) in enumerate(cases):
            with self.subTest(case=index):
                self.assert_same(query, alpha)

    def test_csr_duplicates_empty_rows_and_sparse_array_storage_are_unchanged(self):
        data = np.array([1e16, 1., -1e16, 2., -2., 3.])
        indices = np.array([0, 0, 0, 1, 1, 0])
        indptr = np.array([0, 3, 3, 6])
        for constructor in (csr_matrix, csr_array):
            query = constructor((data.copy(), indices.copy(), indptr.copy()), shape=(3, 2))
            with self.subTest(constructor=constructor.__name__):
                self.assertFalse(query.has_canonical_format)
                self.assert_same(query, np.array([1., 1.]))
                np.testing.assert_array_equal(scores.compensated_kernel_matvec(query, [1., 1.]), [1., 0., 3.])
                self.assertFalse(query.has_canonical_format)
                self.assert_same(constructor((0, 2)), np.array([1., 1.]))

    def test_exact_scalar_product_cancellation_has_independent_rational_reference(self):
        epsilon = 2.**-27
        query = np.array([[1.+epsilon, -1.], [-1.-epsilon, 1.]])
        alpha = np.array([1.-epsilon, 1.])
        expected = np.array([float(sum((Fraction.from_float(float(k))*Fraction.from_float(float(a))
                                       for k, a in zip(row, alpha)), Fraction())) for row in query])
        self.assertEqual(expected[0], -2.**-54)
        for name in FUNCTIONS:
            with self.subTest(helper=name):
                actual = getattr(scores, name)(query, alpha)
                # The adaptive screen may accept ordinary products at this
                # absolute scale; compare it to its own unchanged policy.
                if name == 'compensated_kernel_matvec':
                    self.assertEqual(actual.tobytes(), expected.tobytes())
                self.assertEqual(actual.tobytes(), getattr(self.before, name)(query, alpha).tobytes())


class ScorePreparationLifetimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = before_module()

    def test_empty_and_reliable_dense_queries_do_not_prepare_coefficients(self):
        alpha = np.array([1e16, 1., -1e16])
        with patch.object(scores, '_split_operand', side_effect=AssertionError('Unneeded dense preparation')):
            for name in FUNCTIONS:
                self.assertEqual(getattr(scores, name)(np.empty((0, 3)), alpha).shape, (0,))
            np.testing.assert_array_equal(scores.adaptive_kernel_matvec(np.eye(3), alpha), alpha)

    def test_compensated_dense_prepares_once_and_reuses_the_same_split(self):
        query = np.tile([1., 1., 1.], (7, 1))
        alpha = np.array([1e16, 1., -1e16])
        products = scores._products
        with patch.object(scores, '_split_operand', wraps=scores._split_operand) as prepare, patch.object(
                scores, '_products', wraps=products) as expanded:
            actual = scores.compensated_kernel_matvec(query, alpha)
        self.assertEqual(prepare.call_count, 1)
        calls = expanded.call_args_list
        # Each logical row is expanded once, whether alone or within a tile.
        self.assertEqual(sum(np.shape(call.args[0])[0] if np.ndim(call.args[0]) == 2
                             else 1 for call in calls), len(query))
        prepared = [call.kwargs.get('_split_b') for call in calls]
        self.assertIsNotNone(prepared[0])
        self.assertTrue(all(item is prepared[0] for item in prepared))
        self.assertEqual(actual.tobytes(), self.before.compensated_kernel_matvec(query, alpha).tobytes())

    def test_adaptive_late_and_mixed_blocks_prepare_once_for_uncertain_rows_only(self):
        alpha = np.array([1e16, 1., -1e16])
        query = np.tile([1., 0., 0.], (260, 1))
        uncertain = [128, 129, 255, 256, 259]
        query[uncertain] = [1., 1., 1.]
        original, rows = scores._expanded_dense_rows, []

        def observed(K, coefficients, indices, *args, **kwargs):
            rows.extend(int(index) for index in indices)
            return original(K, coefficients, indices, *args, **kwargs)

        with patch.object(scores, '_split_operand', wraps=scores._split_operand) as prepare, patch.object(
                scores, '_expanded_dense_rows', side_effect=observed):
            actual = scores.adaptive_kernel_matvec(query, alpha)
        self.assertEqual(rows, uncertain)
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(actual.tobytes(), self.before.adaptive_kernel_matvec(query, alpha).tobytes())

    def test_changed_coefficients_between_calls_get_fresh_preparation(self):
        query = np.tile([1., 1., 1.], (3, 1))
        for name in FUNCTIONS:
            alpha = np.array([1e16, 1., -1e16])
            with self.subTest(helper=name), patch.object(scores, '_split_operand', wraps=scores._split_operand) as prepare:
                first = getattr(scores, name)(query, alpha)
                alpha[1] = 3.
                second = getattr(scores, name)(query, alpha)
                self.assertEqual(prepare.call_count, 2)
            np.testing.assert_array_equal(first, np.ones(3))
            np.testing.assert_array_equal(second, np.full(3, 3.))
            self.assertEqual(second.tobytes(), getattr(self.before, name)(query, alpha).tobytes())

    def test_sparse_rows_do_not_receive_or_compute_a_dense_prepared_split(self):
        query = csr_matrix((np.array([1e16, 1., -1e16, 2., -2., 3.]),
                            np.array([0, 0, 0, 1, 1, 0]), np.array([0, 3, 3, 6])), shape=(3, 2))
        alpha = np.array([1., 1.])
        for name in FUNCTIONS:
            with self.subTest(helper=name), patch.object(scores, '_split_operand', side_effect=AssertionError('Dense preparation for sparse rows')), patch.object(
                    scores, '_products', wraps=scores._products) as expanded:
                actual = getattr(scores, name)(query, alpha)
            self.assertTrue(expanded.call_args_list)
            self.assertTrue(all('_split_b' not in call.kwargs for call in expanded.call_args_list))
            self.assertEqual(actual.tobytes(), getattr(self.before, name)(query, alpha).tobytes())


if __name__ == '__main__':
    unittest.main(verbosity=2)
