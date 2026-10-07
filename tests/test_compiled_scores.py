"""Native dense scoring preserves the established expanded-product evaluator."""
import gc
import math
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from dwd import _compiled_residual as loader
from dwd import _compiled_scores as compiled
from dwd import _kernel_scores as scores


def expanded(K, alpha, indices=None, *, batch_enabled=True):
    indices = range(len(K)) if indices is None else indices
    split = scores._split_operand(alpha)
    result = np.full(len(K), -321., dtype=float)
    batching = scores._expanded_dense_rows(K, alpha, indices, split, result,
                                           batch_enabled=batch_enabled)
    return result, batching


def portable(K, alpha, indices=None, *, batch_enabled=True):
    with patch.object(loader, '_ACCEL', None):
        return expanded(K, alpha, indices, batch_enabled=batch_enabled)


def python_entry(tile, alpha, bm, be, bh, bl, output):
    high, low = compiled._PRODUCTS(tile, np.broadcast_to(alpha, tile.shape),
                                   _split_b=(bm, be, bh, bl))
    for i in range(len(tile)):
        output[i] = compiled._FSUM(high[i].tolist() + low[i].tolist())
    return 0


class CompiledScoreDispatchTests(unittest.TestCase):
    def setUp(self):
        self.K = np.arange(45, dtype=float).reshape(9, 5) / 8.
        self.alpha = np.array([1., -2., 3., -4., 1.])

    def assertBitsEqual(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_missing_or_older_extension_and_unsupported_runtime(self):
        expected, state = portable(self.K, self.alpha)
        for extension in (None, SimpleNamespace(evaluate=lambda *args: 0)):
            with self.subTest(extension=extension), patch.object(loader, '_ACCEL', extension):
                actual, actual_state = expanded(self.K, self.alpha)
                self.assertBitsEqual(actual, expected)
                self.assertEqual(actual_state, state)
        entry = unittest.mock.Mock(side_effect=AssertionError('unsupported runtime entered C'))
        with patch.object(loader, '_ACCEL', SimpleNamespace(score_rows=entry)), \
                patch.object(compiled, '_supported', return_value=False):
            actual, _ = expanded(self.K, self.alpha)
        entry.assert_not_called()
        self.assertBitsEqual(actual, expected)

    def test_independent_runtime_guard_and_replaced_math_fsum(self):
        for version in ((3, 11), (3, 13)):
            with patch.object(compiled.sys, 'version_info', version):
                self.assertFalse(compiled._supported())
        with patch.object(compiled.platform, 'machine', return_value='ARM64'):
            self.assertFalse(compiled._supported())
        with patch.object(math, 'fsum', lambda values: sum(values)):
            self.assertFalse(compiled._supported())

    def test_unqualified_operating_system_uses_original_scoring(self):
        expected, state = portable(self.K, self.alpha)
        cancellation = np.array([[1e16, 1., -1e16]] * 11)
        coefficients = np.ones(3)
        functions = (scores.compensated_kernel_matvec, scores.adaptive_kernel_matvec)
        with patch.object(loader, '_ACCEL', None):
            public_expected = [function(cancellation, coefficients) for function in functions]
        entry = unittest.mock.Mock(side_effect=AssertionError('unqualified OS entered C'))
        for operating_system in ('linux', 'darwin'):
            with self.subTest(platform=operating_system), \
                    patch.object(compiled.sys, 'platform', operating_system), \
                    patch.object(compiled.sys, 'version_info', (3, 12)), \
                    patch.object(compiled.platform, 'machine', return_value='AMD64'), \
                    patch.object(loader, '_ACCEL', SimpleNamespace(score_rows=entry)):
                self.assertFalse(compiled._supported())
                actual, actual_state = expanded(self.K, self.alpha)
                self.assertBitsEqual(actual, expected)
                self.assertEqual(actual_state, state)
                for function, reference in zip(functions, public_expected):
                    self.assertBitsEqual(function(cancellation, coefficients), reference)
        entry.assert_not_called()

    def test_dense_layout_indices_and_original_single_row_tail(self):
        entry = unittest.mock.Mock(side_effect=python_entry)
        original_row = scores._expanded_row
        row_hook = unittest.mock.Mock(wraps=original_row)
        for K in (self.K, np.asfortranarray(self.K)):
            for indices in (range(9), np.array([8, 2, 2, 0, 6], dtype=np.intp)):
                with self.subTest(order=K.flags.f_contiguous, indices=str(indices)):
                    expected, state = portable(K, self.alpha, indices)
                    with patch.object(loader, '_ACCEL', SimpleNamespace(score_rows=entry)), \
                            patch.object(compiled, '_supported', return_value=True), \
                            patch.object(scores, '_expanded_row', row_hook):
                        actual, actual_state = expanded(K, self.alpha, indices)
                    self.assertBitsEqual(actual, expected)
                    self.assertEqual(actual_state, state)
        self.assertGreater(entry.call_count, 0)
        self.assertEqual(row_hook.call_count, 2)  # one tail for each nine-row layout

    def test_disabled_batching_strides_dtypes_and_hooks_decline(self):
        entry = unittest.mock.Mock(side_effect=AssertionError('ineligible call entered C'))
        cases = [(self.K[:, ::-1], self.alpha, True),
                 (self.K.astype(np.float32), self.alpha, True),
                 (self.K, self.alpha, False)]
        for K, alpha, batching in cases:
            expected, state = portable(K, alpha, batch_enabled=batching)
            with patch.object(loader, '_ACCEL', SimpleNamespace(score_rows=entry)), \
                    patch.object(compiled, '_supported', return_value=True):
                actual, actual_state = expanded(K, alpha, batch_enabled=batching)
            self.assertBitsEqual(actual, expected)
            self.assertEqual(actual_state, state)
        for hook in ('_products', '_fsum'):
            original = getattr(scores, hook)
            with patch.object(loader, '_ACCEL', SimpleNamespace(score_rows=entry)), \
                    patch.object(compiled, '_supported', return_value=True), \
                    patch.object(scores, hook, wraps=original) as custom:
                expanded(self.K, self.alpha)
                self.assertGreater(custom.call_count, 0)
        entry.assert_not_called()

    def test_call_and_log_error_policies_decline(self):
        entry = unittest.mock.Mock(side_effect=AssertionError('handler bypassed'))
        expected, _ = portable(self.K, self.alpha)
        for mode in ('call', 'log'):
            with patch.object(loader, '_ACCEL', SimpleNamespace(score_rows=entry)), \
                    patch.object(compiled, '_supported', return_value=True), np.errstate(all=mode):
                actual, _ = expanded(self.K, self.alpha)
            self.assertBitsEqual(actual, expected)
        entry.assert_not_called()

    def test_alias_and_unusual_indices_decline(self):
        result = self.K[:, 0]
        split = scores._split_operand(self.alpha)
        entry = unittest.mock.Mock(side_effect=AssertionError('ineligible call entered C'))
        with patch.object(loader, '_ACCEL', SimpleNamespace(score_rows=entry)), \
                patch.object(compiled, '_supported', return_value=True):
            self.assertIsNone(compiled.try_expanded_dense_rows(
                self.K, self.alpha, range(9), split, result, batch_enabled=True))
            for indices in (range(8, -1, -1), np.array([-1, 0], dtype=np.intp), [0, 1]):
                self.assertIsNone(compiled.try_expanded_dense_rows(
                    self.K, self.alpha, indices, split, np.empty(9), batch_enabled=True))
        entry.assert_not_called()

    def test_native_refusal_and_memory_failure_retry_original(self):
        expected, expected_state = portable(self.K, self.alpha)
        for effect in (lambda *args: 2, MemoryError('scratch allocation')):
            with self.subTest(effect=str(effect)), \
                    patch.object(loader, '_ACCEL', SimpleNamespace(score_rows=unittest.mock.Mock(side_effect=effect))), \
                    patch.object(compiled, '_supported', return_value=True):
                actual, state = expanded(self.K, self.alpha)
            self.assertBitsEqual(actual, expected)
            self.assertEqual(state, expected_state)

    def test_scratch_released_before_python_retry(self):
        references = []
        original = scores._products

        def checked_products(*args, **kwargs):
            gc.collect()
            self.assertTrue(references)
            self.assertTrue(all(reference() is None for reference in references))
            return original(*args, **kwargs)

        def failed_entry(tile, alpha, bm, be, bh, bl, output):
            references.extend((weakref.ref(tile), weakref.ref(output)))
            scores._products = checked_products
            raise MemoryError('native scratch')

        expected, _ = portable(self.K, self.alpha)
        with patch.object(loader, '_ACCEL', SimpleNamespace(score_rows=failed_entry)), \
                patch.object(compiled, '_supported', return_value=True), \
                patch.object(scores, '_products', original):
            actual, _ = expanded(self.K, self.alpha)
        self.assertBitsEqual(actual, expected)


_ENTRY = getattr(loader._ACCEL, 'score_rows', None)


@unittest.skipUnless(_ENTRY is not None and compiled._supported(),
                     'Requires the new compiled scorer on CPython 3.12 x86-64.')
class NativeScoreArithmeticTests(unittest.TestCase):
    def assertBitsEqual(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_exact_expanded_products_across_range_and_layout(self):
        rng = np.random.default_rng(913)
        fixtures = [(rng.normal(size=(17, 37)), rng.normal(size=37)),
                    (np.array([[1e16, 1., -1e16]] * 9), np.ones(3)),
                    (np.array([[0., -0., np.finfo(float).smallest_subnormal]] * 9), np.ones(3)),
                    (np.array([[2.**-500, -2.**-500, 2.**-537]] * 9),
                     np.array([2.**-500, 2.**-500, 2.**-537]))]
        for K, alpha in fixtures:
            for matrix in (K, np.asfortranarray(K)):
                for mode in ('ignore', 'raise', 'warn'):
                    with self.subTest(shape=K.shape, fortran=matrix.flags.f_contiguous, mode=mode), np.errstate(all=mode):
                        expected, state = portable(matrix, alpha)
                        actual, actual_state = expanded(matrix, alpha)
                    self.assertBitsEqual(actual, expected)
                    self.assertEqual(actual_state, state)

    def test_public_adaptive_and_compensated_scores_match(self):
        K = np.array([[1e16, 1., -1e16, -0.]] * 11)
        alpha = np.ones(4)
        for function in (scores.compensated_kernel_matvec, scores.adaptive_kernel_matvec):
            with patch.object(loader, '_ACCEL', None):
                expected = function(K, alpha)
            self.assertBitsEqual(function(K, alpha), expected)

    def test_normal_exponent_boundaries_and_libc_fallback(self):
        tiny = np.finfo(float).tiny
        maximum = np.finfo(float).max
        eta = np.finfo(float).smallest_subnormal
        with np.errstate(all='ignore'):
            values = [0., -0., eta, -eta, np.nextafter(tiny, 0.), tiny,
                      np.nextafter(tiny, np.inf), 2.**-500, .5,
                      np.nextafter(1., 0.), 1., np.nextafter(1., np.inf),
                      2.**500, maximum, -maximum]
            scales = [eta, tiny, 2.**-500, .5, 1., 2., 2.**500]
            for value in values:
                for scale in scales:
                    with self.subTest(value=value, scale=scale):
                        K = np.array([[value], [-value]])
                        alpha = np.array([scale])
                        try:
                            expected, state = portable(K, alpha)
                        except FloatingPointError as original_error:
                            with self.assertRaisesRegex(FloatingPointError, str(original_error)):
                                expanded(K, alpha)
                        else:
                            actual, actual_state = expanded(K, alpha)
                            self.assertBitsEqual(actual, expected)
                            self.assertEqual(actual_state, state)

    def test_first_error_order_and_no_partial_success(self):
        maximum = np.finfo(float).max
        K = np.array([[maximum, maximum], [np.inf, 0.]])
        for extension in (None, loader._ACCEL):
            with patch.object(loader, '_ACCEL', extension):
                with self.assertRaisesRegex(FloatingPointError, 'compensated residual sum'):
                    expanded(K, np.ones(2))

    def test_bridge_buffer_contract_and_release(self):
        K, alpha = np.ones((2, 3)), np.ones(3)
        split = scores._split_operand(alpha)
        output = np.empty(2)
        args = [K, alpha, *split, output]
        self.assertEqual(_ENTRY(*args), 0)
        invalid = []
        for replacement in (split[1].astype(np.int64), split[1].astype('>i4'),
                            np.zeros((1, 3), dtype=np.int32)):
            changed = list(args)
            changed[3] = replacement
            invalid.append(changed)
        changed = list(args)
        changed[0] = K[:, ::-1]
        invalid.append(changed)
        changed = list(args)
        changed[-1] = alpha[:2]
        invalid.append(changed)
        readonly = output.copy()
        readonly.flags.writeable = False
        changed = list(args)
        changed[-1] = readonly
        invalid.append(changed)
        for changed in invalid:
            with self.subTest(types=[a.dtype for a in changed]):
                references = [sys.getrefcount(a) for a in changed]
                with self.assertRaises((ValueError, TypeError, BufferError)):
                    _ENTRY(*changed)
                self.assertEqual([sys.getrefcount(a) for a in changed], references)
        self.assertEqual(_ENTRY(*args), 0)


if __name__ == '__main__':
    unittest.main()
