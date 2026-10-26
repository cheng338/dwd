"""Cache-identity and validation contracts for shared numerical test controls."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal

import dwd._eigen as eigen
import numerical_test_support as support
from numerical_test_support import AcceptedNativeReplay


class AcceptedNativeReplayTests(unittest.TestCase):
    def setUp(self):
        self.matrix = np.diag([.25, .5, 2.])
        self.options = dict(driver='evd', eigvals_only=False,
                            check_finite=False, overwrite_a=False)

    def assert_accepted(self, matrix, result):
        # The same real validator is also exercised by the helper; do not
        # manufacture successful validation to test cache behavior.
        checked, statistics = eigen._matrix_statistics(matrix)
        issues, _ = eigen._validate_pairs(
            checked, result[0], result[1], statistics)
        self.assertEqual(issues, [])

    def test_identical_calls_reuse_accepted_result(self):
        with AcceptedNativeReplay() as replay:
            first = eigen.validated_eigh(self.matrix, drivers=('evd',))
            second = eigen.validated_eigh(self.matrix.copy(), drivers=('evd',))
            self.assertEqual(replay.native_calls, 1)
            self.assertEqual(replay.replays, 1)
            self.assertEqual(len(replay.cache), 1)
            for left, right in zip(first, second):
                assert_array_equal(left, right)
                self.assertFalse(np.shares_memory(left, right))

    def test_changed_matrix_identity_does_not_reuse(self):
        changed_bytes = self.matrix.copy()
        changed_bytes[0, 0] = np.nextafter(changed_bytes[0, 0], np.inf)
        storage = np.zeros((6, 6))
        strided = storage[::2, ::2]
        strided[:] = self.matrix
        variants = {
            'bytes': changed_bytes,
            'fortran_layout': self.matrix.copy(order='F'),
            'strided_layout': strided,
            'dtype': self.matrix.astype(np.float32),
            'shape': np.diag([.25, .5, 2., 3.]),
        }
        for name, changed in variants.items():
            with self.subTest(identity=name), AcceptedNativeReplay() as replay:
                # Exercise raw-provider identity directly: validated_eigh
                # promotes inputs before reaching this test-only cache.
                first = replay.call(self.matrix, **self.options)
                second = replay.call(changed, **self.options)
                self.assert_accepted(self.matrix, first)
                self.assert_accepted(changed, second)
                self.assertEqual(replay.native_calls, 2)
                self.assertEqual(replay.replays, 0)
                self.assertEqual(len(replay.cache), 2)

    def test_changed_provider_options_do_not_reuse(self):
        variants = ({'driver': 'evr'}, {'lower': False},
                    {'eigvals_only': True})
        for changed in variants:
            with self.subTest(options=changed), AcceptedNativeReplay() as replay:
                replay.call(self.matrix, **self.options)
                options = dict(self.options, **changed)
                result = replay.call(self.matrix, **options)
                if options['eigvals_only']:
                    _, statistics = eigen._matrix_statistics(self.matrix)
                    issues, _ = eigen._validate_values(result, statistics)
                    self.assertEqual(issues, [])
                else:
                    self.assert_accepted(self.matrix, result)
                self.assertEqual(replay.native_calls, 2)
                self.assertEqual(replay.replays, 0)
                self.assertEqual(len(replay.cache), 2)

    def test_runtime_identity_changes_prevent_reuse(self):
        # Simulate inventory changes without changing the process's actual
        # numerical libraries or thread limits. Native results remain real.
        for field in ('num_threads', 'filepath', 'version'):
            with self.subTest(field=field), AcceptedNativeReplay() as replay:
                pools = support.threadpool_info()
                self.assertTrue(pools)
                changed = copy.deepcopy(pools)
                previous = changed[0].get(field)
                changed[0][field] = ((previous or 0) + 1 if field == 'num_threads'
                                     else str(previous) + '-changed')
                with patch.object(support, 'threadpool_info', return_value=pools):
                    replay.call(self.matrix, **self.options)
                with patch.object(support, 'threadpool_info', return_value=changed):
                    result = replay.call(self.matrix, **self.options)
                self.assert_accepted(self.matrix, result)
                self.assertEqual(replay.native_calls, 2)
                self.assertEqual(replay.replays, 0)

    def test_runtime_inventory_order_does_not_prevent_reuse(self):
        with AcceptedNativeReplay() as replay:
            pools = copy.deepcopy(support.threadpool_info())
            self.assertTrue(pools)
            # Keep this ordering check meaningful even on a single-pool build.
            if len(pools) == 1:
                other = copy.deepcopy(pools[0])
                other['filepath'] = str(other.get('filepath')) + '-second-pool'
                pools.append(other)
            with patch.object(support, 'threadpool_info', return_value=pools):
                first = replay.call(self.matrix, **self.options)
            with patch.object(support, 'threadpool_info', return_value=pools[::-1]):
                second = replay.call(self.matrix, **self.options)
            self.assertEqual(replay.native_calls, 1)
            self.assertEqual(replay.replays, 1)
            for left, right in zip(first, second):
                assert_array_equal(left, right)

    def test_rejected_provider_result_is_never_cached(self):
        with AcceptedNativeReplay() as replay:
            native = replay.native

            def invalid(*args, **kwargs):
                values, vectors = native(*args, **kwargs)
                vectors = vectors.copy(order='K')
                vectors[:, 0] *= 2.
                return values, vectors

            with patch.object(replay, 'native', invalid):
                for _ in range(2):
                    with self.assertRaises(eigen.EigenDecompositionError):
                        eigen.validated_eigh(self.matrix, drivers=('evd',))
            self.assertEqual(replay.native_calls, 2)
            self.assertEqual(replay.replays, 0)
            self.assertEqual(replay.cache, {})

    def test_returned_arrays_cannot_mutate_cached_result(self):
        with AcceptedNativeReplay() as replay:
            first = replay.call(self.matrix, **self.options)
            expected = tuple(item.copy(order='K') for item in first)
            first[0][0] = -123.
            first[1][:] = 0.
            second = replay.call(self.matrix, **self.options)
            for actual, wanted in zip(second, expected):
                assert_array_equal(actual, wanted)
            second[0][0] = -456.
            second[1][:] = 0.
            third = replay.call(self.matrix, **self.options)
            self.assert_accepted(self.matrix, third)
            for actual, wanted in zip(third, expected):
                assert_array_equal(actual, wanted)
            self.assertEqual(replay.native_calls, 1)
            self.assertEqual(replay.replays, 2)


if __name__ == '__main__':
    unittest.main()
