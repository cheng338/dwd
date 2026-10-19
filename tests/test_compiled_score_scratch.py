"""Bounded per-call scratch preserves accurate scores and fallback lifetimes."""
from concurrent.futures import ThreadPoolExecutor
import gc
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from dwd import _compiled_residual as loader
from dwd import _compiled_scores as compiled
from dwd import _kernel_scores as scores


class AllocationProbe:
    def __init__(self, fail_empty=None, fail_copy=None):
        self.empty_calls = self.copy_calls = 0
        self.fail_empty, self.fail_copy = fail_empty, fail_copy
        self.references = []

    def __getattr__(self, name):
        return getattr(np, name)

    def empty(self, *args, **kwargs):
        self.empty_calls += 1
        if self.empty_calls == self.fail_empty:
            raise MemoryError('injected scratch allocation')
        value = np.empty(*args, **kwargs)
        self.references.append(weakref.ref(value))
        return value

    def copyto(self, *args, **kwargs):
        self.copy_calls += 1
        if self.copy_calls == self.fail_copy:
            raise MemoryError('injected bounded row copy')
        return np.copyto(*args, **kwargs)

    def released(self):
        gc.collect()
        return all(reference() is None for reference in self.references)


def reference_entry(tile, alpha, bm, be, bh, bl, output):
    high, low = compiled._PRODUCTS(tile, np.broadcast_to(alpha, tile.shape),
                                   _split_b=(bm, be, bh, bl))
    for i in range(len(tile)):
        output[i] = compiled._FSUM(high[i].tolist() + low[i].tolist())
    return 0


def expanded(K, alpha, indices=None, enabled=True):
    indices = range(len(K)) if indices is None else indices
    result = np.full(len(K), -321., dtype=float)
    state = scores._expanded_dense_rows(K, alpha, indices, scores._split_operand(alpha),
                                        result, batch_enabled=enabled)
    return result, state


class CompiledScoreScratchTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(182608)
        self.K, self.alpha = rng.normal(size=(33, 37)), rng.normal(size=37)

    def portable(self, K, alpha, indices=None, enabled=True):
        with patch.object(loader, '_ACCEL', None):
            return expanded(K, alpha, indices, enabled)

    def assert_bits_equal(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_one_scratch_pair_preserves_layout_selection_and_duplicate_order(self):
        for K in (self.K, np.asfortranarray(self.K)):
            for indices in (range(33), range(1, 32, 2),
                            np.array([32, 2, 2, 1, 17, 8, 4, 22, 0], dtype=np.intp)):
                with self.subTest(fortran=K.flags.f_contiguous, indices=indices):
                    expected, expected_state = self.portable(K, self.alpha, indices)
                    allocation, addresses, rows = AllocationProbe(), [], []
                    before = K.tobytes(), self.alpha.tobytes()
                    def entry(tile, alpha, *args):
                        addresses.append((tile.ctypes.data, args[-1].ctypes.data))
                        rows.append(tile.copy())
                        return reference_entry(tile, alpha, *args)
                    with patch.object(compiled, 'np', allocation), \
                            patch.object(compiled, '_supported', return_value=True), \
                            patch.object(loader, '_ACCEL', SimpleNamespace(score_rows=entry)):
                        actual, state = expanded(K, self.alpha, indices)
                    self.assert_bits_equal(actual, expected)
                    self.assertEqual(state, expected_state)
                    self.assertEqual(allocation.empty_calls, 2)
                    self.assertEqual(len(set(addresses)), 1)
                    selected = np.asarray(list(indices))
                    if len(selected) % 8 == 1:
                        selected = selected[:-1]
                    self.assert_bits_equal(np.concatenate(rows), K[selected])
                    self.assertEqual(before, (K.tobytes(), self.alpha.tobytes()))
                    self.assertTrue(allocation.released())

    def test_scalar_empty_disabled_and_wide_paths_allocate_no_native_scratch(self):
        for K, alpha, indices, enabled in (
                (self.K, self.alpha, range(1), True),
                (self.K, self.alpha, range(0), True),
                (self.K, self.alpha, range(33), False),
                (np.ones((3, 8193)), np.ones(8193), range(3), True)):
            expected, expected_state = self.portable(K, alpha, indices, enabled)
            allocation = AllocationProbe()
            with patch.object(compiled, 'np', allocation), \
                    patch.object(compiled, '_supported', return_value=True), \
                    patch.object(loader, '_ACCEL', SimpleNamespace(score_rows=lambda *a:
                        self.fail('Scalar-only input entered native score path'))):
                actual, state = expanded(K, alpha, indices, enabled)
            self.assert_bits_equal(actual, expected)
            self.assertEqual(state, expected_state)
            self.assertEqual(allocation.empty_calls, 0)

    def test_scratch_is_released_before_scalar_tail(self):
        allocation, tails = AllocationProbe(), []
        original = scores._expanded_row
        def checked_row(*args, **kwargs):
            tails.append(args[3])
            self.assertTrue(allocation.released())
            return original(*args, **kwargs)
        expected, _ = self.portable(self.K, self.alpha)
        with patch.object(compiled, 'np', allocation), \
                patch.object(compiled, '_supported', return_value=True), \
                patch.object(loader, '_ACCEL', SimpleNamespace(score_rows=reference_entry)), \
                patch.object(scores, '_expanded_row', checked_row):
            actual, _ = expanded(self.K, self.alpha)
        self.assert_bits_equal(actual, expected)
        self.assertEqual(tails, [32])

    def test_reused_second_tile_failure_releases_scratch_and_recomputes_all_rows(self):
        expected, expected_state = self.portable(self.K, self.alpha)
        for failure in (2, MemoryError, FloatingPointError, OverflowError):
            with self.subTest(failure=failure):
                allocation, calls = AllocationProbe(), []
                original_products = scores._products
                def checked_products(*args, **kwargs):
                    self.assertTrue(allocation.released())
                    return original_products(*args, **kwargs)
                def entry(tile, alpha, *args):
                    calls.append(1)
                    if len(calls) == 2:
                        args[-1][:] = 987654321.
                        scores._products = checked_products
                        if isinstance(failure, type):
                            raise failure('injected native refusal')
                        return failure
                    return reference_entry(tile, alpha, *args)
                with patch.object(compiled, 'np', allocation), \
                        patch.object(compiled, '_supported', return_value=True), \
                        patch.object(loader, '_ACCEL', SimpleNamespace(score_rows=entry)), \
                        patch.object(scores, '_products', original_products):
                    actual, state = expanded(self.K, self.alpha)
                self.assertEqual(len(calls), 2)
                self.assert_bits_equal(actual, expected)
                self.assertEqual(state, expected_state)
                self.assertTrue(allocation.released())

    def test_allocation_and_copy_failures_preserve_portable_result(self):
        expected, expected_state = self.portable(self.K, self.alpha)
        for allocation in (AllocationProbe(fail_empty=1), AllocationProbe(fail_empty=2),
                           AllocationProbe(fail_copy=1), AllocationProbe(fail_copy=2)):
            with patch.object(compiled, 'np', allocation), \
                    patch.object(compiled, '_supported', return_value=True), \
                    patch.object(loader, '_ACCEL', SimpleNamespace(score_rows=reference_entry)):
                actual, state = expanded(self.K, self.alpha)
            self.assert_bits_equal(actual, expected)
            self.assertEqual(state, expected_state)
            self.assertTrue(allocation.released())

    def test_simultaneous_calls_use_separate_scratch(self):
        barrier, lock = threading.Barrier(4), threading.Lock()
        addresses, first_threads = [], set()
        def entry(tile, alpha, *args):
            tid = threading.get_ident()
            with lock:
                first = tid not in first_threads
                if first:
                    first_threads.add(tid)
                    addresses.append((tile.ctypes.data, args[-1].ctypes.data))
            if first:
                barrier.wait(timeout=10)
            return reference_entry(tile, alpha, *args)
        inputs = [(self.K + i, self.alpha) for i in range(4)]
        expected = [self.portable(*item)[0] for item in inputs]
        with patch.object(compiled, '_supported', return_value=True), \
                patch.object(loader, '_ACCEL', SimpleNamespace(score_rows=entry)), \
                ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(expanded, *item) for item in inputs]
            actual = [future.result(timeout=15)[0] for future in futures]
        self.assertEqual(len({a for a, _ in addresses}), 4)
        self.assertEqual(len({b for _, b in addresses}), 4)
        for first, second in zip(actual, expected):
            self.assert_bits_equal(first, second)


if __name__ == '__main__':
    unittest.main()
