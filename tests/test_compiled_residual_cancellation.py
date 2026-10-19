"""Bounded dispatch preserves arithmetic and joins workers on interruption."""
from concurrent.futures import ThreadPoolExecutor
import threading
import types
import unittest
from unittest.mock import patch

import numpy as np

import dwd._compiled_residual as compiled


class BoundedDispatchTests(unittest.TestCase):
    def inputs(self, n=1025):
        return np.broadcast_to(np.array(1.), (n, n)), np.ones(n), np.zeros(n)

    def test_small_calls_and_term_work_bound(self):
        for n in (8, 64, 255, 256):
            self.assertEqual(compiled._chunk_rows(n), n)
        for n in (257, 1025, 8192, 32768, 1000000):
            size = compiled._chunk_rows(n)
            self.assertGreaterEqual(size, 1)
            self.assertLessEqual(size, compiled._CHUNK_MAX_ROWS)
            self.assertLessEqual(n * size, compiled._CHUNK_MAX_TERMS)
        self.assertEqual(compiled._chunk_rows(compiled._CHUNK_MAX_TERMS + 1), 1)

    def test_partial_outputs_declined_after_single_worker_refusal(self):
        K, x, rhs = self.inputs()
        calls = []

        def evaluate(*args):
            calls.append(args[5:7])
            for output in args[7:]:
                output[:] = 42.
            return 2 if len(calls) == 2 else 0

        with patch.object(compiled, '_ACCEL', types.SimpleNamespace(evaluate=evaluate)), patch.object(
                compiled, '_worker_count', return_value=1):
            self.assertIsNone(compiled.compiled_values(K, x, rhs, .1, .2))
        self.assertEqual(len(calls), 2)

    def test_single_worker_preserves_original_interrupt(self):
        K, x, rhs = self.inputs()
        error = KeyboardInterrupt('owned dispatch interruption')
        calls = []

        def evaluate(*args):
            calls.append(args[5:7])
            if len(calls) == 2:
                raise error
            for output in args[7:]:
                output[:] = 42.
            return 0

        with patch.object(compiled, '_ACCEL', types.SimpleNamespace(evaluate=evaluate)), patch.object(
                compiled, '_worker_count', return_value=1):
            with self.assertRaises(KeyboardInterrupt) as caught:
                compiled.compiled_values(K, x, rhs, .1, .2)
        self.assertIs(caught.exception, error)
        self.assertEqual(len(calls), 2)

    def test_main_interrupt_sets_cancellation_before_join_and_preserves_error(self):
        K, x, rhs = self.inputs()
        cancelled, may_return = threading.Event(), threading.Event()
        started = threading.Barrier(3)
        threads, calls = [], []
        error = KeyboardInterrupt('main thread interruption')
        owner = self

        def evaluate(*args):
            threads.append(threading.current_thread())
            calls.append(args[5:7])
            started.wait(timeout=10)
            if not may_return.wait(timeout=10):
                raise AssertionError('join did not release owned workers')
            return 0

        class InterruptingExecutor(ThreadPoolExecutor):
            def __exit__(self, exc_type, exc_value, traceback):
                try:
                    owner.assertTrue(cancelled.is_set())
                    owner.assertIs(exc_value, error)
                finally:
                    may_return.set()
                return super().__exit__(exc_type, exc_value, traceback)

        def interrupted_wait(futures, timeout):
            self.assertGreater(timeout, 0.)
            self.assertLessEqual(timeout, .05)
            started.wait(timeout=10)
            raise error

        with patch.object(compiled, '_ACCEL', types.SimpleNamespace(evaluate=evaluate)), patch.object(
                compiled, '_worker_count', return_value=2), patch.object(
                compiled, 'Event', return_value=cancelled), patch.object(
                compiled, 'ThreadPoolExecutor', InterruptingExecutor), patch.object(
                compiled, 'wait', side_effect=interrupted_wait):
            with self.assertRaises(KeyboardInterrupt) as caught:
                compiled.compiled_values(K, x, rhs, .1, .2)
        self.assertIs(caught.exception, error)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(not thread.is_alive() for thread in threads))

    def test_worker_refusal_and_exception_cancel_peers_and_join(self):
        for failure in (2, RuntimeError('native failure'), KeyboardInterrupt('worker interrupt')):
            with self.subTest(failure=failure):
                K, x, rhs = self.inputs()
                cancelled = threading.Event()
                started = threading.Barrier(2)
                threads, calls = [], []

                def evaluate(*args):
                    threads.append(threading.current_thread())
                    calls.append(args[5:7])
                    started.wait(timeout=10)
                    if args[5] == 0:
                        if not cancelled.wait(timeout=10):
                            raise AssertionError('peer did not request cancellation')
                        return 0
                    if isinstance(failure, BaseException):
                        raise failure
                    return failure

                with patch.object(compiled, '_ACCEL', types.SimpleNamespace(evaluate=evaluate)), patch.object(
                        compiled, '_worker_count', return_value=2), patch.object(
                        compiled, 'Event', return_value=cancelled):
                    if isinstance(failure, BaseException):
                        with self.assertRaises(type(failure)) as caught:
                            compiled.compiled_values(K, x, rhs, .1, .2)
                        self.assertIs(caught.exception, failure)
                    else:
                        self.assertIsNone(compiled.compiled_values(K, x, rhs, .1, .2))
                self.assertEqual(len(calls), 2)
                self.assertTrue(cancelled.is_set())
                self.assertTrue(all(not thread.is_alive() for thread in threads))


@unittest.skipUnless(compiled._ACCEL is not None, 'Optional compiled extension unavailable.')
class BoundedNativeArithmeticTests(unittest.TestCase):
    def test_chunked_dispatch_exact_across_workers_and_layouts(self):
        rng = np.random.default_rng(20261018)
        n = 257
        for layout in ('C', 'F', 'strided', 'cancelled', 'signed_zero'):
            K = rng.normal(size=(n, n))
            x, rhs = rng.normal(size=n), rng.normal(size=n)
            if layout == 'F':
                K = np.asfortranarray(K)
            elif layout == 'strided':
                K, x, rhs = K[::-1, ::-1], x[::-1], rhs[::-1]
            elif layout == 'cancelled':
                K[:] = 1.
                x[::2], x[1::2] = 1e16, -1e16
                x[-1] = 1.
            elif layout == 'signed_zero':
                K[:], x[:], rhs[:] = -0., -0., -0.
            before = [a.tobytes() for a in (K, x, rhs)]
            expected = [np.empty(n) for _ in range(3)]
            self.assertEqual(compiled._ACCEL.evaluate(K, x, rhs, .125, .3, 0, n, *expected), 0)
            for workers in (1, 2, 4):
                with self.subTest(layout=layout, workers=workers), patch.object(
                        compiled, '_worker_count', return_value=workers):
                    actual = compiled.compiled_values(K, x, rhs, .125, .3)
                    self.assertIsNotNone(actual)
                    self.assertEqual([a.tobytes() for a in actual], [a.tobytes() for a in expected])
                    self.assertEqual([a.tobytes() for a in (K, x, rhs)], before)


if __name__ == '__main__':
    unittest.main()
