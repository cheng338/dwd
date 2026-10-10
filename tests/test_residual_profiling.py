"""Opt-in dispatch observations must preserve numerical calls and results."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import inspect
import json
import math
import threading
import unittest
from unittest.mock import patch

import numpy as np

import dwd._compiled_residual as compiled
import dwd._native_residual as native
import dwd.profiling as profiling
from dwd.profiling import residual_profile


def inputs(n=16):
    rng = np.random.default_rng(94283)
    return (rng.uniform(.5, 1., (n, n)), .125, rng.normal(size=n),
            rng.uniform(.25, .75, n), .3, -.25)


class ProfileContextTests(unittest.TestCase):
    def call_declined(self):
        # An invalid dtype declines before dispatch on supported platforms and
        # remains a valid declined invocation on unsupported platforms.
        args = inputs(2)
        return native.native_compensated_residual(args[0].astype('float32'), *args[1:])

    def test_original_six_positional_interface_and_no_inactive_allocations(self):
        args = inputs(2)
        self.assertEqual(tuple(inspect.signature(native.native_compensated_residual).parameters),
                         ('K', 'shift', 'rhs', 'x', 's', 'target_sum'))
        sentinel = object()
        with patch.object(native, '_native_compensated_residual', return_value=sentinel) as core, \
                patch.object(profiling, '_ResidualCall', side_effect=AssertionError('sample allocated')), \
                patch.object(profiling.time, 'perf_counter', side_effect=AssertionError('timer called')):
            self.assertIs(native.native_compensated_residual(*args), sentinel)
        core.assert_called_once_with(*args)

    def test_inactive_real_call_does_not_read_clock(self):
        with patch.object(profiling.time, 'perf_counter', side_effect=AssertionError('timer called')):
            self.assertIsNone(self.call_declined())

    def test_empty_profile_and_snapshot_are_independent_and_json_compatible(self):
        with residual_profile() as profile:
            empty = profile.as_dict()
            self.assertEqual(empty['counts']['calls'], 0)
            self.call_declined()
        first = profile.as_dict()
        json.dumps(first, allow_nan=False)
        first['counts']['calls'] = -1
        first['routes']['pre_dispatch']['calls'] = -1
        self.assertEqual(profile.as_dict()['counts']['calls'], 1)
        self.assertEqual(empty['counts']['calls'], 0)
        self.assertIsNone(profiling._active_profile())

    def test_nested_scope_excludes_inner_and_restores_after_exception(self):
        with residual_profile() as outer:
            self.call_declined()
            with self.assertRaisesRegex(RuntimeError, 'body failure'):
                with residual_profile() as inner:
                    self.call_declined()
                    raise RuntimeError('body failure')
            self.call_declined()
        self.assertEqual(outer.as_dict()['counts']['calls'], 2)
        self.assertEqual(inner.as_dict()['counts']['calls'], 1)
        self.assertIsNone(profiling._active_profile())

    def test_copied_context_after_exit_cannot_resume_recording_or_read_clock(self):
        for raises in (False, True):
            with self.subTest(exceptional_exit=raises):
                try:
                    with residual_profile() as profile:
                        self.call_declined()
                        saved_context = copy_context()
                        if raises:
                            raise RuntimeError('scope ended')
                except RuntimeError:
                    if not raises:
                        raise
                before = profile.as_dict()
                with patch.object(profiling.time, 'perf_counter',
                                  side_effect=AssertionError('stale scope read timer')):
                    self.assertIsNone(saved_context.run(profiling._active_profile))
                    self.assertIsNone(saved_context.run(self.call_declined))
                self.assertEqual(profile.as_dict(), before)

    def test_threads_collect_independently_and_unscoped_thread_is_excluded(self):
        barrier = threading.Barrier(2)

        def worker(count):
            with residual_profile() as profile:
                barrier.wait(timeout=10)
                for _ in range(count):
                    self.call_declined()
            return profile.as_dict()['counts']['calls']

        with residual_profile() as outer:
            self.call_declined()
            with ThreadPoolExecutor(max_workers=2) as pool:
                a, b = pool.submit(worker, 2), pool.submit(worker, 3)
                self.assertEqual((a.result(timeout=15), b.result(timeout=15)), (2, 3))
                pool.submit(self.call_declined).result(timeout=15)
            self.call_declined()
        self.assertEqual(outer.as_dict()['counts']['calls'], 2)

    def test_async_inherited_scope_is_excluded_and_child_scope_is_independent(self):
        async def unscoped():
            self.assertIsNone(profiling._active_profile())
            self.call_declined()

        async def scoped():
            with residual_profile() as profile:
                await asyncio.sleep(0)
                self.call_declined()
            return profile.as_dict()['counts']['calls']

        async def run():
            with residual_profile() as outer:
                self.call_declined()
                _, child_count = await asyncio.gather(unscoped(), scoped())
                self.call_declined()
            return outer.as_dict()['counts']['calls'], child_count

        self.assertEqual(asyncio.run(run()), (2, 1))

    def test_unsupported_and_pre_dispatch_declines_are_separate(self):
        with residual_profile() as profile:
            with patch.object(native, '_native_supported', return_value=False):
                self.call_declined()
            with patch.object(native, '_native_supported', return_value=True), \
                    patch.object(native, '_round_to_nearest', return_value=True):
                self.call_declined()
        result = profile.as_dict()
        self.assertEqual(result['routes']['unsupported']['declined'], 1)
        self.assertEqual(result['routes']['pre_dispatch']['declined'], 1)
        self.assertEqual(result['compiled_helper']['calls'], 0)
        self.assertEqual(result['counts']['scalar_sumprod_calls_started'], 0)


@unittest.skipUnless(native._native_supported(), 'Requires audited CPython native sumprod.')
class ProfileNumericalTests(unittest.TestCase):
    def assert_bits_equal(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
            self.assertEqual(a.shape, b.shape)
            self.assertEqual(a.tobytes(), b.tobytes())

    def test_scalar_result_and_bounds_are_bitwise_unchanged(self):
        args = inputs(7)
        with patch.object(native, 'compiled_values', return_value=None):
            expected = native.native_compensated_residual(*args)
            with residual_profile() as profile:
                actual = native.native_compensated_residual(*args)
        self.assertIsNotNone(actual)
        self.assert_bits_equal(actual, expected)
        report = profile.as_dict()
        self.assertEqual(report['routes']['scalar_fallback']['bounded_return'], 1)
        self.assertEqual(report['counts']['scalar_sumprod_calls_started'], 14)
        self.assertEqual(report['counts']['scalar_sumprod_calls_completed'], 14)
        self.assertEqual(report['compiled_helper']['declined'], 1)

    @unittest.skipIf(compiled._ACCEL is None, 'Optional native extension is unavailable.')
    def test_compiled_c_fortran_and_strided_results_bounds_inputs_unchanged(self):
        raw = inputs()
        for args in (raw, (np.asfortranarray(raw[0]), *raw[1:]),
                     (raw[0][::-1, ::-1], raw[1], raw[2][::-1], raw[3][::-1], *raw[4:])):
            with self.subTest(strides=args[0].strides):
                snapshots = [np.asarray(args[i]).tobytes() for i in (0, 2, 3)]
                for i in (0, 2, 3):
                    args[i].flags.writeable = False
                expected = native.native_compensated_residual(*args)
                with residual_profile() as profile:
                    actual = native.native_compensated_residual(*args)
                self.assertIsNotNone(actual)
                self.assert_bits_equal(actual, expected)
                result = profile.as_dict()
                self.assertEqual(result['routes']['compiled_values']['bounded_return'], 1)
                self.assertEqual(result['compiled_helper']['values_returned'], 1)
                self.assertEqual(result['counts']['scalar_sumprod_calls_started'], 0)
                self.assertEqual(snapshots, [np.asarray(args[i]).tobytes() for i in (0, 2, 3)])
                self.assertGreaterEqual(result['wall_seconds'], result['compiled_helper']['wall_seconds'])

    def test_partial_scalar_work_is_counted_at_exact_decline_position(self):
        args = inputs(3)
        args[0][-1, 0] = np.nan
        with patch.object(native, 'compiled_values', return_value=None):
            expected = native.native_compensated_residual(*args)
            with residual_profile() as profile:
                actual = native.native_compensated_residual(*args)
        self.assertIsNone(expected)
        self.assertIsNone(actual)
        result = profile.as_dict()
        self.assertEqual(result['counts']['scalar_sumprod_calls_started'], 4)
        self.assertEqual(result['counts']['scalar_sumprod_calls_completed'], 4)
        self.assertEqual(result['decline_reasons'], {'row_operands': 1})

    def test_score_precision_decline_preserves_partial_work_and_original_call(self):
        args = (np.ones((2, 2)), .01, np.ones(2), np.array([1e7, -1e7]), 0., 0.)
        with patch.object(native, 'compiled_values', return_value=None):
            self.assertIsNone(native.native_compensated_residual(*args))
            with residual_profile() as profile:
                self.assertIsNone(native.native_compensated_residual(*args))
        result = profile.as_dict()
        self.assertEqual(result['decline_reasons'], {'score_precision': 1})
        self.assertEqual(result['counts']['scalar_sumprod_calls_completed'], 1)

    def test_compiled_values_do_not_imply_bounded_return_or_solver_acceptance(self):
        args = inputs()
        values = (np.zeros(16), np.zeros(16), np.ones(16))
        with patch.object(native, 'compiled_values', return_value=values):
            with residual_profile() as profile:
                self.assertIsNone(native.native_compensated_residual(*args))
        result = profile.as_dict()
        self.assertEqual(result['compiled_helper']['values_returned'], 1)
        self.assertEqual(result['routes']['compiled_values']['declined'], 1)
        self.assertEqual(result['counts']['bounded_return'], 0)
        self.assertEqual(result['decline_reasons'], {'score_precision': 1})

    def test_scalar_raised_call_is_started_but_not_completed(self):
        with patch.object(native, '_native_supported', return_value=True), \
                patch.object(native, 'compiled_values', return_value=None), \
                patch.object(native.math, 'sumprod', side_effect=ValueError('numerical failure')):
            with residual_profile() as profile:
                self.assertIsNone(native.native_compensated_residual(*inputs(2)))
        result = profile.as_dict()
        self.assertEqual(result['counts']['scalar_sumprod_calls_started'], 1)
        self.assertEqual(result['counts']['scalar_sumprod_calls_completed'], 0)
        self.assertEqual(result['counts']['declined'], 1)
        self.assertEqual(result['decline_reasons'], {'arithmetic_exception': 1})

    def test_compiled_exceptions_keep_existing_caught_and_propagated_behavior(self):
        for exception, caught in ((ValueError('invalid'), True), (RuntimeError('unexpected'), False)):
            with self.subTest(exception=type(exception).__name__):
                with patch.object(native, 'compiled_values', side_effect=exception):
                    with residual_profile() as profile:
                        if caught:
                            self.assertIsNone(native.native_compensated_residual(*inputs()))
                        else:
                            with self.assertRaisesRegex(RuntimeError, 'unexpected'):
                                native.native_compensated_residual(*inputs())
                result = profile.as_dict()
                self.assertEqual(result['compiled_helper']['raised'], 1)
                self.assertEqual(result['routes']['compiled_dispatch']['calls'], 1)
                self.assertEqual(result['counts']['declined' if caught else 'raised'], 1)

    def test_all_route_times_partition_native_scope_and_counts_balance(self):
        with residual_profile() as profile:
            native.native_compensated_residual(*inputs(2))
            native.native_compensated_residual(*inputs())
            with patch.object(native, '_native_supported', return_value=False):
                native.native_compensated_residual(*inputs())
        result = profile.as_dict()
        self.assertEqual(result['counts']['calls'], 3)
        self.assertEqual(sum(route['calls'] for route in result['routes'].values()), 3)
        self.assertTrue(math.isclose(sum(route['wall_seconds'] for route in result['routes'].values()),
                                     result['wall_seconds'], rel_tol=1e-15))
        self.assertEqual(result['counts']['calls'], sum(result['counts'][key]
                         for key in ('bounded_return', 'declined', 'raised')))


if __name__ == '__main__':
    unittest.main(verbosity=2)
