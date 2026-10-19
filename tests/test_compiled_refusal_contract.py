"""Guarded compiled operand refusals preserve arithmetic and fallback contracts.

The genuine-core cases require the optional extension and a qualified native
runtime. The generic-dispatch cases also run without the extension; they do not
impersonate another Python version or make a failed numerical screen acceptable.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from fractions import Fraction
import gc
import math
from threading import Barrier, Event, Lock
import types
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy as np
from threadpoolctl import threadpool_limits
from dwd import _compiled_residual as compiled
from dwd import _native_residual as native
from dwd import _compensated_residual as portable
from dwd.profiling import residual_profile


def fixture(n=16):
    K = (np.arange(n * n).reshape(n, n) % 16 + 1).astype(float) / 16.
    x = (np.arange(n) % 8 + 1).astype(float) / 8.
    rhs = (np.arange(n) % 17 - 8).astype(float) / 16.
    return [K, .125, rhs, x, -.25, .5]


def core_args(args):
    K, shift, rhs, x, intercept, _ = args
    return K, x, rhs, shift, intercept


def exact(value):
    return Fraction.from_float(float(value))


class IteratedRows(np.ndarray):
    """Expose ordinary ndarray storage but intentionally customize iteration."""
    def __new__(cls, storage, iterated):
        value = np.asarray(storage).view(cls)
        value.iterated_rows = np.asarray(iterated)
        return value

    def __array_finalize__(self, original):
        self.iterated_rows = getattr(original, 'iterated_rows', None)

    def __iter__(self):
        return iter(self.iterated_rows)


class _RefusalAssertions:
    def setUp(self):
        super().setUp()
        self.enterContext(threadpool_limits(limits=1))

    def same_values(self, first, second):
        self.assertEqual(len(first), len(second))
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def profiled(self, args):
        with residual_profile() as observed:
            result = native.native_compensated_residual(*args)
        return result, observed.as_dict()

    def generic_baseline(self, args):
        # Original public contract maps internal refusal detail back to None.
        # A distinct imported callable prevents the specialized shortcut, without
        # changing numerical arithmetic or impersonating a different runtime.
        original = self.original_public
        with patch.object(native, 'compiled_values', lambda *values: original(*values)):
            return self.profiled(args)

    def completed_or_portable(self, args, result):
        return portable.compensated_residual(*args) if result is None else result

    def assert_bounds(self, args, result):
        self.assertIsNotNone(result)
        K, shift, rhs, x, intercept, target = args
        residual, constraint, scores, rb, cb, sb = result
        for i, row in enumerate(K):
            score = sum((exact(a) * exact(b) for a, b in zip(row, x)), Fraction())
            wanted = exact(rhs[i]) - exact(intercept) - exact(shift) * exact(x[i]) - score
            self.assertLessEqual(abs(exact(scores[i]) - score), exact(sb[i]))
            self.assertLessEqual(abs(exact(residual[i]) - wanted), exact(rb[i]))
        wanted_constraint = exact(target) - sum(map(exact, x), Fraction())
        self.assertLessEqual(abs(exact(constraint) - wanted_constraint), exact(cb))

    def assert_special_decline(self, args):
        before = [value.copy() if isinstance(value, np.ndarray) else value for value in args]
        self.assertIs(compiled._compiled_values_or_refusal(*core_args(args)), compiled._COMPILED_DOMAIN_REFUSAL)
        self.assertIsNone(compiled.compiled_values(*core_args(args)))
        result, profile = self.profiled(args)
        self.assertIsNone(result)
        self.assertEqual(profile['counts']['calls'], 1)
        self.assertEqual(profile['counts']['declined'], 1)
        self.assertEqual(profile['counts']['raised'], 0)
        self.assertEqual(profile['counts']['scalar_sumprod_calls_started'], 0)
        self.assertEqual(profile['counts']['scalar_sumprod_calls_completed'], 0)
        self.assertEqual(profile['decline_reasons'], {'compiled_operand_domain': 1})
        self.assertEqual(profile['compiled_helper']['declined'], 1)
        self.assertEqual(profile['routes']['scalar_fallback']['calls'], 0)
        for a, b in zip(args, before):
            np.testing.assert_array_equal(a, b)
        return profile

    def assert_generic_scalar_contract(self, args):
        result, profile = self.profiled(args)
        baseline, _ = self.generic_baseline(args)
        self.assertNotIn('compiled_operand_domain', profile['decline_reasons'])
        if result is None or baseline is None:
            self.assertIsNone(result)
            self.assertIsNone(baseline)
        else:
            self.same_values(result, baseline)
        if self.scalar_available:
            self.assertEqual(profile['routes']['scalar_fallback']['calls'], 1)
        else:
            self.assertEqual(profile['counts']['scalar_sumprod_calls_started'], 0)
        return result, profile


class CompiledDomainRefusalTests(_RefusalAssertions, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if compiled._ACCEL is None or not isinstance(
                getattr(compiled._ACCEL, 'evaluate', None), types.BuiltinFunctionType):
            raise unittest.SkipTest('Optional compiled residual extension unavailable.')
        if not native._native_supported():
            raise unittest.SkipTest('Requires an actually qualified native residual runtime.')
        if compiled.compiled_values is not compiled._DEFAULT_COMPILED_VALUES:
            raise AssertionError('Original public compiled dispatcher was not loaded')
        # Avoid turning saved Python functions into TestCase bound methods.
        cls.original_evaluate = staticmethod(compiled._ACCEL.evaluate)
        cls.original_public = staticmethod(compiled.compiled_values)
        cls.original_guard = staticmethod(native._safe_operand)
        cls.scalar_available = native._scalar_supported()

    def test_actual_core_bad_rows_early_middle_late_and_threaded_partitions(self):
        # n=257 exercises chunking; real 2/4 worker partitions may cancel their
        # siblings. No assumption is made about completion order or call count.
        for workers, row in ((1, 0), (1, 128), (1, 256), (2, 128), (4, 256)):
            with self.subTest(workers=workers, row=row), patch.object(compiled, '_worker_count', return_value=workers):
                args = fixture(257)
                args[0][row, 0] = 2.**-220
                self.assert_special_decline(args)
                baseline, old_profile = self.generic_baseline(args)
                self.assertIsNone(baseline)
                if self.scalar_available:
                    self.assertEqual(old_profile['counts']['scalar_sumprod_calls_completed'], 2 * row)
                actual_portable = self.completed_or_portable(args, None)
                reference = portable.compensated_residual(*args)
                self.same_values(actual_portable, reference)

    def test_domain_refusal_boundary_nextafter_and_nonfinite_inputs(self):
        bad = (np.nextafter(native._MIN_OPERAND, 0.),
               np.nextafter(native._MAX_OPERAND, math.inf),
               -np.nextafter(native._MIN_OPERAND, 0.),
               -np.nextafter(native._MAX_OPERAND, math.inf), math.nan, math.inf, -math.inf)
        for value in bad:
            with self.subTest(value=value):
                args = fixture()
                args[0][7, 3] = value
                self.assert_special_decline(args)

    def test_safe_boundary_zero_and_signed_zero_preserve_success_and_bounds(self):
        values = (native._MIN_OPERAND, np.nextafter(native._MIN_OPERAND, math.inf),
                  native._MAX_OPERAND, np.nextafter(native._MAX_OPERAND, 0.), 0., -0.)
        for value in values:
            with self.subTest(value=value):
                args = fixture()
                args[0][7, 3] = value
                detailed = compiled._compiled_values_or_refusal(*core_args(args))
                self.assertIsInstance(detailed, tuple)
                self.same_values(detailed, compiled.compiled_values(*core_args(args)))
                result, profile = self.profiled(args)
                baseline, _ = self.generic_baseline(args)
                self.same_values(result, baseline)
                self.assert_bounds(args, result)
                self.assertNotIn('compiled_operand_domain', profile['decline_reasons'])

    def test_f_order_and_negative_strides_keep_refusal_and_success_contracts(self):
        for layout in ('f', 'negative'):
            with self.subTest(layout=layout):
                args = fixture(32)
                if layout == 'f':
                    args[0] = np.asfortranarray(args[0])
                else:
                    args[0] = args[0][::-1, ::-1]
                    args[2] = args[2][::-1]
                    args[3] = args[3][::-1]
                result, _ = self.profiled(args)
                self.assert_bounds(args, result)
                old = args[0][15, 4]
                args[0][15, 4] = 2.**-220
                self.assert_special_decline(args)
                args[0][15, 4] = old
                next_result, _ = self.profiled(args)
                self.same_values(result, next_result)

    def test_no_status_cache_survives_input_mutation_or_success(self):
        args = fixture()
        first = compiled._compiled_values_or_refusal(*core_args(args))
        old = args[0][12, 2]
        args[0][12, 2] = 2.**-220
        self.assert_special_decline(args)
        args[0][12, 2] = old
        final = compiled._compiled_values_or_refusal(*core_args(args))
        self.same_values(first, final)
        for a in first:
            for b in final:
                self.assertFalse(np.shares_memory(a, b))
        for arrays in (first, final):
            for a in arrays:
                for source in (args[0], args[2], args[3]):
                    self.assertFalse(np.shares_memory(a, source))

    def test_replaced_imported_public_dispatcher_retains_generic_fallback(self):
        for replacement in (lambda *values: None, lambda *values: self.original_public(*values)):
            with self.subTest(replacement=replacement), patch.object(native, 'compiled_values', replacement):
                args = fixture()
                # The first replacement declines a valid matrix; the second
                # follows the original public dispatcher on an invalid matrix.
                args[0][7, 3] = 2.**-220
                _, profile = self.profiled(args)
                self.assertNotIn('compiled_operand_domain', profile['decline_reasons'])
                if self.scalar_available:
                    self.assertEqual(profile['counts']['scalar_sumprod_calls_completed'], 14)

    def test_fake_status_two_is_not_trusted_and_valid_scalar_result_remains(self):
        args = fixture()
        with patch.object(compiled._ACCEL, 'evaluate', lambda *values: 2):
            self.assertIsNone(compiled._compiled_values_or_refusal(*core_args(args)))
            result, profile = self.assert_generic_scalar_contract(args)
        if self.scalar_available:
            self.assert_bounds(args, result)
            self.assertEqual(profile['counts']['scalar_sumprod_calls_completed'], 32)

    def test_other_mocked_refusal_statuses_keep_legacy_scalar_fallback(self):
        for status in (1, 3, 4, 37):
            with self.subTest(status=status), patch.object(compiled._ACCEL, 'evaluate', lambda *values: status):
                args = fixture()
                self.assertIsNone(compiled._compiled_values_or_refusal(*core_args(args)))
                self.assertIsNone(compiled.compiled_values(*core_args(args)))
                result, profile = self.assert_generic_scalar_contract(args)
                if self.scalar_available:
                    self.assert_bounds(args, result)
                    self.assertEqual(profile['counts']['scalar_sumprod_calls_completed'], 32)
                else:
                    self.assertIsNone(result)
                self.same_values(self.completed_or_portable(args, result)[:3],
                                 portable.compensated_residual(*args)[:3])

    def test_wrapped_original_evaluator_does_not_authorize_short_circuit(self):
        args = fixture()
        args[0][7, 3] = 2.**-220
        original = self.original_evaluate
        with patch.object(compiled._ACCEL, 'evaluate', lambda *values: original(*values)):
            self.assertIsNone(compiled._compiled_values_or_refusal(*core_args(args)))
            _, profile = self.assert_generic_scalar_contract(args)
        if self.scalar_available:
            self.assertEqual(profile['counts']['scalar_sumprod_calls_completed'], 14)

    def test_replaced_accelerator_module_even_with_original_builtin_is_untrusted(self):
        args = fixture()
        args[0][7, 3] = 2.**-220
        substitute = types.SimpleNamespace(evaluate=self.original_evaluate)
        with patch.object(compiled, '_ACCEL', substitute):
            self.assertIsNone(compiled._compiled_values_or_refusal(*core_args(args)))
            self.assert_generic_scalar_contract(args)

    def test_modified_safe_operand_function_and_limits_keep_generic_semantics(self):
        for attribute, value in (('_safe_operand', lambda values: self.original_guard(values)),
                                 ('_MIN_OPERAND', 2.**-240), ('_MAX_OPERAND', 2.**220)):
            with self.subTest(attribute=attribute), patch.object(native, attribute, value):
                args = fixture()
                args[0][7, 3] = 2.**210 if attribute == '_MAX_OPERAND' else 2.**-220
                result, profile = self.assert_generic_scalar_contract(args)
                if self.scalar_available:
                    self.assertGreater(profile['counts']['scalar_sumprod_calls_completed'], 0)

    def test_ndarray_subclass_scalar_iteration_remains_in_control(self):
        args = fixture()
        iterated = args[0].copy()
        storage = args[0].copy()
        storage[7, 3] = 2.**-220
        args[0] = IteratedRows(storage, iterated)
        # The C buffer and Python row iterator intentionally disagree. This is
        # a dispatch-boundary contract, not a claim these are the same kernel.
        # The internal core can truthfully report its buffer-domain refusal;
        # only the native dispatch boundary knows that scalar iteration differs.
        # Its public tuple-or-None contract remains unchanged.
        self.assertIsNone(compiled.compiled_values(*core_args(args)))
        result, profile = self.assert_generic_scalar_contract(args)
        if self.scalar_available:
            self.assertIsNotNone(result)
            self.assertEqual(profile['counts']['scalar_sumprod_calls_completed'], 32)
            self.assert_bounds(args, result)

    def test_tiny_problem_and_missing_helper_are_generic_refusals(self):
        for tiny, missing in ((True, False), (False, True)):
            args = fixture(3 if tiny else 16)
            with self.subTest(tiny=tiny, missing=missing), ExitStack() as stack:
                if missing:
                    stack.enter_context(patch.object(compiled, '_ACCEL', None))
                self.assertIsNone(compiled._compiled_values_or_refusal(*core_args(args)))
                self.assertIsNone(compiled.compiled_values(*core_args(args)))
                result, profile = self.profiled(args)
                self.assertNotIn('compiled_operand_domain', profile['decline_reasons'])
                if self.scalar_available:
                    self.assert_bounds(args, result)
                    self.assertEqual(profile['counts']['scalar_sumprod_calls_completed'], 2 * len(args[3]))
                else:
                    self.assertIsNone(result)
                    self.assertEqual(profile['counts']['scalar_sumprod_calls_started'], 0)

    def test_partial_output_arrays_are_discarded_not_exposed_or_cached(self):
        outputs, lock = [], Lock()

        def partial_failure(*values):
            for output in values[7:10]:
                output[:] = -999.
                with lock:
                    outputs.append(weakref.ref(output))
            return 2

        args = fixture(257)
        originals = [args[i].copy() for i in (0, 2, 3)]
        with patch.object(compiled, '_worker_count', return_value=4), patch.object(compiled._ACCEL, 'evaluate', partial_failure):
            self.assertIsNone(compiled._compiled_values_or_refusal(*core_args(args)))
            self.assertIsNone(compiled.compiled_values(*core_args(args)))
        gc.collect()
        self.assertTrue(outputs)
        self.assertTrue(all(reference() is None for reference in outputs))
        for index, original in zip((0, 2, 3), originals):
            np.testing.assert_array_equal(args[index], original)
        result, _ = self.profiled(args)
        self.assertIsNotNone(result)
        self.assertFalse(np.any(result[0] == -999.))

    def test_arbitrary_exceptions_propagate_with_original_identity(self):
        for workers in (1, 2, 4):
            failure = RuntimeError('intentional evaluator failure')

            def fail(*values):
                raise failure

            with self.subTest(workers=workers), patch.object(compiled, '_worker_count', return_value=workers), patch.object(compiled._ACCEL, 'evaluate', fail):
                with self.assertRaises(RuntimeError) as captured:
                    compiled._compiled_values_or_refusal(*core_args(fixture(257)))
                self.assertIs(captured.exception, failure)
                with residual_profile() as profile:
                    with self.assertRaises(RuntimeError) as captured:
                        native.native_compensated_residual(*fixture(257))
                self.assertIs(captured.exception, failure)
                self.assertEqual(profile.as_dict()['counts']['raised'], 1)
                self.assertEqual(profile.as_dict()['compiled_helper']['raised'], 1)

    def test_interruption_joins_owned_workers_and_does_not_return_partial_arrays(self):
        for workers in (2, 4):
            active, lock, started = set(), Lock(), Event()
            interrupted = KeyboardInterrupt('intentional controller interruption')
            original = self.original_evaluate

            def evaluate(*values):
                token = object()
                with lock:
                    active.add(token)
                started.set()
                try:
                    return original(*values)
                finally:
                    with lock:
                        active.remove(token)

            def interrupt_wait(*args, **kwargs):
                self.assertTrue(started.wait(timeout=5), 'No actual worker entered the native evaluator')
                raise interrupted

            with self.subTest(workers=workers), patch.object(compiled, '_worker_count', return_value=workers), patch.object(compiled._ACCEL, 'evaluate', evaluate), patch.object(compiled, 'wait', interrupt_wait):
                with residual_profile() as profile:
                    with self.assertRaises(KeyboardInterrupt) as captured:
                        native.native_compensated_residual(*fixture(257))
                self.assertIs(captured.exception, interrupted)
                self.assertEqual(profile.as_dict()['counts']['raised'], 1)
            self.assertFalse(active, 'An owned evaluator remained active after interrupted dispatch')

    def test_independent_concurrent_success_and_refusal_are_call_local(self):
        valid, invalid = fixture(32), fixture(32)
        invalid[0][15, 3] = 2.**-220
        originals = {name: [args[i].copy() for i in (0, 2, 3)]
                     for name, args in (('valid', valid), ('invalid', invalid))}
        expected, _ = self.profiled(valid)
        self.assertIsNotNone(expected)
        start_each_call = Barrier(2)

        def invoke(name, args):
            records = []
            for _ in range(3):
                start_each_call.wait(timeout=5)
                value, profile = self.profiled(args)
                records.append((value, profile))
            return name, records

        # These are independent calls using the real builtin evaluator. Each
        # small call naturally uses one core worker; no trusted hook is wrapped.
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(invoke, name, args)
                       for name, args in (('valid', valid), ('invalid', invalid))]
            collected = dict(future.result() for future in futures)
        for result, profile in collected['valid']:
            self.same_values(result, expected)
            self.assert_bounds(valid, result)
            self.assertEqual(profile['counts']['calls'], 1)
            self.assertEqual(profile['counts']['bounded_return'], 1)
            self.assertEqual(profile['counts']['declined'], 0)
            self.assertEqual(profile['counts']['scalar_sumprod_calls_completed'], 0)
            self.assertEqual(profile['decline_reasons'], {})
        for result, profile in collected['invalid']:
            self.assertIsNone(result)
            self.assertEqual(profile['counts']['calls'], 1)
            self.assertEqual(profile['counts']['bounded_return'], 0)
            self.assertEqual(profile['counts']['declined'], 1)
            self.assertEqual(profile['counts']['scalar_sumprod_calls_completed'], 0)
            self.assertEqual(profile['decline_reasons'], {'compiled_operand_domain': 1})
        for name, args in (('valid', valid), ('invalid', invalid)):
            for index, original in zip((0, 2, 3), originals[name]):
                np.testing.assert_array_equal(args[index], original)

    def test_peer_exception_wins_over_trusted_domain_refusal_after_join(self):
        # Keep the actual builtin evaluator untouched. The first executor task
        # obtains a genuine trusted status 2; a peer future then raises. This
        # ensures the sentinel cannot hide an exception from another worker.
        original_executor = compiled.ThreadPoolExecutor
        for entry in ('private', 'public', 'native'):
            with self.subTest(entry=entry):
                args = fixture(257)
                args[0][0, 0] = 2.**-220
                failure = RuntimeError('intentional peer task failure after trusted refusal')
                trusted_done, lock = Event(), Lock()
                trusted_results, active, finished = [], set(), []

                class PeerFailureExecutor:
                    def __init__(self, *values, **keywords):
                        self.inner = original_executor(*values, **keywords)

                    def __enter__(self):
                        self.inner.__enter__()
                        return self

                    def submit(self, function, part):
                        def run():
                            with lock:
                                active.add(part)
                            try:
                                if part == 0:
                                    try:
                                        value = function(part)
                                        with lock:
                                            trusted_results.append(value)
                                        return value
                                    finally:
                                        trusted_done.set()
                                if not trusted_done.wait(timeout=5):
                                    raise AssertionError('Trusted peer did not finish')
                                # Its pending chunks may already be cancelled.
                                # A raised future still takes priority over a
                                # numerical refusal returned by the first peer.
                                function(part)
                                raise failure
                            finally:
                                with lock:
                                    active.remove(part)
                                    finished.append(part)
                        return self.inner.submit(run)

                    def __exit__(self, *values):
                        return self.inner.__exit__(*values)

                with patch.object(compiled, '_worker_count', return_value=2), patch.object(compiled, 'ThreadPoolExecutor', PeerFailureExecutor):
                    self.assertIs(compiled._ACCEL.evaluate, self.original_evaluate)
                    with residual_profile() as profile:
                        with self.assertRaises(RuntimeError) as captured:
                            if entry == 'private':
                                compiled._compiled_values_or_refusal(*core_args(args))
                            elif entry == 'public':
                                compiled.compiled_values(*core_args(args))
                            else:
                                native.native_compensated_residual(*args)
                    self.assertIs(captured.exception, failure)
                self.assertEqual(len(trusted_results), 1)
                self.assertIs(trusted_results[0], compiled._COMPILED_DOMAIN_REFUSAL)
                self.assertFalse(active, 'Owned task remained active after exception')
                self.assertCountEqual(finished, [0, 1])
                if entry == 'native':
                    self.assertEqual(profile.as_dict()['counts']['raised'], 1)
                    self.assertEqual(profile.as_dict()['compiled_helper']['raised'], 1)
                    self.assertEqual(profile.as_dict()['counts']['scalar_sumprod_calls_started'], 0)

    def test_malformed_inputs_keep_pre_dispatch_refusal_contract(self):
        cases = []
        for replacement in (fixture()[0].astype(np.float32), fixture()[0].tolist(),
                            np.ones((16, 16, 1)), np.ones((16, 15))):
            args = fixture()
            args[0] = replacement
            cases.append(args)
        for index, replacement in ((2, np.ones(15)), (3, np.ones((16, 1))),
                                   (3, np.array(['not-a-number'] * 16, dtype=object))):
            args = fixture()
            args[index] = replacement
            cases.append(args)
        for index, args in enumerate(cases):
            with self.subTest(case=index):
                actual, profile = self.profiled(args)
                baseline, old_profile = self.generic_baseline(args)
                self.assertIsNone(actual)
                self.assertIsNone(baseline)
                self.assertEqual(profile['decline_reasons'], old_profile['decline_reasons'])
                self.assertNotIn('compiled_operand_domain', profile['decline_reasons'])
                self.assertEqual(profile['compiled_helper']['calls'], 0)
                self.assertEqual(profile['counts']['scalar_sumprod_calls_started'], 0)

    def test_handled_arithmetic_exception_retains_original_decline_reason(self):
        def fail(*values):
            raise FloatingPointError('intentional arithmetic refusal')

        with patch.object(compiled._ACCEL, 'evaluate', fail):
            result, profile = self.profiled(fixture())
        self.assertIsNone(result)
        self.assertEqual(profile['decline_reasons'], {'arithmetic_exception': 1})
        self.assertEqual(profile['compiled_helper']['raised'], 1)
        self.assertEqual(profile['counts']['scalar_sumprod_calls_started'], 0)


class GenericRefusalDispatchTests(_RefusalAssertions, unittest.TestCase):
    """Exercise generic dispatch even on portable wheels and Python 3.11."""

    @classmethod
    def setUpClass(cls):
        cls.original_public = staticmethod(compiled._DEFAULT_COMPILED_VALUES)
        cls.scalar_available = native._scalar_supported()

    def test_missing_helper_preserves_scalar_or_portable_equations(self):
        args = fixture()
        with patch.object(compiled, '_ACCEL', None):
            self.assertIsNone(compiled._compiled_values_or_refusal(*core_args(args)))
            self.assertIsNone(compiled.compiled_values(*core_args(args)))
            result, profile = self.profiled(args)
        self.assertNotIn('compiled_operand_domain', profile['decline_reasons'])
        reference = portable.compensated_residual(*args)
        K, shift, rhs, x, intercept, target = args
        for row, score, residual, right, coefficient in zip(
                K, reference[2], reference[0], rhs, x):
            exact_score = sum((exact(a) * exact(b) for a, b in zip(row, x)), Fraction())
            exact_residual = (exact(right) - exact(intercept)
                              - exact(shift) * exact(coefficient) - exact_score)
            self.assertEqual(score, float(exact_score))
            self.assertEqual(residual, float(exact_residual))
        self.assertEqual(reference[1], float(exact(target) - sum(map(exact, x), Fraction())))
        if self.scalar_available:
            self.assert_bounds(args, result)
            self.same_values(result[:3], reference[:3])
            self.assertEqual(profile['counts']['scalar_sumprod_calls_completed'], 32)
        else:
            self.assertIsNone(result)
            self.assertEqual(profile['counts']['scalar_sumprod_calls_started'], 0)
            self.assertEqual(profile['decline_reasons'], {'runtime_or_rounding': 1})

    def test_tiny_problem_never_calls_an_available_evaluator(self):
        args = fixture(3)
        evaluator = Mock(side_effect=AssertionError('Tiny dispatch reached compiled core'))
        with patch.object(compiled, '_ACCEL', types.SimpleNamespace(evaluate=evaluator)):
            self.assertIsNone(compiled._compiled_values_or_refusal(*core_args(args)))
            self.assertIsNone(compiled.compiled_values(*core_args(args)))
        evaluator.assert_not_called()

    def test_synthetic_refusal_statuses_are_generic_without_a_real_helper(self):
        args = fixture()
        for status in (1, 2, 3, 4, 37):
            with self.subTest(status=status):
                evaluator = Mock(return_value=status)
                with patch.object(compiled, '_ACCEL', types.SimpleNamespace(evaluate=evaluator)):
                    self.assertIsNone(compiled._compiled_values_or_refusal(*core_args(args)))
                    self.assertIsNone(compiled.compiled_values(*core_args(args)))
                    self.assertEqual(evaluator.call_count, 2)
                    result, profile = self.assert_generic_scalar_contract(args)
                self.assertNotIn('compiled_operand_domain', profile['decline_reasons'])
                if self.scalar_available:
                    self.assert_bounds(args, result)
                    self.assertEqual(profile['counts']['scalar_sumprod_calls_completed'], 32)
                    self.same_values(result[:3], portable.compensated_residual(*args)[:3])
                else:
                    self.assertIsNone(result)

    def test_public_dispatcher_hides_private_refusal_detail(self):
        args = fixture()
        with patch.object(compiled, '_compiled_values_or_refusal',
                          return_value=compiled._COMPILED_DOMAIN_REFUSAL) as detail:
            self.assertIsNone(compiled.compiled_values(*core_args(args)))
        detail.assert_called_once()

    def test_replaced_public_dispatcher_keeps_control_at_native_boundary(self):
        args = fixture()
        K, shift, rhs, x, intercept, _ = args
        replacement = Mock(return_value=None)
        with patch.object(native, 'compiled_values', replacement), patch.object(
                compiled, '_compiled_values_or_refusal',
                side_effect=AssertionError('Replacement dispatcher bypassed')):
            self.assertIsNone(native._dispatch_compiled_values(K, x, rhs, shift, intercept))
        replacement.assert_called_once()

    def test_specialized_dispatch_requires_original_operand_contract(self):
        args = fixture()
        K, shift, rhs, x, intercept, _ = args
        refusal = compiled._COMPILED_DOMAIN_REFUSAL
        with patch.object(compiled, '_compiled_values_or_refusal', return_value=refusal):
            self.assertIs(native._dispatch_compiled_values(K, x, rhs, shift, intercept), refusal)
            subclass = IteratedRows(K, K.copy())
            self.assertIsNone(native._dispatch_compiled_values(subclass, x, rhs, shift, intercept))
            guard = native._safe_operand
            for attribute, replacement in (
                    ('_safe_operand', lambda values: guard(values)),
                    ('_MIN_OPERAND', 2.**-240), ('_MAX_OPERAND', 2.**220)):
                with self.subTest(attribute=attribute), patch.object(native, attribute, replacement):
                    self.assertIsNone(native._dispatch_compiled_values(K, x, rhs, shift, intercept))

    def test_replaced_evaluator_exception_is_not_a_domain_refusal(self):
        failure = RuntimeError('intentional evaluator failure')
        evaluator = Mock(side_effect=failure)
        with patch.object(compiled, '_ACCEL', types.SimpleNamespace(evaluate=evaluator)):
            for entry in (compiled._compiled_values_or_refusal, compiled.compiled_values):
                with self.subTest(entry=entry.__name__):
                    with self.assertRaises(RuntimeError) as captured:
                        entry(*core_args(fixture()))
                    self.assertIs(captured.exception, failure)

    def test_unqualified_actual_runtime_never_dispatches(self):
        if native._native_supported():
            self.skipTest('This interpreter has a qualified native residual route.')
        with patch.object(native, '_dispatch_compiled_values',
                          side_effect=AssertionError('Unqualified runtime dispatched')):
            result, profile = self.profiled(fixture())
        self.assertIsNone(result)
        self.assertEqual(profile['decline_reasons'], {'runtime_or_rounding': 1})
        self.assertEqual(profile['compiled_helper']['calls'], 0)
        self.assertEqual(profile['counts']['scalar_sumprod_calls_started'], 0)


if __name__ == '__main__':
    unittest.main()
