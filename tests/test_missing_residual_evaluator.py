"""Missing optional entrypoints decline; callable failures remain observable."""
import threading
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import dwd._compiled_residual as compiled
import dwd._native_residual as native


MISSING_KINDS = ("module-missing", "namespace-missing", "module-none", "namespace-none")


def missing_accelerator(kind):
    owner = ModuleType("missing_residual_evaluator") if kind.startswith("module") else SimpleNamespace()
    if kind.endswith("none"):
        owner.evaluate = None
    return owner


def inputs(n):
    # This represents the large shape without allocating an n-by-n matrix.
    matrix = np.broadcast_to(np.array(1.0), (n, n))
    vector = np.ones(n)
    right = np.arange(n, dtype=float) * 0.25
    return matrix, vector, right


def forbidden(*args, **kwargs):
    raise AssertionError("A missing evaluator must decline before allocation or pool changes")


def call_entry(entry, matrix, vector, right):
    if entry == "compiled":
        return compiled.compiled_values(matrix, vector, right, 0.5, 0.25)
    return native.native_compensated_residual(matrix, 0.5, right, vector, 0.25, float(len(vector)))


@pytest.mark.parametrize("kind", MISSING_KINDS)
@pytest.mark.parametrize("n", (8, 256, 2048))
@pytest.mark.parametrize("entry", ("compiled_values", "_compiled_values_or_refusal"))
def test_missing_evaluator_declines_before_allocation_or_pool_changes(monkeypatch, kind, n, entry):
    matrix, vector, right = inputs(n)
    monkeypatch.setattr(compiled, "_ACCEL", missing_accelerator(kind))
    # Replace only this module's numpy reference, not the process-wide numpy API.
    monkeypatch.setattr(compiled, "np", SimpleNamespace(empty=forbidden))
    monkeypatch.setattr(compiled, "_worker_count", forbidden)
    monkeypatch.setattr(compiled, "retire_for_serial_call", forbidden)
    monkeypatch.setattr(compiled, "executor_for_call", forbidden)
    monkeypatch.setattr(compiled, "ThreadPoolExecutor", forbidden)
    assert getattr(compiled, entry)(matrix, vector, right, 0.5, 0.25) is None


@pytest.mark.parametrize("kind", MISSING_KINDS)
def test_native_facade_uses_identical_existing_scalar_fallback(monkeypatch, kind):
    # Use the actual qualified scalar implementation; do not widen eligibility.
    if not native._scalar_supported():
        pytest.skip("Actual runtime lacks qualified scalar sumprod")
    matrix, vector, right = inputs(8)
    monkeypatch.setattr(compiled, "_worker_count", forbidden)
    monkeypatch.setattr(compiled, "retire_for_serial_call", forbidden)
    monkeypatch.setattr(compiled, "executor_for_call", forbidden)
    original_dispatch = native._dispatch_compiled_values
    dispatch_results = []

    def dispatch(*args):
        result = original_dispatch(*args)
        dispatch_results.append(result)
        return result

    monkeypatch.setattr(native, "_dispatch_compiled_values", dispatch)
    monkeypatch.setattr(compiled, "_ACCEL", None)
    expected = call_entry("native", matrix, vector, right)
    monkeypatch.setattr(compiled, "_ACCEL", missing_accelerator(kind))
    actual = call_entry("native", matrix, vector, right)
    assert dispatch_results == [None, None]
    assert expected is not None and actual is not None
    assert len(actual) == len(expected) == 6
    for got, want in zip(actual, expected):
        assert np.asarray(got).dtype == np.asarray(want).dtype == np.dtype(float)
        np.testing.assert_array_equal(np.asarray(got).view(np.uint64), np.asarray(want).view(np.uint64))
    np.testing.assert_array_equal(actual[0], right - 8.75)
    np.testing.assert_array_equal(actual[2], np.full(8, 8.0))
    assert actual[1] == 0.0


@pytest.mark.parametrize("kind", MISSING_KINDS)
def test_missing_evaluator_with_unsupported_scalar_declines_for_portable(monkeypatch, kind):
    matrix, vector, right = inputs(8)
    monkeypatch.setattr(native, "_scalar_supported", lambda: False)
    monkeypatch.setattr(compiled, "_worker_count", forbidden)
    monkeypatch.setattr(compiled, "executor_for_call", forbidden)
    monkeypatch.setattr(compiled, "_ACCEL", None)
    assert call_entry("native", matrix, vector, right) is None
    monkeypatch.setattr(compiled, "_ACCEL", missing_accelerator(kind))
    assert call_entry("native", matrix, vector, right) is None


@pytest.mark.parametrize("workers", (1, 2))
def test_non_none_noncallable_keeps_low_level_type_error(monkeypatch, workers):
    matrix, vector, right = inputs(16)
    monkeypatch.setattr(compiled, "_ACCEL", SimpleNamespace(evaluate=17))
    monkeypatch.setattr(compiled, "_worker_count", lambda n: workers)
    with pytest.raises(TypeError, match="not callable"):
        call_entry("compiled", matrix, vector, right)


@pytest.mark.parametrize("entry", ("compiled", "native"))
@pytest.mark.parametrize("error_type", (AttributeError, RuntimeError, KeyboardInterrupt))
@pytest.mark.parametrize("workers", (1, 2))
def test_callable_failure_preserves_original_exception_and_retires_workers(
        monkeypatch, entry, error_type, workers):
    if entry == "native" and not native._native_supported():
        pytest.skip("Actual runtime lacks a qualified native residual route")
    matrix, vector, right = inputs(16)
    error = error_type("failure inside a present evaluator")
    observed_threads = []
    entered = []

    def evaluate(K, x, rhs, shift, intercept, start, stop, scores, residual, maxima):
        observed_threads.append(threading.current_thread())
        entered.append((start, stop))
        # Raise one object from one partition, not concurrently from two threads.
        if start == 0:
            raise error
        scores[:] = 16.0
        residual[:] = rhs[start:stop] - 16.75
        maxima[:] = 1.0
        return 0

    monkeypatch.setattr(compiled, "_ACCEL", SimpleNamespace(evaluate=evaluate))
    monkeypatch.setattr(compiled, "_worker_count", lambda n: workers)
    # The real executor/context and cancellation/cleanup functions stay intact.
    with pytest.raises(error_type) as captured:
        call_entry(entry, matrix, vector, right)
    assert captured.value is error
    assert entered and any(start == 0 for start, _ in entered)
    if workers == 2:
        assert all(thread is not threading.current_thread() and not thread.is_alive()
                   for thread in observed_threads)
    else:
        assert observed_threads == [threading.current_thread()]


@pytest.mark.parametrize("workers", (1, 2))
def test_present_callable_preserves_outputs_and_exact_partition_coverage(monkeypatch, workers):
    matrix, vector, right = inputs(16)
    entered = []
    observed_threads = []

    def evaluate(K, x, rhs, shift, intercept, start, stop, scores, residual, maxima):
        assert K is matrix and x is vector and rhs is right
        assert shift == 0.5 and intercept == 0.25
        entered.append((start, stop))
        observed_threads.append(threading.current_thread())
        scores[:] = 16.0
        residual[:] = rhs[start:stop] - 16.75
        maxima[:] = 1.0
        return 0

    monkeypatch.setattr(compiled, "_ACCEL", SimpleNamespace(evaluate=evaluate))
    monkeypatch.setattr(compiled, "_worker_count", lambda n: workers)
    actual = call_entry("compiled", matrix, vector, right)
    assert actual is not None and len(actual) == 3
    np.testing.assert_array_equal(actual[0], np.full(16, 16.0))
    np.testing.assert_array_equal(actual[1], right - 16.75)
    np.testing.assert_array_equal(actual[2], np.ones(16))
    assert sorted(entered) == ([(0, 16)] if workers == 1 else [(0, 8), (8, 16)])
    if workers == 2:
        assert all(thread is not threading.current_thread() and not thread.is_alive()
                   for thread in observed_threads)
    else:
        assert observed_threads == [threading.current_thread()]
