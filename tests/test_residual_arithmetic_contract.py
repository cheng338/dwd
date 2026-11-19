"""Finite exact-oracle validation of residual arithmetic and solver decisions.

The historical eight-term fixture is stored verbatim as hexadecimal binary64
values; no random search or archived probe execution is needed. Computation is
tiny and serial. Actual-runtime tests distinguish their build from the FMA model.
"""
import math

import numpy as np
import pytest

from dwd import _compiled_residual as compiled
from dwd import _native_residual as native
from dwd import _kernel_linear_system as linear_system
from dwd._compensated_residual import compensated_residual
from dwd._kernel_linear_system import KernelLinearSystem

from residual_arithmetic_reference import (
    _dekker_product, _fma_product,
    assert_compiled_values_bounded, assert_public_result_bounded,
    dekker_sumprod, exact, exact_sumprod, fma_sumprod, reference_rows,
)


ROW_HEX = (
    '-0x1.4057356b9bf42p-29', '-0x1.e3b668cb237d8p-36',
    '-0x1.ec5ef23aa8118p+7', '-0x1.60bf8fcda12c0p-36',
    '-0x1.7f1cf419648acp-9', '-0x1.98df1524a7c70p-14',
    '0x1.827a2464310ddp+29', '-0x1.923c6cc80e455p-29',
)
X_HEX = (
    '0x1.d32bdd093f266p+21', '0x1.a334986a78574p-4',
    '0x1.a6d03f9da0dc6p+19', '0x1.6384daf8208afp-6',
    '0x1.7ca51b1cf7ec2p+37', '0x1.336f7fba992c8p+25',
    '0x1.0000000000000p+0', '0x1.0000000000000p+0',
)
ROUTES = ('compiled', 'actual_scalar', 'simulated_fma')


def test_fma_product_preserves_signed_zero_high():
    checked = 0
    for zero in (0.0, -0.0):
        for operand in (0.0, -0.0, 1.0, -1.0, 2.**-200, -2.**-200, 2.**200, -2.**200):
            for a, b in ((zero, operand), (operand, zero)):
                high, low = _fma_product(a, b)
                assert high == low == 0.0
                assert math.copysign(1.0, high) == math.copysign(1.0, a * b)
                # An exact product plus its opposite-signed zero gives +0
                # under the round-to-nearest mode required by these models.
                assert math.copysign(1.0, low) == 1.0
                assert exact(high) + exact(low) == exact(a) * exact(b)
                checked += 1
    assert checked == 32


def test_product_expansions_are_exact_at_guarded_endpoints_and_neighbors():
    pairs = (
        (2.**-200, 2.**-200), (-2.**-200, 2.**200),
        (2.**200, 2.**200), (-2.**200, -2.**200),
        (math.nextafter(2.**-200, math.inf), math.nextafter(2.**200, 0.)),
        (math.nextafter(1., 2.), math.nextafter(1., 0.)),
        (1. + 2.**-27, 1. - 2.**-27),
        (-math.nextafter(1., 2.), math.nextafter(1., 2.)),
    ) + tuple((float.fromhex(a), float.fromhex(b)) for a, b in zip(ROW_HEX, X_HEX, strict=True))
    assert len(pairs) == 16
    for a, b in pairs:
        for product in (_dekker_product, _fma_product):
            high, low = product(a, b)
            assert math.isfinite(high) and math.isfinite(low)
            assert exact(high) + exact(low) == exact(a) * exact(b)
            if product is _fma_product:
                assert high.hex() == (a * b).hex()


def historical_fixture(*, accepted_score=False):
    row = [float.fromhex(value) for value in ROW_HEX]
    x = [float.fromhex(value) for value in X_HEX]
    if accepted_score:
        row.append(1.0)
        x.append(1.0)
    n = len(x)
    return np.tile(row, (n, 1)), np.asarray(x), np.full(n, float(accepted_score))


def core_values(K, x, rhs, shift=0.0, intercept=0.0):
    if compiled._ACCEL is None:
        pytest.skip('Optional residual extension is absent; no native-core claim.')
    outputs = [np.full(len(x), np.nan) for _ in range(3)]
    status = compiled._ACCEL.evaluate(K, x, rhs, shift, intercept, 0, len(x), *outputs)
    assert status == 0
    return outputs


def select_route(monkeypatch, route):
    if route == 'compiled':
        if compiled._ACCEL is None or not native._native_supported():
            pytest.skip('Actual compiled bounded route is not qualified on this runtime.')
        # The historical public fixture has n=9, so the ordinary dispatcher
        # takes the real compiled path. A scalar fallback is a test failure.
        # Preserve runtime qualification: on Linux it uses the scalar build
        # qualification even when the available compiled core does the work.
        dispatch = native._dispatch_compiled_values
        def require_compiled(*args):
            result = dispatch(*args)
            assert isinstance(result, tuple) and len(result) == 3
            return result
        monkeypatch.setattr(native, '_dispatch_compiled_values', require_compiled)
    elif route == 'actual_scalar':
        if not native._scalar_supported():
            pytest.skip('This actual runtime has no qualified builtin math.sumprod.')
        monkeypatch.setattr(native, 'compiled_values', lambda *args: None)
    else:
        # Explicit algorithm model, not a claim to qualify this runtime/build.
        monkeypatch.setattr(native, '_native_supported', lambda: True)
        monkeypatch.setattr(native, '_scalar_supported', lambda: True)
        monkeypatch.setattr(native, 'compiled_values', lambda *args: None)
        monkeypatch.setattr(native.math, 'sumprod', fma_sumprod, raising=False)


def test_historical_counterexample_models_and_old_bitwise_oracle(record_property):
    K, x, rhs = historical_fixture()
    dekker, fma = dekker_sumprod(K[0], x), fma_sumprod(K[0], x)
    correctly_rounded = float(exact_sumprod(K[0], x))
    assert dekker.hex() == '-0x1.4b60a8638bb80p-83'
    assert fma.hex() == '-0x1.4b60a8638bb60p-83'
    assert correctly_rounded.hex() == '-0x1.4b60a8638bb5dp-83'
    # This is the historical test's invalid expectation, deliberately exercised
    # as a negative control. It is not an expected production failure.
    with pytest.raises(AssertionError):
        assert np.float64(dekker).tobytes() == np.float64(fma).tobytes()
    for model in (dekker_sumprod, fma_sumprod):
        scores, residuals = reference_rows(model, K, x, rhs, 0.0, 0.0)
        assert_compiled_values_bounded(K, x, rhs, 0.0, 0.0, scores, residuals)
    record_property('dekker_hex', dekker.hex())
    record_property('simulated_fma_hex', fma.hex())
    record_property('correctly_rounded_hex', correctly_rounded.hex())


def test_historical_actual_core_matches_fixed_dekker_not_fma(record_property):
    K, x, rhs = historical_fixture()
    scores, residuals, maxima = core_values(K, x, rhs)
    expected = reference_rows(dekker_sumprod, K, x, rhs, 0.0, 0.0)
    for actual, reference in zip((scores, residuals), expected, strict=True):
        assert actual.tobytes() == reference.tobytes()
    np.testing.assert_array_equal(maxima, np.max(np.abs(K), axis=1))
    assert scores[0].hex() != fma_sumprod(K[0], x).hex()
    assert_compiled_values_bounded(K, x, rhs, 0.0, 0.0, scores, residuals)
    if native._scalar_supported():
        actual_scalar = math.sumprod(K[0].tolist(), x.tolist())
        scalar_values = reference_rows(math.sumprod, K, x, rhs, 0.0, 0.0)
        assert_compiled_values_bounded(K, x, rhs, 0.0, 0.0, *scalar_values)
        record_property('actual_builtin_sumprod_hex', actual_scalar.hex())
        record_property('actual_scalar_matches_dekker', actual_scalar.hex() == scores[0].hex())
    record_property('actual_compiled_hex', scores[0].hex())


@pytest.mark.parametrize('route', ROUTES)
def test_historical_difference_inside_accepted_bounded_contract(monkeypatch, route, record_property):
    select_route(monkeypatch, route)
    K, x, rhs = historical_fixture(accepted_score=True)
    before = [a.tobytes() for a in (K, x, rhs)]
    result = native.native_compensated_residual(K, 0.0, rhs, x, 0.0, 0.0)
    assert result is not None
    assert_public_result_bounded(K, 0.0, rhs, x, 0.0, 0.0, result)
    assert np.array_equal(result[2], np.ones(len(x)))
    assert np.all(result[0] != 0.0)
    assert before == [a.tobytes() for a in (K, x, rhs)]
    record_property('route', route)
    record_property('residual_hex', float(result[0][0]).hex())
    record_property('residual_bound', float(result[3][0]))
    exact_residual = exact(rhs[0]) - exact_sumprod(K[0], x)
    record_property('correctly_rounded_residual_hex', float(exact_residual).hex())
    record_property('residual_absolute_error_exact', str(abs(exact(result[0][0]) - exact_residual)))
    record_property('score_hex', float(result[2][0]).hex())
    record_property('score_bound', float(result[5][0]))


@pytest.mark.parametrize('kind', ('ordinary', 'cancellation', 'endpoints', 'strided', 'fortran', 'unaligned'))
def test_core_exact_equations_and_fixed_algorithm_on_finite_cases(kind):
    n = 8
    K = np.asarray([[((i * 7 + j * 3) % 19 - 9) / 16 for j in range(n)] for i in range(n)])
    x = np.asarray([1., -1., .125, -.0625, 3., -2., .5, -.25])
    rhs = np.arange(n, dtype=float) / 8
    shift, intercept = .125, -.25
    if kind == 'cancellation':
        K, x, rhs = historical_fixture()
        shift = intercept = 0.0
    elif kind == 'endpoints':
        values = np.asarray([2.**-200, -2.**-200, 2.**200, -2.**200,
                             0., -0., math.nextafter(1., 2.), math.nextafter(1., 0.)])
        K = np.asarray([np.roll(values, i) for i in range(n)])
        x = values[::-1].copy()
    elif kind == 'strided':
        K, x, rhs = K[::-1, ::-1], x[::-1], rhs[::-1]
    elif kind == 'fortran':
        K = np.asfortranarray(K)
    elif kind == 'unaligned':
        raw = bytearray(K.nbytes + 1)
        unaligned = np.ndarray(K.shape, dtype=np.float64, buffer=raw, offset=1)
        unaligned[:] = K
        K = unaligned
    for a in (K, x, rhs):
        a.flags.writeable = False
    before = [a.tobytes() for a in (K, x, rhs)]
    scores, residuals, maxima = core_values(K, x, rhs, shift, intercept)
    expected = reference_rows(dekker_sumprod, K, x, rhs, shift, intercept)
    for actual, reference in zip((scores, residuals), expected, strict=True):
        assert actual.tobytes() == reference.tobytes()
    assert_compiled_values_bounded(K, x, rhs, shift, intercept, scores, residuals)
    np.testing.assert_array_equal(maxima, np.max(np.abs(K), axis=1))
    assert before == [a.tobytes() for a in (K, x, rhs)]


@pytest.mark.parametrize('scale', (2.**-200, -2.**-200, 2.**200, -2.**200))
@pytest.mark.parametrize('route', ROUTES)
def test_inclusive_domain_endpoints_are_bounded(monkeypatch, route, scale):
    select_route(monkeypatch, route)
    K, x, rhs = np.full((8, 8), scale), np.full(8, scale), np.zeros(8)
    result = native.native_compensated_residual(K, 0.0, rhs, x, 0.0, 0.0)
    assert result is not None
    assert_public_result_bounded(K, 0.0, rhs, x, 0.0, 0.0, result)


@pytest.mark.parametrize('field', ('K', 'x', 'rhs', 'shift', 'intercept', 'target_sum'))
@pytest.mark.parametrize('bad', (math.nextafter(2.**-200, 0.), math.nextafter(2.**200, math.inf),
                               float.fromhex('0x0.0000000000001p-1022'), math.inf, math.nan))
def test_outside_domain_and_nonfinite_values_decline(field, bad):
    if not native._native_supported():
        pytest.skip('Actual bounded native route is not qualified on this runtime.')
    values = dict(K=np.eye(8), shift=.125, rhs=np.ones(8), x=np.ones(8), intercept=.25, target_sum=8.)
    if field in ('K', 'x', 'rhs'):
        values[field].flat[0] = bad
    else:
        values[field] = bad
    args = (values['K'], values['shift'], values['rhs'], values['x'], values['intercept'], values['target_sum'])
    assert native.native_compensated_residual(*args) is None
    if compiled._ACCEL is not None and field != 'target_sum':
        outputs = [np.full(8, np.nan) for _ in range(3)]
        status = compiled._ACCEL.evaluate(values['K'], values['x'], values['rhs'], values['shift'],
                                         values['intercept'], 0, 8, *outputs)
        assert status == 2  # Partial buffers carry no valid-value claim.


def test_subnormal_domain_refusal_retains_portable_exact_equation_bounds():
    K, x, rhs = np.eye(8), np.ones(8), np.ones(8)
    K[0, 1] = float.fromhex('0x0.0000000000001p-1022')
    assert native.native_compensated_residual(K, 0.0, rhs, x, 0.0, 8.0) is None
    residuals, constraint, scores, bounds, constraint_bound = compensated_residual(K, 0.0, rhs, x, 0.0, 8.0)
    for i, row in enumerate(K):
        expected = exact(rhs[i]) - exact_sumprod(row, x)
        assert abs(exact(residuals[i]) - expected) <= exact(bounds[i])
    assert abs(exact(constraint)) <= exact(constraint_bound)
    assert np.isfinite(scores).all()


@pytest.mark.parametrize('route', ROUTES)
def test_severe_score_cancellation_declines_even_for_exact_zero(monkeypatch, route):
    select_route(monkeypatch, route)
    K, x, rhs = np.ones((8, 8)), np.asarray([1e7, -1e7] * 4), np.zeros(8)
    assert exact_sumprod(K[0], x) == 0
    assert native.native_compensated_residual(K, 0.0, rhs, x, 0.0, 0.0) is None


@pytest.mark.parametrize('route', ROUTES)
def test_original_solver_acceptance_and_definite_rejection(monkeypatch, route, record_property):
    select_route(monkeypatch, route)
    n = 8
    K = 2 * np.eye(n) + np.ones((n, n)) / 8
    x = np.arange(1., n + 1) / 16
    shift, intercept, target = .5, .25, 2.25
    # All terms are dyadic and small: this rhs represents the exact equations.
    rhs = 2.5 * x + .28125 + intercept
    for i, row in enumerate(K):
        assert exact(rhs[i]) == exact_sumprod(row, x) + exact(shift) * exact(x[i]) + exact(intercept)
    system = KernelLinearSystem(K, shift)
    good = system._bounded_native_measure(rhs, target, x, intercept)
    assert good is not None and bool(good[0])
    bad_x = x.copy()
    bad_x[0] += .0625
    bad_x[1] -= .0625
    bad = system._bounded_native_measure(rhs, target, bad_x, intercept)
    assert bad is not None and not bool(bad[0])
    exact_maximum = max(abs(exact(rhs[i]) - exact_sumprod(row, bad_x)
                            - exact(shift) * exact(bad_x[i]) - exact(intercept)) for i, row in enumerate(K))
    assert exact_maximum == exact(.15625)
    assert exact_maximum > exact(1e-10 * max(1., float(np.max(np.abs(rhs)))))
    assert system.info['native_residual_acceptances'] == 1
    assert system.info['native_residual_equation_failures'] == 1
    record_property('route', route)
    record_property('exact_bad_maximum', str(exact_maximum))


@pytest.mark.parametrize('route', ROUTES)
def test_actual_uncertain_equation_boundary_uses_unchanged_portable_state(monkeypatch, route, record_property):
    select_route(monkeypatch, route)
    n, shift, intercept, target = 8, 2.**-100, 0.0, 8.0
    K, x, rhs = shift * np.eye(n), np.ones(n), np.full(n, 1e-10)
    before = [a.tobytes() for a in (K, x, rhs)]
    values = native.native_compensated_residual(K, shift, rhs, x, intercept, target)
    assert values is not None
    assert_public_result_bounded(K, shift, rhs, x, intercept, target, values)
    exact_residual = exact(rhs[0]) - exact_sumprod(K[0], x) - exact(shift)
    threshold = exact(1e-10)
    assert exact_residual == threshold - exact(2.**-99)
    assert 0 < exact_residual < threshold
    measured, allowance = exact(values[0][0]), exact(values[3][0])
    assert measured - allowance < threshold < measured + allowance
    system = KernelLinearSystem(K, shift)
    assert system._bounded_native_measure(rhs, target, x, intercept) is None
    original_portable = linear_system.compensated_residual
    calls = []
    def record_portable(actual_K, actual_shift, actual_rhs, actual_x, actual_intercept, actual_target):
        assert actual_K is K and actual_rhs is rhs and actual_x is x
        assert (actual_shift, actual_intercept, actual_target) == (shift, intercept, target)
        assert before == [a.tobytes() for a in (actual_K, actual_x, actual_rhs)]
        result = original_portable(actual_K, actual_shift, actual_rhs, actual_x, actual_intercept, actual_target)
        calls.append(result)
        return result
    monkeypatch.setattr(linear_system, 'compensated_residual', record_portable)
    checked = system._compensated_measure(rhs, target, x, intercept)
    assert len(calls) == 1
    assert len(checked) == 6  # Acceptance remains the existing solver's decision.
    portable_residual, constraint, _, bounds, constraint_bound = calls[0]
    for value, bound in zip(portable_residual, bounds, strict=True):
        assert abs(exact(value) - exact_residual) <= exact(bound)
    assert abs(exact(constraint)) <= exact(constraint_bound)
    assert before == [a.tobytes() for a in (K, x, rhs)]
    assert system.info['native_residual_fallbacks'] == 2
    record_property('route', route)
    record_property('exact_residual_below_threshold', str(threshold - exact_residual))
    record_property('native_residual_bound', float(values[3][0]))
    record_property('portable_calls', len(calls))
    record_property('portable_assessment_accepted', bool(checked[0]))
