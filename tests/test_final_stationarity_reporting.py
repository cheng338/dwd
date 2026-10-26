"""Final reporting uses the original equations without changing optimization."""
from decimal import Decimal, localcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from dwd import _compiled_residual as loader
from dwd import _compiled_scores as compiled
from dwd import _kernel_scores as scores
from dwd import _kernel_solver as solver


FIELDS = {'rkhs_gradient_norm', 'gradient_inf_norm', 'intercept_gradient'}


def decimal_stationarity(K, y, alpha, offset, lambd, q):
    """Independent original-equation reference on represented inputs."""
    with localcontext() as context:
        context.prec = 100
        D = lambda value: Decimal.from_float(float(value))
        matrix = [[D(value) for value in row] for row in K]
        a = list(map(D, alpha))
        labels = list(map(D, y))
        exponent, penalty = D(q), D(lambd)
        threshold = exponent / (exponent + 1)
        decision = [sum(x * v for x, v in zip(row, a)) + D(offset) for row in matrix]
        margins = [label * value for label, value in zip(labels, decision)]
        slopes = [Decimal(-1) if margin <= threshold else
                  -(threshold / margin) ** (exponent + 1) for margin in margins]
        z = [label * slope / len(y) for label, slope in zip(labels, slopes)]
        w = [value + 2 * penalty * coefficient for value, coefficient in zip(z, a)]
        action = [sum(x * v for x, v in zip(row, w)) for row in matrix]
        squared = sum(x * v for x, v in zip(w, action))
        bias = sum(z)
        return {'rkhs_gradient_norm': float(max(abs(bias), squared.sqrt())),
                'gradient_inf_norm': float(max(abs(bias), *map(abs, action))),
                'intercept_gradient': float(bias)}


@pytest.mark.parametrize('q', [.2, 1., 2.3])
@pytest.mark.parametrize('layout', ['C', 'F'])
def test_final_estimates_match_independent_original_equations(q, layout):
    X = np.array([[1., 2.], [-1., .5], [.25, -.75], [2., -1.]])
    K = np.array(X @ X.T + np.eye(4) / 8, order=layout)
    y = np.array([1., -1., 1., -1.])
    alpha = np.array([.3, -.1, .2, -.4])
    expected = decimal_stationarity(K, y, alpha, .7, .13, q)
    copies = [value.copy() for value in (K, y, alpha)]
    actual = solver.optimality_diagnostics(K, y, alpha, .7, .13, q)
    for key in FIELDS:
        assert_allclose(actual[key], expected[key], rtol=2e-14, atol=2e-15)
    for value, original in zip((K, y, alpha), copies):
        assert_array_equal(value, original)


def test_joint_affine_cancellation_is_not_replaced_by_supplied_scores():
    # Exact represented K*alpha is 1, even if an ordinary dot loses the small
    # product. The affine margin is .75; the intercept derivative is 5/18.
    K = np.full((4, 4), 2.**52)
    y = np.array([1., -1., 1., -1.])
    alpha = np.array([1., 2.**-52, -1., 0.])
    actual = solver.optimality_diagnostics(K, y, alpha, -.25, 2.**-40, 1.,
                                         scores=np.zeros(4))
    assert_allclose(actual['intercept_gradient'], 5 / 18, rtol=1e-15)


def test_score_roundoff_cannot_create_a_false_negative_stationarity_report():
    K = np.zeros((4, 4))
    y = np.array([1., -1., 1., -1.])
    alpha = np.zeros(4)
    # A tiny discrepancy in the separately reconstructed scores gets amplified
    # by the old square root, despite this model's exactly zero full gradient.
    rounded_scores = -1e-10 * y
    with patch.object(solver, '_final_stationarity', return_value={}):
        original = solver.optimality_diagnostics(K, y, alpha, 0., .2, 1., scores=rounded_scores)
    corrected = solver.optimality_diagnostics(K, y, alpha, 0., .2, 1., scores=rounded_scores)
    assert original['rkhs_gradient_norm'] > 1e-6
    assert corrected['rkhs_gradient_norm'] == 0.
    assert corrected['intercept_gradient'] == 0.
    assert corrected['gradient_inf_norm'] == 0.
    for field in corrected.keys() - FIELDS:
        assert corrected[field] == original[field]


def test_product_roundoff_is_retained_before_the_kernel_action():
    K = np.eye(2)
    y = np.array([1., -1.])
    # 2*lambda*alpha rounds to exactly .5, but its represented product has a
    # small nonzero tail. A conventional z + rounded product erases it.
    alpha = np.array([2.5, -2.5])
    # A supplied offset is irrelevant here: test the small residual under a
    # linear loss by using K=0 for margins and an observed action separately.
    with patch.object(solver, 'affine_scores', return_value=np.zeros(2)):
        actual = solver._final_stationarity(K, y, alpha, 0., .1, 1.)
    with localcontext() as context:
        context.prec = 100
        tail = Decimal(2) * Decimal.from_float(.1) * Decimal.from_float(2.5) - Decimal('.5')
        expected = float((2 * tail * tail).sqrt())
    assert expected > 0
    assert_allclose(actual['rkhs_gradient_norm'], expected, rtol=2e-15, atol=0)


def test_original_objective_dual_norm_and_conversion_fields_are_unchanged():
    K = np.array([[2., .25], [.25, 1.]])
    y, alpha = np.array([1., -1.]), np.array([.2, -.3])
    with patch.object(solver, '_final_stationarity', return_value={}):
        original = solver.optimality_diagnostics(K, y, alpha, .1, .2, 1.7)
    corrected = solver.optimality_diagnostics(K, y, alpha, .1, .2, 1.7)
    assert corrected.keys() == original.keys()
    assert {key: value for key, value in corrected.items() if key not in FIELDS} == {
        key: value for key, value in original.items() if key not in FIELDS}


@pytest.mark.parametrize('alpha, message', [(np.ones(2), 'RKHS norm squared'),
                                          (np.zeros(2), 'RKHS gradient norm squared')])
def test_original_negative_quadratic_refusals_precede_new_reporting(alpha, message):
    with patch.object(solver, '_final_stationarity') as stable:
        with pytest.raises(FloatingPointError, match=message):
            solver.optimality_diagnostics(-np.eye(2), np.array([1., -1.]), alpha, 0., .1, 1.)
    stable.assert_not_called()


def test_original_primal_dual_refusal_precedes_new_reporting():
    with patch.object(solver, 'kernel_dual_lower_bound', return_value=(2., {})), \
            patch.object(solver, '_final_stationarity') as stable:
        with pytest.raises(FloatingPointError, match='Inconsistent primal-dual'):
            solver.optimality_diagnostics(np.eye(2), np.array([1., -1.]), np.zeros(2), 0., .1, 1.)
    stable.assert_not_called()


@pytest.mark.parametrize('backend', ['cholesky', 'spectral', 'lbfgs'])
@pytest.mark.parametrize('stopping', ['fixed', 'optimality'])
def test_final_fields_do_not_feed_back_into_optimizer_or_stopping(backend, stopping):
    K = np.array([[2., .2, -.1, .4], [.2, 1., .3, -.1],
                  [-.1, .3, 1.5, .2], [.4, -.1, .2, 2.]])
    y = np.array([1., -1., 1., -1.])
    arguments = dict(backend=backend, stopping=stopping, max_iter=4, tol=1e-6)
    high = dict.fromkeys(FIELDS, 1.)
    low = dict.fromkeys(FIELDS, 0.)
    with patch.object(solver, '_final_stationarity', return_value=high) as report:
        first = solver.solve_kernel(K, y, .2, **arguments)
        report.assert_called_once()
    with patch.object(solver, '_final_stationarity', return_value=low) as report:
        second = solver.solve_kernel(K, y, .2, **arguments)
        report.assert_called_once()
    for key in ('alpha', 'objective_history', 'offset', 'n_iter', 'returned_iteration',
                'termination_reason', 'final_objective', 'dual_gap', 'objective_tolerance_met'):
        assert_array_equal(first[key], second[key])
    assert first['converged'] is False
    assert second['converged'] is True
    assert first['diagnostics']['optimality_tolerance'] == second['diagnostics']['optimality_tolerance'] == 1e-6


def python_entry(tile, alpha, bm, be, bh, bl, output):
    high, low = compiled._PRODUCTS(tile, np.broadcast_to(alpha, tile.shape),
                                  _split_b=(bm, be, bh, bl))
    for i in range(len(tile)):
        output[i] = compiled._FSUM(high[i].tolist() + low[i].tolist())
    return 0


def test_linux_native_scope_is_only_reporting_kw_and_preserves_fallback():
    K = np.array([[1e16, 1., -1e16]] * 9)
    alpha = np.ones(3)
    with patch.object(loader, '_ACCEL', None):
        expected = scores._reporting_kernel_matvec(K, alpha)
    entry = Mock(side_effect=python_entry)
    with patch.object(compiled.sys, 'platform', 'linux'), \
            patch.object(compiled.platform, 'machine', return_value='x86_64'), \
            patch.object(loader, '_ACCEL', SimpleNamespace(score_rows=entry)):
        assert not compiled._supported()
        assert compiled._supported(reporting=True)
        assert_array_equal(scores.compensated_kernel_matvec(K, alpha), expected)
        scores.adaptive_kernel_matvec(K, alpha)
        entry.assert_not_called()
        assert_array_equal(scores._reporting_kernel_matvec(K, alpha), expected)
        assert entry.call_count > 0
        entry.side_effect = lambda *args: 1
        assert_array_equal(scores._reporting_kernel_matvec(K, alpha), expected)


@pytest.mark.parametrize('guard, value', [('platform', 'darwin'), ('version_info', (3, 10))])
def test_reporting_does_not_bypass_other_platform_guards(guard, value):
    with patch.object(compiled.sys, guard, value):
        assert not compiled._supported(reporting=True)


def test_reporting_preserves_custom_arithmetic_and_numpy_handler_fallback():
    K, alpha = np.ones((3, 3)), np.ones(3)
    entry = Mock(side_effect=AssertionError('guard bypassed'))
    with patch.object(loader, '_ACCEL', SimpleNamespace(score_rows=entry)), \
            patch.object(compiled.sys, 'platform', 'linux'):
        with patch.object(scores, '_products', wraps=scores._products):
            assert_array_equal(scores._reporting_kernel_matvec(K, alpha), np.full(3, 3.))
        with np.errstate(all='call'):
            assert_array_equal(scores._reporting_kernel_matvec(K, alpha), np.full(3, 3.))
    entry.assert_not_called()


@pytest.mark.skipif(getattr(loader._ACCEL, 'score_rows', None) is None,
                    reason='Optional compiled scorer is unavailable')
def test_actual_native_reporting_matches_portable_complete_report():
    if not compiled._supported(reporting=True):
        pytest.skip('Runtime outside the guarded reporting domain')
    K = np.eye(9) + np.ones((9, 9)) / 8
    y = np.array([1., -1., 1., -1., 1., -1., 1., -1., 1.])
    alpha = y / 7
    with patch.object(loader, '_ACCEL', None):
        expected = solver.optimality_diagnostics(K, y, alpha, .3, .1, 2.3)
    entry = loader._ACCEL.score_rows
    with patch.object(loader._ACCEL, 'score_rows', wraps=entry) as invoked:
        actual = solver.optimality_diagnostics(K, y, alpha, .3, .1, 2.3)
    assert invoked.call_count > 0
    assert actual == expected
