"""Independent adversaries for full-precision feature reporting arithmetic.

These direct numerical-helper cases are not claims that the public initial
state guard accepts such coefficients. Public dispatch has its own suite.
"""
from decimal import Decimal, localcontext
from fractions import Fraction
import math

import numpy as np
import pytest

from dwd._linear_feature_diagnostics import (
    linear_feature_diagnostics, _feature_gradient, _feature_coefficients, _weighted_norm,
)


def _report(F, y, alpha, lambd, q):
    # Both fixtures have zero represented scores and objective one. Supplying
    # these observations isolates final-report arithmetic from optimization.
    return linear_feature_diagnostics(F, y, alpha, 0., lambd, q,
        checked_scores=np.zeros(len(y)), checked_objective=1.,
        optimizer_scores=np.zeros(len(y)), optimizer_objective=1.)


def test_feature_gradient_retains_small_terms_between_large_cancelling_coefficients():
    F = np.array([[1.], [1.], [0.], [0.]])
    y = np.array([1., 1., -1., -1.])
    alpha = np.array([2.**53, -2.**53, 0., 0.])
    pins = [value.copy() for value in (F, y, alpha)]
    exact_beta = sum(Fraction.from_float(float(x))*Fraction.from_float(float(a))
                     for x, a in zip(F[:, 0], alpha))
    exact_gradient = sum(Fraction.from_float(float(x))*(-Fraction.from_float(float(label))/4)
                         for x, label in zip(F[:, 0], y))+exact_beta
    assert exact_beta == 0 and exact_gradient == Fraction(-1, 2)
    report, info = _report(F, y, alpha, .5, 1.)
    assert report['final_objective'] == 1.
    assert report['rkhs_norm_squared'] == report['C'] == 0.
    assert report['rkhs_gradient_norm'] == .5
    assert report['gradient_inf_norm'] == .5 and report['intercept_gradient'] == 0.
    assert report['rkhs_gradient_norm'] > 1e-6
    assert Fraction.from_float(report['dual_objective']) <= Fraction(7, 8)
    for actual, expected in zip((F, y, alpha), pins):
        np.testing.assert_array_equal(actual, expected)


def test_conversion_uses_feature_norm_when_squared_norm_underflows():
    F = np.ldexp(np.eye(2), -400)
    y = np.array([-1., 1.])
    alpha = np.array([-2.**-300, 2.**-300])
    q = .2
    with localcontext() as context:
        context.prec = 110
        qq = Decimal.from_float(q)
        norm = Decimal(2).sqrt()*Decimal(2)**-700
        expected_C = (qq+1)*((qq+1)/qq)**qq*norm**(qq+1)
        # Every exact margin is 2^-1100, below the binary64 minimum subnormal.
        exact_gradient_coordinate = -Decimal(2)**-401+2*Decimal(2)**-700
        expected_gradient = float(Decimal(2).sqrt()*abs(exact_gradient_coordinate))
    report, info = _report(F, y, alpha, 1., q)
    assert report['rkhs_norm_squared'] == 0.
    assert 0. < float(expected_C) < 1e-200
    assert math.isfinite(report['C'])
    np.testing.assert_allclose(report['C'], float(expected_C), rtol=2e-12, atol=0.)
    np.testing.assert_allclose(report['rkhs_gradient_norm'], expected_gradient, rtol=2e-14, atol=0.)
    assert report['final_objective'] == 1.
    assert report['dual_objective'] <= 1.


def test_feature_amplification_cannot_hide_regularizer_product_underflow():
    F = np.ldexp(np.eye(2), 500)
    y = np.array([-1., 1.])
    alpha = np.ldexp(y, -500)
    lambd, q = 2.**-700, 2.
    with localcontext() as context:
        context.prec = 110
        margin = Decimal(2)**500
        loss_slope = -(Decimal(2)/3/margin)**3
        coordinate = margin*loss_slope/2+Decimal(2)**-699
        expected = float(Decimal(2).sqrt()*abs(coordinate))
        primal = float(2*(Decimal(2)**-700)+(Decimal(2)/3/margin)**2/3)
    assert expected > 2.**-710
    isolated_regularizer = _feature_gradient(F, np.zeros(2), alpha, 2*lambd)
    np.testing.assert_array_equal(isolated_regularizer, np.ldexp(y, -699))
    try:
        report, info = linear_feature_diagnostics(F, y, alpha, 0., lambd, q,
            checked_scores=np.ldexp(y, 500), checked_objective=primal,
            optimizer_scores=np.ldexp(y, 500), optimizer_objective=primal)
    except FloatingPointError as error:
        # Explicit arithmetic-domain refusal is safe. A zero gradient is not.
        assert any(word in str(error).lower() for word in ('underflow', 'range'))
    else:
        np.testing.assert_allclose(report['rkhs_gradient_norm'], expected, rtol=2e-13, atol=0.)
        assert report['rkhs_gradient_norm'] > 2.**-710
        np.testing.assert_allclose(report['final_objective'], primal, rtol=2e-14, atol=0.)


def test_erased_feature_coefficient_refuses_before_later_amplification():
    F = np.array([[2.**-500], [2.**500]])
    alpha = np.array([2.**-700, 0.])
    exact_beta = Fraction(1, 2**1200)
    exact_second_score = exact_beta*Fraction(2**500)
    assert exact_beta > 0 and float(exact_beta) == 0.
    assert float(exact_second_score) == 2.**-700 > 0.
    with pytest.raises(FloatingPointError, match='underflow'):
        _feature_coefficients(F, alpha)


@pytest.mark.parametrize('beta_exponent,lambda_exponent,expected_exponent',
                         [(-600, 1000, -200), (600, -1000, 200)])
def test_weighted_norm_preserves_range_before_squaring(beta_exponent, lambda_exponent,
                                                      expected_exponent):
    beta = np.array([math.ldexp(1., beta_exponent)])
    penalty = math.ldexp(1., lambda_exponent)
    exact = Fraction.from_float(penalty)*Fraction.from_float(beta[0])**2
    assert exact == Fraction.from_float(math.ldexp(1., expected_exponent))
    assert _weighted_norm(beta, penalty) == float(exact)
