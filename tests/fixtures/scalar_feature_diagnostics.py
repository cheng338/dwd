"""Complete final diagnostics for a trusted dense linear feature geometry.

The target is the real Gram matrix of the represented training features. The
primal and gradient are numerical estimates; the dual endpoint uses outward
arithmetic bounds under the same assumptions as the existing certificates.
Construction provenance is established by the estimator, not by this helper.
Original-kernel and optimizer reconstruction checks remain separate obligations.
Erased loss derivatives and nonnormal gradient-expansion terms are outside this
route's supported arithmetic range and raise instead of indicating convergence.
"""
from fractions import Fraction
import math

import numpy as np

from ._affine_scores import affine_scores
from ._compensated_residual import _gradual_underflow, _products, _fsum
from ._dual_certificate import feasible_dual_slopes, _conjugate_mean_lower, _float_lower
from ._kernel_scores import compensated_kernel_matvec
from ._native_residual import _round_to_nearest
from ._objective_mean import objective_mean
from ._quadratic_bounds import _compensated_dot
from .gen_dwd import V, V_grad


def _dot(left, right):
    high, low = _products(left, right)
    return _fsum(high.tolist()+low.tolist())


def _restore_gradient_terms(terms, exponent):
    """Restore expanded products without accepting erased/subnormal terms.

    No feature multiplication follows this scaling, but cancellation can still
    expose a small term. This route explicitly refuses that arithmetic range
    rather than treating lost expansion terms as a zero stationarity estimate.
    Exact zero terms remain valid, including zero coefficients and features.
    """
    result = []
    for term in terms:
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            restored = np.ldexp(term, exponent)
        if not np.isfinite(restored).all():
            raise FloatingPointError('Nonfinite or overflowing feature gradient products.')
        if np.any((term != 0)&(np.abs(restored) < np.finfo(float).tiny)):
            raise FloatingPointError('Feature gradient expansion underflow is outside the supported arithmetic range.')
        result.append(restored)
    return result


def _feature_gradient(features, z, alpha, twice_penalty):
    """Contract all residual terms before rounding a feature coordinate.

    Rounding each z_i + 2*lambda*alpha_i first can erase loss-gradient terms
    next to large, cancelling coefficients. Keep the regularization expansion
    and all three feature products until the final compensated column sum.
    """
    penalty_mantissa, penalty_exponent = math.frexp(twice_penalty)
    alpha_mantissa, alpha_exponent = np.frexp(alpha)
    z_mantissa, z_exponent = np.frexp(z)
    reg_high, reg_low = _products(np.full(len(alpha), penalty_mantissa), alpha_mantissa)
    result = np.empty(features.shape[1])
    for index in range(features.shape[1]):
        column = features[:, index]
        column_mantissa, column_exponent = np.frexp(column)
        zh, zl = _products(column_mantissa, z_mantissa)
        zh, zl = _restore_gradient_terms((zh, zl), column_exponent+z_exponent)
        hh, hl = _products(column_mantissa, reg_high)
        lh, ll = _products(column_mantissa, reg_low)
        exponent = column_exponent+alpha_exponent+penalty_exponent
        # Scale only the final triple-product terms. Scaling lambda*alpha
        # first can underflow although F*(2*lambda)*alpha is representable.
        hh, hl, lh, ll = _restore_gradient_terms((hh, hl, lh, ll), exponent)
        result[index] = _fsum(zh.tolist()+zl.tolist()+hh.tolist()+hl.tolist()
                              +lh.tolist()+ll.tolist())
    return result


def _feature_coefficients(features, alpha):
    """Retain each feature/coefficient product before its column reduction.

    A product erased while forming beta can later be amplified by a large
    feature in the score. Refuse unrepresentable expansion terms rather than
    interpreting a lost coefficient as exact cancellation.
    """
    alpha_mantissa, alpha_exponent = np.frexp(alpha)
    result = np.empty(features.shape[1])
    for index in range(features.shape[1]):
        mantissa, exponent = np.frexp(features[:, index])
        high, low = _products(mantissa, alpha_mantissa)
        high, low = _restore_gradient_terms((high, low), exponent+alpha_exponent)
        result[index] = _fsum(high.tolist()+low.tolist())
    return result


def _weighted_norm(beta, lambd):
    """Estimate lambda*||beta||**2 without an underflowed square first."""
    norm = math.hypot(*beta)
    if norm == 0:
        return 0.
    penalty_mantissa, penalty_exponent = math.frexp(lambd)
    norm_mantissa, norm_exponent = math.frexp(norm)
    try:
        value = math.ldexp(penalty_mantissa*norm_mantissa*norm_mantissa,
                           penalty_exponent+2*norm_exponent)
    except OverflowError as exc:
        raise FloatingPointError('Overflowing weighted feature norm.') from exc
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite weighted feature norm.')
    return value


def _dual_endpoint(features, y, raw_slopes, lambd, q):
    rho = feasible_dual_slopes(raw_slopes, y)
    signed = y*rho
    method, fallback = None, None
    endpoint = 0.
    try:
        if not _round_to_nearest():
            raise FloatingPointError('Linear feature dual bounds require round-to-nearest.')
        upper = np.empty(features.shape[1])
        for index in range(features.shape[1]):
            value, error = _compensated_dot(features[:, index], signed)
            upper[index] = math.nextafter(abs(value)+error, math.inf)
        if not np.isfinite(upper).all():
            raise FloatingPointError('Nonfinite linear feature dual coordinate bound.')
        value, error = _compensated_dot(upper, upper)
        quadratic_upper = Fraction.from_float(value)+Fraction.from_float(error)
        if quadratic_upper < 0:
            raise ArithmeticError('A feature squared-norm upper bound cannot be negative.')
        method = 'bounded_feature_coordinate_dots_and_squared_norm'
        bound = (_conjugate_mean_lower(rho, q)
                 -quadratic_upper/(4*Fraction.from_float(lambd)*len(y)**2))
        if bound < 0:
            fallback = 'zero_slopes_give_a_stronger_certified_bound'
        else:
            endpoint = _float_lower(bound)
    except (OverflowError, FloatingPointError):
        fallback = 'nonzero_feature_dual_bound_unavailable'
    return float(endpoint), {
        'feasibility': 'exact_dyadic_class_mass_balance',
        'slope_quantum': 2.**-52,
        'dual_quadratic_method': method,
        'fallback_reason': fallback,
        'dual_objective_is_lower_bound': True,
        'kernel_assumption': 'exact real Gram of represented dense training features',
        'primal_objective_and_gap_are_estimates': True,
        'arithmetic_assumptions': 'IEEE binary64 round-to-nearest with gradual underflow; faithfully rounded math.fsum; correctly rounded Decimal ln/exp/sqrt',
    }


def _score_check(actual, expected):
    difference = float(np.max(np.abs(actual-expected)))
    threshold = 5e-7*max(1., float(np.max(np.abs(actual))))
    return dict(max_absolute_difference=difference, threshold=threshold,
                passed=bool(math.isfinite(difference) and difference <= threshold))


def _objective_check(actual, expected):
    difference = float(actual-expected)
    threshold = 5e-8*max(1., abs(actual))
    return dict(signed_difference=difference, threshold=threshold,
                passed=bool(math.isfinite(difference) and abs(difference) <= threshold))


def _conversion(beta, q):
    # A represented beta can have a representable norm and C even when its
    # squared norm underflows. Do not infer zero C from the squared-norm estimate.
    norm = math.hypot(*beta)
    if norm == 0:
        return 0.
    q_term = q*(math.log1p(q)-math.log(q)) if q < 1. else q*math.log1p(1./q)
    logarithm = math.log1p(q)+q_term+(q+1.)*math.log(norm)
    return math.exp(logarithm) if logarithm <= math.log(np.finfo(float).max) else math.inf


def linear_feature_diagnostics(features, y, alpha, offset, lambd, q, *,
                               checked_scores, checked_objective,
                               optimizer_scores, optimizer_objective):
    """Return existing final metrics and scalar geometry/compatibility metadata.

    Inputs come from validated dense training features and the actual returned
    solver state. checked_scores/objective have already passed the original-K
    reconstruction gates. No optimizer state, prediction policy or tolerance is
    changed here; a nonconverged state may still be returned by the solver.
    """
    if (type(features) is not np.ndarray or features.dtype != np.float64
            or features.ndim != 2 or features.shape[0] != len(alpha)
            or features.shape[1] < 1 or np.shape(y) != np.shape(alpha)
            or np.shape(checked_scores) != np.shape(alpha)
            or np.shape(optimizer_scores) != np.shape(alpha)
            or set(np.unique(y)) != {-1., 1.}):
        raise ValueError('Invalid dense linear feature diagnostic geometry.')
    _gradual_underflow()
    if not _round_to_nearest():
        raise FloatingPointError('Linear feature diagnostics require round-to-nearest.')
    twice_penalty = 2*lambd
    if not math.isfinite(twice_penalty):
        raise FloatingPointError('Twice the penalty exceeds finite feature arithmetic.')
    beta = _feature_coefficients(features, alpha)
    scores = compensated_kernel_matvec(features, beta)
    margins = y*affine_scores(features, beta, offset, policy='compensated')
    slopes = V_grad(margins, q=q)
    # For finite margins and positive finite q the exact loss derivative is
    # strictly negative. An erased derivative may be amplified by a feature;
    # it is not evidence of stationarity and is never replaced by tiny.
    if not np.isfinite(slopes).all() or np.any(slopes >= 0):
        raise FloatingPointError('Loss derivative underflow or nonfinite values exceed the supported feature arithmetic range.')
    z = y*slopes/len(y)
    if np.any(np.abs(z) < np.finfo(float).tiny):
        raise FloatingPointError('Mean loss derivative underflow exceeds the supported feature arithmetic range.')
    gradient_features = _feature_gradient(features, z, alpha, twice_penalty)
    gradient_coefficients = compensated_kernel_matvec(features, gradient_features)
    intercept_gradient = _fsum(z.tolist())
    norm_squared = _dot(beta, beta)
    if norm_squared < 0:
        raise ArithmeticError('A feature squared norm cannot be negative.')
    primal = float(objective_mean(V(margins, q=q), _weighted_norm(beta, lambd), inside=False))
    gradient = max(abs(intercept_gradient), math.hypot(*gradient_features))
    gradient_inf = max(abs(intercept_gradient), float(np.max(np.abs(gradient_coefficients))))
    dual, certificate = _dual_endpoint(features, y, -slopes, lambd, q)
    gap = primal-dual
    if not all(math.isfinite(value) for value in
               (primal, norm_squared, gradient, gradient_inf, intercept_gradient, dual, gap)):
        raise FloatingPointError('Nonfinite dense linear feature diagnostic.')
    if gap < -1e-8*max(1., abs(primal), abs(dual)):
        raise FloatingPointError('Inconsistent dense linear feature primal-dual bound.')
    compatibility = {
        'feature_to_checked_original_scores': _score_check(scores, checked_scores),
        'feature_to_optimizer_scores': _score_check(scores, optimizer_scores),
        'feature_to_checked_original_objective': _objective_check(primal, checked_objective),
        'feature_to_optimizer_objective': _objective_check(primal, optimizer_objective),
    }
    if not all(check['passed'] for check in compatibility.values()):
        raise FloatingPointError('Returned coefficients lost dense linear feature score or objective accuracy; '
                                 'check regularization, kernel precision or initialization.')
    metrics = dict(final_objective=primal, rkhs_norm_squared=norm_squared,
                   gradient_inf_norm=gradient_inf, rkhs_gradient_norm=gradient,
                   intercept_gradient=intercept_gradient, dual_objective=dual,
                   dual_gap=float(gap), dual_equality_residual=0.,
                   dual_certificate=certificate, C=_conversion(beta, q))
    info = dict(target='exact real Gram of represented owned dense training features',
                provenance='internally_constructed_named_linear',
                n_samples=int(features.shape[0]), n_features=int(features.shape[1]),
                primal_and_gradient_are_estimates=True,
                gradient_arithmetic='expanded feature contraction; erased derivatives and nonnormal expansion terms refuse',
                prediction_implementation_unchanged=True,
                compatibility=compatibility)
    return metrics, info
