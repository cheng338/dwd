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
import sys as _fusion_sys
from types import ModuleType as _ModuleType, FunctionType as _FunctionType

import numpy as np

from ._affine_scores import affine_scores
from ._compensated_residual import _gradual_underflow, _products, _fsum
from ._dual_certificate import feasible_dual_slopes, _conjugate_mean_lower, _float_lower
from ._kernel_scores import compensated_kernel_matvec
from ._native_residual import _round_to_nearest
from ._objective_mean import objective_mean
from . import _quadratic_bounds
from ._quadratic_bounds import _compensated_dot
from .gen_dwd import V, V_grad
from . import _feature_report_optimization as _report_optimization

_ORIGINAL_REPORT_SUPPORT = _report_optimization
_REPORTING_POLICY_TOKEN = (_report_optimization.__dict__.get('_ORIGINAL_REPORTING_POLICY')
                           if type(_report_optimization) is _ModuleType else None)


def _feature_reporting_available(kind):
    support = _report_optimization
    if support is not _ORIGINAL_REPORT_SUPPORT or type(support) is not _ModuleType:
        return False
    policy = support.__dict__.get('_native_feature_reporting_available')
    return (type(_REPORTING_POLICY_TOKEN) is tuple and len(_REPORTING_POLICY_TOKEN) == 2
            and type(policy) is _FunctionType and policy is _REPORTING_POLICY_TOKEN[0]
            and policy.__code__ is _REPORTING_POLICY_TOKEN[1]
            and policy.__defaults__ is None and policy.__kwdefaults__ is None
            and policy(kind, compensated_kernel_matvec, affine_scores))


def _feature_reporting_matvec(features, values):
    if _feature_reporting_available('raw'):
        return _report_optimization._scores._reporting_kernel_matvec(features, values)
    return compensated_kernel_matvec(features, values)


def _feature_scores_and_margins(features, beta, offset):
    if _feature_reporting_available('raw'):
        scores = _report_optimization._scores._reporting_kernel_matvec(features, beta)
        # Fresh admission after the complete raw operation preserves late hooks.
        if _feature_reporting_available('joint'):
            margins = affine_scores(features, beta, offset, policy='compensated', _reporting=True)
        else:
            margins = affine_scores(features, beta, offset, policy='compensated')
        return scores, margins
    # Native-entry presence only declines sharing; it does not qualify native
    # arithmetic. Read plain-module dictionaries without invoking custom hooks.
    support = _report_optimization
    compiled = support.__dict__.get('_compiled') if type(support) is _ModuleType else None
    residual = compiled.__dict__.get('_compiled_residual') if type(compiled) is _ModuleType else None
    accelerator = residual.__dict__.get('_ACCEL') if type(residual) is _ModuleType else None
    if (type(accelerator) is _ModuleType
            and accelerator.__dict__.get('score_rows') is not None):
        scores = compensated_kernel_matvec(features, beta)
        margins = affine_scores(features, beta, offset, policy='compensated')
        return scores, margins
    return _report_optimization.feature_scores_and_margins(
        features, beta, offset, raw=compensated_kernel_matvec, affine=affine_scores)


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


_ORIGINAL_FEATURE_NUMPY = np
_ORIGINAL_FEATURE_FSUM = _fsum
_ORIGINAL_RESTORE_GRADIENT_TERMS = _restore_gradient_terms
_ORIGINAL_RESTORE_GRADIENT_TERMS_CODE = _restore_gradient_terms.__code__


def _standard_split_callbacks():
    """Do not hoist splits across replaced local contraction callbacks."""
    return (np is _ORIGINAL_FEATURE_NUMPY is _report_optimization._NUMPY_MODULE
            and _fsum is _ORIGINAL_FEATURE_FSUM
            and _restore_gradient_terms is _ORIGINAL_RESTORE_GRADIENT_TERMS
            and _restore_gradient_terms.__code__ is _ORIGINAL_RESTORE_GRADIENT_TERMS_CODE
            and _restore_gradient_terms.__defaults__ is None
            and _restore_gradient_terms.__kwdefaults__ is None)


def _tile_columns(n, p):
    """Bound numerical tile buffers; Python column lists are additional O(n).

    The 512-byte allowance per entry covers simultaneous product-expansion
    arrays and ufunc temporaries. One column above the 1 MiB target uses the
    original scalar path; this is not a bound on total process memory.
    """
    return max(1, min(64, p, (1 << 20)//(512*max(1, n))))


def _gradient_column(column, prepared):
    """Frozen scalar column path, also used to replay a failed tile."""
    z_mantissa, z_exponent, reg_high, reg_low, alpha_exponent, penalty_exponent = prepared
    column_mantissa, column_exponent = np.frexp(column)
    zh, zl = _products(column_mantissa, z_mantissa)
    zh, zl = _restore_gradient_terms((zh, zl), column_exponent+z_exponent)
    hh, hl = _products(column_mantissa, reg_high)
    lh, ll = _products(column_mantissa, reg_low)
    exponent = column_exponent+alpha_exponent+penalty_exponent
    # Scale only the final triple-product terms. Scaling lambda*alpha
    # first can underflow although F*(2*lambda)*alpha is representable.
    hh, hl, lh, ll = _restore_gradient_terms((hh, hl, lh, ll), exponent)
    return _fsum(zh.tolist()+zl.tolist()+hh.tolist()+hl.tolist()
                 +lh.tolist()+ll.tolist())


def _try_fused_gradient(columns, prepared, splits):
    # Experimental only: never hand an unknown cached object to importlib.
    import importlib
    import sys
    name = __package__ + '._feature_expansion'
    module = sys.modules.get(name, _FUSION_ABSENT)
    if module is _FUSION_ABSENT:
        try:
            module = importlib.import_module(name)
        except (ImportError, OSError):
            return None
    if type(module) is not _ModuleType:
        return None
    token = module.__dict__.get('_FUSION_ENTRY_TOKEN')
    function = module.__dict__.get('try_gradient')
    if (type(token) is not tuple or len(token) != 4
            or type(function) is not _FunctionType or function is not token[0]
            or function.__code__ is not token[1] or function.__defaults__ is not None
            or function.__kwdefaults__ is not None):
        return None
    return function(sys.modules[__name__], columns, prepared, splits)


_FUSION_ABSENT = object()
_FUSION_HELPER_TOKEN = (_try_fused_gradient, _try_fused_gradient.__code__, None, None)


def _gradient_tile(columns, prepared, *, _splits=None):
    """Expand B columns together, retaining each scalar six-list reduction."""
    if _splits is not None and len(columns) > 1:
        fused = _try_fused_gradient(columns, prepared, _splits)
        if fused is not None:
            return fused
    z_mantissa, z_exponent, reg_high, reg_low, alpha_exponent, penalty_exponent = prepared
    mantissa, exponent = np.frexp(columns)
    zh, zl = (_products(mantissa, np.broadcast_to(z_mantissa, columns.shape))
              if _splits is None else _products(
                  mantissa, np.broadcast_to(z_mantissa, columns.shape), _split_b=_splits[0]))
    zh, zl = _restore_gradient_terms((zh, zl), exponent+z_exponent)
    hh, hl = (_products(mantissa, np.broadcast_to(reg_high, columns.shape))
              if _splits is None else _products(
                  mantissa, np.broadcast_to(reg_high, columns.shape), _split_b=_splits[1]))
    lh, ll = (_products(mantissa, np.broadcast_to(reg_low, columns.shape))
              if _splits is None else _products(
                  mantissa, np.broadcast_to(reg_low, columns.shape), _split_b=_splits[2]))
    combined_exponent = exponent+alpha_exponent+penalty_exponent
    hh, hl, lh, ll = _restore_gradient_terms((hh, hl, lh, ll), combined_exponent)
    if _splits is not None and len(columns) > 1:
        from . import _linear_feature_diagnostics as _feature_owner
        from ._ordered_gradient_reduction import try_reduce as _try_native_reduce
        native_result = _try_native_reduce(_feature_owner, (zh, zl, hh, hl, lh, ll))
        if native_result is not None:
            return native_result
    result = np.empty(len(columns))
    for index in range(len(columns)):
        result[index] = _fsum(zh[index].tolist()+zl[index].tolist()
                              +hh[index].tolist()+hl[index].tolist()
                              +lh[index].tolist()+ll[index].tolist())
    return result


_ORIGINAL_GRADIENT_TILE = _gradient_tile
_ORIGINAL_GRADIENT_TILE_CODE = _gradient_tile.__code__


def _warm_native_batch_columns(n, remaining, oldwidth, *, beta=False):
    """Return an optional warm size hint; native guards still own admission.

    Keep original scalar/tail groups and do not import, initialize or rebase a
    context here. Changed feature callbacks keep their original group sizes.
    """
    if oldwidth <= 1 or n < 1 or n > 1024 or remaining <= oldwidth:
        return oldwidth
    proposed = oldwidth * (min(64, 16384 // n, remaining) // oldwidth)
    if proposed <= oldwidth:
        return oldwidth
    module = _fusion_sys.modules.get(__package__ + '._feature_expansion')
    if type(module) is not _ModuleType:
        return oldwidth
    context = module.__dict__.get('_CONTEXT')
    if type(context) is not tuple or len(context) != 6:
        return oldwidth
    if beta:
        helper, token = _try_fused_beta, _FUSION_BETA_HELPER_TOKEN
        tile, original, code, keyword = (
            _beta_tile, _ORIGINAL_BETA_TILE, _ORIGINAL_BETA_TILE_CODE, '_split')
        if module.__dict__.get('_BETA_ENTRY') is None:
            return oldwidth
    else:
        helper, token = _try_fused_gradient, _FUSION_HELPER_TOKEN
        tile, original, code, keyword = (
            _gradient_tile, _ORIGINAL_GRADIENT_TILE, _ORIGINAL_GRADIENT_TILE_CODE, '_splits')
    if (type(token) is not tuple or len(token) != 4
            or type(helper) is not _FunctionType or helper is not token[0]
            or helper.__code__ is not token[1] or helper.__defaults__ is not None
            or helper.__kwdefaults__ is not None
            or type(tile) is not _FunctionType or tile is not original
            or tile.__code__ is not code or tile.__defaults__ is not None):
        return oldwidth
    defaults = tile.__kwdefaults__
    if (type(defaults) is not dict or len(defaults) != 1
            or any(type(key) is not str for key in defaults)
            or defaults.get(keyword, _FUSION_ABSENT) is not None):
        return oldwidth
    return proposed


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
    prepared = (z_mantissa, z_exponent, reg_high, reg_low,
                alpha_exponent, penalty_exponent)
    result = np.empty(features.shape[1])
    width = _tile_columns(*features.shape)
    batching = width > 1
    splits = None
    if (batching and features.shape[1] > 1
            and type(features) is _ORIGINAL_FEATURE_NUMPY.ndarray
            and features.dtype == _ORIGINAL_FEATURE_NUMPY.float64
            and _gradient_tile is _ORIGINAL_GRADIENT_TILE
            and _gradient_tile.__code__ is _ORIGINAL_GRADIENT_TILE_CODE
            and _gradient_tile.__defaults__ is None
            and type(_gradient_tile.__kwdefaults__) is dict
            and all(type(key) is str for key in _gradient_tile.__kwdefaults__)
            and tuple(_gradient_tile.__kwdefaults__) == ('_splits',)
            and _gradient_tile.__kwdefaults__['_splits'] is None
            and _standard_split_callbacks()):
        splits = _report_optimization.repeated_operand_splits(
            (z_mantissa, reg_high, reg_low), _products)
    start = 0
    large_batches = True
    while start < features.shape[1]:
        # The first original group always runs naturally. A cold gradient
        # needs another original group before its fused context exists.
        if start and large_batches and batching and splits is not None:
            group_columns = _warm_native_batch_columns(
                features.shape[0], features.shape[1] - start, width)
            if group_columns > width:
                group_stop = start + group_columns
                grouped = None
                try:
                    grouped = _try_fused_gradient(
                        features[:, start:group_stop].T, prepared, splits)
                except (MemoryError, FloatingPointError, OverflowError):
                    pass
                if grouped is not None:
                    result[start:group_stop] = grouped
                    start = group_stop
                    continue
                # Leave failed frames before replaying every original group.
                # The optional attempt does not change original batch flags.
                large_batches = False
        stop = min(start+width, features.shape[1])
        values = None
        if batching and stop-start > 1:
            try:
                values = (_gradient_tile(features[:, start:stop].T, prepared)
                          if splits is None else _gradient_tile(
                              features[:, start:stop].T, prepared, _splits=splits))
            except MemoryError:
                batching = False
            except (FloatingPointError, OverflowError):
                splits = None
        # Outside the except block, failed tile frames/arrays have been
        # released. Replay the original columns in order: a later product
        # refusal must not mask an earlier scalar accumulation refusal.
        if values is None:
            for index in range(start, stop):
                result[index] = _gradient_column(features[:, index], prepared)
        else:
            result[start:stop] = values
        start = stop
    return result


def _beta_column(column, alpha_mantissa, alpha_exponent):
    """Frozen scalar coefficient path, including its range checks."""
    mantissa, exponent = np.frexp(column)
    high, low = _products(mantissa, alpha_mantissa)
    high, low = _restore_gradient_terms((high, low), exponent+alpha_exponent)
    return _fsum(high.tolist()+low.tolist())


def _try_fused_beta(columns, prepared, split):
    # Beta never imports or initializes the fused facade. A prior gradient may
    # have constructed its context; the facade verifies that warm state again.
    module = _fusion_sys.modules.get(__package__ + '._feature_expansion')
    if type(module) is not _ModuleType:
        return None
    token = module.__dict__.get('_FUSION_BETA_ENTRY_TOKEN')
    function = module.__dict__.get('try_beta')
    if (type(token) is not tuple or len(token) != 4
            or type(function) is not _FunctionType or function is not token[0]
            or function.__code__ is not token[1] or function.__defaults__ is not None
            or function.__kwdefaults__ is not None):
        return None
    return function(_fusion_sys.modules[__name__], columns, prepared, split)


_FUSION_BETA_HELPER_TOKEN = (_try_fused_beta, _try_fused_beta.__code__, None, None)


def _beta_tile(columns, alpha_mantissa, alpha_exponent, *, _split=None):
    if _split is not None and len(columns) > 1:
        fused = _try_fused_beta(columns, (alpha_mantissa, alpha_exponent), _split)
        if fused is not None:
            return fused
    mantissa, exponent = np.frexp(columns)
    high, low = (_products(mantissa, np.broadcast_to(alpha_mantissa, columns.shape))
                 if _split is None else _products(
                     mantissa, np.broadcast_to(alpha_mantissa, columns.shape), _split_b=_split))
    high, low = _restore_gradient_terms((high, low), exponent+alpha_exponent)
    result = np.empty(len(columns))
    for index in range(len(columns)):
        result[index] = _fsum(high[index].tolist()+low[index].tolist())
    return result


_ORIGINAL_BETA_TILE = _beta_tile
_ORIGINAL_BETA_TILE_CODE = _beta_tile.__code__


def _feature_coefficients(features, alpha):
    """Retain each feature/coefficient product before its column reduction.

    A product erased while forming beta can later be amplified by a large
    feature in the score. Refuse unrepresentable expansion terms rather than
    interpreting a lost coefficient as exact cancellation.
    """
    alpha_mantissa, alpha_exponent = np.frexp(alpha)
    result = np.empty(features.shape[1])
    width = _tile_columns(*features.shape)
    batching = width > 1
    split = None
    if (batching and features.shape[1] > 1
            and type(features) is _ORIGINAL_FEATURE_NUMPY.ndarray
            and features.dtype == _ORIGINAL_FEATURE_NUMPY.float64
            and _beta_tile is _ORIGINAL_BETA_TILE
            and _beta_tile.__code__ is _ORIGINAL_BETA_TILE_CODE
            and _beta_tile.__defaults__ is None
            and type(_beta_tile.__kwdefaults__) is dict
            and all(type(key) is str for key in _beta_tile.__kwdefaults__)
            and tuple(_beta_tile.__kwdefaults__) == ('_split',)
            and _beta_tile.__kwdefaults__['_split'] is None
            and _standard_split_callbacks()):
        prepared = _report_optimization.repeated_operand_splits((alpha_mantissa,), _products)
        if prepared is not None:
            split = prepared[0]
    start = 0
    large_batches = True
    while start < features.shape[1]:
        if start and large_batches and batching and split is not None:
            group_columns = _warm_native_batch_columns(
                features.shape[0], features.shape[1] - start, width, beta=True)
            if group_columns > width:
                group_stop = start + group_columns
                grouped = None
                try:
                    grouped = _try_fused_beta(
                        features[:, start:group_stop].T,
                        (alpha_mantissa, alpha_exponent), split)
                except (MemoryError, FloatingPointError, OverflowError):
                    pass
                if grouped is not None:
                    result[start:group_stop] = grouped
                    start = group_stop
                    continue
                large_batches = False
        stop = min(start+width, features.shape[1])
        values = None
        if batching and stop-start > 1:
            try:
                values = (_beta_tile(features[:, start:stop].T,
                                     alpha_mantissa, alpha_exponent)
                          if split is None else _beta_tile(
                              features[:, start:stop].T, alpha_mantissa,
                              alpha_exponent, _split=split))
            except MemoryError:
                batching = False
            except (FloatingPointError, OverflowError):
                split = None
        if values is None:
            for index in range(start, stop):
                result[index] = _beta_column(features[:, index], alpha_mantissa,
                                            alpha_exponent)
        else:
            result[start:stop] = values
        start = stop
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


def _dual_dot_constants(n):
    """Hoist the exact n-only checks/bounds from _compensated_dot.

    No callback occurs between these checks and the column contractions. The
    unchanged outer squared-norm dot performs its own arithmetic checks.
    """
    eps = np.finfo(float).eps
    half_normal = np.array([0x0008000000000000], dtype=np.uint64).view(np.float64)[0]
    with np.errstate(under='ignore'):
        halved = np.multiply(np.float64(np.finfo(float).tiny), np.float64(.5))
        doubled = np.multiply(half_normal, np.float64(2.))
    if (np.asarray(halved).view(np.uint64).item() != 0x0008000000000000
            or np.asarray(doubled).view(np.uint64).item() != 0x0010000000000000):
        raise FloatingPointError('Compensated initial-state checking requires gradual float64 underflow.')
    with np.errstate(under='ignore'):
        eta = float(np.nextafter(0., 1.))
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        allowance = _quadratic_bounds._upward_nonnegative((n+1)*eta)
        denominator = float(np.nextafter((1.-eps)*(1.-2*eps), 0.))
        tail = _quadratic_bounds._upward_nonnegative((2*n+2)*eta)
    return eps, allowance, denominator, tail


def _dual_dot_tile_scalar(columns, signed, constants):
    """Return each original dot's value, error and coordinate upper bound."""
    eps, allowance, denominator, tail = constants
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        products = columns*np.broadcast_to(signed, columns.shape)
    if not np.isfinite(products).all():
        raise FloatingPointError('Nonfinite products in compensated initial-state check.')
    result = []
    for row in products:
        try:
            value = _quadratic_bounds.fsum(row.tolist())
            measured_absolute = _quadratic_bounds.fsum(np.abs(row).tolist())
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError('Compensated initial-state accumulation overflowed.') from exc
        if not np.isfinite(value) or not np.isfinite(measured_absolute):
            raise FloatingPointError('Nonfinite compensated initial-state accumulation.')
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            numerator = _quadratic_bounds._upward_nonnegative(measured_absolute+allowance)
            absolute_upper = _quadratic_bounds._upward_nonnegative(numerator/denominator)
            error = _quadratic_bounds._upward_nonnegative(
                _quadratic_bounds._upward_nonnegative((4*eps)*absolute_upper)+tail)
        upper = math.nextafter(abs(value)+error, math.inf)
        result.append((float(value), error, upper))
    return result


def _dual_dot_tile_vector(columns, signed, constants):
    """Keep both per-column fsum orders, then evaluate B outward bounds."""
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        products = columns*np.broadcast_to(signed, columns.shape)
    if not np.isfinite(products).all():
        raise FloatingPointError('Nonfinite products in compensated initial-state check.')
    values = np.empty(len(columns))
    measured_absolute = np.empty(len(columns))
    for index, row in enumerate(products):
        try:
            value = _quadratic_bounds.fsum(row.tolist())
            absolute = _quadratic_bounds.fsum(np.abs(row).tolist())
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError('Compensated initial-state accumulation overflowed.') from exc
        if not np.isfinite(value) or not np.isfinite(absolute):
            raise FloatingPointError('Nonfinite compensated initial-state accumulation.')
        values[index], measured_absolute[index] = value, absolute
    errors, upper = _quadratic_bounds._dual_coordinate_bounds(
        values, measured_absolute, constants)
    return [(float(value), float(error), float(bound))
            for value, error, bound in zip(values, errors, upper)]


def _dual_dot_tile(columns, signed, constants):
    # A replacement hook must observe the existing scalar-bound tile, including
    # its call order. Do not bless a wrapper merely because it was imported early.
    if not _quadratic_bounds._standard_feature_bound_bindings():
        return _dual_dot_tile_scalar(columns, signed, constants)
    try:
        return _dual_dot_tile_vector(columns, signed, constants)
    except (FloatingPointError, OverflowError):
        pass
    # An earlier column's bound refusal can otherwise be masked by a later
    # fsum refusal. Leave the exception frame before replaying the old order.
    # MemoryError is intentionally not caught here: the existing caller
    # disables batching and replays original scalar columns on that exception.
    return _dual_dot_tile_scalar(columns, signed, constants)


def _dual_coordinate_upper(features, signed):
    upper = np.empty(features.shape[1])
    width = _tile_columns(*features.shape)
    batching = width > 1
    constants = None
    for start in range(0, features.shape[1], width):
        stop = min(start+width, features.shape[1])
        values = None
        if batching and stop-start > 1:
            try:
                if constants is None:
                    constants = _dual_dot_constants(features.shape[0])
                values = _dual_dot_tile(features[:, start:stop].T, signed, constants)
            except MemoryError:
                batching = False
            except (FloatingPointError, OverflowError):
                pass
        if values is None:
            for index in range(start, stop):
                value, error = _compensated_dot(features[:, index], signed)
                upper[index] = math.nextafter(abs(value)+error, math.inf)
        else:
            for index, (_, _, bound) in enumerate(values, start):
                upper[index] = bound
    return upper


def _dual_endpoint(features, y, raw_slopes, lambd, q):
    rho = feasible_dual_slopes(raw_slopes, y)
    signed = y*rho
    method, fallback = None, None
    endpoint = 0.
    try:
        if not _round_to_nearest():
            raise FloatingPointError('Linear feature dual bounds require round-to-nearest.')
        upper = _dual_coordinate_upper(features, signed)
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
    scores, joint_scores = _feature_scores_and_margins(features, beta, offset)
    margins = y*joint_scores
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
    gradient_coefficients = _feature_reporting_matvec(features, gradient_features)
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
