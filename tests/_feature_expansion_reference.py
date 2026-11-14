"""Frozen literal contraction oracle, independent of fused dispatch.

Arithmetic and scalar replay are extracted from the reviewed pre-expansion
implementation (SHA256 85f3568f500562db1397269b5bba09d5e1ce69711c036c4f98031a971201fd8b).
Only optimization admission and its native-only branch are removed; product primitives remain
shared and separate exact-Fraction tests independently check dyadic cases.
"""
import math
import numpy as np
from dwd._compensated_residual import _products, _fsum
from dwd import _feature_report_optimization as _report_optimization


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
    # The oracle retains the original scalar/tile arithmetic and replay order.
    # Disable optimization admission so it never calls a native contraction.
    return False


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


def _gradient_tile(columns, prepared, *, _splits=None):
    """Expand B columns together, retaining each scalar six-list reduction."""
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
    result = np.empty(len(columns))
    for index in range(len(columns)):
        result[index] = _fsum(zh[index].tolist()+zl[index].tolist()
                              +hh[index].tolist()+hl[index].tolist()
                              +lh[index].tolist()+ll[index].tolist())
    return result


_ORIGINAL_GRADIENT_TILE = _gradient_tile
_ORIGINAL_GRADIENT_TILE_CODE = _gradient_tile.__code__


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
    for start in range(0, features.shape[1], width):
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
    return result


def _beta_column(column, alpha_mantissa, alpha_exponent):
    """Frozen scalar coefficient path, including its range checks."""
    mantissa, exponent = np.frexp(column)
    high, low = _products(mantissa, alpha_mantissa)
    high, low = _restore_gradient_terms((high, low), exponent+alpha_exponent)
    return _fsum(high.tolist()+low.tolist())


def _beta_tile(columns, alpha_mantissa, alpha_exponent, *, _split=None):
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
    for start in range(0, features.shape[1], width):
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
    return result


