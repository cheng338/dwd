"""Preserve ordinary DWD objective arithmetic and recover reduction overflow."""
from fractions import Fraction
import math

import numpy as np


def _fallback(losses, penalty):
    """Round the exact mean of represented losses plus the represented penalty."""
    values = np.asarray(losses)
    if (not values.size or values.dtype != np.dtype('float64')
            or not np.isfinite(values).all() or not math.isfinite(float(penalty))):
        raise FloatingPointError('Objective inputs are not finite binary64 values.')
    # Exact represented-value arithmetic only after the ordinary route fails.
    total = sum((Fraction.from_float(float(v)) for v in values.flat), Fraction())
    objective = total / values.size + Fraction.from_float(float(penalty))
    try:
        result = float(objective)
    except OverflowError as exc:
        raise FloatingPointError('Objective exceeds floating-point range.') from exc
    if not math.isfinite(result):
        raise FloatingPointError('Objective exceeds floating-point range.')
    return np.float64(result)


def objective_mean(losses, penalty, *, inside=False):
    """Evaluate the existing objective order, recovering only nonfinite results.

    Linear and legacy kernel callers use mean(losses + penalty); other kernel
    callers use mean(losses) + penalty. Retain those orders for ordinary inputs
    so rounding, histories and stopping decisions remain unchanged. A finite
    binary64 loss vector and penalty can have a representable objective even
    when the preliminary sum or per-observation addition overflows. Only then
    use exact represented-value arithmetic before rounding the final result.

    Loss and penalty construction remain the caller's responsibility. Nonfinite
    inputs or an unrepresentable result raise; explicit underflow errors in the
    ordinary calculation propagate. No gradient or solver acceptance rule is
    changed by this helper.
    """
    with np.errstate(over='ignore', invalid='ignore'):
        value = np.mean(losses + penalty) if inside else np.mean(losses) + penalty
    if np.isfinite(value):
        return value
    return _fallback(losses, penalty)
