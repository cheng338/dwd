"""Coefficient-coordinate reference update for corrected kernel DWD MM.

This retains the original coefficient-subtraction formulation. It is a separate
numerical implementation of the same MM algorithm as the optimized score-state
update, not an independent optimization method or mathematical oracle.
"""
import numpy as np
from ._kernel_linear_system import _sum


def reference_update(K, alpha, offset, z, lambd, step_scale, system):
    """Return one corrected MM update and freshly computed kernel scores.

    ``K``, ``alpha``, ``offset``, ``z``, ``lambd`` and ``step_scale`` have already
    passed the common solver validation. ``z`` is the mean-loss derivative with
    respect to the unshifted training decision values. For general q,
    ``step_scale = n*q/(q+1)**2`` and the shared system has shift
    ``2*lambd*step_scale``.

    ``system.solve_constrained(rhs, target_sum)`` must return ``(x, s)`` solving
    ``(K + shift*I) @ x + s*ones = rhs`` and ``sum(x) = target_sum``. Its shared
    numerical checks/refinement must honor the supplied, possibly nonzero sum.
    The intercept is unregularized: ``step_scale*s`` is its subtraction step.

    No caller-owned input is changed. Scores are recomputed from the updated
    coefficients instead of maintained by the optimized score recurrence.
    """
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        rhs = z + 2. * lambd * alpha
        target_sum = float(_sum(alpha) / step_scale)
        if not np.isfinite(rhs).all() or not np.isfinite(target_sum):
            raise FloatingPointError('Nonfinite coefficient MM right-hand side.')
        direction, intercept_direction = system.solve_constrained(
            rhs, target_sum=target_sum)
        updated_alpha = alpha - step_scale * direction
        updated_offset = float(offset - step_scale * intercept_direction)
        scores = K @ updated_alpha
        if (not np.isfinite(updated_alpha).all()
                or not np.isfinite(updated_offset)
                or not np.isfinite(scores).all()):
            raise FloatingPointError('Nonfinite coefficient MM update or scores.')
    return updated_alpha, updated_offset, scores
