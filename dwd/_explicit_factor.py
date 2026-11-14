"""One factorization of the original explicit MM system; checked RHS actions.

No ridge, pseudoinverse, rank truncation or numerical-failure fallback is used.
Residual tests are practical floating-point checks, not formal certificates.
"""
import numpy as np
from scipy.linalg import cho_factor
from scipy.linalg.blas import get_blas_funcs
from dwd._validation import validate_unregularized_design


class ExplicitFactor:
    """Equilibrated Cholesky with original represented-system RHS checks."""

    def __init__(self, P):
        self.P = np.asarray(P, dtype=np.float64)
        if not np.isfinite(self.P).all():
            raise FloatingPointError('Nonfinite explicit GenDWD MM system.')
        diagonal = np.diag(self.P)
        if np.any(diagonal <= 0):
            raise ValueError('The explicit GenDWD MM system must be positive definite.')
        self.diagonal = np.sqrt(diagonal)
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            scaled = (self.P / self.diagonal[:, None]) / self.diagonal[None, :]
        if not np.isfinite(scaled).all():
            raise FloatingPointError('Nonfinite equilibrated explicit GenDWD MM system.')
        self.factor = cho_factor(scaled, lower=True, overwrite_a=True, check_finite=False)
        # Avoid per-action Fortran-layout conversion in low-level BLAS.
        self.factor = (np.asfortranarray(self.factor[0]), self.factor[1])
        self._trsv, = get_blas_funcs(('trsv',), (self.factor[0],))
        self.row_scales = np.max(np.abs(self.P), axis=1)
        self.normalized_rows = self.P / self.row_scales[:, None]
        self.absolute_rows = np.abs(self.normalized_rows)
        self.tolerance = 64 * len(self.P) * np.finfo(np.float64).eps

    def solve(self, rhs):
        rhs = np.asarray(rhs, dtype=np.float64)
        if rhs.shape != (len(self.P),) or not np.isfinite(rhs).all():
            raise FloatingPointError('Invalid explicit GenDWD right-hand side.')
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            scaled_rhs = rhs / self.diagonal
            # Two single-vector triangular actions; only private RHS buffers
            # may be overwritten. Check the returned solution against P below.
            intermediate = self._trsv(
                self.factor[0], scaled_rhs, lower=True, trans=0, diag=0, overwrite_x=True)
            solved = self._trsv(
                self.factor[0], intermediate, lower=True, trans=1, diag=0, overwrite_x=True)
            solution = solved / self.diagonal
        if not np.isfinite(solution).all():
            raise FloatingPointError('Nonfinite explicit GenDWD factor solution.')
        x_scale = float(np.max(np.abs(solution)))
        if x_scale == 0:
            if np.any(rhs != 0):
                raise FloatingPointError('Explicit GenDWD solution underflowed for a nonzero RHS.')
            return solution
        unit_solution = solution / x_scale
        # Check returned x against original stored P, not reconstructed factors.
        # Row and vector scaling bounds products without requiring an inverse.
        scales = self.row_scales
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            rows = self.normalized_rows
            expected_first = rhs / np.maximum(scales, x_scale)
            # The next division can magnify information lost here. Refuse
            # an unresolved nonzero RHS before that second scaling stage.
            if np.any((rhs != 0) &
                      (np.abs(expected_first) < np.finfo(np.float64).tiny)):
                raise FloatingPointError('Explicit GenDWD solve residual RHS '
                                         'normalization lost relative accuracy.')
            expected = expected_first / np.minimum(scales, x_scale)
            residual = np.abs(rows @ unit_solution - expected)
            denominator = self.absolute_rows @ np.abs(unit_solution) + np.abs(expected)
        if not np.isfinite(residual).all() or not np.isfinite(denominator).all():
            raise FloatingPointError('Nonfinite explicit GenDWD solve residual.')
        # Subnormal rounding cannot support the relative-error target.
        # A positive denominator does not alone establish a resolved check.
        if np.any((denominator > 0) & (denominator < np.finfo(np.float64).tiny)):
            raise FloatingPointError('Explicit GenDWD solve residual normalization '
                                     'has unresolved subnormal relative accuracy.')
        nonzero = denominator != 0
        if np.any(~nonzero):
            # Scaling may erase a nonzero equation entirely. Test support
            # in the original stored operands, without multiplying them.
            vanished_rhs = rhs[~nonzero] != 0
            vanished_products = np.any(
                (self.P[~nonzero] != 0) & (solution != 0), axis=1)
            if np.any(vanished_rhs | vanished_products):
                raise FloatingPointError('Explicit GenDWD solve residual normalization '
                                         'erased a nonzero original equation.')
        if (np.any(residual[~nonzero] != 0) or
                np.any(residual[nonzero] / denominator[nonzero] > self.tolerance)):
            raise FloatingPointError('Explicit GenDWD factor solution failed the '
                                     'original-system backward-residual accuracy check.')
        return solution

    def inverse_for_hook(self):
        """Keep the ndarray protocol for an explicitly replaced public helper."""
        inverse = np.linalg.inv(self.P)
        if not np.isfinite(inverse).all():
            raise FloatingPointError('Nonfinite explicit GenDWD inverse for replaced hook.')
        return inverse
