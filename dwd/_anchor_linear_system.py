"""Anchor-coordinate inverse candidates for the original constrained system.

The caller must check every returned candidate against its original equations.
No kernel entry is replaced, symmetrized, regularized or spectrally discarded.
"""
import math
from time import perf_counter
import warnings

import numpy as np
from scipy.linalg import lu_factor, lu_solve, LinAlgError, LinAlgWarning
from scipy.linalg.lapack import dgecon
from ._quadratic_bounds import _compensated_dot


class AnchorLinearSystem:
    """Solve in x[1:] with x[0] = target_sum - sum(x[1:]).

    For A=K+shift*I and P=[e_1-e_0,...,e_(n-1)-e_0], factor
    P.T@A@P. LU preserves any accepted asymmetry in the stored original K.
    This is a candidate preconditioner, not an independent accuracy gate.
    """
    def __init__(self, K, shift):
        started = perf_counter()
        if np.iscomplexobj(K):
            raise ValueError('The anchor kernel must be real.')
        K = np.asarray(K, dtype=np.float64)
        if K.ndim != 2 or K.shape[0] != K.shape[1] or not len(K):
            raise ValueError('The anchor kernel must be a nonempty square matrix.')
        if (isinstance(shift, (bool, np.bool_)) or np.ndim(shift) != 0
                or np.iscomplexobj(shift) or not np.isfinite(shift) or shift <= 0):
            raise ValueError('The anchor shift must be a finite positive real scalar.')
        if not np.isfinite(K).all():
            raise FloatingPointError('Nonfinite anchor kernel entries.')
        self.K, self.shift, self.n = K, float(shift), len(K)
        self.factor = None
        self.info = {'representation': 'anchor_lu', 'anchor_index': 0,
                     'reduced_dimension': self.n-1, 'rcond': None,
                     'reduced_symmetry_error': 0.}
        with np.errstate(over='raise', invalid='raise', under='ignore'):
            self.row_difference = K[0, 1:] - K[0, 0]
            self.column_difference = K[1:, 0] - K[0, 0]
            if self.n > 1:
                m = self.n-1
                matrix = np.empty((m, m), dtype=float, order='F')
                for start in range(0, m, 128):
                    stop = min(start+128, m)
                    # Grouped differences avoid subtracting rounded means.
                    centered = ((K[start+1:stop+1, 1:] - K[start+1:stop+1, 0, None])
                                - self.row_difference[None, :])
                    matrix[start:stop] = centered + self.shift
                matrix.flat[::m+1] += self.shift
                column_sums = np.zeros(m)
                for start in range(0, m, 128):
                    block = matrix[start:start+128]
                    column_sums += np.sum(np.abs(block), axis=0)
                    self.info['reduced_symmetry_error'] = max(
                        self.info['reduced_symmetry_error'],
                        float(np.max(np.abs(block-matrix[:, start:start+len(block)].T))))
                norm = float(column_sums.max())
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('error', LinAlgWarning)
                        self.factor = lu_factor(matrix, overwrite_a=True, check_finite=False)
                except (LinAlgWarning, LinAlgError) as exc:
                    raise FloatingPointError('The anchor reduced factor is singular or invalid.') from exc
                rcond, status = dgecon(self.factor[0], norm, norm='1')
                if status != 0 or not np.isfinite(rcond) or rcond < 0:
                    raise FloatingPointError('Invalid anchor reciprocal-condition estimate.')
                # Conditioning is diagnostic; original-equation checks decide
                # whether a candidate is accurate enough, including rcond=0.
                self.info['rcond'] = float(rcond)
        self.info['factorization_seconds'] = perf_counter()-started

    def candidate(self, rhs, target_sum=0.):
        if np.iscomplexobj(rhs) or np.iscomplexobj(target_sum) or np.ndim(target_sum) != 0:
            raise ValueError('Anchor right-hand side and target must be real.')
        rhs, target_sum = np.asarray(rhs, dtype=float), float(target_sum)
        if rhs.shape != (self.n,) or not np.isfinite(rhs).all() or not np.isfinite(target_sum):
            raise FloatingPointError('Invalid anchor right-hand side or coefficient-sum target.')
        try:
            with np.errstate(over='raise', invalid='raise', under='ignore'):
                reduced_rhs = rhs[1:] - rhs[0]
                if target_sum:
                    reduced_rhs = reduced_rhs + target_sum*(self.shift-self.column_difference)
                u = (lu_solve(self.factor, reduced_rhs, check_finite=False)
                     if self.n > 1 else np.empty(0))
                if not np.isfinite(u).all():
                    raise FloatingPointError('Nonfinite anchor inverse candidate.')
                total = math.fsum(u)
                x = np.empty(self.n)
                x[1:], x[0] = u, target_sum-total
                product = _compensated_dot(self.row_difference, u)[0] if self.n > 1 else 0.
                scalar = math.fsum((float(rhs[0]), -product, self.shift*total,
                                    -float(self.K[0, 0])*target_sum, -self.shift*target_sum))
        except (OverflowError, ValueError, LinAlgError) as exc:
            raise FloatingPointError('Unrepresentable anchor inverse candidate.') from exc
        if not np.isfinite(x).all() or not np.isfinite(scalar):
            raise FloatingPointError('Nonfinite anchor inverse candidate.')
        return x, float(scalar)
