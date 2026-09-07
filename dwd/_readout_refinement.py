"""Bounded adjacent-float refinement of a stored constrained-solve result.

The inverse action can stagnate although a neighboring float64 coefficient
readout satisfies the original equations. This helper changes one coefficient
by exactly one nextafter step and recenters the free scalar. It never changes
the kernel, shift, constraint target or inverse representation. Predicted
improvement only selects a trial; fresh compensated measurements and the
caller's unchanged complete accuracy check authorize a result.
"""
from dataclasses import dataclass

import numpy as np

from ._compensated_residual import compensated_residual


@dataclass
class ReadoutBudget:
    """Shared across all inverse representations of one public solve action.

    The private ``matrix_entry_work`` counter is abstract dense-work accounting:
    one n-by-n measurement or matvec is charged n**2 units. A compensated
    measurement performs several arithmetic passes, so these units are NOT an
    exact count of operations, accessed entries, bytes or elapsed time. The
    fixed charge schedule bounds the number and size of dense actions. All
    extra arrays have length n; inapplicable problems allocate no such arrays.
    """
    max_moves: int = 8
    max_matrix_entry_work: int = 1_048_576
    moves: int = 0
    matrix_entry_work: int = 0

    def charge(self, count):
        if count > self.max_matrix_entry_work - self.matrix_entry_work:
            return False
        self.matrix_entry_work += count
        return True


def polish_readout(K, shift, rhs, x, scalar, target, check, budget):
    """Return ``(x, scalar, checked), details`` or ``None, details``.

    Inputs are read-only. Work on copies, and expose no partial correction
    unless the stored float64 result passes ``check``. Coordinate order is
    ascending, direction order is negative then positive, and exact ties keep
    the first candidate. Accepted private moves must strictly reduce the
    compensated original-equation maximum, including its roundoff allowance.
    """
    n = len(x)
    details = {'status': 'inapplicable', 'moves': 0, 'candidate_scans': 0,
               'matrix_entry_work': 0, 'accepted': False}
    work_before = budget.matrix_entry_work
    moves_before = budget.moves
    matrix_work = n * n

    def finish(result, status):
        details.update(status=status, moves=budget.moves-moves_before,
                       matrix_entry_work=budget.matrix_entry_work-work_before,
                       accepted=result is not None)
        return result, details

    # Reserve enough for the initial measurement and at least one complete
    # candidate scan plus fresh measurement. This gate precedes any O(n) copy.
    if (n == 0 or budget.moves >= budget.max_moves
            or 4 * matrix_work > budget.max_matrix_entry_work - budget.matrix_entry_work):
        return finish(None, 'budget_exhausted')
    if (K.shape != (n, n) or rhs.shape != (n,) or x.shape != (n,)
            or np.iscomplexobj(K) or np.iscomplexobj(rhs) or np.iscomplexobj(x)
            or np.iscomplexobj(scalar) or np.iscomplexobj(target)
            or not np.isfinite(shift) or shift <= 0.
            or not np.isfinite(scalar) or not np.isfinite(target)
            or not np.isfinite(x).all() or not np.isfinite(rhs).all()):
        return finish(None, 'invalid_state')
    budget.charge(matrix_work)
    residual, constraint, _, allowance, constraint_allowance = compensated_residual(
        K, shift, rhs, x, scalar, target)
    maximum = float(np.max(np.abs(residual) + allowance))
    details['initial_maximum'] = maximum
    work_x = x.copy()
    work_scalar = float(scalar)
    # Cheap trigger matches KernelLinearSystem._assess/_equations_acceptable;
    # it never authorizes acceptance. The unchanged full check below remains
    # decisive for the residual, constraint AND RKHS accuracy of the readout.
    tolerance = 1e-10 * max(1., float(np.max(np.abs(rhs))))

    while budget.moves < budget.max_moves:
        if not budget.charge(2 * matrix_work):
            return finish(None, 'budget_exhausted')
        details['candidate_scans'] += 1
        best = None
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            for index in range(n):
                for direction in (-np.inf, np.inf):
                    adjacent = float(np.nextafter(work_x[index], direction))
                    delta = adjacent - work_x[index]
                    if not np.isfinite(adjacent) or not np.isfinite(delta) or delta == 0.:
                        continue
                    proposed = residual - K[:, index] * delta
                    proposed[index] -= shift * delta
                    # Splitting the midpoint avoids overflow of min + max.
                    correction = .5 * float(proposed.min()) + .5 * float(proposed.max())
                    proposed_scalar = float(work_scalar + correction)
                    if not np.isfinite(proposed_scalar):
                        continue
                    predicted = float(np.max(np.abs(proposed - (proposed_scalar-work_scalar))))
                    if np.isfinite(predicted) and predicted < maximum and (best is None or predicted < best[0]):
                        best = (predicted, index, adjacent, proposed_scalar)
        if best is None:
            return finish(None, 'no_strict_prediction_improvement')
        if not budget.charge(matrix_work):
            return finish(None, 'budget_exhausted')
        _, index, adjacent, proposed_scalar = best
        trial_x = work_x.copy()
        trial_x[index] = adjacent
        fresh = compensated_residual(K, shift, rhs, trial_x, proposed_scalar, target)
        fresh_maximum = float(np.max(np.abs(fresh[0]) + fresh[3]))
        if not np.isfinite(fresh_maximum) or fresh_maximum >= maximum:
            return finish(None, 'fresh_residual_did_not_improve')
        budget.moves += 1
        work_x, work_scalar = trial_x, proposed_scalar
        residual, constraint, _, allowance, constraint_allowance = fresh
        maximum = fresh_maximum
        details['final_maximum'] = maximum
        if maximum <= tolerance:
            # The full check may run a compensated matrix measurement, two
            # inverse matvecs, a residual measurement and a quadratic check.
            # Reserve six n**2 dense-work units, not six arithmetic passes;
            # these abstract charges do not count the internal compensation
            # arithmetic. No numerical check is waived.
            if not budget.charge(6 * matrix_work):
                return finish(None, 'budget_exhausted')
            checked = check(rhs, target, work_x, work_scalar)
            if checked[0]:
                return finish((work_x, work_scalar, checked), 'accepted')
            return finish(None, 'full_accuracy_check_failed')
    return finish(None, 'move_budget_exhausted')
