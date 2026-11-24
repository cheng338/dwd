"""Exercise CVXPY's conic HiGHS interface in the minimum solver environment."""
import cvxpy as cp
import numpy as np
from cvxpy.reductions.solvers.conic_solvers.highs_conif import HIGHS

from dwd.svm import SVM


class ConicHiGHS(HIGHS):
    # CVXPY requires a distinct name for an explicitly supplied solver.
    def name(self):
        return "DWD_TEST_CONIC_HIGHS"


def main():
    solver = ConicHiGHS()
    options = {"solver": solver, "threads": 1}
    X, y = [[-1.0], [1.0]], [0, 1]
    model = SVM(C=0.25, solver_kws=options).fit(X, y)
    assert model.problem_.solver_stats.solver_name == solver.name()
    assert model.problem_.status == cp.OPTIMAL
    # The separating slope 1 and intercept 0 give zero hinge loss.
    np.testing.assert_allclose(model.problem_.value, 0.25, rtol=0, atol=1e-8)
    np.testing.assert_array_equal(model.predict(X), y)

    # One equality and one lower bound give opposite signs in the multipliers.
    x = cp.Variable(2)
    fixed, lower = x[0] == 1.0, x[1] >= 2.0
    problem = cp.Problem(cp.Minimize(2.0 * x[0] + 3.0 * x[1]), [fixed, lower])
    problem.solve(**options)
    assert problem.solver_stats.solver_name == solver.name()
    assert problem.status == cp.OPTIMAL
    np.testing.assert_allclose(problem.value, 8.0, rtol=0, atol=1e-8)
    np.testing.assert_allclose(x.value, [1.0, 2.0], rtol=0, atol=1e-8)
    np.testing.assert_allclose([fixed.dual_value, lower.dual_value], [-2.0, 3.0],
                               rtol=0, atol=1e-8)


if __name__ == "__main__":
    main()
