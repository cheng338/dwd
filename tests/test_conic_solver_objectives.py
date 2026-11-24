"""Conic SVM objectives agree with an independently constructed linear program."""
import copy
import warnings
import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import linprog
cp = pytest.importorskip('cvxpy', exc_type=ImportError)
from dwd.svm import SVM
from sklearn.exceptions import NotFittedError
SOLVERS = ('CLARABEL', 'SCS', 'OSQP', 'SCIPY', 'HIGHS')
CASES = ('original', 'separable', 'duplicates', 'all_zero', 'zero_columns_rows',
         'large_scale', 'small_scale', 'near_collinear', 'imbalanced')

def fixture(name):
    rng = np.random.default_rng(340046)
    X = rng.normal(size=(16, 5))
    y = np.tile([0, 1], 8)
    X[:, 0] += y * 1.5
    if name == 'separable': X[:, 0] = (2 * y - 1) * 3.
    elif name == 'duplicates': X[8:], y[8:] = X[:8], y[:8]
    elif name == 'all_zero': X[:] = 0.
    elif name == 'zero_columns_rows':
        X[:, 2:] = 0.
        X[[2, 3], :] = 0.
    elif name == 'large_scale': X *= 1e4
    elif name == 'small_scale': X *= 1e-4
    elif name == 'near_collinear':
        X[:, 1] = X[:, 0] * (1. + 1e-10) + rng.normal(size=16) * 1e-12
        X[:, 2] = -X[:, 0]
    elif name == 'imbalanced':
        y = np.r_[np.zeros(3, dtype=int), np.ones(13, dtype=int)]
        X[:] = 0.
    elif name != 'original': raise ValueError(name)
    return X, y

def solver_options(solver):
    if solver == 'HIGHS':
        return dict(solver=solver, threads=1, primal_feasibility_tolerance=1e-9,
                    dual_feasibility_tolerance=1e-9)
    if solver == 'SCIPY':
        return dict(solver=solver, scipy_options=dict(threads=1,
                    primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9))
    if solver == 'CLARABEL':
        return dict(solver=solver, tol_gap_abs=1e-9, tol_feas=1e-9, tol_gap_rel=1e-9)
    if solver == 'SCS': return dict(solver=solver, eps=1e-7, max_iters=100000)
    return dict(solver=solver, eps_abs=1e-8, eps_rel=1e-8, max_iter=100000)

def true_objective(X, labels, coef, bias):
    decision = X @ np.asarray(coef).reshape(-1) + float(np.asarray(bias).reshape(()))
    return float(np.maximum(0., 1. - (2 * labels - 1) * decision).mean()
                 + .25 * np.abs(coef).sum())

def reference_lp(X, labels):
    n, p = X.shape
    y = 2 * labels - 1
    A = np.zeros((n + 2 * p, p + 1 + n + p))
    A[:n, :p], A[:n, p] = -y[:, None] * X, -y
    A[:n, p + 1:p + 1 + n] = -np.eye(n)
    A[n:n + p, :p], A[n + p:, :p] = np.eye(p), -np.eye(p)
    A[n:n + p, p + 1 + n:] = -np.eye(p)
    A[n + p:, p + 1 + n:] = -np.eye(p)
    rhs = np.r_[-np.ones(n), np.zeros(2 * p)]
    cost = np.r_[np.zeros(p + 1), np.ones(n) / n, np.ones(p) * .25]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = linprog(cost, A_ub=A, b_ub=rhs,
                         bounds=[(None, None)] * (p + 1) + [(0., None)] * (n + p),
                         method='highs', options=dict(threads=1,
                             primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9))
    assert result.success, result.message
    objective = true_objective(X, labels, result.x[:p], result.x[p])
    assert abs(objective - result.fun) < 1e-7
    assert np.max(A @ result.x - rhs) < 1e-7
    return dict(objective=objective, coef=result.x[:p].tolist(), bias=float(result.x[p]),
                warnings=[str(item.message) for item in caught])

@pytest.fixture(scope='module')
def reference_results():
    values = {name: reference_lp(*fixture(name)) for name in CASES}
    return values

@pytest.mark.parametrize('solver', SOLVERS)
@pytest.mark.parametrize('storage', ('dense', 'csr'))
@pytest.mark.parametrize('case', CASES)
def test_actual_svm_objective(case, storage, solver, reference_results, monkeypatch):
    if solver not in cp.installed_solvers(): pytest.skip('Optional solver unavailable: ' + solver)
    X, labels = fixture(case)
    input_X = sparse.csr_matrix(X) if storage == 'csr' else X.copy()
    before = input_X.copy()
    model = SVM(C=.25, solver_kws=solver_options(solver))
    saved_params = copy.deepcopy(model.get_params(deep=False))
    native_stops = []
    if case == 'large_scale' and solver == 'OSQP':
        import osqp
        ordinary_X, ordinary_y = fixture('original')
        model.fit(ordinary_X, ordinary_y)
        assert hasattr(model, 'coef_') and hasattr(model, 'problem_')
        # CVXPY versions expose OSQP's iteration limit through different exceptions.
        # Observe the real solver's return for this refit before CVXPY translates it.
        solve = osqp.OSQP.solve
        def observe_stop(instance, *args, **kwargs):
            result = solve(instance, *args, **kwargs)
            native_stops.append((result.info.status, result.info.iter))
            return result
        monkeypatch.setattr(osqp.OSQP, 'solve', observe_stop)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            model.fit(input_X, labels)
        except (RuntimeError, cp.error.SolverError) as error:
            if case != 'large_scale' or solver != 'OSQP':
                raise
            if native_stops != [('maximum iterations reached', saved_params['solver_kws']['max_iter'])]:
                raise
            user_limit = (isinstance(error, RuntimeError)
                          and str(error) == "SVM conic solver failed with status 'user_limit'.")
            cvxpy_refusal = (isinstance(error, cp.error.SolverError)
                             and str(error) == "Solver 'OSQP' failed. Try another solver, "
                                               "or solve with verbose=True for more information.")
            if not (user_limit or cvxpy_refusal):
                raise
            assert model.get_params(deep=False) == saved_params
            assert not any(name.endswith('_') and not name.startswith('_') for name in vars(model))
            assert '_Xfit' not in vars(model)
            with pytest.raises(NotFittedError):
                model.predict(input_X)
            if sparse.issparse(input_X):
                assert (input_X != before).nnz == 0
            else:
                np.testing.assert_array_equal(input_X, before)
            # A rejected finite-budget result is not a qualified numerical fit.
            return
    if sparse.issparse(input_X): assert (input_X != before).nnz == 0
    else: np.testing.assert_array_equal(input_X, before)
    assert model.coef_.shape == (1, 5) and model.intercept_.shape == (1,)
    assert np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all()
    actual = true_objective(X, labels, model.coef_, model.intercept_[0])
    reference = reference_results[case]['objective']
    gap = actual - reference
    row = dict(case=case, storage=storage, solver=solver, status=model.solver_status_,
               true_objective=actual, reference_objective=reference, gap=gap,
               coef=model.coef_.tolist(), intercept=model.intercept_.tolist(),
               predictions=model.predict(input_X).tolist(),
               tolerance=5e-5 * (1. + abs(reference)),
               qualified=abs(gap) <= 5e-5 * (1. + abs(reference)),
               warnings=[str(item.message) for item in caught])
    assert row['qualified'], row
    if case == 'original' and storage == 'dense' and solver in ('SCIPY', 'HIGHS'):
        assert abs(gap) < 1e-7
    if case == 'imbalanced':
        assert abs(actual - 2 * 3 / 16) < 1e-7  # Free intercept optimum.

@pytest.mark.parametrize('storage', ('dense', 'csr'))
def test_signed_constant_margin_algebra(storage):
    X, labels = fixture('original')
    X = sparse.csr_matrix(X) if storage == 'csr' else X
    y = (2 * labels - 1).reshape(-1, 1)
    beta, bias = cp.Variable((5, 1)), cp.Variable()
    beta.value, bias.value = np.arange(5, dtype=float).reshape(-1, 1) / 3, -.7
    old = cp.multiply(y, X @ beta + bias)
    new = cp.multiply(y, X) @ beta + cp.multiply(y, bias)
    assert old.shape == new.shape == (16, 1)
    np.testing.assert_allclose(old.value, new.value, rtol=2e-15, atol=2e-15)


def test_osqp_infeasible_refit_reports_status_and_clears_state(monkeypatch):
    import osqp

    X, labels = fixture('original')
    original_X, original_labels = X.copy(), labels.copy()
    model = SVM(C=.25, solver_kws={'solver': 'OSQP'}).fit(X, labels)
    assert model.predict(X).shape == labels.shape

    # Loose stopping options make the real solver report primal infeasibility.
    options = {'solver': 'OSQP', 'eps_prim_inf': 1., 'check_termination': 1}
    original_options = options.copy()
    model.set_params(solver_kws=options)
    solve = osqp.OSQP.solve
    native_statuses = []

    def observe_status(instance, *args, **kwargs):
        result = solve(instance, *args, **kwargs)
        native_statuses.append(result.info.status)
        return result

    monkeypatch.setattr(osqp.OSQP, 'solve', observe_status)
    with pytest.raises(RuntimeError, match=r"^SVM conic solver failed with status 'infeasible'\.$"):
        model.fit(X, labels)

    assert native_statuses == ['primal infeasible']
    assert model.solver_kws == original_options
    assert options == original_options
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(labels, original_labels)
    assert not any(name.endswith('_') and not name.startswith('_') for name in vars(model))
    assert '_Xfit' not in vars(model)
    with pytest.raises(NotFittedError):
        model.predict(X)
