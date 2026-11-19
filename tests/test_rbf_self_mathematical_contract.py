"""RBF self-construction and fitted-state checks with independent exact oracles.

The seeded geometries cover small construction asymmetry, ill conditioning,
queries and row-order changes. Numerical refusals in the ill-conditioned panels
are checked outcomes, not claims of successful fitting.
"""
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction
import math
import sys

import numpy as np
import pytest
from scipy import sparse

from dwd.gen_kern_dwd import KernGDWD
from dwd._dual_certificate import kernel_dual_lower_bound, feasible_dual_slopes
from dwd._exact_kernel_factor import exact_kernel_factor


EPS = Fraction.from_float(float(np.finfo(float).eps))
ETA = Fraction(1, 1 << 1074)
MAX_ITERATIONS = 500


def frac(value):
    return Fraction.from_float(float(value))


@pytest.fixture
def evidence(request):
    state = np.geterr().copy(), np.geterrcall(), sys.getprofile(), sys.gettrace()
    record = dict(schema="rbf-self-mathematical-contract-v1",
                  nodeid=request.node.nodeid, public_fit_calls=[], constructions=[],
                  maximum_iterations_per_fit=MAX_ITERATIONS)
    try:
        yield record
    finally:
        record["numerical_state_preserved"] = state == (
            np.geterr().copy(), np.geterrcall(), sys.getprofile(), sys.gettrace())
        assert record["numerical_state_preserved"]


def exact_dot(left, right):
    return sum((frac(a) * frac(b) for a, b in zip(left, right)), Fraction(0))


def exact_psd(K):
    """Independent rational Schur elimination; no floating rank tolerance."""
    if not np.array_equal(K, K.T):
        return dict(status="asymmetric", rank=None)
    work = [[frac(value) for value in row] for row in K]
    active, rank = list(range(len(K))), 0
    while active:
        if any(work[i][i] < 0 for i in active):
            return dict(status="indefinite", rank=None, witness="negative_schur_diagonal")
        pivot = max(active, key=lambda i: work[i][i])
        diagonal = work[pivot][pivot]
        if diagonal == 0:
            if any(work[i][j] != 0 for i in active for j in active):
                return dict(status="indefinite", rank=None, witness="zero_diagonal_nonzero_row")
            return dict(status="psd", rank=rank)
        remaining = [i for i in active if i != pivot]
        for position, i in enumerate(remaining):
            for j in remaining[position:]:
                value = work[i][j] - work[i][pivot] * work[pivot][j] / diagonal
                work[i][j] = work[j][i] = value
        active, rank = remaining, rank + 1
    return dict(status="psd", rank=rank)


def factor_observation(K):
    before = K.tobytes()
    independent = exact_psd(K)
    result = exact_kernel_factor(K)
    record = dict(independent=independent, status=result.status,
                  certified=result.certified, metadata=dict(result.metadata),
                  input_unchanged=K.tobytes() == before)
    valid = record["input_unchanged"]
    if result.certified:
        valid = valid and independent["status"] == "psd"
        valid = valid and result.factor.rank == independent["rank"]
        factor = result.factor
        identities = True
        for i in range(len(K)):
            for j in range(len(K)):
                recovered = sum((frac(factor.C[i, a]) * factor.B_inverse_exact[a][b]
                                 * frac(factor.C[j, b])
                                 for a in range(factor.rank) for b in range(factor.rank)), Fraction(0))
                identities = identities and recovered == frac(K[i, j])
        record["entrywise_identity_matches"] = identities
        valid = valid and identities
    elif independent["status"] == "asymmetric":
        valid = valid and result.status == "not_exactly_symmetric"
    elif independent["status"] == "psd":
        # Resource exhaustion is not a negative PSD finding or a rank cutoff.
        valid = valid and result.status in ("entry_budget_exhausted", "rank_budget_exhausted",
                                           "rational_bit_budget_exhausted", "operation_budget_exhausted")
    record["contract_valid"] = bool(valid)
    return record


def rbf_interval(first, second, gamma=.5):
    squared = sum(((frac(a) - frac(b)) ** 2 for a, b in zip(first, second)), Fraction(0))
    exponent = -frac(gamma) * squared
    with localcontext() as context:
        context.prec, context.rounding = 110, ROUND_FLOOR
        lower_exponent = Decimal(exponent.numerator) / Decimal(exponent.denominator)
        lower = lower_exponent.exp().next_minus()
    with localcontext() as context:
        context.prec, context.rounding = 110, ROUND_CEILING
        upper_exponent = Decimal(exponent.numerator) / Decimal(exponent.denominator)
        upper = upper_exponent.exp().next_plus()
    return Fraction(lower), Fraction(upper)


def kernel_reference(K, X, gamma=.5):
    dense = X.toarray() if sparse.issparse(X) else np.asarray(X)
    largest = Fraction(0)
    for i in range(len(dense)):
        for j in range(len(dense)):
            low, high = rbf_interval(dense[i], dense[j], gamma)
            largest = max(largest, low - frac(K[i, j]), frac(K[i, j]) - high)
    return dict(max_distance_outside_directed_rbf_interval=largest,
                interval_precision_digits=110)


def model(lambd=.1, gamma=.5):
    return KernGDWD(kernel="rbf", kernel_kws={"gamma": gamma}, lambd=lambd, q=1.,
                    backend="auto",
                    initialization="zero", stopping="optimality", tol=1e-6,
                    max_iter=MAX_ITERATIONS, acceleration=None,
                    residual_check_order="refinement_first", random_state=1818)


def prepare(record, X, *, lambd=.1, gamma=.5):
    estimator = model(lambd=lambd, gamma=gamma)
    estimator.cv_init(X)
    K = np.asarray(estimator._cv_K).copy()
    item = dict(X=X.toarray() if sparse.issparse(X) else np.asarray(X), K=K,
                storage="csr" if sparse.issparse(X) else "dense", gamma=gamma,
                asymmetry=float(np.max(np.abs(K - K.T))),
                exactly_symmetric=bool(np.array_equal(K, K.T)),
                computation=estimator.kernel_computation_,
                self_policy=getattr(estimator, "kernel_self_construction_", None),
                factor=factor_observation(K), reference=kernel_reference(K, X, gamma))
    record["constructions"].append(item)
    assert item["factor"]["contract_valid"]
    assert np.isfinite(K).all() and np.all((K >= 0.) & (K <= 1.))
    np.testing.assert_array_equal(np.diag(K), np.ones(len(K)))
    assert item["exactly_symmetric"]
    return estimator, K


def fit_record(record, estimator, X, y, *, allow_refusal=True):
    item = dict(lambd=estimator.lambd, q=estimator.q, labels=np.asarray(y),
                outcome="started", max_iter=estimator.max_iter)
    record["public_fit_calls"].append(item)
    try:
        estimator.fit(X, y)
    except (FloatingPointError, np.linalg.LinAlgError, ValueError) as error:
        # These ill-conditioned panels may legitimately refuse a represented
        # indefinite matrix. Do not convert arbitrary validation/programming
        # errors to a successful experiment outcome.
        numerical_value_error = isinstance(error, ValueError) and (
            str(error) == "A certified negative dual quadratic is incompatible with a PSD kernel."
            or str(error).startswith("K must be positive semidefinite."))
        if not allow_refusal or (isinstance(error, ValueError)
                                and not isinstance(error, np.linalg.LinAlgError)
                                and not numerical_value_error):
            item.update(outcome="unexpected_error", error_type=type(error).__name__, error=str(error))
            raise
        item.update(outcome="numerical_refusal", error_type=type(error).__name__, error=str(error))
        return None
    item.update(outcome="returned", converged=bool(estimator.converged_),
                n_iter=int(estimator.n_iter_), returned_iteration=int(estimator.returned_iteration_),
                termination_reason=estimator.termination_reason_,
                alpha=estimator.dual_coef_.ravel(), offset=float(estimator.intercept_[0]),
                history=estimator.objective_history_, objective=float(estimator.final_objective_),
                diagnostics=estimator.diagnostics_)
    return estimator


def exact_primal(K, alpha, offset, y, lambd):
    scores = [exact_dot(row, alpha) for row in K]
    margins = [frac(label) * (score + frac(offset)) for label, score in zip(y, scores)]
    loss = [1 - margin if margin <= Fraction(1, 2) else Fraction(1, 4) / margin
            for margin in margins]
    norm = sum((frac(a) * score for a, score in zip(alpha, scores)), Fraction(0))
    return dict(scores=scores, norm=norm, primal=sum(loss, Fraction(0))/len(loss) + frac(lambd)*norm)


def query_observation(estimator, query):
    Kq = np.asarray(estimator._compute_kernel(query)).T
    alpha, offset = estimator.dual_coef_.ravel(), float(estimator.intercept_[0])
    actual = np.asarray(estimator.decision_function(query, prediction_batch_size=None))
    batched = np.asarray(estimator.decision_function(query, prediction_batch_size=3))
    fixed_product = np.asarray(estimator._kernel_decision_product(Kq.T)).ravel()
    exact = [exact_dot(row, alpha) + frac(offset) for row in Kq]
    # Conservative ordinary dot-plus-intercept forward bound, also covering
    # more accurate summations. It checks the represented query action only.
    count = 2*len(alpha) + 4
    gamma = count*EPS/(1-count*EPS)
    bounds = [gamma*(sum((abs(frac(k)*frac(a)) for k, a in zip(row, alpha)), Fraction(0))
                     + abs(frac(offset))) + (4*len(alpha)+8)*ETA for row in Kq]
    errors = [abs(frac(value)-target) for value, target in zip(fixed_product, exact)]
    construction_action_difference = float(np.max(np.abs(actual-fixed_product)))
    construction_action_limit = 5e-7*max(1., float(np.max(np.abs(fixed_product))))
    record = dict(query=query.toarray() if sparse.issparse(query) else np.asarray(query),
                  kernel=Kq, scores=actual, batched_scores=batched,
                  fixed_kernel_product=fixed_product,
                  exact_represented_scores=exact, represented_action_errors=errors,
                  represented_action_bounds=bounds,
                  public_vs_fixed_product_difference=construction_action_difference,
                  public_vs_fixed_product_limit=construction_action_limit,
                  public_vs_fixed_comparison_scope="Independent kernel construction plus action; not a dot-only error bound.",
                  batch_score_max_difference=float(np.max(np.abs(actual-batched))),
                  batch_sign_differences=int(np.count_nonzero((actual > 0) != (batched > 0))),
                  predictions=estimator.predict(query))
    record["represented_action_valid"] = all(error <= bound for error, bound in zip(errors, bounds))
    record["public_fixed_compatibility_valid"] = construction_action_difference <= construction_action_limit
    return record


def fitted_checks(record, estimator, K, X, y):
    alpha, offset = estimator.dual_coef_.ravel(), float(estimator.intercept_[0])
    equations = exact_primal(K, alpha, offset, y, estimator.lambd)
    query = query_observation(estimator, X.copy())
    training = np.array([float(s + frac(offset)) for s in equations["scores"]])
    discrepancy = float(np.max(np.abs(query["scores"] - training)))
    record.setdefault("fitted_checks", []).append(dict(equations=equations, query=query,
        score_discrepancy=discrepancy, score_compatibility_limit=5e-7*max(1., float(np.max(np.abs(training)))),
        objective_discrepancy=abs(frac(estimator.final_objective_)-equations["primal"])))
    assert query["represented_action_valid"] and query["public_fixed_compatibility_valid"]
    assert discrepancy <= 5e-7*max(1., float(np.max(np.abs(training))))
    assert abs(frac(estimator.final_objective_)-equations["primal"]) <= frac(5e-8)*max(1, abs(equations["primal"]))
    assert bool(estimator.converged_) == (estimator.rkhs_gradient_norm_ <= estimator.tol)


def positive_dual_control(K, y):
    raw = np.ones(len(y))
    rho = feasible_dual_slopes(raw, y)
    lower, info = kernel_dual_lower_bound(K, y, raw, 1., 1.)
    signed = [frac(label)*frac(value) for label, value in zip(y, rho)]
    equality = sum(signed, Fraction(0))
    box_feasible = all(0 <= frac(value) <= 1 for value in rho)
    quadratic = sum((signed[i]*frac(K[i, j])*signed[j]
                     for i in range(len(y)) for j in range(len(y))), Fraction(0))
    # Projection/quantization guarantees feasibility, not preservation of raw1.
    # Independently bound sqrt(actual rho) at110 decimal digits. Decimal sqrt
    # rounds to nearest; move each nonexact endpoint outward before exact sums.
    conjugate_lower = conjugate_upper = Fraction(0)
    with localcontext() as context:
        context.prec = 110
        for value in rho:
            represented = Decimal.from_float(float(value))
            root = represented.sqrt()
            exact_root = Fraction(root)
            if exact_root*exact_root == frac(value):
                lo = hi = exact_root
            else:
                lo, hi = Fraction(root.next_minus()), Fraction(root.next_plus())
            conjugate_lower += lo
            conjugate_upper += hi
    dual_lower = conjugate_lower/len(y) - quadratic/(4*len(y)*len(y))
    dual_upper = conjugate_upper/len(y) - quadratic/(4*len(y)*len(y))
    raw_quadratic = sum((frac(y[i])*frac(K[i, j])*frac(y[j])
                         for i in range(len(y)) for j in range(len(y))), Fraction(0))
    record = dict(lower=lower, info=info, projected_quadratic=quadratic,
                  symmetric_formula_dual_interval=(dual_lower, dual_upper),
                  raw_one_symmetric_formula=1-raw_quadratic/(4*len(y)*len(y)),
                  exact_equality=equality, box_feasible=box_feasible,
                  oracle_precision_digits=110,
                  formula_requires_symmetric_psd_kernel=True,
                  certificate_lambd=1., certificate_q=1., rho=rho)
    if np.array_equal(K, K.T):
        valid = 0 < frac(lower) <= dual_lower and info["fallback_reason"] is None
    else:
        valid = lower == 0. and info["fallback_reason"] == "asymmetric_kernel_no_nonzero_dual_certificate"
    record["certificate_valid"] = bool(valid and box_feasible and equality == 0)
    return record


@pytest.mark.parametrize("seed", range(11600, 11605))
def test_seeded_construction_and_fit_matches_exact_oracles(seed, evidence):
    X = np.random.RandomState(seed).normal(size=(16, 3))
    y = np.where(np.arange(16) % 2, 1., -1.)
    evidence.update(seed=seed, fit_scope="balanced-label lambda=.1 q=1")
    estimator, K = prepare(evidence, X)
    if evidence["constructions"][-1]["factor"]["independent"]["status"] in ("psd", "asymmetric"):
        evidence["direct_dual_control"] = positive_dual_control(K, y)
        assert evidence["direct_dual_control"]["certificate_valid"]
    fitted = fit_record(evidence, estimator, X, y)
    if fitted is not None:
        fitted_checks(evidence, fitted, K, X, y)


def test_tiny_asymmetric_external_matrix_remains_conservative(evidence):
    K = np.eye(4)
    K[0, 1] = 2.**-52
    y = np.array([-1., 1., -1., 1.])
    before = K.copy()
    evidence["factor"] = factor_observation(K)
    evidence["certificate"] = positive_dual_control(K, y)
    assert evidence["factor"]["contract_valid"]
    assert evidence["certificate"]["certificate_valid"]
    assert not evidence["factor"]["certified"]
    np.testing.assert_array_equal(K, before)


def test_exact_positive_definite_matrix_has_nonzero_certificate(evidence):
    K = np.eye(4) + np.ones((4, 4))/8
    y = np.array([-1., 1., -1., 1.])
    evidence["factor"] = factor_observation(K)
    evidence["certificate"] = positive_dual_control(K, y)
    assert evidence["factor"]["contract_valid"]
    assert evidence["certificate"]["certificate_valid"]
    assert evidence["factor"]["certified"]
    assert evidence["factor"]["independent"] == dict(status="psd", rank=4)
    assert evidence["certificate"]["raw_one_symmetric_formula"] == Fraction(15, 16)


@pytest.mark.parametrize("kind", ["material", "tiny_unit_diagonal"])
def test_symmetry_does_not_establish_positive_semidefiniteness(kind, evidence):
    if kind == "material":
        K, y, raw = -np.eye(4), np.array([-1., 1., -1., 1.]), np.ones(4)
    else:
        K = np.ones((3, 3))
        K[1, 2] = K[2, 1] = 1. - 2.**-52
        y, raw = np.array([-1., 1., 1.]), np.array([1., .5, .5])
    evidence.update(K=K, labels=y, raw_slopes=raw, factor=factor_observation(K))
    assert evidence["factor"]["contract_valid"]
    assert evidence["factor"]["independent"]["status"] == "indefinite"
    assert not evidence["factor"]["certified"]
    # Tiny lambda requests enough quadratic accuracy to enter the existing
    # exact-small diagnostic path. This is not a changed acceptance threshold.
    lower, info = kernel_dual_lower_bound(K, y, raw, 1e-30, 1.)
    assert lower is None
    assert info["dual_objective_is_lower_bound"] is False
    assert info["fallback_reason"] == "certified_negative_dual_quadratic"
    assert info["kernel_assumption"] == (
        "symmetric positive-semidefinite represented kernel (contradicted)")
    evidence["unavailable_dual_certificate"] = info


@pytest.mark.parametrize("storage", ["dense", "csr"])
@pytest.mark.parametrize("kind", ["near_duplicates", "large_offset"])
def test_natural_ill_conditioned_outcomes_are_recorded(kind, storage, evidence):
    if kind == "near_duplicates":
        basis = np.random.RandomState(18181).normal(size=(4, 3))
        X = np.vstack((basis, basis + np.array([2.**-24, 0., 0.])))
    else:
        grid = np.arange(8, dtype=float)
        X = 2.**26 + np.column_stack((grid, grid % 3, (grid*3) % 5))*2.**-10
    y = np.where(np.arange(8) % 2, 1., -1.)
    value = sparse.csr_matrix(X) if storage == "csr" else X
    evidence.update(kind=kind, selected_lambda=1e-12,
                    interpretation="Ill-conditioned represented kernels may refuse; success is not required.")
    estimator, K = prepare(evidence, value, lambd=1e-12)
    fitted = fit_record(evidence, estimator, value, y)
    if fitted is not None:
        fitted_checks(evidence, fitted, K, value, y)


@pytest.mark.parametrize("storage", ["dense", "csr"])
def test_constant_and_duplicate_construction_controls(storage, evidence):
    X = np.repeat(np.array([[0., 0., 0.], [1., .5, -.25], [-.5, 1., .25]]), 2, axis=0)
    value = sparse.csr_matrix(X) if storage == "csr" else X
    _, constant = prepare(evidence, value, gamma=0.)
    np.testing.assert_array_equal(constant, np.ones((6, 6)))
    _, duplicates = prepare(evidence, value)
    for start in (0, 2, 4):
        assert duplicates[start, start+1] == duplicates[start+1, start] == 1.
    assert evidence["constructions"][0]["factor"]["certified"]


@pytest.mark.parametrize("storage", ["dense", "csr"])
def test_fitted_near_boundary_query_actions(storage, evidence):
    positive = np.array([[1., .25, 0.], [2., -.5, .125], [.5, .25, -.25], [1.5, 0., -.5]])
    X = np.vstack((positive, -positive))
    y = np.r_[np.ones(4), -np.ones(4)]
    value = sparse.csr_matrix(X) if storage == "csr" else X
    estimator, K = prepare(evidence, value)
    fitted = fit_record(evidence, estimator, value, y, allow_refusal=False)
    fitted_checks(evidence, fitted, K, value, y)
    query = np.zeros((5, 3))
    query[:, 0] = [-2.**-20, -2.**-40, 0., 2.**-40, 2.**-20]
    queries = sparse.csr_matrix(query) if storage == "csr" else query
    observed = query_observation(fitted, queries)
    evidence["near_boundary"] = observed
    assert observed["represented_action_valid"] and observed["public_fixed_compatibility_valid"]
    alpha, offset = fitted.dual_coef_.ravel(), frac(fitted.intercept_[0])
    intervals = []
    for point in query:
        low = high = offset
        for row, coefficient in zip(X, alpha):
            left, right = rbf_interval(row, point)
            a = frac(coefficient)
            low += a*(left if a >= 0 else right)
            high += a*(right if a >= 0 else left)
        intervals.append((low, high))
    evidence["near_boundary"]["directed_true_rbf_score_intervals_at_returned_coefficients"] = intervals
    evidence["near_boundary"]["interval_signs"] = [1 if low > 0 else -1 if high < 0 else 0
                                                    for low, high in intervals]
    assert np.min(np.abs(observed["scores"])) < 1e-7


def test_reversed_order_is_a_measured_arithmetic_comparison(evidence):
    X = np.random.RandomState(11600).normal(size=(16, 3))
    y = np.where(np.arange(16) % 2, 1., -1.)
    order = np.arange(15, -1, -1)
    first, K = prepare(evidence, X)
    second, reversed_K = prepare(evidence, X[order])
    restored_K = reversed_K[np.ix_(order, order)]
    evidence["permutation"] = dict(order=order,
        kernel_max_difference=float(np.max(np.abs(K-restored_K))),
        kernel_exact_equal=bool(np.array_equal(K, restored_K)),
        interpretation="Order changes can select another rounded triangle or BLAS evaluation; exact parity is not assumed.")
    left = fit_record(evidence, first, X, y)
    right = fit_record(evidence, second, X[order], y[order])
    if left is None or right is None:
        return
    fitted_checks(evidence, left, K, X, y)
    fitted_checks(evidence, right, reversed_K, X[order], y[order])
    query = np.vstack((X, np.zeros((1, 3)), np.random.RandomState(18182).normal(size=(5, 3))))
    a, b = query_observation(left, query), query_observation(right, query)
    evidence["permutation"].update(first_query=a, reversed_query=b,
        score_max_difference=float(np.max(np.abs(a["scores"]-b["scores"]))),
        sign_differences=int(np.count_nonzero((a["scores"] > 0) != (b["scores"] > 0))),
        coefficient_max_difference=float(np.max(np.abs(left.dual_coef_.ravel()-right.dual_coef_.ravel()[order]))))
    assert a["represented_action_valid"] and b["represented_action_valid"]
    assert a["public_fixed_compatibility_valid"] and b["public_fixed_compatibility_valid"]
