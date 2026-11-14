"""Nonlinear-loss reference checks for dense linear-feature final reporting.

These cases make three public fits and three independent primal reference solves.
The deliberately signaled, noisy data must actually exercise material loss-tail
margins and interior dual slopes; nominal q values alone are not coverage.
"""
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize import minimize


HERE = Path(__file__).resolve().parent
HELPER_PATH = HERE / "test_linear_feature_public_full.py"
HELPER_NAME = "_ir13_frozen_nonlinear_oracle_helpers"
spec = importlib.util.spec_from_file_location(HELPER_NAME, HELPER_PATH)
H = importlib.util.module_from_spec(spec)
sys.modules[HELPER_NAME] = H
spec.loader.exec_module(H)

CASES = (
    ("q_half", 64, 7, .5, .002, 911),
    ("q_one", 72, 9, 1., .003, 912),
    ("q_two_point_three", 80, 11, 2.3, .004, 913),
)
CANDIDATE_ITERATIONS = 1000
REFERENCE_ITERATIONS = 400


def _receipt_json(value):
    """Encode exact rationals without decimal integer-string conversions.

    All Fractions use one explicit, lossless schema so consumers need no
    size-dependent interpretation. Hex conversion is independent of CPython's
    decimal-digit safety limit; that interpreter setting is never changed.
    """
    if isinstance(value, Fraction):
        return dict(encoding="fraction-hex-v1", numerator_hex=hex(value.numerator),
                    denominator_hex=hex(value.denominator))
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, np.ndarray):
        return _receipt_json(value.tolist())
    if isinstance(value, np.generic):
        return _receipt_json(value.item())
    if isinstance(value, dict):
        return {str(key): _receipt_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_receipt_json(item) for item in value]
    return value


@pytest.fixture
def evidence(request):
    state = np.geterr().copy(), np.geterrcall(), sys.getprofile(), sys.gettrace()
    record = dict(schema="ir13-nonlinear-reference-v2", nodeid=request.node.nodeid,
                  exact_fraction_encoding="fraction-hex-v1",
                  public_fit_calls=[], primal_reference_calls=0,
                  original_baseline_refits=0,
                  frozen_helper_sha256=hashlib.sha256(HELPER_PATH.read_bytes()).hexdigest(),
                  candidate_iteration_budget=CANDIDATE_ITERATIONS,
                  reference_iteration_budget=REFERENCE_ITERATIONS)
    try:
        yield record
    finally:
        record["numerical_state_preserved"] = state == (
            np.geterr().copy(), np.geterrcall(), sys.getprofile(), sys.gettrace())
        destination = os.environ.get("IR13_FULL_FEATURE_RESULTS")
        if destination:
            root = Path(destination).resolve()
            root.mkdir(parents=True, exist_ok=True)
            name = hashlib.sha256(request.node.nodeid.encode()).hexdigest()+".json"
            payload = json.dumps(_receipt_json(record), indent=2, sort_keys=True, allow_nan=False)
            with (root/name).open("x", encoding="utf-8") as stream:
                stream.write(payload+"\n")
        assert record["numerical_state_preserved"]


def _data(n, p, seed):
    rng = np.random.RandomState(seed)
    X = np.ascontiguousarray(rng.normal(size=(n, p))/math.sqrt(p))
    signal = X[:, 0]+.65*X[:, 1]-.4*X[:, 2]+.1*rng.normal(size=n)
    order = np.argsort(signal, kind="mergesort")
    signed = np.ones(n)
    signed[order[:n//2]] = -1.
    flips_per_class = n//16
    negative_flips = rng.choice(order[:n//2], size=flips_per_class, replace=False)
    positive_flips = rng.choice(order[n//2:], size=flips_per_class, replace=False)
    flips = np.r_[negative_flips, positive_flips]
    signed[flips] *= -1.
    return X, signed, flips


def _primal_decimal(X, signed, beta, offset, lambd, q):
    """Evaluate represented primal coordinates independently at 110 digits.

    This reference takes beta directly, avoiding a pseudoinverse or kernel
    coefficient reconstruction of the separate primal optimizer's state.
    General noninteger powers are high-precision estimates, not intervals.
    """
    with localcontext() as context:
        context.prec = 110
        cast = lambda value: Decimal.from_float(float(value))
        matrix = [[cast(value) for value in row] for row in X]
        labels = list(map(cast, signed))
        vector, bias = list(map(cast, beta)), cast(offset)
        penalty, qq = cast(lambd), cast(q)
        threshold = qq/(qq+1)
        margins = [label*(sum(x*b for x, b in zip(row, vector))+bias)
                   for label, row in zip(labels, matrix)]
        losses, slopes = [], []
        for margin in margins:
            if margin <= threshold:
                losses.append(1-margin)
                slopes.append(-Decimal(1))
            else:
                losses.append((threshold/margin)**qq/(qq+1))
                slopes.append(-(threshold/margin)**(qq+1))
        z = [label*slope/len(X) for label, slope in zip(labels, slopes)]
        gradient = [sum(row[j]*value for row, value in zip(matrix, z))
                    +2*penalty*vector[j] for j in range(X.shape[1])]
        intercept_gradient = sum(z)
        squared = sum(value*value for value in gradient)
        objective = sum(losses)/len(X)+penalty*sum(value*value for value in vector)
        residual = max(abs(intercept_gradient), squared.sqrt())
        return dict(arithmetic="Decimal110 estimate of represented beta and offset",
                    objective=objective, margins=margins, slopes=slopes,
                    gradient=gradient, intercept_gradient=intercept_gradient,
                    stationarity=residual)


def _exposed_integer_grid_witness(X, signed, state, q):
    """Expose the independent frozen helper's exact feasible integer witness.

    This projection is separate from the package's 2^-52 projection. The
    frozen independent lower-bound helper uses this same 2^-40 witness rule;
    exposing it permits actual interior-slope coverage checks and later audits.
    """
    margins = signed*(X@state[:-1]+state[-1])
    threshold = q/(q+1.)
    quantum = 2**40
    units = [int(math.floor((1. if m <= threshold else (threshold/m)**(q+1.))*quantum))
             for m in margins]
    labels = [int(label) for label in signed]
    excess = sum(label*value for label, value in zip(labels, units))
    selected = 1 if excess > 0 else -1
    remaining = abs(excess)
    for index in sorted(range(len(units)), key=lambda i: units[i], reverse=True):
        if labels[index] == selected:
            amount = min(remaining, units[index])
            units[index] -= amount
            remaining -= amount
    rho = [Fraction(value, quantum) for value in units]
    delta = Fraction(1, 2**20)
    material = [delta <= value <= 1-delta for value in rho]
    return dict(quantum=quantum, slopes=rho,
                equality=sum(label*value for label, value in zip(labels, rho)),
                unresolved_mass=remaining, box_feasible=all(0 <= value <= 1 for value in rho),
                exact_class_mass=sum(value for label, value in zip(labels, rho) if label > 0),
                material_interior_delta=delta, material_interior_count=sum(material),
                interior_positive_count=sum(flag and label > 0 for flag, label in zip(material, labels)),
                interior_negative_count=sum(flag and label < 0 for flag, label in zip(material, labels)))


def _tail_coverage(equations, signed, q):
    # A 5% separation excludes mere branch-boundary or last-bit changes.
    threshold = Decimal.from_float(q)/(Decimal.from_float(q)+1)
    cutoff = threshold*Decimal("1.05")
    material = [value >= cutoff for value in equations["margins"]]
    return dict(loss_threshold=str(threshold), material_tail_cutoff=str(cutoff),
                material_tail_count=sum(material),
                tail_positive_count=sum(flag and label > 0 for flag, label in zip(material, signed)),
                tail_negative_count=sum(flag and label < 0 for flag, label in zip(material, signed)))


@pytest.mark.parametrize("name,n,p,q,lambd,seed", CASES, ids=[row[0] for row in CASES])
def test_nonlinear_feature_model_against_independent_reference(name, n, p, q, lambd, seed, evidence):
    X, signed, flips = _data(n, p, seed)
    labels = (signed+1)/2
    evidence.update(case=name, seed=seed, X=X.copy(), signed_labels=signed.copy(),
                    flipped_label_indices=flips.copy(), q=q, lambd=lambd,
                    preprocessing="Gaussian entries / sqrt(p); balanced ranked signal labels; class-preserving flips")
    model = H._fit(evidence, H._model(q=q, lambd=lambd,
                   max_iter=CANDIDATE_ITERATIONS, stopping="optimality", tol=1e-6), X, labels)
    observed = evidence["public_fit_calls"][-1]["feature_observations"]
    beta = observed["coefficients"][-1]["beta"]
    candidate_equations = _primal_decimal(X, signed, beta, model.intercept_[0], lambd, q)
    evidence["candidate_primal_coordinates"] = dict(beta=beta.copy(), offset=float(model.intercept_[0]),
                                                   equations=candidate_equations)
    evaluate = lambda state: H._primal_value_grad_hessian(state, X, signed, lambd, q)
    evidence["primal_reference_calls"] += 1
    reference = minimize(lambda state: evaluate(state)[0], np.zeros(p+1),
        jac=lambda state: evaluate(state)[1], hess=lambda state: evaluate(state)[2],
        method="trust-exact", options={"gtol": 1e-10, "maxiter": REFERENCE_ITERATIONS})
    value, gradient, _ = evaluate(reference.x)
    reference_equations = _primal_decimal(
        X, signed, reference.x[:-1], reference.x[-1], lambd, q)
    evidence["independent_primal_reference"] = dict(
        method="trust-exact", success=bool(reference.success), status=int(reference.status),
        message=str(reference.message), state=reference.x.copy(), iterations=int(reference.nit),
        nfev=int(reference.nfev), njev=int(reference.njev), nhev=int(reference.nhev),
        objective=value, gradient=gradient.copy(), equations=reference_equations)
    witness = _exposed_integer_grid_witness(X, signed, reference.x, q)
    evidence["reference_dual_witness"] = witness
    independent_dual = H._independent_feasible_dual(X, signed, reference.x, lambd, q)
    candidate_coverage = _tail_coverage(candidate_equations, signed, q)
    reference_coverage = _tail_coverage(reference_equations, signed, q)
    # Keep subtraction at high precision; a binary64 display gap of zero is
    # not a statement of exact zero primal-dual gap or exact optimality.
    with localcontext() as context:
        context.prec = 110
        reference_gap = reference_equations["objective"]-independent_dual["lower"]
        candidate_gap = candidate_equations["objective"]-independent_dual["lower"]
    evidence.update(independent_dual=independent_dual,
                    candidate_tail_coverage=candidate_coverage,
                    reference_tail_coverage=reference_coverage,
                    reference_gap_decimal=reference_gap, candidate_gap_decimal=candidate_gap,
                    candidate_reference_objective_difference=(
                        model.final_objective_-float(reference_equations["objective"])))
    # Frozen helper first saves the full alpha/exact-gradient/action evidence,
    # then checks its unchanged scientific and runtime acceptance obligations.
    H._check_report(evidence, model, X, labels, rational=q == 1.)
    assert model.converged_ and model.rkhs_gradient_norm_ <= model.tol == 1e-6
    assert float(reference_equations["stationarity"]) <= 2e-7
    assert max(np.linalg.norm(gradient[:-1]), abs(gradient[-1])) <= 2e-7
    np.testing.assert_allclose(value, float(reference_equations["objective"]), rtol=3e-13, atol=3e-14)
    assert abs(model.final_objective_-float(reference_equations["objective"])) <= (
        2e-8*max(1., abs(float(reference_equations["objective"]))))
    endpoint = model.diagnostics_["dual_certificate"]["dual_objective_lower_bound"]
    assert endpoint <= float(reference_equations["objective"])+2e-13
    assert witness["box_feasible"] and witness["equality"] == 0 and witness["unresolved_mass"] == 0
    assert witness["exact_class_mass"] == independent_dual["exact_class_mass"]
    assert witness["material_interior_count"] >= n//8
    assert witness["interior_positive_count"] >= 2 and witness["interior_negative_count"] >= 2
    for coverage in (candidate_coverage, reference_coverage):
        assert coverage["material_tail_count"] >= n//8
        assert coverage["tail_positive_count"] >= 2 and coverage["tail_negative_count"] >= 2
    assert Decimal("-2e-13") <= reference_gap <= Decimal("2e-7")
    assert Decimal("-2e-13") <= candidate_gap <= Decimal("2e-7")
    assert evidence["primal_reference_calls"] == 1 and len(evidence["public_fit_calls"]) == 1
    assert evidence["original_baseline_refits"] == 0
