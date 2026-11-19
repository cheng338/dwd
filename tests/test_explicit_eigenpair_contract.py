"""P0_eig is unused by explicit solves and validated by implicit solves.

Small deterministic fixed-step controls compare identical routes and reuse the
same supplied eigenpair. No bitwise comparison of independently decomposed
eigensystems is required, and convergence is not inferred from three steps.
"""
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError

import dwd.gen_dwd as gdwd
import dwd._eigen as eigen


def data(case, layout):
    rows = np.arange(20, dtype=float)
    X = np.column_stack(((rows - 9.5) / 8,
                         ((3 * rows) % 7 - 3) / 4,
                         ((5 * rows) % 11 - 5) / 8))
    if case == "rank-deficient-positive":
        X[:, 2] = 2 * X[:, 0]
    penalty = 0.0 if case == "full-rank-zero" else 0.25
    y = np.tile([-1.0, 1.0], 10)
    return (sparse.csr_matrix(X) if layout == "csr" else X), y, penalty


def buffers(X):
    return (X.data, X.indices, X.indptr) if sparse.issparse(X) else (X,)


def snapshot_arrays(*arrays):
    return [(array, array.dtype.str, array.shape, array.strides,
             array.flags.writeable, array.tobytes(order="A")) for array in arrays]


def assert_unchanged(snapshot):
    for array, dtype, shape, strides, writable, raw in snapshot:
        assert array.dtype.str == dtype and array.shape == shape and array.strides == strides
        assert array.flags.writeable == writable
        assert array.tobytes(order="A") == raw


def readonly_pair(pair):
    result = tuple(np.array(array, copy=True, order="K") for array in pair)
    for array in result:
        array.flags.writeable = False
    return result


def options(implicit, penalty):
    return dict(lambd=penalty, q=1.0, implicit_P=implicit, max_iter=3,
                initialization="zero", stopping="fixed", random_state=19)


def model_state(model, X):
    return dict(coef=model.coef_.copy(), intercept=model.intercept_.copy(),
                classes=model.classes_.copy(), objectives=model.objective_history_.copy(),
                C=model.C_, n_iter=model.n_iter_, final_objective=model.final_objective_,
                gradient=model.gradient_norm_, converged=model.converged_,
                termination=model.termination_reason_,
                decisions=model.decision_function(X), predictions=model.predict(X))


def run(entry, X, y, penalty, implicit, pair):
    if entry == "public":
        model = gdwd.GenDWD(**options(implicit, penalty)).fit(X, y, P0_eig=pair)
        assert model.n_iter_ == 3 and model.termination_reason_ == "max_iter"
        assert np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all()
        return model_state(model, X)
    beta, intercept, objectives, conversion = gdwd.solve_gen_dwd(
        X, y, P0_eig=pair, **options(implicit, penalty))
    assert len(objectives) == 4
    assert np.isfinite(beta).all() and np.isfinite(intercept)
    assert np.isfinite(objectives).all()
    return dict(coef=beta, intercept=intercept, objectives=objectives, C=conversion)


def assert_exact_state(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        a, b = np.asarray(actual[name]), np.asarray(expected[name])
        assert a.dtype == b.dtype and a.shape == b.shape, name
        assert a.tobytes() == b.tobytes(), name


def forbid_eigen_work(*args, **kwargs):
    raise AssertionError("This solve must not compute or inspect an unused eigensystem")


class UninspectablePair:
    def __init__(self):
        self.touches = []

    def touched(self, operation):
        self.touches.append(operation)
        raise AssertionError("Unused P0_eig was inspected: " + operation)

    def __array__(self, dtype=None, copy=None):
        return self.touched("array")

    def __iter__(self):
        return self.touched("iter")

    def __len__(self):
        return self.touched("len")

    def __bool__(self):
        return self.touched("bool")


@pytest.mark.parametrize("entry", ["public", "functional"])
@pytest.mark.parametrize("layout", ["dense", "csr"])
@pytest.mark.parametrize("case", ["ordinary-positive", "rank-deficient-positive", "full-rank-zero"])
def test_explicit_outputs_ignore_every_supplied_pair_without_inspection(entry, layout, case, monkeypatch):
    X, y, penalty = data(case, layout)
    valid = readonly_pair(gdwd.get_P0_eig(X))
    malformed = (np.array([[np.nan]]), np.array([complex(1, 2)]))
    opaque = UninspectablePair()
    before = snapshot_arrays(*buffers(X), y, *valid, *malformed)
    with monkeypatch.context() as patch:
        patch.setattr(gdwd, "get_P0_eig", forbid_eigen_work)
        patch.setattr(gdwd, "validated_eigh", forbid_eigen_work)
        patch.setattr(eigen, "eigh", forbid_eigen_work)
        expected = run(entry, X, y, penalty, False, None)
        for supplied in (valid, malformed, opaque):
            actual = run(entry, X, y, penalty, False, supplied)
            assert_exact_state(actual, expected)
    assert opaque.touches == []
    assert_unchanged(before)


@pytest.mark.parametrize("entry", ["public", "functional"])
@pytest.mark.parametrize("layout", ["dense", "csr"])
@pytest.mark.parametrize("case", ["ordinary-positive", "rank-deficient-positive", "full-rank-zero"])
def test_implicit_valid_supplied_pair_is_validated_without_recomputing(entry, layout, case, monkeypatch):
    X, y, penalty = data(case, layout)
    pair = readonly_pair(gdwd.get_P0_eig(X))
    before = snapshot_arrays(*buffers(X), y, *pair)
    # The control also consumes this exact pair; no separate eigensystem is
    # compared bitwise on either numerical runtime.
    expected = run(entry, X, y, penalty, True, pair)
    original_validate = gdwd.validated_eigh
    validations = []

    def validate(matrix, **kwargs):
        assert kwargs.get("supplied") is pair
        validations.append(matrix.shape)
        return original_validate(matrix, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(gdwd, "get_P0_eig", forbid_eigen_work)
        patch.setattr(eigen, "eigh", forbid_eigen_work)
        patch.setattr(gdwd, "validated_eigh", validate)
        actual = run(entry, X, y, penalty, True, pair)
    assert validations == [(4, 4)]
    assert_exact_state(actual, expected)
    assert_unchanged(before)


def invalid_pair(kind, pair, X):
    vectors, values = pair
    if kind == "malformed":
        return (vectors.copy(),)
    if kind == "nonfinite":
        bad_values = values.copy()
        bad_values[0] = np.nan
        return vectors.copy(), bad_values
    if kind == "complex":
        bad_vectors = vectors.astype(complex)
        bad_vectors[0, 0] += 0.5j
        return bad_vectors, values.copy()
    assert kind == "stale"
    other = X.copy()
    if sparse.issparse(other):
        other.data[0] += 0.25
    else:
        other[0, 0] += 0.25
    return gdwd.get_P0_eig(other)


@pytest.mark.parametrize("entry", ["public", "functional"])
@pytest.mark.parametrize("layout", ["dense", "csr"])
@pytest.mark.parametrize("kind", ["malformed", "nonfinite", "complex", "stale"])
def test_implicit_invalid_pair_rejects_and_failed_public_refit_clears_state(entry, layout, kind, monkeypatch):
    X, y, penalty = data("ordinary-positive", layout)
    if entry == "public":
        model = gdwd.GenDWD(**options(True, penalty)).cv_init(X)
        pair = model._P0_eig
        for array in pair:
            array.flags.writeable = False
        model.fit(X, y)
        assert hasattr(model, "coef_")
        private_X = model._P0_X
        private_geometry = model._P0_geometry
        private_before = snapshot_arrays(*buffers(private_X))
    else:
        pair = readonly_pair(gdwd.get_P0_eig(X))
    bad = invalid_pair(kind, pair, X)
    before = snapshot_arrays(*buffers(X), y, *pair, *bad)
    with monkeypatch.context() as patch:
        patch.setattr(gdwd, "get_P0_eig", forbid_eigen_work)
        patch.setattr(eigen, "eigh", forbid_eigen_work)
        with pytest.raises(ValueError, match="Supplied eigenpairs"):
            if entry == "public":
                model.fit(X, y, P0_eig=bad)
            else:
                gdwd.solve_gen_dwd(X, y, P0_eig=bad, **options(True, penalty))
    assert_unchanged(before)
    if entry == "public":
        assert not [name for name in vars(model) if name.endswith("_") and not name.startswith("_")]
        with pytest.raises(NotFittedError):
            model.predict(X)
        assert model._P0_eig is pair and model._P0_X is private_X
        assert model._P0_geometry == private_geometry
        assert model.implicit_P is True and model.lambd == penalty
        assert_unchanged(private_before)
