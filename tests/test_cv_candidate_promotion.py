"""CV must preserve literal sparse data for supported DWD candidates.

The oracle below never calls production promotion helpers. Literal COO row
subsets retain duplicate entries until each direct candidate's fit boundary.
New DWD fits use float64 validation. Candidate order, fit/score/refit
boundaries, pairwise slicing, and unrelated-estimator behavior remain covered.
Retired solver candidates must fail before any fold preparation or computation;
they must not masquerade as a second supported precision policy.
"""
from copy import deepcopy
from fractions import Fraction

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin

from dwd.cv import DoL2LoD, run_cv
from dwd.gen_dwd import GenDWD
from dwd.gen_kern_dwd import KernGDWD


Y = np.tile([-1, 1], 4)
FOLDS = [(np.array([0, 1, 4, 5]), np.array([2, 3, 6, 7])),
         (np.array([2, 3, 6, 7]), np.array([0, 1, 4, 5]))]
PENALTIES = ((.1,), (.2,), (.1, .2), (.2, .1))
LAYOUTS = ("coo_matrix", "coo_array", "csr_matrix", "csr_array",
           "csc_matrix", "csc_array")


def features(layout="coo_matrix", dtype=np.float32, *, readonly=True):
    prefix = [100, 100] if np.dtype(dtype).kind in "iu" else [2**24, 1, -2**24]
    # Column 2 identifies the row without using labels to construct the oracle.
    rows = np.array([0] * len(prefix) + list(range(1, 8)) + list(range(8)),
                    dtype=np.int32)
    cols = np.array([0] * len(prefix) + [0] * 7 + [2] * 8, dtype=np.int32)
    values = np.array(prefix + [-2, -1, -.5, .5, 1, 2, 3]
                      + list(range(1, 9)), dtype=dtype)
    cls = getattr(sparse, layout)
    if layout.startswith("coo"):
        result = cls((values, (rows, cols)), shape=(8, 3))
    else:
        major, minor, count = ((rows, cols, 8) if layout.startswith("csr")
                                else (cols, rows, 3))
        order = np.argsort(major, kind="stable")
        indptr = np.asarray(np.r_[0, np.cumsum(np.bincount(major, minlength=count))],
                            dtype=np.int32)
        result = cls((values[order], minor[order], indptr), shape=(8, 3))
    if readonly:
        for buffer in buffers(result):
            buffer.flags.writeable = False
    return result


def buffers(value):
    return ((value.data, value.row, value.col) if value.format == "coo"
            else (value.data, value.indices, value.indptr))


def snapshot(value):
    flags = {name: deepcopy(item) for name, item in value.__dict__.items()
             if "canonical" in name or "sorted" in name}
    return [(buffer.copy(), buffer.flags.writeable) for buffer in buffers(value)], flags


def unchanged(value, before):
    for buffer, (expected, writeable) in zip(buffers(value), before[0]):
        assert_array_equal(buffer, expected)
        assert buffer.flags.writeable == writeable
    assert snapshot(value)[1] == before[1]


def dense(value):
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def working(raw):
    """Independent supported direct-fit policy: promote before coalescing."""
    return (raw if raw.dtype == np.float64 else raw.astype(np.float64)).tocsr()


def rational_dense(raw):
    """Exact represented coordinates, independent of sparse accumulation."""
    # CSR.tocoo reads (and lazily caches) has_canonical_format on its source,
    # even with copy=True. Keep oracle bookkeeping off the caller's object.
    coo = raw.copy().tocoo(copy=False)
    entries = [[Fraction() for _ in range(raw.shape[1])] for _ in range(raw.shape[0])]
    for row, col, value in zip(coo.row, coo.col, coo.data):
        entries[row][col] += Fraction(float(value))
    return np.asarray([[float(value) for value in row] for row in entries])


def literal_rows(raw, rows):
    """Subset literal COO entries without summing the original duplicates."""
    assert raw.format == "coo"
    positions = np.full(raw.shape[0], -1, dtype=np.int32)
    positions[rows] = np.arange(len(rows), dtype=np.int32)
    keep = positions[raw.row] >= 0
    return sparse.coo_matrix((raw.data[keep].copy(),
                              (positions[raw.row[keep]], raw.col[keep].copy())),
                             shape=(len(rows), raw.shape[1]))


class RecordingBoundary:
    events = []

    def _event(self, kind, value):
        type(self).events.append((kind, np.dtype(value.dtype).name, self.lambd,
                                  getattr(self, "kernel", None), value.copy()))

    def cv_init(self, X):
        self._event("init", X)
        self._cached_dtype = X.dtype
        self._cached_kernel = getattr(self, "kernel", None)
        self._cached_X = X.copy()
        return self

    def _cv_cache_matches(self, X):
        return (hasattr(self, "_cached_X")
                and getattr(self, "_cached_dtype", None) == X.dtype
                and getattr(self, "_cached_kernel", None) == getattr(self, "kernel", None)
                and X.dtype == self._cached_X.dtype
                and np.array_equal(dense(X), dense(self._cached_X)))

    def fit(self, X, y):
        self._event("fit", X)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.fitted_input_ = X.copy()
        return self

    def predict(self, X):
        return np.repeat(self.classes_[0], X.shape[0])


class RecordingLinear(RecordingBoundary, GenDWD):
    pass


class RecordingKernel(RecordingBoundary, KernGDWD):
    pass


def boundary_scorer(preferred):
    def score(model, X, y):
        model._event("score", X)
        return float(model.lambd == preferred)
    return score


@pytest.mark.parametrize("estimator", (RecordingLinear, RecordingKernel))
@pytest.mark.parametrize("penalties", PENALTIES)
@pytest.mark.parametrize("refit", (False, True))
def test_candidate_fit_score_and_refit_boundaries(estimator, penalties, refit):
    raw = features()
    before = snapshot(raw)
    model = estimator()
    parameters = model.get_params(deep=True)
    estimator.events = []
    params = {"lambd": list(penalties)}
    candidates = DoL2LoD(params)
    preferred = penalties[-1]
    result = run_cv(model, raw, Y, params, cv=FOLDS,
                    scoring=boundary_scorer(preferred), refit_best=refit)
    assert result[0] == {"lambd": preferred}
    assert result[1] == 1.
    events = [event for event in estimator.events if event[0] != "init"]
    expected_events = []
    for train, test in FOLDS:
        for candidate in candidates:
            source = working(raw)
            for kind, selected in (("fit", train), ("score", train), ("score", test)):
                expected_events.append((kind, np.dtype(source.dtype).name, candidate["lambd"],
                                        source[selected, :]))
    if refit:
        expected_events.append(("fit", "float64", preferred, working(raw)))
        assert "solver_mode" not in result[2].get_params()
        assert result[2].lambd == preferred
        assert_array_equal(dense(result[2].fitted_input_), dense(working(raw)))
    else:
        assert result[2] is None
    assert len(events) == len(expected_events)
    for actual, expected in zip(events, expected_events):
        assert actual[:3] == expected[:3]
        assert actual[4].dtype == expected[3].dtype
        assert_array_equal(dense(actual[4]), dense(expected[3]))
    assert_array_equal(working(raw).toarray(), rational_dense(raw))
    assert raw.copy().tocsr()[0, 0] != working(raw)[0, 0]
    assert model.get_params(deep=True) == parameters
    unchanged(raw, before)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", (np.float32, np.int8))
def test_sparse_layouts_keep_literal_promotion_and_caller_storage(layout, dtype):
    raw = features(layout, dtype)
    before = snapshot(raw)
    RecordingLinear.events = []
    result = run_cv(RecordingLinear(), raw, Y,
                    {"lambd": [.1, .2]}, cv=FOLDS,
                    scoring=boundary_scorer(.2))
    unchanged(raw, before)
    assert_array_equal(dense(result[2].fitted_input_), rational_dense(raw))
    assert result[2].fitted_input_.dtype == np.float64
    for event in RecordingLinear.events:
        if event[0] == "fit" and event[4].shape[0] == len(Y):
            assert event[1] == "float64"
    unchanged(raw, before)


def actual_model(estimator, penalty=.1):
    options = dict(lambd=penalty, max_iter=2, stopping="fixed", initialization="zero",
                   random_state=7)
    if estimator is KernGDWD:
        options.update(kernel="linear", backend="spectral", affine_computation="standard",
                       acceleration=None)
    return estimator(**options)


@pytest.mark.parametrize("estimator", (GenDWD, KernGDWD))
@pytest.mark.parametrize("penalties", ((.1, .2), (.2, .1)))
def test_actual_fold_fits_match_independent_literal_candidate_fits(monkeypatch, estimator,
                                                                  penalties):
    raw = features()
    before = snapshot(raw)
    probe = rational_dense(raw)
    controls, test_scores, train_scores = [], [], []
    for train, test in FOLDS:
        fold_train, fold_test = literal_rows(raw, train), literal_rows(raw, test)
        fold_test_scores, fold_train_scores = [], []
        for penalty in penalties:
            control = actual_model(estimator, penalty).fit(fold_train, Y[train])
            controls.append(control.decision_function(probe))
            # Scoring uses the same candidate working representation as CV;
            # fitting above independently consumes uncoalesced literal rows.
            fold_train_scores.append(float(np.mean(control.predict(working(fold_train)) == Y[train])))
            fold_test_scores.append(float(np.mean(control.predict(working(fold_test)) == Y[test])))
        test_scores.append(fold_test_scores)
        train_scores.append(fold_train_scores)
    mean_scores = np.mean(test_scores, axis=0)
    winner = penalties[int(np.argmax(mean_scores))]
    full_control = actual_model(estimator, winner).fit(raw, Y)
    controls.append(full_control.decision_function(probe))
    observed = []
    original_fit = estimator.fit

    def recorded_fit(model, X, y, *args, **kwargs):
        result = original_fit(model, X, y, *args, **kwargs)
        observed.append(model.decision_function(probe))
        return result

    monkeypatch.setattr(estimator, "fit", recorded_fit)
    result = run_cv(actual_model(estimator), raw, Y,
                    {"lambd": list(penalties)}, cv=FOLDS)
    assert len(observed) == len(controls)
    for actual, expected in zip(observed, controls):
        # Identical candidate data, but direct and cached eigensystems can
        # differ in final rounding. This is not a convergence experiment.
        assert_allclose(actual, expected, rtol=2e-6, atol=2e-8)
    assert result[0] == {"lambd": winner}
    for i, fold in enumerate(result[4]):
        assert_array_equal(fold["test_score"], test_scores[i])
        assert_array_equal(fold["train_score"], train_scores[i])
    unchanged(raw, before)


@pytest.mark.parametrize("estimator", (GenDWD, KernGDWD))
def test_lambda_path_still_prepares_once_per_fold(monkeypatch, estimator):
    raw = features()
    calls = []
    original_init = estimator.cv_init

    def recorded_init(model, X):
        calls.append((X.dtype, X.copy()))
        return original_init(model, X)

    monkeypatch.setattr(estimator, "cv_init", recorded_init)
    run_cv(actual_model(estimator), raw, Y,
           {"lambd": [.1, .2]}, cv=FOLDS,
           refit_best=False)
    assert len(calls) == len(FOLDS)
    for (_, value), (train, _) in zip(calls, FOLDS):
        assert value.dtype == np.float64
        assert_array_equal(dense(value), rational_dense(raw)[train])


@pytest.mark.parametrize("penalty", (.1, .2))
def test_callable_kernel_receives_selected_candidate_values_and_dtype(penalty):
    raw = features()
    expected = dense(working(raw))
    calls = []

    def kernel(left, right):
        left, right = dense(left), dense(right)
        calls.append((left.copy(), right.copy()))
        return left @ right.T

    model = actual_model(KernGDWD).set_params(kernel=kernel)
    run_cv(model, raw, Y, {"lambd": [penalty]}, cv=FOLDS)
    assert calls
    for pair in calls:
        for value in pair:
            assert value.dtype == expected.dtype
            rows = value[:, 2].astype(int) - 1
            assert_array_equal(value, expected[rows])


@pytest.mark.parametrize("constructor_kernel", ("linear", "precomputed"))
def test_pairwise_candidate_views_are_independent(constructor_kernel):
    rows = np.repeat(np.arange(8, dtype=np.int32), 3)
    raw = sparse.coo_matrix((np.tile(np.array([2**24, 1, -2**24], np.float32), 8),
                             (rows, rows)), shape=(8, 8))
    before = snapshot(raw)
    RecordingKernel.events = []
    params = {"kernel": ["linear", "precomputed"]}
    candidates = DoL2LoD(params)
    run_cv(RecordingKernel(kernel=constructor_kernel),
           raw, Y, params, scoring=boundary_scorer(.1), cv=FOLDS,
           refit_best=False)
    events = [event for event in RecordingKernel.events if event[0] != "init"]
    expected = []
    for train, test in FOLDS:
        for candidate in candidates:
            source = working(raw)
            for kind, selected in (("fit", train), ("score", train), ("score", test)):
                view = source[selected, :]
                if candidate["kernel"] == "precomputed":
                    view = view[:, train]
                expected.append((kind, np.dtype(source.dtype).name, candidate["kernel"], view))
    assert len(events) == len(expected)
    for actual, control in zip(events, expected):
        assert (actual[0], actual[1], actual[3]) == control[:3]
        assert actual[4].dtype == control[3].dtype
        assert_array_equal(dense(actual[4]), dense(control[3]))
    unchanged(raw, before)


@pytest.mark.parametrize("estimator", (RecordingLinear, RecordingKernel))
@pytest.mark.parametrize("bad", (np.nan, np.inf, 1+0j))
def test_invalid_sparse_data_rejects_before_candidate_fit(estimator, bad):
    raw = features(readonly=False)
    if isinstance(bad, complex):
        raw = raw.astype(np.complex128)
    raw.data[0] = bad
    before = snapshot(raw)
    estimator.events = []
    with pytest.raises((TypeError, ValueError)):
        run_cv(estimator(), raw, Y,
               {"lambd": [.1, .2]}, cv=FOLDS)
    assert estimator.events == []
    unchanged(raw, before)


@pytest.mark.parametrize("estimator", (RecordingLinear, RecordingKernel))
def test_dense_inputs_keep_existing_outer_validation_policy(estimator):
    value = np.arange(24, dtype=np.float32).reshape(8, 3)
    before = value.copy()
    estimator.events = []
    run_cv(estimator(), value, Y,
           {"lambd": [.1, .2]}, cv=FOLDS,
           scoring=boundary_scorer(.2))
    assert estimator.events
    for event in estimator.events:
        assert event[4].dtype == np.float32
    assert_array_equal(value, before)


class GenericClassifier(ClassifierMixin, BaseEstimator):
    events = []

    def __init__(self, candidate=0):
        self.candidate = candidate

    def cv_init(self, X):
        return self

    def fit(self, X, y):
        type(self).events.append(X.copy())
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.repeat(self.classes_[0], X.shape[0])


@pytest.mark.parametrize("sparse_input", (False, True))
def test_unrelated_estimators_do_not_gain_solver_policy(sparse_input):
    value = features() if sparse_input else np.arange(24, dtype=np.float32).reshape(8, 3)
    GenericClassifier.events = []
    result = run_cv(GenericClassifier(), value, Y, {"candidate": [0, 1]}, cv=FOLDS)
    assert result[0] == {"candidate": 0}
    assert len(GenericClassifier.events) == 5
    expected = value.tocsr() if sparse_input else value
    assert_array_equal(dense(GenericClassifier.events[-1]), dense(expected))
    assert GenericClassifier.events[-1].dtype == np.float32


@pytest.mark.parametrize("penalties", PENALTIES)
def test_splitter_observes_supported_precision_independent_of_candidate_order(penalties):
    raw = features()
    seen = []

    class Splitter:
        def split(self, X, y=None, groups=None):
            seen.append(X.copy())
            yield from FOLDS

    RecordingLinear.events = []
    run_cv(RecordingLinear(), raw, Y,
           {"lambd": list(penalties)}, cv=Splitter(),
           scoring=boundary_scorer(penalties[-1]), refit_best=False)
    expected = working(raw)
    assert len(seen) == 1
    for value in seen:
        assert value.dtype == expected.dtype
        assert_array_equal(dense(value), dense(expected))


@pytest.mark.parametrize("estimator", (RecordingLinear, RecordingKernel))
@pytest.mark.parametrize("modes", (("legacy",), ("legacy", "schur"), ("schur", "legacy")))
@pytest.mark.parametrize("refit", (False, True))
def test_retired_candidate_preflight_precedes_all_sparse_fold_work(estimator, modes, refit):
    raw = features()
    before = snapshot(raw)
    estimator.events = []
    with pytest.raises(ValueError, match="Invalid parameter.*solver_mode"):
        run_cv(estimator(), raw, Y,
               {"solver_mode": list(modes)}, cv=FOLDS, refit_best=refit)
    assert estimator.events == []
    unchanged(raw, before)
