"""Independent IR-7 contract and real-fit checks; no production helper oracle.

IR-7 adds concrete-type identity to cache comparisons. These tests deliberately
retain same-type signed-zero, NaN, and opaque equality semantics. They do not
claim that arbitrary mutable callable state is a complete cache key.
"""

from collections import OrderedDict
from copy import deepcopy
from unittest.mock import patch
import inspect
import sys

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

import dwd._kernel_solver as kernel_solver
from dwd.cv import run_cv
from dwd.gen_kern_dwd import KernGDWD
from dwd.utils import parameters_equal


class FloatSubclass(float):
    pass


class DictSubclass(dict):
    pass


class ListSubclass(list):
    pass


class ArraySubclass(np.ndarray):
    pass


class OpaqueEqual:
    def __eq__(self, other):
        return True


class ArrayEquality:
    def __eq__(self, other):
        return np.array([True, True])


def _assert_symmetric(left, right, expected):
    assert parameters_equal(left, right) is expected
    assert parameters_equal(right, left) is expected


CROSS_TYPE_PAIRS = [
    pytest.param(np.float16(.1), .1, id="float16-rounded"),
    pytest.param(np.float32(.1), .1, id="float32-rounded"),
    pytest.param(np.float64(.1), .1, id="float64-python"),
    pytest.param(np.longdouble(".1"), .1, id="longdouble-python"),
    pytest.param(np.float16(.5), .5, id="float16-exact"),
    pytest.param(np.float32(.5), .5, id="float32-exact"),
    pytest.param(np.float64(.5), .5, id="float64-exact"),
    pytest.param(np.longdouble(.5), .5, id="longdouble-exact"),
    pytest.param(np.float16(.5), np.float32(.5), id="float16-float32"),
    pytest.param(np.float32(.5), np.float64(.5), id="float32-float64"),
    pytest.param(np.int32(7), 7, id="int32-python"),
    pytest.param(np.int64(2**53), 2**53 + 1, id="int64-large-unequal"),
    pytest.param(np.uint64(2**63), 2**63 + 1, id="uint64-large-unequal"),
    pytest.param(True, 1, id="bool-int"),
    pytest.param(np.bool_(True), True, id="numpy-bool-python"),
    pytest.param(1., 1, id="float-int"),
    pytest.param(FloatSubclass(.5), .5, id="float-subclass"),
]


@pytest.mark.parametrize("left,right", CROSS_TYPE_PAIRS)
@pytest.mark.parametrize("nesting", ("scalar", "nested", "object-array"))
def test_distinct_concrete_scalar_types_invalidate_in_every_nesting(left, right, nesting):
    assert type(left) is not type(right)
    if nesting == "nested":
        left, right = {"settings": [(left,)]}, {"settings": [(right,)]}
    elif nesting == "object-array":
        left, right = np.array([left], dtype=object), np.array([right], dtype=object)
    _assert_symmetric(left, right, False)


@pytest.mark.parametrize("dtype", (float, np.float16, np.float32, np.float64, np.longdouble))
def test_same_float_type_values_and_existing_zero_nan_semantics(dtype):
    _assert_symmetric(dtype(.5), dtype(.5), True)
    _assert_symmetric(dtype(.5), dtype(.75), False)
    # Preserved behavior, not a universal sign-sensitive callable guarantee.
    _assert_symmetric(dtype(0.), dtype(-0.), True)
    left_nan, right_nan = dtype("nan"), dtype("nan")
    _assert_symmetric(left_nan, right_nan, False)
    assert parameters_equal(left_nan, left_nan) is True


@pytest.mark.parametrize("dtype", (int, np.int32, np.int64, np.uint64, bool, np.bool_))
def test_same_integer_and_boolean_types_keep_value_comparison(dtype):
    _assert_symmetric(dtype(1), dtype(1), True)
    _assert_symmetric(dtype(0), dtype(1), False)


def test_longdouble_adjacent_values_are_not_normalized_through_python_float():
    left = np.longdouble(1)
    right = np.nextafter(left, np.longdouble(2))
    assert left != right
    _assert_symmetric(left, right, False)
    if np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant:
        assert float(left) == float(right)


@pytest.mark.parametrize("left,right", [
    pytest.param({"x": 1}, DictSubclass(x=1), id="dict-subclass"),
    pytest.param({"x": 1}, OrderedDict(x=1), id="mapping-kind"),
    pytest.param([1], (1,), id="list-tuple"),
    pytest.param([1], ListSubclass([1]), id="list-subclass"),
    pytest.param(np.array([1.]), [1.], id="array-list"),
    pytest.param(np.array([1.]), np.array([1.]).view(ArraySubclass), id="array-subclass"),
    pytest.param(np.array([1.], dtype=np.float32), np.array([1.]), id="array-dtype"),
    pytest.param(np.array([1.]), np.array([[1.]]), id="array-shape"),
])
def test_container_concrete_types_and_existing_array_metadata_invalidate(left, right):
    _assert_symmetric(left, right, False)


def test_same_container_types_reuse_and_detect_deep_changes():
    value = {"a": [np.array([1., 2.]), (np.float32(.5),)],
             "b": np.array([np.int64(7), {"c": np.float16(.5)}], dtype=object)}
    _assert_symmetric(value, deepcopy(value), True)
    changed = deepcopy(value)
    changed["b"][1]["c"] = .5
    _assert_symmetric(value, changed, False)
    _assert_symmetric(DictSubclass(x=np.float32(.5)),
                      DictSubclass(x=np.float32(.5)), True)
    _assert_symmetric(np.array([0., -0.]), np.array([-0., 0.]), True)
    _assert_symmetric(np.array([np.nan]), np.array([np.nan]), False)


def test_existing_identity_and_opaque_scalar_equality_contract_is_preserved():
    function = lambda: None
    assert parameters_equal(function, function) is True
    assert parameters_equal(function, lambda: None) is False
    _assert_symmetric(OpaqueEqual(), OpaqueEqual(), True)
    _assert_symmetric(ArrayEquality(), ArrayEquality(), False)
    _assert_symmetric(OpaqueEqual(), object(), False)


@pytest.fixture
def data():
    rng = np.random.default_rng(914)
    X = rng.normal(size=(12, 3))
    y = np.tile([-1, 1], 6)
    query = rng.normal(size=(5, 3))
    folds = [(np.arange(6), np.arange(6, 12)),
             (np.arange(6, 12), np.arange(6))]
    return X, y, query, folds


def _model(gamma=.1, **overrides):
    options = dict(kernel="rbf", kernel_kws={"gamma": gamma}, lambd=.09,
                   initialization="zero", stopping="fixed", max_iter=3,
                   acceleration=None, affine_computation="standard",
                   random_state=91, backend="spectral")
    options.update(overrides)
    return KernGDWD(**options)


def _fitted_snapshot(model, query):
    return {"dual": model.dual_coef_.copy(),
            "intercept": model.intercept_.copy(),
            "objective": model.objective_history_.copy(),
            "decision": model.decision_function(query).copy(),
            "iterations": model.n_iter_}


def _assert_snapshot_equal(actual, expected):
    # Existing cache tests use 1e-12; exact represented Grams are checked apart.
    for name in ("dual", "intercept", "objective", "decision"):
        assert_allclose(actual[name], expected[name], rtol=1e-12, atol=1e-12)
    assert actual["iterations"] == expected["iterations"]


BACKENDS = ("cholesky", "spectral")


@pytest.mark.parametrize("dtype", (np.float16, np.float32))
@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("backend", BACKENDS)
def test_changed_gamma_standalone_fit_uses_target_gram_and_matches_uncached_fit(
        dtype, reverse, backend, data):
    X, y, query, _ = data
    old, new = (dtype(.1), .1)
    if reverse:
        old, new = new, old
    options = dict(backend=backend)
    model = _model(old, **options).cv_init(X)
    old_gram = model._cv_K.copy()
    model.set_params(kernel_kws={"gamma": new})
    assert not model._cv_cache_matches(X)

    independently_prepared = _model(new, **options).cv_init(X)
    expected_gram = independently_prepared._cv_K.copy()
    assert not np.array_equal(old_gram, expected_gram)
    expected = _fitted_snapshot(_model(new, **options).fit(X, y), query)
    original_solve = kernel_solver.solve_kernel
    used_grams = []

    def observed_solve(K, *args, **kwargs):
        used_grams.append(np.array(K, copy=True))
        return original_solve(K, *args, **kwargs)

    with patch.object(kernel_solver, "solve_kernel", observed_solve):
        model.fit(X, y)
    assert len(used_grams) == 1
    assert_array_equal(used_grams[0], expected_gram)
    _assert_snapshot_equal(_fitted_snapshot(model, query), expected)
    # fit need not update its old private CV cache after rejecting that cache.


def _continuous_score(model, X, y):
    return -float(np.mean((model.decision_function(X) - y) ** 2))


@pytest.mark.parametrize("dtype,value", [(np.float16, .1), (np.float32, .1),
                                         (np.float64, .5)])
@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("backend", BACKENDS)
def test_real_cv_each_candidate_matches_independent_preparation_in_both_orders(
        dtype, value, reverse, backend, data):
    X, y, query, folds = data
    gammas = [dtype(value), value]
    if reverse:
        gammas.reverse()
    lambdas = [.06, .2]
    settings = [(gamma, lambd) for gamma in gammas for lambd in lambdas]
    expected_fits, expected_test, expected_train = [], [], []
    for train, test in folds:
        fold_test, fold_train = [], []
        for gamma, lambd in settings:
            control = _model(gamma, lambd=lambd, backend=backend)
            control.cv_init(X[train]).fit(X[train], y[train])
            expected_fits.append((control._cv_K.copy(), _fitted_snapshot(control, query)))
            fold_test.append(_continuous_score(control, X[test], y[test]))
            fold_train.append(_continuous_score(control, X[train], y[train]))
        expected_test.append(fold_test)
        expected_train.append(fold_train)

    # Observe the real methods without replacing the provenance-qualified hooks.
    fit_code = inspect.unwrap(KernGDWD.fit).__code__
    init_code = inspect.unwrap(KernGDWD.cv_init).__code__
    observed_fits, preparations, observed_routes = [], [], []
    previous_profile = sys.getprofile()

    def observe(frame, event, result):
        if frame.f_code is init_code and event == "call":
            model = frame.f_locals["self"]
            preparations.append((type(model.kernel_kws["gamma"]), model.lambd))
        elif frame.f_code is fit_code and event == "return":
            model = frame.f_locals["self"]
            if result is model and len(frame.f_locals["X"]) == len(folds[0][0]):
                observed_routes.append((model.kernel_computation_,
                                        model.kernel_self_construction_))
                observed_fits.append((model._cv_K.copy(), _fitted_snapshot(model, query)))

    try:
        sys.setprofile(observe)
        actual = run_cv(_model(backend=backend), X, y,
                        {"kernel_kws": [{"gamma": gamma} for gamma in gammas],
                         "lambd": lambdas}, scoring=_continuous_score, cv=folds,
                        refit_best=True)
    finally:
        sys.setprofile(previous_profile)
    assert sys.getprofile() is previous_profile
    assert observed_routes == [("guarded_norm_sum_rel_v2", "norm_first_v1")] * 8
    assert len(observed_fits) == len(expected_fits) == 8
    for observed, expected in zip(observed_fits, expected_fits):
        assert_array_equal(observed[0], expected[0])
        _assert_snapshot_equal(observed[1], expected[1])
    for index, fold in enumerate(actual[4]):
        assert_allclose(fold["test_score"], expected_test[index], rtol=1e-12, atol=1e-12)
        assert_allclose(fold["train_score"], expected_train[index], rtol=1e-12, atol=1e-12)
    means = np.mean(expected_test, axis=0)
    assert_allclose(actual[3]["mean_test_score"], means, rtol=1e-12, atol=1e-12)
    winner = int(np.argmax(means))
    gamma, lambd = settings[winner]
    assert type(actual[0]["kernel_kws"]["gamma"]) is type(gamma)
    assert actual[0]["lambd"] == lambd
    independent_refit = _model(gamma, lambd=lambd, backend=backend).fit(X, y)
    _assert_snapshot_equal(_fitted_snapshot(actual[2], query),
                           _fitted_snapshot(independent_refit, query))
    # Four preparations: two types per fold; each lambda change is a cache hit.
    # float64(.5) versus Python .5 deliberately rebuilds even with equal Grams.
    assert len(preparations) == 4
    assert [entry[0] for entry in preparations] == [type(g) for _ in folds for g in gammas]


def _type_sensitive_psd_kernel(train, query, *, gamma):
    scale = 1. if type(gamma) is float else 2.
    return scale * (np.asarray(train) @ np.asarray(query).T) + 1.


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("reverse", (False, True))
def test_equal_numeric_values_with_type_sensitive_callable_rebuild(dtype, reverse, data):
    X, y, query, _ = data
    old, new = dtype(.5), .5
    if reverse:
        old, new = new, old
    model = _model(old, kernel=_type_sensitive_psd_kernel).cv_init(X)
    old_gram = model._cv_K.copy()
    model.set_params(kernel_kws={"gamma": new})
    assert not model._cv_cache_matches(X)
    control = _model(new, kernel=_type_sensitive_psd_kernel).cv_init(X).fit(X, y)
    model.cv_init(X).fit(X, y)
    assert not np.array_equal(old_gram, control._cv_K)
    assert_array_equal(model._cv_K, control._cv_K)
    _assert_snapshot_equal(_fitted_snapshot(model, query), _fitted_snapshot(control, query))


@pytest.mark.parametrize("backend", BACKENDS)
def test_same_type_parameters_and_lambda_path_preserve_actual_cache_reuse(
        backend, data):
    X, y, _, _ = data
    model = _model(np.float32(.3), backend=backend).cv_init(X)
    gram, eigen = model._cv_K, model._K_eig
    model.set_params(kernel_kws=deepcopy(model.kernel_kws))
    assert model._cv_cache_matches(X.copy())
    with patch.object(model, "_compute_training_kernel", side_effect=AssertionError("unexpected rebuild")):
        for lambd in (.08, .16):
            model.set_params(lambd=lambd)
            assert model._cv_cache_matches(X)
            model.fit(X, y)
            assert model._cv_K is gram
            assert model._K_eig is eigen


@pytest.mark.parametrize("route", [
    {"backend": "cholesky"}, {"implementation": "reference"},
    {"solver_mode": "legacy"}, {"rbf_computation": "direct"},
    {"affine_computation": "joint"},
])
def test_route_changes_still_invalidate_and_match_independently_prepared_target(route, data):
    X, y, query, _ = data
    model = _model(np.float32(.3)).cv_init(X)
    removed = next((key for key in ('implementation', 'solver_mode') if key in route), None)
    if removed is not None:
        with pytest.raises(ValueError, match='Invalid parameter.*' + removed):
            model.set_params(**route)
        assert model._cv_cache_matches(X)
        return
    model.set_params(**route)
    assert not model._cv_cache_matches(X)
    control = _model(np.float32(.3), **route).cv_init(X).fit(X, y)
    model.cv_init(X).fit(X, y)
    assert_array_equal(model._cv_K, control._cv_K)
    _assert_snapshot_equal(_fitted_snapshot(model, query), _fitted_snapshot(control, query))
