"""Removed solver selectors, current persistence, and immutable historical data."""

import json
import pickle
import warnings

import numpy as np
from _kernel_reference import kernel_route
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import BaseEstimator, ClassifierMixin, clone

import dwd.cv as cv_module
import dwd.gen_dwd as linear
import dwd.gen_kern_dwd as kernel
from legacy_solver_fixtures import CASE_NAMES, MANIFEST, load_legacy_case


MODES = ("legacy", "schur", "unknown", None)
ENTRYPOINTS = (
    "linear_fit", "linear_cv_init", "linear_cv_fit", "linear_solve",
    "kernel_fit", "kernel_cv_init", "kernel_cv_fit", "kernel_solve", "kernel_step",
)


class UntouchableData:
    """Any attempt to inspect or convert data before rejecting mode is a bug."""

    @property
    def shape(self):
        raise AssertionError("Data shape inspected before solver-mode rejection")

    def __array__(self, *args, **kwargs):
        raise AssertionError("Data converted before solver-mode rejection")

    def __len__(self):
        raise AssertionError("Data length inspected before solver-mode rejection")

    def dot(self, *args, **kwargs):
        raise AssertionError("Matrix product attempted before solver-mode rejection")


def unexpected_work(*args, **kwargs):
    raise AssertionError("Numerical/preparation/scoring work occurred before mode rejection")


def prevent_work(monkeypatch):
    # Patch the actual entrypoint modules, not unused upstream aliases. Leave
    # dispatch and cleanup intact so early-refusal behavior is exercised.
    for module in (linear, kernel, cv_module):
        for name in ("promote_sparse_features", "check_real_X_y", "check_array",
                     "get_P0_eig", "get_K_eig", "check_random_state", "check_scoring"):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, unexpected_work)


def assert_mode_error(call, mode, error_class=TypeError):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(error_class) as error:
            call()
    message = str(error.value)
    assert "solver_mode" in message
    assert not [item for item in caught if issubclass(item.category, FutureWarning)]


def entry_call(name, mode):
    data = UntouchableData()
    if name == "linear_fit":
        return lambda: linear.GenDWD(solver_mode=mode).fit(data, data)
    if name == "linear_cv_init":
        return lambda: linear.GenDWD(solver_mode=mode).cv_init(data)
    if name == "linear_cv_fit":
        return lambda: linear.GenDWDCV(solver_mode=mode).fit(data, data)
    if name == "linear_solve":
        return lambda: linear.solve_gen_dwd(data, data, .2, solver_mode=mode)
    if name == "kernel_fit":
        return lambda: kernel.KernGDWD(solver_mode=mode).fit(data, data)
    if name == "kernel_cv_init":
        return lambda: kernel.KernGDWD(solver_mode=mode).cv_init(data)
    if name == "kernel_cv_fit":
        return lambda: kernel.KernGDWDCV(solver_mode=mode).fit(data, data)
    if name == "kernel_solve":
        return lambda: kernel.solve_gen_kern_dwd(data, data, .2, solver_mode=mode)
    if name == "kernel_step":
        return lambda: kernel.get_step_implicit_P(
            data, data, 1., .2, data, 0., data, data, data, data, 1., solver_mode=mode)
    raise AssertionError(name)


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize("mode", MODES)
def test_mode_rejected_at_every_execution_boundary_before_work(entrypoint, mode, monkeypatch):
    prevent_work(monkeypatch)
    before = np.random.get_state()
    assert_mode_error(entry_call(entrypoint, mode), mode)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    assert_array_equal(before[1], after[1])


@pytest.mark.parametrize("estimator_class", (linear.GenDWD, kernel.KernGDWD))
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("path", ("run_cv", "get_path_scores"))
@pytest.mark.parametrize("position", ("first", "last"))
def test_entire_cv_grid_rejects_before_first_candidate(estimator_class, mode, path,
                                                       position, monkeypatch):
    prevent_work(monkeypatch)
    monkeypatch.setattr(estimator_class, "fit", unexpected_work)
    monkeypatch.setattr(estimator_class, "cv_init", unexpected_work)
    model = estimator_class()
    order = [mode, "schur"] if position == "first" else ["schur", mode]
    params = {"solver_mode": order, "lambd": [.2]}
    data = UntouchableData()
    if path == "run_cv":
        call = lambda: cv_module.run_cv(model, data, data, params, cv=2)
    else:
        call = lambda: cv_module.get_path_scores(
            model, data, data, data, data, unexpected_work, params)
    assert_mode_error(call, mode, ValueError)
    # Preflight must not set the caller's parameters to the rejected candidate.
    assert "solver_mode" not in model.get_params()


@pytest.mark.parametrize("estimator_class", (linear.GenDWD, kernel.KernGDWD))
@pytest.mark.parametrize("path", ("run_cv", "get_path_scores"))
def test_removed_selector_set_params_rejects_without_mutating_current_estimator(
        estimator_class, path, monkeypatch):
    prevent_work(monkeypatch)
    model = estimator_class()
    before = model.get_params(deep=False)
    assert_mode_error(lambda: model.set_params(solver_mode="legacy"), "legacy", ValueError)
    assert model.get_params(deep=False) == before


@pytest.mark.parametrize("path", ("run_cv", "get_path_scores"))
def test_unrelated_estimator_solver_mode_name_does_not_acquire_dwd_policy(path):
    events = []

    class ExternalClassifier(ClassifierMixin, BaseEstimator):
        # These labels belong to another estimator's API, not DWD's algebra.
        def __init__(self, solver_mode="legacy"):
            self.solver_mode = solver_mode

        def cv_init(self, X):
            events.append(("prepare", self.solver_mode))
            return self

        def fit(self, X, y):
            events.append(("fit", self.solver_mode))
            self.classes_ = np.unique(y)
            return self

        def predict(self, X):
            return np.repeat(self.classes_[0], len(X))

    X = np.arange(16, dtype=float).reshape(8, 2)
    y = np.tile([-1, 1], 4)
    model = ExternalClassifier()
    params = {"solver_mode": ["legacy", "external"]}
    if path == "run_cv":
        result = cv_module.run_cv(model, X, y, params, cv=2, refit_best=False)
        assert result[0] == {"solver_mode": "legacy"}
        assert model.solver_mode == "legacy"
        expected_fits = ["legacy", "external"] * 2
    else:
        result = cv_module.get_path_scores(
            model, X[:4], y[:4], X[4:], y[4:],
            lambda fitted, values, labels: fitted.score(values, labels), params)
        assert result["params"] == [{"solver_mode": "legacy"},
                                    {"solver_mode": "external"}]
        expected_fits = ["legacy", "external"]
    assert [mode for event, mode in events if event == "fit"] == expected_fits
    assert any(event == "prepare" for event, _ in events)


def current_case(name):
    """Reuse historical inputs for a fresh fit under current public settings."""
    saved = load_legacy_case(name)
    parameters = saved["model"].get_params(deep=False)
    assert "solver_mode" not in parameters
    # Migrate only the detached request for the new estimator. The archived
    # model keeps its original request, fitted policy and legacy attribution.
    if parameters.get("affine_computation") == "auto":
        parameters["affine_computation"] = "joint"
    if parameters.get("initialization") == "auto":
        parameters["initialization"] = "zero"
    return saved, type(saved["model"])(**parameters)


@pytest.mark.parametrize("name", CASE_NAMES)
def test_current_case_migrates_only_new_request_and_preserves_history(name, monkeypatch):
    historical = load_legacy_case(name)
    old_model = historical["model"]
    old_parameters = old_model.get_params(deep=False)
    states = [old_model]
    if hasattr(old_model, "best_estimator_"):
        states.append(old_model.best_estimator_)
    before = [(model, dict(vars(model)), fitted_arrays(model)) for model in states]
    monkeypatch.setitem(globals(), "load_legacy_case", lambda selected: historical)

    saved, current = current_case(name)

    assert saved is historical
    assert current is not old_model
    assert old_model.solver_mode == "legacy"
    if "affine_computation" in old_parameters:
        requested = old_parameters["affine_computation"]
        assert current.affine_computation == ("joint" if requested == "auto" else requested)
    if "initialization" in old_parameters:
        requested = old_parameters["initialization"]
        assert current.initialization == ("zero" if requested == "auto" else requested)
    for model, attributes, arrays in before:
        assert vars(model).keys() == attributes.keys()
        assert all(vars(model)[key] is value for key, value in attributes.items())
        for key, value in arrays.items():
            assert_array_equal(getattr(model, key), value)


@pytest.mark.parametrize("name", ("linear_implicit", "kernel_rbf"))
def test_current_cv_clones_do_not_mutate_the_fitted_caller(name, monkeypatch):
    saved, model = current_case(name)
    model.fit(saved["X"], saved["y"])
    parameters = model.get_params(deep=True)
    original_scores = model.decision_function(saved["query"])
    original_labels = model.predict(saved["query"])
    estimator_class = type(model)
    original_fit, original_init = estimator_class.fit, estimator_class.cv_init
    events = []

    def observed_fit(estimator, X, y, *args, **kwargs):
        assert estimator is not model
        assert "solver_mode" not in estimator.get_params()
        events.append("fit")
        return original_fit(estimator, X, y, *args, **kwargs)

    def observed_init(estimator, X):
        assert estimator is not model
        assert "solver_mode" not in estimator.get_params()
        events.append("prepare")
        return original_init(estimator, X)

    monkeypatch.setattr(estimator_class, "fit", observed_fit)
    monkeypatch.setattr(estimator_class, "cv_init", observed_init)
    params = {"lambd": [.1, .2], "max_iter": [1], "stopping": ["fixed"],
              "initialization": ["zero"]}
    result = cv_module.run_cv(model, saved["X"], saved["y"], params, cv=2)
    assert events.count("fit") == 5
    assert events.count("prepare") == 2
    assert "solver_mode" not in result[2].get_params()
    assert np.isfinite(result[2].decision_function(saved["query"])).all()
    assert model.get_params(deep=True) == parameters
    assert_array_equal(model.decision_function(saved["query"]), original_scores)
    assert_array_equal(model.predict(saved["query"]), original_labels)


def test_fixture_records_genuine_unchanged_pre_retirement_source():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert tuple(manifest["case_names"]) == CASE_NAMES
    expected = {
        "dwd/gen_dwd.py": "54d83898b36bbbe55601dc2f27a1c889bc84aec286ff4da4db54a917e991142c",
        "dwd/gen_kern_dwd.py": "d1513a32453d39a980e12e1d9f9c78ea4bef2cbde69f6ddaaa135a906c867eb7",
    }
    for path, digest in expected.items():
        assert manifest["source_sha256"][path] == digest
        assert manifest["required_source_pins"][path] == digest
    assert "dwd/_solver_mode.py" not in manifest["source_sha256"]
    for name in CASE_NAMES:
        record = manifest["cases"][name]
        assert record["fitted_solver_mode"] == "legacy"
        assert record["n_iter"] > 0
        if name.endswith("_cv"):
            assert record["nested_solver_mode"] == "legacy"


def fitted_arrays(model):
    """Snapshot coefficients and fitted basis; prediction caches may be lazy."""
    result = {}
    for name in ("coef_", "dual_coef_", "intercept_", "classes_", "_Xfit"):
        value = getattr(model, name, None)
        if isinstance(value, np.ndarray):
            result[name] = value.copy()
    return result


@pytest.mark.parametrize("name", CASE_NAMES)
def test_current_pickle_predictions_and_fitted_state_are_preserved(name):
    saved, model = current_case(name)
    model.fit(saved["X"], saved["y"])
    query = saved["query"]
    before = fitted_arrays(model)
    scores, predictions = model.decision_function(query), model.predict(query)
    restored = pickle.loads(pickle.dumps(model, protocol=4))
    assert_array_equal(restored.decision_function(query), scores)
    assert_array_equal(restored.predict(query), predictions)
    assert "solver_mode" not in model.get_params()
    assert "solver_mode" not in restored.get_params()
    for attribute, value in before.items():
        assert_array_equal(getattr(model, attribute), value)
        assert_array_equal(getattr(restored, attribute), value)


@pytest.mark.parametrize("name", ("linear_implicit", "linear_cv", "kernel_rbf", "kernel_cv"))
@pytest.mark.parametrize("mode", MODES)
def test_removed_set_params_preserves_fitted_state_and_private_cache(name, mode):
    saved, model = current_case(name)
    model.fit(saved["X"], saved["y"])
    scores = model.decision_function(saved["query"])
    sentinel = object()
    model._selector_test_cache = sentinel
    assert_mode_error(lambda: model.set_params(solver_mode=mode), mode, ValueError)
    assert model._selector_test_cache is sentinel
    assert_array_equal(model.decision_function(saved["query"]), scores)


@pytest.mark.parametrize("name", ("linear_implicit", "kernel_rbf"))
@pytest.mark.parametrize("mode", MODES)
def test_removed_set_params_preserves_prepared_cache(name, mode):
    saved, model = current_case(name)
    model.cv_init(saved["X"])
    before = dict(vars(model))
    assert_mode_error(lambda: model.set_params(solver_mode=mode), mode, ValueError)
    assert vars(model).keys() == before.keys()
    assert all(vars(model)[key] is value for key, value in before.items())


@pytest.mark.parametrize("name", ("linear_implicit", "linear_explicit", "linear_cv",
                                  "kernel_rbf", "kernel_precomputed", "kernel_cv"))
def test_current_clone_refits_and_leaves_original_usable(name):
    saved, original = current_case(name)
    original.fit(saved["X"], saved["y"])
    before = original.decision_function(saved["query"])
    labels = original.predict(saved["query"])
    copied = clone(original)
    assert "solver_mode" not in copied.get_params()
    copied.fit(saved["X"], saved["y"])
    assert np.isfinite(copied.decision_function(saved["query"])).all()
    assert_array_equal(original.decision_function(saved["query"]), before)
    assert_array_equal(original.predict(saved["query"]), labels)


@pytest.mark.parametrize("implicit", (False, True))
def test_corrected_linear_fitting_still_supports_explicit_and_implicit_systems(implicit):
    saved = load_legacy_case("linear_implicit")
    model = linear.GenDWD(implicit_P=implicit, lambd=.2,
                          max_iter=3, stopping="fixed", initialization="zero").fit(
                              saved["X"], saved["y"])
    assert model.n_iter_ == 3
    assert np.isfinite(model.decision_function(saved["query"])).all()
    assert "solver_mode" not in model.get_params()


@pytest.mark.parametrize("implementation", ("reference", "optimized"))
def test_corrected_kernel_reference_and_optimized_controls(implementation):
    saved = load_legacy_case("kernel_rbf")
    model = kernel_route(implementation=implementation,
                            kernel="rbf", kernel_kws={"gamma": .2}, lambd=.2,
                            max_iter=3, stopping="fixed", initialization="zero",
                            acceleration=None, residual_check_order="refinement_first").fit(
                                saved["X"], saved["y"])
    assert model.n_iter_ == 3
    assert np.isfinite(model.decision_function(saved["query"])).all()
    assert "solver_mode" not in model.get_params()


def test_fixture_loads_are_independent_objects_and_arrays():
    first, second = load_legacy_case("kernel_rbf"), load_legacy_case("kernel_rbf")
    assert first["model"] is not second["model"]
    assert not np.shares_memory(first["model"].dual_coef_, second["model"].dual_coef_)
    for name in ("X", "y", "query", "scores", "labels"):
        assert not np.shares_memory(first[name], second[name])
    before = second["model"].dual_coef_.copy()
    first["model"].dual_coef_[0, 0] += 1
    assert_array_equal(second["model"].dual_coef_, before)
