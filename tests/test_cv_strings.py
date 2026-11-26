"""Check StringDType targets at package-owned CV and delegated sklearn boundaries."""
from functools import partial

import numpy as np
import pytest
from sklearn import config_context
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, get_scorer, make_scorer, recall_score,
)
from sklearn.model_selection import (
    GridSearchCV, KFold, RepeatedStratifiedKFold, StratifiedKFold, StratifiedShuffleSplit,
)

from dwd.cv import run_cv
from dwd.gen_dwd import GenDWD, GenDWDCV
from dwd.gen_kern_dwd import KernGDWD, KernGDWDCV
from dwd.kern_md import KernMD
from dwd.model_selection import fit_shared_fold_search


X = np.array([[-3.], [3.], [-2.], [2.], [0.5], [-0.5], [-1.], [1.],
              [-1.5], [1.5], [-2.5], [2.5]])


def string_dtype():
    if not hasattr(np.dtypes, "StringDType"):
        pytest.skip("Requires NumPy variable-width strings")
    return np.dtypes.StringDType()


def target():
    return np.array(["q", "q\0"] * 6, dtype=string_dtype())


def cv_inputs(kind):
    if kind == "default":
        return None, StratifiedKFold(5)
    if kind == "integer":
        return 2, StratifiedKFold(2)
    splitter = {
        "stratified": StratifiedKFold(3, shuffle=True, random_state=7),
        "repeated": RepeatedStratifiedKFold(n_splits=2, n_repeats=2, random_state=9),
        "shuffle": StratifiedShuffleSplit(n_splits=3, test_size=4, random_state=13),
        "plain": KFold(3),
    }[kind]
    return splitter, splitter


@pytest.mark.parametrize("cv_model,base", [(GenDWDCV, GenDWD), (KernGDWDCV, KernGDWD)])
@pytest.mark.parametrize("kind", ["default", "integer", "stratified", "repeated", "shuffle", "plain"])
@pytest.mark.parametrize("scoring", ["accuracy", "balanced_accuracy"])
def test_named_cv_matches_manual_string_folds(cv_model, base, kind, scoring):
    y = target()
    boxed = y.astype(object)
    cv, reference = cv_inputs(kind)
    common = dict(stopping="fixed", max_iter=5, random_state=42)
    lambdas = [.1, .6]
    options = dict(lambd_vals=lambdas, q_vals=[1.], cv=cv, scoring=scoring, **common)
    result = cv_model(**options).fit(X, y)
    object_result = cv_model(**options).fit(X, np.array(['a','b'] * 6, dtype=object))
    expected = []
    for (train, test), actual, boxed_actual in zip(reference.split(X, boxed), result.all_cv_results_,
                                                   object_result.all_cv_results_, strict=True):
        scores = []
        for i, lambd in enumerate(lambdas):
            model = base(lambd=lambd, q=1., **common).fit(X[train], boxed[train])
            for name, rows in (("train_score", train), ("test_score", test)):
                truth = boxed[rows]
                matches = model.predict(X[rows]).astype(object) == truth
                value = matches.mean() if scoring == "accuracy" else np.mean(
                    [matches[[v == label for v in truth]].mean() for label in set(truth)])
                assert actual[name][i] == pytest.approx(value)
                assert actual[name][i] == boxed_actual[name][i]
                if name == "test_score":
                    scores.append(value)
        expected.append(scores)
    means = np.mean(expected, axis=0)
    assert result.best_score_ == pytest.approx(means.max())
    assert result.best_params_ == object_result.best_params_
    assert result.classes_.dtype == result.best_estimator_.classes_.dtype == y.dtype
    assert result.classes_.tolist() == ["q", "q\0"]
    assert [(v == "q") for v in result.predict(X)] == [(v == "a") for v in object_result.predict(X)]


@pytest.mark.parametrize("cv_model", [GenDWDCV, KernGDWDCV])
def test_callable_and_custom_splitter_see_original_string_labels(cv_model):
    y = target()
    visits = []
    class DtypeAwareSplitter(StratifiedKFold):
        def split(self, x, labels, groups=None):
            assert labels.dtype == y.dtype and np.shares_memory(labels, y)
            yield from super().split(x, labels.astype(object), groups)
    def scorer(model, x, labels):
        assert labels.dtype == model.classes_.dtype == y.dtype
        visits.append(len(labels))
        return np.mean(model.predict(x).astype(object) == labels.astype(object))
    model = cv_model(lambd_vals=[.1], q_vals=[1.], cv=DtypeAwareSplitter(2),
                     scoring=scorer, stopping="fixed", max_iter=5).fit(X, y)
    assert len(visits) == 4 and len(model.all_cv_results_) == 2
    assert model.best_estimator_.classes_.dtype == y.dtype


@pytest.mark.parametrize("use_helper", [False, True])
def test_delegated_search_keeps_custom_splitter_and_scorer(use_helper):
    y = target()
    class Splitter:
        def get_n_splits(self, *args, **kwargs):
            return 2
        def split(self, x, labels, groups=None):
            assert labels is y
            yield from StratifiedKFold(2).split(x, labels.astype(object))
    def scorer(model, x, labels):
        assert labels.dtype == model.classes_.dtype == y.dtype
        return model.score(x, labels)
    splitter = Splitter()
    search = GridSearchCV(KernMD(), {}, n_jobs=1, scoring=scorer, cv=splitter, error_score="raise")
    if use_helper:
        fit_shared_fold_search(search, X, y)
    else:
        search.fit(X, y)
    assert search.cv is splitter and search.scoring is scorer
    assert search.best_estimator_.classes_.dtype == y.dtype


@pytest.mark.parametrize("scoring", [None, "balanced_accuracy", "roc_auc"])
def test_ordinary_strings_match_boxed_labels(scoring):
    y = np.array(["alpha", "beta"] * 6, dtype=string_dtype())
    kwargs = dict(lambd_vals=[.1, .6], q_vals=[1.], cv=3,
                  scoring=scoring, stopping="fixed", max_iter=5)
    actual = GenDWDCV(**kwargs).fit(X, y)
    boxed = GenDWDCV(**kwargs).fit(X, y.astype(object))
    np.testing.assert_allclose(actual.agg_cv_results_['mean_test_score'],
                               boxed.agg_cv_results_['mean_test_score'])
    assert actual.best_params_ == boxed.best_params_
    assert actual.predict(X).tolist() == boxed.predict(X).tolist()


@pytest.mark.parametrize("scoring", ["f1", "precision", "recall"])
def test_binary_positive_label_is_still_validated(scoring):
    with pytest.raises(ValueError, match="pos_label=1"):
        GenDWDCV(lambd_vals=[.1], q_vals=[1.], cv=2, scoring=scoring,
                 stopping="fixed", max_iter=5).fit(X, target())


def test_delegated_default_search_matches_stock_result_or_failure():
    y = target()
    def new_search():
        return GridSearchCV(KernMD(), {}, cv=2, scoring="accuracy", n_jobs=1, error_score="raise")
    search = new_search()
    try:
        stock = new_search().fit(X, y)
    except (TypeError, ValueError) as error:
        with pytest.raises(type(error)) as caught:
            fit_shared_fold_search(search, X, y)
        assert str(caught.value) == str(error)
    else:
        fit_shared_fold_search(search, X, y)
        np.testing.assert_equal(stock.cv_results_['mean_test_score'],
                                search.cv_results_['mean_test_score'])
        assert stock.best_params_ == search.best_params_
    assert search.cv == 2 and search.scoring == "accuracy"


@pytest.mark.parametrize("labels", [("cat", "dog"), ("q", "q\0"), ("id\0A", "id\0B")])
@pytest.mark.parametrize("which", ["get", "make", "positive", "selected"])
def test_sklearn_scorer_objects_match_manual_folds(labels, which):
    class Inherited(StratifiedKFold):
        pass

    y = np.asarray(labels * 6, dtype=string_dtype())
    positive = labels[1]
    original_options = np.array([positive], dtype=object)
    scorers = {
        "get": get_scorer("accuracy"),
        "make": make_scorer(accuracy_score),
        "positive": make_scorer(f1_score, pos_label=positive),
        "selected": make_scorer(recall_score, average="macro", labels=original_options,
                                pos_label=None, greater_is_better=False),
    }
    cv = Inherited(3)
    result = GenDWDCV(lambd_vals=[.1], q_vals=[1.], cv=cv, scoring=scorers[which],
                      stopping="fixed", max_iter=5).fit(X, y)
    encoded = np.arange(len(y)) % 2
    for (train, test), fold in zip(cv.split(X, encoded), result.all_cv_results_, strict=True):
        model = GenDWD(lambd=.1, q=1., stopping="fixed", max_iter=5).fit(X[train], encoded[train])
        truth, pred = encoded[test], model.predict(X[test])
        if which == "positive":
            expected = f1_score(truth, pred, pos_label=1)
        elif which == "selected":
            expected = -recall_score(truth, pred, average="macro", labels=[1])
        else:
            expected = accuracy_score(truth, pred)
        assert fold["test_score"][0] == pytest.approx(expected)
    assert original_options.tolist() == [positive]
    assert result.best_estimator_.classes_.dtype == y.dtype


@pytest.mark.parametrize("wrap", [False, True])
def test_make_scorer_custom_metric_sees_original_prediction_and_target(wrap):
    y = target()
    calls = []

    def metric(truth, predictions, *, pos_label):
        assert truth.dtype == predictions.dtype == y.dtype
        assert pos_label == "q\0"
        assert set(truth.tolist() + predictions.tolist()) == {"q", "q\0"}
        calls.append(len(truth))
        return np.mean(truth.astype(object) == predictions.astype(object))

    score = make_scorer(metric, pos_label="q\0")
    if wrap:
        score = partial(make_scorer(partial(metric, pos_label="q\0")))
    result = GenDWDCV(lambd_vals=[.1], q_vals=[1.], cv=3, scoring=score,
                      stopping="fixed", max_iter=5).fit(X, y)
    assert len(calls) == 6
    assert result.best_estimator_.classes_.dtype == y.dtype


@pytest.mark.parametrize("scoring", ["accuracy", "roc_auc"])
def test_public_run_cv_keeps_subclass_fit_and_response_on_the_original_fold(scoring):
    y = target()

    class Plain(GenDWD):
        pass

    class Observed(GenDWD):
        def __copy__(self):
            raise AssertionError("scoring must use the fitted model")

        def fit(self, X, labels):
            assert labels.dtype == y.dtype and set(labels.tolist()) == set(y.tolist())
            super().fit(X, labels)
            self.fit_owner_ = id(self)
            return self

        def decision_function(self, X):
            assert self.fit_owner_ == id(self) and self.classes_.dtype == y.dtype
            return super().decision_function(X)

        def predict(self, X):
            assert self.fit_owner_ == id(self) and self.classes_.dtype == y.dtype
            return super().predict(X)

    results = []
    for cls in (GenDWD, Plain, Observed):
        best, score, estimator, _, folds = run_cv(
            cls(stopping="fixed", max_iter=5), X, y, {"lambd": [.1, .6]},
            cv=3, scoring=scoring,
        )
        assert estimator.classes_.dtype == y.dtype
        results.append((best, score, [f["test_score"] for f in folds]))
    assert results[0] == results[1] == results[2]


def test_custom_probability_metric_uses_original_estimator_and_positive_class():
    y = target()

    class ProbabilityDWD(GenDWD):
        def fit(self, X, labels):
            super().fit(X, labels)
            self.fit_owner_ = id(self)
            return self

        def predict_proba(self, X):
            assert id(self) == self.fit_owner_ and self.classes_.dtype == y.dtype
            return np.tile([.25, .75], (len(X), 1))

    class Metric:
        __name__ = None

        def __call__(self, truth, probabilities, *, pos_label):
            assert truth.dtype == y.dtype and pos_label == "q\0"
            assert (probabilities == .75).all()
            return probabilities.mean()

    metric = make_scorer(Metric(), response_method="predict_proba", pos_label="q\0")
    result = run_cv(ProbabilityDWD(stopping="fixed", max_iter=5), X, y,
                    {"lambd": [.1]}, cv=3, scoring=metric)
    assert result[1] == .75


def test_custom_repeated_split_does_not_read_unused_splitter_descriptor():
    y = target()

    class Custom(RepeatedStratifiedKFold):
        @property
        def cv(self):
            raise AssertionError("the overridden split never uses this descriptor")

        @cv.setter
        def cv(self, value):
            pass

        def split(self, X, labels, groups=None):
            assert labels.dtype == y.dtype
            yield from KFold(3).split(X)

    result = GenDWDCV(lambd_vals=[.1], q_vals=[1.], cv=Custom(),
                      stopping="fixed", max_iter=5).fit(X, y)
    assert len(result.all_cv_results_) == 3


@pytest.mark.parametrize("cv_model", [GenDWDCV, KernGDWDCV])
@pytest.mark.parametrize("case", ["absent-NUL", "present-StringDType", "embedded-NUL"])
def test_explicit_scorer_labels_preserve_fold_scores_and_selection(cv_model, case):
    x = np.random.default_rng(1).integers(-3, 4, (12, 2)).astype(float)
    reference_y = np.array(["q", "r"] * 6, dtype=object)
    classes = ["id\0A", "id\0B"] if case == "embedded-NUL" else ["q", "r"]
    y = np.array(classes * 6, dtype=string_dtype())
    if case == "absent-NUL":
        labels, reference_labels = ["q\0"], ["absent"]
    elif case == "present-StringDType":
        labels, reference_labels = np.unique(y), ["q", "r"]
    else:
        labels, reference_labels = np.array(classes, dtype=string_dtype()), ["q", "r"]

    def search(target, selected):
        scorer = make_scorer(recall_score, labels=selected, average="macro",
                             pos_label=None, zero_division=0)
        return cv_model(lambd_vals=[.001, 1., 100.], q_vals=[1.], cv=3, scoring=scorer,
                        stopping="fixed", max_iter=5, random_state=7).fit(x, target)

    actual = search(y, labels)
    expected = search(reference_y, reference_labels)
    for fold, reference in zip(actual.all_cv_results_, expected.all_cv_results_, strict=True):
        for key in ("train_score", "test_score"):
            np.testing.assert_array_equal(fold[key], reference[key])
    assert actual.best_params_ == expected.best_params_
    predictions = actual.predict(x)
    assert predictions.dtype == actual.classes_.dtype == actual.best_estimator_.classes_.dtype == y.dtype
    assert actual.classes_.tolist() == classes
    assert predictions.tolist() == [classes[int(v == "r")] for v in expected.predict(x)]
    if case == "absent-NUL":
        assert actual.agg_cv_results_["mean_test_score"] == [0., 0., 0.]
        assert actual.best_params_["lambd"] == .001
    else:
        assert labels.dtype == y.dtype and labels.tolist() == classes


@pytest.mark.parametrize("cv_model", [GenDWDCV, KernGDWDCV])
@pytest.mark.parametrize("name,ratio", [("positive_likelihood_ratio", 2.),
                                       ("neg_negative_likelihood_ratio", -.5)])
@pytest.mark.parametrize("use_name", [True, False], ids=["name", "scorer-object"])
def test_registered_likelihood_ratio_scorers(cv_model, name, ratio, use_name):
    # Each fold has two correct and one incorrect prediction of each class.
    x = np.array([[-2.], [2.], [-2.], [2.], [2.], [-2.]] * 3)
    y = np.array(["q", "r"] * 9, dtype=string_dtype())
    options = dict(lambd_vals=[.1, .6], q_vals=[1.], cv=3,
                   scoring=name if use_name else get_scorer(name),
                   stopping="fixed", max_iter=5, random_state=7)
    actual = cv_model(**options).fit(x, y)
    expected = cv_model(**options).fit(x, y.astype(object))
    for fold, reference in zip(actual.all_cv_results_, expected.all_cv_results_, strict=True):
        for key in ("train_score", "test_score"):
            np.testing.assert_array_equal(fold[key], reference[key])
            np.testing.assert_allclose(fold[key], [ratio, ratio])
    assert actual.best_params_ == expected.best_params_
    assert actual.predict(x).tolist() == expected.predict(x).tolist()


@pytest.mark.parametrize("name,constant", [("positive_likelihood_ratio", 0),
                                          ("neg_negative_likelihood_ratio", 1)])
def test_undefined_likelihood_ratio_follows_installed_scorer(name, constant):
    class Constant(GenDWD):
        def predict(self, X):
            return self.classes_[np.full(len(X), constant)]

    y = np.array(["q", "r"] * 6, dtype=string_dtype())
    boxed = y.astype(object)
    model = Constant(stopping="fixed", max_iter=5)
    stock = get_scorer(name)(model.fit(X, boxed), X, boxed)
    grid = {"lambd": [.1]}
    if np.isfinite(stock):
        actual = run_cv(model, X, y, grid, cv=3, scoring=name)
        assert actual[1] == stock
    else:
        with pytest.raises(ValueError, match="finite real scalar score.*fold 0, train score"):
            run_cv(model, X, y, grid, cv=3, scoring=name)


def compare_search(y, scorer, reference_scorer):
    x = np.random.default_rng(1).integers(-3, 4, (12, 2)).astype(float)
    settings = dict(lambd_vals=[.001, 1., 100.], q_vals=[1.], cv=3,
                    stopping="fixed", max_iter=5, random_state=7)
    actual = GenDWDCV(scoring=scorer, **settings).fit(x, y)
    reference_y = np.array(["a", "b"] * 6, dtype=object)
    expected = GenDWDCV(scoring=reference_scorer, **settings).fit(x, reference_y)
    for fold, control in zip(actual.all_cv_results_, expected.all_cv_results_, strict=True):
        for key in ("train_score", "test_score"):
            np.testing.assert_allclose(fold[key], control[key])
    assert actual.best_params_ == expected.best_params_
    assert [p == y[1] for p in actual.predict(x)] == [p == "b" for p in expected.predict(x)]
    return actual


@pytest.mark.parametrize("storage", ["variable", "list", "tuple", "object"])
@pytest.mark.parametrize("kind", ["scorer", "metric"])
def test_keyword_partials_preserve_full_class_names(storage, kind):
    y = ["q", "q\0"] * 6
    if storage == "tuple":
        y = tuple(y)
    elif storage in ("variable", "object"):
        y = np.array(y, dtype=string_dtype() if storage == "variable" else object)
    if kind == "scorer":
        scorer = partial(get_scorer("accuracy"))
        reference = get_scorer("accuracy")
    else:
        scorer = make_scorer(partial(recall_score, average="macro", pos_label=None))
        reference = make_scorer(recall_score, average="macro", pos_label=None)
    result = compare_search(y, scorer, reference)
    assert result.classes_.tolist() == ["q", "q\0"]


@pytest.mark.parametrize("storage", ["variable", "list", "object"])
@pytest.mark.parametrize("binding", ["metric", "make_scorer", "call"])
def test_absent_class_in_partial_options_and_overrides(storage, binding):
    y = ["q", "r"] * 6
    if storage != "list":
        y = np.array(y, dtype=string_dtype() if storage == "variable" else object)
    absent = ["q\0"]
    metric = partial(recall_score, labels=absent if binding == "metric" else ["q"],
                     average="macro", pos_label=None, zero_division=0)
    scorer = make_scorer(metric, labels=absent) if binding == "make_scorer" else make_scorer(metric)
    if binding == "call":
        scorer = partial(scorer, labels=absent)
    reference = make_scorer(recall_score, labels=["absent"], average="macro",
                            pos_label=None, zero_division=0)
    with config_context(enable_metadata_routing=binding == "call"):
        result = compare_search(y, scorer, reference)
    assert result.agg_cv_results_["mean_test_score"] == [0., 0., 0.]
    assert result.best_params_["lambd"] == .001
    assert absent == ["q\0"]


@pytest.mark.parametrize("override", [False, True])
def test_partial_positive_class_selects_correct_decision_response(override):
    y = target()
    metric = partial(average_precision_score, pos_label="q\0")
    options = {"pos_label": "q"} if override else {}
    scorer = make_scorer(metric, response_method="decision_function", **options)
    reference = make_scorer(average_precision_score, response_method="decision_function",
                            pos_label="a" if override else "b")
    compare_search(y, scorer, reference)
    assert metric.keywords == {"pos_label": "q\0"}


@pytest.mark.parametrize("override", [False, True], ids=["configured", "call-time"])
def test_invalid_positive_class_diagnostic_names_the_original_labels(override):
    y = target()
    rejected = "q\0\0"
    responses = []

    class Observed(GenDWD):
        def decision_function(self, x):
            responses.append(len(x))
            return super().decision_function(x)

    scorer = make_scorer(average_precision_score, response_method="decision_function",
                         pos_label="q\0" if override else rejected)
    if override:
        scorer = partial(scorer, pos_label=rejected)
    with config_context(enable_metadata_routing=override), pytest.raises(ValueError) as caught:
        run_cv(Observed(stopping="fixed", max_iter=5), X, y, {"lambd": [.1]}, cv=3, scoring=scorer)
    assert repr(rejected) in str(caught.value) and "'q'" in str(caught.value)
    assert not getattr(caught.value, "__notes__", ())
    assert len(responses) == int(override)


def test_custom_scorer_keeps_its_own_label_error():
    error = ValueError("pos_label=custom is not a valid label")

    def metric(y, predictions):
        raise error

    with pytest.raises(ValueError) as caught:
        run_cv(GenDWD(stopping="fixed", max_iter=5), X, target(), {"lambd": [.1]},
               cv=3, scoring=make_scorer(metric))
    assert caught.value is error and not getattr(error, "__notes__", ())

    class OwnPrediction(GenDWD):
        def predict(self, x):
            raise error

    with pytest.raises(ValueError) as caught:
        run_cv(OwnPrediction(stopping="fixed", max_iter=5), X, target(), {"lambd": [.1]},
               cv=3, scoring="accuracy")
    assert caught.value is error and not getattr(error, "__notes__", ())


def test_sklearn_validates_positive_class_before_calling_custom_metric():
    y = target()
    rejected = "q\0\0"

    def metric(truth, predictions, *, pos_label):
        pytest.fail("sklearn must reject the positive class before calling the metric")

    with pytest.raises(ValueError) as caught:
        run_cv(GenDWD(stopping="fixed", max_iter=5), X, y, {"lambd": [.1]}, cv=3,
               scoring=make_scorer(metric, pos_label=rejected))
    assert repr(rejected) in str(caught.value)


def test_nested_metric_error_keeps_its_own_classes():
    class OwnMetric(GenDWD):
        def predict(self, x):
            return f1_score(["off", "on"], ["on", "off"], pos_label="invalid")

    with pytest.raises(ValueError, match="off.*on") as caught:
        run_cv(OwnMetric(stopping="fixed", max_iter=5), X, target(), {"lambd": [.1]},
               cv=3, scoring="accuracy")
    assert not getattr(caught.value, "__notes__", ())

    class OwnProperty(GenDWD):
        @property
        def decision_function(self):
            return f1_score(["off", "on"], ["on", "off"], pos_label="invalid")

    with pytest.raises(ValueError, match="off.*on") as caught:
        run_cv(OwnProperty(stopping="fixed", max_iter=5), X, target(), {"lambd": [.1]},
               cv=3, scoring="roc_auc")
    assert not getattr(caught.value, "__notes__", ())


@pytest.mark.parametrize("kind", ["fixed", "native"])
@pytest.mark.parametrize("provider", ["list", "series"])
def test_passive_string_rows_preserve_absent_class_recall(kind, provider):
    class Text(np.str_):
        pass

    dtype = str if kind == "fixed" else string_dtype()
    rows = [np.array("q", dtype=dtype), Text("q\0")] * 6
    if provider == "series":
        pd = pytest.importorskip("pandas")

        class SelectedRows(pd.Series):
            @property
            def __array__(self):
                raise AttributeError

            def __iter__(self):
                self.attrs["reads"] += 1
                assert self.attrs["reads"] == 1
                return iter(rows)

        y = SelectedRows(np.arange(len(rows)))
        y.attrs["reads"] = 0
    else:
        y = rows

    scorer = make_scorer(recall_score, labels=["q\0\0"], average="macro",
                         pos_label=None, zero_division=0)
    result = GenDWDCV(lambd_vals=[.02, 2.], q_vals=[1.], cv=3, stopping="fixed",
                      max_iter=4, random_state=7, scoring=scorer).fit(X, y)
    assert result.classes_.tolist() == ["q", "q\0"]
    for fold in result.all_cv_results_:
        assert fold["train_score"] == fold["test_score"] == [0., 0.]
