"""Plot artists and signed predictions retain the documented helper behavior."""
from copy import deepcopy
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib", reason="Plot helpers require optional matplotlib.")
matplotlib.use("Agg", force=True)
from matplotlib import colors
import matplotlib.pyplot as plt

from dwd import sim_fun


@pytest.fixture
def axes():
    # A known ordinary style makes line-style assertions independent of the
    # caller's matplotlibrc. This context changes no persistent configuration.
    with matplotlib.rc_context({"lines.linestyle": "-", "text.usetex": False}):
        figure, axis = plt.subplots()
        try:
            yield figure, axis
        finally:
            plt.close(figure)


@pytest.mark.parametrize("kind,std,log,color", [
    pytest.param("test", True, True, "red", id="test-default-red-std-log"),
    pytest.param("train", True, False, "blue", id="train-blue-std-linear"),
    pytest.param("test", True, False, "#2864aa", id="test-hex-std-linear"),
    pytest.param("train", True, True, (0.125, 0.5, 0.75, 0.75), id="train-rgba-std-log"),
    pytest.param("test", False, True, "blue", id="test-blue-no-std-log"),
    pytest.param("train", False, False, "#a63d40", id="train-hex-no-std-linear"),
])
def test_cv_tuning_preserves_artists_without_redundant_color_warning(axes, kind, std, log, color):
    figure, axis = axes
    parameter = "model__C"
    values = np.array([0.25, 1.0, 4.0])
    results = {
        "params": [{parameter: value} for value in values],
        "mean_test_score": np.array([0.25, 0.5, 0.75]),
        "std_test_score": np.array([0.125, 0.125, 0.0625]),
        "mean_train_score": np.array([0.375, 0.625, 0.875]),
        "std_train_score": np.array([0.0625, 0.125, 0.0625]),
    }
    original = deepcopy(results)
    fitted_search = SimpleNamespace(cv_results_=results)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="color is redundantly defined.*", category=UserWarning)
        returned = sim_fun.cv_tuning(fitted_search, parameter, kind=kind, std=std, log=log, color=color)
        figure.canvas.draw()
    assert returned is None
    mean, spread = results["mean_" + kind + "_score"], results["std_" + kind + "_score"]
    expected_y = [mean] + ([mean + spread, mean - spread] if std else [])
    assert len(axis.lines) == len(expected_y)
    for position, (line, expected) in enumerate(zip(axis.lines, expected_y, strict=True)):
        np.testing.assert_array_equal(line.get_xdata(), values)
        np.testing.assert_array_equal(line.get_ydata(), expected)
        assert colors.to_rgba(line.get_color()) == colors.to_rgba(color)
        assert line.get_linestyle() == ("-" if position == 0 else "--")
    assert axis.lines[0].get_label() == "cv " + kind
    assert [text.get_text() for text in axis.get_legend().get_texts()] == ["cv " + kind]
    assert axis.get_xscale() == ("log" if log else "linear")
    assert axis.get_xlabel() == parameter
    assert axis.get_ylim() == (0.0, 1.0)
    assert results["params"] == original["params"]
    for key in ("mean_test_score", "std_test_score", "mean_train_score", "std_train_score"):
        np.testing.assert_array_equal(results[key], original[key])


def test_tuning_curve_labels_conic_C_and_preserves_solver_error_contract(axes):
    figure, axis = axes
    X_train = np.array([[-2., 0.], [-1., 1.], [0., -1.], [2., 0.]])
    y_train = np.array([-1., -1., 1., 1.])
    X_test = np.array([[-3., 1.], [0., 0.], [1., -1.], [2., 1.]])
    y_test = np.array([-1., -1., -1., 1.])
    inputs_before = [value.copy() for value in (X_train, y_train, X_test, y_test)]
    values = [0.25, 2.0]
    options = {"solver": "fixture-no-real-solver", "max_iter": 7}
    options_before = deepcopy(options)
    seen = []

    def solver(X, y, C, *, solver_kws):
        assert X is X_train and y is y_train
        assert solver_kws is options
        seen.append(C)
        # Exact affine scores produce different nontrivial errors. A zero
        # score retains the helper's signed-label tie value, rather than +1.
        if C == 0.25:
            return np.array([1., 0.]), 0., {"unused_solver_diagnostic": True}
        assert C == 2.0
        return np.array([-1., 0.]), 1., {"unused_solver_diagnostic": True}

    train, test = sim_fun.tuning_curve(X_train, y_train, X_test, y_test,
                                     values, solver, solver_kws=options)
    figure.canvas.draw()  # Includes actual rendering of the C mathtext label.
    assert seen == values
    assert options == options_before
    np.testing.assert_array_equal(train, [0.25, 0.75])
    np.testing.assert_array_equal(test, [0.5, 1.0])
    assert train.shape == test.shape == (2,)
    assert len(axis.lines) == 2
    for line, errors, label in zip(axis.lines, (train, test), ("Train error", "Test error"), strict=True):
        np.testing.assert_array_equal(line.get_xdata(), values)
        np.testing.assert_array_equal(line.get_ydata(), errors)
        assert line.get_label() == label
        assert line.get_linestyle() == "-"
    assert [text.get_text() for text in axis.get_legend().get_texts()] == ["Train error", "Test error"]
    assert axis.get_xscale() == "log"
    assert axis.get_xlabel() == r"$C$"
    assert axis.get_ylim() == (0.0, 1.0)
    for value, original in zip((X_train, y_train, X_test, y_test), inputs_before, strict=True):
        np.testing.assert_array_equal(value, original)


@pytest.mark.parametrize("features,offset,expected", [
    pytest.param(np.array([2., 1.]), 0.25, np.array([1.]), id="vector-positive-length-one"),
    pytest.param(np.array([2., 1.]), 0.0, np.array([0.]), id="vector-zero-tie-length-one"),
    pytest.param(np.array([[2., 1.], [1., 1.], [3., 1.]]), 0.0,
                 np.array([0., -1., 1.]), id="batch-zero-negative-positive"),
])
def test_linear_prediction_vector_batch_and_zero_tie(features, offset, expected):
    beta = np.array([0.5, -1.0])
    before, beta_before = features.copy(), beta.copy()
    result = sim_fun.lin_clf_predict(features, beta, offset)
    assert isinstance(result, np.ndarray)
    assert result.shape == expected.shape
    np.testing.assert_array_equal(result, expected)
    if features.ndim == 2:
        for row, label in zip(features, expected, strict=True):
            one = sim_fun.lin_clf_predict(row, beta, offset)
            assert one.shape == (1,)
            np.testing.assert_array_equal(one, [label])
    np.testing.assert_array_equal(features, before)
    np.testing.assert_array_equal(beta, beta_before)
