"""CV grid defaults are immutable and keep sklearn parameter contracts.

These tests require no external fixtures and do not run numerical fits. HTML
asset decoding belongs to sklearn and is tested separately from plain repr.
"""

from copy import deepcopy
import importlib
import inspect
import pickle
from unittest.mock import patch

import numpy as np
import pytest
from sklearn import config_context
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_parameters_default_constructible

from dwd.gen_dwd import GenDWDCV
from dwd.gen_kern_dwd import KernGDWDCV


CLASSES = (GenDWDCV, KernGDWDCV)
GRID_NAMES = ('lambd_vals', 'q_vals')


def expected_grids():
    return dict(lambd_vals=np.logspace(-2, 2, 10),
                q_vals=np.logspace(-2, 2, 5))


def explicit_grids(kind):
    values = dict(lambd_vals=[.125, .5], q_vals=[1., 2.3])
    if kind == 'array':
        return {name: np.asarray(value) for name, value in values.items()}
    if kind == 'tuple':
        return {name: tuple(value) for name, value in values.items()}
    assert kind == 'list'
    return values


def assert_grid_equal(actual, expected):
    assert type(actual) is type(expected)
    np.testing.assert_array_equal(actual, expected)


def assert_default_grid(value, expected):
    assert type(value) is tuple
    assert all(type(element) is np.float64 for element in value)
    np.testing.assert_array_equal(value, expected)


def assert_plain_representations(model):
    for subject in (model, Pipeline([('cv', model)])):
        for rendered in (repr(subject), str(subject)):
            assert type(rendered) is str
            assert type(subject).__name__ in rendered
            assert type(model).__name__ in rendered


@pytest.mark.parametrize('cls', CLASSES)
@pytest.mark.parametrize('changed_only', [True, False])
@pytest.mark.parametrize('custom', [False, True])
def test_plain_representations_and_pipeline(cls, changed_only, custom):
    model = cls(**(explicit_grids('array') if custom else {}))
    with config_context(print_changed_only=changed_only):
        assert_plain_representations(model)


@pytest.mark.parametrize('cls', CLASSES)
def test_sklearn_default_constructibility(cls):
    # Some sklearn versions return early on a signature-comparison error. The
    # independent direct signature assertions below also have to pass.
    check_parameters_default_constructible(cls.__name__, cls())


@pytest.mark.parametrize('cls', CLASSES)
def test_signature_defaults_match_stored_immutable_grids(cls):
    signature = inspect.signature(cls.__init__)
    model = cls()
    parameters = model.get_params(deep=False)
    for name, expected in expected_grids().items():
        default = signature.parameters[name].default
        assert (default == inspect.Parameter.empty) is False
        assert_default_grid(default, expected)
        assert parameters[name] is default
        assert getattr(model, name) is default


@pytest.mark.parametrize('cls', CLASSES)
@pytest.mark.parametrize('kind', ['list', 'tuple', 'array'])
@pytest.mark.parametrize('route', ['constructor', 'set_params'])
def test_explicit_grid_identity_and_clone(cls, kind, route):
    grids = explicit_grids(kind)
    if route == 'constructor':
        model = cls(**grids)
    else:
        model = cls()
        assert model.set_params(**grids) is model
    assert model.set_params() is model
    parameters = model.get_params(deep=False)
    for name, value in grids.items():
        assert getattr(model, name) is value
        assert parameters[name] is value
    copied = clone(model)
    assert type(copied) is cls
    for name, value in grids.items():
        assert_grid_equal(getattr(copied, name), value)
        if kind in ('list', 'array'):
            assert getattr(copied, name) is not value
    assert not hasattr(copied, 'best_estimator_')


@pytest.mark.parametrize('cls', CLASSES)
@pytest.mark.parametrize('custom', [False, True])
@pytest.mark.parametrize('method', ['deepcopy', 'pickle4', 'pickle5'])
def test_roundtrips_keep_requested_grid_representation(cls, custom, method):
    model = cls(**(explicit_grids('array') if custom else {}))
    restored = (deepcopy(model) if method == 'deepcopy'
                else pickle.loads(pickle.dumps(model, protocol=int(method[-1]))))
    assert type(restored) is cls
    for name in GRID_NAMES:
        assert_grid_equal(getattr(restored, name), getattr(model, name))
    with config_context(print_changed_only=True):
        assert_plain_representations(restored)


@pytest.mark.parametrize('cls', CLASSES)
def test_default_instances_cannot_share_mutable_default_grids(cls):
    first, second = cls(), cls()
    before = second.get_params(deep=False)
    for name, expected in expected_grids().items():
        assert_default_grid(getattr(first, name), expected)
        assert_default_grid(getattr(second, name), expected)
        with pytest.raises(TypeError):
            getattr(first, name)[0] = 17.
    replacement = explicit_grids('array')
    first.set_params(**replacement)
    replacement['lambd_vals'][0] = 17.
    for name, expected in expected_grids().items():
        assert getattr(second, name) is before[name]
        assert_default_grid(getattr(second, name), expected)
        assert_default_grid(getattr(cls(), name), expected)


@pytest.mark.parametrize('cls', CLASSES)
@pytest.mark.parametrize('name', ['not_a_parameter', 'unknown__value'])
def test_invalid_root_does_not_render_or_mutate(cls, name):
    model = cls()
    before = model.get_params(deep=False)
    with patch.object(cls, '__repr__', side_effect=AssertionError('rendered')):
        with pytest.raises(ValueError, match=name):
            model.set_params(**{name: 'invalid'})
    assert model.get_params(deep=False).keys() == before.keys()
    assert all(getattr(model, key) is value for key, value in before.items())
    assert not hasattr(model, name)


@pytest.mark.parametrize('cls', CLASSES)
def test_mixed_valid_invalid_roots_still_reject_atomically(cls):
    model = cls()
    before = model.get_params(deep=False)
    marker = model.observed_state_ = object()
    with pytest.raises(ValueError, match='unknown'):
        model.set_params(max_iter=17, unknown='invalid')
    assert all(getattr(model, key) is value for key, value in before.items())
    assert model.observed_state_ is marker
    assert not hasattr(model, 'unknown')


class CapturedGrid(Exception):
    """Expected boundary: no real CV preparation or numerical fit starts."""


@pytest.mark.parametrize('cls', CLASSES)
@pytest.mark.parametrize('explicit', [None, 'lambd_vals', 'q_vals'])
def test_fit_uses_exact_grids_without_mutating_requests(cls, explicit, monkeypatch):
    grids = explicit_grids('array')
    model = cls(**({explicit: grids[explicit]} if explicit else {}))
    before = model.get_params(deep=False)
    copied_grids = {name: deepcopy(before[name]) for name in GRID_NAMES}
    # A CV failure must still invalidate an old predictor and any partial state.
    model.best_estimator_ = object()
    model.classes_ = np.array([0, 1])
    model._Xfit = np.ones((2, 1))
    captured = []

    def intercept(**kwargs):
        captured.append(kwargs)
        raise CapturedGrid('captured before CV')

    module = importlib.import_module(cls.__module__)
    monkeypatch.setattr(module, 'run_cv', intercept)
    with pytest.raises(CapturedGrid, match='before CV'):
        model.fit(np.array([[0., 1.], [1., 0.], [2., 1.], [1., 2.]]),
                  np.array([0, 0, 1, 1]))
    assert len(captured) == 1
    actual = captured[0]['params']
    for name, expected in expected_grids().items():
        effective = grids[name] if name == explicit else expected
        values = actual[{'lambd_vals': 'lambd', 'q_vals': 'q'}[name]]
        np.testing.assert_array_equal(values, effective)
        assert all(type(element) is np.float64 for element in values)
        assert getattr(model, name) is before[name]
        assert_grid_equal(getattr(model, name), copied_grids[name])
    assert all(getattr(model, key) is value for key, value in before.items())
    assert not any(name.endswith('_') and not name.startswith('_')
                   for name in vars(model))
    assert not hasattr(model, '_Xfit')
