"""Sharing RHS splits must preserve callbacks, recovery and input ownership.

Existing feature-batching tests cover scalar/tile arithmetic and report fields.
These tests target boundaries crossed specifically by preparing a split once.
All numerical expectations below are exact dyadic actions or analytic zeros.
"""
import sys

import numpy as np
import pytest

from dwd import _compensated_residual as arithmetic
from _feature_report_test_utils import (feature, support, same, pins, contract,
    contraction_data, exact_contract, numpy_policy, portable_scores)

pytestmark = pytest.mark.usefixtures('numpy_policy')


@pytest.mark.parametrize('family', ['coefficients', 'gradient'])
@pytest.mark.parametrize('callback', ['_fsum', '_restore_gradient_terms'])
def test_local_callback_keeps_later_splitter_observable(monkeypatch, family, callback):
    data = contraction_data()
    before, expected = pins(data), exact_contract(family, *data)
    original_split = arithmetic._split_operand
    original_callback = getattr(feature, callback)
    observations, changed = [], False

    def observing_split(operand):
        observations.append(tuple(operand.shape))
        return original_split(operand)

    def observing_callback(*args, **kwargs):
        nonlocal changed
        if not changed:
            monkeypatch.setattr(arithmetic, '_split_operand', observing_split)
            changed = True
        return original_callback(*args, **kwargs)

    monkeypatch.setattr(feature, callback, observing_callback)
    same(contract(family, *data), expected)
    # Two full64-column tiles. Restoration also precedes the two remaining
    # products in the first gradient tile; fsum occurs after those products.
    count = 1 if family == 'coefficients' else 3 if callback == '_fsum' else 5
    assert changed and observations == [(64, 4)] * count
    assert pins(data) == before


@pytest.mark.parametrize('family', ['coefficients', 'gradient'])
def test_scalar_recovery_invalidates_splits_before_later_tile(monkeypatch, family):
    # A finite expansion whose first column sum overflows. A caller-supplied
    # scalar callback owns its replacement0; no correctness claim is made for
    # that replacement. Later original scalar and tiled calls must see its hook.
    features = np.full((4, 128), .125)
    features[:, 0] = 2.**1023
    alpha = np.ones(4) if family == 'coefficients' else np.zeros(4)
    z = np.ones(4)
    before = pins((features, alpha, z))
    original_split = arithmetic._split_operand
    attribute = '_beta_column' if family == 'coefficients' else '_gradient_column'
    original_column = getattr(feature, attribute)
    observations, recoveries = [], []

    def observing_split(operand):
        observations.append(tuple(operand.shape))
        return original_split(operand)

    def recover_column(column, *args):
        try:
            return original_column(column, *args)
        except FloatingPointError as error:
            assert str(error) == 'Nonfinite or overflowing compensated residual sum.'
            np.testing.assert_array_equal(column, np.full(4, 2.**1023))
            assert not recoveries
            recoveries.append(str(error))
            monkeypatch.setattr(arithmetic, '_split_operand', observing_split)
            return 0.

    monkeypatch.setattr(feature, attribute, recover_column)
    actual = (feature._feature_coefficients(features, alpha) if family == 'coefficients'
              else feature._feature_gradient(features, z, alpha, 1.))
    products = 1 if family == 'coefficients' else 3
    assert observations == [(4,)] * (63 * products) + [(64, 4)] * products
    assert len(recoveries) == 1
    expected = np.full(128, .5)
    expected[0] = 0.
    same(actual, expected)
    assert pins((features, alpha, z)) == before


def _stack_contains(code):
    frame = sys._getframe(1)
    try:
        while frame is not None:
            if frame.f_code is code:
                return True
            frame = frame.f_back
        return False
    finally:
        del frame


def _delayed_finfo(monkeypatch, family, operation):
    original_finfo, original_split = np.finfo, arithmetic._split_operand
    target = (feature._feature_coefficients if family == 'coefficients'
              else feature._feature_gradient).__code__
    restoration = feature._restore_gradient_terms.__code__
    observations = []
    state = dict(changed=False, gradual_before_change=0)
    with monkeypatch.context() as scoped:
        def observing_split(operand):
            if _stack_contains(target):
                observations.append(tuple(operand.shape))
            return original_split(operand)

        def delayed_finfo(*args, **kwargs):
            if not state['changed'] and _stack_contains(arithmetic._gradual_underflow.__code__):
                state['gradual_before_change'] += 1
            if not state['changed'] and _stack_contains(restoration) and _stack_contains(target):
                scoped.setattr(arithmetic, '_split_operand', observing_split)
                state['changed'] = True
            return original_finfo(*args, **kwargs)

        scoped.setattr(np, 'finfo', delayed_finfo)
        result = operation()
    assert np.finfo is original_finfo and arithmetic._split_operand is original_split
    assert state['changed']
    assert observations == [(64, 4)] * (1 if family == 'coefficients' else 5)
    return result, state


@pytest.mark.parametrize('family', ['coefficients', 'gradient'])
def test_private_contraction_observes_restoration_delayed_finfo(monkeypatch, family):
    data = contraction_data()
    before, expected = pins(data), exact_contract(family, *data)
    result, state = _delayed_finfo(monkeypatch, family, lambda: contract(family, *data))
    same(result, expected)
    assert state['gradual_before_change'] == 0
    assert pins(data) == before


@pytest.mark.parametrize('family', ['coefficients', 'gradient'])
def test_public_report_observes_restoration_delayed_finfo(monkeypatch, portable_scores, family):
    features = np.zeros((4, 128))
    labels = np.array([-1., 1., -1., 1.])
    alpha = np.array([.5, -.25, .125, -.0625])
    checked, optimizer = np.zeros(4), np.zeros(4)
    arrays = features, labels, alpha, checked, optimizer
    before = pins(arrays)

    def report():
        return feature.linear_feature_diagnostics(features, labels, alpha, 0., 1., 1.,
            checked_scores=checked, optimizer_scores=optimizer,
            checked_objective=1., optimizer_objective=1.)

    expected = report()
    actual, state = _delayed_finfo(monkeypatch, family, report)
    same(actual, expected)
    assert actual[0]['final_objective'] == 1.
    assert actual[0]['rkhs_gradient_norm'] == actual[0]['gradient_inf_norm'] == 0.
    # Initial range checking must not activate the hook intended for restoration.
    assert state['gradual_before_change'] >= 1
    assert pins(arrays) == before


@pytest.mark.parametrize('family', ['coefficients', 'gradient'])
@pytest.mark.parametrize('n,p', [(4, 1), (1025, 2), (2049, 2)])
def test_scalar_width_does_not_prepare_unused_splits(monkeypatch, family, n, p):
    features = np.full((n, p), .125)
    alpha = np.full(n, .25)
    z = np.full(n, -.125)
    expected = exact_contract(family, features, alpha, z)

    def forbidden(*args, **kwargs):
        raise AssertionError('no tiled product can use a prepared split')

    monkeypatch.setattr(support, 'repeated_operand_splits', forbidden)
    same(contract(family, features, alpha, z), expected)


@pytest.mark.parametrize('family', ['coefficients', 'gradient'])
def test_changed_restore_body_preserves_exception(monkeypatch, family):
    def changed(terms, exponent):
        raise LookupError('restoration body remains observable')

    def forbidden(*args, **kwargs):
        raise AssertionError('replaced restoration must decline preparation')

    monkeypatch.setattr(feature._restore_gradient_terms, '__code__', changed.__code__)
    monkeypatch.setattr(support, 'repeated_operand_splits', forbidden)
    with pytest.raises(LookupError, match='^restoration body remains observable$'):
        contract(family, *contraction_data())


@pytest.mark.parametrize('family', ['coefficients', 'gradient'])
@pytest.mark.parametrize('attribute,replacement', [('__defaults__', ()), ('__kwdefaults__', {})])
def test_changed_restore_defaults_decline_preparation(monkeypatch, family, attribute, replacement):
    data = contraction_data()
    expected = exact_contract(family, *data)
    monkeypatch.setattr(feature._restore_gradient_terms, attribute, replacement)

    def forbidden(*args, **kwargs):
        raise AssertionError('replaced restoration must decline preparation')

    monkeypatch.setattr(support, 'repeated_operand_splits', forbidden)
    same(contract(family, *data), expected)


@pytest.mark.parametrize('family,attribute', [('coefficients', '_beta_tile'), ('gradient', '_gradient_tile')])
def test_replaced_tile_keeps_positional_callback_contract(monkeypatch, family, attribute):
    data = contraction_data()
    expected = exact_contract(family, *data)
    original, calls = getattr(feature, attribute), []

    def positional_only(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(feature, attribute, positional_only)
    same(contract(family, *data), expected)
    assert calls == [True, True]


@pytest.mark.parametrize('family', ['coefficients', 'gradient'])
def test_replaced_products_keep_positional_callback_contract(monkeypatch, family):
    data = contraction_data()
    expected = exact_contract(family, *data)
    original, calls = feature._products, []

    def positional_only(left, right):
        calls.append(left.shape)
        return original(left, right)

    monkeypatch.setattr(feature, '_products', positional_only)
    same(contract(family, *data), expected)
    assert calls
