"""Exact original-feature MM coordinates, rounding cells and finite resources."""
import importlib.util
from pathlib import Path
import struct

import numpy as np
import pytest

from dwd._exact_linear_mm import ExactCoordinateMM, ExactCoordinatePolicy, ExactCoordinateLimitError


_spec = importlib.util.spec_from_file_location(
    '_f11_exact_test_oracle', Path(__file__).with_name('fixtures') / 'f11_exact_oracle.py')
oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oracle)
word, bits = oracle.word, oracle.bits
ExactAction, assert_receipt = oracle.ExactAction, oracle.assert_receipt
_SIGN, _MAX = oracle._SIGN, oracle._MAX


@pytest.mark.parametrize('case,q', [('ordinary', 1), ('ordinary', 2),
    ('original_far_outlier', 1), ('constant_positive', 1), ('wide_small_positive', 1)])
def test_original_feature_gaussian_oracle(case, q):
    X, y, penalty = oracle.case(case)
    shift = float(2*len(X)*penalty/float(q+2+1/q))
    old = np.zeros(X.shape[1]+1)
    h = np.asarray(-y/float(q+2+1/q), dtype=np.float64)
    action = ExactAction(X, shift, h, old)
    theta, receipt = ExactCoordinateMM(X, shift).next_iterate(h, old[1:], float(old[0]))
    assert_receipt(action, theta, receipt)


@pytest.mark.parametrize('kind', ['even_intercept', 'odd_intercept', 'coefficient',
                                  'half_subnormal', 'nonzero_subnormal'])
def test_hardcoded_rounding_cells(kind):
    records = []
    for sign in (1.0, -1.0):
        X = np.zeros((2, 1))
        shift = 1.0
        old = np.zeros(2)
        if kind in ('even_intercept', 'odd_intercept'):
            old[0] = sign*(1.0 if kind == 'even_intercept' else 1.0+2.0**-52)
            h = np.array([-sign*2.0**-52, 0.0])
            expected = sign*(1.0 if kind == 'even_intercept' else 1.0+2.0**-51)
        elif kind == 'coefficient':
            X[:, 0] = [-1.0, 1.0]
            shift = 2.0
            h = sign*np.array([2.0, -(2.0+2.0**-51)])
            expected = sign
        else:
            minimum = struct.unpack('>d', struct.pack('>Q', 1))[0]
            hword = 1 | (_SIGN if sign > 0 else 0)
            h = np.frombuffer(struct.pack('=QQ', hword,
                              0 if kind == 'half_subnormal' else hword), dtype=np.float64).copy()
            shift = minimum if kind == 'nonzero_subnormal' else 1.0
            expected_word = (_SIGN if sign < 0 else 0) | (0 if kind == 'half_subnormal' else 1)
            expected = struct.unpack('>d', struct.pack('>Q', expected_word))[0]
        action = ExactAction(X, shift, h, old)
        theta, receipt = ExactCoordinateMM(X, shift).next_iterate(h, old[1:], float(old[0]))
        assert_receipt(action, theta, receipt)
        assert word(theta[1 if kind == 'coefficient' else 0]) == word(expected)
        if kind != 'coefficient':
            assert word(theta[1]) == 0
        records.append(receipt)
        if kind == 'nonzero_subnormal':
            # Mean of largest subnormal and minnormal is their exact midpoint;
            # the minnormal significand is even. Construct operands as words.
            sign_word = _SIGN if sign < 0 else 0
            hsign = _SIGN if sign > 0 else 0
            h = np.frombuffer(struct.pack('=QQ', hsign | ((1 << 52)-1),
                                         hsign | (1 << 52)), dtype=np.float64).copy()
            action = ExactAction(X, 1.0, h, old)
            theta, receipt = ExactCoordinateMM(X, 1.0).next_iterate(h, old[1:], float(old[0]))
            assert_receipt(action, theta, receipt)
            assert word(theta[0]) == sign_word | (1 << 52)
            records.append(receipt)


def test_nonfinite_rounding_refuses_instead_of_clipping():
    X = np.zeros((2, 1))
    maximum = struct.unpack('>d', struct.pack('>Q', _MAX))[0]
    for sign in (1.0, -1.0):
        old = np.array([sign*maximum, 0.0])
        theta, receipt = ExactCoordinateMM(X, 1.0).next_iterate(np.zeros(2), old[1:], float(old[0]))
        assert_receipt(ExactAction(X, 1.0, np.zeros(2), old), theta, receipt)
        assert word(theta[0]) == word(old[0])
        for h in (np.array([-sign*2.0**971, 0.0]), np.array([-sign*maximum, -sign*maximum])):
            with pytest.raises(FloatingPointError, match='finite binary64 rounding'):
                ExactCoordinateMM(X, 1.0).next_iterate(h, np.zeros(1), sign*maximum)


@pytest.mark.parametrize('cap', ['features', 'work', 'integer'])
def test_resource_caps_refuse_with_accounting(cap):
    X = np.zeros((2, 17 if cap == 'features' else 1))
    policy = ExactCoordinatePolicy(**({'max_exact_work': 0} if cap == 'work' else
                                     {'max_integer_bits': 1} if cap == 'integer' else {}))
    with pytest.raises(ExactCoordinateLimitError) as caught:
        system = ExactCoordinateMM(X, 1.0, policy=policy)
        system.next_iterate(np.ones(2), np.zeros(X.shape[1]), 0.0)
    assert caught.value.reason and caught.value.stage and type(caught.value.usage) is dict


def test_domain_and_owned_input_isolation():
    X = np.array([[-1.0], [1.0]])
    for shift in (0.0, -1.0, float('inf')):
        with pytest.raises(ValueError):
            ExactCoordinateMM(X, shift)
    for invalid in (np.array([1.0, 2.0]), np.array([[float('nan')]])):
        with pytest.raises(ValueError):
            ExactCoordinateMM(invalid, 1.0)
    original = X.copy()
    system = ExactCoordinateMM(X, 1.0)
    X[:] = 100.0
    h, old = np.array([0.25, -0.25]), np.zeros(2)
    theta, receipt = system.next_iterate(h, old[1:], 0.0)
    assert_receipt(ExactAction(original, 1.0, h, old), theta, receipt)
    with pytest.raises(ValueError):
        system.next_iterate(np.ones(1), old[1:], 0.0)


def test_cached_factor_and_cumulative_budget():
    X = np.array([[-1.0], [1.0]])
    h, old = np.array([0.25, -0.25]), np.zeros(2)
    system = ExactCoordinateMM(X, 1.0)
    records = []
    for index in range(2):
        theta, receipt = system.next_iterate(h, old[1:], float(old[0]))
        assert_receipt(ExactAction(X, 1.0, h, old), theta, receipt)
        assert receipt['actions_completed'] == index+1
        records.append(receipt)
        old = theta.copy()
    assert records[1]['budget_usage']['exact_work'] > records[0]['budget_usage']['exact_work']
    assert records[1]['factorizations_completed'] == records[0]['factorizations_completed'] == 1
