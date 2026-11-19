"""Small independent oracles and isolated imports for feature-report tests."""
from contextlib import contextmanager
from fractions import Fraction
import importlib.util
from pathlib import Path
import struct
import sys
import types

import numpy as np
import pytest

import dwd
from dwd import _compiled_residual as native
from dwd import _feature_report_optimization as support
from dwd import _linear_feature_diagnostics as feature


def fraction(value):
    return Fraction.from_float(float(value))


def exact_dot(left, right):
    return sum((fraction(x) * fraction(y) for x, y in zip(left, right)), Fraction())


def same(actual, expected):
    """Bitwise arrays/floats, including signed zeros; recursive report values."""
    if isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert actual.tobytes(order='C') == expected.tobytes(order='C')
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            same(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected):
            same(left, right)
    elif isinstance(expected, (float, np.floating)):
        assert struct.pack('!d', actual) == struct.pack('!d', expected)
    else:
        assert actual == expected


def pins(arrays):
    return [(a.dtype.str, a.shape, a.strides, a.flags.writeable,
             a.tobytes(order='C')) for a in arrays]


def separate(matrix, beta, offset):
    raw = feature.compensated_kernel_matvec(matrix, beta)
    joint = feature.affine_scores(matrix, beta, offset, policy='compensated')
    return raw, joint


def contraction_data():
    features = np.arange(4 * 128, dtype=float).reshape(4, 128) / 1024 + .125
    alpha = np.array([.5, -.25, .125, -.0625])
    return features, alpha, -.5 * alpha


def contract(family, features, alpha, z):
    if family == 'coefficients':
        return feature._feature_coefficients(features, alpha)
    return feature._feature_gradient(features, z, alpha, .5)


def exact_contract(family, features, alpha, z):
    # The test data are dyadic and representable; no package arithmetic is used.
    values = alpha if family == 'coefficients' else [
        fraction(zz) + Fraction(1, 2) * fraction(aa) for zz, aa in zip(z, alpha)]
    return np.array([float(sum((fraction(x) * (v if isinstance(v, Fraction)
                                             else fraction(v))
                               for x, v in zip(column, values)), Fraction()))
                     for column in features.T])


@pytest.fixture
def portable_scores(monkeypatch):
    """Exercise the supported extension-absent path even on native hosts."""
    monkeypatch.setattr(native, '_ACCEL', None)


@pytest.fixture
def numpy_policy():
    geterr, geterrcall = np.geterr, np.geterrcall
    policy, callback = geterr().copy(), geterrcall()
    yield
    assert geterr() == policy and geterrcall() is callback


@contextmanager
def fresh_support(monkeypatch):
    """Load the installed support after a hook; restore both module registries.

    This tests pre-import replacements without reloading production modules or
    carrying another implementation/baseline in the test distribution.
    """
    name = 'dwd._feature_report_test_fresh_support'
    leaf = name.rsplit('.', 1)[1]
    with monkeypatch.context() as scoped:
        placeholder = types.ModuleType(name)
        scoped.setitem(sys.modules, name, placeholder)
        scoped.setattr(dwd, leaf, placeholder, raising=False)
        spec = importlib.util.spec_from_file_location(name, Path(support.__file__))
        module = importlib.util.module_from_spec(spec)
        scoped.setitem(sys.modules, name, module)
        scoped.setattr(dwd, leaf, module)
        spec.loader.exec_module(module)
        scoped.setattr(feature, '_report_optimization', module)
        yield module
