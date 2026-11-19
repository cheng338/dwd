"""Caller-policy oracle for underflow that occurs only after distance repair."""
import warnings

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from dwd.gen_kern_dwd import KernGDWD


DEFAULT = dict(divide="warn", over="warn", invalid="warn", under="ignore")


@pytest.fixture(autouse=True)
def restore_error_policy():
    policy, handler = np.geterr(), np.geterrcall()
    np.seterr(**DEFAULT)
    np.seterrcall(None)
    try:
        yield
    finally:
        np.seterr(**policy)
        np.seterrcall(handler)


class Sink:
    def __init__(self):
        self.events = []

    def __call__(self, description, flag):
        self.events.append((description, flag))

    def write(self, description):
        self.events.append(description)


def construct(X):
    estimator = KernGDWD(kernel="rbf", kernel_kws={"gamma": 1000.})
    estimator._Xfit = X.copy()
    return estimator, estimator._compute_training_kernel(estimator._Xfit)


@pytest.mark.parametrize("kind", ("dense", "csr"))
@pytest.mark.parametrize("under", ("warn", "raise", "call", "log"))
def test_underflow_from_repaired_distance_honors_caller_policy(kind, under):
    # Both coordinates and their difference 1 are represented exactly. The
    # norm/dot formula loses that difference at this offset; repaired exp(-1000)
    # rounds to zero in binary64 and must obey the requested underflow policy.
    X = np.array([[float(2**40)], [float(2**40 + 1)]])
    assert X[1, 0] - X[0, 0] == 1.
    if kind == "csr":
        X = csr_matrix(X)
    sink = Sink()
    np.seterrcall(sink)
    policy = dict(DEFAULT, under=under)
    with np.errstate(**policy), warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        if under == "raise":
            with pytest.raises(FloatingPointError, match="underflow"):
                construct(X)
            result = None
        else:
            estimator, result = construct(X)
            assert estimator.kernel_computation_ == "guarded_norm_sum_rel_v2"
        assert np.geterr() == policy and np.geterrcall() is sink
    if result is not None:
        np.testing.assert_array_equal(result, np.eye(2))
    observed_warning = any(issubclass(item.category, RuntimeWarning)
                           and "underflow" in str(item.message).lower() for item in seen)
    observed_handler = any("underflow" in str(event).lower() for event in sink.events)
    assert observed_warning is (under == "warn")
    assert observed_handler is (under in ("call", "log"))
