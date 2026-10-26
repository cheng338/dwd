"""Numerical controls for testing solver contracts independently of LAPACK."""
from decimal import Decimal, localcontext
import json
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_info

import dwd._eigen as eigen


def _array_identity(value):
    return (value.shape, value.dtype.str, value.strides,
            bool(value.flags.c_contiguous), bool(value.flags.f_contiguous),
            value.tobytes(order='C'))


def _copy_result(value):
    if isinstance(value, tuple):
        return tuple(_copy_result(item) for item in value)
    result = value.copy(order='K')
    if _array_identity(result) != _array_identity(value):
        raise AssertionError('Native result layout cannot be replayed exactly')
    return result


class AcceptedNativeReplay:
    """Reuse accepted raw eigensystems for identical provider calls.

    Patch the private provider, not ``validated_eigh`` or the solver. Every
    replay still passes through the ordinary package validators. Rejected
    provider results are not cached, preserving normal fallback behavior.
    Public ``K_eig`` stays absent, so automatic recovery remains eligible.
    """

    def __init__(self):
        self.native = eigen.eigh
        self.cache = {}
        self.native_calls = 0
        self.replays = 0

    def __enter__(self):
        # A runtime can load its OpenMP library on its first eigensolve. Warm
        # a separate tiny matrix before fixing complete runtime identities;
        # a later identity change must still prevent reuse.
        eigen.validated_eigh(np.diag([1., 2.]), eigvals_only=True,
                             drivers=('evd',))
        self.patch = patch.object(eigen, 'eigh', self.call)
        self.patch.__enter__()
        return self

    def __exit__(self, *exc):
        return self.patch.__exit__(*exc)

    def call(self, matrix, *args, **kwargs):
        if args:
            raise AssertionError('Unexpected positional native options')
        pools = sorted(threadpool_info(),
                       key=lambda item: json.dumps(item, sort_keys=True))
        key = (_array_identity(matrix), json.dumps(kwargs, sort_keys=True),
               json.dumps(pools, sort_keys=True))
        if key in self.cache:
            self.replays += 1
            return _copy_result(self.cache[key])
        self.native_calls += 1
        before = _array_identity(matrix)
        result = self.native(matrix, **kwargs)
        if _array_identity(matrix) != before:
            raise AssertionError('Native eigensolver changed its input')
        checked, statistics = eigen._matrix_statistics(matrix)
        if kwargs.get('eigvals_only', False):
            issues, _ = eigen._validate_values(result, statistics)
        else:
            issues, _ = eigen._validate_pairs(
                checked, result[0], result[1], statistics)
        if not issues:
            self.cache[key] = _copy_result(result)
        return result


def objective_oracle(K, y, alpha, offset, penalty):
    """Evaluate the q=1 objective at represented inputs with 80-digit arithmetic.

    This checks objective evaluation, not convergence or optimality.
    """
    D = Decimal.from_float
    with localcontext() as context:
        context.prec = 80
        coefficients = [D(float(value)) for value in alpha]
        products = [sum((D(float(K[i, j])) * coefficients[j]
                         for j in range(len(y))), Decimal(0))
                    for i in range(len(y))]
        margins = [D(float(y[i])) * (products[i] + D(float(offset)))
                   for i in range(len(y))]
        losses = [1 - value if value <= Decimal('.5') else 1 / (4 * value)
                  for value in margins]
        return float(sum(losses) / len(y) + D(float(penalty)) * sum(
            (a * product for a, product in zip(coefficients, products)),
            Decimal(0)))
