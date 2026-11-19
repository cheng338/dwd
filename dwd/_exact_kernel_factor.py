"""Resource-bounded, exact stored-float64 PSD/rank factor.

Certification means an entrywise rational identity K=L D L.T, positive D,
unit-lower-triangular selected rows, and an explicitly verified inverse of
B=K[P,P]. Hence K=C B_inverse C.T for C=K[:,P]. There is no numerical rank
threshold and no use of a generating feature matrix. Incomplete work returns
no factor. Work/bit caps bound exact arithmetic, not wall-clock seconds.
"""
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from numbers import Integral
import struct
from time import perf_counter
from types import MappingProxyType

import numpy as np


class BudgetExhausted(Exception):
    """Exact computation could not finish inside an explicit resource cap."""


class CertificateFailure(Exception):
    """A complete exact identity or structural check failed."""


def _integer_budget(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer.')
    return int(value)


class ExactArithmetic:
    """Count every exact primitive/check and preflight unreduced intermediates.

    One operation means a rational construction, arithmetic operation, sign,
    equality or order check. Every arithmetic result is checked as well. Before
    Fraction executes, conservative bit bounds cover its unreduced products,
    denominator and numerator sums, including intermediate integers. Thus an
    operation whose cancellation *might* make it small can still be refused.
    This conservatism never turns incomplete arithmetic into a certificate.
    """
    def __init__(self, *, max_bits=4096, max_operations=1000000):
        self.max_bits = _integer_budget(max_bits, 'max_bits')
        self.max_operations = _integer_budget(max_operations, 'max_operations')
        self.operations = 0
        self.maximum_rational_bits = 0
        self.maximum_preflight_bits = 0
        self.operation_counts = {}

    def _tick(self, name):
        if self.operations >= self.max_operations:
            raise BudgetExhausted('operation_budget_exhausted')
        self.operations += 1
        self.operation_counts[name] = self.operation_counts.get(name, 0) + 1

    def _preflight(self, *bounds):
        largest = max(bounds, default=0)
        self.maximum_preflight_bits = max(self.maximum_preflight_bits, largest)
        if largest > self.max_bits:
            raise BudgetExhausted('rational_bit_budget_exhausted')

    def _checked(self, value):
        if type(value) is not Fraction:
            raise TypeError('Exact arithmetic requires Fraction operands.')
        bits = max(value.numerator.bit_length(), value.denominator.bit_length())
        self.maximum_rational_bits = max(self.maximum_rational_bits, bits)
        if bits > self.max_bits:
            raise BudgetExhausted('rational_bit_budget_exhausted')
        return value

    def integer(self, value):
        self._tick('integer')
        if isinstance(value, bool):
            value = int(value)
        if not isinstance(value, Integral):
            raise TypeError('Exact integer construction requires an integer.')
        value = int(value)
        self._preflight(max(value.bit_length(), 1))
        return self._checked(Fraction(value))

    def from_float(self, value):
        self._tick('from_float')
        value = float(value)
        bits = struct.unpack('<Q', struct.pack('<d', value))[0]
        exponent, mantissa = (bits >> 52) & 2047, bits & ((1 << 52)-1)
        if exponent == 2047:
            raise ValueError('An exact float must be finite.')
        if exponent:
            mantissa |= 1 << 52
            exponent = int(exponent)-1023-52
        else:
            exponent = -1074
        if not mantissa:
            self._preflight(1)
            return self._checked(Fraction(0))
        # Fixed-width IEEE decoding precedes any potentially large rational
        # shift. The normalized numerator/denominator sizes are known first.
        trailing = (mantissa & -mantissa).bit_length()-1
        mantissa >>= trailing
        exponent += trailing
        self._preflight(mantissa.bit_length()+max(exponent, 0), 1+max(-exponent, 0))
        numerator = mantissa << max(exponent, 0)
        if bits >> 63:
            numerator = -numerator
        denominator = 1 << max(-exponent, 0)
        return self._checked(Fraction(numerator, denominator))

    def _operands(self, a, b):
        return self._checked(a), self._checked(b)

    def add(self, a, b):
        self._tick('add')
        a, b = self._operands(a, b)
        if not a.numerator:
            return b
        if not b.numerator:
            return a
        first = a.numerator.bit_length()+b.denominator.bit_length()
        second = b.numerator.bit_length()+a.denominator.bit_length()
        self._preflight(first, second, max(first, second)+1,
                        a.denominator.bit_length()+b.denominator.bit_length())
        return self._checked(a+b)

    def sub(self, a, b):
        self._tick('sub')
        a, b = self._operands(a, b)
        if not b.numerator:
            return a
        if not a.numerator:
            return self._checked(-b)
        first = a.numerator.bit_length()+b.denominator.bit_length()
        second = b.numerator.bit_length()+a.denominator.bit_length()
        self._preflight(first, second, max(first, second)+1,
                        a.denominator.bit_length()+b.denominator.bit_length())
        return self._checked(a-b)

    def mul(self, a, b):
        self._tick('mul')
        a, b = self._operands(a, b)
        if not a.numerator or not b.numerator:
            self._preflight(1)
            return self._checked(Fraction(0))
        self._preflight(a.numerator.bit_length()+b.numerator.bit_length(),
                        a.denominator.bit_length()+b.denominator.bit_length())
        return self._checked(a*b)

    def div(self, a, b):
        self._tick('div')
        a, b = self._operands(a, b)
        if not b.numerator:
            raise CertificateFailure('zero_division_in_exact_inverse')
        if not a.numerator:
            self._preflight(1)
            return self._checked(Fraction(0))
        self._preflight(a.numerator.bit_length()+b.denominator.bit_length(),
                        a.denominator.bit_length()+b.numerator.bit_length())
        return self._checked(a/b)

    def sign(self, a):
        self._tick('sign')
        a = self._checked(a)
        return (a.numerator > 0)-(a.numerator < 0)

    def eq(self, a, b):
        self._tick('eq')
        a, b = self._operands(a, b)
        # Canonical fractions compare without cross-products.
        return a.numerator == b.numerator and a.denominator == b.denominator

    def lt(self, a, b):
        self._tick('lt')
        a, b = self._operands(a, b)
        self._preflight(a.numerator.bit_length()+b.denominator.bit_length(),
                        b.numerator.bit_length()+a.denominator.bit_length())
        return a < b

    def dot(self, a, b):
        if len(a) != len(b):
            raise ValueError('Exact dot operands must have equal length.')
        result = self.integer(0)
        for first, second in zip(a, b):
            result = self.add(result, self.mul(first, second))
        return result

    def metadata(self):
        return dict(max_bits=self.max_bits, max_operations=self.max_operations,
                    operations=self.operations, maximum_rational_bits=self.maximum_rational_bits,
                    maximum_preflight_bits=self.maximum_preflight_bits,
                    operation_counts=dict(self.operation_counts))


def _input_status(K, max_entries):
    # No np.asarray, np.isfinite full-matrix temporary, or rational allocation
    # occurs before the input-entry cap. Require a concrete stored matrix.
    if type(K) is not np.ndarray or K.ndim != 2 or K.shape[0] != K.shape[1]:
        return 'invalid_matrix_shape_or_type'
    if K.size > max_entries:
        return 'entry_budget_exhausted'
    if K.dtype != np.dtype(np.float64):
        return 'requires_native_float64'
    for i in range(len(K)):
        if not np.isfinite(K[i]).all():
            return 'nonfinite_matrix'
        if not np.array_equal(K[i], K[:, i]):
            return 'not_exactly_symmetric'
    return None


def _matrix_hash(K):
    h = sha256()
    for row in K:
        # At most one row is copied for noncontiguous input.
        h.update(np.ascontiguousarray(row).tobytes())
    return h.hexdigest()


def _factor_hash(n, pivots, C, B, inverse):
    h = sha256()
    h.update(str(n).encode('ascii'))
    h.update(repr(pivots).encode('ascii'))
    h.update(_matrix_hash(C).encode('ascii'))
    for matrix in (B, inverse):
        for row in matrix:
            for value in row:
                for integer in (value.numerator, value.denominator):
                    # Hash bounded existing integers without decimal-string limits.
                    data = abs(integer).to_bytes(max(1, (integer.bit_length()+7)//8), 'little')
                    h.update(b'-' if integer < 0 else b'+')
                    h.update(len(data).to_bytes(8, 'little'))
                    h.update(data)
    return h.hexdigest()


@dataclass(frozen=True)
class ExactKernelFactor:
    n: int
    rank: int
    pivots: tuple
    C: np.ndarray
    B_exact: tuple
    B_inverse_exact: tuple
    K_sha256: str
    factor_sha256: str
    metadata: object

    def validate(self, K):
        """Check exact input identity and mutation of the returned factor data."""
        if _input_status(K, self.n*self.n) is not None or K.shape != (self.n, self.n):
            return False
        if _matrix_hash(K) != self.K_sha256:
            return False
        if self.C.shape != (self.n, self.rank) or self.C.dtype != np.dtype(np.float64):
            return False
        return self.factor_sha256 == _factor_hash(self.n, self.pivots, self.C,
                                                 self.B_exact, self.B_inverse_exact)


@dataclass(frozen=True)
class FactorResult:
    status: str
    factor: object
    metadata: object

    @property
    def certified(self):
        return self.status == 'certified_exact_psd_rank' and self.factor is not None


def _verify_ldlt(original, columns, diagonals, pivots, arithmetic):
    zero, one = arithmetic.integer(0), arithmetic.integer(1)
    for diagonal in diagonals:
        if arithmetic.sign(diagonal) <= 0:
            return False
    for i in range(len(pivots)):
        for j in range(i, len(pivots)):
            if not arithmetic.eq(columns[j][pivots[i]], one if i == j else zero):
                return False
    for i in range(len(original)):
        for j in range(len(original)):
            value = zero
            for k in range(len(diagonals)):
                product = arithmetic.mul(columns[k][i], diagonals[k])
                product = arithmetic.mul(product, columns[k][j])
                value = arithmetic.add(value, product)
            if not arithmetic.eq(value, original[i][j]):
                return False
    return True


def exact_inverse(matrix, arithmetic):
    """Gauss-Jordan inverse and complete product check, using the same budget."""
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError('Exact inverse requires a square matrix.')
    zero, one = arithmetic.integer(0), arithmetic.integer(1)
    rows = [list(row)+[one if i == j else zero for j in range(n)] for i, row in enumerate(matrix)]
    for j in range(n):
        pivot = next((i for i in range(j, n) if arithmetic.sign(rows[i][j]) != 0), None)
        if pivot is None:
            raise CertificateFailure('singular_selected_block')
        rows[j], rows[pivot] = rows[pivot], rows[j]
        value = rows[j][j]
        rows[j] = [arithmetic.div(x, value) for x in rows[j]]
        for i in range(n):
            if i == j:
                continue
            value = rows[i][j]
            for k in range(2*n):
                product = arithmetic.mul(value, rows[j][k])
                rows[i][k] = arithmetic.sub(rows[i][k], product)
    inverse = tuple(tuple(row[n:]) for row in rows)
    for i in range(n):
        for j in range(n):
            value = arithmetic.dot(matrix[i], tuple(inverse[k][j] for k in range(n)))
            if not arithmetic.eq(value, one if i == j else zero):
                raise CertificateFailure('exact_inverse_identity_failed')
    return inverse


def _compute_factor(K, max_rank, arithmetic, metadata):
    n = len(K)
    original = [[arithmetic.from_float(value) for value in row] for row in K]
    schur = [row.copy() for row in original]
    zero = arithmetic.integer(0)
    active, pivots, diagonals, columns = list(range(n)), [], [], []
    while active:
        p = active[0]
        for i in active:
            if arithmetic.sign(schur[i][i]) < 0:
                metadata.update(witness_index=i, partial_pivot_count=len(pivots))
                raise CertificateFailure('exact_negative_schur_diagonal')
            if arithmetic.lt(schur[p][p], schur[i][i]):
                p = i
        pivot = schur[p][p]
        if arithmetic.sign(pivot) == 0:
            for i in active:
                for j in active:
                    if arithmetic.sign(schur[i][j]) != 0:
                        metadata.update(witness_indices=(i, j), partial_pivot_count=len(pivots))
                        raise CertificateFailure('zero_diagonal_nonzero_schur')
            break
        if len(pivots) >= max_rank:
            metadata['partial_pivot_count'] = len(pivots)
            raise BudgetExhausted('rank_budget_exhausted')
        remaining = [i for i in active if i != p]
        column = [zero]*n
        for i in active:
            column[i] = arithmetic.div(schur[i][p], pivot)
        for position, i in enumerate(remaining):
            for j in remaining[position:]:
                product = arithmetic.mul(pivot, column[i])
                product = arithmetic.mul(product, column[j])
                schur[i][j] = schur[j][i] = arithmetic.sub(schur[i][j], product)
        for i in active:
            schur[i][p] = schur[p][i] = zero
        pivots.append(p)
        diagonals.append(pivot)
        columns.append(column)
        active = remaining
    metadata['arithmetic_after_schur'] = arithmetic.metadata()
    if not _verify_ldlt(original, columns, diagonals, pivots, arithmetic):
        raise CertificateFailure('exact_ldlt_or_triangular_identity_failed')
    metadata['arithmetic_after_ldlt_identity'] = arithmetic.metadata()
    B = tuple(tuple(original[i][j] for j in pivots) for i in pivots)
    inverse = exact_inverse(B, arithmetic)
    metadata['arithmetic_after_inverse_identity'] = arithmetic.metadata()
    rank = len(pivots)
    C = K[:, pivots].copy()
    C.setflags(write=False)
    pivots = tuple(pivots)
    metadata.update(rank=rank, complete_ldlt_identity=True, positive_pivots=True,
                    selected_L_unit_lower_triangular=True, exact_inverse_identity=True,
                    positive_directions_discarded=0, kernel_changed=False,
                    identity='Exact stored K = C @ B_inverse_exact @ C.T; derived from complete LDL identity.')
    return ExactKernelFactor(n, rank, pivots, C, B, inverse, _matrix_hash(K),
                             _factor_hash(n, pivots, C, B, inverse), MappingProxyType(dict(metadata)))


def exact_kernel_factor(K, *, max_entries=65536, max_rank=16,
                        max_bits=4096, max_operations=1000000):
    """Return a complete exact factor or an inapplicable result with factor=None.

    Input must be an existing, exactly symmetric, finite native-float64 ndarray.
    Nonnegative integer budgets are validated before inspecting matrix values.
    Max rank limits work; it never authorizes dropping a positive direction.
    A negative Schur witness means no exact PSD certificate for these bytes;
    this private result does not change any public kernel-validation policy.
    """
    started = perf_counter()
    max_entries = _integer_budget(max_entries, 'max_entries')
    max_rank = _integer_budget(max_rank, 'max_rank')
    arithmetic = ExactArithmetic(max_bits=max_bits, max_operations=max_operations)
    metadata = dict(max_entries=max_entries, max_rank=max_rank,
                    kernel_changed=False, positive_directions_discarded=0)
    status = _input_status(K, max_entries)
    factor = None
    if status is None:
        metadata['K_sha256'] = _matrix_hash(K)
        metadata['shape'] = tuple(K.shape)
        try:
            factor = _compute_factor(K, max_rank, arithmetic, metadata)
            status = 'certified_exact_psd_rank'
        except (BudgetExhausted, CertificateFailure) as exc:
            status = str(exc)
            # Do not return exceptions/tracebacks retaining partial rational
            # matrices. Only a completed factor can escape this scope.
            exc.__traceback__ = None
            factor = None
    metadata.update(arithmetic=arithmetic.metadata(), seconds=perf_counter()-started,
                    certified=status == 'certified_exact_psd_rank')
    return FactorResult(status, factor, MappingProxyType(metadata))
