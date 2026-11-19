"""Bounded true-MM action on an exactly certified kernel.

K is unchanged and its entire certified function space is retained. The sparse
pivot representative changes the otherwise irrelevant nullspace gauge. It does
not satisfy, or claim to check, the stronger (K+delta I)/sum-alpha-zero gauge.
Every step has at most two candidate coefficients: a validated small spectral
solve, then the rounded exact reduced solution. The three acceptance gates are
exact rational comparisons on the actual exported float64 coefficients/offset.
Resource caps apply separately to preparation and EACH call, not wall time.
"""
from time import perf_counter
from numbers import Integral

import numpy as np

from ._exact_kernel_factor import ExactArithmetic, ExactKernelFactor, exact_inverse
from ._eigen import validated_eigh


class MMFunctionAccuracyError(FloatingPointError):
    def __init__(self, message, info):
        super().__init__(message)
        self.info = info


def _cap(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(name + ' must be a nonnegative integer.')
    return int(value)


def _scalar(value, name):
    if np.ndim(value) != 0 or np.iscomplexobj(value) or isinstance(value, (bool, np.bool_)):
        raise ValueError(name + ' must be a finite real scalar.')
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + ' must be a finite real scalar.') from exc
    if not np.isfinite(value):
        raise ValueError(name + ' must be a finite real scalar.')
    return value


def _abs(q, ar):
    return ar.sub(ar.integer(0), q) if ar.sign(q) < 0 else q


def _max(values, ar, floor=None):
    answer = ar.integer(0) if floor is None else floor
    for value in values:
        if ar.lt(answer, value):
            answer = value
    return answer


def _sum(values, ar):
    result = ar.integer(0)
    for value in values:
        result = ar.add(result, value)
    return result


def _float(q):
    value = float(q)
    if not np.isfinite(value):
        raise FloatingPointError('An exact result has no finite float64 readout.')
    return value


def _diagnostic(q):
    try:
        value = float(q)
        return value if np.isfinite(value) else 'outside_float64_range'
    except OverflowError:
        return 'outside_float64_range'


class CertifiedKernelMM:
    """Prepare one fixed stored K/delta action, without labels or hidden features.

    ``a`` in ``step(a, previous_offset)`` excludes the old intercept. Returned
    ``new_offset`` is absolute and already checked: never add the old offset
    again. The returned kernel scores are rounded exact K@alpha products.
    """
    def __init__(self, K, factor, delta, *, max_entries=65536, max_rank=16,
                 max_bits=4096, max_operations=1000000):
        started = perf_counter()
        self.max_entries = _cap(max_entries, 'max_entries')
        self.max_rank = _cap(max_rank, 'max_rank')
        self.max_bits = _cap(max_bits, 'max_bits')
        self.max_operations = _cap(max_operations, 'max_operations')
        if type(K) is not np.ndarray or K.ndim != 2 or not len(K) or K.size > self.max_entries:
            raise ValueError('A nonempty stored matrix inside the entry budget is required.')
        if not isinstance(factor, ExactKernelFactor) or factor.rank > self.max_rank:
            raise ValueError('An exact factor inside the rank budget is required.')
        if not factor.validate(K):
            raise ValueError('Exact factor does not match the current stored kernel.')
        self.delta = _scalar(delta, 'delta')
        if self.delta <= 0:
            raise ValueError('delta must be strictly positive.')
        self.K, self.factor = K, factor
        self.n, self.rank = factor.n, factor.rank
        self._factor_hash = factor.factor_sha256
        ar = self._arithmetic()
        self._nq, self._zero = ar.integer(self.n), ar.integer(0)
        self._deltaq = ar.from_float(self.delta)
        self._C = tuple(tuple(ar.from_float(x) for x in row) for row in factor.C)
        self._B = factor.B_exact
        # Validation of each existing rational is charged through arithmetic.
        self._B = tuple(tuple(ar.add(self._zero, x) for x in row) for row in self._B)
        self._mean = tuple(ar.div(_sum((row[j] for row in self._C), ar), self._nq)
                           for j in range(self.rank))
        self._T = tuple(tuple(ar.sub(row[j], self._mean[j]) for row in self._C)
                        for j in range(self.rank))
        self._M = tuple(tuple(ar.add(ar.dot(self._T[i], self._T[j]),
                                         ar.mul(self._deltaq, self._B[i][j]))
                              for j in range(self.rank)) for i in range(self.rank))
        self._inverse = exact_inverse(self._M, ar)
        self._native = None
        native = dict(attempted=bool(self.rank), available=False, kind='validated_small_full_eigenbasis')
        if self.rank:
            try:
                M = np.array([[_float(x) for x in row] for row in self._M])
                (values, vectors), detail = validated_eigh(M, return_info=True)
                with np.errstate(over='raise', divide='raise', invalid='raise'):
                    inverse_values = 1. / values
                if not (np.isfinite(values).all() and np.all(values > 0.) and
                        np.isfinite(inverse_values).all() and np.all(inverse_values > 0.)):
                    raise FloatingPointError('Small eigenvalues/inverses must be finite and strictly positive.')
                self._native = (vectors, inverse_values)
                native.update(available=True, validation=detail)
            except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
                native['failure'] = type(exc).__name__ + ': ' + str(exc)
        self.info = dict(status='prepared',
                         gauge='certified sparse pivot; true MM function, not reduced A/sum gauge',
                         n=self.n, rank=self.rank, pivots=list(factor.pivots),
                         K_sha256=factor.K_sha256, factor_sha256=self._factor_hash,
                         delta=self.delta, positive_directions_discarded=0,
                         original_kernel_unchanged=True, native_preparation=native,
                         preparation_arithmetic=ar.metadata(),
                         preparation_seconds=perf_counter()-started,
                         budget_scope='each preparation / each step; fixed maximum two candidates')
        self.steps = 0

    def _arithmetic(self):
        return ExactArithmetic(max_bits=self.max_bits, max_operations=self.max_operations)

    def _native_beta(self, rhs):
        vectors, weights = self._native
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            return vectors @ (weights * (vectors.T @ np.array([_float(x) for x in rhs])))

    def _quadratic(self, beta, ar):
        return ar.dot(beta, tuple(ar.dot(row, beta) for row in self._B))

    def _assess(self, beta, aq, oldb, ideal, ideal_scores, ideal_b, ar, name):
        if np.iscomplexobj(beta) or np.shape(beta) != (self.rank,) or not np.isfinite(beta).all():
            raise FloatingPointError('Candidate coefficients are not a finite real rank vector.')
        beta = np.asarray(beta, dtype=np.float64)
        bq = tuple(ar.from_float(x) for x in beta)
        scoresq = tuple(ar.dot(row, bq) for row in self._C)
        # Exactly re-center the actual rounded beta, then round the ABSOLUTE
        # intercept once. The same correction is used for both candidates.
        increment = ar.div(_sum((ar.sub(x, y) for x, y in zip(aq, scoresq)), ar), self._nq)
        offset = _float(ar.add(oldb, increment))
        offsetq = ar.from_float(offset)
        scores = np.array([_float(x) for x in scoresq], dtype=np.float64)
        with np.errstate(over='raise', invalid='raise'):
            decisions = scores + offset
        if not np.isfinite(decisions).all():
            raise FloatingPointError('Exported decisions are not finite.')
        actual = tuple(ar.add(x, offsetq) for x in scoresq)
        desired = tuple(ar.add(x, ideal_b) for x in ideal_scores)
        e = tuple(ar.sub(x, y) for x, y in zip(bq, ideal))
        qerror, qmodel = self._quadratic(e, ar), self._quadratic(bq, ar)
        if ar.sign(qerror) < 0 or ar.sign(qmodel) < 0:
            raise FloatingPointError('Certified positive quadratic unexpectedly became negative.')
        one = ar.integer(1)
        feature_tol, scalar_tol = ar.from_float(5e-7), ar.from_float(1e-10)
        qlimit = ar.mul(ar.mul(feature_tol, feature_tol), _max((qmodel,), ar, one))
        decision_scale = _max((_abs(x, ar) for x in actual), ar, one)
        decision_limit = ar.mul(feature_tol, decision_scale)
        model_error = _max((_abs(ar.sub(x, y), ar) for x, y in zip(actual, desired)), ar)
        # Also check the actual score-vector/offset floating readout that the
        # optimizer will reuse, not merely the exact represented function.
        readout_error = _max((_abs(ar.sub(ar.from_float(x), y), ar)
                             for x, y in zip(decisions, desired)), ar)
        offset_increment = ar.sub(offsetq, oldb)
        mean_residual = ar.div(_sum((ar.sub(ar.add(x, offset_increment), y)
                                    for x, y in zip(scoresq, aq)), ar), self._nq)
        scalar_limit = ar.mul(scalar_tol, _max((_abs(x, ar) for x in aq), ar, one))
        gates = dict(rkhs=not ar.lt(qlimit, qerror),
                     actual_model_decisions=not ar.lt(decision_limit, model_error),
                     exported_decision_readout=not ar.lt(decision_limit, readout_error),
                     free_offset_stationarity=not ar.lt(scalar_limit, _abs(mean_residual, ar)))
        result = dict(candidate=name, accepted=all(gates.values()), gates=gates,
                      rkhs_error_squared=_diagnostic(qerror), rkhs_error_squared_limit=_diagnostic(qlimit),
                      rkhs_norm_squared=_diagnostic(qmodel),
                      actual_model_decision_error=_diagnostic(model_error),
                      exported_decision_error=_diagnostic(readout_error),
                      decision_error_limit=_diagnostic(decision_limit),
                      free_offset_mean_residual=_diagnostic(mean_residual),
                      free_offset_residual_limit=_diagnostic(scalar_limit),
                      absolute_offset_error=_diagnostic(_abs(ar.sub(offsetq, ideal_b), ar)),
                      offset_policy='round exact old_offset + mean(a-C beta_float) once')
        alpha = np.zeros(self.n, dtype=np.float64)
        alpha[list(self.factor.pivots)] = beta
        return alpha, offset, scores, result

    def step(self, a, previous_offset):
        started = perf_counter()
        if np.iscomplexobj(a) or np.shape(a) != (self.n,):
            raise ValueError('a must be a real vector with one entry per kernel row.')
        a = np.asarray(a, dtype=np.float64)
        if not np.isfinite(a).all():
            raise ValueError('a must contain only finite values.')
        previous_offset = _scalar(previous_offset, 'previous_offset')
        if self.factor.factor_sha256 != self._factor_hash or not self.factor.validate(self.K):
            raise ValueError('Kernel or exact factor changed after preparation.')
        ar = self._arithmetic()
        aq = tuple(ar.from_float(x) for x in a)
        oldb = ar.from_float(previous_offset)
        rhs = tuple(ar.dot(row, aq) for row in self._T)
        ideal = tuple(ar.dot(row, rhs) for row in self._inverse)
        ideal_scores = tuple(ar.dot(row, ideal) for row in self._C)
        ideal_increment = ar.div(_sum((ar.sub(x, y) for x, y in zip(aq, ideal_scores)), ar), self._nq)
        ideal_b = ar.add(oldb, ideal_increment)
        attempts = []
        names = ('native_small_spectral', 'rounded_exact') if self._native is not None else ('rounded_exact',)
        for name in names:
            try:
                beta = self._native_beta(rhs) if name == 'native_small_spectral' else np.array([_float(x) for x in ideal])
                alpha, offset, scores, detail = self._assess(beta, aq, oldb, ideal, ideal_scores, ideal_b, ar, name)
                attempts.append(detail)
                if detail['accepted']:
                    self.steps += 1
                    info = dict(status='accepted', accepted_candidate=name, attempts=attempts,
                                absolute_offset_checked=True, exact_Kalpha_scores_rounded=True,
                                reduced_A_gauge_checked=False, true_MM_function_checked=True,
                                rkhs_norm_squared=detail['rkhs_norm_squared'],
                                arithmetic=ar.metadata(), elapsed_seconds=perf_counter()-started)
                    return alpha, offset, scores, info
            except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
                attempts.append(dict(candidate=name, accepted=False, failure=type(exc).__name__ + ': ' + str(exc)))
        raise MMFunctionAccuracyError('Neither bounded candidate meets the true MM function accuracy gates.',
                                      dict(attempts=attempts, arithmetic=ar.metadata(),
                                           elapsed_seconds=perf_counter()-started))
