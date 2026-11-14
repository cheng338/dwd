"""Scale-relative numerical PSD admission regression tests.

The policy tolerates relative roundoff; it is not an exact PSD certificate.
The original admission function is retained inline for ordinary-path comparisons.
"""
from fractions import Fraction
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from dwd import _kernel_solver as kernel
from dwd import _psd_policy as policy
from dwd import _spectral_linear_system as spectral
from dwd._eigen import validated_eigh
from dwd._psd_policy import spectral_psd_status
from dwd.gen_kern_dwd import KernGDWD, KernGDWDCV


EPS = float(np.finfo(float).eps)
MIN_SUBNORMAL = math.ldexp(1., -1074)


def original_check_spectrum(values, n, shift):
    """Unchanged pre-IR28 function body; no alternate installed package."""
    if np.shape(values) != (n,) or not np.isfinite(values).all():
        raise FloatingPointError('Invalid kernel eigenvalues from spectral preparation.')
    limit = 100 * np.finfo(float).eps * n * max(1., float(np.max(np.abs(values))))
    if values.min() < -limit:
        raise ValueError('K must be positive semidefinite. Recompute a rounded '
                         'float32 Gram matrix from float64 features if necessary.')
    return float(limit)


def exact_relative_oracle(values):
    """Independent exact arithmetic on the supplied binary64 eigenvalues.

    Selected boundary fixtures have an exactly representable normalized ratio.
    This oracle does not claim every rounded division is an exact comparison.
    """
    numbers = [Fraction.from_float(float(x)) for x in values]
    scale = max(map(abs, numbers))
    return scale == 0 or min(numbers) >= -(100 * len(numbers) * Fraction(1, 2**52)) * scale


def quadratic_fraction(K, vector):
    return sum((Fraction.from_float(float(vector[i]))
                * Fraction.from_float(float(K[i, j]))
                * Fraction.from_float(float(vector[j]))
                for i in range(len(vector)) for j in range(len(vector))), Fraction())


def recovery_system(K, shift=.2):
    # _build_original needs only these state fields.  Avoid another inverse
    # preparation when testing the fresh-spectrum admission boundary itself.
    system = object.__new__(spectral.SpectralLinearSystem)
    system.K, system.n, system.shift = K, len(K), shift
    return system


@pytest.mark.parametrize('exponent', [-1000, -80, 0, 500, 1000])
@pytest.mark.parametrize('kind', ['psd', 'zero_rank', 'relative_roundoff', 'mixed_material', 'negative_only'])
def test_scale_grid_matches_independent_exact_oracle(exponent, kind):
    raw = {
        'psd': [1., 2., 3., 4.],
        'zero_rank': [0., 0., 1., 2.],
        'relative_roundoff': [-EPS, 1., 2., 3.],
        'mixed_material': [-1., 1., 0., 0.],
        'negative_only': [-1., -2., -3., -4.],
    }[kind]
    values = np.array([math.ldexp(x, exponent) for x in raw])
    before = values.tobytes()
    with np.errstate(all='raise'):
        accepted, tolerance = spectral_psd_status(values, len(values))
    assert type(accepted) is bool and type(tolerance) is float
    assert accepted is exact_relative_oracle(values)
    assert accepted is (kind in ('psd', 'zero_rank', 'relative_roundoff'))
    assert math.isfinite(tolerance) and tolerance >= 0
    assert values.tobytes() == before


@pytest.mark.parametrize('negative_zero', [False, True])
def test_zero_spectrum_has_exact_zero_tolerance(negative_zero):
    values = np.full(4, -0. if negative_zero else 0.)
    before = values.tobytes()
    with np.errstate(all='raise'):
        assert spectral_psd_status(values, 4) == (True, 0.)
    assert values.tobytes() == before


@pytest.mark.parametrize('n', [2, 20])
@pytest.mark.parametrize('exponent', [-900, 0, 900])
@pytest.mark.parametrize('neighbor', ['inside', 'boundary', 'outside'])
def test_exact_power_of_two_boundary_and_adjacent_binary64(n, exponent, neighbor):
    relative = float(100 * n) * EPS
    magnitude = {'inside': math.nextafter(relative, 0.),
                 'boundary': relative,
                 'outside': math.nextafter(relative, math.inf)}[neighbor]
    values = np.zeros(n)
    values[0] = -math.ldexp(magnitude, exponent)
    values[-1] = math.ldexp(1., exponent)
    with np.errstate(all='raise'):
        accepted, _ = spectral_psd_status(values, n)
    assert accepted is exact_relative_oracle(values)
    assert accepted is (neighbor != 'outside')


@pytest.mark.parametrize('case,negative,scale,expected', [
    ('all_tiny', -MIN_SUBNORMAL, MIN_SUBNORMAL, False),
    ('rounded_product_boundary', -2 * MIN_SUBNORMAL, math.ldexp(1., -1030), False),
    ('tiny_relative_roundoff', -MIN_SUBNORMAL, math.ldexp(1., -1030), True),
    ('tiny_positive', MIN_SUBNORMAL, MIN_SUBNORMAL, True),
])
def test_subnormal_decision_survives_caller_allraise(case, negative, scale, expected):
    values = np.array([negative, scale, 0., 0.])
    with np.errstate(all='raise'):
        accepted, tolerance = spectral_psd_status(values, 4)
    assert accepted is expected is exact_relative_oracle(values)
    assert tolerance == float(Fraction(400, 2**52) * Fraction.from_float(scale))
    if case == 'rounded_product_boundary':
        # Removing max(1, scale) but comparing a rounded product still admits
        # this outside-boundary value.  C's normalized policy rejects it.
        assert tolerance == 2 * MIN_SUBNORMAL
        assert negative >= -tolerance


@pytest.mark.parametrize('scale', [MIN_SUBNORMAL, math.ldexp(1., -1030)])
def test_naive_numpy_product_raises_but_scalar_normalization_does_not(scale):
    values = np.array([-MIN_SUBNORMAL, scale, 0., 0.])
    with np.errstate(all='raise'):
        with pytest.raises(FloatingPointError, match='underflow'):
            np.float64(400 * EPS) * np.float64(scale)
        accepted, _ = spectral_psd_status(values, 4)
    assert accepted is exact_relative_oracle(values)


@pytest.mark.parametrize('minimum,expected', [
    (-MIN_SUBNORMAL, True), (float(np.finfo(float).max), True),
    (-float(np.finfo(float).max), False),
])
def test_largest_finite_scale_and_underflowing_normalized_ratio(minimum, expected):
    values = np.array([minimum, float(np.finfo(float).max)])
    with np.errstate(all='raise'):
        accepted, tolerance = spectral_psd_status(values, 2)
    assert accepted is expected is exact_relative_oracle(values)
    assert math.isfinite(tolerance)


@pytest.mark.parametrize('case', ['ordinary', 'normal_scale_subnormal_threshold', 'subnormal', 'zero'])
def test_existing_underflow_guard_is_only_reached_for_tiny_reported_threshold(case, monkeypatch):
    tiny = float(np.finfo(float).tiny)
    values = {'ordinary': np.array([-EPS, 1.]),
              'normal_scale_subnormal_threshold': np.array([-tiny / 2, tiny]),
              'subnormal': np.array([-MIN_SUBNORMAL, MIN_SUBNORMAL]),
              'zero': np.zeros(2)}[case]
    calls = []
    original = policy._gradual_underflow
    def observe():
        calls.append(True)
        return original()
    monkeypatch.setattr(policy, '_gradual_underflow', observe)
    with np.errstate(all='raise'):
        accepted, _ = spectral_psd_status(values, 2)
    assert len(calls) == (0 if case == 'ordinary' else 1)
    assert accepted is exact_relative_oracle(values)


@pytest.mark.parametrize('scale', [float(np.finfo(float).tiny), MIN_SUBNORMAL, 0.])
def test_unsupported_underflow_mode_error_propagates_without_changing_cpu_mode(scale, monkeypatch):
    failure = FloatingPointError('test-only unavailable gradual underflow')
    def unavailable():
        raise failure
    monkeypatch.setattr(policy, '_gradual_underflow', unavailable)
    with pytest.raises(FloatingPointError) as caught:
        spectral_psd_status(np.array([0., scale]), 2)
    assert caught.value is failure


@pytest.mark.parametrize('bad', [np.array([1.]), np.array([np.nan, 1.]), np.array([np.inf, 1.])])
@pytest.mark.parametrize('caller', ['solver', 'fresh_recovery'])
def test_unchanged_caller_shape_and_finite_guards(bad, caller, monkeypatch):
    if caller == 'solver':
        with pytest.raises(FloatingPointError, match='Invalid kernel eigenvalues'):
            kernel._check_spectrum(bad, 2, .2)
    else:
        monkeypatch.setattr(spectral, 'validated_eigh', lambda *a, **kw: ((bad, np.eye(2)), {}))
        with pytest.raises(FloatingPointError, match='Invalid fresh original-kernel eigenvalues'):
            recovery_system(np.eye(2))._build_original('evd')


@pytest.mark.parametrize('path', ['full', 'supplied', 'eigenvalues_only'])
def test_tiny_material_indefiniteness_rejected_before_final_reporting(path, monkeypatch):
    scale = math.ldexp(1., -80)
    K = np.diag([-scale, scale])
    y = np.array([-1., 1.])
    before = K.tobytes()
    assert quadratic_fraction(K, y) == 0  # No19 contradiction is not a substitute.
    assert original_check_spectrum(K.diagonal(), 2, 1.) > scale
    seen = []
    original = kernel.validated_eigh
    def observe(*args, **kwargs):
        seen.append(dict(kwargs))
        return original(*args, **kwargs)
    monkeypatch.setattr(kernel, 'validated_eigh', observe)
    monkeypatch.setattr(kernel, 'optimality_diagnostics', lambda *a, **kw: pytest.fail('Final reporting reached'))
    kwargs = {'backend': 'cholesky' if path == 'eigenvalues_only' else 'spectral'}
    if path == 'supplied':
        kwargs['K_eig'] = (np.eye(2), K.diagonal().copy())
    with pytest.raises(ValueError, match='positive semidefinite'):
        kernel.solve_kernel(K, y, 1., max_iter=0, **kwargs)
    assert len(seen) == 1
    assert bool(seen[0].get('eigvals_only')) is (path == 'eigenvalues_only')
    assert (seen[0].get('supplied') is not None) is (path == 'supplied')
    assert K.tobytes() == before


@pytest.mark.parametrize('exponent', [-500, -80, 0, 500])
def test_recovery_shares_relative_policy_and_preserves_input(exponent):
    scale = math.ldexp(1., exponent)
    positive = np.array([1., 2., 3.]) * scale
    K = np.diag(np.r_[-EPS * scale, positive])
    before = K.tobytes()
    prepared = recovery_system(K)._build_original('evd')
    details = prepared[-1]
    assert details['negative_roundoff_eigenvalues_clipped'] == 1
    assert details['positive_eigenvalues_retained'] == 3
    assert details['psd_tolerance'] == kernel._check_spectrum(K.diagonal(), 4, .2)
    assert details['inverse_preconditioner_only'] is True
    assert K.tobytes() == before
    bad = np.diag(np.r_[-scale, positive])
    with pytest.raises(ValueError, match='positive semidefinite'):
        kernel._check_spectrum(bad.diagonal(), 4, .2)
    with pytest.raises(FloatingPointError, match='PSD tolerance'):
        recovery_system(bad)._build_original('evd')


def exact_rank_three_gram(exponent):
    H = np.array([[(-1.) ** ((i & j).bit_count()) for j in range(8)] for i in range(8)])
    B = H[:, :3]
    K = np.ldexp(B @ B.T, exponent)
    values = np.array([math.ldexp(8., exponent)] * 3 + [0.] * 5)
    return K, H / math.sqrt(8.), values


@pytest.mark.parametrize('exponent', [-900, -80, 0, 900])
@pytest.mark.parametrize('path', ['full', 'supplied', 'eigenvalues_only'])
def test_exact_low_rank_gram_remains_admitted_across_scales(exponent, path):
    K, vectors, exact_values = exact_rank_three_gram(exponent)
    before = K.tobytes()
    # H^T H=8I proves rank3 and PSD without a floating eigensolver oracle.
    if path == 'eigenvalues_only':
        values = validated_eigh(K, eigvals_only=True)
        kernel._check_spectrum(values, 8, .2)
        assert spectral_psd_status(values, 8)[0] is True
        assert_allclose(np.sort(values)[-3:] / exact_values[0], np.ones(3), rtol=2e-14, atol=0)
    else:
        supplied = (vectors, exact_values) if path == 'supplied' else None
        _, retained, info = kernel._eigensystem(K, supplied, .2)
        assert np.all(retained >= 0)
        assert info['positive_eigenvalues_retained'] >= 3
        if path == 'supplied':
            assert_array_equal(retained, exact_values)
            assert info['positive_eigenvalues_retained'] == 3
            assert info['eigendecomposition_supplied'] is True
    assert K.tobytes() == before


def ordinary_data():
    X = np.array([[i / 8., ((i * 5) % 13 - 6) / 8., ((i * 7) % 11 - 5) / 8.]
                  for i in range(12)], dtype=float)
    return X, np.tile([-1., 1.], 6)


def assert_model_science_equal(a, b, X):
    for name in ('dual_coef_', 'intercept_', 'objective_history_'):
        assert_allclose(getattr(a, name), getattr(b, name), rtol=2e-12, atol=2e-13)
    for name in ('n_iter_', 'returned_iteration_', 'termination_reason_', 'converged_', 'backend_'):
        assert getattr(a, name) == getattr(b, name)
    assert_allclose(a.decision_function(X), b.decision_function(X), rtol=2e-12, atol=2e-13)
    assert_array_equal(a.predict(X), b.predict(X))
    assert a.diagnostics_['dual_certificate'] == b.diagnostics_['dual_certificate']
    for model in (a, b):
        system = model.diagnostics_['linear_system_diagnostics']
        assert system['linear_recoveries'] == 0
        assert not system.get('spectral_recovery_attempts', [])


@pytest.mark.parametrize('family', ['linear', 'rbf'])
@pytest.mark.parametrize('path', ['full', 'cached', 'eigenvalues_only', 'known_family'])
def test_ordinary_public_fits_match_original_policy_at_identical_input(family, path, monkeypatch):
    X, y = ordinary_data()
    before = X.tobytes()
    options = dict(kernel=family, kernel_kws={'gamma': .4} if family == 'rbf' else None,
                   backend='cholesky' if path in ('eigenvalues_only', 'known_family') else 'spectral',
                   lambd=1., max_iter=4, stopping='fixed', initialization='zero', random_state=31)
    def fit_one():
        model = KernGDWD(**options)
        kwargs = {}
        if path == 'cached':
            model.cv_init(X)
            cache = model._K_eig
            assert cache is not None
        elif path == 'eigenvalues_only':
            # The same package's training construction; explicitly supplied K
            # requires admission instead of borrowing the known-family hint.
            model.cv_init(X)
            kwargs['K'] = model._cv_K.copy()
        model.fit(X, y, **kwargs)
        if path == 'cached':
            assert model._K_eig is cache
            assert model.diagnostics_['eigendecomposition_supplied'] is True
        return model
    with monkeypatch.context() as patch:
        patch.setattr(kernel, '_check_spectrum', original_check_spectrum)
        baseline = fit_one()
    candidate = fit_one()
    assert_model_science_equal(baseline, candidate, X)
    assert X.tobytes() == before
    expected = ('internally constructed PSD kernel (known family)' if path == 'known_family'
                else 'checked eigenvalues of supplied kernel' if path == 'eigenvalues_only'
                else 'validated eigensystem')
    assert candidate.diagnostics_['psd_validation'] == expected


def test_new_small_cv_matches_original_policy_scores_selection_and_refit(monkeypatch):
    X, y = ordinary_data()
    folds = [(np.arange(6), np.arange(6, 12)), (np.arange(6, 12), np.arange(6))]
    def fit_cv():
        return KernGDWDCV(kernel='rbf', kernel_kws_vals=[{'gamma': .4}],
                         backend='spectral', lambd_vals=[.5, 1.], q_vals=[1.], cv=folds,
                         max_iter=3, stopping='fixed', initialization='zero').fit(X, y)
    with monkeypatch.context() as patch:
        patch.setattr(kernel, '_check_spectrum', original_check_spectrum)
        baseline = fit_cv()
    candidate = fit_cv()
    assert baseline.best_params_ == candidate.best_params_
    assert baseline.best_score_ == candidate.best_score_
    # Actual CV timing is recorded by run_cv but is not a scientific output.
    ignored = {'mean_runtime', 'std_runtime'}
    assert baseline.agg_cv_results_.keys() == candidate.agg_cv_results_.keys()
    for key in baseline.agg_cv_results_:
        if key not in ignored:
            assert baseline.agg_cv_results_[key] == candidate.agg_cv_results_[key]
    for a, b in zip(baseline.all_cv_results_, candidate.all_cv_results_, strict=True):
        assert a.keys() == b.keys()
        for key in a:
            if key not in {'runtime', 'init_time'}:
                assert a[key] == b[key]
    assert_model_science_equal(baseline.best_estimator_, candidate.best_estimator_, X)


@pytest.mark.parametrize('case', ['two', 'four', 'three'])
def test_historical_no19_relative_near_psd_admission_is_preserved(case):
    if case == 'two':
        K = np.array([[1., 1. + 2.**-45], [1. + 2.**-45, 1.]])
        v = np.array([-1., 1.])
    elif case == 'four':
        v = np.array([-1., 1., -1., 1.])
        K = np.ones((4, 4)) - 2.**-47 * np.outer(v, v)
    else:
        K = np.ones((3, 3)); K[1, 2] = K[2, 1] = 1. - 2.**-52
        v = np.array([-1., .5, .5])
    before = K.tobytes()
    assert quadratic_fraction(K, v) < 0
    raw = validated_eigh(K, eigvals_only=True)
    original_check_spectrum(raw, len(K), 1.)
    kernel._check_spectrum(raw, len(K), 1.)
    assert spectral_psd_status(raw, len(K))[0] is True
    assert K.tobytes() == before


def test_every_strictly_positive_supplied_mode_is_retained():
    values = np.array([MIN_SUBNORMAL, math.ldexp(1., -100), 1e-14, 1.])
    K = np.diag(values)
    _, retained, info = kernel._eigensystem(K, (np.eye(4), values), 1.)
    assert_array_equal(retained, values)
    assert info['positive_eigenvalues_retained'] == 4
    assert info['negative_roundoff_eigenvalues_clipped'] == 0
