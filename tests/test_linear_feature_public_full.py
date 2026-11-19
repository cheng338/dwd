"""Independent public contracts for dense linear-feature final reporting.

No fixed-fixture solver substitution, old-package fit replay, or relaxed model
acceptance rule is used here. Optional JSON receipts contain observations; the
pytest outcome remains authoritative. Reference optimization is a distinct
primal trust-region solve and is counted separately from public estimator fits.
"""
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
from types import MethodType
from unittest.mock import patch

import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import minimize
from sklearn.exceptions import NotFittedError
import sklearn.metrics.pairwise as sklearn_pairwise

from dwd.gen_kern_dwd import KernGDWD
from _kernel_reference import PrivateReferenceKernGDWD, kernel_route
from dwd.kernel_utils import KernelClfMixin
import dwd.kernel_utils as kernel_utils


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (Decimal, Fraction)):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    return value


@pytest.fixture
def evidence(request):
    state = np.geterr().copy(), np.geterrcall()
    record = dict(schema='ir13-full-public-test-v1', nodeid=request.node.nodeid,
                  public_fit_calls=[], private_kernel_reference_fit_calls=[], primal_reference_calls=0,
                  original_baseline_refits=0)
    yield record
    assert (np.geterr().copy(), np.geterrcall()) == state
    if os.environ.get('IR13_FULL_FEATURE_RESULTS'):
        root = Path(os.environ['IR13_FULL_FEATURE_RESULTS']).resolve()
        root.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha256(request.node.nodeid.encode()).hexdigest()+'.json'
        with (root/name).open('x', encoding='utf-8') as stream:
            json.dump(_json(record), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')


def _model(**options):
    params = dict(kernel='linear', backend='lbfgs', lambd=.5, q=1.,
                  max_iter=500, stopping='optimality', tol=1e-6,
                  initialization='zero', random_state=173)
    params.update(options)
    return KernGDWD(**params)


def _fit(evidence, model, X, y, **kwargs):
    attempt = dict(shape=list(X.shape), backend=model.backend, q=model.q,
                   lambd=model.lambd, max_iter=model.max_iter,
                   stopping=model.stopping, outcome='started')
    private_reference = isinstance(model, PrivateReferenceKernGDWD)
    attempt['entry_point'] = ('private_reference_solver' if private_reference
                              else 'public_estimator_fit')
    evidence['private_kernel_reference_fit_calls' if private_reference
             else 'public_fit_calls'].append(attempt)
    # Observe the actual final contraction; never reconstruct a substitute
    # fitted state or change returned arrays. Both bindings restore on failure.
    from dwd import _linear_feature_diagnostics as feature_core
    original_coefficients = feature_core._feature_coefficients
    original_gradient = feature_core._feature_gradient
    observed = dict(coefficients=[], gradients=[])
    attempt['feature_observations'] = observed
    def coefficients(features, alpha):
        result = original_coefficients(features, alpha)
        observed['coefficients'].append(dict(alpha=alpha.copy(), beta=result.copy()))
        return result
    def gradient(features, z, alpha, twice_penalty):
        result = original_gradient(features, z, alpha, twice_penalty)
        observed['gradients'].append(dict(z=z.copy(), feature_gradient=result.copy(),
                                          twice_penalty=twice_penalty))
        return result
    try:
        with patch.object(feature_core, '_feature_coefficients', coefficients), \
                patch.object(feature_core, '_feature_gradient', gradient):
            returned = model.fit(X, y, **kwargs)
    except BaseException as error:
        attempt.update(outcome='raised', error_type=type(error).__name__, error=str(error))
        raise
    assert returned is model
    attempt.update(outcome='returned', geometry=model.diagnostics_.get('rkhs_geometry'),
                   converged=model.converged_, gradient=model.rkhs_gradient_norm_,
                   objective=model.final_objective_, n_iter=model.n_iter_,
                   returned_iteration=model.returned_iteration_,
                   termination_reason=model.termination_reason_)
    return model


def _feature_route(model):
    assert model.diagnostics_.get('rkhs_geometry') == 'linear_features'
    info = model.diagnostics_['rkhs_geometry_info']
    assert info['provenance'] == 'internally_constructed_named_linear'
    assert 'represented' in info['target'] and 'features' in info['target']
    assert model.backend_ == 'lbfgs'
    assert model.decision_function.__func__ is KernelClfMixin.decision_function
    assert model.predict.__func__ is KernelClfMixin.predict
    def no_arrays(value):
        assert not isinstance(value, np.ndarray)
        if isinstance(value, dict):
            for item in value.values():
                no_arrays(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                no_arrays(item)
    no_arrays(info)


def _equations(X, y, alpha, b, lambd, q, *, rational=False):
    """Independent exact q=1 or 110-digit general-q feature equations."""
    with localcontext() as context:
        context.prec = 110
        cast = (lambda x: Fraction.from_float(float(x))) if rational else (
            lambda x: Decimal.from_float(float(x)))
        one = cast(1)
        n = len(y)
        matrix = [[cast(x) for x in row] for row in X]
        a, labels = list(map(cast, alpha)), list(map(cast, y))
        penalty, qq, offset = cast(lambd), cast(q), cast(b)
        if rational:
            assert qq == 1
        beta = [sum(row[j]*coefficient for row, coefficient in zip(matrix, a))
                for j in range(X.shape[1])]
        scores = [sum(x*w for x, w in zip(row, beta)) for row in matrix]
        margins = [label*(score+offset) for label, score in zip(labels, scores)]
        threshold = qq/(qq+one)
        slopes, losses = [], []
        for margin in margins:
            if margin <= threshold:
                losses.append(one-margin)
                slopes.append(-one)
            else:
                losses.append((threshold/margin)**qq/(qq+one))
                slopes.append(-(threshold/margin)**(qq+one))
        z = [label*slope/n for label, slope in zip(labels, slopes)]
        gradient = [sum(row[j]*value for row, value in zip(matrix, z))
                    +2*penalty*beta[j] for j in range(X.shape[1])]
        action = [sum(x*g for x, g in zip(row, gradient)) for row in matrix]
        bias = sum(z)
        squared = sum(g*g for g in gradient)
        norm2 = sum(w*w for w in beta)
        primal = sum(losses)/n+penalty*norm2
        norm = max(abs(float(bias)), math.sqrt(float(squared)))
        infnorm = max(abs(float(bias)), *(abs(float(x)) for x in action))
        if norm2 == 0:
            conversion = 0.
        elif rational:
            conversion = float(4*norm2)
        else:
            conversion = float((qq+one)*((qq+one)/qq)**qq*norm2**((qq+one)/2))
        return dict(beta=list(map(float, beta)), beta_exact=beta,
                    scores=list(map(float, scores)),
                    primal=float(primal), primal_exact=str(primal), norm2=float(norm2),
                    gradient=norm, gradient_squared=float(squared), bias=float(bias),
                    gradient_vector_exact=gradient, bias_exact=bias,
                    gradient_inf=infnorm, C=conversion,
                    arithmetic='Fraction' if rational else 'Decimal110')


def _check_report(evidence, model, X, y, *, rational=False):
    _feature_route(model)
    signed = np.where(y == model.classes_[1], 1., -1.)
    expected = _equations(X, signed, model.dual_coef_.ravel(), model.intercept_[0],
                          model.lambd, model.q, rational=rational)
    observed = evidence['public_fit_calls'][-1]['feature_observations']
    assert len(observed['coefficients']) == len(observed['gradients']) == 1
    beta = observed['coefficients'][0]['beta']
    actual_gradient = observed['gradients'][0]['feature_gradient']
    z = observed['gradients'][0]['z']
    as_fraction = lambda value: Fraction.from_float(float(value))
    ghat = list(map(as_fraction, actual_gradient))
    exact_matrix = [[as_fraction(value) for value in row] for row in X]
    action = [sum(value*g for value, g in zip(row, ghat)) for row in exact_matrix]
    bias = sum(map(as_fraction, z), Fraction())
    action_metric = max(abs(bias), *map(abs, action))
    eps, eta = Fraction(1, 2**52), Fraction(1, 2**1074)
    product_allowance = 2*X.shape[1]*eta
    action_allowances = [product_allowance+2*eps*(abs(value)+product_allowance)+eta
                         for value in action]
    rounding_allowance = max(2*eps*abs(bias)+eta, *action_allowances)
    oracle_gradient = list(map(Fraction, expected['gradient_vector_exact']))
    oracle_bias = Fraction(expected['bias_exact'])
    propagated = max(abs(bias-oracle_bias), *(sum(abs(value)*abs(g-reference)
        for value, g, reference in zip(row, ghat, oracle_gradient)) for row in exact_matrix))
    oracle_action = [sum(value*g for value, g in zip(row, oracle_gradient))
                     for row in exact_matrix]
    oracle_metric = max(abs(oracle_bias), *map(abs, oracle_action))
    # Save complete observed/oracle state before any numerical assertion so a
    # failed validation remains diagnosable without refitting the model.
    evidence.update(independent_equations=expected,
        returned_alpha=model.dual_coef_.ravel().copy(), returned_offset=float(model.intercept_[0]),
        reported_metrics=dict(primal=model.final_objective_, gradient=model.rkhs_gradient_norm_,
                              gradient_inf=model.gradient_inf_norm_, C=model.C_),
        coefficient_action_check=dict(exact_action=action, exact_bias=bias,
            exact_metric_from_observed_gradient=action_metric,
            arithmetic_rounding_allowance=rounding_allowance,
            propagated_component_discrepancy=propagated,
            true_oracle_metric=oracle_metric,
            actual_oracle_discrepancy=abs(as_fraction(model.gradient_inf_norm_)-oracle_metric)))
    np.testing.assert_allclose(beta, expected['beta'], rtol=3e-11, atol=8e-12)
    np.testing.assert_allclose(actual_gradient, list(map(float, oracle_gradient)),
                               rtol=3e-11, atol=8e-12)
    np.testing.assert_allclose(float(bias), float(oracle_bias), rtol=3e-11, atol=8e-12)
    for actual, key in ((model.final_objective_, 'primal'),
                        (model.rkhs_gradient_norm_, 'gradient'), (model.C_, 'C')):
        np.testing.assert_allclose(actual, expected[key], rtol=3e-11, atol=8e-12)
    assert abs(as_fraction(model.gradient_inf_norm_)-action_metric) <= rounding_allowance
    assert abs(as_fraction(model.gradient_inf_norm_)-oracle_metric) <= propagated+rounding_allowance
    assert model.converged_ == (model.rkhs_gradient_norm_ <= model.tol)
    assert model.optimality_met_ == model.converged_
    assert model.stationarity_checked_
    # This independent oracle is an estimate in Decimal mode. Avoid declaring
    # an inconsistency solely inside its tight scalar comparison allowance.
    if abs(expected['gradient']-model.tol) > 8e-12:
        assert model.converged_ == (expected['gradient'] <= model.tol)
    endpoint = model.diagnostics_['dual_certificate']['dual_objective_lower_bound']
    assert endpoint <= expected['primal']+2e-13*max(1., abs(expected['primal']))
    assert model.dual_equality_residual_ == 0.
    assert abs(model.final_objective_-model.objective_history_[model.returned_iteration_]) <= (
        5e-8*max(1., abs(model.final_objective_)))
    evidence['independent_equations'] = expected
    evidence['dual_lower_endpoint'] = endpoint
    return expected


PANELS = [
    ('small_q_small_scale', 6, 1, .1, .03, .2, 301, 'C'),
    ('ordinary', 10, 3, 1., 1., 1., 302, 'C'),
    ('large_q_scale', 12, 5, 4., 3., 2.3, 303, 'C'),
    ('wide_features', 8, 24, .5, .2, 1., 304, 'C'),
    ('duplicated_rows', 10, 3, 1., .7, 1., 305, 'C'),
    ('zero_rank', 6, 2, 0., 2., 1., 306, 'C'),
    ('fortran', 8, 4, 2., .1, 2.3, 307, 'F'),
    ('known_scale1000', 20, 1, 1000., 1., 1., 1, 'C'),
]


@pytest.mark.parametrize('case,n,p,scale,lambd,q,seed,order', PANELS, ids=[x[0] for x in PANELS])
def test_public_feature_numerical_panel(case, n, p, scale, lambd, q, seed, order, evidence):
    X = np.array(np.random.RandomState(seed).normal(size=(n, p))*scale, order=order)
    if case == 'duplicated_rows':
        X[1::2] = X[::2]
    y = np.r_[np.zeros(n//2), np.ones(n-n//2)]
    original = X.copy()
    model = _fit(evidence, _model(lambd=lambd, q=q, random_state=0), X, y)
    _check_report(evidence, model, X, y, rational=q == 1.)
    np.testing.assert_array_equal(X, original)
    assert not np.shares_memory(X, model._Xfit)
    if case == 'known_scale1000':
        assert model.converged_
    evidence.update(case=case, alpha=model.dual_coef_, intercept=model.intercept_,
                    objective_history=model.objective_history_)


@pytest.mark.parametrize('q,lambd', [(.2, .13), (1., .125), (2.3, 2.)])
def test_step_zero_independent_full_report(q, lambd, evidence):
    X = np.array([[1., .5], [-.5, 1.], [.25, -.75], [-1., -.25]])
    y = np.array([-1., 1., -1., 1.])
    alpha = np.array([.125, -.0625, .1875, -.125])
    model = _fit(evidence, _model(q=q, lambd=lambd, max_iter=0), X, y,
                 alpha_init=alpha, offset_init=.125)
    np.testing.assert_array_equal(model.dual_coef_.ravel(), alpha)
    assert model.intercept_[0] == .125 and model.n_iter_ == 0
    assert len(model.objective_history_) == 1
    _check_report(evidence, model, X, y, rational=q == 1.)


@pytest.mark.parametrize('initialization', ['zero', 'random', 'explicit'])
def test_initialization_and_nonconvergence_are_preserved(initialization, evidence):
    X, y = np.eye(4), np.array([0, 0, 1, 1])
    options = {}
    model = _model(max_iter=0, initialization='random' if initialization == 'random' else 'zero')
    expected = np.zeros(4)
    if initialization == 'random':
        expected = np.random.RandomState(173).normal(size=4)
        expected /= np.linalg.norm(expected)
    elif initialization == 'explicit':
        expected = np.array([.125, -.0625, .1875, -.125])
        options['alpha_init'] = expected.copy()
    _fit(evidence, model, X, y, **options)
    np.testing.assert_array_equal(model.dual_coef_.ravel(), expected)
    assert model.n_iter_ == model.returned_iteration_ == 0
    assert model.termination_reason_ == 'max_iter' and not model.converged_
    _check_report(evidence, model, X, y, rational=True)


class _Inherited(KernGDWD):
    pass


def _changed_compute(self, X):
    return 2*self._Xfit@X.T


def _changed_training(self, X):
    return 2*(X@X.T)


def _changed_capture(self, X):
    return KernGDWD._capture_kernel_basis(self, X)


def _changed_known_psd(self):
    return True


ROUTES = ['supplied_K', 'supplied_eigenpairs', 'precomputed', 'callable', 'subclass',
          'compute_hook', 'training_hook', 'capture_hook', 'known_psd_hook',
          'cv_cache', 'changed_cv_cache', 'sparse', 'spectral', 'cholesky', 'reference']


@pytest.mark.parametrize('route', ROUTES)
def test_untrusted_or_other_routes_keep_original_kernel_reporting(route, evidence):
    X, y = np.eye(4), np.array([0, 0, 1, 1])
    alpha = np.array([.125, -.0625, .1875, -.125])
    params = _model(max_iter=0).get_params()
    kind = _Inherited if route == 'subclass' else KernGDWD
    if route in ('spectral', 'cholesky'):
        params['backend'] = route
    if route == 'reference':
        params.update(implementation='reference', backend='spectral')
        # Numerical private-reference control, not a public estimator route.
        kind = kernel_route
    if route == 'precomputed':
        params['kernel'] = 'precomputed'
    if route == 'callable':
        params['kernel'] = lambda train, query: train@query.T
    model = kind(**params)
    hook = {'compute_hook': ('_compute_kernel', _changed_compute),
            'training_hook': ('_compute_training_kernel', _changed_training),
            'capture_hook': ('_capture_kernel_basis', _changed_capture),
            'known_psd_hook': ('_known_psd_kernel', _changed_known_psd)}.get(route)
    if hook is not None:
        setattr(model, hook[0], MethodType(hook[1], model))
    fit_options = dict(alpha_init=alpha, offset_init=.0625)
    expected_K = np.eye(4)
    if route == 'supplied_K':
        expected_K = 2*np.eye(4)
        fit_options['K'] = expected_K.copy()
    if route == 'supplied_eigenpairs':
        fit_options['K_eig'] = (np.eye(4), np.ones(4))
    if route in ('compute_hook', 'training_hook'):
        expected_K = 2*np.eye(4)
    if route in ('cv_cache', 'changed_cv_cache'):
        model.cv_init(X)
        if route == 'changed_cv_cache':
            expected_K = 2*np.eye(4)
            model._cv_K = expected_K.copy()
            model._K_eig = (np.eye(4), np.full(4, 2.))
    if route == 'sparse':
        X = sparse.csr_matrix(X)
    _fit(evidence, model, X, y, **fit_options)
    assert model.diagnostics_.get('rkhs_geometry') != 'linear_features'
    signed = np.where(y == 1, 1., -1.)
    margins = signed*(expected_K@alpha+.0625)
    assert np.all(margins <= .5)
    primal = np.mean(1.-margins)+model.lambd*(alpha@expected_K@alpha)
    assert model.final_objective_ == pytest.approx(primal, rel=2e-14, abs=2e-15)
    assert model.C_ == pytest.approx(4*(alpha@expected_K@alpha), rel=2e-14)
    evidence['route'] = route


@pytest.mark.parametrize('hook', ['kernel_utils_pairwise', 'linear_registry',
                                 'linear_attribute', 'pairwise_arrays', 'sparse_dot'])
def test_replaced_external_linear_builders_do_not_gain_feature_provenance(hook, monkeypatch, evidence):
    X, y = np.eye(4), np.array([0, 0, 1, 1])
    if hook == 'kernel_utils_pairwise':
        original = kernel_utils.pairwise_kernels
        monkeypatch.setattr(kernel_utils, 'pairwise_kernels',
                            lambda *args, **kwargs: 2*original(*args, **kwargs))
    elif hook == 'linear_registry':
        original = sklearn_pairwise.PAIRWISE_KERNEL_FUNCTIONS['linear']
        monkeypatch.setitem(sklearn_pairwise.PAIRWISE_KERNEL_FUNCTIONS, 'linear',
                           lambda *args, **kwargs: 2*original(*args, **kwargs))
    elif hook == 'linear_attribute':
        original = sklearn_pairwise.linear_kernel
        monkeypatch.setattr(sklearn_pairwise, 'linear_kernel',
                            lambda *args, **kwargs: 2*original(*args, **kwargs))
    elif hook == 'pairwise_arrays':
        original = sklearn_pairwise.check_pairwise_arrays
        def replacement(*args, **kwargs):
            left, right = original(*args, **kwargs)
            return 2*left, right
        monkeypatch.setattr(sklearn_pairwise, 'check_pairwise_arrays', replacement)
    else:
        original = sklearn_pairwise.safe_sparse_dot
        monkeypatch.setattr(sklearn_pairwise, 'safe_sparse_dot',
                            lambda *args, **kwargs: 2*original(*args, **kwargs))
    # At alpha=0 all score/objective comparisons agree for every operator.
    # Routing therefore has to establish construction, not infer it from the
    # currently represented zero predictor.
    model = _fit(evidence, _model(max_iter=0), X, y)
    assert model.diagnostics_.get('rkhs_geometry') != 'linear_features'
    assert model.final_objective_ == 1.
    evidence['external_builder_hook'] = hook


def test_preimport_linear_wrapper_is_outside_standard_construction(evidence):
    source = r'''
import functools
import json
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import sklearn.metrics.pairwise as pairwise
assert 'dwd' not in sys.modules
original = pairwise.PAIRWISE_KERNEL_FUNCTIONS['linear']
@functools.wraps(original)
def replacement(*args, **kwargs):
    return 2*original(*args, **kwargs)
pairwise.PAIRWISE_KERNEL_FUNCTIONS['linear'] = replacement
pairwise.linear_kernel = replacement
import dwd
from dwd.gen_kern_dwd import KernGDWD
model = KernGDWD(kernel='linear', backend='lbfgs', initialization='zero',
                 max_iter=0, random_state=173).fit(np.eye(4), [0, 0, 1, 1])
print(json.dumps(dict(package=str(Path(dwd.__file__).resolve()),
                     geometry=model.diagnostics_.get('rkhs_geometry'),
                     objective=model.final_objective_, fit_calls=1)))
'''
    evidence['public_fit_calls'].append(dict(outcome='isolated_child_started',
        max_iter=0, preimport_builder_wrapper=True))
    import dwd
    package_root = str(Path(dwd.__file__).resolve().parents[1])
    result = subprocess.run([sys.executable, '-B', '-c', source, package_root],
                            text=True, capture_output=True, timeout=60, check=False)
    evidence['preimport_child'] = dict(returncode=result.returncode,
                                     stdout=result.stdout, stderr=result.stderr)
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    import dwd
    assert observed['package'] == str(Path(dwd.__file__).resolve())
    assert observed['fit_calls'] == 1 and observed['geometry'] != 'linear_features'
    assert observed['objective'] == 1.
    evidence['public_fit_calls'][-1]['outcome'] = 'isolated_child_returned'


def test_missing_private_builder_helper_declines_without_breaking_import(evidence):
    source = r'''
import json
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
import sklearn.metrics.pairwise as pairwise
assert 'dwd' not in sys.modules
original = getattr(pairwise, '_parallel_pairwise', None)
if original is not None:
    delattr(pairwise, '_parallel_pairwise')
import dwd
import dwd.gen_kern_dwd as kernel_module
assert not kernel_module._LINEAR_KERNEL_SOURCES_OK
assert not kernel_module._linear_kernel_sources_unchanged()
if original is not None:
    pairwise._parallel_pairwise = original
# Restoring a link after import cannot retroactively qualify the snapshot.
assert not kernel_module._linear_kernel_sources_unchanged()
print(json.dumps(dict(package=str(Path(dwd.__file__).resolve()),
                     imported=True, provenance_qualified=False, fit_calls=0)))
'''
    import dwd
    package_root = str(Path(dwd.__file__).resolve().parents[1])
    result = subprocess.run([sys.executable, '-B', '-c', source, package_root],
                            text=True, capture_output=True, timeout=60, check=False)
    evidence['missing_private_helper_child'] = dict(returncode=result.returncode,
        stdout=result.stdout, stderr=result.stderr)
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    import dwd
    assert observed['package'] == str(Path(dwd.__file__).resolve())
    assert observed['imported'] and not observed['provenance_qualified']
    assert observed['fit_calls'] == 0


def test_objective_stopping_keeps_its_original_meaning(evidence):
    X = np.array([[1., 0.], [0., 1.], [-1., -.5], [.5, -.25]])
    y = np.array([0, 0, 1, 1])
    model = _fit(evidence, _model(stopping='objective', obj_tol=100., max_iter=20), X, y)
    assert model.termination_reason_ == 'objective_tolerance' and model.n_iter_ == 1
    assert model.objective_tolerance_met_
    _check_report(evidence, model, X, y, rational=True)


@pytest.mark.parametrize('stop_kind', ['return_true', 'stop_iteration'])
def test_callback_stop_reports_actual_returned_state(stop_kind, evidence):
    X = np.random.RandomState(405).normal(size=(8, 3))
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    observed = []
    def callback(state):
        assert not state['alpha'].flags.writeable
        assert not state['training_scores'].flags.writeable
        observed.append(dict(iteration=state['iteration'], alpha=state['alpha'].copy(),
                             offset=state['offset'], objective=state['objective']))
        if state['iteration'] == 1:
            if stop_kind == 'stop_iteration':
                raise StopIteration
            return True
    model = _fit(evidence, _model(stopping='fixed', max_iter=8, callback=callback), X, y)
    assert model.termination_reason_ == 'callback_stop' and model.n_iter_ == 1
    assert [state['iteration'] for state in observed] == [0, 1]
    np.testing.assert_array_equal(model.dual_coef_.ravel(), observed[-1]['alpha'])
    assert model.intercept_[0] == observed[-1]['offset']
    assert model.objective_history_[-1] == observed[-1]['objective']
    _check_report(evidence, model, X, y, rational=True)


def test_validation_reports_the_selected_best_state(evidence):
    X = np.random.RandomState(406).normal(size=(12, 4))
    y = np.r_[np.zeros(6), np.ones(6)]
    query = np.random.RandomState(407).normal(size=(7, 4))
    validation_y = np.array([0, 1, 0, 1, 0, 1, 0])
    observed = {}
    def callback(state):
        observed[state['iteration']] = (state['alpha'].copy(), state['offset'], state['objective'])
    model = _fit(evidence, _model(stopping='validation', max_iter=6, patience=2,
                 min_delta=1., callback=callback), X, y, validation_data=(query, validation_y))
    assert model.validation_history_ and model.returned_iteration_ == 1
    assert model.returned_iteration_ <= model.n_iter_
    alpha, offset, objective = observed[1]
    np.testing.assert_array_equal(model.dual_coef_.ravel(), alpha)
    assert model.intercept_[0] == offset and model.objective_history_[1] == objective
    _check_report(evidence, model, X, y, rational=True)


@pytest.mark.parametrize('failure', ['exception', 'invalid_return'])
def test_failed_refit_clears_feature_state_and_preserves_exception(failure, evidence):
    X, y = np.eye(4), np.array([0, 0, 1, 1])
    model = _fit(evidence, _model(max_iter=0), X, y)
    _feature_route(model)
    marker = RuntimeError('independent callback sentinel')
    def callback(state):
        if failure == 'exception':
            raise marker
        return 'invalid callback result'
    model.set_params(callback=callback)
    with pytest.raises(RuntimeError if failure == 'exception' else ValueError) as caught:
        _fit(evidence, model, X, y)
    if failure == 'exception':
        assert caught.value is marker
    assert not hasattr(model, '_Xfit')
    assert not any(name.endswith('_') and not name.startswith('_') for name in vars(model))
    with pytest.raises(NotFittedError):
        model.predict(X)


def test_owned_basis_pickle_and_prediction_policy(evidence):
    X = np.random.RandomState(410).normal(size=(10, 3))
    y = np.r_[np.zeros(5), np.ones(5)]
    model = _fit(evidence, _model(), X, y)
    _feature_route(model)
    original_X = X.copy()
    query = np.random.RandomState(411).normal(size=(6, 3))
    before = model.decision_function(query)
    X[:] = 100.
    np.testing.assert_array_equal(model._Xfit, original_X)
    assert not np.shares_memory(X, model._Xfit)
    np.testing.assert_allclose(model.decision_function(query), before, rtol=0, atol=5e-7)
    restored = pickle.loads(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    assert type(restored) is KernGDWD
    for name in ('_Xfit', 'dual_coef_', 'intercept_', 'objective_history_'):
        np.testing.assert_array_equal(getattr(restored, name), getattr(model, name))
    assert restored.diagnostics_['rkhs_geometry_info'] == model.diagnostics_['rkhs_geometry_info']
    assert restored.affine_computation_ == model.affine_computation_ == 'joint'
    assert restored.prediction_precision_ == model.prediction_precision_
    np.testing.assert_allclose(restored.decision_function(query), before, rtol=0, atol=5e-7)
    np.testing.assert_array_equal(restored.predict(query), model.predict(query))
    expected = _check_report(evidence, model, original_X, y, rational=True)
    beta = np.asarray(expected['beta'])
    pivot = int(np.argmax(abs(beta)))
    assert beta[pivot] != 0
    center = -model.intercept_[0]/beta[pivot]
    boundary = np.zeros((3, 3))
    boundary[:, pivot] = [np.nextafter(center, -np.inf), center, np.nextafter(center, np.inf)]
    scores = model.decision_function(boundary)
    assert np.isfinite(scores).all()
    evidence['boundary_observation'] = dict(public=scores,
        feature_formula=boundary@beta+model.intercept_[0],
        universal_class_parity_required=False, observed_public_prediction=True)


@pytest.mark.parametrize('case,q', [('identity', 1.), ('zero_imbalanced_small_q', .2),
                                    ('zero_imbalanced_large_q', 2.3)])
def test_analytic_optima_bound_the_reported_dual(case, q, evidence):
    if case == 'identity':
        X, y, penalty = np.eye(2), np.array([0, 1]), 1./16.
        optimum = Fraction(3, 8)
    else:
        X, y, penalty = np.zeros((3, 2)), np.array([0, 1, 1]), .7
        with localcontext() as context:
            context.prec = 110
            qq = Decimal.from_float(q)
            bias = (qq/(qq+1))*Decimal(2)**(1/(qq+1))
            optimum = (1+bias+2*(qq/(qq+1)/bias)**qq/(qq+1))/3
    model = _fit(evidence, _model(lambd=penalty, q=q, max_iter=1000, tol=1e-8), X, y)
    expected = _check_report(evidence, model, X, y, rational=q == 1.)
    assert model.converged_
    endpoint = Fraction.from_float(model.diagnostics_['dual_certificate']['dual_objective_lower_bound'])
    assert endpoint <= Fraction(optimum)
    assert abs(model.final_objective_-float(optimum)) <= 2e-12
    if case != 'identity':
        assert abs(model.intercept_[0]-float(bias)) <= 1e-7
        assert expected['norm2'] == model.C_ == 0.
    evidence['analytic_optimum'] = optimum


def _primal_value_grad_hessian(state, X, signed, lambd, q):
    """Independent scalar loss equations; no package loss/report helper."""
    beta, b = state[:-1], state[-1]
    margins = signed*(X@beta+b)
    threshold = q/(q+1.)
    loss, slope, second = [], [], []
    for margin in margins:
        if margin <= threshold:
            loss.append(1.-margin); slope.append(-1.); second.append(0.)
        else:
            power = (threshold/margin)**(q+1.)
            loss.append((threshold/margin)**q/(q+1.))
            slope.append(-power); second.append((q+1.)*power/margin)
    augmented = np.column_stack((X, np.ones(len(X))))
    z = signed*np.asarray(slope)/len(X)
    gradient = augmented.T@z
    gradient[:-1] += 2*lambd*beta
    hessian = augmented.T@(np.asarray(second)[:, None]*augmented)/len(X)
    hessian[:-1, :-1] += 2*lambd*np.eye(X.shape[1])
    value = math.fsum(loss)/len(X)+lambd*float(beta@beta)
    return value, gradient, hessian


def _directed_fraction(value, rounding):
    with localcontext() as context:
        context.prec = 110
        context.rounding = rounding
        return Decimal(value.numerator)/Decimal(value.denominator)


def _independent_feasible_dual(X, signed, state, lambd, q):
    """Independent integer-grid feasibility and directed Decimal dual bound.

    This deliberately does not call the package's slope projection, conjugate,
    quadratic, or report helpers. It supplies a separate reference-quality
    check for the trust-region primal solve, not a new package acceptance rule.
    """
    margins = signed*(X@state[:-1]+state[-1])
    threshold = q/(q+1.)
    quantum = 2**40
    units = [int(math.floor((1. if m <= threshold else (threshold/m)**(q+1.))*quantum))
             for m in margins]
    labels = [int(label) for label in signed]
    excess = sum(label*value for label, value in zip(labels, units))
    selected = 1 if excess > 0 else -1
    remaining = abs(excess)
    for index in sorted(range(len(units)), key=lambda i: units[i], reverse=True):
        if labels[index] == selected:
            amount = min(remaining, units[index])
            units[index] -= amount
            remaining -= amount
    assert remaining == 0 and sum(label*value for label, value in zip(labels, units)) == 0
    rho = [Fraction(value, quantum) for value in units]
    assert all(0 <= value <= 1 for value in rho)
    vector = [label*value for label, value in zip(labels, rho)]
    projection = [sum(Fraction.from_float(float(row[j]))*value
                      for row, value in zip(X, vector)) for j in range(X.shape[1])]
    penalty = sum(value*value for value in projection)/(4*Fraction.from_float(lambd)*len(X)**2)
    exponent = Fraction.from_float(q)/(1+Fraction.from_float(q))
    lower_powers = []
    with localcontext() as context:
        context.prec = 110
        for value in rho:
            if value in (0, 1):
                lower_powers.append(Decimal(value.numerator))
                continue
            number = Decimal.from_float(float(value))
            assert Fraction(number) == value
            logarithm = number.ln().next_minus()
            upper_exponent = _directed_fraction(exponent, ROUND_CEILING)
            context.rounding = ROUND_FLOOR
            lower_powers.append((upper_exponent*logarithm).exp().next_minus())
        context.rounding = ROUND_FLOOR
        lower = sum(lower_powers, Decimal())/len(X)
        lower -= _directed_fraction(penalty, ROUND_CEILING)
    return dict(lower=lower, exact_feasible=True, quantum=quantum,
                exact_class_mass=sum(rho[i] for i in range(len(rho)) if labels[i] > 0))


@pytest.mark.parametrize('n,p,lambd,q,seed', [(64, 7, .2, 1., 501),
    (80, 5, 1., .2, 502), (32, 48, .7, 2.3, 503)])
def test_larger_public_model_against_independent_primal_reference(n, p, lambd, q, seed, evidence):
    X = np.random.RandomState(seed).normal(size=(n, p))/math.sqrt(p)
    y = np.r_[np.zeros(n//2), np.ones(n-n//2)]
    model = _fit(evidence, _model(lambd=lambd, q=q, max_iter=1000), X, y)
    expected = _check_report(evidence, model, X, y)
    signed = 2*y-1
    evaluate = lambda state: _primal_value_grad_hessian(state, X, signed, lambd, q)
    evidence['primal_reference_calls'] += 1
    reference = minimize(lambda state: evaluate(state)[0], np.zeros(p+1),
        jac=lambda state: evaluate(state)[1], hess=lambda state: evaluate(state)[2],
        method='trust-exact', options={'gtol': 1e-10, 'maxiter': 250})
    value, gradient, _ = evaluate(reference.x)
    # SciPy status alone is not a certificate. Check its independent equations.
    assert max(np.linalg.norm(gradient[:-1]), abs(gradient[-1])) <= 2e-7
    assert abs(model.final_objective_-value) <= 2e-8*max(1., abs(value))
    endpoint = model.diagnostics_['dual_certificate']['dual_objective_lower_bound']
    assert endpoint <= value+2e-13*max(1., abs(value))
    independent_dual = _independent_feasible_dual(X, signed, reference.x, lambd, q)
    reference_gap = value-float(independent_dual['lower'])
    assert -2e-13 <= reference_gap <= 2e-7*max(1., abs(value))
    evidence['independent_primal_reference'] = dict(method='trust-exact',
        success=bool(reference.success), status=int(reference.status), message=str(reference.message),
        objective=value, gradient=gradient, iterations=int(reference.nit),
        candidate_objective_error=model.final_objective_-value,
        candidate_stationarity=expected['gradient'],
        independent_dual=independent_dual, independent_reference_gap=reference_gap)


def test_strict_numpy_error_policy_survives_public_feature_report(evidence):
    X, y = np.eye(4), np.array([0, 0, 1, 1])
    with np.errstate(all='raise'):
        model = _fit(evidence, _model(max_iter=0), X, y)
        _check_report(evidence, model, X, y, rational=True)
        assert all(value == 'raise' for value in np.geterr().values())
