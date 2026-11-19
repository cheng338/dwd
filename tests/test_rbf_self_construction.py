"""Finite construction/cache/provenance contracts; no solver-policy replacement."""
import pickle
from types import MethodType
from unittest.mock import patch

import numpy as np
import pytest
from scipy.sparse import csr_matrix, csr_array
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import pairwise_kernels

from dwd.gen_kern_dwd import KernGDWD
from dwd import kernel_utils

POLICY = 'norm_first_v1'


def features(kind='dense'):
    X = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.],
                  [-1., 0.], [0., -1.], [-1., -1.], [2., 1.]]) / 4.
    if kind == 'fortran':
        return np.asfortranarray(X)
    if kind == 'readonly':
        X.flags.writeable = False
    elif kind == 'strided':
        owner = np.zeros((len(X), 4)); owner[:, ::2] = X; X = owner[:, ::2]
    elif kind == 'csr':
        return csr_matrix(X)
    elif kind == 'csr_array':
        return csr_array(X)
    return X


def model(**options):
    args = dict(kernel='rbf', kernel_kws={'gamma': .5}, lambd=.13,
                initialization='zero', max_iter=2, stopping='fixed', random_state=173)
    args.update(options)
    return KernGDWD(**args)


def dense(X):
    return X.toarray() if hasattr(X, 'toarray') else np.asarray(X)


def marker(estimator):
    return getattr(estimator, 'kernel_self_construction_', None)


@pytest.fixture(autouse=True)
def unchanged_policy():
    errors, callback = np.geterr(), np.geterrcall()
    try:
        yield
        assert np.geterr() == errors
        assert np.geterrcall() is callback
    finally:
        np.seterr(**errors); np.seterrcall(callback)


@pytest.mark.parametrize('kind', ['dense', 'fortran', 'readonly', 'strided', 'csr', 'csr_array'])
def test_owned_preparation_and_cross_query_routing(kind):
    X = features(kind); original = dense(X).copy()
    current = model().cv_init(X)
    assert marker(current) == POLICY
    assert getattr(current, '_cv_rbf_self_generation', None) == POLICY
    assert current._cv_cache_matches(current._cv_X)
    np.testing.assert_array_equal(dense(X), original)
    assert current._cv_X is not X
    if not hasattr(X, 'toarray'):
        assert not np.shares_memory(current._cv_X, X)
    np.testing.assert_array_equal(current._cv_K, current._cv_K.T)
    # The separate self marker must not change the evaluator for cross inputs.
    query = current._cv_X[[5, 1, 5, 0]].copy()
    first = current._compute_kernel(query)
    current.kernel_self_construction_ = 'unrelated_self_marker'
    second = current._compute_kernel(query)
    np.testing.assert_array_equal(first, second)
    assert current.kernel_computation_ == 'guarded_norm_sum_rel_v2'


@pytest.mark.parametrize('kind', ['dense', 'csr'])
def test_existing_large_asymmetry_reconstruction_survives_untrusted_hook(kind):
    X = features(kind)
    def perturbed(X, Y, metric, **kwargs):
        K = pairwise_kernels(X, Y, metric=metric, **kwargs)
        if X is Y:
            K[0, 1] += 1e-9
        return K
    with patch.object(kernel_utils, 'pairwise_kernels', side_effect=perturbed):
        current = model().cv_init(X)
    assert current.kernel_computation_ == 'norm_sum'
    assert current.kernel_symmetry_correction_ > 1e-10
    assert marker(current) is None
    np.testing.assert_array_equal(current._cv_K, current._cv_K.T)


@pytest.mark.parametrize('change', ['missing', 'foreign'])
def test_stale_cache_generation_is_rebuilt_instead_of_relabelled(change):
    X = features(); current = model().cv_init(X)
    old_K = current._cv_K.copy()
    if change == 'missing':
        current.__dict__.pop('_cv_rbf_self_generation', None)
    else:
        current._cv_rbf_self_generation = 'different_generation'
    assert not current._cv_cache_matches(current._cv_X)
    # Missing or foreign generations must be rebuilt, not merely relabelled.
    current.cv_init(X)
    assert marker(current) == POLICY
    assert getattr(current, '_cv_rbf_self_generation', None) == POLICY
    assert current._cv_cache_matches(current._cv_X)
    np.testing.assert_array_equal(current._cv_K, old_K)


@pytest.mark.parametrize('change', ['values', 'gamma', 'gamma_type'])
def test_data_and_parameter_cache_guards_remain(change):
    X = features(); current = model().cv_init(X)
    probe = X.copy()
    if change == 'values':
        probe[0, 0] += .125
    elif change == 'gamma':
        current.kernel_kws = {'gamma': .75}
    else:
        current.kernel_kws = {'gamma': np.float32(.5)}
    assert not current._cv_cache_matches(probe)


@pytest.mark.parametrize('kind', ['dense', 'csr'])
def test_fresh_cached_pickle_and_failed_refit_lifecycle(kind):
    X = features(kind); y = np.array([-1, 1, -1, 1, -1, 1, -1, 1])
    fresh = model().fit(X, y)
    cached = model().cv_init(X).fit(X, y)
    query = X[[5, 1, 5, 0]].copy()
    expected = fresh.decision_function(query)
    for fitted in (fresh, cached, pickle.loads(pickle.dumps(cached))):
        assert marker(fitted) == POLICY
        for batch in (None, 1, 3):
            scores = fitted.decision_function(query, prediction_batch_size=batch)
            np.testing.assert_allclose(scores, expected, rtol=0.,
                atol=5e-7*max(1., float(np.max(np.abs(expected)))))
        # Fitted v1/v2 policy remains independent of the new construction marker.
        del_marker = pickle.loads(pickle.dumps(fitted))
        del_marker.__dict__.pop('kernel_self_construction_', None)
        np.testing.assert_array_equal(del_marker.decision_function(query),
                                      fitted.decision_function(query))
    with pytest.raises(ValueError, match='exactly two classes'):
        cached.fit(X, np.ones(len(y)))
    assert 'kernel_self_construction_' not in vars(cached)
    assert 'kernel_computation_' not in vars(cached)
    with pytest.raises(NotFittedError):
        cached.predict(query)
    assert cached._cv_cache_matches(cached._cv_X)


@pytest.mark.parametrize('route', ['supplied', 'precomputed', 'callable'])
def test_small_asymmetric_external_matrices_are_not_repaired(route):
    X = features(); y = np.where(np.arange(len(X)) % 2, 1., -1.)
    K = np.eye(len(X)); K[0, 1] = np.finfo(float).eps
    original = K.copy()
    if route == 'supplied':
        fitted = model(max_iter=0).fit(X, y, K=K)
    elif route == 'precomputed':
        fitted = model(kernel='precomputed', kernel_kws=None, max_iter=0).fit(K, y)
    else:
        fitted = model(kernel=lambda A, B: K.copy(), kernel_kws=None, max_iter=0).fit(X, y)
    assert marker(fitted) is None
    np.testing.assert_array_equal(K, original)
    assert fitted.diagnostics_['dual_certificate']['fallback_reason'] == 'asymmetric_kernel_no_nonzero_dual_certificate'


@pytest.mark.parametrize('hook', ['subclass', 'instance_compute', 'class_compute',
                                  'training_method', 'known_psd', 'cv_init',
                                  'pairwise', 'registry', 'guarded', 'sparse_dot'])
def test_custom_construction_never_gets_new_provenance(hook):
    import sklearn.metrics.pairwise as pairwise
    import sklearn.utils.extmath as extmath
    import dwd._guarded_rbf as guarded
    X = features()
    current = model()
    if hook == 'subclass':
        class Child(KernGDWD):
            pass
        current = Child(kernel='rbf', kernel_kws={'gamma': .5})
        current.cv_init(X)
    else:
        if hook == 'instance_compute':
            original = current._compute_kernel
            current._compute_kernel = MethodType(lambda self, data: original(data), current)
            current.cv_init(X)
        elif hook in ('class_compute', 'training_method', 'known_psd', 'cv_init'):
            name = {'class_compute': '_compute_kernel', 'training_method': '_compute_training_kernel',
                    'known_psd': '_known_psd_kernel', 'cv_init': 'cv_init'}[hook]
            original = getattr(KernGDWD, name)
            def wrapper(self, *args, **kwargs):
                return original(self, *args, **kwargs)
            with patch.object(KernGDWD, name, wrapper):
                current.cv_init(X)
        elif hook == 'registry':
            original = pairwise.PAIRWISE_KERNEL_FUNCTIONS['rbf']
            with patch.dict(pairwise.PAIRWISE_KERNEL_FUNCTIONS,
                            {'rbf': lambda *a, **k: original(*a, **k)}):
                current.cv_init(X)
        else:
            owner, name = {'pairwise': (kernel_utils, 'pairwise_kernels'),
                           'guarded': (guarded, 'guarded_rbf'),
                           'sparse_dot': (extmath, 'safe_sparse_dot')}[hook]
            original = getattr(owner, name)
            with patch.object(owner, name, lambda *a, **k: original(*a, **k)):
                current.cv_init(X)
    assert marker(current) is None
    assert getattr(current, '_cv_rbf_self_generation', None) is None


@pytest.mark.parametrize('route', ['direct', 'linear', 'legacy'])
def test_unrelated_routes_remain_outside_self_policy(route):
    options = {'direct': {'rbf_computation': 'direct'},
               'linear': {'kernel': 'linear', 'kernel_kws': None},
               'legacy': {'solver_mode': 'legacy'}}[route]
    if route == 'legacy':
        with pytest.raises(TypeError, match='solver_mode'):
            model(**options)
        return
    current = model(**options)
    current.cv_init(features())
    assert marker(current) is None
    assert getattr(current, '_cv_rbf_self_generation', None) is None


def test_upfront_policy_uses_one_unchanged_guarded_self_evaluation():
    import sys
    from dwd._guarded_rbf import guarded_rbf
    code = guarded_rbf.__code__
    calls = []
    previous = sys.getprofile()
    def observe(frame, event, result):
        if event == 'call' and frame.f_code is code:
            calls.append((frame.f_locals['mode'], frame.f_locals['self_kernel']))
    try:
        sys.setprofile(observe)
        prepared = model().cv_init(features())
    finally:
        sys.setprofile(previous)
    assert sys.getprofile() is previous
    assert calls == [('norm_sum', True)]
    assert marker(prepared) == POLICY
    assert prepared.kernel_symmetry_correction_ == 0.
