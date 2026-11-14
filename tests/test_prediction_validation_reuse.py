"""Public prediction validation reuse preserves numerical and extension behavior.

The oracle disables only norm reuse and uses the installed package's unchanged
per-batch numerical core. These tiny correctness controls do not fit models.
These controls do not claim cross-environment or different-BLAS bit equality.
"""
import importlib
import pickle
import sys
import warnings
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn import config_context, get_config
from threadpoolctl import threadpool_limits

from dwd.gen_kern_dwd import KernGDWD
from dwd import _prediction_norms as norms
from dwd import _guarded_rbf as guarded
from dwd import kernel_utils as ku


@pytest.fixture(autouse=True)
def deterministic_small_controls():
    # Correctness-only, tiny matrices; no performance evidence is generated.
    before = np.geterr(), np.geterrcall()
    with threadpool_limits(limits=1), np.errstate(divide='warn', over='warn',
                                                invalid='warn', under='ignore'):
        yield
    assert np.geterr() == before[0]
    assert np.geterrcall() is before[1]


def manual_model():
    """Public query state, deliberately no fitted optimum/convergence claim."""
    rng = np.random.default_rng(351612)
    m = KernGDWD(kernel='rbf', kernel_kws={'gamma': .17}, prediction_batch_size=3)
    m._Xfit = rng.normal(size=(9, 4))
    m.dual_coef_ = np.linspace(-.4, .5, 9)[None, :]
    m.intercept_ = np.array([.125])
    m.classes_ = np.array([-3, 7])
    m.n_features_in_ = 4
    m.kernel_ = 'rbf'
    m.kernel_kws_ = {'gamma': .17}
    m.kernel_computation_ = 'guarded_norm_sum_rel_v2'
    m.prediction_precision_ = 'adaptive'
    m.affine_computation_ = 'joint'
    return m, rng.normal(size=(11, 4))


def exact(actual, expected):
    assert type(actual) is type(expected) is np.ndarray
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


def compare(m, q, **kwargs):
    before = pickle.dumps(m, protocol=5)
    with patch.object(norms, 'prepare_norm_kernel', return_value=None):
        expected = m.decision_function(q, **kwargs)
    actual = m.decision_function(q, **kwargs)
    exact(actual, expected)
    assert pickle.dumps(m, protocol=5) == before
    # predict has no per-call batch override; use its own corresponding oracle.
    with patch.object(norms, 'prepare_norm_kernel', return_value=None):
        labels = m.predict(q)
    exact(m.predict(q), labels)
    return actual


def count_pairwise(operation):
    count = 0
    code = norms._CHECK.__code__
    prior = sys.getprofile()
    def observe(frame, event, argument):
        nonlocal count
        if event == 'call' and frame.f_code is code:
            count += 1
    sys.setprofile(observe)
    try:
        result = operation()
    finally:
        sys.setprofile(prior)
    return result, count


def test_eligible_public_call_reduces_validation_without_state_change():
    m, q = manual_model()
    assert callable(norms.prepare_public_norm_kernel)
    assert norms._VALIDATION_SOURCE_OK and norms._validation_sources_unchanged()
    before = pickle.dumps(m, protocol=5)
    with patch.object(norms, 'prepare_norm_kernel', return_value=None):
        expected, old_count = count_pairwise(lambda: m.decision_function(q))
    result, new_count = count_pairwise(lambda: m.decision_function(q))
    exact(result, expected)
    assert old_count == 4
    assert new_count == 0
    assert pickle.dumps(m, protocol=5) == before


@pytest.mark.parametrize('layout', ['C', 'F', 'reverse', 'strided', 'readonly',
                                   'float32', 'int', 'list', 'subclass', 'csr'])
def test_input_representation_contract(layout):
    m, q = manual_model()
    class Subclass(np.ndarray):
        pass
    if layout == 'F': q = np.asfortranarray(q)
    elif layout == 'reverse': q = q[::-1]
    elif layout == 'strided': q = np.repeat(q, 2, axis=0)[::2]
    elif layout == 'readonly': q.setflags(write=False)
    elif layout == 'float32': q = q.astype(np.float32)
    elif layout == 'int': q = np.rint(q).astype(np.int64)
    elif layout == 'list': q = q.tolist()
    elif layout == 'subclass': q = q.view(Subclass)
    elif layout == 'csr': q = csr_matrix(q)
    compare(m, q)
    checked = ku.check_array(q, dtype=np.float64, accept_sparse='csr')
    eligible = norms._dense_validation_candidate(m._Xfit, checked, q,
                                                 m.kernel_computation_)
    assert eligible == (layout in ('C', 'readonly'))


@pytest.mark.parametrize('case', ['F_basis', 'strided_basis', 'near_duplicate',
                                'large_offset', 'tiny', 'zero_gamma', 'alias'])
def test_numerical_boundary_and_basis_storage(case):
    m, q = manual_model()
    if case == 'F_basis': m._Xfit = np.asfortranarray(m._Xfit)
    elif case == 'strided_basis': m._Xfit = np.repeat(m._Xfit, 2, axis=0)[::2]
    elif case == 'near_duplicate':
        q[:3] = m._Xfit[:3]
        q[1, 0] = np.nextafter(q[1, 0], np.inf)
    elif case == 'large_offset':
        m._Xfit += 1e8
        q += 1e8
    elif case == 'tiny':
        m._Xfit *= 1e-160
        q *= 1e-160
    elif case == 'zero_gamma': m.kernel_kws_ = {'gamma': 0.}
    elif case == 'alias': q = m._Xfit
    compare(m, q)


@pytest.mark.parametrize('size', [1, 2, 10, 11, None])
def test_batch_tail_and_unbatched_route(size):
    m, q = manual_model()
    compare(m, q, prediction_batch_size=size)


def test_mutations_are_revalidated_each_call():
    m, q = manual_model()
    first = compare(m, q)
    q[0] += .75
    second = compare(m, q)
    assert first.tobytes() != second.tobytes()
    m._Xfit[0] -= .5
    third = compare(m, q)
    assert second.tobytes() != third.tobytes()
    m.kernel_kws_['gamma'] = .31
    fourth = compare(m, q)
    assert third.tobytes() != fourth.tobytes()
    q[0, 0] = np.nan
    with pytest.raises(ValueError): m.decision_function(q)


def outcome(m, q, reference):
    events = []
    class Handler:
        def __call__(self, *args): events.append(('call', args))
        def write(self, message): events.append(('log', message))
    handler = Handler()
    prior_handler = np.geterrcall()
    np.seterrcall(handler)
    try:
        with warnings.catch_warnings(record=True) as caught, ExitStack() as stack:
            warnings.simplefilter('always')
            if reference:
                stack.enter_context(patch.object(norms, 'prepare_norm_kernel', return_value=None))
            try:
                a = m.decision_function(q)
                result = ('returned', a.dtype.str, a.shape, a.tobytes())
            except Exception as error:
                result = ('error', type(error), str(error))
            return result, [(type(w.message), str(w.message)) for w in caught], events
    finally:
        np.seterrcall(prior_handler)


@pytest.mark.parametrize('where,value', [('query', np.nan), ('query', np.inf),
                                        ('basis', np.nan), ('basis', np.inf),
                                        ('basis', 1e200)])
def test_invalid_and_overflowed_inputs_keep_original_outcome(where, value):
    m, q = manual_model()
    (q if where == 'query' else m._Xfit)[0, 0] = value
    assert outcome(m, q, False) == outcome(m, q, True)


@pytest.mark.parametrize('setting,policy', [('under','raise'), ('under','warn'),
    ('under','call'), ('under','log'), ('invalid','raise'), ('over','raise'), ('divide','raise')])
def test_error_policy_and_handler_are_preserved(setting, policy):
    m, q = manual_model()
    q *= 1e3
    with np.errstate(**{setting:policy}):
        before = np.geterr(), np.geterrcall()
        assert norms.prepare_norm_kernel(m, q, q) is None
        assert outcome(m, q, False) == outcome(m, q, True)
        assert np.geterr() == before[0] and np.geterrcall() is before[1]


@pytest.mark.parametrize('invalid', [False, True])
def test_assume_finite_is_preserved_and_declines_validation_reuse(invalid):
    m, q = manual_model()
    if invalid: q[0, 0] = np.nan
    before = get_config()
    with config_context(assume_finite=True):
        assert not norms._validation_sources_unchanged()
        assert outcome(m, q, False) == outcome(m, q, True)
        assert get_config()['assume_finite'] is True
    assert get_config() == before


@pytest.mark.parametrize('target', ['pairwise_check', 'pairwise_array', 'public_array',
                                   'finite_check', 'guarded'])
def test_replaced_live_validation_helpers_remain_observable(target):
    targets = {'pairwise_check': (guarded, 'check_pairwise_arrays'),
               'pairwise_array': (norms.pairwise, 'check_array'),
               'public_array': (ku, 'check_array'),
               'finite_check': (norms._validation_module, '_assert_all_finite'),
               'guarded': (guarded, 'guarded_rbf')}
    module, name = targets[target]
    original = getattr(module, name)
    counts = []
    for reference in (True, False):
        calls = []
        def wrapped(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)
        m, q = manual_model()
        with patch.object(module, name, wrapped), ExitStack() as stack:
            if reference:
                stack.enter_context(patch.object(norms, 'prepare_norm_kernel', return_value=None))
            value = m.decision_function(q)
            if reference: expected = value
            else: exact(value, expected)
        counts.append(len(calls))
    assert counts[0] == counts[1] > 0


def test_replacement_after_prepare_is_seen_by_each_batch():
    m, q = manual_model()
    compute = norms.prepare_public_norm_kernel(m, q, q)
    assert compute is not None
    calls = []
    original = guarded.check_pairwise_arrays
    def wrapped(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    with patch.object(guarded, 'check_pairwise_arrays', wrapped):
        for start in range(0, len(q), 3): compute(q[start:start+3])
    assert len(calls) == 4


def test_preimport_nonfunction_wrapper_declines_without_import_failure():
    original = norms.pairwise.check_array
    class Callable:
        def __call__(self, *args, **kwargs): return original(*args, **kwargs)
    try:
        with patch.object(norms.pairwise, 'check_array', Callable()):
            importlib.reload(norms)
            assert norms._VALIDATION_SOURCE_OK is False
            m, q = manual_model()
            compare(m, q)
    finally:
        importlib.reload(norms)
    assert norms._VALIDATION_SOURCE_OK


def test_equal_view_is_supported_but_equal_copy_is_not_a_proof():
    a = np.arange(12, dtype=float).reshape(3, 4)
    assert norms._same_dense_storage(a.view(), a)
    assert not norms._same_dense_storage(a.copy(), a)
    assert not norms._same_dense_storage(a[::-1], a)


def test_direct_prepare_and_three_argument_extension_contract():
    m, q = manual_model()
    calls = []
    original = norms.prepare_norm_kernel
    def wrapped(estimator, query, original_query):
        calls.append(1)
        return original(estimator, query, original_query)
    with patch.object(norms, 'prepare_norm_kernel', wrapped):
        m.decision_function(q)
    assert calls == [1]
    # Direct preparation did not observe public validation, so must not bypass it.
    q[0, 0] = np.nan
    with pytest.raises(ValueError):
        compute = original(m, q, q)
        compute(q[:3])

