"""Finite metadata guards for sharing feature-report products and splits.

No admission check may execute a replaced callback. Numerical and public
report coverage lives in the arithmetic and feature-batching test modules.
"""
from functools import cached_property
import math
import sys
import types

import numpy as np
import pytest

from dwd import _affine_scores as affine
from dwd import _compensated_residual as arithmetic
from dwd import _compiled_residual as native
from dwd import _kernel_scores as scores

from _feature_report_test_utils import (pins as _pins, same as _same,
    feature, support, contract as _contract_current, exact_contract, fresh_support)


@pytest.fixture(autouse=True)
def portable_and_policy(monkeypatch):
    original_geterr, original_geterrcall = np.geterr, np.geterrcall
    policy, callback = original_geterr().copy(), original_geterrcall()
    monkeypatch.setattr(native, '_ACCEL', None)
    yield
    assert original_geterr() == policy and original_geterrcall() is callback


def _pair_data():
    return (np.array([[1., .25, -.5], [.5, -.25, 1.]]),
            np.array([.5, .25, .125]), .25)


def _separate(module, matrix, beta, offset):
    raw = module.compensated_kernel_matvec(matrix, beta)
    joint = module.affine_scores(matrix, beta, offset, policy='compensated')
    return raw, joint


def _forbidden_fusion(*args, **kwargs):
    raise AssertionError('a replaced input boundary must take the original pair')


def _check_pair_fallback(monkeypatch, events, *, active_support=None):
    module = feature
    active_support = support if active_support is None else active_support
    data = _pair_data()
    before = _pins(data[:2])
    events.clear()
    assert not active_support.can_share_products(
        data[0], data[1], module.compensated_kernel_matvec, module.affine_scores)
    # Qualification itself must not invoke a replaced predicate/descriptor.
    assert events == []
    monkeypatch.setattr(active_support, '_fused_pair', _forbidden_fusion)
    expected = _separate(feature, *data)
    expected_events = list(events)
    events.clear()
    actual = module._feature_scores_and_margins(*data)
    _same(actual, expected)
    assert events == expected_events
    _same(expected, (np.array([.5, .3125]), np.array([.75, .5625])))
    assert _pins(data[:2]) == before
    return expected_events


def _copy_function(function):
    result = types.FunctionType(function.__code__, function.__globals__,
                                function.__name__, function.__defaults__,
                                function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def _replace_body(monkeypatch, function, events):
    """Change the body in place while preserving its actual original result."""
    assert type(function) is types.FunctionType and not function.__code__.co_freevars
    original = _copy_function(function)
    namespace = function.__globals__
    monkeypatch.setitem(namespace, '_prospective_original_call', original)
    monkeypatch.setitem(namespace, '_prospective_call_events', events)

    def changed(*args, **kwargs):
        _prospective_call_events.append('body')
        return _prospective_original_call(*args, **kwargs)

    monkeypatch.setattr(function, '__code__', changed.__code__)


def _plain_method(value):
    return value.__func__ if type(value) in (staticmethod, classmethod) else value


def _tiny_body(descriptor):
    if type(descriptor) is cached_property:
        return descriptor.func
    assert type(descriptor) is property, 'a new finfo layout needs explicit review'
    return descriptor.fget


FINFO_CASES = ['owner_alias', 'new_code', 'new_defaults', 'new_kwdefaults',
               'init_code', 'tiny_body_code', 'tiny_body_defaults',
               'tiny_descriptor', 'smallest_normal_descriptor']


def _change_finfo(monkeypatch, kind, events):
    cls = np.finfo
    original_new = _plain_method(vars(cls)['__new__'])
    # Read the real values before installing the callback. A guard must neither
    # populate this cache nor execute the newly installed descriptors.
    info = cls(float)
    tiny, smallest = info.tiny, info.smallest_normal
    state = dict(vars(info))
    if kind == 'owner_alias':
        owner = sys.modules[original_new.__module__]
        assert vars(owner)['finfo'] is cls
        monkeypatch.setattr(owner, 'finfo', object())
    elif kind == 'new_code':
        _replace_body(monkeypatch, original_new, events)
    elif kind == 'new_defaults':
        assert original_new.__defaults__ is None
        monkeypatch.setattr(original_new, '__defaults__', ())
    elif kind == 'new_kwdefaults':
        assert original_new.__kwdefaults__ is None
        monkeypatch.setattr(original_new, '__kwdefaults__', {'unused': object()})
    elif kind == 'init_code':
        _replace_body(monkeypatch, _plain_method(vars(cls)['_init']), events)
    elif kind.startswith('tiny_body_'):
        body = _tiny_body(vars(cls)['tiny'])
        if kind == 'tiny_body_code':
            _replace_body(monkeypatch, body, events)
        else:
            assert body.__defaults__ is None
            monkeypatch.setattr(body, '__defaults__', ())
    else:
        attribute, value = ('tiny', tiny) if kind == 'tiny_descriptor' else ('smallest_normal', smallest)

        def observe(instance):
            events.append(attribute)
            return value

        monkeypatch.setattr(cls, attribute, property(observe), raising=False)
    return info, state


class TestCommonFusionGuard:
    def test_ordinary_fusion_remains_enabled(self, monkeypatch):
        data = _pair_data()
        module = feature
        assert support.can_share_products(data[0], data[1], module.compensated_kernel_matvec, module.affine_scores)
        original, observed = support._fused_pair, []

        def observe(*args):
            observed.append(True)
            return original(*args)

        monkeypatch.setattr(support, '_fused_pair', observe)
        _same(module._feature_scores_and_margins(*data), _separate(feature, *data))
        assert observed == [True]

    @pytest.mark.parametrize('entry', ['finfo', 'iscomplexobj', 'issparse', 'zeros'])
    def test_preimport_input_wrappers_are_not_blessed_by_capture(self, monkeypatch, entry):
        owner = scores if entry == 'issparse' else np
        original, events = getattr(owner, entry), []

        def observe(*args, **kwargs):
            events.append(entry)
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, entry, observe)
        with fresh_support(monkeypatch) as loaded_support:
            assert events == [], 'capturing guard metadata must not probe the replacement'
            calls = _check_pair_fallback(monkeypatch, events, active_support=loaded_support)
        assert len(calls) == {'finfo': 2, 'iscomplexobj': 5, 'issparse': 2, 'zeros': 2}[entry]

    @pytest.mark.parametrize('kind', FINFO_CASES)
    def test_finfo_methods_and_descriptors_decline_without_probing(self, monkeypatch, kind):
        events = []
        info, state = _change_finfo(monkeypatch, kind, events)
        _check_pair_fallback(monkeypatch, events)
        # No replacement-induced entry is installed in the real cached record.
        assert vars(info).keys() == state.keys()
        assert all(vars(info)[key] is value for key, value in state.items())

    @pytest.mark.parametrize('token', ['code', 'defaults', 'kwdefaults'])
    def test_unchanged_iscomplexobj_dispatcher_does_not_hide_body_edit(self, monkeypatch, token):
        outer, events = np.iscomplexobj, []
        body = outer.__wrapped__
        if token == 'code':
            _replace_body(monkeypatch, body, events)
        elif token == 'defaults':
            assert body.__defaults__ is None
            monkeypatch.setattr(body, '__defaults__', ())
        else:
            assert body.__kwdefaults__ is None
            monkeypatch.setattr(body, '__kwdefaults__', {'unused': object()})
        assert np.iscomplexobj is outer
        calls = _check_pair_fallback(monkeypatch, events)
        assert len(calls) == (5 if token == 'code' else 0)

    @pytest.mark.parametrize('kind', ['code', 'defaults', 'owner_alias'])
    def test_imported_sparse_predicate_checks_its_actual_definition(self, monkeypatch, kind):
        function, events = scores.issparse, []
        if kind == 'code':
            _replace_body(monkeypatch, function, events)
        elif kind == 'defaults':
            assert function.__defaults__ is None
            monkeypatch.setattr(function, '__defaults__', ())
        else:
            owner = sys.modules[function.__module__]
            assert vars(owner)['issparse'] is function
            monkeypatch.setattr(owner, 'issparse', object())
        calls = _check_pair_fallback(monkeypatch, events)
        assert len(calls) == (2 if kind == 'code' else 0)

    @pytest.mark.parametrize('owner,alias', [(scores, 'np'), (affine, 'np'),
                                           (arithmetic, 'np'), (affine, 'math'),
                                           (arithmetic, 'math')],
                             ids=['scores-numpy', 'affine-numpy', 'arithmetic-numpy',
                                  'affine-math', 'arithmetic-math'])
    def test_common_module_alias_replacement_uses_original_pair(self, monkeypatch, owner, alias):
        original = np if alias == 'np' else math
        proxy = types.ModuleType(original.__name__)
        proxy.__dict__.update(vars(original))
        monkeypatch.setattr(owner, alias, proxy)
        _check_pair_fallback(monkeypatch, [])


def _contraction_data():
    features = np.arange(4 * 128, dtype=float).reshape(4, 128) / 1024 + .125
    alpha = np.array([.5, -.25, .125, -.0625])
    return features, alpha, -.5 * alpha


def _check_split_fallback(monkeypatch, family, events, data=None):
    data = _contraction_data() if data is None else data
    before = _pins(data)
    expected = exact_contract(family, *data)
    original, preparation = support.repeated_operand_splits, []

    def observe(*args):
        value = original(*args)
        preparation.append(value)
        assert value is None, 'unsupported callbacks must not produce a cached split'
        return value

    monkeypatch.setattr(support, 'repeated_operand_splits', observe)
    events.clear()
    actual = _contract_current(family, *data)
    observed = list(events)
    _same(actual, expected)
    assert _pins(data) == before
    # Local unsupported-input admission may return before the support function.
    assert all(value is None for value in preparation)
    return observed


def _check_positive_split(monkeypatch, family):
    data = _contraction_data()
    expected = exact_contract(family, *data)
    original, preparation = support.repeated_operand_splits, []

    def observe(*args):
        value = original(*args)
        assert value is not None
        preparation.append(len(value))
        return value

    monkeypatch.setattr(support, 'repeated_operand_splits', observe)
    _same(_contract_current(family, *data), expected)
    assert preparation == [1 if family == 'coefficients' else 3]

class TestSplitCompatibility:
    @pytest.mark.parametrize('family', ['coefficients', 'gradient'])
    def test_ordinary_reuse_is_enabled(self, monkeypatch, family):
        _check_positive_split(monkeypatch, family)

    @pytest.mark.parametrize('family', ['coefficients', 'gradient'])
    @pytest.mark.parametrize('kind', ['owner_alias', 'new_code', 'tiny_body_code', 'tiny_descriptor'])
    def test_finfo_boundary_also_protects_contraction_reuse(self, monkeypatch, family, kind):
        events = []
        _change_finfo(monkeypatch, kind, events)
        _check_split_fallback(monkeypatch, family, events)

    @pytest.mark.parametrize('family', ['coefficients', 'gradient'])
    @pytest.mark.parametrize('method', ['__init__', '__enter__', '__exit__'])
    def test_errstate_method_body_is_an_observable_boundary(self, monkeypatch, family, method):
        events = []
        _replace_body(monkeypatch, vars(np.errstate)[method], events)
        calls = _check_split_fallback(monkeypatch, family, events)
        assert calls

    def test_public_gradient_phase_rechecks_after_V_grad(self, monkeypatch):
        """A real public-report callback changes a later contraction boundary."""
        features = np.zeros((4, 128))
        labels = np.array([-1., 1., -1., 1.])
        alpha = np.array([.5, -.25, .125, -.0625])
        kwargs = dict(checked_scores=np.zeros(4), optimizer_scores=np.zeros(4),
                      checked_objective=1., optimizer_objective=1.)
        before = _pins((features, labels, alpha, kwargs['checked_scores'], kwargs['optimizer_scores']))

        def run(module, observe_preparation):
            calls, preparation = [], []
            original_v, original_finfo = module.V_grad, np.finfo
            with monkeypatch.context() as scoped:
                def finfo(*args, **kwargs):
                    calls.append('finfo')
                    return original_finfo(*args, **kwargs)

                def derivative(*args, **kwargs):
                    calls.append('V_grad')
                    result = original_v(*args, **kwargs)
                    scoped.setattr(np, 'finfo', finfo)
                    return result

                scoped.setattr(module, 'V_grad', derivative)
                if observe_preparation:
                    original = support.repeated_operand_splits

                    def observe(*args):
                        value = original(*args)
                        preparation.append(value is not None)
                        return value

                    scoped.setattr(support, 'repeated_operand_splits', observe)
                result = module.linear_feature_diagnostics(features, labels, alpha, 0., 1., 1., **kwargs)
            return result, calls, preparation

        actual, calls, preparation = run(feature, True)
        assert calls[0] == 'V_grad' and 'finfo' in calls
        assert actual[0]['final_objective'] == 1.
        assert actual[0]['rkhs_gradient_norm'] == 0.
        assert actual[0]['gradient_inf_norm'] == 0.
        assert preparation == [True, False]
        assert _pins((features, labels, alpha, kwargs['checked_scores'], kwargs['optimizer_scores'])) == before


class TestNarrowSplitGuard:
    @pytest.mark.parametrize('family', ['coefficients', 'gradient'])
    @pytest.mark.parametrize('entry', ['asarray', 'frexp', 'ldexp', 'isfinite',
                                      'broadcast_to', 'empty', 'any', 'abs'])
    def test_reachable_numpy_entry_preserves_callback_and_declines_reuse(self, monkeypatch, family, entry):
        data, events = _contraction_data(), []
        original = getattr(np, entry)

        def observe(*args, **kwargs):
            events.append(entry)
            return original(*args, **kwargs)

        monkeypatch.setattr(np, entry, observe)
        assert _check_split_fallback(monkeypatch, family, events, data)

    @pytest.mark.parametrize('family', ['coefficients', 'gradient'])
    def test_replaced_local_numpy_alias_does_not_prepare_splits(self, monkeypatch, family):
        events, data = [], _contraction_data()
        proxy = types.ModuleType('numpy')
        proxy.__dict__.update(vars(np))
        original = np.frexp

        def observe(*args, **kwargs):
            events.append('frexp')
            return original(*args, **kwargs)

        proxy.frexp = observe
        monkeypatch.setattr(feature, 'np', proxy)
        assert _check_split_fallback(monkeypatch, family, events, data)

    @pytest.mark.parametrize('family', ['coefficients', 'gradient'])
    def test_private_feature_subclass_retains_slicing_callbacks(self, monkeypatch, family):
        events = []

        class ObservableFeatures(np.ndarray):
            def __getitem__(self, key):
                events.append(repr(key))
                return super().__getitem__(key)

        features, alpha, z = _contraction_data()
        data = features.view(ObservableFeatures), alpha, z
        assert _check_split_fallback(monkeypatch, family, events, data)

    def test_prepared_vector_subclass_is_declined_without_array_callbacks(self):
        events = []

        class ObservableVector(np.ndarray):
            def __array_ufunc__(self, *args, **kwargs):
                events.append('ufunc')
                raise AssertionError('admission must not evaluate a vector subclass')

        vector = np.array([.25, -.5, .125, 1.]).view(ObservableVector)
        assert support.repeated_operand_splits((vector,), arithmetic._products) is None
        assert events == []

    @pytest.mark.parametrize('family', ['coefficients', 'gradient'])
    @pytest.mark.parametrize('irrelevant', ['affine', 'zeros'])
    def test_irrelevant_fusion_replacement_does_not_disable_split_reuse(self, monkeypatch, family, irrelevant):
        def forbidden(*args, **kwargs):
            raise AssertionError('this fusion entry is not part of a feature contraction')

        if irrelevant == 'affine':
            monkeypatch.setattr(affine, '_finish_row', forbidden)
        else:
            monkeypatch.setattr(np, 'zeros', forbidden)
        _check_positive_split(monkeypatch, family)

@pytest.mark.parametrize('entry,count', [('iscomplexobj', 5), ('zeros', 2)])
def test_preimport_unknown_callable_is_not_probed(monkeypatch, entry, count):
    original, events = getattr(np, entry), []

    class Callable:
        def __call__(self, *args, **kwargs):
            events.append(entry)
            return original(*args, **kwargs)

        def __getattr__(self, name):
            raise AssertionError('metadata admission probed an unknown callable')

    monkeypatch.setattr(np, entry, Callable())
    with fresh_support(monkeypatch) as loaded_support:
        assert events == []
        observed = _check_pair_fallback(monkeypatch, events, active_support=loaded_support)
    assert len(observed) == count
