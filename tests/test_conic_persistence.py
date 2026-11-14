"""Persistence, state ownership, and re-solving for optional conic estimators."""
import copy
import importlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import time
import types

import joblib
from joblib import Parallel, delayed, parallel_config
try:
    import cloudpickle
except ImportError:
    try:
        from joblib.externals import cloudpickle
    except ImportError:
        cloudpickle = None
from joblib.externals.loky import reusable_executor
import numpy as np
import psutil
import pytest
from scipy import sparse
from sklearn import config_context
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import StratifiedKFold, cross_validate

cp = pytest.importorskip('cvxpy', exc_type=ImportError)
from dwd.socp_dwd import DWD, solve_dwd_socp
from dwd.svm import SVM, solve_svm


SERIALIZERS = ('pickle', 'joblib', 'cloudpickle')
CASES = (('DWD', 'CLARABEL'), ('DWD', 'SCS'),
         ('SVM', 'CLARABEL'), ('SVM', 'SCS'), ('SVM', 'OSQP'),
         ('SVM', 'SCIPY'), ('SVM', 'HIGHS'))
CLASSES = {'DWD': DWD, 'SVM': SVM}
from dwd._conic_persistence import PersistentProblem

ProblemBase = PersistentProblem


class UserProblemSlots(ProblemBase):
    __slots__ = ('user_slot',)


def _freeze(value):
    """Observe diagnostic data, including pybind native solver-info fields."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else {'nonfinite': repr(number)}
    if isinstance(value, np.generic):
        return _freeze(value.item())
    if isinstance(value, np.ndarray):
        return _freeze(value.tolist())
    if sparse.issparse(value):
        return dict(shape=list(value.shape), dense=_freeze(value.toarray()))
    if isinstance(value, dict):
        return {str(key): _freeze(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_freeze(item) for item in value]
    if isinstance(value, types.SimpleNamespace):
        fields = vars(value)
        return dict(data={name: _freeze(item) for name, item in fields.items()
                          if not callable(item)},
                    excluded_runtime_callable_fields=sorted(
                        name for name, item in fields.items() if callable(item)))
    # Solver info wrappers expose public scalar fields, sometimes without a
    # Python __dict__. Do not call methods or infer missing values.
    fields = {}
    excluded = []
    for name in dir(value):
        if name.startswith('__'):
            continue
        item = getattr(value, name)
        if callable(item):
            excluded.append(name)
        else:
            fields[name] = _freeze(item)
    if not fields:
        raise TypeError('Unobserved diagnostic type: ' + type(value).__qualname__)
    return dict(data=fields, excluded_runtime_callable_fields=sorted(excluded))


def _diagnostic_values(value):
    """Compare native and portable diagnostic data without representation metadata."""
    if isinstance(value, dict):
        return {key: _diagnostic_values(item) for key, item in value.items()
                if not key.endswith('_time')
                and key not in ('compilation_time', '_dwd_source_type',
                                'excluded_runtime_callable_fields')
                and not key.startswith('_pybind11_conduit_')}
    if isinstance(value, list):
        return [_diagnostic_values(item) for item in value]
    return value


def _problem_observation(problem):
    assert isinstance(problem, cp.Problem)
    metrics = problem.size_metrics
    stats = problem.solver_stats
    solution = problem.solution
    return dict(
        status=problem.status, value=_freeze(problem.value),
        variables=[_freeze(v.value) for v in problem.variables()],
        parameters=[_freeze(v.value) for v in problem.parameters()],
        duals=[_freeze(c.dual_value) for c in problem.constraints],
        identity_order=dict(variables=[str(v.id) for v in problem.variables()],
                            constraints=[str(c.id) for c in problem.constraints]),
        size={name: _freeze(getattr(metrics, name)) for name in
              ('num_scalar_variables', 'num_scalar_data', 'num_scalar_eq_constr',
               'num_scalar_leq_constr', 'max_data_dimension', 'max_big_small_squared')},
        solver_stats=_freeze(vars(stats)) if stats is not None else None,
        solution=(dict(status=solution.status, opt_val=_freeze(solution.opt_val),
                       primal_vars=_freeze(solution.primal_vars),
                       dual_vars=_freeze(solution.dual_vars), attr=_freeze(solution.attr))
                  if solution is not None else None),
        compilation_time=_freeze(problem.compilation_time),
    )


def _model_observation(model, query):
    result = dict(prediction=model.predict(query).tolist(),
                  decision=_freeze(model.decision_function(query)),
                  coefficient=_freeze(model.coef_), intercept=_freeze(model.intercept_),
                  classes=model.classes_.tolist(), n_features=int(model.n_features_in_),
                  solver_status=model.solver_status_, problem=_problem_observation(model.problem_))
    for name in ('C_', 'eta_', 'd_'):
        if hasattr(model, name):
            result[name] = _freeze(getattr(model, name))
    if isinstance(model, DWD):
        result['direction'] = _freeze(model.direction)
    return result


def _guard(model):
    problem = model.problem_ if hasattr(model, 'problem_') else model
    observed = _problem_observation(problem)
    state = vars(problem).copy()
    cache = getattr(problem, '_solver_cache', None)
    cache_items = dict(cache) if isinstance(cache, dict) else None
    model_state = vars(model).copy() if model is not problem else None
    # Check stored diagnostic identities, including excluded native methods.
    # Native descriptor access itself can create a new bound-method object;
    # only stored dictionaries provide an identity that can be compared.
    nodes, seen = [], set()
    def remember(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, dict):
            fields = value
        elif hasattr(value, '__dict__') and not isinstance(value, type):
            fields = vars(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                remember(item)
            return
        else:
            return
        nodes.append((fields, dict(fields)))
        for item in fields.values():
            if not callable(item):
                remember(item)
    remember(problem.solver_stats)
    remember(problem.solution)
    return problem, state, cache, cache_items, model_state, observed, nodes


def _assert_unchanged(model, guard):
    problem, state, cache, cache_items, model_state, observed, nodes = guard
    assert vars(problem).keys() == state.keys()
    assert all(vars(problem)[key] is value for key, value in state.items())
    assert getattr(problem, '_solver_cache', None) is cache
    if cache_items is not None:
        assert cache.keys() == cache_items.keys()
        assert all(cache[key] is value for key, value in cache_items.items())
    if model_state is not None:
        assert model.problem_ is problem
        assert vars(model).keys() == model_state.keys()
        assert all(vars(model)[key] is value for key, value in model_state.items())
    for fields, old_fields in nodes:
        assert fields.keys() == old_fields.keys()
        assert all(fields[key] is value for key, value in old_fields.items())
    assert _problem_observation(problem) == observed


def _dumps(value, serializer):
    if serializer == 'pickle':
        return pickle.dumps(value, protocol=5)
    if serializer == 'cloudpickle':
        if cloudpickle is None:
            pytest.skip('Optional cloudpickle serializer is unavailable')
        return cloudpickle.dumps(value, protocol=5)
    stream = io.BytesIO()
    joblib.dump(value, stream, compress=0, protocol=5)
    return stream.getvalue()


def _loads(raw, serializer):
    return joblib.load(io.BytesIO(raw)) if serializer == 'joblib' else pickle.loads(raw)


def _transfer(value, serializer, source):
    guard = _guard(source)
    try:
        return _loads(_dumps(value, serializer), serializer)
    finally:
        _assert_unchanged(source, guard)


def _solver_options(solver):
    if solver == 'default':
        return None
    if solver not in cp.installed_solvers():
        pytest.skip('Optional solver unavailable: ' + solver)
    if solver == 'CLARABEL':
        return dict(solver=solver, verbose=False, max_iter=250,
                    tol_gap_abs=1e-8, tol_feas=1e-8)
    if solver == 'SCS':
        return dict(solver=solver, verbose=False, eps=1e-5, max_iters=10000)
    if solver == 'OSQP':
        return dict(solver=solver, verbose=False, eps_abs=1e-7, eps_rel=1e-7,
                    max_iter=10000)
    if solver == 'HIGHS':
        return dict(solver=solver, verbose=False, threads=1)
    if solver == 'SCIPY':
        return dict(solver=solver, verbose=False, scipy_options={'threads': 1})
    return dict(solver=solver, verbose=False)


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(461104)
    X = rng.normal(scale=.4, size=(16, 5))
    labels = np.tile(np.array(['left', 'right']), 8)
    X[:, 0] += np.where(labels == 'right', .8, -.8)
    query = rng.normal(scale=.5, size=(9, 5))
    return X, labels, query


@pytest.fixture(scope='module')
def models(data):
    cache = {}
    def get(family, solver='CLARABEL', storage='dense', penalty=None):
        key = family, solver, storage, penalty
        if key not in cache:
            X, y, _ = data
            features = sparse.csr_matrix(X) if storage == 'csr' else X
            options = _solver_options(solver)
            saved_options = copy.deepcopy(options)
            C = penalty if penalty is not None else (.3 if family == 'DWD' else .05)
            model = CLASSES[family](C=C, solver_kws=options)
            with config_context(enable_metadata_routing=True):
                model.set_score_request(sample_weight='evaluation_weight')
            model.fit(features, y)
            assert options == saved_options
            assert model.solver_kws == saved_options
            cache[key] = model
        return cache[key]
    return get


@pytest.mark.parametrize('family', ('DWD', 'SVM'))
@pytest.mark.parametrize('serializer', SERIALIZERS)
def test_unfitted_constructor_and_score_request_roundtrip(family, serializer, data):
    model = CLASSES[family](C=.3, solver_kws=_solver_options('CLARABEL'))
    with config_context(enable_metadata_routing=True):
        model.set_score_request(sample_weight='evaluation_weight')
    routing = repr(model.get_metadata_routing())
    state = vars(model).copy()
    restored = _loads(_dumps(model, serializer), serializer)
    assert type(restored) is type(model)
    assert restored.get_params(deep=False) == model.get_params(deep=False)
    assert repr(restored.get_metadata_routing()) == routing
    assert 'problem_' not in vars(restored) and 'coef_' not in vars(restored)
    with pytest.raises(NotFittedError):
        restored.predict(data[2])
    assert vars(model).keys() == state.keys()
    assert all(vars(model)[key] is value for key, value in state.items())


@pytest.mark.parametrize('family,solver', CASES)
@pytest.mark.parametrize('storage', ('dense', 'csr'))
@pytest.mark.parametrize('serializer', SERIALIZERS)
def test_fitted_bundle_preserves_problem_graph_and_all_diagnostics(family, solver, storage, serializer, models, data):
    model = models(family, solver, storage)
    problem = model.problem_
    expected = _model_observation(model, data[2])
    aliases = dict(problem=problem, variables=problem.variables(),
                   constraints=problem.constraints, parameters=problem.parameters(),
                   coefficient=model.coef_, solver_stats=problem.solver_stats,
                   solution=problem.solution)
    aliases['cycle'] = aliases
    restored = _transfer((model, problem, aliases), serializer, model)
    copied, copied_problem, copied_aliases = restored
    assert type(copied) is type(model)
    assert copied.problem_ is copied_problem is copied_aliases['problem']
    assert isinstance(copied_problem, cp.Problem)
    assert copied_aliases['cycle'] is copied_aliases
    # Joblib's ndarray storage need not retain ndarray alias identity.
    np.testing.assert_array_equal(copied_aliases['coefficient'], copied.coef_)
    if serializer != 'joblib':
        assert copied_aliases['coefficient'] is copied.coef_
    assert copied_aliases['solver_stats'] is copied_problem.solver_stats
    assert copied_aliases['solution'] is copied_problem.solution
    if (problem.solver_stats.extra_stats is
            problem.solution.attr.get('solver_specific_stats')):
        assert (copied_problem.solver_stats.extra_stats is
                copied_problem.solution.attr.get('solver_specific_stats'))
    for label, actual in (('variables', copied_problem.variables()),
                          ('constraints', copied_problem.constraints),
                          ('parameters', copied_problem.parameters())):
        assert len(actual) == len(copied_aliases[label])
        assert all(a is b for a, b in zip(actual, copied_aliases[label]))
    assert copied.get_params(deep=False) == model.get_params(deep=False)
    assert repr(copied.get_metadata_routing()) == repr(model.get_metadata_routing())
    assert _model_observation(copied, data[2]) == expected
    if solver == 'SCS':
        assert _freeze(copied_problem._solver_cache) == _freeze(problem._solver_cache)
    elif solver in ('CLARABEL', 'OSQP', 'HIGHS'):
        assert solver not in copied_problem._solver_cache


@pytest.mark.parametrize('family,solver', CASES)
def test_standalone_public_solver_problem_roundtrip(family, solver, data):
    options = _solver_options(solver)
    saved_options = copy.deepcopy(options)
    function = solve_dwd_socp if family == 'DWD' else solve_svm
    result = function(data[0], data[1], C=.3 if family == 'DWD' else .05,
                      solver_kws=options)
    problem = result[-1]
    expected = _problem_observation(problem)
    restored = _transfer(problem, 'pickle', problem)
    assert options == saved_options
    assert isinstance(restored, cp.Problem)
    assert _problem_observation(restored) == expected


@pytest.mark.parametrize('family,solver', CASES)
@pytest.mark.parametrize('serializer', SERIALIZERS)
def test_solver_extra_statistics_serialize_independently(family, solver, serializer, models):
    model = models(family, solver)
    extra = model.problem_.solver_stats.extra_stats
    expected = _freeze(extra)
    guard = _guard(model)
    try:
        restored = _loads(_dumps(extra, serializer), serializer)
        assert _freeze(restored) == expected
    finally:
        _assert_unchanged(model, guard)


@pytest.mark.parametrize('family,solver', (('DWD', 'CLARABEL'), ('SVM', 'CLARABEL'),
                                        ('DWD', 'SCS'), ('SVM', 'SCS'),
                                        ('SVM', 'OSQP'), ('SVM', 'HIGHS'),
                                        ('SVM', 'SCIPY')))
@pytest.mark.parametrize('serializer', SERIALIZERS)
def test_restored_problem_can_resolve_and_serialize_again(family, solver, serializer, models, data):
    model = models(family, solver)
    restored = _transfer(model, serializer, model)
    expected_predictions = restored.predict(data[2]).copy()
    expected_coef = restored.coef_.copy()
    old_value = float(restored.problem_.value)
    rtol, atol, feasibility = {
        'CLARABEL': (2e-6, 2e-7, 2e-6), 'SCS': (2e-4, 2e-5, 5e-4),
        'OSQP': (2e-5, 2e-6, 2e-5), 'HIGHS': (2e-6, 2e-7, 2e-6),
        'SCIPY': (2e-6, 2e-7, 2e-6),
    }[solver]
    for warm_start in (False, True):
        options = dict(restored.solver_kws, warm_start=warm_start)
        value = restored.problem_.solve(**options)
        assert restored.problem_.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        assert np.isfinite(value)
        np.testing.assert_allclose(value, old_value, rtol=rtol, atol=atol)
        for variable in restored.problem_.variables():
            assert variable.value is not None and np.isfinite(variable.value).all()
        violations = [float(np.max(np.asarray(c.violation()))) for c in restored.problem_.constraints]
        assert not violations or max(violations) <= feasibility
    # Explicitly solving the diagnostic Problem does not refit the estimator.
    np.testing.assert_array_equal(restored.coef_, expected_coef)
    np.testing.assert_array_equal(restored.predict(data[2]), expected_predictions)
    expected = _model_observation(restored, data[2])
    again = _loads(_dumps(restored, serializer), serializer)
    assert _model_observation(again, data[2]) == expected


@pytest.mark.parametrize('family', ('DWD', 'SVM'))
def test_default_solver_and_fitted_auto_penalty_survive(family, models, data):
    model = models(family, 'default', penalty='auto' if family == 'DWD' else .05)
    expected = _model_observation(model, data[2])
    restored = _transfer(model, 'pickle', model)
    assert restored.solver_kws is None
    assert restored.C == model.C
    assert _model_observation(restored, data[2]) == expected


@pytest.mark.parametrize('family', ('DWD', 'SVM'))
def test_clone_and_failed_refit_keep_constructor_contract(family, models, data):
    model = models(family)
    duplicate = clone(model)
    assert type(duplicate) is type(model)
    assert duplicate.get_params(deep=False) == model.get_params(deep=False)
    assert repr(duplicate.get_metadata_routing()) == repr(model.get_metadata_routing())
    assert not any(k.endswith('_') and not k.startswith('_') for k in vars(duplicate))
    restored = _transfer(model, 'pickle', model)
    restored.set_params(C=-1)
    with pytest.raises(ValueError):
        restored.fit(data[0], data[1])
    assert restored.C == -1
    assert not any(k.endswith('_') and not k.startswith('_') for k in vars(restored))
    assert restored.solver_kws == model.solver_kws
    with pytest.raises(NotFittedError):
        restored.predict(data[2])
    restored.set_params(C=model.C).fit(data[0], data[1])
    np.testing.assert_allclose(restored.decision_function(data[2]),
                               model.decision_function(data[2]), rtol=2e-6, atol=2e-7)


class AnnotatedDWD(DWD):
    def __init__(self, C=.3, solver_kws=None, annotation='user annotation'):
        super().__init__(C=C, solver_kws=solver_kws)
        self.annotation = annotation


class AnnotatedSVM(SVM):
    def __init__(self, C=.05, solver_kws=None, annotation='user annotation'):
        super().__init__(C=C, solver_kws=solver_kws)
        self.annotation = annotation


@pytest.fixture(scope='module')
def subclass_models(data):
    cache = {}
    def get(family):
        if family not in cache:
            cls = AnnotatedDWD if family == 'DWD' else AnnotatedSVM
            model = cls(solver_kws=_solver_options('CLARABEL')).fit(data[0], data[1])
            model.user_state = {'problem': model.problem_, 'coef': model.coef_, 'tokens': ['a', 2]}
            model.user_state['cycle'] = model.user_state
            cache[family] = model
        return cache[family]
    return get


@pytest.mark.parametrize('family', ('DWD', 'SVM'))
@pytest.mark.parametrize('serializer', SERIALIZERS)
def test_legitimate_subclass_state_and_cycles_survive(family, serializer, subclass_models, data):
    model = subclass_models(family)
    expected = _model_observation(model, data[2])
    restored = _transfer(model, serializer, model)
    assert type(restored) is type(model)
    assert restored.annotation == model.annotation
    assert restored.user_state['problem'] is restored.problem_
    np.testing.assert_array_equal(restored.user_state['coef'], restored.coef_)
    if serializer != 'joblib':
        assert restored.user_state['coef'] is restored.coef_
    assert restored.user_state['cycle'] is restored.user_state
    assert restored.user_state['tokens'] == model.user_state['tokens']
    assert _model_observation(restored, data[2]) == expected


class UserSerializationFailure:
    def __reduce__(self):
        raise RuntimeError('intentional user-state refusal')


@pytest.mark.parametrize('family', ('DWD', 'SVM'))
@pytest.mark.parametrize('serializer', SERIALIZERS)
def test_unrelated_user_serialization_error_propagates_without_mutation(family, serializer, models):
    model = copy.copy(models(family))
    # Exercise both an outer user object and user state owned by the model.
    payload = (UserSerializationFailure(), model)
    guard = _guard(model)
    with pytest.raises(RuntimeError, match='intentional user-state refusal'):
        _dumps(payload, serializer)
    _assert_unchanged(model, guard)
    model.user_payload = UserSerializationFailure()
    guard = _guard(model)
    with pytest.raises(RuntimeError, match='intentional user-state refusal'):
        _dumps(model, serializer)
    _assert_unchanged(model, guard)


@pytest.mark.parametrize('family', ('DWD', 'SVM'))
def test_unknown_transient_solver_cache_entry_is_not_serialized(family, models, data):
    model = copy.copy(models(family, 'SCS'))
    original_problem = model.problem_
    problem = object.__new__(type(original_problem))
    problem.__dict__.update(vars(original_problem))
    problem._solver_cache = {'future_solver_name': UserSerializationFailure()}
    model.problem_ = problem
    guard = _guard(model)
    expected = _model_observation(model, data[2])
    restored = _loads(_dumps(model, 'pickle'), 'pickle')
    assert isinstance(restored.problem_, cp.Problem)
    assert restored.problem_._solver_cache == {}
    assert _model_observation(restored, data[2]) == expected

    _assert_unchanged(model, guard)


@pytest.mark.parametrize('family', ('DWD', 'SVM'))
@pytest.mark.parametrize('serializer', SERIALIZERS)
def test_problem_subclass_slots_and_cyclic_state_survive(family, serializer, models):
    original = models(family).problem_
    problem = object.__new__(UserProblemSlots)
    problem.__dict__.update(vars(original))
    problem.user_slot = {'note': ['caller', 3], 'problem': problem}
    slot = problem.user_slot
    expected = _problem_observation(problem)
    restored = _transfer(problem, serializer, problem)
    assert problem.user_slot is slot and slot['problem'] is problem
    assert type(restored) is UserProblemSlots
    assert restored.user_slot['note'] == ['caller', 3]
    assert restored.user_slot['problem'] is restored
    assert _problem_observation(restored) == expected


@pytest.mark.parametrize('family', ('DWD', 'SVM'))
@pytest.mark.parametrize('layout', ('missing', 'nonmapping'))
def test_private_cache_layout_does_not_mutate_source(family, layout, models):
    original = models(family).problem_
    problem = object.__new__(type(original))
    problem.__dict__.update(vars(original))
    if layout == 'missing':
        del problem.__dict__['_solver_cache']
    else:
        problem._solver_cache = UserSerializationFailure()
    guard = _guard(problem)
    expected = _problem_observation(problem)
    restored = _loads(_dumps(problem, 'pickle'), 'pickle')
    assert _problem_observation(restored) == expected
    if layout == 'missing':
        assert '_solver_cache' not in vars(restored)
    else:
        assert restored._solver_cache == {}

    _assert_unchanged(problem, guard)


class OpaqueUserInfo:
    __slots__ = ('value',)

    def __init__(self):
        self.value = 7.

    def __reduce__(self):
        raise RuntimeError('intentional opaque-diagnostic refusal')


def test_unknown_user_diagnostic_and_private_callback_are_not_rewritten():
    opaque = OpaqueUserInfo()
    data = np.array([1., 2.])
    namespace = types.SimpleNamespace(opaque=opaque, data=data,
                                     _user_callback=opaque.__reduce__)
    namespace.cycle = namespace
    helper = importlib.import_module('dwd._conic_persistence')
    result = helper._portable_extra_stats(namespace)
    assert result is namespace
    assert result.opaque is opaque and result.data is data
    assert result.cycle is result
    assert result._user_callback is namespace._user_callback
    with pytest.raises(RuntimeError, match='intentional opaque-diagnostic refusal'):
        _dumps(result, 'pickle')


@pytest.mark.parametrize('order', ('namespace_alias_first', 'outer_alias_first'))
@pytest.mark.parametrize('serializer', SERIALIZERS)
def test_real_native_bridge_alias_order_and_cycles(order, serializer, models):
    model = models('SVM', 'OSQP')
    entry = model.problem_._solver_cache.get('OSQP')
    if not isinstance(entry, tuple) or len(entry) < 3 or not hasattr(entry[2], 'info'):
        pytest.skip('Installed OSQP does not expose a cached native information record')
    raw_info = entry[2].info
    names = sorted(name for name in dir(raw_info)
                   if name.startswith('_pybind11_conduit_') and name.endswith('_'))
    if not names:
        pytest.skip('Installed OSQP does not expose a native conduit bridge')
    name = names[0]
    native_bridge = getattr(raw_info, name)
    assert callable(native_bridge) and native_bridge.__name__ == name
    native_owner = native_bridge.__self__
    native_values = _freeze(native_owner)
    namespace = types.SimpleNamespace(alias=native_bridge)
    setattr(namespace, name, native_bridge)
    namespace.info = native_owner
    if order == 'outer_alias_first':
        graph = {'early_alias': native_bridge, 'namespace': namespace, 'owner': native_owner}
    else:
        graph = {'namespace': namespace, 'early_alias': native_bridge, 'owner': native_owner}
    graph['cycle'] = graph
    tuple_cycle = (namespace, [])
    tuple_cycle[1].append(tuple_cycle)
    graph['tuple_cycle'] = tuple_cycle
    original_graph, original_namespace = dict(graph), vars(namespace).copy()
    guard = _guard(model)
    try:
        helper = importlib.import_module('dwd._conic_persistence')
        converted = helper._portable_extra_stats(graph)
        restored = _loads(_dumps(converted, serializer), serializer)
        reference = restored['early_alias']
        assert isinstance(reference, helper.RuntimeMethodReference)
        assert not callable(reference) and reference.method_name == name
        assert reference is restored['namespace'].alias
        assert reference is getattr(restored['namespace'], name)
        assert restored['owner'] is restored['namespace'].info
        assert restored['cycle'] is restored
        assert restored['tuple_cycle'][0] is restored['namespace']
        assert restored['tuple_cycle'][1][0] is restored['tuple_cycle']
        assert _diagnostic_values(_freeze(restored['owner'])) == _diagnostic_values(native_values)
    finally:
        _assert_unchanged(model, guard)
        assert graph.keys() == original_graph.keys()
        assert all(graph[key] is value for key, value in original_graph.items())
        assert vars(namespace).keys() == original_namespace.keys()
        assert all(vars(namespace)[key] is value for key, value in original_namespace.items())
        assert _freeze(native_owner) == native_values


def _worker_predict(model, query, directory):
    actor = psutil.Process()
    identity = dict(pid=actor.pid, create_time=actor.create_time())
    path = Path(directory)
    with (path / f'{actor.pid}.json').open('x', encoding='utf-8') as stream:
        json.dump(identity, stream)
    deadline = time.monotonic() + 45
    while len(list(path.glob('*.json'))) < 2:
        if time.monotonic() >= deadline:
            raise AssertionError('Two workers did not reach the bounded transport barrier')
        time.sleep(.02)
    return identity, _model_observation(model, query)


def _close_own_executor():
    executor = reusable_executor._executor
    if executor is not None:
        executor.shutdown(wait=True, kill_workers=False)


@pytest.mark.parametrize('family', ('DWD', 'SVM'))
def test_real_two_process_fitted_model_transport(family, models, data, tmp_path):
    model = models(family)
    expected = _model_observation(model, data[2])
    guard = _guard(model)
    directory = tmp_path / 'parallel-barrier'
    directory.mkdir()
    identities = []
    try:
        with parallel_config(backend='loky', n_jobs=2, inner_max_num_threads=1):
            rows = Parallel()(delayed(_worker_predict)(model, data[2], str(directory))
                              for _ in range(2))
        identities = [row[0] for row in rows]
        assert len({row['pid'] for row in identities}) == 2
        assert all(row['pid'] != os.getpid() for row in identities)
        assert all(row[1] == expected for row in rows)
    finally:
        _close_own_executor()
        _assert_unchanged(model, guard)
    for identity in identities:
        try:
            actor = psutil.Process(identity['pid'])
            assert actor.create_time() != identity['create_time']
        except psutil.NoSuchProcess:
            pass


@pytest.mark.parametrize('family', ('DWD', 'SVM'))
def test_parallel_cross_validate_returns_real_fitted_estimators(family, data):
    model = CLASSES[family](C=.3 if family == 'DWD' else .05,
                            solver_kws=_solver_options('CLARABEL'))
    splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=17)
    try:
        with parallel_config(backend='loky', n_jobs=2, inner_max_num_threads=1):
            result = cross_validate(model, data[0], data[1], cv=splitter,
                                    scoring='accuracy', return_estimator=True,
                                    error_score='raise', n_jobs=2)
    finally:
        _close_own_executor()
    assert len(result['estimator']) == 2
    assert np.isfinite(result['test_score']).all()
    for model, (train, test) in zip(result['estimator'], splitter.split(data[0], data[1])):
        assert isinstance(model.problem_, cp.Problem)
        manual = clone(model).fit(data[0][train], data[1][train])
        np.testing.assert_allclose(model.decision_function(data[0][test]),
                                   manual.decision_function(data[0][test]),
                                   rtol=2e-6, atol=2e-7)


def _caller_scipy_options(failure):
    _solver_options('SCIPY')
    nested = {'threads': 1}
    if failure:
        nested.update(method='highs', bounds=[(None, None)])
    return {'solver': 'SCIPY', 'verbose': False, 'scipy_options': nested}


def _assert_options_outcome(options, before, nested, failure):
    assert options['scipy_options'] is nested
    assert options == before


@pytest.mark.parametrize('failure', (False, True), ids=('successful_solve', 'failed_solve'))
def test_real_scipy_estimator_preserves_caller_nested_options(failure, data):
    options = _caller_scipy_options(failure)
    before = copy.deepcopy(options)
    nested = options['scipy_options']
    model = SVM(C=.05, solver_kws=options)
    if failure:
        with pytest.raises(ValueError, match='Please do not specify bounds'):
            model.fit(data[0], data[1])
        assert not any(k.endswith('_') and not k.startswith('_') for k in vars(model))
    else:
        model.fit(data[0], data[1])
        assert model.solver_status_ in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        assert np.isfinite(model.decision_function(data[2])).all()
    assert model.solver_kws is options
    _assert_options_outcome(options, before, nested, failure)


@pytest.mark.parametrize('failure', (False, True), ids=('successful_solve', 'failed_solve'))
def test_restored_problem_preserves_caller_nested_options(failure, models):
    original = models('SVM', 'SCS').problem_
    guard = _guard(original)
    problem = _loads(_dumps(original, 'pickle'), 'pickle')
    _assert_unchanged(original, guard)
    options = _caller_scipy_options(failure)
    before = copy.deepcopy(options)
    nested = options['scipy_options']
    if failure:
        with pytest.raises(ValueError, match='Please do not specify bounds'):
            problem.solve(**options)
    else:
        value = problem.solve(**options)
        assert problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        assert np.isfinite(value)
    _assert_options_outcome(options, before, nested, failure)


class IdentityCLARABEL(importlib.import_module(
        'cvxpy.reductions.solvers.conic_solvers.clarabel_conif').CLARABEL):
    def name(self):
        return 'DWD_TEST_CALLER_IDENTITY_CLARABEL'

    def __deepcopy__(self, memo):
        raise AssertionError('A caller-supplied solver must not be deep-copied')


class OpaqueOptionMapping(dict):
    def __deepcopy__(self, memo):
        raise AssertionError('An opaque option object must not be deep-copied')


@pytest.mark.parametrize('failure', (False, True), ids=('successful_forward', 'failed_forward'))
def test_solve_option_graph_copy_preserves_aliases_cycles_and_opaque_identity(failure, monkeypatch):
    solver = IdentityCLARABEL()
    array = np.arange(4., dtype=np.float64)
    callback = lambda value: value
    opaque = OpaqueOptionMapping(value=3)
    nested = {'entries': [1, 2]}
    sequence = (nested, array)
    nested['cycle'] = nested
    nested['sequence'] = sequence
    forwarded = {}

    def downstream(problem, *args, **kwargs):
        received = kwargs['options']
        assert received is kwargs['options_alias'] is args[0][0]
        assert received['cycle'] is received
        assert received['sequence'] is args[0]
        assert kwargs['solver'] is solver
        assert kwargs['array'] is array and args[0][1] is array
        assert kwargs['callback'] is callback and kwargs['opaque'] is opaque
        assert received is not nested
        assert args[0] is not sequence
        received['entries'].append('backend')
        received['backend_added'] = True
        forwarded['checked'] = True
        if failure:
            raise RuntimeError('controlled downstream solve failure')
        return 42.

    monkeypatch.setattr(cp.Problem, 'solve', downstream)
    problem = ProblemBase(cp.Minimize(cp.Constant(0.)))
    kwargs = dict(solver=solver, options=nested, options_alias=nested,
                  array=array, callback=callback, opaque=opaque)
    if failure:
        with pytest.raises(RuntimeError, match='controlled downstream solve failure'):
            problem.solve(sequence, **kwargs)
    else:
        assert problem.solve(sequence, **kwargs) == 42.
    assert forwarded == {'checked': True}
    assert nested['cycle'] is nested and nested['sequence'] is sequence
    assert sequence[0] is nested and sequence[1] is array
    assert nested['entries'] == [1, 2] and 'backend_added' not in nested
    np.testing.assert_array_equal(array, np.arange(4.))
    assert opaque == {'value': 3}


def test_real_custom_solver_instance_identity_is_preserved(data):
    _solver_options('CLARABEL')
    solver = IdentityCLARABEL()
    options = {'solver': solver, 'verbose': False}
    model = SVM(C=.05, solver_kws=options).fit(data[0], data[1])
    assert model.solver_kws is options and options['solver'] is solver
    assert options == {'solver': solver, 'verbose': False}
    assert model.solver_status_ in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    assert np.isfinite(model.decision_function(data[2])).all()
