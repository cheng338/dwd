"""Mandatory installed-wheel route witnesses for explicitly selected CI contracts.

This is test tooling, not a production dispatch policy. Expected routes are
independent of the production predicates whose regressions it must detect.
"""
from contextlib import contextmanager
import importlib
import importlib.util
import inspect
import math
from pathlib import Path
import platform
import sys
import sysconfig
import types


_PREFIX = 'Native route preflight: '


def _require(condition, message):
    if not condition:
        raise RuntimeError(_PREFIX + message)


def expected_routes(backend, contract, expected_platform, expected_python, numpy_version):
    """Return the declared CI contract without consulting production guards."""
    _require(backend in ('native', 'portable'), 'unknown wheel backend')
    _require(contract in ('current', 'released-1.3.19'), 'unknown route contract')
    _require(expected_platform in ('windows', 'linux'), 'unknown expected platform')
    _require(expected_python in ('3.11', '3.12'), 'unknown expected Python')
    extension = backend == 'native'
    scalar = expected_python == '3.12'
    guarded = scalar or (extension and expected_platform == 'windows')
    routes = {
        'raw_compiled_residual': extension,
        'guarded_residual_n8': ('compiled_values' if extension and guarded else
                                'scalar_fallback' if scalar else 'declined'),
        'guarded_residual_n4': 'scalar_fallback' if scalar else 'declined',
        'ordinary_scores': extension and expected_platform == 'windows',
        'reporting_scores': extension,
        'reporting_affine': extension and contract == 'current',
        'ordinary_affine': False,
    }
    if contract == 'current':
        routes['ordered_gradient'] = extension
        # NumPy 1.26 errstate inherits Python callbacks from ContextDecorator.
        routes['feature_expansion'] = extension and scalar and not numpy_version.startswith('1.26.')
    return routes


def _runtime_identity(expected_platform, expected_python):
    actual = {
        'implementation': sys.implementation.name,
        'platform': sys.platform,
        'python': '.'.join(str(v) for v in sys.version_info[:2]),
        'machine': platform.machine().lower(),
        'pointer_bits': 64 if sys.maxsize > 2**32 else 32,
        'gil_disabled': sysconfig.get_config_var('Py_GIL_DISABLED'),
        'float_format': [sys.float_info.radix, sys.float_info.mant_dig,
                         sys.float_info.max_exp],
    }
    _require(actual['implementation'] == 'cpython', 'CI contract requires CPython')
    _require(actual['platform'] == {'windows': 'win32', 'linux': 'linux'}[expected_platform],
             'actual platform does not match expected platform')
    _require(actual['python'] == expected_python, 'actual Python does not match expected Python')
    _require(actual['machine'] in ('amd64', 'x86_64') and actual['pointer_bits'] == 64,
             'CI contract requires x86-64')
    _require(actual['gil_disabled'] in (None, 0), 'CI contract requires a GIL-enabled build')
    _require(actual['float_format'] == [2, 53, 1024], 'CI contract requires binary64')
    return actual


def _array_equal(np, value, expected, route):
    _require(type(value) is np.ndarray and value.dtype == np.dtype('float64')
             and value.shape == expected.shape and bool(np.isfinite(value).all())
             and bool(np.array_equal(value, expected)), route + ' returned incorrect values or shape')


def _native_entry(accelerator, name):
    entry = getattr(accelerator, name, None)
    _require(isinstance(entry, types.BuiltinFunctionType)
             and getattr(entry, '__self__', None) is accelerator,
             name + ' is missing or is not the real installed native entry')
    return entry


@contextmanager
def _entry_witness(accelerator, name, expected_native):
    """Forward the actual native callable, restoring it on every exit."""
    original = getattr(accelerator, name, None)
    if expected_native:
        _native_entry(accelerator, name)
    observed = {'observed_calls': 0, 'completed_calls': 0, 'statuses': []}
    def forward(*args, **kwargs):
        observed['observed_calls'] += 1
        status = original(*args, **kwargs)
        observed['completed_calls'] += 1
        observed['statuses'].append(status)
        return status
    if original is not None:
        setattr(accelerator, name, forward)
    try:
        yield observed
    finally:
        if original is not None:
            setattr(accelerator, name, original)


def _check_witness(observed, expected_native, route):
    if expected_native:
        _require(observed['observed_calls'] > 0, route + ' required native entry did not execute')
        _require(observed['completed_calls'] == observed['observed_calls']
                 and all(type(status) is int and status == 0 for status in observed['statuses']),
                 route + ' native entry did not complete successfully')
    else:
        _require(observed['observed_calls'] == 0, route + ' unexpectedly entered native code')


def _score_probe(np, accelerator, name, expected_native, function, expected, route):
    try:
        with _entry_witness(accelerator, name, expected_native) as observed:
            result = function()
    except Exception as error:
        raise RuntimeError(_PREFIX + route + ' raised during its route witness') from error
    _array_equal(np, result, expected, route)
    _check_witness(observed, expected_native, route)
    return {'expected_native': expected_native, **observed, 'output_verified': True}


def _residual_probe(np, accelerator, native, residual_profile, n, expected_route):
    K, x, rhs = np.eye(n), np.ones(n), np.zeros(n)
    route = 'guarded_residual_n' + str(n)
    try:
        with _entry_witness(accelerator, 'evaluate', expected_route == 'compiled_values') as entry_observed:
            with residual_profile() as profile:
                result = native.native_compensated_residual(K, .5, rhs, x, .25, 0.)
            observed = profile.as_dict()
    except Exception as error:
        raise RuntimeError(_PREFIX + route + ' raised') from error
    _check_witness(entry_observed, expected_route == 'compiled_values', route)
    counts = observed['counts']
    _require(counts['calls'] == 1 and counts['raised'] == 0, route + ' has invalid call evidence')
    if expected_route == 'declined':
        _require(result is None and counts['declined'] == 1 and counts['bounded_return'] == 0
                 and counts['scalar_sumprod_calls_started'] == 0
                 and observed['compiled_helper']['values_returned'] == 0,
                 route + ' unexpectedly admitted an unsupported route')
    else:
        _require(isinstance(result, tuple) and len(result) == 6,
                 route + ' silently declined')
        _require(counts['bounded_return'] == 1 and counts['declined'] == 0
                 and observed['routes'][expected_route]['bounded_return'] == 1,
                 route + ' did not use expected ' + expected_route)
        _array_equal(np, result[0], np.full(n, -1.75), route + ' residual')
        _array_equal(np, result[2], np.ones(n), route + ' scores')
        _require(type(result[1]) is float and result[1] == -float(n),
                 route + ' returned an incorrect constraint')
        for index in (3, 5):
            bound = result[index]
            _require(type(bound) is np.ndarray and bound.dtype == np.dtype('float64')
                     and bound.shape == (n,) and bool(np.isfinite(bound).all())
                     and bool((bound >= 0).all()), route + ' returned invalid bounds')
        _require(math.isfinite(result[4]) and result[4] >= 0,
                 route + ' returned an invalid constraint bound')
        expected_scalar_calls = 2 * n if expected_route == 'scalar_fallback' else 0
        _require(counts['scalar_sumprod_calls_started'] == expected_scalar_calls
                 and counts['scalar_sumprod_calls_completed'] == expected_scalar_calls,
                 route + ' has incorrect scalar-call evidence')
        _require(observed['compiled_helper']['values_returned'] ==
                 (1 if expected_route == 'compiled_values' else 0),
                 route + ' has incorrect compiled-dispatch evidence')
    _array_equal(np, K, np.eye(n), route + ' kernel immutability')
    _array_equal(np, x, np.ones(n), route + ' coefficient immutability')
    _array_equal(np, rhs, np.zeros(n), route + ' rhs immutability')
    return {'expected_route': expected_route, 'observed': observed,
            'entry_witness': entry_observed,
            'bounded_return': result is not None, 'output_verified': True}


def _check_ordered_witness(observed, expected_native):
    """A C call followed by fallback is not a successful accelerated route."""
    expected_calls = 1 if expected_native else 0
    _require(observed['native_calls'] == expected_calls
             and observed['native_returns'] == expected_calls,
             'ordered_gradient did not execute the required genuine C route')
    _require(observed['facade_returns'] == 1
             and observed['accepted_native_returns'] == expected_calls,
             'ordered_gradient did not retain the required facade result')


def _ordered_gradient_probe(np, expected_native):
    """Observe untouched native/guard bindings; never time this CI witness."""
    from dwd import _linear_feature_diagnostics as feature
    from dwd import _ordered_gradient_reduction as facade

    name = 'dwd._ordered_reduction_accel'
    spec = importlib.util.find_spec(name)
    _require((spec is not None) is expected_native,
             'ordered_gradient artifact presence does not match wheel backend')
    accelerator = importlib.import_module(name) if expected_native else None
    entry = _native_entry(accelerator, 'reduce_rows') if expected_native else None
    origin = None
    if expected_native:
        origin = Path(accelerator.__file__).resolve()
        _require(origin.parent == Path(feature.__file__).resolve().parent,
                 'ordered_gradient accelerator is outside the installed package')
        _require(type(accelerator.API_VERSION) is int and accelerator.API_VERSION == 1
                 and type(accelerator.MAX_ROWS) is int and accelerator.MAX_ROWS == 64
                 and type(accelerator.MAX_TERMS) is int and accelerator.MAX_TERMS == 12288,
                 'ordered_gradient native API does not match the current contract')
    else:
        _require(name not in sys.modules, 'portable wheel has an ordered native module')
    X = np.array([[1., .5], [-.25, 2.]])
    z, alpha = np.array([.25, -.5]), np.array([.125, .25])
    # Exact dyadic arithmetic: X.T @ (z + .5*alpha) = (13/32, -19/32).
    # This oracle does not invoke the implementation's product or sum helpers.
    expected = np.array([13./32., -19./32.])
    before = X.tobytes(), z.tobytes(), alpha.tobytes()
    observed = {'native_calls': 0, 'native_returns': 0,
                'facade_returns': 0, 'accepted_native_returns': 0}
    returned = []
    code = facade.try_reduce.__code__
    previous = sys.getprofile()
    def profile(frame, event, arg):
        if entry is not None and arg is entry:
            if event == 'c_call':
                observed['native_calls'] += 1
            elif event == 'c_return':
                observed['native_returns'] += 1
        if event == 'return' and frame.f_code is code:
            returned.append(arg)
    try:
        sys.setprofile(profile)
        actual = feature._feature_gradient(X, z, alpha, .5)
    finally:
        sys.setprofile(previous)
    _array_equal(np, actual, expected, 'ordered_gradient')
    _require(actual.tobytes() == expected.tobytes(), 'ordered_gradient changed exact output bits')
    _require(before == (X.tobytes(), z.tobytes(), alpha.tobytes()), 'ordered_gradient mutated input')
    observed['facade_returns'] = len(returned)
    for value in returned:
        if value is not None:
            _array_equal(np, value, expected, 'ordered_gradient facade')
            _require(value.tobytes() == expected.tobytes(), 'ordered_gradient facade changed exact bits')
            observed['accepted_native_returns'] += 1
    _check_ordered_witness(observed, expected_native)
    _require(facade._LOAD_ATTEMPTED is True, 'ordered_gradient loader was bypassed')
    _require(facade._ACCEL is accelerator and facade._ENTRY is entry,
             'ordered_gradient cache does not match the required route')
    if not expected_native:
        _require(name not in sys.modules and importlib.util.find_spec(name) is None,
                 'portable wheel acquired an ordered native module')
    return {'expected_native': expected_native, **observed, 'output_verified': True,
            'accelerator_origin': str(origin) if origin is not None else None,
            'witness': 'Untimed sys.setprofile observation; no entry or guard replacement.'}


def _feature_expansion_probe(np, expected_native, expected_artifact):
    """Unprofiled genuine facade acceptance; hooks deliberately force fallback."""
    from dwd import _linear_feature_diagnostics as feature
    from dwd import _feature_expansion as facade
    from dwd import _compensated_residual as arithmetic
    _require(sys.getprofile() is None and sys.gettrace() is None,
             'feature_expansion witness requires an unhooked thread')
    name = 'dwd._feature_expansion_accel'
    spec = importlib.util.find_spec(name)
    _require((spec is not None) is expected_artifact,
             'feature_expansion artifact presence does not match the build contract')
    X = np.array([[1., .5], [-.25, 2.]])
    z, alpha = np.array([.25, -.5]), np.array([.125, .25])
    columns = X.T
    before = X.tobytes(), z.tobytes(), alpha.tobytes()
    pm, pe = math.frexp(.5)
    zm, ze = np.frexp(z)
    am, ae = np.frexp(alpha)
    high, low = arithmetic._products(np.full(len(alpha), pm), am)
    prepared = (zm, ze, high, low, ae, pe)
    splits = tuple(arithmetic._split_operand(value) for value in (zm, high, low))
    expected_gradient = np.array([13./32., -19./32.])
    expected_beta = np.array([1./16., 9./16.])
    # Earlier ordered-route preflight has naturally qualified the old facade.
    # Direct try_gradient has no Python numerical success return: an ndarray
    # proves the genuine entry returned OK through the unchanged C barrier.
    gradient = facade.try_gradient(feature, columns, prepared, splits)
    beta = facade.try_beta(feature, columns, (am, ae), arithmetic._split_operand(am))
    if expected_native:
        accelerator = sys.modules.get(name)
        _require(type(accelerator) is types.ModuleType,
                 'feature_expansion required native module did not load')
        _require(facade._ENTRY is _native_entry(accelerator, 'gradient_from_splits')
                 and facade._BETA_ENTRY is _native_entry(accelerator, 'beta_from_split'),
                 'feature_expansion cache does not own the genuine native entries')
        for key, actual, expected in (('gradient', gradient, expected_gradient),
                                      ('beta', beta, expected_beta)):
            _array_equal(np, actual, expected, 'feature_expansion ' + key)
            _require(actual.tobytes() == expected.tobytes() and actual.flags.owndata,
                     'feature_expansion ' + key + ' changed bits/private output')
    else:
        _require(gradient is None and beta is None and name not in sys.modules,
                 'unsupported/portable build unexpectedly used feature expansion')
    _require(feature._feature_gradient(X, z, alpha, .5).tobytes() == expected_gradient.tobytes()
             and feature._feature_coefficients(X, alpha).tobytes() == expected_beta.tobytes(),
             'feature_expansion full contraction/fallback changed exact bits')
    _require(before == (X.tobytes(), z.tobytes(), alpha.tobytes()),
             'feature_expansion mutated inputs')
    loaded_by_facade = name in sys.modules
    _require(loaded_by_facade is expected_native,
             'feature_expansion full contraction loaded an unexpected module')
    if expected_artifact and not expected_native:
        from contextlib import ContextDecorator
        _require(np.errstate.__bases__ == (ContextDecorator,)
                 and facade._LOAD_ATTEMPTED is False and facade._CONTEXT is None,
                 'feature_expansion must decline the NumPy 1.26 callback family before loading')
    origin = None
    if expected_artifact:
        # Inspect the installed binary only after observing genuine public dispatch.
        accelerator = importlib.import_module(name)
        origin = Path(accelerator.__file__).resolve()
        _require(origin.parent == Path(feature.__file__).resolve().parent,
                 'feature_expansion module is outside the installed package')
        _require(accelerator.API_VERSION == 2 and accelerator.BETA_API_VERSION == 1
                 and accelerator.COMPILED_PY_HEX == sys.hexversion,
                 'feature_expansion ABI/capability does not match this interpreter')
        _native_entry(accelerator, 'gradient_from_splits')
        _native_entry(accelerator, 'beta_from_split')
    return {'expected_native': expected_native, 'artifact_present': spec is not None,
            'loaded_by_facade': loaded_by_facade,
            'gradient_native_accepted': gradient is not None,
            'beta_native_accepted': beta is not None,
            'output_verified': True,
            'accelerator_origin': str(origin) if origin is not None else None,
            'witness': 'Unprofiled facade returns and exact full contractions before explicit artifact inspection.'}


def verify_routes(backend, contract, expected_platform, expected_python):
    """Require intended routes and exact tiny results; never skip from a guard."""
    actual_runtime = _runtime_identity(expected_platform, expected_python)
    import numpy as np
    actual_runtime['numpy'] = np.__version__
    expected = expected_routes(backend, contract, expected_platform, expected_python, np.__version__)
    from dwd import _compiled_residual as compiled
    from dwd import _native_residual as native
    from dwd import _kernel_scores as scores
    from dwd import _affine_scores as affine
    from dwd.profiling import residual_profile

    accelerator = compiled._ACCEL
    if contract == 'released-1.3.19':
        _require(getattr(accelerator, 'score_affine_rows', None) is None
                 and '_reporting' not in inspect.signature(affine.affine_scores).parameters,
                 'released-1.3.19 affine absence contract does not match installed artifact')
    if backend == 'native':
        _require(accelerator is not None, 'native wheel has no loaded accelerator')
        _native_entry(accelerator, 'evaluate')
    else:
        _require(accelerator is None, 'portable wheel unexpectedly loaded an accelerator')
    K, alpha, rhs = np.eye(8), np.ones(8), np.zeros(8)
    try:
        with _entry_witness(accelerator, 'evaluate', expected['raw_compiled_residual']) as raw_witness:
            raw = compiled.compiled_values(K, alpha, rhs, .5, .25)
    except Exception as error:
        raise RuntimeError(_PREFIX + 'raw_compiled_residual raised') from error
    _check_witness(raw_witness, expected['raw_compiled_residual'], 'raw_compiled_residual')
    if expected['raw_compiled_residual']:
        _require(isinstance(raw, tuple) and len(raw) == 3, 'raw_compiled_residual silently declined')
        for value, reference in zip(raw, (np.ones(8), np.full(8, -1.75), np.ones(8))):
            _array_equal(np, value, reference, 'raw_compiled_residual')
    else:
        _require(raw is None, 'portable wheel unexpectedly used raw compiled residual')
    routes = {'raw_compiled_residual': {
        'expected_native': expected['raw_compiled_residual'],
        **raw_witness, 'values_returned': raw is not None, 'output_verified': True}}
    for n in (8, 4):
        key = 'guarded_residual_n' + str(n)
        routes[key] = _residual_probe(np, accelerator, native, residual_profile, n, expected[key])
    routes['ordinary_scores'] = _score_probe(
        np, accelerator, 'score_rows', expected['ordinary_scores'],
        lambda: scores.compensated_kernel_matvec(K, alpha), np.ones(8), 'ordinary_scores')
    routes['reporting_scores'] = _score_probe(
        np, accelerator, 'score_rows', expected['reporting_scores'],
        lambda: scores._reporting_kernel_matvec(K, alpha), np.ones(8), 'reporting_scores')
    affine_report_call = (
        (lambda: affine.affine_scores(K, alpha, .25, policy='compensated', _reporting=True))
        if contract == 'current' else
        (lambda: affine.affine_scores(K, alpha, .25, policy='compensated')))
    routes['reporting_affine'] = _score_probe(
        np, accelerator, 'score_affine_rows', expected['reporting_affine'],
        affine_report_call, np.full(8, 1.25), 'reporting_affine')
    routes['reporting_affine']['api_scope'] = (
        'current reporting opt-in' if contract == 'current' else
        'released portable affine API; no reporting opt-in or affine native entry')
    routes['ordinary_affine'] = _score_probe(
        np, accelerator, 'score_affine_rows', False,
        lambda: affine.affine_scores(K, alpha, .25, policy='compensated'),
        np.full(8, 1.25), 'ordinary_affine')
    _array_equal(np, K, np.eye(8), 'fixture kernel immutability')
    _array_equal(np, alpha, np.ones(8), 'fixture coefficient immutability')
    _array_equal(np, rhs, np.zeros(8), 'fixture rhs immutability')
    if contract == 'current':
        routes['ordered_gradient'] = _ordered_gradient_probe(np, expected['ordered_gradient'])
        routes['feature_expansion'] = _feature_expansion_probe(
            np, expected['feature_expansion'], backend == 'native' and expected_python == '3.12')
    return {'contract': contract, 'backend': backend,
            'expected_runtime': {'platform': expected_platform, 'python': expected_python},
            'actual_runtime': actual_runtime, 'expected_routes': expected, 'routes': routes}
