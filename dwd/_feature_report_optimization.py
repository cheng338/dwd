"""Private sharing of feature-report product arithmetic.

This internal helper changes neither the geometry nor any diagnostic gate.
Hooked, native-score, exceptional and unsupported calls keep the original
whole-raw-then-whole-affine sequence. No cache survives a report invocation.
"""
import math
import sys
import types
from functools import cached_property as _CachedProperty

import numpy as np

from . import _affine_scores as _affine
from . import _compensated_residual as _arithmetic
from . import _compiled_scores as _compiled
from . import _kernel_scores as _scores
from . import _quadratic_bounds as _bounds


def _definition(module, name, function):
    """Recognize the ordinary definition, not a preimport wrapper.

    These are finite compatibility checks, not hostile-code attestation.
    """
    return (type(function) is types.FunctionType
            and function.__globals__ is module.__dict__
            and function.__module__ == module.__name__
            and function.__name__ == name
            and function.__qualname__ == name
            and function.__code__.co_name == name
            and function.__code__.co_qualname == name
            and function.__code__.co_filename == module.__file__
            and not hasattr(function, '__wrapped__'))


_FUNCTIONS = tuple(
    (module, name, getattr(module, name),
     getattr(getattr(module, name), '__code__', None),
     getattr(getattr(module, name), '__defaults__', None),
     dict(getattr(getattr(module, name), '__kwdefaults__', None) or {}))
    for module, names in (
        (_arithmetic, ('_products', '_split_operand', '_fsum', '_gradual_underflow')),
        (_scores, ('compensated_kernel_matvec', '_compensated_kernel_matvec',
                   '_inputs', '_expanded_row', '_expanded_dense_rows')),
        (_affine, ('affine_scores', '_intercept', '_row_data', '_exact_row',
                   '_finish_row', '_finish_score', '_zero_has_exact_products',
                   '_round_to_nearest_now', '_expanded_row', '_expanded_dense_rows')),
        (_compiled, ('try_expanded_dense_rows', '_supported')),
        (_bounds, ('_standard_feature_bound_bindings',)),
    ) for name in names)
_NUMPY = tuple((name, getattr(np, name)) for name in (
    'asarray', 'empty', 'arange', 'broadcast_to', 'frexp', 'ldexp',
    'isfinite', 'any', 'all', 'abs', 'multiply', 'nextafter', 'geterr'))
_MATH = tuple((name, getattr(math, name)) for name in ('fsum', 'nextafter', 'isfinite'))


def _numpy_definition(name, function):
    if type(function) is np.ufunc:
        return function.__name__ == ('absolute' if name == 'abs' else name)
    if type(function) is types.BuiltinFunctionType:
        return (function.__name__ == name
                and str(function.__module__).startswith('numpy'))
    # NumPy's public array-function dispatcher owns an original Python body.
    body = getattr(function, '__wrapped__', function)
    return (type(body) is types.FunctionType
            and body.__code__.co_name == name
            and body.__code__.co_qualname == name
            and str(body.__globals__.get('__name__', '')).startswith('numpy.')
            and (function is body or
                 (type(function).__name__ == '_ArrayFunctionDispatcher'
                  and (type(function).__module__ == 'numpy'
                       or type(function).__module__.startswith('numpy.')))))


# These tokens inspect ordinary entries without invoking replacement callbacks.
# Unknown layouts fall back; this is not private-global/cache attestation.
_NUMPY_MODULE = np
_MATH_MODULE = math
_ABSENT = object()


def _function_token(function):
    if type(function) is not types.FunctionType:
        return None
    defaults = function.__kwdefaults__
    if defaults is not None and (type(defaults) is not dict
            or any(type(key) is not str for key in defaults)):
        return None
    return (function, function.__code__, function.__defaults__,
            None if defaults is None else tuple(defaults.items()))


def _matches_function(function, token):
    if (token is None or type(function) is not types.FunctionType
            or function is not token[0] or function.__code__ is not token[1]
            or function.__defaults__ is not token[2]):
        return False
    defaults, expected = function.__kwdefaults__, token[3]
    if expected is None:
        return defaults is None
    return (type(defaults) is dict and len(defaults) == len(expected)
            and all(type(key) is str for key in defaults)
            and all(key in defaults and defaults[key] is value
                    for key, value in expected))


def _owned_function(owner, function, qualified):
    name = qualified.rsplit('.', 1)[-1]
    return (type(owner) is types.ModuleType
            and type(function) is types.FunctionType
            and function.__globals__ is owner.__dict__
            and function.__module__ == owner.__dict__.get('__name__')
            and function.__name__ == name and function.__qualname__ == qualified
            and function.__code__.co_name == name
            and function.__code__.co_qualname == qualified
            and function.__code__.co_filename == owner.__dict__.get('__file__')
            and '__wrapped__' not in function.__dict__)


def _member_function(entry):
    if type(entry) is staticmethod:
        return entry.__func__
    if type(entry) is property:
        return entry.fget
    if type(entry) is _CachedProperty:
        return entry.__dict__.get('func')
    return entry


def _member_token(cls, name):
    entry = cls.__dict__.get(name, _ABSENT) if type(cls) is type else _ABSENT
    function = _member_function(entry)
    return entry, function, _function_token(function)


def _matches_member(cls, name, token, owner, qualified):
    entry, function, state = token
    if cls.__dict__.get(name, _ABSENT) is not entry:
        return False
    if entry is _ABSENT:
        return True
    return (_member_function(entry) is function and _matches_function(function, state)
            and _owned_function(owner, function, qualified))


def _owner_of(function, allowed):
    if type(function) is not types.FunctionType:
        return None
    name = function.__globals__.get('__name__')
    if type(name) is not str or name not in allowed:
        return None
    owner = sys.modules.get(name)
    return owner if type(owner) is types.ModuleType else None


_FINFO = np.__dict__.get('finfo')
_FINFO_MEMBERS = tuple((name, _member_token(_FINFO, name)) for name in
                       ('__new__', '__init__', '_init', 'tiny', 'smallest_normal'))
_FINFO_OWNER = _owner_of(_FINFO_MEMBERS[0][1][1],
                       ('numpy._core.getlimits', 'numpy.core.getlimits'))
_ISSPARSE = _scores.__dict__.get('issparse')
_ISSPARSE_TOKEN = _function_token(_ISSPARSE)
_ISSPARSE_OWNER = _owner_of(_ISSPARSE,
                          ('scipy._lib._sparse', 'scipy.sparse._base'))


def _numpy_dispatcher_type():
    # Inspect only already-loaded plain modules; never probe a callable object.
    for name in ('numpy._core._multiarray_umath', 'numpy.core._multiarray_umath'):
        owner = sys.modules.get(name)
        if type(owner) is types.ModuleType:
            cls = owner.__dict__.get('_ArrayFunctionDispatcher')
            if (type(cls) is type and cls.__name__ == '_ArrayFunctionDispatcher'
                    and cls.__module__ in ('numpy', name)):
                return cls
    return None


_NUMPY_DISPATCHER = _numpy_dispatcher_type()


def _numpy_body_token(function):
    if type(function) is types.FunctionType:
        return _function_token(function)
    if type(function) is _NUMPY_DISPATCHER:
        return _function_token(function.__wrapped__)
    return None


_INPUT_NUMPY = tuple((name, np.__dict__.get(name),
                     _numpy_body_token(np.__dict__.get(name)))
                    for name in ('iscomplexobj', 'zeros'))


def _numpy_entry(name, original, body):
    function = np.__dict__.get(name)
    kind = type(function)
    if (kind is not np.ufunc and kind is not types.BuiltinFunctionType
            and kind is not types.FunctionType and kind is not _NUMPY_DISPATCHER):
        return False
    if function is not original or not _numpy_definition(name, function):
        return False
    if body is None:
        return type(function) is np.ufunc or type(function) is types.BuiltinFunctionType
    current = (function if type(function) is types.FunctionType
               else function.__wrapped__)
    return _matches_function(current, body)


def _standard_modules(modules):
    if (np is not _NUMPY_MODULE or math is not _MATH_MODULE
            or type(np) is not types.ModuleType or type(math) is not types.ModuleType
            or sys.modules.get('numpy') is not np or sys.modules.get('math') is not math):
        return False
    return all(type(module) is types.ModuleType
               and module.__dict__.get('np') is np
               and ('math' not in module.__dict__ or module.__dict__['math'] is math)
               for module in modules)


def _standard_finfo():
    cls, owner = np.__dict__.get('finfo'), _FINFO_OWNER
    if (cls is not _FINFO or type(cls) is not type or cls.__name__ != 'finfo'
            or cls.__bases__ != (object,) or type(owner) is not types.ModuleType
            or owner.__dict__.get('finfo') is not cls):
        return False
    for name, token in _FINFO_MEMBERS:
        entry = token[0]
        if name in ('__new__', '_init') and entry is _ABSENT:
            return False
        if name == 'tiny' and type(entry) not in (property, _CachedProperty):
            return False
        if (name == 'smallest_normal' and entry is not _ABSENT
                and type(entry) not in (property, _CachedProperty)):
            return False
        if not _matches_member(cls, name, token, owner, 'finfo.' + name):
            return False
    return True


def _standard_input_bindings():
    if (not _standard_modules((_arithmetic, _scores, _affine))
            or not _standard_finfo()
            or not all(_numpy_entry(*entry) for entry in _INPUT_NUMPY)):
        return False
    function, owner = _scores.__dict__.get('issparse'), _ISSPARSE_OWNER
    return (_matches_function(function, _ISSPARSE_TOKEN)
            and type(owner) is types.ModuleType
            and owner.__dict__.get('issparse') is function
            and _owned_function(owner, function, 'issparse'))


def _standard_bindings():
    if not _standard_input_bindings():
        return False
    for module, name, original, code, defaults, kwdefaults in _FUNCTIONS:
        function = getattr(module, name, None)
        if (function is not original or not _definition(module, name, function)
                or function.__code__ is not code
                or function.__defaults__ is not defaults):
            return False
        actual_defaults = function.__kwdefaults__ or {}
        if (type(actual_defaults) is not dict
                or any(type(key) is not str for key in actual_defaults)
                or actual_defaults.keys() != kwdefaults.keys()
                or any(actual_defaults[key] is not value for key, value in kwdefaults.items())):
            return False
    for name, original in _NUMPY:
        function = getattr(np, name, None)
        if function is not original or not _numpy_definition(name, function):
            return False
    for name, original in _MATH:
        function = getattr(math, name, None)
        if (function is not original or type(function) is not types.BuiltinFunctionType
                or function.__module__ != 'math' or function.__name__ != name):
            return False
    if not _bounds._standard_feature_bound_bindings():
        return False
    if any(mode in ('call', 'log') for mode in np.geterr().values()):
        return False
    return (_scores._products is _affine._products is _arithmetic._products
            and _scores._fsum is _affine._fsum is _arithmetic._fsum
            and _scores._split_operand is _affine._split_operand is _arithmetic._split_operand
            and _scores._gradual_underflow is _arithmetic._gradual_underflow
            and _affine._inputs is _scores._inputs
            and _scores.try_expanded_dense_rows is _compiled.try_expanded_dense_rows
            and _scores._SCORE_TILE_BYTES == _affine._TILE_BYTES == 1024 * 1024
            and _scores._SCORE_TILE_MAX_ROWS == _affine._TILE_MAX_ROWS == 8)


def can_share_products(features, beta, raw, affine):
    """Counter-free prospective dispatch query for tests and saved receipts."""
    if (raw is not _scores.compensated_kernel_matvec
            or affine is not _affine.affine_scores
            or not _standard_bindings()
            or type(features) is not np.ndarray or features.dtype != np.float64
            or features.ndim != 2 or min(features.shape) < 1
            or type(beta) is not np.ndarray or beta.dtype != np.float64
            or beta.ndim != 1 or features.shape[1] != len(beta)
            or not _affine._round_to_nearest_now()):
        return False
    # Do not replace an available qualified C raw-score path with Python work.
    # A conservative decline also covers layouts the native entry may refuse.
    entry = getattr(_compiled._compiled_residual._ACCEL, 'score_rows', None)
    return not (entry is not None and _compiled._supported(reporting=False))


def _fused_pair(features, beta, offset):
    features, beta, sparse = _scores._inputs(features, beta)
    assert sparse is False
    offset = _affine._intercept(offset)
    raw, joint = np.empty(len(features)), np.empty(len(features))
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        split_beta = _arithmetic._split_operand(beta)
    width = min(8, max(1, (1024 * 1024) // (128 * len(beta))))
    for start in range(0, len(features), width):
        stop = min(start + width, len(features))
        columns = features[start:stop]
        coefficients = np.broadcast_to(beta, columns.shape)
        high, low = _arithmetic._products(columns, coefficients, _split_b=split_beta)
        for local, index in enumerate(range(start, stop)):
            terms = high[local].tolist() + low[local].tolist()
            raw[index] = _arithmetic._fsum(terms)
            # A separate sum is essential: fl(dot)+b is not the joint result.
            try:
                value = _arithmetic._fsum(terms + [offset])
            except FloatingPointError:
                value = _affine._exact_row(features, beta, offset, False, index)
            else:
                value, _ = _affine._finish_score(
                    value, len(beta), features, beta, offset, False, index)
            joint[index] = value
    return raw, joint


def feature_scores_and_margins(features, beta, offset, *, raw, affine):
    """Share ordinary products; retain original raw-first failure precedence."""
    if can_share_products(features, beta, raw, affine):
        try:
            return _fused_pair(features, beta, offset)
        except (MemoryError, FloatingPointError, OverflowError, ValueError):
            pass
    # Leave the exception frame before re-entering the two original helpers.
    # In particular, an early affine refusal cannot mask a later raw refusal.
    scores = raw(features, beta)
    margins = affine(features, beta, offset, policy='compensated')
    return scores, margins


_SPLIT_FUNCTIONS = tuple(entry for entry in _FUNCTIONS
                         if entry[0] is _arithmetic and entry[1] != '_gradual_underflow')
_SPLIT_NUMPY = tuple((name, np.__dict__.get(name),
                     _numpy_body_token(np.__dict__.get(name))) for name in (
    'asarray', 'frexp', 'ldexp', 'isfinite', 'broadcast_to', 'empty',
    'any', 'abs', 'geterr'))
_ERRSTATE = np.__dict__.get('errstate')
_ERRSTATE_MEMBERS = tuple((name, _member_token(_ERRSTATE, name))
                         for name in ('__init__', '__enter__', '__exit__'))
_ERRSTATE_OWNER = _owner_of(_ERRSTATE_MEMBERS[0][1][1],
                          ('numpy._core._ufunc_config', 'numpy.core._ufunc_config'))
_SPLIT_MATH = tuple(entry for entry in _MATH if entry[0] in ('fsum', 'isfinite'))


def _standard_split_bindings():
    """Check only callbacks reachable between preparation and cached uses."""
    if not _standard_modules((_arithmetic,)) or not _standard_finfo():
        return False
    for module, name, original, code, defaults, kwdefaults in _SPLIT_FUNCTIONS:
        function = module.__dict__.get(name)
        if (function is not original or not _definition(module, name, function)
                or function.__code__ is not code
                or function.__defaults__ is not defaults):
            return False
        actual_defaults = function.__kwdefaults__ or {}
        if (type(actual_defaults) is not dict
                or any(type(key) is not str for key in actual_defaults)
                or actual_defaults.keys() != kwdefaults.keys()
                or any(actual_defaults[key] is not value for key, value in kwdefaults.items())):
            return False
    if not all(_numpy_entry(*entry) for entry in _SPLIT_NUMPY):
        return False
    for name, original in _SPLIT_MATH:
        function = math.__dict__.get(name)
        if (function is not original or type(function) is not types.BuiltinFunctionType
                or function.__module__ != 'math' or function.__name__ != name):
            return False
    cls, owner = np.__dict__.get('errstate'), _ERRSTATE_OWNER
    if (cls is not _ERRSTATE or type(cls) is not type or cls.__name__ != 'errstate'
            or type(owner) is not types.ModuleType
            or owner.__dict__.get('errstate') is not cls):
        return False
    if not all(_matches_member(cls, name, token, owner, 'errstate.' + name)
               for name, token in _ERRSTATE_MEMBERS):
        return False
    return not any(mode in ('call', 'log') for mode in np.geterr().values())


def repeated_operand_splits(vectors, product):
    """Prepare identical 1-D right-hand splits for broadcast column tiles.

    The original product helper still validates both full operands and forms
    every high/low term at the same rounding points. Only repeated coefficient
    frexp/Dekker splitting is removed. Scalar/error paths are unchanged.
    """
    if (product is not _arithmetic._products or not _standard_split_bindings()
            or type(vectors) is not tuple or not vectors
            or any(type(vector) is not np.ndarray or vector.dtype != np.float64
                   or vector.ndim != 1 for vector in vectors)):
        return None
    try:
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            return tuple(_arithmetic._split_operand(vector) for vector in vectors)
    except (MemoryError, FloatingPointError, OverflowError, ValueError):
        return None


# Feature-only opt-in to the existing reporting entries. Unknown bindings use
# the original feature dispatcher; this does not enable general prediction.
_REPORT_COMPILED, _REPORT_SCORES, _REPORT_AFFINE = _compiled, _scores, _affine
_REPORT_RESIDUAL = _compiled.__dict__.get('_compiled_residual')
_REPORT_SYS = sys
_REPORT_PLATFORM = _compiled.__dict__.get('platform')
_REPORT_MACHINE = (_function_token(_REPORT_PLATFORM.__dict__.get('machine'))
                   if type(_REPORT_PLATFORM) is types.ModuleType else None)
_REPORT_FAMILY_GUARD = _function_token(_standard_bindings)
_REPORT_MATVEC = _function_token(_scores.__dict__.get('_reporting_kernel_matvec'))
_REPORT_ARRAY = _function_token(_compiled.__dict__.get('_array'))


def _native_feature_reporting_available(kind, raw, affine):
    if (type(kind) is not str or kind not in ('raw', 'joint')
            or _compiled is not _REPORT_COMPILED or _scores is not _REPORT_SCORES
            or _affine is not _REPORT_AFFINE
            or any(type(owner) is not types.ModuleType
                   for owner in (_compiled, _scores, _affine, _REPORT_RESIDUAL))):
        return False
    if (_compiled.__dict__.get('_compiled_residual') is not _REPORT_RESIDUAL
            or raw is not _scores.__dict__.get('compensated_kernel_matvec')
            or affine is not _affine.__dict__.get('affine_scores')):
        return False
    accelerator = _REPORT_RESIDUAL.__dict__.get('_ACCEL')
    if type(accelerator) is not types.ModuleType:
        return False
    names = ('score_rows',) if kind == 'raw' else ('score_affine_rows',)
    for name in names:
        entry = accelerator.__dict__.get(name)
        if (type(entry) is not types.BuiltinFunctionType
                or entry.__self__ is not accelerator or entry.__name__ != name
                or type(entry.__module__) is not str
                or entry.__module__ != __package__+'._residual_accel'):
            return False
    system, platform_module = _compiled.__dict__.get('sys'), _compiled.__dict__.get('platform')
    if (system is not _REPORT_SYS or system is not sys or type(system) is not types.ModuleType
            or platform_module is not _REPORT_PLATFORM or type(platform_module) is not types.ModuleType):
        return False
    implementation = system.__dict__.get('implementation')
    platform_name = system.__dict__.get('platform')
    if (type(implementation) is not types.SimpleNamespace
            or type(implementation.__dict__.get('name')) is not str
            or implementation.__dict__['name'] != 'cpython'
            or type(platform_name) is not str or platform_name not in ('win32', 'linux')
            or not _matches_function(platform_module.__dict__.get('machine'), _REPORT_MACHINE)):
        return False
    reporting = _scores.__dict__.get('_reporting_kernel_matvec')
    array = _compiled.__dict__.get('_array')
    if (not _matches_function(reporting, _REPORT_MATVEC)
            or not _matches_function(array, _REPORT_ARRAY)
            or any(type(value) is not str for owner, function in
                   ((_scores, reporting), (_compiled, array)) for value in
                   (function.__module__, function.__name__, function.__qualname__,
                    owner.__dict__.get('__name__'), owner.__dict__.get('__file__')))
            or not _owned_function(_scores, reporting, '_reporting_kernel_matvec')
            or not _owned_function(_compiled, array, '_array')
            or not _matches_function(_standard_bindings, _REPORT_FAMILY_GUARD)
            or not _standard_bindings() or _compiled._supported(reporting=True) is not True):
        return False
    if kind == 'joint':
        name = __package__+'._compiled_affine_scores'
        cached = sys.modules.get(name, _ABSENT)
        if cached is not _ABSENT:
            # Avoid importlib probing a replaced cached module. If absent, its
            # ordinary lazy import stays inside affine evaluation, after raw.
            if type(cached) is not types.ModuleType:
                return False
            function = cached.__dict__.get('try_expanded_dense_rows')
            token = cached.__dict__.get('_FEATURE_REPORTING_ENTRY')
            if (type(token) is not tuple or len(token) != 4
                    or not _matches_function(function, token)
                    or type(function) is not types.FunctionType
                    or any(type(value) is not str for value in (
                        function.__module__, function.__name__, function.__qualname__,
                        cached.__dict__.get('__name__'), cached.__dict__.get('__file__')))
                    or not _owned_function(cached, function, 'try_expanded_dense_rows')
                    or cached.__dict__['__name__'] != name
                    or cached.__dict__.get('_compiled_residual') is not _REPORT_RESIDUAL
                    or cached.__dict__.get('_supported') is not _compiled._supported
                    or cached.__dict__.get('_array') is not array):
                return False
    return True


_ORIGINAL_REPORTING_POLICY = (_native_feature_reporting_available,
                              _native_feature_reporting_available.__code__)
