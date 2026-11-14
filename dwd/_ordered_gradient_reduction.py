"""Optional ordered reduction of already-restored gradient terms.

This module is imported only by the existing qualified multi-column route.
It changes neither expansion/restoration nor numerical acceptance. Missing,
unsupported or replaced native entries keep the original checked-fsum loop.
Compatibility checks are not hostile-code or concurrent-mutation attestation.
"""
import importlib
from importlib.machinery import ExtensionFileLoader, EXTENSION_SUFFIXES, ModuleSpec
import math
import platform
import sys
import sysconfig
import types

import numpy as np

from . import _compensated_residual as _arithmetic
from . import _feature_report_optimization as _support


_MODULE_NAME = __package__ + '._ordered_reduction_accel'
_GIL_DISABLED = sysconfig.get_config_var('Py_GIL_DISABLED')
_LOAD_ATTEMPTED = False
# Attempted is set before importing, so it does not imply a terminal cache.
# Unexpected exceptions leave completion false and retain the original fallback.
_LOAD_COMPLETE = False
_ACCEL = None
_ENTRY = None
_ABSENT = object()
_NUMPY = np
_MATH = math
_NDARRAY = np.__dict__.get('ndarray') if type(np) is types.ModuleType else None
_FLOAT64 = np.__dict__.get('float64') if type(np) is types.ModuleType else None
_UFUNC = np.__dict__.get('ufunc') if type(np) is types.ModuleType else None


def _owned(owner, function, name):
    if type(owner) is not types.ModuleType or type(function) is not types.FunctionType:
        return False
    values = (function.__module__, function.__name__, function.__qualname__,
              owner.__dict__.get('__name__'), owner.__dict__.get('__file__'))
    if not all(type(value) is str for value in values):
        return False
    return (function.__globals__ is owner.__dict__ and values[0] == values[3]
            and values[1] == values[2] == name
            and function.__code__.co_name == function.__code__.co_qualname == name
            and function.__code__.co_filename == values[4]
            and '__wrapped__' not in function.__dict__)


def _token(owner, name):
    function = owner.__dict__.get(name) if type(owner) is types.ModuleType else None
    if not _owned(owner, function, name):
        return None
    defaults = function.__kwdefaults__
    if defaults is not None and (type(defaults) is not dict
            or any(type(key) is not str for key in defaults)):
        return None
    return (function, function.__code__, function.__defaults__,
            None if defaults is None else tuple(defaults.items()))


def _matches(owner, name, token):
    if token is None or type(owner) is not types.ModuleType:
        return False
    function = owner.__dict__.get(name)
    if (function is not token[0] or not _owned(owner, function, name)
            or function.__code__ is not token[1] or function.__defaults__ is not token[2]):
        return False
    actual, expected = function.__kwdefaults__, token[3]
    if expected is None:
        return actual is None
    return (type(actual) is dict and len(actual) == len(expected)
            and all(type(key) is str for key in actual)
            and all(key in actual and actual[key] is value for key, value in expected))


_FEATURE = sys.modules.get(__package__ + '._linear_feature_diagnostics')


def _original_arithmetic_token(name):
    """Use the earlier pre-preparation tokens, not a lazy import snapshot."""
    entries = _support.__dict__.get('_SPLIT_FUNCTIONS') if type(_support) is types.ModuleType else None
    if type(entries) is not tuple or type(_arithmetic) is not types.ModuleType:
        return None
    matches = tuple(entry for entry in entries if type(entry) is tuple and len(entry) == 6
                    and entry[0] is _arithmetic and type(entry[1]) is str and entry[1] == name)
    if len(matches) != 1:
        return None
    _, _, function, code, defaults, kwdefaults = matches[0]
    if (not _owned(_arithmetic, function, name) or type(code) is not types.CodeType
            or type(kwdefaults) is not dict or any(type(key) is not str for key in kwdefaults)):
        return None
    return function, code, defaults, tuple(kwdefaults.items()) if kwdefaults else None


def _original_restore_token():
    if type(_FEATURE) is not types.ModuleType:
        return None
    function = _FEATURE.__dict__.get('_ORIGINAL_RESTORE_GRADIENT_TERMS')
    code = _FEATURE.__dict__.get('_ORIGINAL_RESTORE_GRADIENT_TERMS_CODE')
    if not _owned(_FEATURE, function, '_restore_gradient_terms') or type(code) is not types.CodeType:
        return None
    return function, code, None, None


def _dispatcher_type():
    for name in ('numpy._core._multiarray_umath', 'numpy.core._multiarray_umath'):
        owner = sys.modules.get(name)
        if type(owner) is not types.ModuleType:
            continue
        cls = owner.__dict__.get('_ArrayFunctionDispatcher')
        if (type(cls) is type and type(cls.__name__) is str and type(cls.__module__) is str
                and cls.__name__ == '_ArrayFunctionDispatcher' and cls.__module__ in ('numpy', name)):
            return cls
    return None


_PRODUCT = _original_arithmetic_token('_products')
_FSUM = _original_arithmetic_token('_fsum')
_RESTORE = _original_restore_token()
_DISPATCHER = _dispatcher_type()


def _support_numpy_token(name):
    entries = _support.__dict__.get('_SPLIT_NUMPY') if type(_support) is types.ModuleType else None
    if type(entries) is not tuple:
        return None
    matches = tuple(entry for entry in entries if type(entry) is tuple and len(entry) == 3
                    and type(entry[0]) is str and entry[0] == name)
    return matches[0] if len(matches) == 1 else None


_EMPTY = _support_numpy_token('empty')
_ISFINITE = _support_numpy_token('isfinite')
_GETERR = _support_numpy_token('geterr')
_MATH_FUNCTIONS = (tuple((name, math.__dict__.get(name)) for name in ('fsum', 'isfinite'))
                   if type(math) is types.ModuleType else ())
_SUPPORT_GUARDS = tuple((name, _token(_support, name)) for name in
                        ('_numpy_entry', '_numpy_definition', '_matches_function'))


def _numpy_entry(entry):
    """Type-check mutable metadata before the existing finite entry proof."""
    if entry is None:
        return False
    name = entry[0]
    function = np.__dict__.get(name)
    if function is not entry[1]:
        return False
    kind = type(function)
    if kind is _UFUNC:
        pass
    elif kind is types.BuiltinFunctionType:
        if not all(type(value) is str for value in
                   (function.__name__, function.__qualname__, function.__module__)):
            return False
    elif kind is types.FunctionType or kind is _DISPATCHER:
        body = function if kind is types.FunctionType else function.__wrapped__
        if (type(body) is not types.FunctionType
                or type(body.__globals__.get('__name__')) is not str
                or not all(type(value) is str for value in
                           (body.__name__, body.__qualname__, body.__module__))):
            return False
    else:
        return False
    return _support._numpy_entry(*entry)


def _callbacks(feature):
    if (feature is not _FEATURE or type(feature) is not types.ModuleType
            or type(_arithmetic) is not types.ModuleType or type(_support) is not types.ModuleType
            or np is not _NUMPY or math is not _MATH
            or type(np) is not types.ModuleType or type(math) is not types.ModuleType
            or type(_NDARRAY) is not type or type(_FLOAT64) is not type or type(_UFUNC) is not type
            or np.__dict__.get('ndarray') is not _NDARRAY
            or np.__dict__.get('float64') is not _FLOAT64 or np.__dict__.get('ufunc') is not _UFUNC
            or sys.modules.get('numpy') is not np or sys.modules.get('math') is not math
            or feature.__dict__.get('np') is not np or _arithmetic.__dict__.get('np') is not np
            or _arithmetic.__dict__.get('math') is not math
            or _support.__dict__.get('np') is not np
            or _support.__dict__.get('_NUMPY_DISPATCHER') is not _DISPATCHER
            or not _matches(_arithmetic, '_products', _PRODUCT)
            or not _matches(_arithmetic, '_fsum', _FSUM)
            or not _matches(feature, '_restore_gradient_terms', _RESTORE)
            or feature.__dict__.get('_products') is not _PRODUCT[0]
            or feature.__dict__.get('_fsum') is not _FSUM[0]):
        return False
    if (not all(_matches(_support, name, token) for name, token in _SUPPORT_GUARDS)
            or not all(_numpy_entry(entry) for entry in (_EMPTY, _ISFINITE, _GETERR))):
        return False
    for name, original in _MATH_FUNCTIONS:
        function = math.__dict__.get(name)
        if (function is not original or type(function) is not types.BuiltinFunctionType
                or type(function.__module__) is not str
                or function.__module__ != 'math' or function.__name__ != name
                or function.__self__ is not math):
            return False
    return not any(value in ('call', 'log') for value in np.geterr().values())


def _native_module(module, feature):
    if type(module) is not types.ModuleType or type(feature) is not types.ModuleType:
        return None
    d = module.__dict__
    if (type(d.get('__name__')) is not str or d['__name__'] != _MODULE_NAME
            or sys.modules.get(_MODULE_NAME) is not module):
        return None
    for name, expected in (('API_VERSION', 1), ('MAX_ROWS', 64), ('MAX_TERMS', 12288)):
        value = d.get(name)
        if type(value) is not int or value != expected:
            return None
    spec, loader = d.get('__spec__'), d.get('__loader__')
    if type(spec) is not ModuleSpec or type(loader) is not ExtensionFileLoader:
        return None
    sd, ld = spec.__dict__, loader.__dict__
    origin, filename, feature_file = sd.get('origin'), d.get('__file__'), feature.__dict__.get('__file__')
    values = (sd.get('name'), ld.get('name'), ld.get('path'), origin, filename, feature_file)
    if not all(type(value) is str for value in values):
        return None
    if (values[0] != _MODULE_NAME or values[1] != _MODULE_NAME
            or sd.get('loader') is not loader or values[2] != origin or origin != filename):
        return None
    normalized = filename.replace('\\', '/')
    if normalized.rpartition('/')[0] != feature_file.replace('\\', '/').rpartition('/')[0]:
        return None
    if not any(normalized.rpartition('/')[2] == '_ordered_reduction_accel'+suffix
               for suffix in EXTENSION_SUFFIXES):
        return None
    entry = d.get('reduce_rows')
    if type(entry) is not types.BuiltinFunctionType or entry.__self__ is not module:
        return None
    values = (entry.__name__, entry.__qualname__, entry.__module__)
    if not all(type(value) is str for value in values):
        return None
    return entry if values == ('reduce_rows', 'reduce_rows', _MODULE_NAME) else None


def _native_entry(feature):
    global _LOAD_ATTEMPTED, _LOAD_COMPLETE, _ACCEL, _ENTRY
    if not _LOAD_ATTEMPTED:
        _LOAD_ATTEMPTED = True
        if not (sys.implementation.name == 'cpython'
                and sys.version_info[0] == 3 and sys.version_info[:2] >= (3, 11)
                and sys.platform in ('win32', 'linux')
                and (_GIL_DISABLED is None or (type(_GIL_DISABLED) is int and _GIL_DISABLED == 0))
                and platform.machine().lower() in ('amd64', 'x86_64')
                and sys.maxsize > 2**32):
            _LOAD_COMPLETE = True
            return None
        # importlib can inspect a cached object's __spec__ before returning it.
        # Qualify every existing value directly, including an explicit None.
        module = sys.modules.get(_MODULE_NAME, _ABSENT)
        if module is _ABSENT:
            try:
                module = importlib.import_module(_MODULE_NAME)
            except (ImportError, OSError):
                _LOAD_COMPLETE = True
                return None
        entry = _native_module(module, feature)
        if entry is None:
            _LOAD_COMPLETE = True
            return None
        _ACCEL, _ENTRY = module, entry
        _LOAD_COMPLETE = True
    if _ACCEL is None:
        return None
    entry = _native_module(_ACCEL, feature)
    return entry if entry is _ENTRY else None


def try_reduce(feature, planes):
    """Return a complete private native result, or None for the old loop.

    Allocation/exception failures unwind to the existing outer replay. This
    facade never catches unexpected exceptions or returns a partial tile.
    """
    if _LOAD_ATTEMPTED is True and _ACCEL is None:
        return None
    if not _callbacks(feature) or type(planes) is not tuple or len(planes) != 6:
        return None
    if not all(type(plane) is np.ndarray and plane.dtype == np.float64
               and plane.ndim == 2 and plane.flags.aligned for plane in planes):
        return None
    shape = planes[0].shape
    if (not all(plane.shape == shape for plane in planes)
            or not 1 < shape[0] <= 64 or shape[1] < 1 or shape[0]*shape[1] > 2048):
        return None
    entry = _native_entry(feature)
    if entry is None:
        return None
    rows, width = shape
    packed = np.empty((rows, 6*width))
    for index, plane in enumerate(planes):
        packed[:, index*width:(index+1)*width] = plane
    output = np.empty(rows)
    status = entry(packed, output)
    if type(status) is not int:
        raise TypeError('Native ordered reduction status must be an exact integer.')
    if status != 0 or not np.isfinite(output).all():
        return None
    return output


# Experimental fused overlay: definition-owned originals, never lazy snapshots.
# All existing bodies and both existing native artifacts remain unchanged.
_FUSION_CALLBACK_TOKENS = tuple(
    (name, (function, function.__code__, function.__defaults__,
            None if function.__kwdefaults__ is None else tuple(function.__kwdefaults__.items())))
    for name in ('try_reduce', '_callbacks', '_native_entry', '_native_module',
                 '_matches', '_owned', '_numpy_entry')
    for function in (globals()[name],))
