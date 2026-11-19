"""Private exact-arithmetic gradient and warm beta tiles for qualified builds.

The first eligible tile follows the original ordered-reduction path. A
previously loaded, definition-qualified ordered facade admits native use.
Missing expansion binaries and all refusals retain the complete original tile.
"""
import importlib
from importlib.machinery import ExtensionFileLoader, EXTENSION_SUFFIXES, ModuleSpec
import math
import sys
import sysconfig
import types

from . import _compensated_residual as _arithmetic
from . import _feature_report_optimization as _support

_NAME = __package__ + '._feature_expansion_accel'
_OLD_NAME = __package__ + '._ordered_gradient_reduction'
_FEATURE_NAME = __package__ + '._linear_feature_diagnostics'
_ABSENT = object()
_LOAD_ATTEMPTED = False
_ACCEL = None
_ENTRY = None
_BETA_ENTRY = None
_CONTEXT = None
_GIL_DISABLED = sysconfig.get_config_var('Py_GIL_DISABLED')
_GETPROFILE, _GETTRACE = sys.getprofile, sys.gettrace


def _hooks_absent():
    for name, function in (('getprofile', _GETPROFILE), ('gettrace', _GETTRACE)):
        if (type(function) is not types.BuiltinFunctionType
                or function.__self__ is not sys or function.__name__ != name
                or sys.__dict__.get(name) is not function):
            return False
    return _GETPROFILE() is None and _GETTRACE() is None


def _matches(function, token):
    if (type(token) is not tuple or len(token) != 4
            or type(function) is not types.FunctionType or function is not token[0]
            or function.__code__ is not token[1] or function.__defaults__ is not token[2]):
        return False
    actual, expected = function.__kwdefaults__, token[3]
    if expected is None:
        return actual is None
    return (type(actual) is dict and len(actual) == len(expected)
            and all(type(key) is str for key in actual)
            and all(type(pair) is tuple and len(pair) == 2 and type(pair[0]) is str
                    and actual.get(pair[0], _ABSENT) is pair[1] for pair in expected))


def _definition_token(entry):
    if type(entry) is not tuple or len(entry) != 6:
        return None
    owner, name, function, code, defaults, kwdefaults = entry
    if (type(owner) is not types.ModuleType or type(name) is not str
            or type(kwdefaults) is not dict or any(type(k) is not str for k in kwdefaults)):
        return None
    return function, code, defaults, tuple(kwdefaults.items()) if kwdefaults else None


def _instance_dictionary(instance, cls):
    """Only the genuine exact-type __dict__ getset may materialize a dict."""
    if type(cls) is not type or type(instance) is not cls:
        return None
    descriptor = cls.__dict__.get('__dict__')
    if (type(descriptor) is not types.GetSetDescriptorType
            or descriptor.__objclass__ is not cls or descriptor.__name__ != '__dict__'):
        return None
    value = descriptor.__get__(instance, cls)
    return value if type(value) is dict else None


def _member(cls, name, token):
    if type(token) is not tuple or len(token) != 3 or type(cls) is not type:
        return False
    entry, function, state = token
    if cls.__dict__.get(name, _support._ABSENT) is not entry:
        return False
    if entry is _support._ABSENT:
        return True
    if type(entry) is staticmethod:
        actual = entry.__func__
    elif type(entry) is property:
        actual = entry.fget
    elif type(entry) is _support._CachedProperty:
        values = _instance_dictionary(entry, _support._CachedProperty)
        if values is None:
            return False
        actual = values.get('func')
    else:
        actual = entry
    return actual is function and _matches(function, state)


def _initial(feature, old):
    if (not _hooks_absent() or type(feature) is not types.ModuleType
            or type(old) is not types.ModuleType or type(_support) is not types.ModuleType
            or type(_arithmetic) is not types.ModuleType
            or sys.modules.get(_FEATURE_NAME) is not feature
            or sys.modules.get(_OLD_NAME) is not old):
        return False
    # Importing the facade does not finish its lazy native loader. Never bind
    # a cold or in-flight tuple: the original tile settles it naturally first.
    if (old.__dict__.get('_LOAD_ATTEMPTED') is not True
            or old.__dict__.get('_LOAD_COMPLETE') is not True):
        return False
    tokens = old.__dict__.get('_FUSION_CALLBACK_TOKENS')
    if type(tokens) is not tuple or len(tokens) != 7:
        return False
    for item in tokens:
        if type(item) is not tuple or len(item) != 2:
            return False
        name, token = item
        if type(name) is not str or not _matches(old.__dict__.get(name), token):
            return False
    # These helper identities are checked before invoking any old helper.
    if not old._callbacks(feature):
        return False
    for entry in _support._SPLIT_FUNCTIONS:
        token = _definition_token(entry)
        if token is None or not _matches(entry[0].__dict__.get(entry[1]), token):
            return False
    if not all(old._numpy_entry(entry) for entry in _support._SPLIT_NUMPY):
        return False
    np = _support._NUMPY_MODULE
    for name, cls, owner, members in (
            ('finfo', _support._FINFO, _support._FINFO_OWNER, _support._FINFO_MEMBERS),
            ('errstate', _support._ERRSTATE, _support._ERRSTATE_OWNER, _support._ERRSTATE_MEMBERS)):
        if (type(cls) is not type or type(owner) is not types.ModuleType
                or np.__dict__.get(name) is not cls or owner.__dict__.get(name) is not cls
                or cls.__bases__ != (object,)
                or not all(_member(cls, member, token) for member, token in members)):
            return False
    return True


def _native_module(module, feature):
    if type(module) is not types.ModuleType:
        return None
    d = module.__dict__
    for key, expected in (('API_VERSION', 2), ('COMPILED_PY_HEX', sys.hexversion),
                          ('MAX_ROWS', 64), ('MAX_ENTRIES', 16384), ('MAX_WIDTH', 1024)):
        if type(d.get(key)) is not int or d[key] != expected:
            return None
    if (type(d.get('__name__')) is not str or d['__name__'] != _NAME
            or sys.modules.get(_NAME) is not module):
        return None
    spec, loader = d.get('__spec__'), d.get('__loader__')
    if type(spec) is not ModuleSpec or type(loader) is not ExtensionFileLoader:
        return None
    sd, ld = spec.__dict__, loader.__dict__
    values = (sd.get('name'), ld.get('name'), ld.get('path'), sd.get('origin'),
              d.get('__file__'), feature.__dict__.get('__file__'))
    if not all(type(v) is str for v in values):
        return None
    if (values[0] != _NAME or values[1] != _NAME or sd.get('loader') is not loader
            or values[2] != values[3] or values[3] != values[4]):
        return None
    path = values[4].replace('\\', '/')
    if (path.rpartition('/')[0] != values[5].replace('\\', '/').rpartition('/')[0]
            or not any(path.rpartition('/')[2] == '_feature_expansion_accel' + suffix
                       for suffix in EXTENSION_SUFFIXES)):
        return None
    name = 'gradient_from_splits'
    function = d.get(name)
    if (type(function) is not types.BuiltinFunctionType or function.__self__ is not module
            or not all(type(v) is str for v in
                       (function.__name__, function.__qualname__, function.__module__))):
        return None
    return function if (function.__name__, function.__qualname__, function.__module__) == (name, name, _NAME) else None


def _native_entry(feature):
    global _LOAD_ATTEMPTED, _ACCEL, _ENTRY
    if not _LOAD_ATTEMPTED:
        _LOAD_ATTEMPTED = True
        if (sys.implementation.name != 'cpython' or sys.version_info[:2] != (3, 12)
                or sys.platform not in ('linux', 'win32') or sys.maxsize <= 2**32
                or _GIL_DISABLED not in (None, 0)):
            return None
        module = sys.modules.get(_NAME, _ABSENT)
        if module is _ABSENT:
            try:
                module = importlib.import_module(_NAME)
            except (ImportError, OSError):
                return None
        entry = _native_module(module, feature)
        if entry is None:
            return None
        _ACCEL, _ENTRY = module, entry
    if _ACCEL is None:
        return None
    entry = _native_module(_ACCEL, feature)
    return entry if entry is _ENTRY else None


def _make_context(feature, old):
    """Bind only the finite callback family this exact gradient skips.

    Expected code/default states originate before preparation, or at the
    definition in this experimental overlay. No current code is blessed here.
    The C bridge rechecks live state after all allocations and acquisitions.
    """
    global _BETA_ENTRY
    np = _support._NUMPY_MODULE
    beta_entry = _beta_native_entry(feature)
    dicts, bindings, functions, classes, instances = [], [], [], [], []

    def bind(d, name, value):
        if type(d) is not dict or type(name) is not str:
            raise TypeError('Invalid private fused context owner')
        if not any(d is previous for previous in dicts):
            dicts.append(d)
        bindings.append((d, name, value))

    def module(owner):
        if type(owner) is not types.ModuleType:
            raise TypeError('Invalid private fused module owner')
        name = owner.__dict__.get('__name__')
        if type(name) is not str:
            raise TypeError('Invalid private fused module name')
        bind(sys.modules, name, owner)

    def function(owner, name, token):
        if token is None or not _matches(owner.__dict__.get(name), token):
            return False
        f, code, defaults, kw = token
        bind(owner.__dict__, name, f)
        # Metadata must agree with original code and exact owning globals.
        if (type(f.__module__) is not str or type(f.__name__) is not str
                or type(f.__qualname__) is not str or f.__globals__ is not owner.__dict__
                or f.__module__ != owner.__dict__.get('__name__')
                or f.__name__ != code.co_name or f.__qualname__ != code.co_qualname
                or '__wrapped__' in f.__dict__):
            return False
        functions.append((f, code, defaults, kw, owner.__dict__,
                          f.__module__, f.__name__, f.__qualname__))
        return True

    def body(token):
        if token is None:
            return True
        f = token[0]
        if type(f) is not types.FunctionType:
            return False
        owner = sys.modules.get(f.__globals__.get('__name__'))
        if type(owner) is not types.ModuleType or f.__globals__ is not owner.__dict__:
            return False
        module(owner)
        # NumPy decorated bodies need not be the public owner binding.
        if not _matches(f, token):
            return False
        owner_name = owner.__dict__.get('__name__')
        if (not all(type(value) is str for value in (owner_name, f.__module__, f.__name__, f.__qualname__))
                or not (f.__module__ == owner_name
                        or (f.__module__ == 'numpy' and owner_name.startswith('numpy.')))
                or f.__name__ != token[1].co_name or f.__qualname__ != token[1].co_qualname):
            return False
        functions.append((f, token[1], token[2], token[3], owner.__dict__,
                          f.__module__, f.__name__, f.__qualname__))
        return True

    for owner in (feature, old, _support, _arithmetic, np, math, sys, _ACCEL,
                  sys.modules[__name__]):
        module(owner)
    for owner in (feature, _support, _arithmetic, old):
        bind(owner.__dict__, 'np', np)
    for owner in (feature, _support, _arithmetic, old):
        bind(owner.__dict__, 'math', math)
    for entry in _support._SPLIT_FUNCTIONS:
        if not function(entry[0], entry[1], _definition_token(entry)):
            return None
    for name in ('_products', '_fsum'):
        bind(feature.__dict__, name, _arithmetic.__dict__[name])
    if not function(feature, '_restore_gradient_terms', old._RESTORE):
        return None
    if not function(feature, '_gradient_tile', (
            feature._ORIGINAL_GRADIENT_TILE, feature._ORIGINAL_GRADIENT_TILE_CODE,
            None, (('_splits', None),))):
        return None
    if not function(feature, '_try_fused_gradient', feature._FUSION_HELPER_TOKEN):
        return None
    if beta_entry is not None:
        if not function(feature, '_beta_tile', (
                feature._ORIGINAL_BETA_TILE, feature._ORIGINAL_BETA_TILE_CODE,
                None, (('_split', None),))):
            return None
        if not function(feature, '_try_fused_beta', feature._FUSION_BETA_HELPER_TOKEN):
            return None
        bind(feature.__dict__, '_fusion_sys', sys)
    for name, token in old._FUSION_CALLBACK_TOKENS:
        if not function(old, name, token):
            return None
    for name, token in old._SUPPORT_GUARDS:
        if not function(_support, name, token):
            return None
    for name in ('_PRODUCT', '_FSUM', '_RESTORE', '_LOAD_ATTEMPTED', '_ACCEL', '_ENTRY',
                 '_NDARRAY', '_FLOAT64', '_UFUNC', '_DISPATCHER'):
        bind(old.__dict__, name, old.__dict__[name])
    bind(old.__dict__, '_LOAD_COMPLETE', True)
    if old._ACCEL is not None:
        if old._native_module(old._ACCEL, feature) is not old._ENTRY:
            return None
        module(old._ACCEL)
        bind(old._ACCEL.__dict__, 'reduce_rows', old._ENTRY)
    for name, original in _support._SPLIT_MATH:
        bind(math.__dict__, name, original)
    for name in ('ndarray', 'float64', 'ufunc'):
        bind(np.__dict__, name, old.__dict__[{'ndarray': '_NDARRAY', 'float64': '_FLOAT64', 'ufunc': '_UFUNC'}[name]])
    for name, original, token in _support._SPLIT_NUMPY:
        bind(np.__dict__, name, original)
        if not body(token):
            return None
        if type(original) is old._DISPATCHER:
            values = _instance_dictionary(original, old._DISPATCHER)
            if values is None or values.get('__wrapped__') is not token[0]:
                return None
            instances.append((original, old._DISPATCHER, values))
            bind(values, '__wrapped__', token[0])
    for name, cls, owner, members in (
            ('finfo', _support._FINFO, _support._FINFO_OWNER, _support._FINFO_MEMBERS),
            ('errstate', _support._ERRSTATE, _support._ERRSTATE_OWNER, _support._ERRSTATE_MEMBERS)):
        module(owner)
        bind(np.__dict__, name, cls)
        bind(owner.__dict__, name, cls)
        # C reads tp_dict directly and compares all snapshot entries; it does
        # not call mappingproxy descriptors at the final admission boundary.
        members_dict = dict(cls.__dict__)
        classes.append((cls, members_dict, cls.__bases__))
        for member, token in members:
            entry, f, state = token
            if entry is _support._ABSENT:
                continue
            bind(members_dict, member, entry)
            if not body(state):
                return None
            if type(entry) is _support._CachedProperty:
                values = _instance_dictionary(entry, _support._CachedProperty)
                if values is None or values.get('func') is not f:
                    return None
                instances.append((entry, _support._CachedProperty, values))
                bind(values, 'func', f)
    # The bridge resolves exact type dictionaries before the final gate. The
    # copied dictionaries above are snapshots, with live member checks in C.
    owner = sys.modules[__name__]
    for name, token in _OWN_TOKENS:
        if not function(owner, name, token):
            return None
    bind(owner.__dict__, '_ACCEL', _ACCEL)
    bind(owner.__dict__, '_ENTRY', _ENTRY)
    bind(owner.__dict__, '_BETA_ENTRY', beta_entry)
    bind(_ACCEL.__dict__, _ENTRY.__name__, _ENTRY)
    if beta_entry is not None:
        bind(_ACCEL.__dict__, 'BETA_API_VERSION', _ACCEL.__dict__['BETA_API_VERSION'])
        bind(_ACCEL.__dict__, 'beta_from_split', beta_entry)
    # Captured only in the natural gradient context-construction path. Beta
    # never calls this routine, never rebases and never initializes a loader.
    _BETA_ENTRY = beta_entry
    return (tuple(dicts), tuple(bindings), tuple(functions), tuple(classes),
            tuple(instances), (owner.__dict__, '_CONTEXT'))


def _array(value, old, dimensions, integer=False):
    if type(value) is not old._NDARRAY:
        return False
    dtype = value.dtype
    return (value.ndim == dimensions and value.flags.aligned and dtype.isnative
            and dtype.kind == ('i' if integer else 'f')
            and dtype.itemsize == (4 if integer else 8))


def try_gradient(feature, columns, prepared, splits):
    global _CONTEXT
    if _LOAD_ATTEMPTED is True and _ACCEL is None:
        return None
    old = sys.modules.get(_OLD_NAME)
    if not _initial(feature, old):
        return None
    if (type(prepared) is not tuple or len(prepared) != 6
            or type(splits) is not tuple or len(splits) != 3
            or not _array(columns, old, 2)):
        return None
    rows, width = columns.shape
    if not 2 <= rows <= 64 or not 1 <= width <= 1024 or rows*width > 16384:
        return None
    z, ze, high, low, ae, pe = prepared
    if (type(pe) is not int or not all(_array(v, old, 1) and len(v) == width for v in (z, high, low))
            or not all(_array(v, old, 1, True) and len(v) == width for v in (ze, ae))):
        return None
    rhs = []
    for split in splits:
        if type(split) is not tuple or len(split) != 4:
            return None
        if not all(_array(v, old, 1, i == 1) and len(v) == width
                   and v.flags.c_contiguous for i, v in enumerate(split)):
            return None
        rhs.extend(split)
    if not ze.flags.c_contiguous or not ae.flags.c_contiguous:
        return None
    entry = _native_entry(feature)
    if entry is None or not _initial(feature, old):
        return None
    if _CONTEXT is None:
        _CONTEXT = _make_context(feature, old)
    if _CONTEXT is None:
        return None
    np = _support._NUMPY_MODULE
    tile = columns if columns.flags.c_contiguous else columns.copy(order='C')
    # Invoke the previously qualified genuine allocator. A late alias change
    # must be refused at the C barrier and replayed at its original position.
    output = old._EMPTY[1](rows, dtype='float64')
    arrays = (tile, *rhs, ze, ae, output)
    buffers = tuple(memoryview(value) for value in arrays)
    bundle = (_CONTEXT, buffers, pe)
    status = entry(bundle)
    if type(status) is not int:
        raise TypeError('Native fused gradient status must be an exact integer.')
    if status != 0:
        return None
    # The native success contract includes finite complete output; no extra
    # numerical operation or mutable NumPy callback is needed after success.
    return output


def _beta_native_entry(feature):
    # Validate only an already loaded genuine gradient module; no import/cache
    # initialization. An old G artifact remains usable for gradients alone.
    if (_ACCEL is None or _ENTRY is None
            or _native_module(_ACCEL, feature) is not _ENTRY):
        return None
    d = _ACCEL.__dict__
    if type(d.get('BETA_API_VERSION')) is not int or d['BETA_API_VERSION'] != 1:
        return None
    function = d.get('beta_from_split')
    if (type(function) is not types.BuiltinFunctionType or function.__self__ is not _ACCEL
            or not all(type(value) is str for value in
                       (function.__name__, function.__qualname__, function.__module__))):
        return None
    return function if (function.__name__, function.__qualname__, function.__module__) == (
        'beta_from_split', 'beta_from_split', _NAME) else None


def try_beta(feature, columns, prepared, split):
    # Warm-only: do not inspect unknown inputs or import/initialize on a miss.
    if _CONTEXT is None or _BETA_ENTRY is None:
        return None
    old = sys.modules.get(_OLD_NAME)
    if not _initial(feature, old):
        return None
    entry = _beta_native_entry(feature)
    if entry is None or entry is not _BETA_ENTRY:
        return None
    if (type(prepared) is not tuple or len(prepared) != 2
            or type(split) is not tuple or len(split) != 4
            or not _array(columns, old, 2)):
        return None
    rows, width = columns.shape
    if not 2 <= rows <= 64 or not 1 <= width <= 1024 or rows*width > 16384:
        return None
    mantissa, exponent = prepared
    if (not _array(mantissa, old, 1) or len(mantissa) != width
            or not _array(exponent, old, 1, True) or len(exponent) != width
            or not exponent.flags.c_contiguous):
        return None
    if not all(_array(value, old, 1, i == 1) and len(value) == width
               and value.flags.c_contiguous for i, value in enumerate(split)):
        return None
    tile = columns if columns.flags.c_contiguous else columns.copy(order='C')
    output = old._EMPTY[1](rows, dtype='float64')
    arrays = (tile, *split, exponent, output)
    buffers = tuple(memoryview(value) for value in arrays)
    status = entry((_CONTEXT, buffers))
    if type(status) is not int:
        raise TypeError('Native fused beta status must be an exact integer.')
    return output if status == 0 else None


_OWN_TOKENS = tuple((name, (function, function.__code__, function.__defaults__,
                          None if function.__kwdefaults__ is None else tuple(function.__kwdefaults__.items())))
                    for name, function in tuple(globals().items())
                    if type(function) is types.FunctionType and function.__globals__ is globals())
_FUSION_ENTRY_TOKEN = (try_gradient, try_gradient.__code__, None, None)
_FUSION_BETA_ENTRY_TOKEN = (try_beta, try_beta.__code__, None, None)
