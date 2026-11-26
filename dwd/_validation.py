"""Validation shared by real-valued estimator inputs."""

from collections import UserString, deque
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from fractions import Fraction
from inspect import getattr_static
from itertools import chain, repeat
from math import gcd, isfinite
from numbers import Integral, Real
from operator import is_
from sys import modules, version_info
from types import GetSetDescriptorType, MethodType, SimpleNamespace

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
    assert_all_finite, check_array, check_X_y, check_consistent_length, column_or_1d,
    validate_data,
)


_SKLEARN_NATIVE_FLOAT_DTYPES = frozenset(map(np.dtype, (np.float16, np.float32, np.float64)))


def _comparison_labels(y):
    """Preserve complete variable-width strings during class comparisons."""
    # Some NumPy versions stop comparing same-length StringDType values at
    # their first embedded NUL, including during sorting and unique grouping.
    return np.asarray(y, dtype=object) if y.dtype.kind == 'T' else y


def _sorted_class_labels(y):
    """Return sorted, distinct representatives in the training label dtype."""
    classes = np.unique(_comparison_labels(y))
    return classes.astype(y.dtype) if y.dtype.kind == 'T' else classes


def _check_other_float_labels(y, *, discrete=False):
    """Handle floating dtypes outside the set recognized by sklearn 1.6."""
    if y.dtype.kind != 'f' or y.dtype in _SKLEARN_NATIVE_FLOAT_DTYPES:
        return False
    # Check in the represented precision. Both float64 and signed integer
    # casts can erase distinctions between legitimate extended-precision IDs.
    if not np.isfinite(y).all():
        raise ValueError('Input y contains NaN or infinity.')
    if discrete and not (y == np.trunc(y)).all():
        raise ValueError('Unknown label type: continuous. Class labels must be discrete.')
    return True


def _discrete_training_labels(y):
    """Return a checked label vector suitable for nested sklearn classifiers."""
    if _check_other_float_labels(y, discrete=True):
        # A nested SVC checks its own target again. Supply exact native IDs
        # when possible; for wider domains retain the original dtype and its
        # classification support in the installed sklearn version.
        y = _wide_integral_float_target(y, native_only=True)
    else:
        _reject_nat_training_labels(y)
    check_classification_targets(_comparison_labels(y))
    return y


def _pandas_target_array(y, *, include_boxed=False):
    """Get extension storage for vector targets without importing pandas."""
    pd = modules.get('pandas')
    if pd is None or isinstance(y, np.ndarray):
        return None
    if isinstance(y, pd.DataFrame):
        if len(y._mgr.items) != 1:
            return None
        values = y._mgr.iget_values(0)
    elif isinstance(y, pd.Series):
        values = y._values
    elif isinstance(y, pd.Index):
        if isinstance(y, (pd.RangeIndex, pd.MultiIndex)):
            return None
        values = y._data
    else:
        return y if isinstance(y, pd.api.extensions.ExtensionArray) else None
    # pandas projects the backing values. Its public selectors and .array
    # accessor can be overridden independently of this conversion.
    if isinstance(values, pd.api.extensions.ExtensionArray) or include_boxed and values.dtype == object:
        return values
    return None


def _numeric_abc_instance(value, abc):
    try:
        return isinstance(value, abc)
    except TypeError:
        cls = type(value)
        if type(cls).__hash__ is not None:
            raise
    # ABC caches and virtual registries require hashable classes. An
    # unhashable subclass still inherits its bases' numeric registrations.
    return any(issubclass(base, abc) for base in type.__getattribute__(cls, '__mro__')
               if type(base).__hash__ is not None)


def _is_integral_label(value):
    if isinstance(value, Decimal):
        return value.is_finite() and value == value.to_integral_value()
    if _numeric_abc_instance(value, Fraction):
        return value.denominator == 1
    return (isinstance(value, np.bool_) or _numeric_abc_instance(value, Integral) or
            isinstance(value, (float, np.floating)) and np.isfinite(value)
            and value == np.trunc(value))


def _python_label(value):
    """Unbox numeric IDs so scalar comparisons never promote an integer to float."""
    if isinstance(value, np.ndarray) and value.ndim == 0 and value.dtype.kind in 'biuf':
        value = value[()]
    if isinstance(value, np.generic) and np.generic.dtype.__get__(value).kind in 'biuf':
        # A numeric subclass can override item(), __int__ or __float__.
        # Read its represented scalar directly through NumPy's descriptor.
        value = np.generic.item(value)
        if isinstance(value, np.floating) and np.isfinite(value):
            # Extended precision has no Python float counterpart.
            if value == np.trunc(value):
                return int(value)
            return Fraction(*value.as_integer_ratio())
    if (isinstance(value, Decimal) or _numeric_abc_instance(value, Fraction)) and _is_integral_label(value):
        return int(value)
    return value


def _scalar_reader(base):
    if issubclass(base, np.integer):
        return base.__int__
    if issubclass(base, np.floating) and base is not np.longdouble:
        return base.__float__
    # Extended precision must not pass through binary64.
    return _python_label


# Compare types by identity, including for user classes with unhashable metaclasses.
_NUMPY_READERS = {id(t): _scalar_reader(t) for t in
                  (np.dtype(code).type for code in '?' + np.typecodes['AllInteger'] + np.typecodes['Float'])}
_NUMPY_REAL_TYPES = frozenset(_NUMPY_READERS)
_PLAIN_INTEGER_TYPES = frozenset(id(np.dtype(code).type) for code in
                                 '?' + np.typecodes['AllInteger']) | {id(bool), id(int)}
_PLAIN_REAL_TYPES = _NUMPY_REAL_TYPES.union(map(id, (bool, int, float, Fraction, Decimal)))
_PLAIN_FLOAT_TYPES = {id(float)} | {id(np.dtype(code).type) for code in np.typecodes['Float']}
_FLOAT_OR_DECIMAL_TYPES = _PLAIN_FLOAT_TYPES | {id(Decimal)}
_TEXT_TYPES = frozenset(map(id, (str, bytes, np.str_, np.bytes_)))
_COMPLEX_TYPES = frozenset(id(np.dtype(code).type) for code in np.typecodes['Complex']) | {id(complex)}
_TEXT_OR_NONE_TYPES = _TEXT_TYPES | {id(type(None))}
_COMPARISONS = ('__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__')
_ARRAY_PROTOCOLS = ('__array__', '__array_interface__', '__array_struct__')
_EXTENSION_PROTOCOLS = (*_ARRAY_PROTOCOLS, 'to_numpy')
_TEXT_FIELDS = (*_COMPARISONS, '__hash__', '__getattr__', '__getattribute__', '__str__', '__bytes__',
                '__int__', '__float__', '__complex__', '__bool__', '__index__', '__buffer__', '__len__', '__iter__', '__getitem__',
                'encode', 'decode', 'dtype', '__array_priority__', '__array_ufunc__', '__array_function__',
                *_ARRAY_PROTOCOLS)
_ITERATED_NUMBER_TYPES = _PLAIN_REAL_TYPES | _COMPLEX_TYPES
_NUMBER_TEXT_FIELDS = (*_TEXT_FIELDS, '__repr__', '__format__', '__array_wrap__', '__array_prepare__')
_MISSING = object()
_array_shape = np.ndarray.shape.__get__


def _inherits_numeric_order(cls, base, operations):
    return all(getattr(cls, op, None) is getattr(base, op, None) for op in operations)


def _overrides_numeric_conversion(cls):
    base = next((b for b in cls.__mro__ if id(b) in _NUMPY_REAL_TYPES or b is int or b is float), None)
    return base is not None and not _inherits_numeric_order(cls, base, ('__int__', '__float__', '__index__'))


def _base_numeric_scalar(value):
    value = _unbox_numeric_array(value)
    if isinstance(value, np.generic):
        return np.generic.__array__(value)[()]
    if isinstance(value, int):
        return int.__int__(value)
    if isinstance(value, float):
        return float.__float__(value)
    return value


def _inherited_numeric_types(types):
    """Identify subclasses whose comparisons still use their scalar base."""
    readers = _NUMPY_READERS.copy()
    tagged_types = set()
    for key, t in types.items():
        base = next((b for b in t.__mro__ if id(b) in _NUMPY_REAL_TYPES), None)
        if base is not None:
            if _inherits_numeric_order(t, base, (*_COMPARISONS, '__array_ufunc__')):
                readers[key] = readers[id(base)]
        elif issubclass(t, (int, float)):
            base = int if issubclass(t, int) else float
            if _inherits_numeric_order(t, base, (*_COMPARISONS, '__array_ufunc__')):
                tagged_types.add(key)
    return readers, tagged_types


def _integer_dtype(low, high):
    if -(2**63) <= low and high < 2**63:
        return np.int64
    if 0 <= low and high < 2**64:
        return np.uint64
    return object


def _integer_array(integers, shape):
    dtype = _integer_dtype(min(integers), max(integers))
    return np.array(integers, dtype=dtype).reshape(shape)


def _integral_target(original):
    return _integer_array(list(map(int, original.flat)), original.shape)


def _numeric_key(value, tagged_types):
    if id(type(value)) in tagged_types:
        return int.__int__(value) if isinstance(value, int) else float.__float__(value)
    return _python_label(value)


def _same_binary_membership(raw, representatives, negative, positive):
    return (np.all(negative ^ positive)
            and np.array_equal(raw == representatives[:1], negative)
            and np.array_equal(raw == representatives[1:2], positive))


def _same_binary_order(values, exact, tagged_types):
    """Retain representatives only if actual class ordering and both masks agree."""
    try:
        representatives = np.unique(values)
        if len(representatives) != 2:
            return False
        low, high = (_numeric_key(v, tagged_types) for v in representatives)
        if not low < high:
            return False
        if tagged_types:
            exact = [_numeric_key(v, tagged_types) for v in exact]
        keys = np.asarray(exact, dtype=object)
        negative, positive = keys == low, keys == high
        return _same_binary_membership(values.ravel(), representatives, negative, positive)
    except (TypeError, ValueError, ArithmeticError):
        return False


def _consistent_custom_groups(values, ordinary, tagged_types):
    assert_all_finite(values, input_name='y')
    raw = values.ravel()
    peer_rows = np.fromiter((id(type(v)) in ordinary for v in raw), dtype=bool, count=len(raw))
    peers = raw[peer_rows]
    keys = np.fromiter((_numeric_key(v, tagged_types) for v in peers), dtype=object, count=len(peers))
    exact_ids, exact_groups = np.unique(keys, return_inverse=True)
    _, peer_groups = np.unique(peers, return_inverse=True)
    if not np.array_equal(peer_groups, exact_groups):
        return False
    representatives = np.unique(raw)
    if len(representatives) == 2:
        negative = exact_label_matches(raw, representatives[:1])
        positive = exact_label_matches(raw, representatives[1:2])
        if not _same_binary_membership(raw, representatives, negative, positive):
            return False
        # The full-target class masks, restricted to ordinary numeric peers,
        # must separate and order their exact IDs even for a custom representative.
        _, raw_groups = np.unique(positive[peer_rows], return_inverse=True)
        return np.array_equal(raw_groups, exact_groups)
    return len(exact_ids) <= len(representatives)


def _unbox_numeric_array(value, *, include_text=False):
    """Read scalar labels through ordinary zero-dimensional NumPy boxes."""
    if type(value) is not np.ndarray or value.ndim != 0:
        return value
    if value.dtype.kind in 'biuf' or include_text and value.dtype.kind in 'cUST':
        return value[()]
    if value.dtype.kind == 'O':
        scalar = value[()]
        if type(scalar) is np.ndarray:
            seen = {id(value)}
            while type(scalar) is np.ndarray and scalar.ndim == 0:
                if id(scalar) in seen:
                    raise ValueError('A class label contains a cyclic scalar array.')
                seen.add(id(scalar))
                scalar = scalar[()]
        if (isinstance(scalar, (int, float, Decimal)) or
                include_text and isinstance(scalar, (str, bytes, complex, np.complexfloating)) or
                isinstance(scalar, np.generic) and np.generic.dtype.__get__(scalar).kind in 'biuf' or
                _numeric_abc_instance(scalar, Fraction)):
            return scalar
    return value


def _require_finite_boxed_reals(values, kinds):
    # Object-array validation in sklearn detects NaN but can accept infinity.
    # Read the stored float without invoking a label's custom conversion.
    if len(kinds) == 1 and kinds <= _PLAIN_FLOAT_TYPES:
        if not np.isfinite(values.astype(type(values.flat[0]))).all():
            raise ValueError('Input y contains NaN or infinity.')
        return
    for v in values.flat:
        if isinstance(v, np.floating):
            number = np.generic.item(v)
            finite = np.isfinite(number) if isinstance(number, np.floating) else isfinite(number)
        elif isinstance(v, float):
            finite = isfinite(float.__float__(v))
        elif isinstance(v, Decimal):
            finite = Decimal.is_finite(v)
        else:
            continue
        if not finite:
            raise ValueError('Input y contains NaN or infinity.')


def _boxed_numeric_target(source, values, *, mode='normalize'):
    """Prevent mixed NumPy scalars from comparing adjacent class IDs as equal."""
    if values.ndim not in (1, 2) or (values.ndim == 2 and values.shape[1] != 1):
        return source
    kinds = set(map(id, map(type, values.flat)))
    has_arrays = id(np.ndarray) in kinds
    if has_arrays:
        # A scalar array can carry a different integer or floating dtype at
        # each position, even if its Python type is identical throughout.
        values = np.fromiter(map(_unbox_numeric_array, values.flat), dtype=object,
                             count=values.size).reshape(values.shape)
        kinds = set(map(id, map(type, values.flat)))
    if not has_arrays and len(kinds) < 2 and (
            not kinds or kinds <= _PLAIN_REAL_TYPES or
            not isinstance(values.flat[0], (int, float, np.integer, np.floating, Decimal))):
        if not kinds.isdisjoint(_FLOAT_OR_DECIMAL_TYPES):
            _require_finite_boxed_reals(values, kinds)
        return source
    shape = getattr(source, 'shape', values.shape)
    if not kinds <= _PLAIN_REAL_TYPES:
        # Inspect each user type once. An inherited NumPy comparison can
        # round a peer just like the base scalar's comparison. A user ufunc
        # override, however, can define reflected equality and ordering.
        types = {id(t): t for t in map(type, values.flat) if id(t) not in _PLAIN_REAL_TYPES}
        readers, tagged_types = _inherited_numeric_types(types)
        numpy_types = readers.keys()
        ordinary = _PLAIN_REAL_TYPES | numpy_types | tagged_types
        numeric_subclass = any(issubclass(t, (int, float, np.integer, np.floating)) for t in types.values())
        if numeric_subclass:
            # Check original operands before projection or paired comparisons.
            # Other observations cannot change a label's declared domain.
            require_real(values, 'y')
        converted = {k for k, t in types.items() if _overrides_numeric_conversion(t)}
        has_float = not kinds.isdisjoint(_PLAIN_FLOAT_TYPES) or any(
            issubclass(t, (float, np.floating)) for t in types.values())
        other_types = kinds - ordinary
        mixed_text = bool(other_types) and other_types <= _TEXT_OR_NONE_TYPES
        if (not mixed_text or converted) and (
                has_float or id(Decimal) in kinds or any(issubclass(t, Decimal) for t in types.values())):
            _require_finite_boxed_reals(values, kinds)
        if kinds <= ordinary:
            if converted:
                scalars = [_base_numeric_scalar(v) if id(type(v)) in converted else v for v in values.flat]
                if mode == 'fit':
                    # The fallback needs scalar dtypes for normalization. The
                    # membership oracle needs numbers that cannot round an ID.
                    keys = list(map(_python_label, scalars))
                    if _same_binary_order(values, keys, tagged_types):
                        return values.reshape(shape)
                return np.asarray(_exact_numeric_sequence(scalars, mode=mode)).reshape(shape)
            if kinds.isdisjoint(numpy_types) or (mode == 'fit' and not has_arrays and len(kinds) == 1):
                return values.reshape(shape) if has_arrays else source
            # Keep ordinary tagged Python labels when unboxing their NumPy
            # peers. Only NumPy storage supplies the value of a NumPy subclass.
            extended = id(np.longdouble) in kinds or any(issubclass(t, np.longdouble) for t in types.values())
            if kinds <= numpy_types:
                exact = [readers[id(type(v))](v) for v in values.flat]
            else:
                exact = [readers[id(type(v))](v) if id(type(v)) in numpy_types else v for v in values.flat]
            if (mode == 'fit' and not types.keys().isdisjoint(numpy_types)
                    and _same_binary_order(values, exact, tagged_types)):
                return values.reshape(shape)
            if not tagged_types:
                if extended or not kinds.isdisjoint((id(Fraction), id(Decimal))):
                    integral = all(map(_is_integral_label, exact))
                else:
                    integral = all(v.is_integer() for v in exact if type(v) is float)
                if integral:
                    return _integer_array(list(map(int, exact)), shape)
            return np.asarray(exact, dtype=object).reshape(shape)
        if mode in ('observations', 'score'):
            # Scoring compares aligned pairs and monitoring has fixed training
            # representatives. Held observations never define their own classes.
            return values.reshape(shape)
        if mixed_text:
            # Plain text, bytes and None have no class ordering with ordinary
            # numbers. Let the caller validate these mixed object targets.
            return values.reshape(shape) if has_arrays else source
        if (kinds.isdisjoint(ordinary) and
                not any(issubclass(t, (int, float, np.integer, np.floating)) for t in types.values())):
            return values.reshape(shape) if has_arrays else source
        if not numeric_subclass:
            require_real(values, 'y')
        if _consistent_custom_groups(values, ordinary, tagged_types):
            return values.reshape(shape)
        raise ValueError('Boxed numeric class IDs compare inconsistently alongside custom labels. '
                         'Encode the classes with distinct exact labels before fitting.')
    if not has_arrays and kinds.isdisjoint(_NUMPY_REAL_TYPES):
        if not kinds.isdisjoint(_FLOAT_OR_DECIMAL_TYPES):
            _require_finite_boxed_reals(values, kinds)
        return source
    if kinds.isdisjoint((id(Fraction), id(Decimal))):
        integral = all(v.is_integer() for v in values.flat if isinstance(v, (float, np.floating)))
    else:
        integral = all(_is_integral_label(v) for v in values.flat)
    if integral:
        return _integral_target(values).reshape(shape)
    _require_finite_boxed_reals(values, kinds)
    return np.fromiter(map(_python_label, values.flat), dtype=object,
                       count=values.size).reshape(shape)


def _boxed_sequence(y):
    values = np.asarray(y, dtype=object)
    if values.ndim == 2 and values.shape[1] == 1:
        # Whole-array boxing converts native ndarray rows to Python numbers.
        # Recover the original NumPy scalar from each such row.
        values = np.fromiter((row[0] if isinstance(row, np.ndarray) else value
                              for row, value in zip(y, values[:, 0])),
                             dtype=object, count=len(y)).reshape(values.shape)
    return values


def _label_row(value):
    return issubclass(type(value), (list, tuple)) or type(value) is np.ndarray and value.ndim != 0


def _singleton_label_rows(rows):
    return all((issubclass(type(row), (list, tuple)) and len(row) == 1)
               or (type(row) is np.ndarray and row.shape == (1,)) for row in rows)


def _boxed_label_vector(rows, *, column=False):
    # The selected projection has already established the target shape. Box
    # its captured scalar operands without another round of dtype discovery.
    if column:
        if not _singleton_label_rows(rows):
            return _boxed_sequence(rows).reshape(-1)
        return np.fromiter((row[0] for row in rows), dtype=object, count=len(rows))
    return rows if type(rows) is np.ndarray else np.fromiter(rows, dtype=object, count=len(rows))


def _stored_sequence_number(value):
    value = _unbox_numeric_array(value)
    if isinstance(value, int):
        return int.__int__(value)
    if isinstance(value, float):
        return float.__float__(value)
    return _python_label(value)


def _custom_sequence_target(y, inferred):
    """Recover a numeric sequence if native inference discarded a comparator."""
    if not len(y):
        return None
    if inferred.ndim == 1:
        values = y
    elif inferred.ndim == 2 and inferred.shape[1] == 1:
        row_type = type(y[0])
        if row_type is list or row_type is tuple or row_type is np.ndarray:
            first = type(y[0][0])
            if id(first) in _PLAIN_REAL_TYPES:
                for row in y:
                    if type(row) is not row_type or type(row[0]) is not first:
                        break
                else:
                    return None
        values = _boxed_label_vector(y, column=True)
    else:
        return None
    # Most numeric lists contain a single ordinary scalar type. Identity
    # checks also work for user types with nonstandard metaclass hashing.
    first = type(values[0])
    if all(map(is_, map(type, values), repeat(first))):
        if id(first) in _PLAIN_REAL_TYPES or first is np.ndarray:
            return None
        types_to_check = values[:1]
    else:
        types_to_check = values
    if not _custom_comparison_types(types_to_check, ()):
        types = {id(t): t for t in map(type, types_to_check) if id(t) not in _PLAIN_REAL_TYPES}
        if not any(map(_overrides_numeric_conversion, types.values())):
            if any(issubclass(t, (int, float, np.integer, np.floating)) for t in types.values()):
                require_real(_boxed_label_vector(values), 'y')
            return None
    return _boxed_label_vector(values).reshape(inferred.shape)


def _can_use_native_sequence(boxed, inferred):
    """Keep native classification targets when projection preserves their classes."""
    raw = boxed.ravel()
    native = inferred.ravel()
    actual, actual_groups = np.unique(raw, return_inverse=True)
    representatives, groups = np.unique(native, return_inverse=True)
    if not np.array_equal(actual_groups, groups):
        return False
    for value, projected in zip(raw, native):
        # Compare the stored numbers independently of a user-defined equality
        # that might itself round a large integer or accept a different ID.
        if not isinstance(value, (int, float, np.generic)):
            return False
        if _stored_sequence_number(value) != _python_label(projected):
            return False
    # A comparator can match a different native class even when every
    # observation matches its own. Verify one mask at a time in both operand
    # orders, retaining NumPy scalars when boxing native representatives.
    projected = np.fromiter(representatives, dtype=object, count=len(representatives))
    for i in range(len(actual)):
        mask = groups == i
        original = actual[i:i + 1]
        if not np.array_equal(raw == original, mask) or not np.array_equal(original == raw, mask):
            return False
        label = projected[i:i + 1]
        try:
            forward, reverse = raw == label, label == raw
        except (MemoryError, RecursionError):
            raise
        except Exception:
            # A custom comparison can reject a foreign operand with its own
            # exception. The original class comparisons above must succeed.
            return False
        if not np.array_equal(forward, mask) or not np.array_equal(reverse, mask):
            return False
    return True


def _plain_sequence_container(value):
    cls = type(value)
    if cls is list or cls is tuple:
        return True
    if not isinstance(value, (list, tuple)):
        return False
    base = list if isinstance(value, list) else tuple
    operations = ('__len__', '__iter__', '__getitem__', '__getattribute__', '__getattr__')
    return (all(getattr_static(cls, op, _MISSING) is getattr_static(base, op, _MISSING) for op in operations)
            and all(getattr_static(value, attr, _MISSING) is _MISSING for attr in _ARRAY_PROTOCOLS))


def _numeric_sequence_after_conversion_error(y):
    """Recover a plain vector or column only when a real scalar overrides conversion."""
    if not _plain_sequence_container(y) or not len(y):
        return None
    first = y[0]
    column = _plain_sequence_container(first) or type(first) is np.ndarray and first.ndim != 0
    if column:
        for row in y:
            if type(row) is np.ndarray:
                if row.shape != (1,):
                    return None
            elif not _plain_sequence_container(row) or len(row) != 1:
                return None
        values = (row[0] for row in y)
    else:
        values = iter(y)
    boxed = np.fromiter(map(_unbox_numeric_array, values), dtype=object, count=len(y))
    if not all(np.generic.dtype.__get__(v).kind in 'biuf' if isinstance(v, np.generic)
               else isinstance(v, (int, float)) for v in boxed):
        return None
    types = {id(t): t for t in map(type, boxed) if id(t) not in _PLAIN_REAL_TYPES}
    if not any(map(_overrides_numeric_conversion, types.values())):
        return None
    return boxed[:, None] if column else boxed


def _unbox_unicode_scalar(value):
    if type(value) is np.ndarray and value.ndim == 0 and value.dtype.kind in 'UT':
        return value[()]
    return value


def _plain_unicode_scalar(value, cache):
    cls = type(value)
    return (cls is str or cls is np.str_ or issubclass(cls, str)
            and _passive_scalar(value, _TEXT_TYPES, _TEXT_FIELDS, cache))


def _native_string_subclass_rows(rows):
    """Replace inherited Unicode scalars only in a plain native-string sequence."""
    if not _plain_sequence_container(rows) or not rows:
        return None
    column = _label_row(rows[0])
    native = False
    result = None
    cache = {}
    for i, row in enumerate(rows):
        value = row
        if column and type(row) is not np.ndarray:
            if not _plain_sequence_container(row) or len(row) != 1:
                return None
            value = row[0]
        if type(value) is np.ndarray:
            shape = (1,) if column and row is value else ()
            if value.shape != shape or value.dtype.kind not in 'UT':
                return None
            native |= value.dtype.kind == 'T'
        elif type(value) is not str and type(value) is not np.str_:
            if not _plain_unicode_scalar(value, cache):
                return None
            if result is None:
                result = list(rows)
            text = str.__str__(value)
            result[i] = [text] if column else text
    return result if native else None


def _infer_label_sequence(rows):
    try:
        return np.asarray(rows)
    except ValueError as error:
        # Older NumPy can reject a passive str subclass next to a strict
        # StringDType box. Retain the box, including all its dtype metadata.
        if (len(error.args) != 1 or type(error.args[0]) is not str
                or error.args[0] != 'StringDType only allows string data when string coercion is disabled.'):
            raise
        compatible = _native_string_subclass_rows(rows)
        if compatible is None:
            raise
        return np.asarray(compatible)


def _string_sequence_target(rows, inferred):
    """Preserve observable Unicode scalar values through string dtype inference."""
    variable = inferred.dtype.kind == 'T'
    if inferred.ndim == 1:
        values = rows
    elif inferred.ndim == 2 and inferred.shape[1] == 1 and _singleton_label_rows(rows):
        values = (row[0] for row in rows)
    else:
        return inferred if variable else rows
    scalar_types = {}
    if variable:
        # Casting a NumPy Unicode scalar to StringDType can trim its trailing
        # NULs. Assign the full Python string into the already inferred dtype,
        # which also retains the selected coercion and missing-value metadata.
        for i, value in enumerate(values):
            if _plain_unicode_scalar(value, scalar_types) and str.endswith(value, '\0'):
                inferred.flat[i] = str.__getitem__(value, slice(None))
        return inferred
    padded = False
    for value in values:
        if type(value) is np.ndarray:
            value = _unbox_unicode_scalar(value)
        if not _plain_unicode_scalar(value, scalar_types):
            return rows
        padded = padded or str.endswith(value, '\0')
    if not padded:
        return rows
    boxed = _boxed_label_vector(rows, column=inferred.ndim == 2)
    for i, value in enumerate(boxed):
        if type(value) is np.ndarray:
            boxed[i] = _unbox_unicode_scalar(value)
    return boxed.reshape(inferred.shape)


def _exact_numeric_sequence(y, *, mode='normalize', ordinary=False, iterated=False):
    """Recover integer observations lost by mixed numeric type inference."""
    if ordinary and len(y) and type(y[0]) is int and np.dtype(int).itemsize == 8:
        # The source scan already established a flat vector of built-in ints.
        # fromiter skips repeated dtype discovery and rejects out-of-range IDs.
        try:
            return np.fromiter(y, dtype=np.int64, count=len(y))
        except OverflowError:
            pass
    try:
        inferred = _infer_label_sequence(y)
    except (MemoryError, RecursionError):
        raise
    except Exception:
        temporal = _temporal_sequence_target(y, mode)
        if temporal is not None:
            return temporal
        boxed = _numeric_sequence_after_conversion_error(y)
        if boxed is None:
            raise
        return _boxed_numeric_target(boxed, boxed, mode=mode)
    if iterated and not ordinary:
        exact = _iterated_text_target(y, inferred, mode)
        if exact is not None:
            return _label_source_and_target(exact, mode)[1]
        if inferred.dtype.kind in 'SUT':
            return inferred
    if inferred.dtype.kind in 'UT':
        return _string_sequence_target(y, inferred)
    if inferred.dtype.kind == 'O':
        temporal = _temporal_sequence_target(y, mode)
        if temporal is not None:
            return temporal
        if inferred.ndim == 2 and inferred.shape[1] == 1:
            inferred = _boxed_sequence(y)
        return _boxed_numeric_target(y, inferred, mode=mode)
    if not ordinary and inferred.dtype.kind in 'biuf':
        boxed = _custom_sequence_target(y, inferred)
        if boxed is not None:
            target = _boxed_numeric_target(boxed, boxed, mode=mode)
            if (mode == 'normalize' and (inferred.dtype.kind != 'f' or np.isfinite(inferred).all())
                    and _can_use_native_sequence(target, inferred)):
                return inferred
            return target
    if inferred.dtype.kind in 'Mm':
        temporal = _temporal_sequence_target(y, mode)
        return y if temporal is None else temporal
    if inferred.dtype.kind != 'f':
        return inferred if inferred.dtype.kind in 'biu' else y
    # Small integers survive promotion and sklearn's float classification
    # check. Windows NumPy 1.x uses a 32-bit native int for the latter.
    limit = 2 ** min(np.finfo(inferred.dtype).nmant + 1, np.iinfo(int).bits - 1)
    if not np.any((inferred >= limit) | (inferred <= -limit)):
        return inferred
    # sklearn's native numeric target validation rejects NaN and infinity;
    # its object-array check does not detect infinities. Preserve the finite
    # check before producing an exact object target.
    if not np.isfinite(inferred).all():
        raise ValueError('Input y contains NaN or infinity.')
    boxed = np.asarray(y, dtype=object)
    original = np.fromiter(map(_python_label, boxed.flat), dtype=object,
                           count=boxed.size).reshape(boxed.shape)
    if any(_numeric_abc_instance(v, Integral) for v in original.flat):
        if not all(_is_integral_label(v) for v in original.flat):
            if not any(_numeric_abc_instance(v, Integral) and (not np.isfinite(f) or int(v) != int(f))
                       for v, f in zip(original.flat, inferred.flat)):
                return inferred
            return original
        # Preserve integral observations even when their mixed inference was
        # exactly representable but exceeds the platform's native int range.
        return _integral_target(original)
    return inferred


def _integer_label_fits(value, dtype):
    if not _is_integral_label(value):
        return False
    bounds = np.iinfo(dtype)
    return int(bounds.min) <= int(value) <= int(bounds.max)


class _SparseTextObservations(np.ndarray):
    """Exact rows containing both numeric sparse points and a text fill."""


def _passive_scalar(value, bases, fields, cache):
    """Recognize inherited scalar protocols without evaluating user descriptors."""
    cls = type(value)
    key = id(cls)
    namespace = _MISSING if cache is None else cache.get(key, _MISSING)
    if namespace is _MISSING:
        mro = type.__getattribute__(cls, '__mro__')
        base = next((t for t in mro if id(t) in bases), None)
        if base is None:
            return False
        inherited = {id(t) for t in base.__mro__}
        declared = {name for t in mro if id(t) not in inherited for name in type.__getattribute__(t, '__dict__')}
        namespace = getattr_static(cls, '__dict__')
        if (type(namespace) is not GetSetDescriptorType or namespace.__name__ != '__dict__'
                or _custom_fields(value, base, declared.intersection(fields))):
            namespace = False
        elif not type.__subclasscheck__(namespace.__objclass__, cls):
            namespace = None  # A slotted scalar has no instance dictionary.
        if cache is not None:
            cache[key] = namespace
    return namespace is not False and (namespace is None or dict.keys(namespace.__get__(value)).isdisjoint(fields))


def _sparse_text_fill(fill, mode=None, cache=None):
    cls = type(fill)
    if id(cls) in _TEXT_TYPES:
        return True
    # A bytes subclass can compare differently against a NumPy and a Python
    # integer. Preserve its actual operand for scoring.
    return (mode == 'score' and issubclass(cls, (str, bytes))
            and _passive_scalar(fill, _TEXT_TYPES, _TEXT_FIELDS, cache))


def _all_sparse_text(values, mode):
    cache = {}
    return all(_sparse_text_fill(v, mode, cache) for v in values)


def _sparse_text_rows(points, index, fill, shape, codes=None, *, mode=None):
    # NumPy's common text dtype can make integer 1 indistinguishable from
    # '1'. Fill an object array directly from the selected compressed rows.
    if points.dtype.kind not in 'biufc' or not index.ngaps or not _sparse_text_fill(fill, mode):
        return None
    if codes is None:
        positions = index.to_int_index().indices
    else:
        if (codes < 0).any():
            raise ValueError('Input y contains NaN.')
        located = index.lookup_array(codes.ravel().astype(np.int32, copy=False))
        positions = located >= 0
        if positions.all():
            return points[located].reshape(shape)
        points = points[located[positions]]
    plain_text_only = not points.size and id(type(fill)) in _TEXT_TYPES
    if plain_text_only:
        # With only a text fill, preserve the dense text metric's usual
        # target validation. Fixed-width text can strip a trailing NUL.
        native = np.asarray(fill)
        if native.item() == fill:
            return np.full(shape, fill, dtype=native.dtype)
    rows = np.empty(shape, dtype=object)
    rows.fill(fill)
    if plain_text_only:
        return rows
    # Even an all-fill bytes subclass can be inferred as a numeric array.
    # Retain the actual subclass for paired scoring, including trailing NULs.
    rows.flat[positions] = np.fromiter(points, dtype=object, count=points.size)
    return rows.view(_SparseTextObservations)


def _sparse_text_target(rows, mode):
    if isinstance(rows, _SparseTextObservations):
        if mode in ('fit', 'normalize'):
            raise ValueError('Sparse target mixes numeric and text class labels.')
        for value in rows.flat:
            if issubclass(type(value), (complex, np.complexfloating)):
                raise ValueError('Sparse target contains complex class labels.')
            if mode == 'score' and not (isinstance(value, (str, bytes)) or _is_integral_label(value)):
                raise ValueError('Numeric class labels mixed with sparse text must be finite integers for scoring.')
    return rows


def _occupied_sparse_levels(array, codes, *, mode=None):
    """Read the compressed entries for observed levels without scalar indexing."""
    locations = array.sp_index.lookup_array(codes.astype(np.int32, copy=False))
    points = array.sp_values
    fill = None
    if ((locations < 0) & (codes >= 0)).any():
        fill = array.fill_value
        _check_training_fill(fill, mode)
    return [np.nan if code < 0 else fill if location < 0 else points[location]
            for code, location in zip(codes, locations)]


def _categorical_integer_target(array, pd, mode):
    categories = array.categories
    dtype = categories.dtype
    sparse = isinstance(dtype, pd.SparseDtype)
    if dtype.kind not in 'iu' and not (sparse and (dtype.kind in 'bf'
            or dtype.kind == 'c' and _sparse_text_fill(dtype.fill_value, mode))):
        return None
    codes = array._codes
    if (codes < 0).any():
        raise ValueError('Input y contains NaN.')
    if sparse:
        # Taking the Index, casting the Categorical, or iterating its values
        # can round a level on older pandas. Read each observed scalar once.
        levels_array = categories._values
        rows = _sparse_text_rows(levels_array.sp_values, levels_array.sp_index, levels_array.fill_value,
                                 codes.shape, codes, mode=mode)
        if rows is not None:
            return _sparse_text_target(rows, mode)
        positions, used = pd.factorize(codes)
        stored = _occupied_sparse_levels(levels_array, used, mode=mode)
        if any(_custom_numeric_fill(v) for v in stored):
            observations = np.fromiter(stored, dtype=object, count=len(stored))[positions]
            return _boxed_numeric_target(observations, observations, mode=mode)
        levels = [_python_label(v) for v in stored]
        if dtype.kind in 'iu' and all(_integer_label_fits(v, dtype.subtype) for v in levels):
            labels = np.asarray(levels, dtype=dtype.subtype)
        elif all(isinstance(v, (Real, np.bool_)) for v in levels):
            # A sparse category's fill can itself be a non-integer label.
            labels = np.asarray(_exact_numeric_sequence(levels))
        else:
            # In particular, a text fill must not stringify the integer IDs.
            labels = np.asarray(levels, dtype=object)
        return labels[positions]
    native = getattr(dtype, 'numpy_dtype', dtype)
    return np.asarray(categories._values, dtype=native)[codes]


def _custom_numeric_fill(value):
    return (id(type(value)) not in _PLAIN_REAL_TYPES
            and isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, np.timedelta64))


def _boxed_sparse_scalars(values, pd, *, include_numeric=False):
    # IntIndex.lookup can scan the compressed positions for each scalar.
    # Read native object references directly, inserting the actual fill at gaps.
    if not isinstance(values, pd.arrays.SparseArray) or (not include_numeric and values.dtype.subtype.kind != 'O'):
        return values
    if not values.sp_index.ngaps:
        return values.sp_values
    scalars = np.empty(len(values), dtype=object)
    scalars.fill(values.fill_value)
    stored = values.sp_values
    if stored.dtype != object:
        # Native-to-object array assignment would convert NumPy scalars to
        # Python numbers before the custom label can inspect its operand.
        stored = np.fromiter(stored, dtype=object, count=len(stored))
    scalars[values.sp_index.to_int_index().indices] = stored
    return scalars


_TICK_ATTOSECONDS = {'as': 1, 'fs': 10**3, 'ps': 10**6, 'ns': 10**9, 'us': 10**12,
                     'ms': 10**15, 's': 10**18, 'm': 60 * 10**18, 'h': 3600 * 10**18,
                     'D': 86400 * 10**18, 'W': 604800 * 10**18}


def _temporal_label_key(value, pd):
    aware = False
    if pd is not None and isinstance(value, pd.Timestamp):
        aware = value.tz is not None
        value = value.asm8  # UTC ticks for a timezone-aware timestamp.
    elif pd is not None and isinstance(value, pd.Timedelta):
        value = value.asm8
    elif isinstance(value, datetime):
        offset = value.utcoffset()
        elapsed = value.replace(tzinfo=None) - datetime(1970, 1, 1)
        if offset is not None:
            elapsed -= offset
        return ('aware datetime' if offset is not None else 'datetime',
                (elapsed // timedelta(microseconds=1)) * _TICK_ATTOSECONDS['us'])
    elif isinstance(value, timedelta):
        return 'duration', (value // timedelta(microseconds=1)) * _TICK_ATTOSECONDS['us']
    elif isinstance(value, date):
        return 'date', value.toordinal()

    unit, step = np.datetime_data(value.dtype)
    ticks = int(value.view('i8')) * step
    duration = isinstance(value, np.timedelta64)
    if unit in ('Y', 'M'):
        months = ticks * 12 if unit == 'Y' else ticks
        if duration:
            return 'calendar duration', months
        # Calendar datetime ticks denote the first day of the period. The
        # Gregorian 400-year cycle also handles years outside Python's range.
        cycles, remaining = divmod(months, 4800)
        years, month = divmod(remaining, 12)
        ticks = cycles * 146097 + (date(1970 + years, 1 + month, 1) - date(1970, 1, 1)).days
        unit = 'D'
    if unit == 'generic':
        return ('unscaled duration' if duration else 'unscaled datetime'), ticks
    kind = 'duration' if duration else ('aware datetime' if aware else 'datetime')
    return kind, ticks * _TICK_ATTOSECONDS[unit]


def _temporal_missing(value):
    # A Python duration can exceed the range of every native microsecond
    # count. Such observations remain exact Python values while testing units.
    return type(value) is not timedelta and np.isnat(value)


def _temporal_cast(values, dtype, pd):
    if values.dtype == dtype:
        return values
    # Multiplication before division can wrap even when the final count in
    # a stepped unit fits int64. Work from exact instants or durations.
    unit, step = np.datetime_data(dtype)
    # Fixed units with integral scale can be multiplied exactly in int64 once
    # both finite extremes fit. Scalar conversion handles short groups, NaT,
    # calendar units and conversions that may divide or round.
    if (type(values) is np.ndarray and values.size >= 8 and isinstance(dtype, np.dtype)
            and values.dtype.kind == dtype.kind and dtype.kind in 'Mm' and values.dtype.isnative and dtype.isnative
            and step > 0 and unit in _TICK_ATTOSECONDS):
        source_unit, source_step = np.datetime_data(values.dtype)
        if source_unit in _TICK_ATTOSECONDS:
            factor, remainder = divmod(source_step * _TICK_ATTOSECONDS[source_unit],
                                       step * _TICK_ATTOSECONDS[unit])
            if not remainder and 0 < factor < 2**63:
                ticks = values.view('i8')
                low = int(ticks.min())
                if low != -(2**63):
                    limit = (2**63 - 1) // factor
                    if low < -limit or int(ticks.max()) > limit:
                        return None
                    scaled = np.empty(values.shape, dtype=dtype)
                    np.multiply(ticks, factor, out=scaled.view('i8'))
                    return scaled
    domain = _temporal_label_key(np.zeros((), dtype)[()], pd)[0]
    quantum = step * _TICK_ATTOSECONDS.get(unit, 12 if unit == 'Y' else 1)
    counts = np.empty(values.shape, dtype=np.int64)
    for i, old in enumerate(values.flat):
        if _temporal_missing(old):
            count = -(2**63)
        else:
            kind, tick = _temporal_label_key(old, pd)
            if kind != domain:
                return None
            if domain == 'datetime' and unit in ('Y', 'M'):
                days, remainder = divmod(tick, _TICK_ATTOSECONDS['D'])
                if remainder:
                    return None
                cycles, days = divmod(days, 146097)
                calendar = date(1970, 1, 1) + timedelta(days=days)
                if calendar.day != 1:
                    return None
                tick = cycles * 4800 + (calendar.year - 1970) * 12 + calendar.month - 1
            count, remainder = divmod(tick, quantum)
            if remainder or not -(2**63) < count < 2**63:
                return None
        counts.flat[i] = count
    return counts.astype(dtype)


def _stepped_temporal_dtypes(arrays, pd, kind):
    # A grid such as 3 ns may hold both 3 ns and 12 billion seconds even
    # though the latter exceeds int64 in nanoseconds. Only occupied ticks
    # constrain the grid. NaT reserves the minimum signed count.
    stride = largest = 0
    domains = set()
    for values in arrays:
        for value in values.flat:
            if not _temporal_missing(value):
                domain, tick = _temporal_label_key(value, pd)
                domains.add(domain)
                stride = gcd(stride, tick)
                largest = max(largest, abs(tick))
    if not stride or len(domains) != 1:
        return
    domain = domains.pop()
    if domain == 'calendar duration':
        units = (('Y', 12), ('M', 1))
    elif domain in ('duration', 'datetime'):
        units = reversed(_TICK_ATTOSECONDS.items())
    else:
        return
    for unit, width in units:
        step, remainder = divmod(stride, width)
        if remainder:
            continue
        # NumPy stores a dtype's multiplier in a signed C int. Reduce a
        # large stride by common clock factors only when the counts still fit.
        for factor in (2, 3, 5, 7):
            while step > 2**31 - 1 and step % factor == 0:
                step //= factor
        if 1 < step < 2**31 and largest <= (2**63 - 1) * width * step:
            yield np.dtype(f'{kind}8[{step}{unit}]')


def _common_temporal_arrays(arrays, pd):
    candidates = [values.dtype for values in arrays if values.dtype.kind in 'Mm'] or [np.dtype('m8[us]')]
    kind = candidates[0].kind
    try:
        candidates.append(np.result_type(*candidates))
    except (TypeError, ValueError, OverflowError):
        pass
    # A calendar month and a week, for example, can both be exact days even
    # when NumPy's common week dtype truncates the beginning of the month.
    standard = (np.dtype(f'{kind}8[{unit}]') for unit in reversed(_TICK_ATTOSECONDS))
    checked = set()
    for dtype in chain(candidates, standard, _stepped_temporal_dtypes(arrays, pd, kind)):
        if dtype in checked:
            continue
        checked.add(dtype)
        converted = []
        for values in arrays:
            group = _temporal_cast(values, dtype, pd)
            if group is None:
                break
            converted.append(group)
        else:
            return converted
    return None


class _SelectedTemporalObservations(np.ndarray):
    """Temporal rows prepared for comparison, independent of the stored class representatives."""


class _MixedTemporalObservations(_SelectedTemporalObservations):
    """Same-domain temporal observations without a shared native unit."""


def _boxed_temporal_scalars(values, kinds):
    if len(kinds) == 1 and (id(np.datetime64) in kinds or id(np.timedelta64) in kinds):
        return values
    pd = modules.get('pandas')
    if pd is None:
        return None
    missing = id(type(pd.NaT))
    if kinds == {missing}:
        # pandas NaT alone identifies no family and cannot match a class.
        return np.full(values.shape, np.datetime64('NaT')).view(_MixedTemporalObservations)
    if kinds <= {id(np.datetime64), id(pd.Timestamp), id(datetime), missing}:
        # Preserve timezone semantics, including custom tzinfo callbacks.
        if any(type(v) in (pd.Timestamp, datetime) and v.tzinfo is not None for v in values.flat):
            return None
        scalar = np.datetime64
    elif kinds <= {id(np.timedelta64), id(pd.Timedelta), id(timedelta), missing}:
        scalar = np.timedelta64
    else:
        return None
    dtype = np.dtype(scalar)
    return np.fromiter((_temporal_fill(v, dtype, pd) for v in values.flat),
                       dtype=object, count=values.size).reshape(values.shape)


def _selected_boxed_temporal_target(rows, mode):
    """Recover temporal units from the scalar rows already selected by pandas."""
    if isinstance(rows, _MixedTemporalObservations):
        return _projected_pandas_target(rows, mode)[1]
    if type(rows) in (list, tuple) and rows:
        if id(type(rows[0])) in _PLAIN_REAL_TYPES or id(type(rows[0])) in _TEXT_OR_NONE_TYPES:
            return None
        # Iteration has already captured the actual rows. Object fromiter keeps
        # every scalar's unit without invoking another projection on a label.
        if type(rows[0]) in (list, tuple, np.ndarray):
            if not all((type(row) in (list, tuple) and len(row) == 1)
                       or (type(row) is np.ndarray and row.shape == (1,)) for row in rows):
                return None
            values = np.fromiter((row[0] for row in rows), dtype=object, count=len(rows))[:, None]
        else:
            values = np.fromiter(rows, dtype=object, count=len(rows))
    elif (type(rows) is np.ndarray and rows.dtype.kind == 'O' and rows.size
          and (rows.ndim == 1 or rows.ndim == 2 and rows.shape[1] == 1)):
        values = rows
    else:
        return None
    if id(type(values.flat[0])) in _PLAIN_REAL_TYPES or id(type(values.flat[0])) in _TEXT_OR_NONE_TYPES:
        return None
    kinds = set(map(id, map(type, values.flat)))
    scalars = _boxed_temporal_scalars(values, kinds)
    if scalars is None:
        return None
    if isinstance(scalars, _MixedTemporalObservations):
        return _projected_pandas_target(scalars, mode)[1]
    if mode in ('fit', 'normalize') and len(kinds) == 1:
        # Keep homogeneous class representatives when scalar comparison needs
        # no unit promotion. Held observations need only exact paired equality.
        units = {np.dtype('m8[us]') if type(v) is timedelta else v.dtype
                 for v in scalars.flat if not _temporal_missing(v)}
        if len(units) < 2:
            return None
    target = _temporal_sequence_target(scalars, mode, python_duration=True)
    if mode in ('score', 'observations') and not isinstance(target, _MixedTemporalObservations):
        return target.view(_SelectedTemporalObservations)
    return target


def _temporal_sequence_target(rows, mode, *, python_duration=False):
    if not len(rows):
        return None
    first = rows[0]
    if ((type(first) in (list, tuple) and len(first) == 1)
            or (type(first) is np.ndarray and first.shape == (1,))):
        first = first[0]
    if type(first) not in (np.datetime64, np.timedelta64) and not (python_duration and type(first) is timedelta):
        return None
    values = _boxed_sequence(rows)
    if values.ndim not in (1, 2) or (values.ndim == 2 and values.shape[1] != 1):
        return None
    positions = {}
    family = np.timedelta64 if type(first) is timedelta else type(first)
    for i, scalar in enumerate(values.flat):
        if type(scalar) is family:
            dtype = scalar.dtype
        elif python_duration and family is np.timedelta64 and type(scalar) is timedelta:
            dtype = np.dtype(object)
        else:
            return None
        positions.setdefault(dtype, []).append(i)
    if len(positions) == 1 and dtype.kind != 'O':
        return values.astype(dtype)
    pd = modules.get('pandas')
    groups = [values.flat[indices].astype(dtype) for dtype, indices in positions.items()]
    common = _common_temporal_arrays(groups, pd)
    if common is None:
        if len({_temporal_label_key(v, pd)[0] for group in groups for v in group if not _temporal_missing(v)}) > 1:
            raise ValueError('Temporal target has incompatible temporal units.')
        return _projected_pandas_target(values.view(_MixedTemporalObservations), mode)[1]
    target = np.empty(values.shape, dtype=common[0].dtype)
    for indices, group in zip(positions.values(), common):
        target.flat[indices] = group
    return target


def _temporal_fill(fill, dtype, pd):
    if (type(fill) is type(pd.NaT) or
            issubclass(type(fill), type(pd.NaT)) and not bool(fill == fill)):
        scalar = dtype.type('NaT')
    elif isinstance(fill, (pd.Timestamp, pd.Timedelta)):
        scalar = fill.asm8
    elif type(fill) is timedelta and dtype.kind == 'm':
        ticks = fill // timedelta(microseconds=1)
        # NumPy wraps an out-of-range Python duration on conversion. Keep
        # its exact value for testing native grids and for paired comparisons.
        return np.timedelta64(ticks, 'us') if -(2**63) < ticks < 2**63 else fill
    elif type(fill) is datetime:
        if fill.utcoffset() is not None:
            raise ValueError('A timezone-aware fill cannot share a naive temporal label dtype.')
        scalar = np.datetime64(fill)
    else:
        scalar = fill
    if (type(scalar) is not dtype.type or
            (isinstance(fill, pd.Timestamp) and fill.tz is not None)):
        raise ValueError('Sparse temporal target has an incompatible occupied fill.')
    return scalar


def _sparse_temporal_levels(levels, codes, pd):
    """Decode only referenced temporal levels, preserving each selected unit."""
    dtype = levels.sp_values.dtype
    used, positions = np.unique(codes.ravel(), return_inverse=True)
    if not used.size:
        return np.empty(codes.shape, dtype=dtype)
    if used[0] < -1 or used[-1] >= len(levels):
        raise ValueError('Sparse categorical code is outside the level catalogue.')
    values = _occupied_sparse_levels(levels, used)
    if dtype.kind == 'O':
        # A missing category code has no instant of its own. Infer its family
        # only from the observed temporal levels, never the unused catalogue.
        rows = np.fromiter((pd.NaT if code < 0 else v for code, v in zip(used, values)),
                           dtype=object, count=len(used))
        kinds = set(map(id, map(type, rows)))
        if _boxed_temporal_scalars(rows, kinds) is None:
            return None
        return rows[positions].reshape(codes.shape)
    missing = np.array('NaT', dtype=dtype)[()]
    rows = [missing if code < 0 else _temporal_fill(value, dtype, pd)
            for code, value in zip(used, values)]
    return _temporal_sequence_target(rows, 'observations', python_duration=True)[positions].reshape(codes.shape)


def _sparse_temporal_target(points, fill, index, shape, pd):
    """Expand the occupied fill without changing its instant or duration."""
    if type(fill) is points.dtype.type and fill.dtype == points.dtype:
        return None
    scalar = _temporal_fill(fill, points.dtype, pd)
    common = _common_temporal_arrays([points, np.asarray(scalar)], pd)
    if common is None:
        if _temporal_label_key(np.zeros((), points.dtype)[()], pd)[0] != _temporal_label_key(scalar, pd)[0]:
            raise ValueError('Sparse temporal target has incompatible temporal units.')
        # Object assignment from a native array can turn individual temporal
        # scalars into integer ticks. Keep each scalar with its own unit.
        dense = np.empty(shape, dtype=object).view(_MixedTemporalObservations)
        dense.fill(scalar)
        stored = np.fromiter(points, dtype=object, count=len(points))
    else:
        stored, scalar = common
        dense = np.full(shape, scalar[()], dtype=stored.dtype)
    dense.flat[index.to_int_index().indices] = stored
    return dense


def _sparse_real_target(array, pd, mode):
    """Keep occupied numeric fills and compressed points exact before densifying."""
    subtype = array.dtype.subtype
    if subtype.kind not in 'biufc' or not array.sp_index.ngaps:
        return None
    fill = array.fill_value
    if subtype.kind == 'c' and not _sparse_text_fill(fill, mode):
        return None
    if type(fill) is subtype.type:
        return None
    kind = id(type(fill))
    if kind not in _PLAIN_REAL_TYPES:
        # Older pandas can turn an occupied temporal NaT in numeric sparse
        # storage into None. Inspect the selected fill before that conversion.
        _check_training_fill(fill, mode)
        rows = _sparse_text_rows(array.sp_values, array.sp_index, fill, array.shape, mode=mode)
        if rows is not None:
            return _sparse_text_target(rows, mode)
        if _custom_numeric_fill(fill):
            scalars = _boxed_sparse_scalars(array, pd, include_numeric=True)
            return _boxed_numeric_target(scalars, scalars, mode=mode)
        return None
    points = array.sp_values
    if kind in _PLAIN_INTEGER_TYPES or kind in _PLAIN_FLOAT_TYPES:
        dtype = np.result_type(subtype, type(fill))
        if dtype.kind == 'b' or dtype.kind in 'iu' and _integer_label_fits(fill, dtype):
            return pd.arrays.SparseArray.to_numpy(array, dtype=dtype)
        if dtype.kind == 'f':
            if subtype.kind == 'f' and kind in _PLAIN_FLOAT_TYPES:
                return None
            # Small integers survive floating promotion and sklearn's native
            # integer cast when it checks that a floating target is discrete.
            limit = 2 ** min(np.finfo(dtype).nmant + 1, np.iinfo(int).bits - 1)
            if (-limit < _python_label(fill) < limit and
                    (not points.size or (-limit < points.min().item() and points.max().item() < limit))):
                return pd.arrays.SparseArray.to_numpy(array, dtype=dtype)
            if subtype.kind in 'iu' and _integer_label_fits(fill, subtype):
                return pd.arrays.SparseArray.to_numpy(array, dtype=subtype)
    # Infer from the actual fill and stored scalars. Running the sequence
    # check on compressed values also handles an all-gap array in O(1).
    values = np.asarray(_exact_numeric_sequence([fill, *points], mode=mode))
    dense = np.full(len(array), values[0], dtype=values.dtype)
    dense[array.sp_index.to_int_index().indices] = values[1:]
    return dense


def _declares_array(value, *, class_only=False):
    if version_info >= (3, 12) and getattr_static(type(value), '__buffer__', None) is not None:
        return True
    owner = type(value) if class_only else value
    return any(getattr_static(owner, attr, _MISSING) is not _MISSING for attr in _ARRAY_PROTOCOLS)


def _dynamic_attributes(value, base=object):
    cls = type(value)
    return (getattr_static(cls, '__getattr__', _MISSING) is not _MISSING
            or getattr_static(cls, '__getattribute__', _MISSING) is not getattr_static(base, '__getattribute__'))


def _sequence_kind(value):
    if isinstance(value, (list, tuple)):
        return 'sequence'
    if isinstance(value, (str, bytes, bytearray, memoryview)):
        return None
    cls = type(value)
    try:
        if issubclass(cls, (Mapping, UserString)):
            return None
        if issubclass(cls, Sequence):
            return 'sequence'
    except TypeError:
        # ABC caches require hashable classes.
        if type.__subclasscheck__(Mapping, cls) or type.__subclasscheck__(UserString, cls):
            return None
        if type.__subclasscheck__(Sequence, cls):
            return 'sequence'
    # NumPy also accepts sized positional containers without ABC registration.
    if (getattr_static(cls, '__getitem__', None) is not None
            and getattr_static(cls, '__len__', None) is not None
            and getattr_static(cls, 'keys', None) is None):
        return 'indexed'
    return None


def _sequence_representation(value, *, indexed=False):
    if type(value) is list or type(value) is tuple:
        return value
    if type(value) is deque:
        return list(value)
    if type(value) is range:
        # Only mixed signed/unsigned inference can round an integer range.
        if not value or max(value[0], value[-1]) < 2**63:
            return np.asarray(value)
        low, high = sorted((value[0], value[-1]))
        dtype = np.uint64 if 0 <= low and high < 2**64 else object
        return np.fromiter(value, dtype=dtype, count=len(value))
    # inspect.getattr_static cannot inspect instances of unhashable classes on
    # some Pythons. Resolve their instance protocols in NumPy order below.
    unhashable = type(type(value)).__hash__ is None
    if _declares_array(value, class_only=unhashable):
        # NumPy resolves declared buffers and array protocols on their owner.
        return np.asarray(value)
    if version_info < (3, 12):
        # Older interpreters do not expose a native exporter's __buffer__.
        try:
            exported = memoryview(value)
        except TypeError:
            pass
        else:
            return np.asarray(exported)
    base = object
    if isinstance(value, (list, tuple)):
        base = list if isinstance(value, list) else tuple
    if unhashable or _dynamic_attributes(value, base):
        # A dynamic lookup can also report that no array protocol is present.
        # Resolve protocols in NumPy's order. Hand the first bound result to
        # NumPy without replaying its descriptor, retaining the storage owner.
        for name in ('__array_struct__', '__array_interface__', '__array__'):
            protocol = getattr(value, name, _MISSING)
            if protocol is not _MISSING:
                return np.asarray(SimpleNamespace(owner=value, **{name: protocol}))
    # Iteration and indexing can be customized independently of the array
    # protocols. Snapshot their sequence output with its actual scalar types.
    if indexed:
        # NumPy treats a class without a usable sequence length as a scalar
        # (including scipy sparse matrices). Keep it with its original caller.
        try:
            len(value)
        except (MemoryError, RecursionError):
            raise
        except Exception:
            return value
        return [item for item in value]
    return value if _plain_sequence_container(value) else list(value)


def _projected_rows(rows, *, in_column=False):
    snapshot = None
    for i, row in enumerate(rows):
        if (type(row) is np.ndarray or row is None
                or isinstance(row, (int, float, complex, str, bytes, np.generic, Decimal))
                or _numeric_abc_instance(row, Fraction)):
            continue
        kind = _sequence_kind(row)
        if kind:
            projected = _sequence_representation(row, indexed=kind == 'indexed')
            if isinstance(projected, (list, tuple)):
                # Visit earlier row elements before converting a later row.
                # Deeper nesting is outside the vector/column target shapes.
                projected = np.asarray(projected) if in_column else _projected_rows(projected, in_column=True)
        elif isinstance(row, np.ndarray) or _declares_array(row) or _dynamic_attributes(row):
            projected = np.asarray(row)
            if projected.ndim == 0 and projected.dtype.kind == 'O' and projected[()] is row:
                continue
        else:
            continue
        if projected is not row:
            if snapshot is None:
                snapshot = list(rows)
            snapshot[i] = projected
    return rows if snapshot is None else snapshot


def _sequence_source(y):
    """Snapshot provider-backed rows, leaving ordinary scalar operands intact."""
    if not len(y):
        return y, True
    first = y[0]
    cls = type(first)
    if id(cls) in _PLAIN_REAL_TYPES and all(map(is_, map(type, y), repeat(cls))):
        return y, True
    if (cls is list or cls is tuple) and len(first) == 1:
        scalar = type(first[0])
        if id(scalar) in _PLAIN_REAL_TYPES:
            for row in y:
                if type(row) is not cls or len(row) != 1 or type(row[0]) is not scalar:
                    break
            else:
                return y, True
    return _projected_rows(y), False


def _pandas_projection_kind(value, base, protocols):
    """Distinguish a declared representation from pandas' inherited conversion."""
    cls = type(value)
    try:
        namespace = object.__getattribute__(value, '__dict__')
    except AttributeError:
        namespace = {}
    if not namespace.keys().isdisjoint(protocols):
        return 'declared'
    if cls is base:
        return None
    if any(getattr_static(cls, name, _MISSING) is not getattr_static(base, name, _MISSING)
           for name in (*protocols, '__buffer__')):
        return 'declared'
    if version_info < (3, 12):
        # Native buffer slots were not visible as __buffer__ on these versions.
        try:
            memoryview(value)
        except TypeError:
            pass
        else:
            return 'declared'
    if (getattr_static(cls, '__getattr__', _MISSING) is not getattr_static(base, '__getattr__', _MISSING)
            or getattr_static(cls, '__getattribute__') is not getattr_static(base, '__getattribute__')):
        return 'dynamic'
    return None


def _bound_implementation(method, value, implementation):
    return isinstance(method, MethodType) and method.__self__ is value and method.__func__ is implementation


def _manager_base(manager):
    block = modules['pandas.core.internals.managers'].BlockManager
    if isinstance(manager, block):
        return block
    return getattr(modules.get('pandas.core.internals.array_manager'), 'ArrayManager', None)


def _custom_fields(value, base, names):
    cls = type(value)
    return cls is not base and any(getattr_static(cls, name, _MISSING) is not getattr_static(base, name, _MISSING)
                                   for name in names)


def _storage_layout(storage, carrier, pd):
    if not isinstance(carrier, pd.DataFrame):
        return storage
    frame = pd.DataFrame({0: storage}, copy=False)
    blocks = hasattr(carrier._mgr, 'blocks')
    if blocks != hasattr(frame._mgr, 'blocks'):
        frame = frame._as_manager('block' if blocks else 'array', copy=False)
    return frame


def _stored_sparse(array, pd):
    # Read only the backing dictionary, bypassing subclass accessors.
    state = getattr_static(pd.arrays.SparseArray, '__dict__').__get__(array)
    points, index, dtype = (state.get(k) for k in ('_sparse_values', '_sparse_index', '_dtype'))
    if type(points) is not np.ndarray or type(dtype) is not pd.SparseDtype or index is None:
        return None
    return pd.arrays.SparseArray._simple_new(points, index, dtype)


def _native_sparse_fill(fill, dtype):
    if dtype.kind not in 'biuf' or id(type(fill)) not in _PLAIN_REAL_TYPES:
        return False
    if type(fill) is dtype.type:
        return True
    if dtype.kind in 'iu':
        return _integer_label_fits(fill, dtype)
    if dtype.kind == 'b':
        return _is_integral_label(fill) and int(fill) in (0, 1)
    with np.errstate(over='ignore', invalid='ignore'):
        try:
            converted = dtype.type(fill)
        except (TypeError, ValueError, OverflowError):
            return False
    return _python_label(fill) == _python_label(converted)


def _ordinary_sparse_level_rows(levels, codes, legacy_frame, pd):
    # These are occupied, already validated categories. Reconstructing a
    # Categorical would revalidate their rounded dense view on older pandas.
    index = pd.Index._simple_new(levels)
    selected = SimpleNamespace(ndim=1, categories=index, _codes=codes)
    if legacy_frame:
        selected.codes = codes
        selected.dtype = pd.CategoricalDtype._from_fastpath(categories=index, ordered=False)
        return pd.Categorical.astype(selected, np.dtype(object), copy=False)
    return pd.Categorical.__array__(selected)


def _plain_temporal_levels(levels, pd, *, legacy=False, observed_array=False, array_fields=()):
    base = pd.arrays.SparseArray
    # Compare operations selected by pandas, without evaluating subclass
    # descriptors. Application methods and metadata do not participate.
    fields = ('__getattribute__', '__getattr__', '__len__', 'sp_index', 'sp_values',
              'dtype', 'fill_value', *array_fields)
    if not observed_array:
        fields += (*_ARRAY_PROTOCOLS, '__buffer__')
    if legacy:
        # ArrayManager casts the levels to object before taking the codes.
        fields += ('astype', '_densify')
    else:
        # SparseArray.take selects _take_with_fill, then constructs type(self)
        # from the selected rows. Its constructor checks isinstance(data, cls)
        # and assigns the new sparse index, points and dtype.
        if _custom_fields(type(levels), type(base), ('__call__', '__instancecheck__', '__subclasscheck__',
                                                      '__getattribute__', '__getattr__')):
            return False
        fields += ('take', '_take_with_fill', 'kind', '__init__', '__new__', '__setattr__', '__del__')
    state = getattr_static(base, '__dict__').__get__(levels)
    storage = ('_sparse_index', '_sparse_values', '_dtype')
    return state.keys().isdisjoint(fields) and not _custom_fields(levels, base, (*fields, *storage))


def _same_temporal_rows(left, right, pd):
    if left.size != right.size:
        return False
    if left.dtype == right.dtype and left.dtype.kind in 'Mm':
        return np.array_equal(left.ravel().view('i8'), right.ravel().view('i8'))
    temporal = (np.datetime64, np.timedelta64, pd.Timestamp, pd.Timedelta, type(pd.NaT), datetime, timedelta, date)
    for a, b in zip(left.flat, right.flat):
        if a is b:
            continue
        # pandas may box native temporal points as integer ticks alongside
        # its temporal fill. Compare like representations without equating
        # a boxed integer to an instant or duration.
        if type(a) is int and type(b) is int and a == b:
            continue
        if type(a) not in temporal or type(b) not in temporal:
            return False
        missing_a, missing_b = _is_temporal_nat(a, pd), _is_temporal_nat(b, pd)
        if missing_a or missing_b:
            if not (missing_a and missing_b):
                return False
        elif _temporal_label_key(a, pd) != _temporal_label_key(b, pd):
            return False
    return True


def _plain_temporal_fill(fill, pd):
    if type(fill) in (datetime, pd.Timestamp):
        zone = fill.tzinfo
        return (zone is None or type(zone) is timezone
                or type(zone) is getattr(modules.get('zoneinfo'), 'ZoneInfo', None))
    return type(fill) in (type(pd.NaT), np.datetime64, np.timedelta64, pd.Timedelta, timedelta, date)


def _can_replay_temporal_fill(fill, pd):
    # A detached sparse take still compares rows with its original fill.
    # Preserve a custom scalar's actual projection without calling it again.
    return (_plain_temporal_fill(fill, pd) or id(type(fill)) in _PLAIN_REAL_TYPES
            or type(fill) in (str, bytes, type(None)) or fill is pd.NA)


def _native_temporal_take_error(error, seed, indices, pd):
    sparse = modules['pandas._libs.sparse']
    if type(seed.sp_index) not in (sparse.IntIndex, sparse.BlockIndex):
        return False
    if type(error) is ValueError and seed.sp_values.dtype.kind == 'O' and _can_replay_temporal_fill(seed.fill_value, pd):
        # Pandas may infer a native dtype from the selected object points,
        # then reject the original (even unused) fill for that new dtype.
        trace = error.__traceback__
        while trace.tb_next is not None:
            trace = trace.tb_next
        check = getattr(pd.SparseDtype, '_check_fill_value', None)
        return check is not None and trace.tb_frame.f_code is check.__code__
    if type(error) is OverflowError:
        # Rebuilding the sparse rows compares their native ticks to the fill.
        # Either operand can be out of range in the other one's resolution.
        return _plain_temporal_fill(seed.fill_value, pd)
    if type(error).__name__ != 'DTypePromotionError' or not np.any(indices == -1):
        return False
    # Older pandas asks NumPy to promote native temporal points and the
    # missing-code NaN before constructing any rows. Preserve exceptions
    # originating inside a caller's fill or conversion callback.
    trace = error.__traceback__
    while trace.tb_next is not None:
        trace = trace.tb_next
    return trace.tb_frame.f_code is pd.arrays.SparseArray._take_with_fill.__code__


def _nested_sparse_projection(levels, codes, carrier, visible, pd, array_fields, mode):
    # pandas can coerce levels both while taking the codes and on the final
    # cast. Consult the actual outer conversion first, including any callback
    # on the selected rows. A detached reference contains no live descriptors.
    plain = _stored_sparse(levels, pd)
    if (plain is not None and type(codes) is np.ndarray and codes.size == visible.size
            and plain.sp_values.dtype.kind in 'biufc' and _sparse_text_fill(plain.fill_value, mode)
            and plain.sp_index.ngaps):
        legacy = isinstance(carrier, pd.DataFrame) and not hasattr(carrier._mgr, 'blocks')
        if _plain_temporal_levels(levels, pd, legacy=legacy, observed_array=True, array_fields=array_fields):
            return _sparse_text_rows(plain.sp_values, plain.sp_index, plain.fill_value, visible.shape, codes, mode=mode)
        try:
            ordinary = _ordinary_sparse_level_rows(plain, codes, legacy, pd)
        except (ValueError, TypeError, OverflowError):
            return visible
        if (_same_plain_projection(ordinary, visible)
                and not _same_numeric_ids(_sparse_text_rows(plain.sp_values, plain.sp_index, plain.fill_value,
                                                            visible.shape, codes, mode=mode).flat, visible.flat)):
            raise ValueError('Custom pandas sparse category conversion matches the ordinary conversion, '
                             'which changes a stored class ID. Pass exact labels or the intended projected labels.')
        return visible
    if (plain is not None and plain.sp_values.dtype.kind in 'MmO' and type(codes) is np.ndarray
            and codes.size == visible.size):
        if not _can_replay_temporal_fill(plain.fill_value, pd):
            return visible
        exact = None
        if plain.sp_values.dtype.kind == 'O':
            exact = _sparse_temporal_levels(plain, codes, pd)
            if exact is None:
                return visible
        legacy = isinstance(carrier, pd.DataFrame) and not hasattr(carrier._mgr, 'blocks')
        try:
            ordinary = _ordinary_sparse_level_rows(plain, codes, legacy, pd)
        except (TypeError, ValueError, OverflowError):
            return visible  # The selected operation succeeded where the detached default failed.
        if ordinary.dtype == visible.dtype and _same_temporal_rows(ordinary, visible, pd):
            if exact is None:
                exact = _sparse_temporal_levels(plain, codes, pd)
            if _plain_temporal_levels(levels, pd, legacy=legacy, observed_array=True, array_fields=array_fields):
                return exact.reshape(visible.shape)
            if not _same_temporal_rows(exact, visible, pd):
                raise ValueError('Custom pandas sparse category conversion matches the ordinary conversion, '
                                 'which changes a stored class ID. Pass np.asarray(y) to use the '
                                 'projected labels, or pass exact labels built from the stored scalars.')
        return visible
    if (plain is None or type(codes) is not np.ndarray or codes.ndim != 1 or (codes < 0).any()
            or visible.size != codes.size or visible.dtype.kind not in 'biufO'
            or plain.sp_values.dtype.kind not in 'biuf' or id(type(plain.fill_value)) not in _PLAIN_REAL_TYPES
            or visible.dtype.kind == 'O' and any(id(type(v)) not in _PLAIN_REAL_TYPES for v in visible.flat)):
        return visible
    used, codes = np.unique(codes, return_inverse=True)
    compact = plain.take(used)
    exact = _boxed_sparse_scalars(compact, pd, include_numeric=True)[codes]
    if _same_numeric_ids(exact, visible.flat):
        return visible
    legacy = isinstance(carrier, pd.DataFrame) and not hasattr(carrier._mgr, 'blocks')
    ordinary = _ordinary_sparse_level_rows(compact, codes, legacy, pd)
    if ordinary.dtype != visible.dtype or not _same_numeric_ids(ordinary.flat, visible.flat):
        return visible
    # An overridden fill leaves occupied points untouched when the point
    # buffer, index and conversion are all inherited. Their exact values can
    # then be recovered even if pandas rounded them during its sparse take.
    base = pd.arrays.SparseArray
    if ((legacy or getattr_static(levels, '_take_with_fill') is getattr_static(base, '_take_with_fill'))
            and not _custom_fields(levels, base, ('sp_values', 'sp_index', 'dtype', '_sparse_values',
                                                  '_sparse_index', '__getattribute__', '__getattr__'))):
        points = compact.sp_index.lookup_array(codes.astype(np.int32, copy=False)) >= 0
        if (not _custom_fields(levels, base, ('fill_value', '_dtype'))
                or all(is_point or _python_label(a) == _python_label(b)
                       for is_point, a, b in zip(points, exact, visible.flat))):
            return np.asarray(_exact_numeric_sequence(list(exact))).reshape(visible.shape)
    raise ValueError('Custom pandas sparse category conversion matches the ordinary conversion, '
                     'which changes a stored class ID. Pass np.asarray(y) to use the '
                     'projected labels, or pass exact labels built from the stored scalars.')


def _observed_levels(levels, selected, pd, *, taken=False, trace_fields=False):
    """Observe the actual categorical take or cast, including its returned array."""
    base = pd.arrays.SparseArray
    seed = _stored_sparse(levels, pd)
    if seed is None:
        return levels
    selected['observed_levels'] = True
    trace_fields = (trace_fields or seed.sp_values.dtype.kind in 'MmO'
                    or seed.sp_values.dtype.kind in 'biufc' and _sparse_text_fill(seed.fill_value, selected.get('mode')))
    array_fields = None

    class Levels(base):
        def __len__(self):
            return len(levels)

        def __iter__(self):
            if selected.pop('declined_levels', False):
                return _observed_iteration(levels, selected, 'rows' if taken else 'level_rows')
            return iter(levels)

        def __getitem__(self, key):
            return levels[key]

        def __getattribute__(self, name):
            nonlocal array_fields
            if array_fields is not None:
                array_fields.add(name)
            try:
                method = getattr(levels, name)
            except AttributeError:
                if name == '__array__':
                    selected['declined_levels'] = True
                raise
            if name == 'take':
                def take(*args, **kwargs):
                    indices = args[0] if args else kwargs.get('indices')
                    native = (not taken and seed.sp_values.dtype.kind in 'MmO' and kwargs.get('allow_fill')
                              and type(indices) is np.ndarray and indices.dtype.kind in 'iu'
                              and _bound_implementation(method, levels, base.take)
                              and _plain_temporal_levels(levels, pd))
                    if native and type(levels) is base and _can_replay_temporal_fill(seed.fill_value, pd):
                        # The built-in take would construct a fresh base
                        # array. A subclass can customize the returned rows.
                        rows = _sparse_temporal_levels(seed, indices, pd)
                        if rows is not None:
                            selected['temporal_rows'] = rows
                            return rows
                    try:
                        rows = method(*args, **kwargs)
                    except (OverflowError, TypeError, ValueError) as error:
                        # No returned array or getter exists when the native
                        # constructor itself fails on the catalogue's units.
                        if not native or not _native_temporal_take_error(error, seed, indices, pd):
                            raise
                        rows = _sparse_temporal_levels(seed, indices, pd)
                        if rows is None:
                            raise
                        selected['temporal_rows'] = rows
                        return rows
                    if _bound_implementation(method, levels, base.take) and isinstance(rows, base):
                        return _observed_levels(rows, selected, pd, taken=True, trace_fields=trace_fields)
                    return _read_taken_rows(rows, selected, pd)
                return take
            if name == 'astype' and _bound_implementation(method, levels, base.astype):
                return MethodType(base.astype, self)
            if name == '__array__' and _bound_implementation(method, levels, base.__array__):
                selected['default_level_array'] = True
                if trace_fields:
                    # Run pandas' selected method with its real operands. Its
                    # accesses depend on version and on gaps in the taken rows.
                    array_fields = selected.setdefault('array_fields', set())
                    return MethodType(base.__array__, self)
            return method

        if version_info >= (3, 12) and getattr_static(type(levels), '__buffer__', None) is not None:
            def __buffer__(self, flags):
                return memoryview(levels)

    return Levels._simple_new(seed.sp_values, seed.sp_index, seed.dtype)


class _LevelReads:
    def __init__(self, index, selected):
        self.index = index
        self.selected = selected

    def __len__(self):
        return len(self.index)

    def __getattr__(self, name):
        value = getattr(self.index, name)
        if name == '_values':
            self.selected['levels'] = value
            pd = modules['pandas']
            base = pd.arrays.SparseArray
            if isinstance(value, base) and (type(value) is not base or value.dtype.subtype.kind in 'MmO'
                    or _pandas_projection_kind(value, base, (*_ARRAY_PROTOCOLS, 'take', 'astype',
                                                             '_take_with_fill')) is not None):
                return _observed_levels(value, self.selected, pd)
            if isinstance(value, pd.api.extensions.ExtensionArray) and not isinstance(value, base):
                return _generic_array_reads(value, self.selected, follow_take=True)
        return value


class _CategoryReads:
    def __init__(self, owner):
        self.owner = owner
        self.selected = {}

    def __getattr__(self, name):
        value = getattr(self.owner, name)
        if name == 'categories':
            return _LevelReads(value, self.selected)
        if name == '_codes':
            self.selected['codes'] = value
        return value


class _SparseReads:
    def __init__(self, owner, selected):
        self.owner = owner
        self.selected = selected

    def __getattr__(self, name):
        value = getattr(self.owner, name)
        if name in ('sp_values', 'sp_index', 'fill_value'):
            self.selected[name] = value
        return value


class _SelectedVector:
    """Run an inherited pandas vector conversion on its already selected input."""

    def __init__(self, values, base):
        self.base = base
        self._data = values
        self._values = values
        self.__array__ = MethodType(base.__array__, self)


def _observed_iteration(array, selected, key='rows'):
    # NumPy materializes an iterator's rows before inferring their common
    # dtype. Freeze provider-backed rows now so normalization cannot replay them.
    source = iter(array)

    def capture():
        # Exceptions while yielding rows belong to consumption, after NumPy
        # has successfully obtained the iterator.
        rows, ordinary = _sequence_source(list(source))
        selected[key] = rows
        # NumPy consumes a pandas-selected iterator outside our own inference
        # call. Keep the original operands for comparison and yield compatible
        # text to the native dtype without invoking the provider again.
        compatible = None if ordinary else _native_string_subclass_rows(rows)
        yield from rows if compatible is None else compatible

    return capture()


def _iterated_text_target(rows, visible, mode):
    # NumPy may promote mixed scalar IDs to text after pandas selected
    # iteration. Inspect the captured rows without invoking another conversion.
    # The descriptor also surfaces a pending C-level conversion exception,
    # which some NumPy versions defer until the first use of the result.
    shape = _array_shape(visible)
    if not rows or len(shape) not in (1, 2) or shape[0] != len(rows) or visible.size != len(rows):
        return None
    if visible.dtype.kind not in 'SUTO' and not (mode == 'score' and visible.dtype.kind in 'biuf'):
        return None
    scalars = rows
    # _sequence_source has already materialized active row providers. A
    # retained list or tuple subclass still exposes its inherited elements.
    if _label_row(rows[0]):
        if not _singleton_label_rows(rows):
            return None
        scalars = [row[0] for row in rows]
    first_type = type(scalars[0])
    if mode != 'score' and issubclass(first_type, (str, bytes)):
        # The unbound method accepts only its own text family. In the common
        # case this checks every suffix without hashing rows or calling a hook.
        ends, nul = (str.endswith, '\0') if issubclass(first_type, str) else (bytes.endswith, b'\0')
        try:
            if not any(map(ends, scalars, repeat(nul))):
                return None
        except TypeError:
            pass  # A different scalar family requires the membership checks below.
    kinds = {id(cls): cls for cls in map(type, scalars)}
    if id(np.ndarray) in kinds:
        scalars = [_unbox_numeric_array(v, include_text=True) for v in scalars]
        kinds = {id(cls): cls for cls in map(type, scalars)}
    if kinds.keys() <= _ITERATED_NUMBER_TYPES:
        return None
    text = {key for key, cls in kinds.items() if issubclass(cls, (str, bytes))}
    if not text:
        return None
    numbers = kinds.keys() - text
    unicode = {key for key in text if issubclass(kinds[key], str)}
    if not numbers:
        # Mixing byte strings and Unicode can change a class. Within one
        # family only fixed-width truncation needs recovery, except for the
        # original subclass operands used in paired scoring.
        if (not unicode or unicode == text) and not (mode == 'score' and text - _TEXT_TYPES):
            ends, nul = (str.endswith, '\0') if unicode else (bytes.endswith, b'\0')
            if not any(map(ends, scalars, repeat(nul))):
                return None
    other = kinds.keys() - _TEXT_TYPES - _ITERATED_NUMBER_TYPES
    if other:
        protocols = {}
        for key in other:
            if key in text:
                protocols[key] = _TEXT_TYPES, _TEXT_FIELDS
                continue
            fields = _NUMBER_TEXT_FIELDS
            if type.__subclasscheck__(Fraction, kinds[key]):
                # Fraction implements comparison in Python using these operands
                # and, for floating peers, a conversion through the subclass.
                fields += ('numerator', 'denominator', '_numerator', '_denominator',
                           '_richcmp', 'from_float', '_from_coprime_ints', '__new__')
            elif issubclass(kinds[key], Decimal):
                fields += ('is_finite', 'to_integral_value')
            protocols[key] = _ITERATED_NUMBER_TYPES, fields
        cache = {}
        for value in scalars:
            check = protocols.get(id(type(value)))
            if check is not None and not _passive_scalar(value, check[0], check[1], cache):
                return None
    values = np.fromiter(scalars, dtype=object, count=len(scalars))
    # Inherited bytes can genuinely equal a NumPy integer. For those operands
    # fit and monitoring need the existing boxed class checks. Built-in bytes
    # and a passive numeric subclass remain distinct.
    inherited_bytes = not unicode and bool(text - _TEXT_TYPES)
    if mode == 'score' or numbers and not inherited_bytes:
        values = values.view(_SparseTextObservations)
    return values.reshape(visible.shape)


def _iterated_labels(rows, visible, mode):
    exact = _iterated_text_target(rows, visible, mode)
    if exact is not None:
        return exact
    if visible.ndim not in (1, 2) or len(rows) != visible.shape[0]:
        return visible
    if visible.dtype.kind not in 'biufOMm':
        return visible
    if visible.ndim == 1:
        return rows
    if visible.shape[1] != 1:
        return visible
    # A frame inserts a column axis after converting a one-dimensional
    # extension. A sequence of singleton rows already supplies that axis.
    if rows and _label_row(rows[0]):
        return rows
    return [[row] for row in rows]


def _read_taken_rows(rows, selected, pd):
    if isinstance(rows, (pd.api.extensions.ExtensionArray, list, tuple)):
        return _generic_array_reads(rows, selected)
    return rows


def _generic_array_reads(array, selected, *, follow_take=False):
    """Follow the actual NumPy protocol, retaining a selected sequence's rows."""
    class Reads:
        def __getattr__(self, name):
            method = getattr(array, name)
            if name in ('to_numpy', 'astype'):
                base = modules['pandas'].api.extensions.ExtensionArray
                implementation = getattr_static(base, name)
                if _bound_implementation(method, array, implementation):
                    # These pandas implementations convert self through NumPy.
                    return MethodType(implementation, self)
            if name == 'take' and follow_take:
                def take(*args, **kwargs):
                    return _read_taken_rows(method(*args, **kwargs), selected, modules['pandas'])
                return take
            return method

        def __len__(self):
            return len(array)

        def __getitem__(self, key):
            return array[key]

        def __iter__(self):
            return _observed_iteration(array, selected, 'level_rows' if follow_take else 'rows')

        if version_info >= (3, 12) and getattr_static(type(array), '__buffer__', None) is not None:
            def __buffer__(self, flags):
                return memoryview(array)

    return Reads()


def _generic_projection(array, carrier, pd, mode):
    selected = {}
    reads = _generic_array_reads(array, selected)
    if carrier is array:
        projected = reads
    elif isinstance(carrier, _SelectedVector):
        projected = _SelectedVector(reads, carrier.base)
    elif isinstance(carrier, (pd.Series, pd.Index)):
        base = pd.Series if isinstance(carrier, pd.Series) else pd.Index
        projected = _SelectedVector(reads, base)
    else:
        return _FrameDataReads(carrier, pd, mode).labels()
    visible = np.asarray(projected)
    return _iterated_labels(selected['rows'], visible, mode) if 'rows' in selected else visible


def _observed_manager(manager, observer):
    """Substitute only the data used by this pandas manager's own conversion."""
    series = manager.ndim == 1
    if hasattr(manager, 'blocks'):
        original = manager.blocks[0]

        class ObservedBlock(type(original)):
            def __getattribute__(self, name):
                if name == 'values':
                    return observer
                # Let pandas resolve and cache block properties on the real
                # block. Warm and cold conversions can select different
                # extension getters, including on the next fit or score.
                return getattr(original, name)

        block = ObservedBlock(original.values, placement=original.mgr_locs, ndim=original.ndim)
        if series:
            shadow = type(manager)(block, manager.axes[0])
        else:
            shadow = type(manager)((block,), manager.axes, verify_integrity=False)
    else:
        shadow = type(manager)([observer], manager._axes, verify_integrity=False)
    return shadow


def _observed_carrier(carrier, observer, pd):
    if isinstance(carrier, _SelectedVector):
        return _SelectedVector(observer, carrier.base)
    if isinstance(carrier, pd.Index):
        return pd.Index._simple_new(observer)
    if not isinstance(carrier, (pd.Series, pd.DataFrame)):
        return observer
    shadow = _observed_manager(carrier._mgr, observer)
    container = pd.Series if isinstance(carrier, pd.Series) else pd.DataFrame
    return container._from_mgr(shadow, axes=shadow.axes)


def _storage_observer(array, pd, mode):
    """Capture only operands selected by an inherited extension conversion."""
    category = isinstance(array, pd.Categorical)
    base = pd.Categorical if category else pd.arrays.SparseArray
    selected = {}

    class Observer(base):
        def __len__(self):
            return len(array)

        def __iter__(self):
            if selected.pop('declined_array', False):
                return _observed_iteration(array, selected)
            return iter(array)

        def __getitem__(self, key):
            return array[key]

        if version_info >= (3, 12) and getattr_static(type(array), '__buffer__', None) is not None:
            def __buffer__(self, flags):
                return memoryview(array)

        def __getattribute__(self, name):
            try:
                value = getattr(array, name)
            except AttributeError:
                if name == '__array__':
                    selected['declined_array'] = True
                raise
            if name in ('__array__', 'to_numpy', 'astype'):
                implementation = getattr_static(base, name, None)
                if _bound_implementation(value, array, implementation):
                    if name == '__array__' or category and name == 'astype':
                        def convert(*args, **kwargs):
                            if category:
                                inputs = _CategoryReads(array)
                                inputs.selected['mode'] = mode
                                result = implementation(inputs, *args, **kwargs)
                                selected.update(inputs.selected)
                                return result
                            result = implementation(_SparseReads(array, selected), *args, **kwargs)
                            selected['array_dtype'] = result.dtype
                            return result
                        return convert
                    # A built-in outer conversion can call the array protocol
                    # itself. Keep following pandas until it selects the data.
                    return MethodType(implementation, self)
            return value

    if category:
        raw = getattr_static(pd.Categorical, '_ndarray').__get__(array)
        seed = pd.Categorical._simple_new(np.broadcast_to(np.int8(0), raw.shape), pd.CategoricalDtype([0]))
        observer = Observer._simple_new(seed._codes, seed.dtype)
    else:
        seed = _stored_sparse(array, pd)
        if seed is None:
            return None, None
        # A fill and point buffer with the same native integer subtype
        # have an exact common representation, including all uint64 IDs.
        if (seed.sp_values.dtype.kind in 'biu' and type(seed.fill_value) is seed.sp_values.dtype.type
                and getattr_static(array, '__array__', None) is getattr_static(base, '__array__')
                and not _custom_fields(array, base, ('__getattribute__', '__getattr__', 'fill_value', 'dtype',
                                                     'sp_values', 'sp_index', '_dtype', '_sparse_values',
                                                     '_sparse_index'))):
            return None, None
        observer = Observer._simple_new(seed.sp_values, seed.sp_index, seed.dtype)
    return observer, selected


def _observed_projection(array, carrier, pd, mode):
    observer, selected = _storage_observer(array, pd, mode)
    if observer is None:
        return np.asarray(carrier)
    visible = np.asarray(_observed_carrier(carrier, observer, pd))
    return _observed_labels(array, carrier, selected, visible, pd, mode)


def _observed_labels(array, carrier, selected, visible, pd, mode):
    if 'temporal_rows' in selected and selected['temporal_rows'].size == visible.size:
        return selected['temporal_rows'].reshape(visible.shape)
    if 'rows' in selected:
        return _iterated_labels(selected['rows'], visible, mode)
    if isinstance(array, pd.Categorical):
        levels, codes = selected.get('levels'), selected.get('codes')
        if 'level_rows' in selected and type(codes) is np.ndarray and codes.size == visible.size:
            rows = selected['level_rows']
            observed = [rows[code] if code >= 0 else missing for code, missing in zip(codes.ravel(), visible.flat)]
            return _iterated_labels(observed, visible, mode)
        if not isinstance(levels, pd.arrays.SparseArray) or type(codes) is not np.ndarray:
            return visible
        if selected.get('observed_levels'):
            if not selected.get('default_level_array'):
                return visible
            return _nested_sparse_projection(levels, codes, carrier, visible, pd, selected.get('array_fields', ()), mode)
        if codes.size != visible.size:
            return visible
        text_rows = _sparse_text_rows(levels.sp_values, levels.sp_index, levels.fill_value, visible.shape, codes, mode=mode)
        if text_rows is not None:
            return text_rows
        if levels.dtype.subtype.kind in 'Mm':
            return _sparse_temporal_levels(levels, codes, pd).reshape(visible.shape)
        if levels.dtype.subtype.kind not in 'biufO' or visible.dtype.kind not in 'biufO':
            return visible
        # A valid catalogue can have sparse levels whose dense floating view
        # is no longer unique. Expand only the selected operands and preserve
        # their original scalar objects for later paired comparisons.
        positions, used = pd.factorize(codes.ravel())
        scalars = _occupied_sparse_levels(levels, used)
        rows = np.fromiter(scalars, dtype=object, count=len(scalars))[positions].reshape(visible.shape)
        if rows.size and all(isinstance(v, (Real, np.bool_)) for v in scalars):
            return rows.tolist()
        return rows
    else:
        index = selected.get('sp_index')
        if index is None or not index.ngaps:
            return visible
        points, fill = selected['sp_values'], selected['fill_value']
        if type(points) is np.ndarray:
            text_rows = _sparse_text_rows(points, index, fill, visible.shape, mode=mode)
            if text_rows is not None:
                return text_rows
        if type(points) is np.ndarray and points.dtype.kind in 'Mm':
            dense = _sparse_temporal_target(points, fill, index, visible.shape, pd)
            return visible if dense is None else dense
        if (carrier is array and type(points) is np.ndarray and points.dtype.kind == 'c'
                and visible.dtype.kind in 'SU'):
            raise ValueError('Sparse complex target has an incompatible occupied fill.')
        if type(points) is not np.ndarray or points.dtype.kind not in 'biufO':
            return visible
        # Defer an observed temporal sentinel to the consumer's fit/score
        # policy when rebuilding the selected sparse rows below.
        if (points.dtype == np.uint64 and not _is_temporal_nat(fill, pd)
                and not _integer_label_fits(_base_numeric_scalar(fill), np.uint64)):
            raise ValueError('Sparse uint64 target contains a fill value that is '
                             'not representable as a uint64 class label.')
        if selected['array_dtype'] == points.dtype == visible.dtype and _native_sparse_fill(fill, points.dtype):
            return visible
        if index.length != visible.size:
            # A sparse subclass may extend its logical length. Its inherited
            # conversion fills the entire exposed shape before writing points.
            rows = np.empty(visible.size, dtype=object)
            rows.fill(fill)
            rows[index.to_int_index().indices] = np.fromiter(points, dtype=object, count=len(points))
            return rows.reshape(visible.shape)
        try:
            dtype = pd.SparseDtype(points.dtype, fill)
        except (TypeError, ValueError):
            points = np.fromiter(points, dtype=object, count=len(points))
            dtype = pd.SparseDtype(object, fill)
        frozen = pd.arrays.SparseArray._simple_new(points, index, dtype)
    stored = _storage_layout(frozen, carrier, pd)
    return stored if stored.shape == visible.shape else visible


def _same_numeric_ids(left, right):
    return all(a == b for a, b in zip(map(_python_label, left), map(_python_label, right)))


def _same_plain_projection(ordinary, selected):
    if ordinary.dtype != selected.dtype or ordinary.size != selected.size:
        return False
    scalars = _PLAIN_REAL_TYPES | _TEXT_TYPES
    if selected.dtype.kind == 'O' and any(id(type(v)) not in scalars for v in selected.flat):
        return False
    return _same_numeric_ids(ordinary.flat, selected.flat)


def _ordinary_sparse_frame_rows(plain, codes, manager, pd):
    if codes is not None:
        return _ordinary_sparse_level_rows(plain, codes, not hasattr(manager, 'blocks'), pd)
    reference = pd.DataFrame({0: plain}, copy=False)
    if hasattr(manager, 'blocks') != hasattr(reference._mgr, 'blocks'):
        # pandas 2.x permits frames created under different manager settings.
        reference = reference._as_manager('block' if hasattr(manager, 'blocks') else 'array', copy=False)
    return np.asarray(reference)


def _selected_pandas_data(value, base, pd, mode):
    data = value._data if base is pd.Index else value._values
    carrier = _SelectedVector(data, base)
    if isinstance(data, (pd.Categorical, pd.arrays.SparseArray)):
        return _observed_projection(data, carrier, pd, mode)
    if isinstance(data, pd.api.extensions.ExtensionArray):
        return _generic_projection(data, carrier, pd, mode)
    return np.asarray(carrier)


def _check_dynamic_sparse_frame(frame, projected, pd, mode):
    """Refuse an unwitnessed projection indistinguishable from rounded sparse IDs."""
    if not projected.size or projected.dtype.kind not in 'biufOMmSU':
        return
    # The original data is a comparison reference only. No user protocol or
    # storage accessor can be replayed to infer which lookup actually won.
    state = getattr_static(pd.DataFrame, '__dict__').__get__(frame)
    manager = state.get('_mgr')
    if manager is None or len(manager.arrays) != 1:
        return
    storage = manager.arrays[0]
    codes = None
    if isinstance(storage, pd.Categorical):
        dtype = getattr_static(pd.Categorical, '_dtype').__get__(storage)
        if type(dtype) is not pd.CategoricalDtype:
            return
        index = getattr_static(pd.CategoricalDtype, '__dict__').__get__(dtype).get('_categories')
        if not isinstance(index, pd.Index):
            return
        storage_data = getattr_static(pd.Index, '__dict__').__get__(index).get('_data')
        codes = getattr_static(pd.Categorical, '_ndarray').__get__(storage)
        if type(codes) is not np.ndarray or codes.ndim != 1:
            return
        storage = storage_data
    if not isinstance(storage, pd.arrays.SparseArray):
        return
    plain = _stored_sparse(storage, pd)
    if (plain is not None and plain.sp_values.dtype.kind in 'biufc' and _sparse_text_fill(plain.fill_value, mode)
            and plain.sp_index.ngaps and projected.shape == (len(plain) if codes is None else codes.size, 1)):
        try:
            ordinary = _ordinary_sparse_frame_rows(plain, codes, manager, pd)
        except (ValueError, TypeError, OverflowError):
            return
        if _same_plain_projection(ordinary, projected):
            exact = _sparse_text_rows(plain.sp_values, plain.sp_index, plain.fill_value, projected.shape, codes, mode=mode)
            if not _same_numeric_ids(exact.flat, projected.flat):
                raise ValueError('Dynamic pandas conversion matches the ordinary sparse conversion, which changes '
                                 'a stored class ID. Pass exact labels or the intended projected labels.')
        return
    if projected.dtype.kind in 'SU':
        return
    if plain is not None and plain.sp_values.dtype.kind in 'Mm':
        if (projected.shape != (len(plain) if codes is None else codes.size, 1)
                or not _can_replay_temporal_fill(plain.fill_value, pd)):
            return
        try:
            ordinary = _ordinary_sparse_frame_rows(plain, codes, manager, pd)
        except (TypeError, ValueError, OverflowError):
            return
        if ordinary.dtype != projected.dtype or not _same_temporal_rows(ordinary, projected, pd):
            return
        if codes is None:
            if not plain.sp_index.ngaps:
                return
            exact = _sparse_temporal_target(plain.sp_values, plain.fill_value, plain.sp_index, plain.shape, pd)
            if exact is None:
                return
        else:
            exact = _sparse_temporal_levels(plain, codes, pd)
        if not _same_temporal_rows(exact, projected, pd):
            raise ValueError('Dynamic pandas conversion matches the ordinary sparse conversion, '
                             'which changes a stored class ID. Pass np.asarray(y) to use the '
                             'projected labels, or pass exact labels built from the stored scalars.')
        return
    if (plain is None or plain.sp_values.dtype.kind not in 'biufO'
            or id(type(plain.fill_value)) not in _PLAIN_REAL_TYPES or (codes is not None and (codes < 0).any())):
        return
    if codes is not None:
        # Positional take preserves the sparse subtype, fill and exact points.
        # Unused catalogue entries are never boxed or supplied to the reference.
        used, codes = np.unique(codes, return_inverse=True)
        plain = plain.take(used)
    if ((plain.sp_values.dtype.kind == 'O' and any(id(type(v)) not in _PLAIN_REAL_TYPES for v in plain.sp_values))
            or projected.dtype.kind == 'O' and any(id(type(v)) not in _PLAIN_REAL_TYPES for v in projected.flat)):
        return
    exact = _boxed_sparse_scalars(plain, pd, include_numeric=True)
    if codes is not None:
        exact = exact[codes]
    if projected.shape != (len(exact), 1) or _same_numeric_ids(exact, projected.flat):
        return
    ordinary = _ordinary_sparse_frame_rows(plain, codes, manager, pd)
    if ordinary.dtype == projected.dtype and _same_numeric_ids(ordinary.flat, projected.flat):
        raise ValueError('Dynamic pandas conversion matches the ordinary sparse conversion, '
                         'which changes a stored class ID. Pass np.asarray(y) to use the '
                         'projected labels, or pass exact labels built from the stored scalars.')


def _pandas_sequence_data(value):
    # NumPy checks the length before obtaining the iterator. Preserve its raw
    # rows for the same numeric and shape validation as any other sequence.
    try:
        len(value)
    except (MemoryError, RecursionError):
        raise
    except Exception:
        scalar = np.empty((), dtype=object)
        scalar[()] = value
        return scalar
    try:
        rows = iter(value)
    except TypeError as error:
        raise TypeError('Could not convert object to sequence') from error
    return list(rows)


class _FrameDataReads:
    """Observe the inherited values property's actual manager.as_array call."""

    def __init__(self, frame, pd, mode):
        self.mode = mode
        self.frame = frame
        self.pd = pd
        self.reading_values = False
        self.witness = None
        self.iteration = {}

    def __getattr__(self, name):
        if name in ('_values', 'values'):
            previous = self.reading_values
            self.reading_values = name == 'values'
            try:
                data = getattr_static(self.pd.DataFrame, name).__get__(self)
                if name == '_values' and isinstance(data, self.pd.api.extensions.ExtensionArray):
                    return _generic_array_reads(data, self.iteration)
                return data
            finally:
                self.reading_values = previous
        value = getattr(self.frame, name)
        if name != '_mgr' or not self.reading_values:
            return value

        def as_array(*args, **kwargs):
            method = value.as_array
            # An explicitly supplied manager conversion owns its result,
            # including any chosen rounding or exception.
            base = _manager_base(value)
            if base is None or not _bound_implementation(method, value, base.as_array) or len(value.items) != 1:
                return method(*args, **kwargs)
            data = value.arrays[0]
            if not isinstance(data, self.pd.api.extensions.ExtensionArray):
                return method(*args, **kwargs)
            if isinstance(data, (self.pd.Categorical, self.pd.arrays.SparseArray)):
                observer, selected = _storage_observer(data, self.pd, self.mode)
            else:
                selected = self.iteration
                observer = _generic_array_reads(data, selected)
            if observer is None:
                return method(*args, **kwargs)
            visible = _observed_manager(value, observer).as_array(*args, **kwargs)
            if selected is not self.iteration:
                self.witness = data, value, selected
            return visible

        return SimpleNamespace(as_array=as_array)

    def labels(self):
        visible = self.pd.DataFrame.__array__(self)
        if 'rows' in self.iteration:
            return _iterated_labels(self.iteration['rows'], visible, self.mode)
        if self.witness is None:
            return visible
        data, manager, selected = self.witness
        carrier = self.pd.DataFrame._from_mgr(manager, axes=manager.axes)
        return _observed_labels(data, carrier, selected, visible, self.pd, self.mode)


def _pandas_projection(value, base, carrier, mode, *, extension=False):
    pd = modules['pandas']
    sparse = extension and isinstance(value, pd.arrays.SparseArray)
    category = extension and isinstance(value, pd.Categorical)
    frame = extension and isinstance(carrier, pd.DataFrame)
    legacy_frame = frame and getattr(carrier._mgr, 'blocks', None) is None
    protocols = _EXTENSION_PROTOCOLS if extension else _ARRAY_PROTOCOLS
    if base is pd.DataFrame:
        # The frame conversion reads _values, which can delegate to values
        # for a one-dimensional extension block.
        protocols = (*protocols, '_values', 'values')
    if (sparse or category) and (not frame or legacy_frame):
        # These __array__ implementations do not delegate to to_numpy.
        protocols = _ARRAY_PROTOCOLS
    if legacy_frame:
        protocols = (*protocols, 'astype')
    kind = _pandas_projection_kind(value, base, protocols)
    if base is pd.DataFrame and kind is None:
        if _custom_fields(value, base, ('_mgr',)):
            kind = 'declared'
        else:
            manager = value._mgr
            if type(manager) is not _manager_base(manager) or 'as_array' in vars(manager):
                kind = 'declared'
    cls = type(value)
    user_vector = ((base is pd.Index or base is pd.Series) and cls is not base
                   and getattr(pd, cls.__name__, None) is not cls)
    if user_vector and kind is None:
        # A Series can consult its manager while pandas looks up a missing
        # NumPy protocol, before __array__ selects the data.
        kind = 'dynamic'
    if extension and not (sparse or category):
        # pandas selects the conversion for the public carrier, including
        # the transposed storage of a two-dimensional extension block. Only
        # sparse and categorical storage needs recovery from a lossy default.
        return None if kind is None else _generic_projection(value, carrier, pd, mode)
    if sparse or category:
        if cls is not base or kind is not None:
            return _observed_projection(value, carrier, pd, mode)
        return None
    if kind is None:
        return None
    if version_info < (3, 12) or getattr_static(cls, '__buffer__', None) is not None:
        # NumPy first requests a readable buffer and falls through if the
        # exporter declines. Keep the acquired view and its owner alive.
        try:
            exported = memoryview(value)
        except Exception:
            pass
        else:
            return np.asarray(exported)
    for name in ('__array_struct__', '__array_interface__', '__array__'):
        protocol = getattr(value, name, _MISSING)
        if protocol is _MISSING:
            continue
        inherited = name == '__array__' and _bound_implementation(protocol, value, base.__array__)
        if not inherited:
            return np.asarray(SimpleNamespace(owner=value, **{name: protocol}))
        if base is pd.DataFrame:
            custom_values = _custom_fields(value, base, ('_values', 'values'))
            dynamic_lookup = _custom_fields(value, base, ('__getattribute__',))
            if custom_values or dynamic_lookup:
                visible = np.asarray(SimpleNamespace(owner=value, __array__=protocol))
                if dynamic_lookup and not custom_values:
                    _check_dynamic_sparse_frame(value, visible, pd, mode)
                return visible
            return _FrameDataReads(value, pd, mode).labels()
        return _selected_pandas_data(value, base, pd, mode)
    return _pandas_sequence_data(value)


def _pandas_extension_base(value, pd):
    # Include third-party extensions as well as subclasses of pandas' concrete
    # arrays. Inherited pandas implementations keep their exact-storage paths.
    for cls in type(value).__mro__:
        if getattr(pd.arrays, cls.__name__, None) is cls:
            return cls
    return pd.api.extensions.ExtensionArray


def _categorical_level_projection(array, carrier, pd, mode):
    categories = array.categories
    legacy_frame = isinstance(carrier, pd.DataFrame) and getattr(carrier._mgr, 'blocks', None) is None
    cls = type(categories)
    if getattr(pd, cls.__name__, None) is not cls:
        base = next(t for t in cls.__mro__ if getattr(pd, t.__name__, None) is t)
        if _custom_fields(categories, base, ('_values', '_data', '__getattribute__', '__getattr__')):
            return _observed_projection(array, carrier, pd, mode)
    if isinstance(categories, (pd.RangeIndex, pd.MultiIndex)):
        return None
    levels = categories._values
    if not isinstance(levels, pd.api.extensions.ExtensionArray):
        return None
    base = _pandas_extension_base(levels, pd)
    sparse = isinstance(levels, pd.arrays.SparseArray)
    protocols = (*_ARRAY_PROTOCOLS, '_take_with_fill') if sparse else _EXTENSION_PROTOCOLS
    kind = _pandas_projection_kind(levels, base, (*protocols, 'astype' if legacy_frame else 'take'))
    if sparse and (type(levels) is not base or kind is not None
                   or levels.dtype.subtype.kind in 'Mm' and not _can_replay_temporal_fill(levels.fill_value, pd)):
        return _observed_projection(array, carrier, pd, mode)
    if kind is not None:
        return _observed_projection(array, carrier, pd, mode)
    return None


def _wide_integral_float_target(values, *, native_only=False):
    """Keep exact integer class IDs outside sklearn's signed target-check range."""
    if not values.size or not (values.ndim == 1 or values.ndim == 2 and values.shape[1] == 1):
        return values
    low, high = values.min(), values.max()
    if np.isfinite(low) and np.isfinite(high):
        low, high = int(low), int(high)
        bounds = np.iinfo(int)
        if (low < bounds.min or high > bounds.max) and (values == np.trunc(values)).all():
            dtype = _integer_dtype(low, high)
            if dtype is object:
                return values if native_only else _integral_target(values)
            return values.astype(dtype)
    return values


def _projected_pandas_target(projected, mode, *, selected=False):
    if isinstance(projected, _SparseTextObservations):
        return projected, _sparse_text_target(projected, mode)
    if isinstance(projected, _MixedTemporalObservations):
        if mode in ('score', 'observations'):
            return projected, projected
        _reject_nat_training_labels(projected)
        raise ValueError('Could not infer an exact shared unit for the temporal class labels. '
                         'Supply a NumPy temporal array in an explicit unit that represents '
                         'every instant or duration exactly.')
    if selected:
        temporal = _selected_boxed_temporal_target(projected, mode)
        if temporal is not None:
            return projected, temporal
        if type(projected) in (list, tuple):
            # The outer sequence fallback has only captured rows. Use the
            # same inference for text recovery and ordinary numeric labels.
            source, ordinary = _sequence_source(projected)
            target = _exact_numeric_sequence(source, mode=mode, ordinary=ordinary, iterated=True)
            return (target if isinstance(target, _SparseTextObservations) else source), target
    if isinstance(projected, np.ndarray):
        if (projected.dtype.kind == 'O' and projected.size
                and (projected.ndim == 1 or projected.ndim == 2 and projected.shape[1] == 1)
                and all(id(type(v)) in _PLAIN_INTEGER_TYPES for v in projected.flat)):
            return projected, _integral_target(projected)
        if projected.dtype.kind == 'f':
            return projected, _wide_integral_float_target(projected)
    return _label_source_and_target(projected, mode)


def _label_source_and_target(y, mode):
    """Keep the represented source for paired equality and normalize it once."""
    if isinstance(y, (list, tuple)):
        y = _sequence_representation(y)
        if isinstance(y, (list, tuple)):
            source, ordinary = _sequence_source(y)
            return source, _exact_numeric_sequence(source, mode=mode, ordinary=ordinary)
    elif not isinstance(y, np.ndarray):
        pd = modules.get('pandas')
        pandas_target = pd is not None and isinstance(
            y, (pd.Series, pd.Index, pd.DataFrame, pd.api.extensions.ExtensionArray))
        if pandas_target:
            outer = (pd.Series if isinstance(y, pd.Series) else pd.Index if isinstance(y, pd.Index)
                     else pd.DataFrame if isinstance(y, pd.DataFrame) else None)
            projected = _pandas_projection(y, outer, y, mode) if outer is not None else None
            if projected is not None:
                return _projected_pandas_target(projected, mode, selected=True)
            storage = _pandas_target_array(y, include_boxed=True)
            if storage is not None and not isinstance(storage, np.ndarray):
                projected = _pandas_projection(storage, _pandas_extension_base(storage, pd), y, mode, extension=True)
                if projected is None and isinstance(storage, pd.Categorical):
                    projected = _categorical_level_projection(storage, y, pd, mode)
            if projected is not None:
                return _projected_pandas_target(projected, mode, selected=True)
            target = _pandas_array_target(y, storage, pd, mode)
            return (target if isinstance(target, _SparseTextObservations) else y), target
        kind = _sequence_kind(y)
        if kind:
            source = _sequence_representation(y, indexed=kind == 'indexed')
            if source is not y:
                return _label_source_and_target(source, mode)
        if _declares_array(y) or _dynamic_attributes(y):
            array = np.asarray(y)
            if array.ndim or array.dtype.kind != 'O' or array[()] is not y:
                y = array
    if (mode == 'score' and isinstance(y, np.ndarray) and y.dtype.kind == 'f'
            and y.dtype not in _SKLEARN_NATIVE_FLOAT_DTYPES):
        return y, _wide_integral_float_target(np.asarray(y))
    if isinstance(y, np.ndarray) and y.dtype.kind == 'O':
        if isinstance(y, _SparseTextObservations):
            return y, _sparse_text_target(y, mode)
        if isinstance(y, _MixedTemporalObservations):
            return _projected_pandas_target(y, mode)
        return y, _boxed_numeric_target(y, np.asarray(y), mode=mode)
    return y, y


def exact_sparse_uint64_target(y, *, mode='normalize'):
    """Read IDs before dtype inference; fit keeps safe representatives, observations need only paired equality."""
    return _label_source_and_target(y, mode)[1]


def _pandas_array_target(y, array, pd, mode):
    if array is None:
        return y
    if isinstance(array, np.ndarray):
        return _boxed_numeric_target(y, array, mode=mode)
    frame_shape = y._mgr.shape[::-1] if isinstance(y, pd.DataFrame) else None
    if isinstance(array, pd.Categorical):
        dense = _categorical_integer_target(array, pd, mode)
        categories = array.categories
        sparse = isinstance(categories.dtype, pd.SparseDtype)
        if sparse and categories.dtype.subtype.kind in 'Mm':
            dense = _sparse_temporal_levels(categories._values, array._codes, pd).reshape(frame_shape or array.shape)
            return _projected_pandas_target(dense, mode)[1]
        sparse_objects = (sparse
                          and categories.dtype.subtype.kind == 'O'
                          and isinstance(categories._values, pd.arrays.SparseArray))
        if dense is None and (categories.dtype == object or sparse_objects):
            codes = array._codes
            if (codes < 0).any():
                if sparse_objects:
                    temporal = _sparse_temporal_levels(categories._values, codes, pd)
                    if temporal is not None:
                        rows = temporal.reshape(frame_shape or array.shape)
                        normalized = _selected_boxed_temporal_target(rows, mode)
                        return rows if normalized is None else normalized
                return y  # Keep the caller's missing-value validation.
            positions, used = pd.factorize(codes)
            if sparse_objects:
                # A sparse fill can lose its scalar type during densification.
                # Read only the levels referenced by observations.
                levels_array = categories._values
                levels = np.fromiter(_occupied_sparse_levels(levels_array, used),
                                     dtype=object, count=len(used))
            else:
                levels = np.asarray(categories._values)[used]
            normalized = _selected_boxed_temporal_target(levels, mode) if sparse_objects else None
            if normalized is None:
                normalized = _boxed_numeric_target(levels, levels, mode=mode)
            if sparse_objects or normalized is not levels:
                dense = normalized[positions]
        return y if dense is None else dense.reshape(frame_shape or array.shape)
    if not isinstance(array, pd.arrays.SparseArray):
        if (mode in ('score', 'observations') and type(array) in (pd.arrays.DatetimeArray, pd.arrays.TimedeltaArray)
                and isinstance(array.dtype, np.dtype)):
            # The selected timezone-naive array already has exact native ticks.
            # Expose them once, preserving a one-column frame's shape.
            return np.asarray(array).reshape(frame_shape or array.shape)
        if isinstance(array, getattr(pd.arrays, 'NumpyExtensionArray', ())):
            if array.dtype.kind == 'T':
                return np.asarray(y)
            if array.dtype.kind == 'O':
                return _boxed_numeric_target(y, array.to_numpy(copy=False), mode=mode)
        if array.dtype.kind in 'iu' and hasattr(array.dtype, 'numpy_dtype'):
            return array.to_numpy(dtype=array.dtype.numpy_dtype, copy=True).reshape(frame_shape or array.shape)
        return y
    scalars = _boxed_sparse_scalars(array, pd)
    if scalars is not array:
        temporal = _selected_boxed_temporal_target(scalars, mode)
        if temporal is not None:
            return temporal.reshape(frame_shape or array.shape)
        return _boxed_numeric_target(scalars, scalars, mode=mode).reshape(frame_shape or array.shape)
    if array.dtype.subtype.name != 'uint64':
        if array.dtype.subtype.kind in 'Mm':
            shape = frame_shape or array.shape
            if array.sp_index.ngaps:
                dense = _sparse_temporal_target(array.sp_values, array.fill_value, array.sp_index, shape, pd)
                if dense is not None:
                    return _projected_pandas_target(dense, mode)[1]
            if frame_shape is not None or mode in ('observations', 'score'):
                return np.asarray(array).reshape(shape)
            return y
        dense = _sparse_real_target(array, pd, mode)
        if dense is not None:
            return dense.reshape(frame_shape or array.shape)
        # sklearn rejects bare SparseArrays in classification metrics and
        # converts sparse frames to SciPy matrices. Use the selected values.
        bare_observations = y is array and mode in ('observations', 'score')
        if (bare_observations or frame_shape is not None) and array.dtype.subtype.kind in 'biuf':
            # An occupied nonnumeric fill can coerce numeric points to text.
            if array.sp_index.ngaps and id(type(array.fill_value)) not in _PLAIN_REAL_TYPES:
                return y
            values = np.asarray(array).reshape(frame_shape or array.shape)
            return _projected_pandas_target(values, mode)[1]
        return y

    # A SparseDtype can describe a non-integer fill even with uint64 stored
    # values. Validate it only when there really are implicit observations.
    if array.sp_index.ngaps:
        fill = array.fill_value
        _check_training_fill(fill, mode)
        rows = _sparse_text_rows(array.sp_values, array.sp_index, fill, frame_shape or array.shape, mode=mode)
        if rows is not None:
            return _sparse_text_target(rows, mode)
        custom = _custom_numeric_fill(fill)
        if not _integer_label_fits(_base_numeric_scalar(fill) if custom else fill, np.uint64):
            raise ValueError('Sparse uint64 target contains a fill value that is '
                             'not representable as a uint64 class label.')
        if custom:
            scalars = _boxed_sparse_scalars(array, pd, include_numeric=True)
            return _boxed_numeric_target(scalars, scalars, mode=mode).reshape(frame_shape or array.shape)
    # Specifying the dtype during densification avoids a float64 intermediate.
    # Make an owning copy even when all positions have stored values.
    dense = pd.arrays.SparseArray.to_numpy(array, dtype=np.uint64, copy=True)
    return dense.reshape(frame_shape or array.shape)


def _accuracy_for_mixed_ids(integers, floats, float_source, weights):
    """Compare integral floats and 64-bit IDs without rounding the integer side."""
    integers, floats = column_or_1d(integers), column_or_1d(floats)
    check_consistent_length(integers, floats)
    if getattr(getattr(float_source, 'dtype', None), 'kind', None) == 'f':
        bounds = np.iinfo(integers.dtype)
        in_range = (floats >= np.float64(bounds.min)) & (floats < np.float64(int(bounds.max) + 1))
        matches = np.zeros(len(floats), dtype=bool)
        matches[in_range] = integers[in_range] == floats[in_range].astype(integers.dtype)
    else:
        # A Python sequence can contain integer scalars already rounded in
        # the inferred floating array. Read those scalars from the source.
        original = column_or_1d(np.asarray(float_source, dtype=object))
        matches = np.fromiter((int(u) == int(v) for u, v in zip(integers, original)),
                              dtype=bool, count=len(integers))
    return accuracy_score(np.ones_like(matches), matches, sample_weight=weights)


def _temporal_label_matches(truth, labels, *, paired):
    left, right = column_or_1d(truth), column_or_1d(labels)
    if paired:
        check_consistent_length(left, right)
    if left.dtype.kind in 'Mm' and left.dtype == right.dtype:
        return left == right

    pd = modules.get('pandas')
    kinds, domains = set(), []
    for source, values in ((truth, left), (labels, right)):
        mixed = isinstance(source, _MixedTemporalObservations)
        if not mixed and values.dtype.kind in 'Mm':
            kinds.add(values.dtype.kind)
        elif not mixed:
            raise ValueError('Mixed-unit temporal observations require temporal labels.')
        # A NaT's unit carries no observed instant or duration. A missing-only
        # operand cannot match a class or create a fixed/calendar unit conflict.
        sample = next((value for value in values if not _temporal_missing(value)), None)
        if sample is not None:
            if mixed:
                kinds.add('m' if type(sample) is timedelta else sample.dtype.kind)
            domains.append(_temporal_label_key(sample, pd)[0])
    if len(kinds) > 1 or len(set(domains)) > 1:
        raise ValueError('Temporal observations and predictions have incompatible units.')
    if len(domains) < 2:
        return np.zeros(len(left), dtype=bool)
    others = right if paired else repeat(right[0])
    return np.fromiter((not _temporal_missing(a) and not _temporal_missing(b)
                        and _temporal_label_key(a, pd) == _temporal_label_key(b, pd)
                        for a, b in zip(left, others)), dtype=bool, count=len(left))


def _temporal_comparison_targets(observations, peer):
    """Recover boxed scalars when the other operand establishes a temporal domain."""
    if isinstance(peer, np.ndarray) and (
            np.ndarray.dtype.__get__(peer).kind in 'Mm' or isinstance(peer, _SelectedTemporalObservations)):
        rows = observations
        if not isinstance(rows, (np.ndarray, list, tuple)):
            # A custom projection already supplied its selected rows upstream.
            # Ordinary pandas carriers can still hold the original boxed vector.
            pd = modules.get('pandas')
            if pd is None or not isinstance(rows, (pd.Series, pd.Index, pd.DataFrame, pd.api.extensions.ExtensionArray)):
                return observations, peer
            rows = np.asarray(rows)
        if not isinstance(rows, _SelectedTemporalObservations):
            if (isinstance(rows, np.ndarray) and type(rows) is not np.ndarray
                    and np.ndarray.dtype.__get__(rows).kind == 'O'
                    and not _custom_fields(rows, np.ndarray, (
                        '__array__', '__array_interface__', '__array_struct__', '__array_function__',
                        '__array_ufunc__', '__iter__',
                        '__getitem__', '__getattribute__', '__getattr__', 'shape', 'dtype', 'flat'))):
                # A view with inherited array access exposes the same scalar
                # objects as its base ndarray.
                rows = np.asarray(rows)
            boxed = _selected_boxed_temporal_target(rows, 'observations')
            if boxed is not None:
                observations = boxed
    elif (type(peer) is np.ndarray and peer.dtype.kind == 'O'
          and (isinstance(observations, _SelectedTemporalObservations)
               or isinstance(observations, np.ndarray) and np.ndarray.dtype.__get__(observations).kind in 'Mm')):
        boxed = _selected_boxed_temporal_target(peer, 'observations')
        if boxed is not None:
            peer = boxed
    return observations, peer


def _integral_id_kind(values):
    """Distinguish plain integral operands from those requiring scalar dispatch."""
    if values.dtype.kind in 'biu':
        return 'plain'
    if values.dtype.kind == 'f':
        return 'plain' if np.isfinite(values).all() and (values == np.trunc(values)).all() else None
    if values.dtype.kind != 'O':
        return None
    kinds = set(map(id, map(type, values.flat)))
    if kinds <= _PLAIN_INTEGER_TYPES:
        return 'plain'
    if kinds <= _PLAIN_REAL_TYPES:
        return 'plain' if all(map(_is_integral_label, values.flat)) else None
    # Domain validation uses the represented number. Equality below may use
    # a different protocol for class membership.
    return 'boxed' if all(_is_integral_label(float.__float__(v) if isinstance(v, float)
                                             else _python_label(v)) for v in values.flat) else None


def _original_label_vector(source):
    """Recover pre-normalization operands for a type-sensitive comparison."""
    array = _pandas_target_array(source, include_boxed=True)
    if array is None:
        if isinstance(source, (list, tuple)):
            return _boxed_label_vector(source, column=len(source) != 0 and _label_row(source[0]))
        return np.asarray(source).reshape(-1)
    if isinstance(array, np.ndarray):
        return array.reshape(-1)
    pd = modules['pandas']
    if isinstance(array, getattr(pd.arrays, 'NumpyExtensionArray', ())) and array.dtype.kind == 'O':
        return array.to_numpy(copy=False).reshape(-1)
    if isinstance(array, pd.Categorical):
        positions, used = pd.factorize(array._codes)
        levels_array = array.categories._values
        stored = (_occupied_sparse_levels(levels_array, used) if isinstance(levels_array, pd.arrays.SparseArray)
                  else (levels_array[int(code)] if code >= 0 else np.nan for code in used))
        levels = np.fromiter(stored, dtype=object, count=len(used))
        return levels[positions]
    scalars = _boxed_sparse_scalars(array, pd, include_numeric=True)
    return scalars if isinstance(scalars, np.ndarray) else np.fromiter(scalars, dtype=object, count=len(scalars))


def _custom_comparison_types(values, labels):
    # An exact ndarray around a scalar ID is unboxed by _python_label for
    # ordinary comparisons. Its contents do not supply a custom protocol.
    types = {id(t): t for t in map(type, chain(values, labels))
             if t is not np.ndarray and id(t) not in _PLAIN_REAL_TYPES}
    readers, tagged_types = _inherited_numeric_types(types)
    return types.keys() - readers.keys() - tagged_types


def exact_label_matches(values, labels, *, value_source=None, label_source=None):
    """Compare aligned scalar IDs, or broadcast a single class representative."""
    custom = _custom_comparison_types(values, labels)
    if custom:
        # Normalizing a vector of otherwise ordinary NumPy subclasses on
        # its own can discard a type recognized by the opposite operand.
        if value_source is not None:
            values = _original_label_vector(value_source)
        if label_source is not None:
            labels = _original_label_vector(label_source)
        # A stored numeric subclass can itself define the paired comparison.
        values = list(map(_unbox_numeric_array, values))
        labels = list(map(_unbox_numeric_array, labels))
        custom = _custom_comparison_types(values, labels)
        text_types = {}
        if custom and (custom <= _TEXT_TYPES or all(
                _plain_unicode_scalar(v, text_types) for v in chain(values, labels)
                if id(type(v)) in custom)):
            # A zero-dimensional Unicode array compared with a text scalar
            # can trim the scalar's trailing NULs during dtype promotion.
            values = list(map(_unbox_unicode_scalar, values))
            labels = list(map(_unbox_unicode_scalar, labels))
    others = repeat(labels[0]) if len(labels) == 1 else iter(labels)
    # A custom comparator sees the same scalars as during fitting, including
    # the NumPy scalar stored inside an individual zero-dimensional array.
    return np.fromiter(
        (a == b if id(type(a)) in custom or id(type(b)) in custom
         else _python_label(a) == _python_label(b) for a, b in zip(values, others)),
        dtype=bool, count=len(values))


class ExactLabelClassifierMixin(ClassifierMixin):
    """ClassifierMixin accuracy with exact integer class IDs."""

    def score(self, X, y, sample_weight=None):
        """Return mean accuracy on the given test data and labels."""
        y, exact_y = _label_source_and_target(y, 'score')
        predicted, result = _label_source_and_target(self.predict(X), 'score')
        exact_y, result = _temporal_comparison_targets(exact_y, result)
        predictions = np.asarray(result)
        if isinstance(exact_y, _SparseTextObservations) and (predictions.dtype.kind in 'UST'
                or _integral_id_kind(predictions)
                or predictions.dtype.kind == 'O' and _all_sparse_text(predictions.flat, 'score')):
            left, right = column_or_1d(exact_y), column_or_1d(predictions)
            check_consistent_length(left, right)
            matches = exact_label_matches(left, right, label_source=predicted)
            return accuracy_score(np.ones_like(matches), matches, sample_weight=sample_weight)
        if predictions.dtype.kind in 'Mm' and not isinstance(exact_y, np.ndarray):
            truth = np.asarray(exact_y)
            if truth.dtype.kind in 'Mm':
                exact_y = truth
        if (isinstance(exact_y, _MixedTemporalObservations) or isinstance(result, _MixedTemporalObservations)
                or (predictions.dtype.kind in 'Mm' and isinstance(exact_y, np.ndarray)
                    and exact_y.dtype.kind in 'Mm')):
            labels = result if isinstance(result, _MixedTemporalObservations) else predictions
            matches = _temporal_label_matches(exact_y, labels, paired=True)
            return accuracy_score(np.ones_like(matches), matches, sample_weight=sample_weight)
        if predictions.dtype.kind in 'biufO':
            truth = np.asarray(exact_y)
            # Two integral IDs can lie outside any common native integer
            # dtype. Validate the metric's integer-label domain first, then
            # compare the boxed observations using their scalar semantics.
            if 'O' in (predictions.dtype.kind, truth.dtype.kind):
                truth_kind, prediction_kind = _integral_id_kind(truth), _integral_id_kind(predictions)
                if truth_kind and prediction_kind:
                    left, right = column_or_1d(truth), column_or_1d(predictions)
                    check_consistent_length(left, right)
                    if truth_kind == prediction_kind == 'plain':
                        matches = np.fromiter((int(a) == int(b) for a, b in zip(left, right)),
                                              dtype=bool, count=len(left))
                    else:
                        matches = exact_label_matches(left, right, value_source=y, label_source=predicted)
                    return accuracy_score(np.ones_like(matches), matches, sample_weight=sample_weight)
            wide = ('uint64', 'int64')
            mixed = ((predictions.dtype.name in wide and truth.dtype.kind == 'f')
                     or (truth.dtype.name in wide and predictions.dtype.kind == 'f'))
            if mixed:
                # Mixed comparisons can round either the prediction or the truth.
                # Keep sklearn's normal validation for continuous and nonfinite data.
                if truth.dtype.kind == 'f':
                    integers, floats, float_source = predictions, truth, exact_y
                else:
                    integers, floats, float_source = truth, predictions, result
                if np.isfinite(floats).all() and (floats == np.trunc(floats)).all():
                    return _accuracy_for_mixed_ids(integers, floats, float_source, sample_weight)
        # sklearn attaches dtype metadata while checking class labels. NumPy's
        # variable-width strings cannot carry that metadata. Object arrays
        # keep the original string values, including trailing NULs.
        if isinstance(exact_y, np.ndarray):
            exact_y = _comparison_labels(exact_y)
        return accuracy_score(exact_y, _comparison_labels(predictions), sample_weight=sample_weight)


def _refused_real_conversion(value):
    """A failed cast can establish no new dtype for an otherwise real scalar."""
    if not isinstance(value, (int, float)) or not _overrides_numeric_conversion(type(value)):
        return False
    cls = type(value)
    base = int if isinstance(value, int) else float
    if (getattr_static(cls, '__getattribute__') is not getattr_static(base, '__getattribute__')
            or getattr_static(cls, '__getattr__', _MISSING) is not _MISSING):
        return False
    return all(getattr_static(value, op, _MISSING) is _MISSING
               for op in ('dtype', '__array_function__', *_ARRAY_PROTOCOLS))


def contains_complex(value, *, training_labels=False):
    """Recognize complex scalars hidden in an object-dtype array too.

    Ordinary numeric arrays need only a dtype check. Object arrays require
    checking their entries before a float cast can discard imaginary parts.
    For training, reject missing temporal labels during the same scan.
    """
    if np.iscomplexobj(value):
        return True
    values = np.asarray(value)
    if values.dtype.kind != 'O':
        if training_labels and not _check_other_float_labels(values):
            _reject_nat_training_labels(values)
        return False
    pd = modules.get('pandas') if training_labels else None
    pending, visited = [values], set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        for item in current.flat:
            if type(item) is int or type(item) is float or id(type(item)) in _NUMPY_REAL_TYPES:
                continue
            if training_labels and current is values:
                if type(item) is str:
                    continue
                if _is_boxed_nat(item, pd):
                    raise ValueError('Training labels contain NaT.')
            try:
                complex_item = np.iscomplexobj(item)
            except (MemoryError, RecursionError):
                raise
            except Exception as error:
                # NumPy consults dtype and dispatch before attempting a cast.
                # Preserve failures from an explicit representation or dispatch.
                if (isinstance(error, ValueError) and issubclass(type(item), bytes)
                        and _passive_scalar(item, _TEXT_TYPES, _TEXT_FIELDS, None)):
                    # An isolated bytes subclass can select integer inference
                    # even though its containing object array selected text.
                    continue
                if not _refused_real_conversion(item):
                    raise
                continue
            if complex_item:
                return True
            if isinstance(item, np.ndarray) and item.dtype.kind == 'O':
                pending.append(item)
    return False


def require_real(value, name, *, training_labels=False):
    """Reject unsupported complex values without converting their real parts."""
    if contains_complex(value, training_labels=training_labels):
        raise ValueError(f'{name} must be real.')


def _is_boxed_nat(value, pd):
    label = value
    if issubclass(type(value), np.ndarray):
        seen = set()
        while issubclass(type(value), np.ndarray):
            array = np.asarray(value)
            if array.size != 1 or id(value) in seen:
                return False
            seen.add(id(value))
            if (array.dtype.kind in 'Mm' and issubclass(type(value), np.ma.MaskedArray)
                    and np.ma.getmaskarray(value).any()):
                return not bool(label == label)
            value = array.flat[0]
    if not _is_temporal_nat(value, pd):
        return False
    # A custom label can define a reflexive class independent of its NaT
    # storage. Check that label's own membership only for a missing payload.
    return not bool(label == label)


def _is_temporal_nat(value, pd):
    return ((pd is not None and issubclass(type(value), type(pd.NaT))) or
            (issubclass(type(value), (np.datetime64, np.timedelta64))
             and np.isnat(np.generic.__array__(value))[()]))


def _check_training_fill(fill, mode):
    if (mode in ('fit', 'normalize') and id(type(fill)) not in _PLAIN_REAL_TYPES
            and _is_boxed_nat(fill, modules.get('pandas'))):
        raise ValueError('Training labels contain NaT.')


def _reject_nat_training_labels(y):
    """Check the actual training rows after conversion and before class discovery."""
    if y.dtype.kind in 'Mm':
        missing = np.isnat(y).any()
    elif y.dtype.kind == 'O':
        pd = modules.get('pandas')
        missing = any(_is_boxed_nat(value, pd) for value in y.flat
                      if type(value) is not str and type(value) is not int)
    elif y.dtype.kind == 'T':
        if not hasattr(y.dtype, 'na_object'):
            return
        sentinel = y.dtype.na_object
        # String sentinels compare as ordinary class names. For any other
        # sentinel, NumPy returns the configured object at a missing row.
        if issubclass(type(sentinel), str) or not any(value is sentinel for value in y.flat):
            return
        missing = _is_temporal_nat(sentinel, modules.get('pandas'))
        if not missing:
            raise ValueError('Training labels contain missing values.')
    else:
        return
    if missing:
        raise ValueError('Training labels contain NaT.')


def _validate_prepared_X_y(X, y, *, estimator=None, **kwargs):
    """Validate a single classifier target and retain its original labels."""
    if isinstance(y, np.ndarray) and y.dtype.kind == 'T':
        # Older NumPy versions reject StringDType in sklearn's floating-dtype
        # query. Validate features normally and preserve the string scalars
        # and their missing-value policy for the estimator's label checks.
        if estimator is None:
            X = check_array(X, **kwargs)
        else:
            X = validate_data(estimator, X, **kwargs)
        y = column_or_1d(y, warn=True)
        check_consistent_length(X, y)
        return X, y
    if estimator is None:
        return check_X_y(X, y, **kwargs)
    return validate_data(estimator, X, y, **kwargs)


def check_real_X_y(X, y, *, target_mode='fit', **kwargs):
    """Apply sklearn data validation and reject object-wrapped complex labels."""
    prepared = exact_sparse_uint64_target(y, mode=target_mode)
    X, y = _validate_prepared_X_y(X, prepared, **kwargs)
    if y.dtype.kind == 'O' and y is not prepared:
        y = _boxed_numeric_target(y, y, mode=target_mode)
    require_real(y, 'y', training_labels=target_mode == 'fit')
    return X, y


def validate_unregularized_design(X):
    """Refuse design rank loss when the represented penalty shift is zero."""
    n, p = X.shape
    if n < p + 1:
        raise ValueError('The zero-shift augmented design has fewer rows than columns; '
                         'the MM system is singular.')
    # Own the augmented design. Scaling columns changes neither its exact rank
    # nor the model, and avoids making a rank decision from the squared system.
    features = X.toarray() if hasattr(X, 'toarray') else X
    design = np.empty((n, p + 1), dtype=np.float64)
    design[:, 0] = 1.
    design[:, 1:] = features
    del features
    if not np.isfinite(design).all():
        raise FloatingPointError('Nonfinite zero-shift augmented design.')
    scales = np.max(np.abs(design), axis=0)
    if np.any(scales == 0):
        raise ValueError('The zero-shift augmented design has a zero column; '
                         'the MM system is singular.')
    design /= scales
    singular_values = np.linalg.svd(design, compute_uv=False)
    if not np.isfinite(singular_values).all():
        raise FloatingPointError('Nonfinite zero-shift design singular values.')
    threshold = np.finfo(np.float64).eps * max(design.shape) * singular_values[0]
    if singular_values[-1] <= threshold:
        raise ValueError('The zero-shift augmented design is rank deficient or '
                         'numerically unresolved at float64 precision.')
