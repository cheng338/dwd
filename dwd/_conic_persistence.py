"""Portable fitted CVXPY state for the package's own conic problems.

The mathematical Problem and its live solver cache are unchanged. Persistence
omits nonportable solver-cache state while retaining known data-only warm
starts. Native data-only diagnostics become
attribute-readable Python records, retaining all statistical values. Unknown
user/solver objects are left alone, including their serialization errors.
"""
from copy import copy
from dataclasses import dataclass
from types import GetSetDescriptorType, MemberDescriptorType, SimpleNamespace

import cvxpy as cp
from cvxpy import settings
import numpy as np


class PortableSolverInfo(SimpleNamespace):
    """Native information fields with ``_dwd_source_type`` provenance.

    This data record does not implement live native solver methods.
    """


@dataclass(frozen=True)
class RuntimeMethodReference:
    """Explicit description of a private extension bridge, not a statistic."""

    owner_type: str
    method_name: str


def _native_info_fields(value):
    """Recognize extension data records structurally, without solver names.

    Unknown interfaces, public methods and failing getters remain untouched.
    Python classes with slots/properties are not treated as native records.
    """
    cls = type(value)
    if (getattr(cls, '__dictoffset__', None) != 0
            or '__slots__' in vars(cls)
            or hasattr(getattr(cls, '__init__', None), '__code__')):
        return None
    fields = {}
    for base in reversed(cls.__mro__[:-1]):
        for name, descriptor in vars(base).items():
            if name.startswith('_'):
                continue
            native_property = (
                isinstance(descriptor, property)
                and descriptor.fget is not None
                and callable(descriptor.fget)
                and not hasattr(descriptor.fget, '__code__')
                and type(descriptor.fget).__module__ == 'builtins'
            )
            if not (native_property or isinstance(
                    descriptor, (GetSetDescriptorType, MemberDescriptorType))):
                return None
            try:
                fields[name] = getattr(value, name)
            except Exception:
                return None
    return fields or None


def _portable_extra_stats(value):
    """Convert only graphs containing recognized native diagnostic objects.

    Arrays are retained by reference, never scanned or copied. Dict, list,
    tuple and plain namespace aliases/cycles survive conversion. Arbitrary
    subclasses and opaque payloads retain their original objects and behavior.
    """
    memo = {}
    native_fields = {}
    converted = False

    def fields(item):
        identity = id(item)
        if identity not in native_fields:
            native_fields[identity] = _native_info_fields(item)
        return native_fields[identity]

    def bridge(item):
        # This reserved pybind11 interoperability protocol is not an arbitrary
        # user callback or a solver/version-name exception.
        name = getattr(item, '__name__', None)
        if (not isinstance(name, str) or not name.startswith('_pybind11_conduit_')
                or not name.endswith('_')):
            return None
        owner = getattr(item, '__self__', None)
        if owner is None or fields(owner) is None:
            return None
        cls = type(owner)
        return RuntimeMethodReference(cls.__module__ + '.' + cls.__qualname__, name)

    def convert(item):
        nonlocal converted
        kind = type(item)
        if (item is None or kind in (str, bytes, bool, int, float, complex)
                or isinstance(item, (np.ndarray, np.generic))):
            return item
        identity = id(item)
        if identity in memo:
            return memo[identity]
        if callable(item):
            reference = bridge(item)
            if reference is not None:
                converted = True
                memo[identity] = reference
                return reference
            return item
        if kind is dict:
            result = {}
            memo[identity] = result
            for key, child in item.items():
                result[key] = convert(child)
            return result
        if kind is list:
            result = []
            memo[identity] = result
            result.extend(convert(child) for child in item)
            return result
        if kind is tuple:
            children = [convert(child) for child in item]
            if identity in memo:  # A tuple/list cycle may have returned here.
                return memo[identity]
            result = tuple(children)
            memo[identity] = result
            return result
        if kind is SimpleNamespace:
            result = SimpleNamespace()
            memo[identity] = result
            for name, child in vars(item).items():
                setattr(result, name, convert(child))
            return result
        info = fields(item)
        if info is None:
            return item
        result = PortableSolverInfo()
        memo[identity] = result
        converted = True
        for name, child in info.items():
            setattr(result, name, convert(child))
        result._dwd_source_type = kind.__module__ + '.' + kind.__qualname__
        return result

    result = convert(value)
    return result if converted else value


def _portable_cache_entries(cache):
    """Retain proven data-only entries without testing pickle or copying data.

    Native solver objects, arbitrary container subclasses and object arrays
    are opaque. Their cache entries are rebuilt on the next solve. Builtin
    cycles and shared arrays are portable and retain their graph identity.
    """
    if type(cache) is not dict:
        return {}

    def portable(value, seen):
        kind = type(value)
        if value is None or kind in (bool, int, float, complex, str, bytes):
            return True
        if kind is np.ndarray:
            return not value.dtype.hasobject
        if isinstance(value, np.generic):
            return not value.dtype.hasobject
        identity = id(value)
        if identity in seen:
            return True
        seen.add(identity)
        if kind is dict:
            return all(portable(key, seen) and portable(child, seen)
                       for key, child in value.items())
        if kind in (list, tuple, set, frozenset):
            return all(portable(child, seen) for child in value)
        if kind is SimpleNamespace:
            return all(portable(child, seen) for child in vars(value).values())
        return False

    retained = {key: value for key, value in cache.items()
                if portable(key, set()) and portable(value, set())}
    return cache if len(retained) == len(cache) else retained


def _copy_solve_options(value):
    """Isolate builtin option containers without cloning solver/data objects.

    Solver adapters may edit nested options in place. Only exact builtin dict,
    list and tuple containers are copied; aliases and container cycles survive.
    Arrays, custom solvers, callbacks and opaque/container-subclass leaves retain
    identity. This does not claim to isolate mutations of those opaque objects.
    """
    memo = {}

    def convert(item):
        kind = type(item)
        if kind not in (dict, list, tuple):
            return item
        identity = id(item)
        if identity in memo:
            return memo[identity]
        if kind is dict:
            result = {}
            memo[identity] = result
            for key, child in item.items():
                result[convert(key)] = convert(child)
            return result
        if kind is list:
            result = []
            memo[identity] = result
            result.extend(convert(child) for child in item)
            return result
        children = [convert(child) for child in item]
        if identity in memo:  # A tuple/list cycle may have returned here.
            return memo[identity]
        result = tuple(children)
        memo[identity] = result
        return result

    return convert(value)


class PersistentProblem(cp.Problem):
    """A CVXPY Problem retaining fitted diagnostics through persistence.

    Loading preserves statistics, primal/dual values and known data-only warm
    starts. Native solver/factorization handles are rebuilt on the next solve.
    Saving never changes the original problem's live cache.
    """

    def solve(self, *args, **kwargs):
        """Solve without sharing caller-owned builtin option containers.

        CVXPY controls the method, solver selection and option interpretation.
        Copying only containers preserves custom solver and data identity.
        """
        args, kwargs = _copy_solve_options((args, kwargs))
        return super().solve(*args, **kwargs)

    def unpack_results(self, solution, chain, inverse_data):
        super().unpack_results(solution, chain, inverse_data)
        stats = self.solver_stats
        solved = self.solution
        if stats is None or solved is None or not isinstance(solved.attr, dict):
            return
        extra = stats.extra_stats
        portable = _portable_extra_stats(extra)
        if portable is extra:
            return
        stats_copy = copy(stats)
        solution_copy = copy(solved)
        solution_copy.attr = solved.attr.copy()
        stats_copy.extra_stats = portable
        if solution_copy.attr.get(settings.EXTRA_STATS) is extra:
            solution_copy.attr[settings.EXTRA_STATS] = portable
        self._solver_stats = stats_copy
        self._solution = solution_copy

    def __getstate__(self):
        state = super().__getstate__()
        if isinstance(state, dict):
            state = state.copy()
            if '_solver_cache' in state:
                state['_solver_cache'] = _portable_cache_entries(state['_solver_cache'])
            return state
        # Preserve user-subclass slots and future base serialization contracts.
        if (isinstance(state, tuple) and len(state) == 2
                and isinstance(state[0], dict)):
            dictionary = state[0].copy()
            if '_solver_cache' in dictionary:
                dictionary['_solver_cache'] = _portable_cache_entries(dictionary['_solver_cache'])
            return dictionary, state[1]
        return state
