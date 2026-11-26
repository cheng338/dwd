"""Fixed-grid binary kernel DWD CV with validated, resumable fold receipts.

This example leaves ``dwd.cv.run_cv`` and the estimator API unchanged. It uses
the same candidate order, fold-local preparation, equal-fold weighting and first
maximum selection, using exact accuracy counts. Only named feature kernels and accuracy scoring are in
scope. Completed receipts contain scores and diagnostics, never fitted models.
"""
from concurrent.futures import Future, wait
from contextlib import ExitStack
from copy import copy, deepcopy
from decimal import Decimal
from functools import partial
from importlib import import_module
from inspect import getattr_static
from numbers import Integral, Real
import os
import signal
import sys
import threading
from time import perf_counter
import warnings

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs, parallel_config
from joblib.externals.loky import ProcessPoolExecutor
from joblib.parallel import AutoBatchingMixin, ParallelBackendBase
from joblib._parallel_backends import LokyBackend, ThreadingBackend
from scipy.sparse import issparse
from sklearn.base import clone
from sklearn.metrics import check_scoring
from sklearn.model_selection import check_cv
from sklearn.preprocessing import StandardScaler
from threadpoolctl import ThreadpoolController

from dwd._validation import (_comparison_labels, _label_source_and_target, _pandas_target_array,
                             _reject_nat_training_labels,
                             _validate_prepared_X_y)
from dwd.cv import (DoL2LoD, _validate_score, _accuracy_correct, _accuracy_means,
                    _stock_stratified_splitter)
from dwd.gen_kern_dwd import KernGDWD

try:
    from ._grid_report import safe_summarize_grid
except ImportError:
    from _grid_report import safe_summarize_grid

try:
    from ._cv_checkpoint import (
        CheckpointError, CheckpointBusyError, CheckpointRun, stable, digest,
        array_fingerprint, runtime_fingerprint, io_path, write_receipt, write_json_atomic,
        read_receipt, worker_lease, assert_generation)
except ImportError:
    from _cv_checkpoint import (
        CheckpointError, CheckpointBusyError, CheckpointRun, stable, digest,
        array_fingerprint, runtime_fingerprint, io_path, write_receipt, write_json_atomic,
        read_receipt, worker_lease, assert_generation)


_SCHEMA = 2
_KERNELS = frozenset(('linear', 'poly', 'polynomial', 'rbf', 'laplacian',
                      'sigmoid', 'cosine', 'additive_chi2', 'chi2'))
_TIMINGS = ('preparation_seconds', 'runtime', 'score_seconds', 'scaling_seconds')
_RECEIPT_KEYS = frozenset((
    'schema', 'identity_sha256', 'fold_index', 'candidate_index',
    'parameter_sha256', 'train_count', 'test_count', 'train_score',
    'test_score', 'train_correct', 'test_correct', 'diagnostics') + _TIMINGS)
_ITERATION_FIELDS = ('n_iter', 'returned_iteration', 'max_iter', 'total_iterations')
_FLAG_FIELDS = ('converged', 'criterion_reached', 'optimality_met',
                'objective_tolerance_met', 'stationarity_checked', 'budget_exhausted')
_VALUE_FIELDS = ('final_objective', 'gradient_inf_norm', 'rkhs_gradient_norm',
                 'dual_gap', 'dual_equality_residual', 'stationarity_residual')
_DIAGNOSTIC_KEYS = frozenset((
    'termination_reason', 'backend', 'solver_summary') + _ITERATION_FIELDS
    + _FLAG_FIELDS + _VALUE_FIELDS)
_SOLVER_FIELDS = {
    'summary': ('setup_seconds', 'optimization_seconds', 'total_seconds',
                'auto_fallback', 'fallback_reason', 'kernel_approximation_used',
                'positive_eigenvalues_discarded', 'symmetry_error'),
    'dual_certificate': ('dual_objective_lower_bound', 'dual_objective_is_lower_bound',
                         'primal_objective_and_gap_are_estimates', 'kernel_assumption',
                         'dual_quadratic_method', 'fallback_reason'),
    'linear_system_diagnostics': ('linear_recoveries', 'linear_solves',
                                  'compensated_residual_checks', 'native_residual_checks',
                                  'ordinary_residual_checks', 'ordinary_residual_acceptances',
                                  'refinement_steps',
                                  'added_objective_regularization',
                                  'native_refinement_trials', 'native_refinement_discarded',
                                  'native_refinement_acceptances', 'native_refinement_seconds',
                                  'compensated_residual_seconds', 'native_residual_seconds'),
    'spectral_restart': ('attempts', 'restarts', 'discarded_completed_updates',
                         'successful_updates', 'total_completed_updates',
                         'total_attempted_updates', 'max_iter_per_attempt', 'history_scope'),
    'mm_function_recovery': ('attempted', 'accepted_actions',
                             'certificate_status', 'kernel_approximation_used',
                             'positive_directions_discarded', 'true_mm_function_checked'),
}


def _integer(value, name, minimum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(name + ' must be an integer.')
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(name + ' must be at least ' + str(minimum) + '.')
    return value


def _memory_allowances(budget_mib, worker_mib):
    """Validate caller-provided allowances, not estimates of model memory."""
    if (budget_mib is None) != (worker_mib is None):
        raise ValueError('cv_memory_budget_mib and cv_worker_memory_mib must be supplied together.')
    if budget_mib is None:
        return None, None
    values = []
    for name, value in (('cv_memory_budget_mib', budget_mib),
                        ('cv_worker_memory_mib', worker_mib)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value <= 0):
            raise ValueError(name + ' must be finite and positive.')
        values.append(float(value))
    if values[0] < values[1]:
        raise ValueError('The CV memory budget must admit at least one worker allowance.')
    return tuple(values)


def _worker_plan(jobs, missing_folds, native_threads, budget_mib, worker_mib):
    automatic_cpu_budget = (os.cpu_count() or 1) if jobs == -1 else None
    requested = max(1, automatic_cpu_budget // native_threads) if jobs == -1 else jobs
    workers = min(requested, missing_folds)
    memory_cap = None
    if budget_mib is not None:
        # Exact integer ratios avoid rounding up an allowance boundary.
        from fractions import Fraction
        memory_cap = int(Fraction(budget_mib) // Fraction(worker_mib))
        workers = min(workers, memory_cap)
    return {'jobs': jobs, 'requested_workers': requested,
            'automatic_cpu_budget': automatic_cpu_budget, 'unfinished_folds': missing_folds,
            'workers': workers, 'native_threads_per_worker': native_threads,
            'nominal_compute_threads': workers * native_threads,
            'memory_budget_mib': budget_mib, 'worker_memory_mib': worker_mib,
            'memory_worker_cap': memory_cap,
            'memory_basis': 'caller_supplied_allowances' if budget_mib is not None else None}


def _simple_parameters(value):
    """Exclude stateful, executable and opaque objects before building identity."""
    if value is None or isinstance(value, (str, bool, np.bool_, Integral)):
        return
    if isinstance(value, Real):
        if not np.isfinite(value):
            raise ValueError('Estimator parameters must be finite.')
        return
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError('Parameter dictionaries require string keys.')
        for item in value.values():
            _simple_parameters(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _simple_parameters(item)
        return
    raise ValueError('Parameters must contain only deterministic scalar values, '
                     'lists, tuples and dictionaries; callables and RNG objects '
                     'are unsupported.')


def _validate_estimator(clf):
    if type(clf) is not KernGDWD:
        raise TypeError('This example supports exactly KernGDWD.')
    parameters = clf.get_params(deep=False)
    _simple_parameters(parameters)
    if clf.callback is not None:
        raise ValueError('Callbacks are unsupported in resumable CV.')
    if clf.kernel == 'precomputed':
        raise ValueError('Precomputed kernels are outside this feature-kernel example.')
    if not isinstance(clf.kernel, str) or clf.kernel not in _KERNELS:
        raise ValueError('Use a supported named feature kernel; callable kernels are unsupported.')
    if clf.stopping == 'validation':
        raise ValueError('Validation stopping requires a separate monitoring split '
                         'inside each CV training fold and is unsupported here.')
    seed = _integer(clf.random_state, 'random_state', 0)
    if seed > np.iinfo(np.uint32).max:
        raise ValueError('random_state must be at most 2**32 - 1.')


def _fold_indices(indices, n_rows, fold_index, name):
    values = np.asarray(indices)
    if (values.ndim != 1 or values.dtype.kind not in 'iu'
            or not len(values) or np.any(values < 0) or np.any(values >= n_rows)):
        raise ValueError(f'Fold {fold_index} {name} indices must be a nonempty '
                         'one-dimensional integer array within the data range.')
    result = np.array(values, dtype=np.int64, copy=True)
    if len(np.unique(result)) != len(result):
        raise ValueError(f'Fold {fold_index} {name} indices contain duplicates.')
    result.flags.writeable = False
    return result


def _materialize_folds(cv, X, y, *, class_codes=None):
    if isinstance(cv, (bool, np.bool_)):
        raise ValueError('cv must be a fold count, splitter or explicit folds.')
    classes = y if class_codes is None else class_codes
    splitter = check_cv(cv, y=classes, classifier=True)
    split_y = classes if class_codes is not None and _stock_stratified_splitter(splitter) else y
    if hasattr(splitter, 'random_state'):
        seed = splitter.random_state
        if seed is not None:
            _integer(seed, 'CV random_state', 0)
        elif getattr(splitter, 'shuffle', False):
            raise ValueError('A shuffled CV splitter requires an integer random_state.')
    folds = []
    for fold_index, (train, test) in enumerate(splitter.split(X, split_y)):
        train = _fold_indices(train, len(y), fold_index, 'training')
        test = _fold_indices(test, len(y), fold_index, 'test')
        if np.intersect1d(train, test).size:
            raise ValueError(f'Fold {fold_index} training and test rows overlap.')
        if len(np.unique(classes[train])) != 2 or len(np.unique(classes[test])) != 2:
            raise ValueError(f'Fold {fold_index} must contain both classes in '
                             'its training and test rows.')
        folds.append((train, test))
    if not folds:
        raise ValueError('Cross-validation must contain at least one fold.')
    return folds


def _string_dtype_policy(dtype):
    """Identify deterministic sentinel values without reading object addresses."""
    if not hasattr(dtype, 'na_object'):
        return [dtype.coerce, 'default']
    value = dtype.na_object
    cls = type(value)
    pd = sys.modules.get('pandas')
    if value is None or cls is object or (pd is not None and (value is pd.NA or value is pd.NaT)):
        # All plain object() tokens have identical behavior on nonmissing rows.
        data = None
    elif cls is str:
        data = value.encode('utf-8', 'surrogatepass').hex()
    elif cls in (bool, int):
        data = value
    elif cls is bytes:
        data = value.hex()
    elif cls is Decimal:
        data = str(value)
    elif cls in (float, complex):
        data = np.asarray(value).tobytes().hex()
    elif isinstance(value, np.generic) and cls is value.dtype.type and value.dtype.kind in 'biufcmMSU':
        array = np.asarray(value)
        if array.dtype.kind in 'fc' and array.real.dtype.itemsize > 8:
            # Extended precision has platform-dependent padding bytes.
            data = [array.dtype.str, [(np.format_float_scientific(v, unique=True), bool(np.signbit(v)))
                                      for v in (array.real[()], array.imag[()])]]
        else:
            data = [array.dtype.str, array.tobytes().hex()]
    else:
        raise CheckpointError('Cannot identify this custom StringDType missing-value policy for a checkpoint.')
    return [dtype.coerce, cls.__module__, cls.__qualname__, data]


def _owned_inputs(X, y):
    if type(y) is not np.ndarray:
        # Keep the exact values selected by DWD. Redensifying a pandas
        # numeric extension can round an integer before fitting and hashing.
        source, labels = _label_source_and_target(y, 'fit')
        if isinstance(labels, np.ndarray) and labels.dtype.kind in 'biuf':
            storage = _pandas_target_array(y, include_boxed=True)
            # A categorical can have object dtype while its numeric category
            # levels are exact. Plain boxed numeric arrays keep their existing rule.
            y = labels if storage is not None and not isinstance(storage, np.ndarray) else source
        else:
            y = labels if isinstance(labels, np.ndarray) and labels.dtype.kind in 'UTO' else source
    X, y = _validate_prepared_X_y(X, y, accept_sparse=False, dtype='numeric', copy=True)
    y = np.array(y, copy=True)
    if y.dtype.kind == 'O':
        if not all(isinstance(value, str) for value in y):
            raise ValueError('Object labels must contain only strings.')
        kinds = {id(type(value)): type(value) for value in y}
        if any(cls is not str for cls in kinds.values()):
            for cls in kinds.values():
                base = np.str_ if issubclass(cls, np.str_) else str
                if any(getattr_static(cls, op) is not getattr_static(base, op)
                       for op in ('__eq__', '__ne__', '__lt__', '__le__', '__gt__', '__ge__')):
                    raise CheckpointError('Cannot identify custom string comparisons for a checkpoint.')
            # __str__ may describe a value without returning its class name.
            y = np.array([str.__getitem__(value, slice(None)) for value in y], dtype=object)
        if not any(str.endswith(value, '\0') for value in y):
            # Ordinary object strings previously used this fixed-width identity.
            y = y.astype(str)
    if y.dtype.kind not in 'biufUSTO' or (y.dtype.kind == 'f' and not np.isfinite(y).all()):
        raise ValueError('Labels must be finite real numbers or strings.')
    if y.dtype.kind == 'T':
        _reject_nat_training_labels(y)
    class_codes = None
    if y.dtype.kind in 'OT':
        classes, class_codes = np.unique(_comparison_labels(y), return_inverse=True)
        class_codes = class_codes.reshape(-1)
    else:
        classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError('This example requires exactly two classes.')
    if class_codes is None:
        identity = array_fingerprint(y)
    else:
        if y.dtype.metadata is not None:
            raise CheckpointError('Unsupported array dtype metadata for checkpoint identity')
        # Pointer bytes do not identify strings. Hash the complete class names
        # and row membership without storing training labels in the manifest.
        class_codes = class_codes.astype(np.uint8)
        # JSON escapes a supplementary character and a literal surrogate
        # pair identically. Encode each class before framing it in JSON.
        names = [name.encode('utf-8', 'surrogatepass').hex() for name in classes]
        label_identity = [names, array_fingerprint(class_codes)]
        if y.dtype.kind == 'T':
            label_identity.append(_string_dtype_policy(y.dtype))
        identity = {'dtype': y.dtype.kind, 'shape': list(y.shape), 'sha256': digest(label_identity)}
    X.flags.writeable = False
    y.flags.writeable = False
    return X, y, class_codes, identity


def _accuracy(value, candidate, context, fold_index=None):
    value = _validate_score(value, candidate, context, fold_index)
    if not 0 <= value <= 1:
        raise ValueError('Accuracy must be between zero and one; '
                         f'candidate {candidate!r}, fold {fold_index}, {context}.')
    return float(value)


def _diagnostics(clf):
    result = {}
    for key in ('n_iter', 'returned_iteration', 'total_iterations'):
        value = getattr(clf, key + '_', None)
        result[key] = None if value is None else int(value)
    result['max_iter'] = int(clf.max_iter)
    result['termination_reason'] = getattr(clf, 'termination_reason_', None)
    result['backend'] = getattr(clf, 'backend_', None)
    for key in _FLAG_FIELDS[:-1]:
        value = getattr(clf, key + '_', None)
        result[key] = None if value is None else bool(value)
    reason = result['termination_reason']
    result['budget_exhausted'] = None if reason is None else reason == 'max_iter'
    for key in _VALUE_FIELDS:
        value = getattr(clf, key + '_', None)
        result[key] = (float(value) if isinstance(value, Real) and np.isfinite(value)
                       else None)
    source = getattr(clf, 'diagnostics_', {})
    result['solver_summary'] = {}
    if isinstance(source, dict):
        for section, names in _SOLVER_FIELDS.items():
            values = source if section == 'summary' else source.get(section, {})
            if not isinstance(values, dict):
                continue
            compact = {}
            for name in names:
                if name not in values:
                    continue
                value = values[name]
                if isinstance(value, np.generic):
                    value = value.item()
                if (value is None or type(value) in (str, bool, int)
                        or (type(value) is float and np.isfinite(value))):
                    compact[name] = value
            if compact:
                result['solver_summary'][section] = compact
    return result


def _fit_and_score(clf, X_train, y_train, X_test, y_test, scorer,
                   candidate, fold_index):
    """Fit once, then validate both scores before the caller can save a receipt."""
    started = perf_counter()
    clf.fit(X_train, y_train)
    fit_seconds = perf_counter() - started
    started = perf_counter()
    train_score = _accuracy(scorer(clf, X_train, y_train), candidate,
                            'train score', fold_index)
    test_score = _accuracy(scorer(clf, X_test, y_test), candidate,
                           'test score', fold_index)
    score_seconds = perf_counter() - started
    return (fit_seconds, train_score, test_score, score_seconds, _diagnostics(clf),
            _accuracy_correct(train_score, len(y_train)),
            _accuracy_correct(test_score, len(y_test)))


def _receipt_path(directory, fold_index, candidate_index):
    return (directory / 'receipts' / f'fold-{fold_index:04d}'
            / f'candidate-{candidate_index:06d}.json')


def _validate_receipt(record, identity_hash, fold_index, candidate_index,
                      candidate, train_count, test_count):
    context = f'fold {fold_index}, candidate {candidate_index}'
    if not isinstance(record, dict) or set(record) != _RECEIPT_KEYS:
        raise CheckpointError('Malformed receipt fields for ' + context)
    expected = {
        'schema': _SCHEMA, 'identity_sha256': identity_hash,
        'fold_index': fold_index, 'candidate_index': candidate_index,
        'parameter_sha256': digest(stable(candidate)),
        'train_count': train_count, 'test_count': test_count}
    for name, value in expected.items():
        if type(record[name]) is not type(value) or record[name] != value:
            raise CheckpointError('Receipt ' + name + ' mismatch for ' + context)
    for name in _TIMINGS:
        value = record[name]
        if (type(value) not in (int, float) or not np.isfinite(value) or value < 0):
            raise CheckpointError('Invalid receipt timing for ' + context)
    for name in ('train_score', 'test_score'):
        value = record[name]
        if (type(value) not in (int, float) or not np.isfinite(value)
                or not 0 <= value <= 1):
            raise CheckpointError('Invalid receipt accuracy for ' + context)
    for prefix in ('train', 'test'):
        correct, count = record[prefix + '_correct'], record[prefix + '_count']
        if (type(correct) is not int or not 0 <= correct <= count
                or correct / count != record[prefix + '_score']):
            raise CheckpointError('Receipt accuracy count mismatch for ' + context)
    diagnostic = record['diagnostics']
    if not isinstance(diagnostic, dict) or set(diagnostic) != _DIAGNOSTIC_KEYS:
        raise CheckpointError('Malformed receipt diagnostics for ' + context)
    for name in _ITERATION_FIELDS:
        value = diagnostic[name]
        if (value is not None and (type(value) is not int or value < 0)):
            raise CheckpointError('Invalid receipt iteration count for ' + context)
    if (diagnostic['termination_reason'] is not None
            and not isinstance(diagnostic['termination_reason'], str)):
        raise CheckpointError('Invalid receipt termination reason for ' + context)
    if diagnostic['backend'] is not None and not isinstance(diagnostic['backend'], str):
        raise CheckpointError('Invalid receipt backend for ' + context)
    for name in _FLAG_FIELDS:
        if diagnostic[name] is not None and type(diagnostic[name]) is not bool:
            raise CheckpointError('Invalid receipt convergence flag for ' + context)
    for name in _VALUE_FIELDS:
        value = diagnostic[name]
        if value is not None and (type(value) not in (int, float) or not np.isfinite(value)):
            raise CheckpointError('Invalid receipt numerical diagnostic for ' + context)
    solver = diagnostic['solver_summary']
    if not isinstance(solver, dict) or not set(solver) <= set(_SOLVER_FIELDS):
        raise CheckpointError('Invalid compact solver diagnostics for ' + context)
    for section, values in solver.items():
        if not isinstance(values, dict) or not set(values) <= set(_SOLVER_FIELDS[section]):
            raise CheckpointError('Invalid compact solver fields for ' + context)
        for value in values.values():
            if not (value is None or type(value) in (str, bool, int)
                    or (type(value) is float and np.isfinite(value))):
                raise CheckpointError('Nonscalar compact solver diagnostic for ' + context)


def _scan_receipts(directory, identity_hash, folds, candidates):
    """Read and validate the entire completed set before any new fit starts."""
    expected = {
        _receipt_path(directory, f, c): (f, c)
        for f in range(len(folds)) for c in range(len(candidates))}
    root = directory / 'receipts'
    if root.exists():
        for path in root.rglob('*.json'):
            if path not in expected:
                raise CheckpointError('Unexpected receipt path: ' + str(path))
    completed = [{} for _ in folds]
    for path, (fold_index, candidate_index) in expected.items():
        if path.exists():
            record = read_receipt(path, identity_hash)
            train, test = folds[fold_index]
            _validate_receipt(record, identity_hash, fold_index, candidate_index,
                              candidates[candidate_index], len(train), len(test))
            completed[fold_index][candidate_index] = record
    return completed


def _check_runtime(expected, *, check_inventory=True, native_threads=1):
    options = {'check_inventory': check_inventory}
    if native_threads != 1:
        options['expected_native_threads'] = native_threads
    observed = runtime_fingerprint(**options)
    if observed != expected:
        raise CheckpointError('Source or numerical runtime changed during the run.')
    return observed


def _initialize_fold_worker(solver_modules):
    """Load the parent's optional solver libraries before limiting and checking pools."""
    for name in solver_modules:
        import_module(name)


def _reap_fold_worker(worker):
    if worker.pid is None:
        return
    if worker.is_alive():
        if os.name == 'posix':
            try:
                os.kill(worker.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            worker.terminate()
    worker.join()


def _stop_managed_folds(processes, reason=''):
    from joblib.externals.loky.backend.utils import kill_process_tree

    while processes:
        _, worker = processes.popitem()
        try:
            kill_process_tree(worker)
        except Exception:
            # Tree discovery is best effort. Missing tools, permission errors
            # and promoted warnings must not skip reaping the owned worker.
            pass
        _reap_fold_worker(worker)


class _CheckpointExecutor(ProcessPoolExecutor):
    @property
    def _executor_manager_thread(self):
        return self._manager

    @_executor_manager_thread.setter
    def _executor_manager_thread(self, manager):
        # Loky assigns the manager before starting its thread. Customize only
        # this pool so its own queue and future cleanup can finish normally.
        if manager is not None:
            manager.kill_workers = partial(_stop_managed_folds, manager.processes)
        self._manager = manager


class _CheckpointFoldBackend(AutoBatchingMixin, ParallelBackendBase):
    """Run joblib batches on processes owned by one checkpoint search."""
    supports_inner_max_num_threads = True
    supports_retrieve_callback = True

    def __init__(self):
        super().__init__()
        self.pool = None
        self.maps = None
        self.starting = []
        self._startup = Future()
        self._stopped = Future()
        self._stop_requested = threading.Event()
        self._abort = False

    def _own_workers(self, make_pool):
        # Python signal handlers run in the main thread. Claim ownership
        # before allocating queues or starting any OS child.
        if not self._startup.set_running_or_notify_cancel():
            return
        try:
            self.pool = make_pool()
            self.pool._ensure_executor_running()
        except BaseException as error:
            self._startup.set_exception(error)
        else:
            self._startup.set_result(None)

        self._stop_requested.wait()
        try:
            self._close_pool()
        except BaseException as error:
            self._stopped.set_exception(error)
        else:
            self._stopped.set_result(None)

    def _close_pool(self):
        if self.pool is None:
            return
        try:
            manager = self.pool._executor_manager_thread
            managed = manager is not None and manager.ident is not None
            if not managed:
                for worker in self.starting:
                    _reap_fold_worker(worker)
            self.pool.shutdown(wait=managed, kill_workers=self._abort)
        finally:
            if self.maps is not None:
                self.maps._clean_temporary_resources(force=True)

    def finish_workers(self):
        self._stop_requested.set()
        self._stopped.result()

    def abort_workers(self):
        if not self._startup.cancel():
            self._abort = True
            self._stop_requested.set()
            return self._stopped.exception()

    def effective_n_jobs(self, n_jobs):
        # Retain loky's sequential fallback for nested or daemonic callers.
        return LokyBackend(nesting_level=self.nesting_level).effective_n_jobs(n_jobs)

    def configure(self, n_jobs=1, parallel=None, **options):
        count = super().configure(n_jobs=n_jobs, parallel=parallel)
        if count == 1:
            return count
        from joblib._memmapping_reducer import TemporaryResourcesManager, get_memmapping_reducers
        from joblib.externals.loky.backend.context import get_context

        def memmap_folder():
            if self.maps is None:
                self.maps = TemporaryResourcesManager(options['temp_folder'])
            return self.maps.resolve_temp_folder_name()

        forward, backward = get_memmapping_reducers(
            temp_folder_resolver=memmap_folder, max_nbytes=options['max_nbytes'],
            mmap_mode=options['mmap_mode'], unlink_on_gc_collect=False)
        # Loky starts its manager after the initial workers. Keep their handles
        # so a partial startup can be unwound.
        context = copy(get_context())
        make_process = context.Process

        def make_worker(*args, **kwargs):
            worker = make_process(*args, **kwargs)
            self.starting.append(worker)
            return worker

        context.Process = make_worker
        solvers = tuple(name for name in ('cvxpy', 'scs') if sys.modules.get(name) is not None)
        make_pool = partial(
            _CheckpointExecutor,
            max_workers=count, context=context, job_reducers=forward, result_reducers=backward,
            env=self._prepare_worker_env(count), initializer=_initialize_fold_worker,
            initargs=(solvers,))
        threading.Thread(target=self._own_workers, args=(make_pool,)).start()
        self._startup.result()
        return count

    def submit(self, func, callback=None):
        future = self.pool.submit(func)
        self.starting.clear()
        if callback is not None:
            future.add_done_callback(callback)
        return future

    def retrieve_result_callback(self, future):
        return future.result()


class _JoblibConfig(parallel_config):
    """Bind the actual override before joblib's constructor changes thread-local state."""

    def __init__(self, **settings):
        self._settings = settings

    def activate(self):
        super().__init__(**self._settings)
        return self

    def __exit__(self, *exc_info):
        # Joblib saves the original mapping before installing the new one.
        if hasattr(self, 'old_parallel_config'):
            return super().__exit__(*exc_info)


def _resource_override(previous, activate, operation, abort=None, claim=None):
    """Restore settings even if the provider's constructor or exit is interrupted."""
    entered = completed = False
    first = cleanup_error = None
    aborted = restored = False
    ambient = sys.exc_info()[1]

    def stop():
        nonlocal first, cleanup_error, aborted
        if abort is not None and not aborted:
            # Recovery retries the same saved exception if a signal arrives
            # while claiming it or before shutdown starts.
            if claim is not None:
                first = claim(first)
            cleanup_error = abort()
            aborted = True

    def run():
        nonlocal entered, completed, first
        with activate():
            try:
                entered = True
                result = operation()
                completed = True
            except BaseException as error:
                first = error
                stop()
                raise
        return result

    def recover():
        try:
            stop()
        finally:
            previous.__exit__(*sys.exc_info())

    def guarded_call():
        nonlocal first, restored
        try:
            return run()
        except BaseException as interrupted:
            if first is None and entered and not completed and interrupted.__context__ is not ambient:
                first = interrupted.__context__
            if first is None:
                first = interrupted
            recover()
            restored = True
            raise

    try:
        # Include the entire first recovery in this guard. Even a second
        # interrupt before that handler starts must restore the settings.
        return guarded_call()
    except BaseException as interrupted:
        if first is None:
            # An interrupt while accepting the first error retains it in
            # context. Exclude errors the caller was handling before entry.
            context = interrupted.__context__
            first = interrupted if context is None or context is ambient else context
        if not restored:
            # Recovery is idempotent. Retry a further interruption, while
            # allowing a persistent provider failure to reach the caller.
            try:
                recover()
            except BaseException:
                recover()
        if cleanup_error is not None and cleanup_error is not first:
            first.__cause__ = cleanup_error
        if first is interrupted:
            raise
        context = first.__context__
        try:
            raise first
        finally:
            first.__context__ = context


class _JoinedFoldThreads(ThreadingBackend):
    def terminate(self):
        # The standard multiprocessing thread-pool terminator joins its
        # handlers, but only explicitly joins process workers.
        pool = self._pool
        if pool is not None:
            super().terminate()
            pool.join()


class _SharedFoldThreads:
    """The supervisor owns joblib and its pool; the caller receives signals."""

    def __init__(self, workers):
        # Resolve caller options and nesting before switching threads.
        self.backend = _JoinedFoldThreads()
        self.parallel = Parallel(n_jobs=workers, backend=self.backend)
        self.finished = Future()
        self.stopping = threading.Event()
        self.error_lock = threading.Lock()
        self.error = None

    def fail(self, error):
        while True:
            try:
                with self.error_lock:
                    # First to publish owns the diagnostic, even if a caller
                    # signal separates this assignment from closing admission.
                    if self.error is None:
                        self.error = error
                    self.stopping.set()
                    return self.error
            except KeyboardInterrupt:
                continue

    def run(self, *args):
        if not self.stopping.is_set():
            try:
                return _run_fold(*args, stop_requested=self.stopping)
            except BaseException as error:
                # Keep the original traceback and cause. Complete the joblib
                # batch normally so its siblings can finish their receipts.
                self.fail(error)

    def tasks(self, iterator):
        # Advancing the iterator registers a checkpoint lock name. Finish an
        # admitted registration before shutdown, then stop consuming the grid.
        while not self.stopping.is_set():
            try:
                args = next(iterator)
            except StopIteration:
                return
            except BaseException as error:
                self.fail(error)
                return
            yield delayed(self.run)(*args)

    def supervise(self, arguments):
        if not self.finished.set_running_or_notify_cancel():
            return
        results = None
        try:
            results = self.parallel(self.tasks(iter(arguments)))
        except BaseException as error:
            self.fail(error)
        finally:
            try:
                self.backend.terminate()
            except BaseException as error:
                self.fail(error)
            self.finished.set_result(results)

    def dispatch(self, arguments):
        threading.Thread(target=self.supervise, args=(arguments,)).start()
        # On Windows an unbounded native wait can delay a pending SIGINT.
        while not wait((self.finished,), timeout=.05).done:
            pass
        if self.error is not None:
            context = self.error.__context__
            try:
                raise self.error
            finally:
                self.error.__context__ = context
        return self.finished.result()

    def abort(self):
        while True:
            try:
                if not self.finished.cancel():
                    self.finished.result()
                return
            except KeyboardInterrupt:
                continue


def _parallel_folds(arguments, workers, native_threads):
    backend = _CheckpointFoldBackend()
    config = _JoblibConfig(backend=backend, inner_max_num_threads=native_threads)
    threads = None

    def dispatch():
        nonlocal threads
        if config.parallel_config.get('require') == 'sharedmem' and effective_n_jobs(workers) > 1:
            threads = _SharedFoldThreads(workers)
            return threads.dispatch(arguments)
        results = Parallel(n_jobs=workers)(delayed(_run_fold)(*args) for args in arguments)
        if backend.pool is not None:
            backend.finish_workers()
        return results

    def abort():
        if threads is not None:
            return threads.abort()
        return backend.abort_workers()

    def claim(error):
        return threads.fail(error) if threads is not None else error

    return _resource_override(config, config.activate, dispatch, abort=abort, claim=claim)


class _NativeThreadSnapshot:
    def __init__(self, controller):
        self.limits = [(pool, pool.get_num_threads()) for pool in controller.lib_controllers]

    def __exit__(self, *exc_info):
        # A failing setter must not prevent later libraries from recovering.
        # Rebuild the stack on retry, keeping setter order and exception context.
        restore = ExitStack()
        for pool, count in reversed(self.limits):
            restore.callback(pool.set_num_threads, count)
        restore.__exit__(*exc_info)


def _native_thread_limit(operation, *, limits):
    # Keep distinct libraries even when they report the same prefix.
    controller = ThreadpoolController()
    return _resource_override(_NativeThreadSnapshot(controller),
                              partial(controller.limit, limits=limits), operation)


def _run_fold(clf, X, y, train, test, candidates, fold_index, existing,
              run_dir, token, identity_hash, expected_runtime, scale, cv_native_threads=1,
              *, stop_requested=None):
    """A single worker owns a fold; candidates stay in their original order."""
    directory = io_path(run_dir)
    records = dict(existing)
    guard_options = {} if cv_native_threads == 1 else {'native_threads': cv_native_threads}

    def score_candidates():
        _check_runtime(expected_runtime, **guard_options)
        assert_generation(directory, token)
        X_train, X_test = X[train, :], X[test, :]
        y_train, y_test = y[train], y[test]
        scaling_seconds = 0.0
        if scale:
            started = perf_counter()
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            scaling_seconds = perf_counter() - started
        model = clone(clf)
        # The estimator's score compares complete text values and retains
        # their dtype for fit and predict, including on older sklearn versions.
        scorer = check_scoring(estimator=model, scoring=None if y.dtype.kind in 'OT' else 'accuracy')
        for candidate_index, candidate in enumerate(candidates):
            if stop_requested is not None and stop_requested.is_set():
                return
            if candidate_index in records:
                continue
            assert_generation(directory, token)
            _check_runtime(expected_runtime, check_inventory=False, **guard_options)
            model.set_params(**candidate)
            preparation_seconds = 0.0
            if not model._cv_cache_matches(X_train):
                started = perf_counter()
                model.cv_init(X_train)
                preparation_seconds = perf_counter() - started
            (runtime, train_score, test_score, score_seconds, diagnostics,
             train_correct, test_correct) = _fit_and_score(
                model, X_train, y_train, X_test, y_test, scorer, candidate, fold_index)
            record = {
                'schema': _SCHEMA, 'identity_sha256': identity_hash,
                'fold_index': fold_index, 'candidate_index': candidate_index,
                'parameter_sha256': digest(stable(candidate)),
                'train_count': len(train), 'test_count': len(test),
                'train_score': train_score, 'test_score': test_score,
                'train_correct': train_correct, 'test_correct': test_correct,
                'runtime': runtime, 'preparation_seconds': preparation_seconds,
                'score_seconds': score_seconds, 'scaling_seconds': scaling_seconds,
                'diagnostics': diagnostics}
            _validate_receipt(record, identity_hash, fold_index, candidate_index,
                              candidate, len(train), len(test))
            _check_runtime(expected_runtime, check_inventory=False, **guard_options)
            assert_generation(directory, token)
            write_receipt(_receipt_path(directory, fold_index, candidate_index), record)
            records[candidate_index] = record
            scaling_seconds = 0.0
        _check_runtime(expected_runtime, **guard_options)
        assert_generation(directory, token)
    lease = worker_lease(directory, token, fold_index)
    _resource_override(lease, lambda: lease,
                       partial(_native_thread_limit, score_candidates, limits=cv_native_threads))
    return fold_index, records


def _aggregate(records_by_fold, candidates):
    all_cv_results = []
    for records in records_by_fold:
        all_cv_results.append({
            'params': deepcopy(candidates),
            'train_score': [records[c]['train_score'] for c in range(len(candidates))],
            'test_score': [records[c]['test_score'] for c in range(len(candidates))],
            **{name: [records[c][name] for c in range(len(candidates))]
               for name in ('train_correct', 'test_correct', 'train_count', 'test_count')},
            'runtime': [records[c]['runtime'] for c in range(len(candidates))],
            'init_time': sum(records[c]['preparation_seconds']
                             for c in range(len(candidates)))})
    aggregate = {'params': deepcopy(candidates)}
    metrics = ('test_score', 'train_score', 'runtime')
    for metric in metrics:
        aggregate['mean_' + metric] = []
        aggregate['std_' + metric] = []
    accuracy_means = {metric: _accuracy_means(all_cv_results, metric, len(candidates))
                      for metric in ('test_score', 'train_score')}
    for metric, values in accuracy_means.items():
        aggregate['mean_' + metric + '_fraction'] = [
            [value.numerator, value.denominator] for value in values]
    for candidate_index, candidate in enumerate(candidates):
        for metric in metrics:
            values = [fold[metric][candidate_index] for fold in all_cv_results]
            mean = (float(accuracy_means[metric][candidate_index]) if metric in accuracy_means
                    else np.mean(values))
            if metric in ('test_score', 'train_score'):
                _validate_score(mean, candidate, 'mean ' + metric + ' across folds')
            aggregate['mean_' + metric].append(mean)
            aggregate['std_' + metric].append(np.std(values))
    return aggregate, all_cv_results


def _work_summary(records, replayed, started, refit_seconds, final_scaling_seconds,
                   refit_best, workers, refit_diagnostics, final_native_threads,
                   final_runtime, cv_native_threads=1):
    current = {key: 0.0 for key in ('preparation', 'fit', 'score', 'scaling')}
    historical = dict(current)
    mapping = {'preparation_seconds': 'preparation', 'runtime': 'fit',
               'score_seconds': 'score', 'scaling_seconds': 'scaling'}
    new_count = old_count = 0
    for fold_index, fold in enumerate(records):
        for candidate_index, record in fold.items():
            if candidate_index in replayed[fold_index]:
                target = historical
                old_count += 1
            else:
                target = current
                new_count += 1
            for field, key in mapping.items():
                target[key] += record[field]
    current.update(final_refit=refit_seconds, final_scaling=final_scaling_seconds,
                   wall=perf_counter() - started)
    return {
        'schema': _SCHEMA, 'scope': 'current_invocation',
        'executed_evaluations': new_count, 'replayed_evaluations': old_count,
        'fold_count': len(records), 'candidate_count': len(records[0]),
        'fold_workers': workers, 'native_threads_per_worker': cv_native_threads,
        'current_seconds': current, 'historical_seconds': historical,
        'final_refit': {'performed': refit_best, 'fresh_model': refit_best,
                        'diagnostics': refit_diagnostics,
                        'estimator_workers': 1 if refit_best else 0,
                        'native_threads_requested': final_native_threads,
                        'runtime': final_runtime},
        'interpretation': 'Preparation, fit, score and scaling are summed elapsed '
            'worker intervals, not CPU time or wall time. Replayed receipt timings '
            'describe historical evaluations. Final refitting always trains a fresh '
            'model. Wall time ends at this summary and excludes subsequent summary '
            'publication, context teardown and caller data loading.'}


def _json_values(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_values(item) for item in value]
    return value


def _publish_summaries(directory, identity_hash, result, candidates):
    """Public summaries are conveniences; only validated receipts enable reuse."""
    try:
        cv_result = {
            'schema': _SCHEMA, 'identity_sha256': identity_hash,
            'best_params': _json_values(result['best_params']),
            'best_params_identity': stable(result['best_params']),
            'candidate_parameters_identity': stable(candidates),
            'best_score': float(result['best_score']),
            'agg_results': _json_values(result['agg_results']),
            'all_cv_results': _json_values(result['all_cv_results']),
            'finite_grid_report': result['finite_grid_report'],
            'final_refit': result['work_summary']['final_refit']}
    except Exception as error:
        _warn_summary(error)
        cv_result = None
    for filename, payload in (('cv-results.json', cv_result),
                              ('work-summary.json', result['work_summary'])):
        if payload is not None:
            try:
                write_json_atomic(directory / filename, payload, replace=True)
            except Exception as error:
                _warn_summary(error)


def _warn_summary(error):
    try:
        warnings.warn('Could not publish the CV summary: ' + str(error), RuntimeWarning)
    except Exception:
        pass


def run_resumable_cv(clf, X, y, params, run_dir, *, cv=5, scoring='accuracy',
                     jobs=1, resume=False, scale=False, refit_best=True,
                     final_native_threads=1, cv_native_threads=1,
                     cv_memory_budget_mib=None, cv_worker_memory_mib=None):
    """Run fixed-grid binary kernel DWD CV, optionally reusing completed folds.

    ``params`` follows ``dwd.cv.run_cv``: a dictionary of values or value lists.
    Duplicate candidates remain distinct. Jobs run across folds only; each
    worker uses ``cv_native_threads`` (default 1) and processes candidates serially.
    ``jobs=-1`` divides logical CPU count by the native-thread count, allowing
    at least one worker and capped by the number of unfinished folds.
    Scaling, when requested, is fitted using each training fold only.
    Full-data scaling and the timed final fit use one estimator worker and
    one native thread. The compatibility argument ``final_native_threads``
    accepts only 1. The optional paired memory
    allowances cap simultaneous CV workers; they are supplied by the caller,
    not measured memory limits or a guarantee against allocation failure.
    The caller's thread limits are restored before return. Changing native
    threads may change floating-point reductions; no faster fit is guaranteed.

    ``resume=True`` requires matching data, materialized folds, parameters,
    source and numerical runtime. Jobs and refit choice may change. Each saved
    evaluation is checksummed and checked semantically before new fits begin.
    An unfinished evaluation is rerun; models and partial solver state are not
    serialized. ``refit_best=True`` therefore performs fresh full-data fitting
    even when every CV evaluation was replayed. ``best_clf`` consumes scaled
    features when ``scaler`` is returned; apply that scaler before prediction.

    Termination diagnostics are observational: they never exclude a finite
    scored candidate or change the first-maximum selection rule.
    """
    started = perf_counter()
    if type(resume) is not bool or type(scale) is not bool or type(refit_best) is not bool:
        raise ValueError('resume, scale and refit_best must be booleans.')
    jobs = _integer(jobs, 'jobs')
    if jobs == 0 or jobs < -1:
        raise ValueError('jobs must be positive or -1.')
    final_native_threads = _integer(final_native_threads, 'final_native_threads', 1)
    if final_native_threads != 1:
        raise ValueError('final_native_threads must be 1 for timed final refitting.')
    cv_native_threads = _integer(cv_native_threads, 'cv_native_threads', 1)
    cv_memory_budget_mib, cv_worker_memory_mib = _memory_allowances(
        cv_memory_budget_mib, cv_worker_memory_mib)
    if scoring != 'accuracy' or not isinstance(scoring, str):
        raise ValueError('This example supports accuracy scoring only.')
    _validate_estimator(clf)
    if not isinstance(params, dict):
        raise ValueError('params must be a dictionary of values or value lists.')
    if 'random_state' in params:
        raise ValueError('Tuning random_state is unsupported; set one explicit estimator seed.')
    candidates = DoL2LoD(params)
    if not candidates:
        raise ValueError('The parameter grid must contain at least one candidate.')
    for candidate in candidates:
        _validate_estimator(clone(clf).set_params(**candidate))
    if issparse(X):
        raise ValueError('This example requires dense real feature data.')
    raw = np.asarray(X)
    if raw.dtype.kind not in 'biuf':
        raise ValueError('This example requires dense real numeric feature data.')
    X, y, class_codes, label_identity = _owned_inputs(X, y)
    folds = _materialize_folds(cv, X, y, class_codes=class_codes)
    clf = clone(clf)
    guard_options = {} if cv_native_threads == 1 else {'native_threads': cv_native_threads}
    fingerprint_options = ({} if cv_native_threads == 1 else
                           {'expected_native_threads': cv_native_threads})

    def run_search():
        expected_runtime = runtime_fingerprint(check_inventory=True, **fingerprint_options)
        identity = {
            'schema': _SCHEMA, 'example': 'resumable_kernel_cv',
            'estimator': stable(clf.get_params(deep=False)),
            'candidates': stable(candidates), 'X': array_fingerprint(X),
            'y': label_identity, 'scale': scale, 'scoring': scoring,
            'accuracy_aggregation': 'exact_equal_fold_proportions_v1',
            'folds': [{'train': array_fingerprint(train), 'test': array_fingerprint(test)}
                      for train, test in folds],
            'runtime': expected_runtime, 'native_threads': cv_native_threads}
        checkpoint = CheckpointRun(run_dir, identity, resume=resume)

        def evaluate():
            checkpoint.assert_current()
            completed = _scan_receipts(checkpoint.directory, checkpoint.identity_hash,
                                        folds, candidates)
            replayed = [set(records) for records in completed]
            missing_folds = [f for f, records in enumerate(completed)
                             if len(records) < len(candidates)]
            plan = _worker_plan(jobs, len(missing_folds), cv_native_threads,
                                cv_memory_budget_mib, cv_worker_memory_mib)
            workers = plan['workers']
            def pending_folds():
                for f in missing_folds:
                    checkpoint.register_fold(f)
                    yield (clf, X, y, *folds[f], candidates, f, completed[f],
                           str(checkpoint.directory), checkpoint.token, checkpoint.identity_hash,
                           expected_runtime, scale, cv_native_threads)

            if workers <= 1:
                results = [_run_fold(*args) for args in pending_folds()]
            else:
                results = _parallel_folds(pending_folds(), workers, cv_native_threads)
            for fold_index, records in results:
                completed[fold_index] = records
            checkpoint.assert_current()
            _check_runtime(expected_runtime, **guard_options)
            aggregate, all_cv_results = _aggregate(completed, candidates)
            exact_means = _accuracy_means(all_cv_results, 'test_score', len(candidates))
            best_index = max(range(len(candidates)), key=exact_means.__getitem__)
            best_params = deepcopy(candidates[best_index])
            best_score = aggregate['mean_test_score'][best_index]
            best_clf = scaler = None
            refit_diagnostics = None
            final_runtime = None
            refit_seconds = final_scaling_seconds = 0.0
            if refit_best:
                expected_final_runtime = expected_runtime
                if cv_native_threads != 1:
                    expected_final_runtime = deepcopy(expected_runtime)
                    for pool in expected_final_runtime['native_pools']:
                        pool['num_threads'] = 1

                def refit():
                    nonlocal scaler, best_clf, final_runtime, refit_diagnostics
                    nonlocal refit_seconds, final_scaling_seconds
                    final_X = X
                    if scale:
                        scale_started = perf_counter()
                        scaler = StandardScaler()
                        final_X = scaler.fit_transform(X)
                        final_scaling_seconds = perf_counter() - scale_started
                    best_clf = clone(clf).set_params(**best_params)
                    checkpoint.assert_current()
                    final_runtime = _check_runtime(expected_final_runtime, check_inventory=False)
                    fit_started = perf_counter()
                    best_clf.fit(final_X, y)
                    refit_seconds = perf_counter() - fit_started
                    if cv_native_threads != 1:
                        _check_runtime(expected_final_runtime, check_inventory=False)
                    try:
                        refit_diagnostics = _diagnostics(best_clf)
                    except Exception as error:
                        _warn_summary(error)

                # A named kernel can request joblib workers through kernel_kws.
                # Activate the sequential backend after recording the caller's settings.
                refit_config = _JoblibConfig(backend='sequential', n_jobs=1)
                refit_serially = partial(_resource_override, refit_config, refit_config.activate, refit)
                if cv_native_threads == 1:
                    refit_serially()
                else:
                    _native_thread_limit(refit_serially, limits=1)
            checkpoint.assert_current()
            _check_runtime(expected_runtime, **guard_options)
            try:
                summary = _work_summary(completed, replayed, started, refit_seconds,
                                         final_scaling_seconds, refit_best, workers,
                                         refit_diagnostics, final_native_threads,
                                         final_runtime, cv_native_threads=cv_native_threads)
                summary['cv_resource_plan'] = plan
            except Exception as error:
                _warn_summary(error)
                summary = {
                    'schema': _SCHEMA, 'scope': 'current_invocation',
                    'coverage': 'unavailable', 'reporting_error': type(error).__name__,
                    'native_threads_per_worker': cv_native_threads,
                    'cv_resource_plan': plan,
                    'final_refit': {'performed': refit_best, 'fresh_model': refit_best,
                                    'diagnostics': refit_diagnostics,
                                    'estimator_workers': 1 if refit_best else 0,
                                    'native_threads_requested': final_native_threads,
                                    'runtime': final_runtime}}
            result = {'best_params': best_params, 'best_score': best_score,
                      'best_clf': best_clf, 'scaler': scaler,
                      'agg_results': aggregate, 'all_cv_results': all_cv_results,
                      'work_summary': summary,
                      'finite_grid_report': safe_summarize_grid(
                          candidates, all_cv_results, best_params)}
            _publish_summaries(checkpoint.directory, checkpoint.identity_hash, result,
                                candidates)
            return result

        return _resource_override(checkpoint, lambda: checkpoint, evaluate)

    return _native_thread_limit(run_search, limits=cv_native_threads)
