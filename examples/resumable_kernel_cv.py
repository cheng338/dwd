"""Fixed-grid binary kernel DWD CV with validated, resumable fold receipts.

This example leaves ``dwd.cv.run_cv`` and the estimator API unchanged. It uses
the same candidate order, fold-local preparation, equal-fold weighting and first
maximum selection, using exact accuracy counts. Only named feature kernels and accuracy scoring are in
scope. Completed receipts contain scores and diagnostics, never fitted models.
"""
from copy import deepcopy
from contextlib import nullcontext
from numbers import Integral, Real
import os
from time import perf_counter
import warnings

import numpy as np
from joblib import Parallel, delayed, parallel_config
from scipy.sparse import issparse
from sklearn.base import clone
from sklearn.metrics import check_scoring
from sklearn.model_selection import check_cv
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y
from threadpoolctl import threadpool_limits

from dwd.cv import (DoL2LoD, _validate_score, _accuracy_correct, _accuracy_means)
from dwd.gen_kern_dwd import KernGDWD

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


def _materialize_folds(cv, X, y):
    if isinstance(cv, (bool, np.bool_)):
        raise ValueError('cv must be a fold count, splitter or explicit folds.')
    splitter = check_cv(cv, y=y, classifier=True)
    if hasattr(splitter, 'random_state'):
        seed = splitter.random_state
        if seed is not None:
            _integer(seed, 'CV random_state', 0)
        elif getattr(splitter, 'shuffle', False):
            raise ValueError('A shuffled CV splitter requires an integer random_state.')
    folds = []
    for fold_index, (train, test) in enumerate(splitter.split(X, y)):
        train = _fold_indices(train, len(y), fold_index, 'training')
        test = _fold_indices(test, len(y), fold_index, 'test')
        if np.intersect1d(train, test).size:
            raise ValueError(f'Fold {fold_index} training and test rows overlap.')
        if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
            raise ValueError(f'Fold {fold_index} must contain both classes in '
                             'its training and test rows.')
        folds.append((train, test))
    if not folds:
        raise ValueError('Cross-validation must contain at least one fold.')
    return folds


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


def _run_fold(clf, X, y, train, test, candidates, fold_index, existing,
              run_dir, token, identity_hash, expected_runtime, scale):
    """A single worker owns a fold; candidates stay in their original order."""
    directory = io_path(run_dir)
    records = dict(existing)
    with worker_lease(directory, token, fold_index), threadpool_limits(limits=1):
        _check_runtime(expected_runtime)
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
        scorer = check_scoring(estimator=model, scoring='accuracy')
        for candidate_index, candidate in enumerate(candidates):
            if candidate_index in records:
                continue
            assert_generation(directory, token)
            _check_runtime(expected_runtime, check_inventory=False)
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
            _check_runtime(expected_runtime, check_inventory=False)
            assert_generation(directory, token)
            write_receipt(_receipt_path(directory, fold_index, candidate_index), record)
            records[candidate_index] = record
            scaling_seconds = 0.0
        _check_runtime(expected_runtime)
        assert_generation(directory, token)
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
                   final_runtime):
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
        'fold_workers': workers, 'native_threads_per_worker': 1,
        'current_seconds': current, 'historical_seconds': historical,
        'final_refit': {'performed': refit_best, 'fresh_model': refit_best,
                        'diagnostics': refit_diagnostics,
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
                     final_native_threads=1):
    """Run fixed-grid binary kernel DWD CV, optionally reusing completed folds.

    ``params`` follows ``dwd.cv.run_cv``: a dictionary of values or value lists.
    Duplicate candidates remain distinct. Jobs run across folds only; each
    worker uses one native thread and processes its candidate path serially.
    ``jobs=-1`` allows all logical CPUs, capped by the number of unfinished folds.
    Scaling, when requested, is fitted using each training fold only.
    ``final_native_threads`` is a positive integer (default 1) used only for
    full-data scaling and the fresh final fit. CV always uses one native thread.
    The caller's thread limits are restored before return. Changing native
    threads may change floating-point reductions; no faster fit is guaranteed.

    ``resume=True`` requires matching data, materialized folds, parameters,
    source and numerical runtime. Jobs, refit choice and final native-thread
    count may change. Each saved
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
    X, y = check_X_y(X, y, accept_sparse=False, dtype='numeric', copy=True)
    y = np.array(y, copy=True)
    if y.dtype.kind == 'O':
        if not all(isinstance(value, str) for value in y):
            raise ValueError('Object labels must contain only strings.')
        y = y.astype(str)
    if y.dtype.kind not in 'biufUS' or (y.dtype.kind in 'f' and not np.isfinite(y).all()):
        raise ValueError('Labels must be finite real numbers or strings.')
    if len(np.unique(y)) != 2:
        raise ValueError('This example requires exactly two classes.')
    X.flags.writeable = False
    y.flags.writeable = False
    folds = _materialize_folds(cv, X, y)
    clf = clone(clf)
    with threadpool_limits(limits=1):
        expected_runtime = runtime_fingerprint(check_inventory=True)
        identity = {
            'schema': _SCHEMA, 'example': 'resumable_kernel_cv',
            'estimator': stable(clf.get_params(deep=False)),
            'candidates': stable(candidates), 'X': array_fingerprint(X),
            'y': array_fingerprint(y), 'scale': scale, 'scoring': scoring,
            'accuracy_aggregation': 'exact_equal_fold_proportions_v1',
            'folds': [{'train': array_fingerprint(train), 'test': array_fingerprint(test)}
                      for train, test in folds],
            'runtime': expected_runtime, 'native_threads': 1}
        with CheckpointRun(run_dir, identity, resume=resume) as checkpoint:
            checkpoint.assert_current()
            completed = _scan_receipts(checkpoint.directory, checkpoint.identity_hash,
                                        folds, candidates)
            replayed = [set(records) for records in completed]
            missing_folds = [f for f, records in enumerate(completed)
                             if len(records) < len(candidates)]
            requested = (os.cpu_count() or 1) if jobs == -1 else jobs
            workers = min(requested, len(missing_folds))
            arguments = [
                (clf, X, y, *folds[f], candidates, f, completed[f],
                 str(checkpoint.directory), checkpoint.token, checkpoint.identity_hash,
                 expected_runtime, scale)
                for f in missing_folds]
            if workers <= 1:
                results = [_run_fold(*args) for args in arguments]
            else:
                with parallel_config(backend='loky', inner_max_num_threads=1):
                    results = Parallel(n_jobs=workers)(delayed(_run_fold)(*args)
                                                       for args in arguments)
            for fold_index, records in results:
                completed[fold_index] = records
            checkpoint.assert_current()
            _check_runtime(expected_runtime)
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
                if final_native_threads != 1:
                    expected_final_runtime = deepcopy(expected_runtime)
                    for pool in expected_final_runtime['native_pools']:
                        pool['num_threads'] = final_native_threads
                final_limits = (threadpool_limits(limits=final_native_threads)
                                if final_native_threads != 1 else nullcontext())
                with final_limits:
                    final_X = X
                    if scale:
                        scale_started = perf_counter()
                        scaler = StandardScaler()
                        final_X = scaler.fit_transform(X)
                        final_scaling_seconds = perf_counter() - scale_started
                    best_clf = clone(clf).set_params(**best_params)
                    checkpoint.assert_current()
                    if final_native_threads == 1:
                        final_runtime = _check_runtime(expected_runtime, check_inventory=False)
                    else:
                        final_runtime = _check_runtime(
                            expected_final_runtime, check_inventory=False,
                            native_threads=final_native_threads)
                    fit_started = perf_counter()
                    best_clf.fit(final_X, y)
                    refit_seconds = perf_counter() - fit_started
                    if final_native_threads != 1:
                        _check_runtime(expected_final_runtime, check_inventory=False,
                                       native_threads=final_native_threads)
                    try:
                        refit_diagnostics = _diagnostics(best_clf)
                    except Exception as error:
                        _warn_summary(error)
            checkpoint.assert_current()
            _check_runtime(expected_runtime)
            try:
                summary = _work_summary(completed, replayed, started, refit_seconds,
                                         final_scaling_seconds, refit_best, workers,
                                         refit_diagnostics, final_native_threads,
                                         final_runtime)
            except Exception as error:
                _warn_summary(error)
                summary = {
                    'schema': _SCHEMA, 'scope': 'current_invocation',
                    'coverage': 'unavailable', 'reporting_error': type(error).__name__,
                    'final_refit': {'performed': refit_best, 'fresh_model': refit_best,
                                    'diagnostics': refit_diagnostics,
                                    'native_threads_requested': final_native_threads,
                                    'runtime': final_runtime}}
            result = {'best_params': best_params, 'best_score': best_score,
                      'best_clf': best_clf, 'scaler': scaler,
                      'agg_results': aggregate, 'all_cv_results': all_cv_results,
                      'work_summary': summary}
            _publish_summaries(checkpoint.directory, checkpoint.identity_hash, result,
                                candidates)
            return result
