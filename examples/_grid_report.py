"""Describe a completed finite accuracy grid without changing its results.

This reporting helper does not fit, rank by timing, extend a grid,
discard candidates, or alter the caller's selected model. Candidate indices
always refer to the original sequence, including repeated parameter labels.
"""

from fractions import Fraction
import math

import numpy as np


_DETAIL_LIMIT = 10
_SCOPE = 'best_among_supplied_candidates'
_MISSING = object()


class _UnknownParameter(ValueError):
    pass


def _scalar(value):
    if isinstance(value, np.generic):
        converted = value.item()
        if isinstance(converted, np.generic):
            raise _UnknownParameter('Unsupported scalar precision.')
        return converted
    return value


def _identity(value):
    """Exact, type-aware identities; never invoke an opaque object's equality."""
    if isinstance(value, np.generic):
        # Fixed parameters and selected labels retain their original scalar
        # types, just as checkpoint identity does. Numeric axis ordering is a
        # separate operation and may compare their exact numeric values.
        return ('numpy_scalar', type(value).__module__, type(value).__qualname__,
                value.dtype.str, _identity(_scalar(value)))
    if value is None:
        return ('none',)
    if type(value) in (str, bool, int):
        return (type(value).__name__, value)
    if type(value) is float and math.isfinite(value):
        return ('float', value.hex())
    if type(value) in (list, tuple):
        return (type(value).__name__, tuple(_identity(item) for item in value))
    if type(value) is dict and all(type(key) is str for key in value):
        return ('dict', tuple((key, _identity(value[key])) for key in sorted(value)))
    raise _UnknownParameter('Unsupported parameter value.')


def _json_parameter(value):
    """Copy supported labels into JSON without modifying their original objects."""
    try:
        value = _scalar(value)
    except _UnknownParameter:
        return {'status': 'unknown', 'reason': 'unsupported_scalar_precision'}
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) in (list, tuple):
        return [_json_parameter(item) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _json_parameter(value[key]) for key in sorted(value)}
    return {'status': 'unknown', 'reason': 'unsupported_parameter',
            'type': type(value).__name__}


def _flatten(parameters, prefix=()):
    if type(parameters) is not dict or not all(type(key) is str for key in parameters):
        raise ValueError('Candidate parameters require dictionaries with string keys.')
    result = {}
    for key in sorted(parameters):
        value = parameters[key]
        path = prefix + (key,)
        if type(value) is dict and value:
            result.update(_flatten(value, path))
        else:
            result[path] = value
    return result


def _relation(left, right, omitted=None):
    """Whether all other flattened parameters are exactly equal, or unknown."""
    left_keys = set(left) - {omitted}
    right_keys = set(right) - {omitted}
    if left_keys != right_keys:
        return False
    unknown = False
    for key in sorted(left_keys):
        try:
            a, b = _identity(left[key]), _identity(right[key])
        except _UnknownParameter:
            unknown = True
            continue
        if a != b:
            return False
    return None if unknown else True


def _numeric(value):
    try:
        value = _scalar(value)
    except _UnknownParameter:
        return None
    if type(value) is int:
        return Fraction(value), value
    if type(value) is float and math.isfinite(value):
        return Fraction.from_float(value), value
    return None


def _exact(value):
    return None if value is None else {'numerator': value.numerator,
                                       'denominator': value.denominator}


def _axis_report(path, selected, flattened):
    selected_value = selected.get(path, _MISSING)
    report = {
        'path': '.'.join(path), 'path_parts': list(path),
        'value': None if selected_value is _MISSING else _json_parameter(selected_value),
        'status': 'unknown', 'position': None, 'slice_candidate_count': None,
        'distinct_numeric_value_count': None, 'minimum': None, 'maximum': None,
        'missing_coordinate_candidate_count': sum(path not in item for item in flattened),
        'incomparable_candidate_count': 0, 'reason': None,
    }
    number = _numeric(selected_value)
    if selected_value is _MISSING:
        report['reason'] = 'selected_coordinate_missing'
        return report
    if number is None:
        report['reason'] = 'selected_coordinate_not_finite_numeric'
        return report
    values = []
    unknown = 0
    for candidate in flattened:
        relation = _relation(selected, candidate, omitted=path)
        if relation is False:
            continue
        neighbor = _numeric(candidate.get(path, _MISSING))
        if relation is None or neighbor is None:
            unknown += 1
        else:
            values.append(neighbor)
    report['incomparable_candidate_count'] = unknown
    if unknown:
        report['reason'] = 'conditional_slice_contains_unknown_parameters'
        return report
    if not values:
        report['reason'] = 'conditional_slice_unavailable'
        return report
    coordinates = {value[0] for value in values}
    lower, upper = min(coordinates), max(coordinates)
    chosen = number[0]
    position = ('singleton' if len(coordinates) == 1 else
                'lower' if chosen == lower else 'upper' if chosen == upper else 'interior')
    report.update(status='known', position=position, slice_candidate_count=len(values),
                  distinct_numeric_value_count=len(coordinates),
                  minimum=next(raw for exact, raw in values if exact == lower),
                  maximum=next(raw for exact, raw in values if exact == upper))
    return report


def summarize_grid(candidates, all_cv_results, best_params):
    """Return an observational JSON-compatible summary of a completed grid.

    ``all_cv_results`` uses ``dwd.cv``'s per-fold count arrays. Equal-fold
    accuracy is computed by its existing ``_accuracy_means`` helper. Missing
    counts, incomplete folds, inconsistent scores or a known selected-label
    mismatch raise an exception; use ``safe_summarize_grid`` at a reporting
    boundary. Unsupported parameter labels remain explicitly unknown.

    Numeric coordinate slices fix every other supplied parameter exactly.
    Missing coordinates are never replaced by estimator defaults. Singleton
    slices do not establish a boundary optimum or convergence. Details are
    limited to ten entries, while total counts cover all supplied candidates.
    """
    from dwd.cv import _accuracy_means

    if type(candidates) not in (list, tuple) or not candidates:
        raise ValueError('A nonempty original candidate sequence is required.')
    if type(all_cv_results) not in (list, tuple) or not all_cv_results:
        raise ValueError('Completed fold results are required.')
    flattened = [_flatten(candidate) for candidate in candidates]
    n = len(candidates)
    alignment = 'verified'
    score_validation = 'verified'
    for fold in all_cv_results:
        if type(fold) is not dict:
            raise ValueError('Each fold requires a result dictionary.')
        for field in ('test_correct', 'test_count'):
            if field not in fold or len(fold[field]) != n:
                raise ValueError('Incomplete candidate count arrays.')
        if 'params' not in fold:
            alignment = 'unknown_missing_fold_parameters'
        else:
            if len(fold['params']) != n:
                raise ValueError('Incomplete candidate parameter labels.')
            for expected, actual in zip(flattened, fold['params']):
                relation = _relation(expected, _flatten(actual))
                if relation is False:
                    raise ValueError('Fold parameter labels do not match original candidates.')
                if relation is None and alignment == 'verified':
                    alignment = 'unknown_unsupported_parameter'
        if 'test_score' in fold:
            if len(fold['test_score']) != n:
                raise ValueError('Incomplete candidate score arrays.')
        else:
            score_validation = 'unknown_missing_test_score'
    means = _accuracy_means(all_cv_results, 'test_score', n)
    for fold in all_cv_results:
        if 'test_score' not in fold:
            continue
        for score, correct, count in zip(fold['test_score'], fold['test_correct'], fold['test_count']):
            value = _scalar(score)
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or value != float(Fraction(int(correct), int(count)))):
                raise ValueError('Fold scores disagree with exact accuracy counts.')
    best_index = max(range(n), key=means.__getitem__)
    match = _relation(flattened[best_index], _flatten(best_params))
    if match is False:
        raise ValueError('Supplied best parameters do not match the first exact maximum.')
    best = means[best_index]
    tied = [i for i, score in enumerate(means) if score == best]
    ranked = sorted(means, reverse=True)
    second_gap = None if n == 1 else best - ranked[1]
    lower = [score for score in means if score < best]
    lower_gap = None if not lower else best - max(lower)
    # Keep categorical/unsupported coordinates visible as unknown instead of
    # silently omitting, for example, a gamma label such as "scale".
    paths = sorted({path for candidate in flattened for path in candidate})
    axes = [_axis_report(path, flattened[best_index], flattened) for path in paths]
    details = [{'candidate_index': i, 'params': _json_parameter(candidates[i])}
               for i in tied[:_DETAIL_LIMIT]]
    return {
        'status': 'available', 'scope': _SCOPE,
        'selection_rule': 'first_maximum_of_exact_equal_fold_accuracy',
        'candidate_identity_policy': 'original_order_and_labels_without_deduplication',
        'candidate_count': n, 'fold_count': len(all_cv_results),
        'candidate_alignment': alignment, 'score_count_validation': score_validation,
        'best_params_validation': 'verified' if match is True else 'unknown_unsupported_parameter',
        'best_index': best_index, 'best_params': _json_parameter(candidates[best_index]),
        'best_score': float(best), 'best_score_exact': _exact(best),
        'tied_best_count': len(tied), 'tied_best_indices': tied[:_DETAIL_LIMIT],
        'tied_best_details': details, 'tied_best_details_truncated': len(tied) > _DETAIL_LIMIT,
        'gap_to_second_candidate': None if second_gap is None else float(second_gap),
        'gap_to_second_candidate_exact': _exact(second_gap),
        'gap_to_next_strictly_lower': None if lower_gap is None else float(lower_gap),
        'gap_to_next_strictly_lower_exact': _exact(lower_gap),
        'all_flat': n >= 2 and len(tied) == n, 'singleton': n == 1,
        'selected_axes': axes[:_DETAIL_LIMIT], 'selected_axis_count': len(axes),
        'selected_axes_truncated': len(axes) > _DETAIL_LIMIT,
        'known_axis_count': sum(axis['status'] == 'known' for axis in axes),
        'unknown_axis_count': sum(axis['status'] == 'unknown' for axis in axes),
        'detail_limit': _DETAIL_LIMIT,
        'interpretation': 'The winner is best only among the supplied candidates. '
            'Ties, gaps and numeric slice positions describe this finite grid; '
            'they do not establish an optimum beyond it or search convergence.',
    }


def safe_summarize_grid(candidates, all_cv_results, best_params):
    """Reporting failures must not replace a valid result or prevent refitting."""
    try:
        return summarize_grid(candidates, all_cv_results, best_params)
    except Exception as error:
        return {'status': 'unavailable', 'scope': _SCOPE,
                'reporting_error': type(error).__name__}


def format_grid_report(report):
    """Return bounded readable text; formatter errors are reporting-only too."""
    try:
        if type(report) is not dict or report.get('status') != 'available':
            return 'Finite-grid report unavailable; the selected result is unchanged.'
        lines = [
            f"Best among {report['candidate_count']} supplied candidates: "
            f"candidate {report['best_index']} (zero-based), "
            f"equal-fold CV accuracy {100 * report['best_score']:.6f}%.",
            f"Exactly tied best candidates: {report['tied_best_count']}; "
            'selection retains the first exact maximum.',
        ]
        for label, key in (
            ('Gap to the second candidate', 'gap_to_second_candidate'),
            ('Gap to the next strictly lower distinct score', 'gap_to_next_strictly_lower'),
        ):
            value = report[key]
            lines.append(label + (': unavailable (no such candidate).' if value is None
                                  else f': {100 * value:.6g} percentage points.'))
        if report['all_flat']:
            lines.append('All supplied candidates have exactly the same mean CV accuracy.')
        elif report['singleton']:
            lines.append('Only one candidate was supplied; no score comparison is available.')
        if report['candidate_alignment'] != 'verified':
            lines.append('Candidate-label alignment could not be fully verified.')
        if report['best_params_validation'] != 'verified':
            lines.append('Selected-label identity could not be fully verified.')
        if report['score_count_validation'] != 'verified':
            lines.append('Accuracy counts were available, but recorded scores were incomplete.')
        for axis in report['selected_axes'][:_DETAIL_LIMIT]:
            # Escape embedded newlines/control characters in parameter labels.
            import json
            path = json.dumps(axis['path'], ensure_ascii=True)
            if len(path) > 160:
                path = path[:157] + '...'
            if axis['status'] == 'known':
                if axis['position'] == 'singleton':
                    position = 'only one distinct numeric value; no endpoint evidence'
                else:
                    position = axis['position'] + ' within its conditional slice'
                lines.append(f"Coordinate {path}: {position}; "
                             f"{axis['distinct_numeric_value_count']} distinct values, "
                             f"{axis['slice_candidate_count']} candidate labels.")
            else:
                lines.append(f'Coordinate {path}: unknown (missing, unsupported, '
                             'or incomparable parameter information).')
        if report['selected_axes_truncated']:
            lines.append(f"Coordinate details limited to {_DETAIL_LIMIT} of "
                         f"{report['selected_axis_count']} coordinates.")
        lines.append('This finite-grid summary does not establish an optimum beyond '
                     'the supplied candidates or search convergence.')
        return '\n'.join(lines)
    except Exception:
        return 'Finite-grid report unavailable; the selected result is unchanged.'
