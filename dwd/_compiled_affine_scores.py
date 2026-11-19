"""Optional joint affine sums for final reporting, with Python row fallback.

The native entry preserves ordered high products, low products, then b. The
original Python post-sum checks still own cancellation and exact recovery.
General prediction does not opt into this reporting-only dispatch.
"""
import math

import numpy as np

from . import _compiled_residual
from . import _compensated_residual as _arithmetic
from ._compiled_scores import _array, _supported


_PRODUCTS = _arithmetic._products
_FSUM = _arithmetic._fsum
_SPLIT = _arithmetic._split_operand


def try_expanded_dense_rows(K, alpha, b, indices, split_alpha, result, exact_used,
                            *, batch_enabled, reporting=False):
    """Return batching state on success, or None for original Python rows.

    Native scratch is private, bounded and released before fallback. A block
    is published only after all of its original post-sum checks have succeeded.
    Unexpected postcheck exceptions retain their original behavior.
    """
    from . import _affine_scores as scores

    if reporting is not True:
        return None
    entry = getattr(_compiled_residual._ACCEL, 'score_affine_rows', None)
    hooks = (_PRODUCTS, _FSUM, _SPLIT)
    genuine_hooks = (
        scores._products is scores._TRUSTED_PRODUCTS is _PRODUCTS is _arithmetic._products
        and scores._fsum is scores._TRUSTED_FSUM is _FSUM is _arithmetic._fsum
        and scores._split_operand is scores._TRUSTED_SPLIT is _SPLIT is _arithmetic._split_operand
        and _PRODUCTS.__globals__.get('_split_operand') is _SPLIT
        and math.fsum is scores._TRUSTED_MATH_FSUM
        and all(value is original for value, original in zip(
            (scores._finish_row, scores._finish_score, scores._exact_row,
             scores._zero_has_exact_products, scores._round_to_nearest_now,
             scores._expanded_row), scores._NATIVE_REPORTING_HOOKS))
        and all(getattr(value, '__module__', None) == _arithmetic.__name__
                and getattr(value, '__name__', None) == name
                and not hasattr(value, '__wrapped__')
                for value, name in zip(hooks, ('_products', '_fsum', '_split_operand'))))
    compatible = (entry is not None and _supported(reporting=True)
        and batch_enabled and genuine_hooks and type(b) is float and math.isfinite(b)
        and not any(mode in ('call', 'log') for mode in np.geterr().values())
        and _array(K, np.dtype('float64'), 2, contiguous=False)
        and (K.flags.c_contiguous or K.flags.f_contiguous)
        and _array(alpha, np.dtype('float64'), 1)
        and _array(result, np.dtype('float64'), 1) and result.flags.writeable
        and _array(exact_used, np.dtype('bool'), 1) and exact_used.flags.writeable
        and K.shape[1] == len(alpha) and result.shape == exact_used.shape == (K.shape[0],)
        and 0 < len(alpha) <= (1 << 20)
        and isinstance(split_alpha, tuple) and len(split_alpha) == 4
        and all(_array(value, np.dtype('int32') if i == 1 else np.dtype('float64'), 1)
                and value.shape == alpha.shape for i, value in enumerate(split_alpha))
        and (_array(indices, np.dtype(np.intp), 1) or type(indices) is range))
    if compatible:
        compatible = not np.may_share_memory(result, exact_used)
        inputs = (K, alpha, *split_alpha)
        if type(indices) is np.ndarray:
            inputs += (indices,)
        compatible = compatible and not any(
            np.may_share_memory(destination, value)
            for destination in (result, exact_used) for value in inputs)
    if not compatible:
        return None
    if len(indices) == 0:
        return batch_enabled
    if type(indices) is range:
        good_indices = indices.step > 0 and indices[0] >= 0 and indices[-1] < len(K)
    else:
        good_indices = bool(np.all(indices >= 0) and np.all(indices < len(K)))
    if not good_indices:
        return None
    width = len(alpha)
    capacity = min(scores._TILE_MAX_ROWS, max(1, scores._TILE_BYTES // (128 * width)))
    scratch_tile = scratch_output = scratch_exact = tile = output = exact = None
    try:
        for start in range(0, len(indices), capacity):
            selected = indices[start:start + capacity]
            if len(selected) == 1:
                scratch_tile = scratch_output = scratch_exact = tile = output = exact = None
                index = int(selected[0])
                result[index], exact_used[index] = scores._expanded_row(
                    K, alpha, b, False, index, split_alpha)
                continue
            refused = False
            try:
                if scratch_tile is None:
                    rows = min(capacity, len(indices))
                    scratch_tile = np.empty((rows, width), dtype=np.float64)
                    scratch_output = np.empty(rows, dtype=np.float64)
                    scratch_exact = np.empty(rows, dtype=bool)
                tile = scratch_tile[:len(selected)]
                output = scratch_output[:len(selected)]
                exact = scratch_exact[:len(selected)]
                if type(selected) is range and selected.step == 1:
                    np.copyto(tile, K[selected.start:selected.stop])
                else:
                    for local, index in enumerate(selected):
                        np.copyto(tile[local], K[int(index)])
                refused = bool(entry(tile, alpha, *split_alpha, output, b))
            except (MemoryError, FloatingPointError, OverflowError):
                refused = True
            if not refused:
                # Keep numerical/other postcheck exceptions outside the native
                # refusal handler, as in the original Python dense helper.
                try:
                    for local, index in enumerate(selected):
                        output[local], exact[local] = scores._finish_score(
                            float(output[local]), width, K, alpha, b, False, int(index))
                    result[selected] = output
                    exact_used[selected] = exact
                except MemoryError:
                    refused = True
            tile = output = exact = None
            if refused:
                scratch_tile = scratch_output = scratch_exact = None
                return None
        return batch_enabled
    finally:
        scratch_tile = scratch_output = scratch_exact = tile = output = exact = None


# Definition-owned token for the feature-only reporting admission.
_FEATURE_REPORTING_ENTRY = (try_expanded_dense_rows, try_expanded_dense_rows.__code__,
                            try_expanded_dense_rows.__defaults__,
                            tuple(try_expanded_dense_rows.__kwdefaults__.items()))
