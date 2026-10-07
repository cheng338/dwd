"""Optional dense expanded scores with the original Python arithmetic fallback.

This is a second entry in the existing optional extension, not a replacement
for residual evaluation. It preserves the ordered high/low products and fsum
used by dense accurate scoring. No solver or acceptance policy changes here.
"""
import math
import platform
import sys
import types

import numpy as np

from . import _compiled_residual
from . import _compensated_residual as _arithmetic


_PRODUCTS = _arithmetic._products
_FSUM = _arithmetic._fsum


def _supported():
    """Limit the new entry to the validated Windows CPython/x86-64 runtime."""
    return (sys.implementation.name == 'cpython'
            and sys.platform == 'win32'
            and sys.version_info[:2] == (3, 12)
            and platform.machine().lower() in ('amd64', 'x86_64')
            and sys.maxsize > 2**32
            and sys.float_info.radix == 2 and sys.float_info.mant_dig == 53
            and sys.float_info.max_exp == 1024
            and isinstance(math.fsum, types.BuiltinFunctionType)
            and math.fsum.__module__ == 'math' and math.fsum.__name__ == 'fsum')


def _array(value, dtype, ndim, *, contiguous=True):
    return (type(value) is np.ndarray and value.dtype == dtype and value.ndim == ndim
            and value.flags.aligned and (not contiguous or value.flags.c_contiguous))


def try_expanded_dense_rows(K, alpha, indices, split_alpha, result, *, batch_enabled):
    """Return the batching state on success, or None to use the Python helper.

    The original single-row/tail route is retained. A native refusal, unsupported
    layout, missing/older extension, custom arithmetic hook or active NumPy
    call/log handler declines. Scratch is released before the caller retries.
    """
    from . import _kernel_scores as scores

    accel = _compiled_residual._ACCEL
    entry = getattr(accel, 'score_rows', None)
    hooks = (_PRODUCTS, _FSUM)
    genuine_hooks = (scores._products is _PRODUCTS is _arithmetic._products
                     and scores._fsum is _FSUM is _arithmetic._fsum
                     and all(getattr(value, '__module__', None) == _arithmetic.__name__
                             and getattr(value, '__name__', None) == name
                             and not hasattr(value, '__wrapped__')
                             for value, name in zip(hooks, ('_products', '_fsum'))))
    compatible = (entry is not None and _supported() and batch_enabled and genuine_hooks
        and not any(mode in ('call', 'log') for mode in np.geterr().values())
        and _array(K, np.dtype('float64'), 2, contiguous=False)
        and (K.flags.c_contiguous or K.flags.f_contiguous)
        and _array(alpha, np.dtype('float64'), 1)
        and _array(result, np.dtype('float64'), 1) and result.flags.writeable
        and K.shape[1] == len(alpha) and result.shape == (K.shape[0],)
        and 0 < len(alpha) <= (1 << 20)
        and isinstance(split_alpha, tuple) and len(split_alpha) == 4
        and all(_array(value, np.dtype('int32') if i == 1 else np.dtype('float64'), 1)
                and value.shape == alpha.shape for i, value in enumerate(split_alpha))
        and (_array(indices, np.dtype(np.intp), 1) or type(indices) is range))
    if compatible:
        compatible = not any(np.may_share_memory(result, value)
                             for value in (K, alpha, *split_alpha))
        if type(indices) is np.ndarray:
            compatible = compatible and not np.may_share_memory(result, indices)
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
    capacity = min(scores._SCORE_TILE_MAX_ROWS,
                   max(1, scores._SCORE_TILE_BYTES // (128 * width)))
    for start in range(0, len(indices), capacity):
        selected = indices[start:start + capacity]
        if len(selected) == 1:
            index = int(selected[0])
            result[index] = scores._expanded_row(K, alpha, False, index, _split_b=split_alpha)
            continue
        tile = output = None
        refused = False
        try:
            tile = np.ascontiguousarray(K[selected], dtype=np.float64)
            output = np.empty(len(selected), dtype=np.float64)
            status = entry(tile, alpha, *split_alpha, output)
            if status:
                refused = True
            else:
                result[selected] = output
        except (MemoryError, FloatingPointError, OverflowError):
            refused = True
        # Leave the exception scope so traceback references release scratch too.
        tile = output = None
        if refused:
            return None
    return batch_enabled
