"""Scale-relative PSD admission of an already validated finite spectrum."""
import numpy as np
from ._compensated_residual import _gradual_underflow


def spectral_psd_status(values, n):
    """Return numerical PSD admission and its tolerance in original units.

    This is a roundoff policy, not an exact PSD certificate. Callers validate
    the vector shape and finiteness first. All positive modes are retained.
    Python scalar division keeps the decision meaningful when the original-unit
    tolerance underflows; the diagnostic tolerance can then round to zero.
    """
    scale = float(np.max(np.abs(values)))
    relative = 100 * float(np.finfo(float).eps) * n
    tolerance = relative * scale
    if tolerance < float(np.finfo(float).tiny):
        _gradual_underflow()
    if scale == 0.:
        return True, 0.
    minimum = float(np.min(values))
    return minimum / scale >= -relative, tolerance
