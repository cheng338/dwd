"""Resolve requested momentum policy without changing solver acceptance."""


def resolve_acceleration(requested, *, implementation='optimized',
                         backend='auto'):
    """Resolve by declared solver capability, never data or stopping policy.

    ``None`` is an explicit request for ordinary MM. ``auto`` selects restart
    on optimized MM, including spectral MM, and no acceleration on
    reference and L-BFGS routes. Explicit restart keeps its existing
    incompatibility errors. Other option validation belongs to the caller.
    """
    if requested is not None and (
            not isinstance(requested, str) or requested not in ('auto', 'restart')):
        raise ValueError("acceleration must be None, 'auto' or 'restart'.")
    compatible = (implementation == 'optimized'
                  and backend in ('auto', 'cholesky', 'spectral'))
    if requested == 'auto':
        return 'restart' if compatible else None
    if requested == 'restart' and not compatible:
        raise ValueError('Acceleration is supported only for optimized MM, '
                         'not reference or L-BFGS.')
    return requested
