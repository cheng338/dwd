"""Resolve automatic residual scheduling without changing a numerical test."""


def resolve_residual_check_order(requested, *, implementation='optimized',
                                 solver_mode='schur', backend='auto', K_eig=None):
    """Return the supported concrete policy for one actual solve route.

    Explicit values preserve their historical validation. Auto is conservative
    around any supplied/cached eigenbasis, even explicit Cholesky: those routes
    keep their previous refinement-first behavior. Resolution changes no
    objective, backend, tolerance or eigensystem, and waives no acceptance
    criterion. Different checked states can still change finite-iteration fits.
    """
    if not isinstance(requested, str) or requested not in ('auto', 'refinement_first', 'adaptive'):
        raise ValueError("residual_check_order must be 'auto', 'refinement_first' or 'adaptive'.")
    compatible = (implementation == 'optimized' and solver_mode == 'schur'
                  and backend in ('auto', 'cholesky'))
    if requested == 'auto':
        return 'adaptive' if compatible and K_eig is None else 'refinement_first'
    if requested == 'adaptive' and (not compatible or (backend == 'auto' and K_eig is not None)):
        raise ValueError('Adaptive residual ordering requires optimized Cholesky MM.')
    return requested
