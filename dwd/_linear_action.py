"""One opt-in computation policy; no numerical acceptance lives here."""


def validate_linear_action(action, *, implementation='optimized',
                           solver_mode='schur', backend='auto', K_eig=None):
    if not isinstance(action, str) or action not in ('factor', 'inverse'):
        raise ValueError("linear_action must be 'factor' or 'inverse'.")
    if action == 'inverse' and (implementation != 'optimized'
            or solver_mode != 'schur' or backend not in ('auto', 'cholesky')
            or (backend == 'auto' and K_eig is not None)):
        raise ValueError("linear_action='inverse' requires optimized Schur MM, "
                         "and a Cholesky route (auto with no eigensystem, or explicit cholesky).")
    return action
