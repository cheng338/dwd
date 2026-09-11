"""Bounded automatic retry of a specifically exhausted constrained MM solve.

No numerical update lives here. The successful attempt uses the existing
spectral solver, with the same kernel, initialization and stopping settings.
"""
from time import perf_counter
from traceback import clear_frames


class ConstrainedSolveFailure(FloatingPointError):
    """The original, centered and anchor constrained solves were exhausted."""


class MMRecoveryExhausted(FloatingPointError):
    """Constrained solving and its existing MM function recovery both failed."""

    def __init__(self, message, details):
        super().__init__(message)
        # Solver-created scalar/string diagnostics, never numerical state.
        self.details = dict(details)

    def __reduce__(self):
        # Explicit/ineligible failures may cross a joblib worker boundary.
        return type(self), (str(self), self.details)


def _release_failed_frames(error):
    """Release unwound solver frames before allocating the retry's eigenbasis."""
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        for linked in (current.__cause__, current.__context__):
            if linked is not None:
                pending.append(linked)
        if current.__traceback__ is not None:
            clear_frames(current.__traceback__)
        current.__traceback__ = None
        current.__cause__ = None
        current.__context__ = None


def solve_with_spectral_restart(solve, K, y, lambd, *, eligible=False, **options):
    """Try auto once, then explicit spectral once for a typed eligible failure.

    The caller establishes kernel-construction provenance. Callback and
    accelerated fits are excluded because replay would alter their contracts.
    Existing histories and counters describe the successful attempt; retry
    diagnostics explicitly account for work discarded from the failed one.
    """
    eligible = (eligible and options.get('backend', 'auto') == 'auto'
                and options.get('implementation', 'optimized') == 'optimized'
                and options.get('acceleration') is None
                and options.get('callback') is None
                and options.get('K_eig') is None)
    if not eligible:
        return solve(K, y, lambd, **options)
    started = perf_counter()
    try:
        return solve(K, y, lambd, **options)
    except MMRecoveryExhausted as error:
        # Copy only the solver's deliberately small diagnostic receipt, then
        # detach every linked traceback. Exiting this block also drops error.
        failed = dict(error.details)
        failed['message'] = str(error)
        failed['elapsed_seconds'] = perf_counter() - started
        _release_failed_frames(error)

    # No recursive dispatch, public refit, second initialization or new kernel.
    retry_options = dict(options, backend='spectral')
    result = solve(K, y, lambd, **retry_options)
    diagnostics = result['diagnostics']
    successful_timing = {
        name: float(diagnostics.get(name, 0.))
        for name in ('setup_seconds', 'optimization_seconds', 'total_seconds')
    }
    failed_updates = int(failed['completed_iterations'])
    failed_iteration = int(failed['failed_iteration'])
    successful_updates = int(result['n_iter'])
    diagnostics.update(
        requested_backend='auto', initial_backend='cholesky',
        attempted_backends=['cholesky', 'spectral'], effective_backend='spectral',
        auto_fallback=True, fallback_reason='exhausted_constrained_mm_spectral_restart',
        spectral_restart={
            'attempts': 2, 'restarts': 1,
            'failed_attempt': failed,
            'successful_attempt_timing': successful_timing,
            'discarded_completed_updates': failed_updates,
            'failed_attempted_iteration': failed_iteration,
            'successful_updates': successful_updates,
            'total_completed_updates': failed_updates + successful_updates,
            'total_attempted_updates': failed_iteration + successful_updates,
            'max_iter_per_attempt': int(options.get('max_iter', 100)),
            'initialization_reused': True, 'original_kernel_reused': True,
            'history_scope': 'successful_spectral_attempt',
        })
    # Keep successful setup/optimization timing identifiable while total time
    # and the combined phase figures include the abandoned first attempt.
    diagnostics['setup_seconds'] = (successful_timing['setup_seconds']
                                    + float(failed['setup_seconds']))
    diagnostics['optimization_seconds'] = (successful_timing['optimization_seconds']
                                           + float(failed['optimization_seconds']))
    diagnostics['total_seconds'] = perf_counter() - started
    return result
