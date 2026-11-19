"""Preserve a primary exception while reporting secondary cleanup failures.

Diagnostic records contain text only, never secondary exceptions or frames.
Formatting and annotation are best effort; they must not replace the primary.
"""

_STATE_FIELDS = (BaseException.args, BaseException.__traceback__,
                 BaseException.__cause__, BaseException.__context__,
                 BaseException.__suppress_context__)


def capture_exception_state(error):
    """Capture the primary before cleanup can raise another exception."""
    return tuple(field.__get__(error) for field in _STATE_FIELDS)


def restore_exception_state(error, state):
    # Use the base descriptors, not user-defined attribute hooks. Restore
    # suppression last: setting __cause__ also sets __suppress_context__.
    for field, value in zip(_STATE_FIELDS, state):
        field.__set__(error, value)


def reraise_preserving(error, state):
    """Re-raise the same object without replacing its original exception chain."""
    try:
        raise error
    except BaseException:
        # A plain raise error would set its context to the active cleanup
        # exception. Restore after that automatic change, as ExitStack does.
        # Python 3.11+ bare raise respects the restored traceback.
        restore_exception_state(error, state)
        raise


def _type_name(error):
    kind = type(error)
    return (type.__getattribute__(kind, '__module__') + '.'
            + type.__getattribute__(kind, '__qualname__'))


def _trim_primary(snapshot, error, primary, seen):
    """Remove primary backedges from a formatting snapshot, not the errors."""
    if snapshot is None or id(error) in seen:
        return
    seen.add(id(error))
    for name, field in (('__cause__', BaseException.__cause__),
                        ('__context__', BaseException.__context__)):
        original = field.__get__(error)
        if original is primary:
            setattr(snapshot, name, None)
        elif original is not None:
            _trim_primary(getattr(snapshot, name), original, primary, seen)
    if isinstance(error, BaseExceptionGroup) and snapshot.exceptions is not None:
        for child, original in zip(snapshot.exceptions,
                                   BaseExceptionGroup.exceptions.__get__(error)):
            if original is primary:
                # Keep the group's member count, but do not echo the primary's
                # accumulated cleanup notes or chain back into its new note.
                child.__notes__ = None
                child.__cause__ = child.__context__ = None
            else:
                _trim_primary(child, original, primary, seen)


def _format_failure(secondary, primary):
    # Failure-only import: healthy residual calls do not format tracebacks.
    import traceback
    try:
        snapshot = traceback.TracebackException.from_exception(
            secondary, capture_locals=False)
        _trim_primary(snapshot, secondary, primary, set())
        return ''.join(snapshot.format(chain=True))
    except BaseException:
        # A hostile __str__, __repr__, or __notes__ must not mask the primary.
        # Preserve the raise sites where ordinary stack formatting is possible.
        try:
            stack = ''.join(traceback.format_tb(
                BaseException.__traceback__.__get__(secondary)))
        except BaseException:
            stack = '[secondary traceback formatting unavailable]\n'
        try:
            message = str(secondary)
        except BaseException:
            message = '[exception text unavailable]'
        return stack + _type_name(secondary) + ': ' + message + '\n'


def _add_note(error, text):
    try:
        BaseException.add_note(error, text)
    except BaseException:
        # Do not overwrite malformed caller-supplied notes to repair diagnostics.
        pass


def note_cleanup_failure(primary, secondary, phase):
    """Attach stage-specific text evidence without changing the primary chain."""
    try:
        record = {'phase': phase, 'exception_type': _type_name(secondary),
                  'traceback': _format_failure(secondary, primary)}
        try:
            attributes = BaseException.__dict__['__dict__'].__get__(primary)
            previous = attributes.get('_dwd_cleanup_failures', ())
            if type(previous) is not tuple:
                previous = ()
            attributes['_dwd_cleanup_failures'] = previous + (record,)
        except BaseException:
            pass
        _add_note(primary, phase + '\nSecondary cleanup failure (original '
                  'exception retained):\n' + record['traceback'])
    except BaseException:
        # Even an annotation/import failure must not become the raised error.
        _add_note(primary, phase + '\nSecondary cleanup diagnostics unavailable.')


def note_cleanup_receipt(error, receipt, *, source='unclassified'):
    """Label a receipt's actual scope without making new cleanup claims."""
    prefix = 'Cleanup diagnostic receipt (unclassified): '
    if type(source) is str and source == 'owned_executor':
        prefix = 'Owned executor cleanup: '
    elif type(source) is str and source == 'startup_only':
        prefix = ('Optional startup-recovery diagnostic (qualification unclassified; '
                  'not an executor-context shutdown result): ')
        if type(receipt) is dict:
            try:
                qualified, drained = receipt.get('qualified'), receipt.get('drained')
                scope = receipt.get('scope')
                # The startup helper's positive result concerns its exact admitted
                # startup window. A declined check is not a failed context shutdown.
                # These labels do not qualify workers or alter either drain policy.
                if (qualified is True and drained is True and type(scope) is str
                        and scope == 'Exact CPython successful-start wait window only'):
                    prefix = 'Interrupted-start recovery: '
                elif qualified is False and drained is False:
                    prefix = ('Optional startup recovery not qualified '
                              '(not an executor-context shutdown result): ')
            except BaseException:
                # Even an exact dict can contain hostile colliding keys. Failed
                # diagnostic classification must remain neutral, not mask error.
                pass
    try:
        text = prefix + repr(receipt)
    except BaseException:
        text = prefix + '[receipt formatting unavailable]'
    _add_note(error, text)
