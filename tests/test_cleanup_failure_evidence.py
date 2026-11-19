"""Failure-only residual cleanup controls; no workers or numerical fits run.

All branch injections use the actual dispatcher/scope/shutdown code with
finite fake owned resources. They complement, rather than replace, the existing
owned-thread subprocess tests proving real cancellation and retirement.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from dwd import _compiled_residual as compiled
from dwd import _residual_startup_cleanup as startup
from dwd import _residual_worker_scope as scopes


CHAINS = ('plain', 'explicit', 'implicit', 'suppressed')
BRANCHES = ('dispatcher_cleanup', 'dispatcher_context', 'owned_cleanup',
            'scope_unproved', 'shutdown_nonblocking', 'shutdown_unproved',
            'owned_first_interrupt')
_DESCRIPTORS = {name: BaseException.__dict__[name] for name in
                ('args', '__cause__', '__context__', '__suppress_context__', '__traceback__', '__dict__')}


def _state(error):
    return {name: descriptor.__get__(error) for name, descriptor in _DESCRIPTORS.items()
            if name != '__dict__'}


def _primary_origin(error):
    raise error


def _original_chain_origin(error):
    raise error


def _make_primary(chain='explicit', cls=KeyboardInterrupt):
    error = cls('original worker failure')
    root = OSError('original source or remote traceback cause')
    _DESCRIPTORS['__dict__'].__get__(error)['__notes__'] = ['existing primary note']
    try:
        if chain == 'plain':
            _primary_origin(error)
        else:
            try:
                _original_chain_origin(root)
            except OSError:
                if chain == 'explicit':
                    raise error from root
                if chain == 'suppressed':
                    raise error from None
                _primary_origin(error)
    except BaseException as caught:
        assert caught is error
        return error, root, _state(error)
    raise AssertionError('Primary fixture did not raise.')


def _cleanup_origin(error):
    raise error


def _context_cleanup_origin(error):
    raise error


def _notes(error):
    values = _DESCRIPTORS['__dict__'].__get__(error)
    notes = values.get('__notes__', [])
    return [value for value in notes if type(value) is str] if type(notes) is list else []


def _records(error):
    records = _DESCRIPTORS['__dict__'].__get__(error).get('_dwd_cleanup_failures', ())
    assert type(records) is tuple
    for record in records:
        assert type(record) is dict
        assert set(record) == {'phase', 'exception_type', 'traceback'}
        assert all(type(value) is str for value in record.values())
    return records


def _assert_primary(error, expected, before, *, existing_note=True):
    assert error is expected
    after = _state(error)
    assert after['args'] is before['args']
    assert after['__cause__'] is before['__cause__']
    assert after['__context__'] is before['__context__']
    assert after['__suppress_context__'] is before['__suppress_context__']
    # Propagation adds frames. The original traceback object and its unchanged
    # tail must remain reachable, rather than being replaced by a cleanup stack.
    trace = after['__traceback__']
    while trace is not None and trace is not before['__traceback__']:
        trace = trace.tb_next
    assert trace is before['__traceback__']
    if existing_note:
        assert 'existing primary note' in _notes(error)


class _FakePool:
    def __init__(self):
        self._threads = set()
        self._shutdown = False


class _Cancellation:
    def __init__(self, events):
        self.events = events
        self.published = False

    def set(self):
        self.events.append('cancel')
        self.published = True

    def is_set(self):
        return self.published


def _dispatch_injection(monkeypatch, primary, secondary, branch, events, extra=None, receipt=None):
    cancelled = _Cancellation(events)
    class Pool:
        def submit(self, *args):
            events.append('submit')
            _primary_origin(primary)
    class Executor:
        def __enter__(self):
            events.append('enter')
            return Pool()
        def __exit__(self, *exc):
            events.append('context-exit')
            assert cancelled.is_set()
            if branch == 'dispatcher_context':
                _context_cleanup_origin(secondary)
            if extra is not None:
                _context_cleanup_origin(extra)
            return False
    def cleanup(pool, error):
        events.append('cleanup')
        assert error is primary and cancelled.is_set()
        if branch == 'dispatcher_cleanup':
            _cleanup_origin(secondary)
        return {'qualified': False, 'drained': False} if receipt is None else receipt
    monkeypatch.setattr(compiled, '_ACCEL', SimpleNamespace(evaluate=lambda *args: 0))
    monkeypatch.setattr(compiled, '_worker_count', lambda n: 2)
    monkeypatch.setattr(compiled, 'Event', lambda: cancelled)
    monkeypatch.setattr(compiled, 'executor_for_call', lambda *args: Executor())
    monkeypatch.setattr(compiled, 'drain_started_before_registration', cleanup)
    compiled.compiled_values(np.eye(8), np.ones(8), np.zeros(8), .125, .3)
    raise AssertionError('Cleanup failure returned numerical output.')


def _invoke(monkeypatch, branch, primary, secondary, events):
    if branch.startswith('dispatcher_'):
        return _dispatch_injection(monkeypatch, primary, secondary, branch, events)
    pool = _FakePool()
    if branch == 'owned_cleanup':
        def fail_owned(executor, error=None):
            assert executor is pool and error is primary
            events.append('cleanup')
            _cleanup_origin(secondary)
        monkeypatch.setattr(scopes, 'drain_owned_executor', fail_owned)
        return scopes._drain_default_executor(pool, primary)
    if branch == 'scope_unproved':
        def unproved(executor, error=None):
            assert executor is pool and error is None
            events.append('unproved')
            return {'drained': False}, primary
        monkeypatch.setattr(scopes, 'drain_owned_executor', unproved)
        return scopes._drain_default_executor(pool)
    if branch == 'shutdown_unproved':
        def unproved(executor, **kwargs):
            assert executor is pool
            events.append('unproved')
            return {'drained': False}, primary
        monkeypatch.setattr(startup, 'drain_owned_executor', unproved)
        return startup._OwnedShutdown(pool)(wait=True)
    monkeypatch.setattr(startup, '_check_shutdown_binding', lambda executor: None)
    if branch == 'shutdown_nonblocking':
        def fail_shutdown(executor, **kwargs):
            assert executor is pool and kwargs['wait'] is False
            events.append('wake')
            if events.count('wake') == 1:
                _primary_origin(primary)
            _cleanup_origin(secondary)
        monkeypatch.setattr(startup, '_SHUTDOWN', fail_shutdown)
        return startup._OwnedShutdown(pool)(wait=False)
    assert branch == 'owned_first_interrupt'
    class InterruptedCancellation:
        def set(self):
            events.append('cancel')
            if events.count('cancel') == 1:
                _primary_origin(primary)
    def fail_wake(executor, **kwargs):
        assert executor is pool and kwargs['wait'] is False
        events.append('wake')
        _cleanup_origin(secondary)
    monkeypatch.setattr(startup, '_owned_shutdown', lambda executor: None)
    monkeypatch.setattr(startup, '_qualified_startup', lambda: None)
    monkeypatch.setattr(startup, '_SHUTDOWN', fail_wake)
    return startup.drain_owned_executor(pool, cancelled=InterruptedCancellation())


def test_baseline_existing_cause_is_not_replaced(monkeypatch):
    """Standalone current-source defect proof; imports no candidate helper."""
    primary, _, before = _make_primary('explicit')
    secondary, events = RuntimeError('secondary startup cleanup failure'), []
    with pytest.raises(KeyboardInterrupt) as caught:
        _invoke(monkeypatch, 'dispatcher_cleanup', primary, secondary, events)
    _assert_primary(caught.value, primary, before)
    assert events == ['enter', 'submit', 'cancel', 'cleanup', 'context-exit']


@pytest.mark.parametrize('branch', BRANCHES)
@pytest.mark.parametrize('chain', CHAINS)
def test_seven_cleanup_paths_preserve_original_chain(monkeypatch, chain, branch):
    primary, _, before = _make_primary(chain)
    secondary, events = RuntimeError('secondary cleanup sentinel'), []
    with pytest.raises(KeyboardInterrupt) as caught:
        _invoke(monkeypatch, branch, primary, secondary, events)
    _assert_primary(caught.value, primary, before)
    notes = '\n'.join(_notes(primary))
    if branch.endswith('unproved'):
        assert 'completion is unproved' in notes or 'drain not established' in notes
    else:
        assert 'RuntimeError' in notes and 'secondary cleanup sentinel' in notes
        assert '_cleanup_origin' in notes or '_context_cleanup_origin' in notes
    if branch.startswith('dispatcher_'):
        assert events.index('cancel') < events.index('cleanup') < events.index('context-exit')
    if branch == 'owned_first_interrupt':
        assert events == ['cancel', 'cancel', 'wake']


class FatalFailure(BaseException):
    pass


@pytest.mark.parametrize('cls', (RuntimeError, SystemExit, FatalFailure))
@pytest.mark.parametrize('branch', ('dispatcher_cleanup', 'dispatcher_context', 'owned_cleanup'))
def test_non_interrupt_primary_types_are_not_wrapped(monkeypatch, branch, cls):
    primary, _, before = _make_primary('explicit', cls)
    secondary = PermissionError('secondary permission failure')
    with pytest.raises(cls) as caught:
        _invoke(monkeypatch, branch, primary, secondary, [])
    _assert_primary(caught.value, primary, before)
    assert 'secondary permission failure' in '\n'.join(_notes(primary))


def test_two_cleanup_failures_keep_original_chain_and_both_tracebacks(monkeypatch):
    primary, _, before = _make_primary('explicit')
    first, second = RuntimeError('first cleanup sentinel'), OSError('second context sentinel')
    events = []
    with pytest.raises(KeyboardInterrupt) as caught:
        _dispatch_injection(monkeypatch, primary, first, 'dispatcher_cleanup', events, extra=second)
    _assert_primary(caught.value, primary, before)
    notes = '\n'.join(_notes(primary))
    assert 'first cleanup sentinel' in notes and 'second context sentinel' in notes
    assert '_cleanup_origin' in notes and '_context_cleanup_origin' in notes
    assert events.count('cleanup') == events.count('context-exit') == 1


def test_scope_failure_retains_owner_and_resets_active_scope(monkeypatch):
    primary, _, before = _make_primary('explicit')
    secondary = OSError('owned drain failed')
    pool, events = _FakePool(), []
    previous = scopes._SCOPE.get()
    def fail_owned(executor, error=None):
        assert executor is pool and error is primary
        events.append('cleanup')
        _cleanup_origin(secondary)
    monkeypatch.setattr(scopes, 'drain_owned_executor', fail_owned)
    scope = scopes.ResidualWorkerScope()
    with pytest.raises(KeyboardInterrupt) as caught:
        with scope:
            scope.executor, scope.workers = pool, 2
            _primary_origin(primary)
    _assert_primary(caught.value, primary, before)
    assert scope.closed and scope.executor is pool and scope.workers == 2
    assert scopes._SCOPE.get() is previous and events == ['cleanup']


@pytest.mark.parametrize('branch', ('scope', 'shutdown_wait', 'shutdown_nonblocking', 'owned_drain'))
def test_cleanup_failure_without_body_error_is_not_suppressed(monkeypatch, branch):
    secondary, _, before = _make_primary('explicit', RuntimeError)
    pool = _FakePool()
    def failure(*args, **kwargs):
        _cleanup_origin(secondary)
    if branch == 'scope':
        monkeypatch.setattr(scopes, 'drain_owned_executor', failure)
        call = lambda: scopes._drain_default_executor(pool)
    elif branch == 'shutdown_wait':
        monkeypatch.setattr(startup, 'drain_owned_executor', failure)
        call = lambda: startup._OwnedShutdown(pool)(wait=True)
    else:
        monkeypatch.setattr(startup, '_check_shutdown_binding', lambda executor: None)
        monkeypatch.setattr(startup, '_SHUTDOWN', failure)
        if branch == 'shutdown_nonblocking':
            call = lambda: startup._OwnedShutdown(pool)(wait=False)
        else:
            monkeypatch.setattr(startup, '_owned_shutdown', lambda executor: None)
            monkeypatch.setattr(startup, '_qualified_startup', lambda: None)
            call = lambda: startup.drain_owned_executor(pool)
    with pytest.raises(RuntimeError) as caught:
        call()
    _assert_primary(caught.value, secondary, before)


class HostileAnnotationError(RuntimeError):
    def add_note(self, note):
        raise FatalFailure('overridden add_note must not replace primary')


class HostileTextError(RuntimeError):
    def __str__(self):
        raise FatalFailure('secondary str failed')
    def __repr__(self):
        raise FatalFailure('secondary repr failed')


class HostileAttributeError(RuntimeError):
    protected = frozenset(_DESCRIPTORS) | {'__notes__', '_dwd_cleanup_failures'}
    def __getattribute__(self, name):
        if name in HostileAttributeError.protected:
            raise FatalFailure('custom exception getter must not run')
        return super().__getattribute__(name)
    def __setattr__(self, name, value):
        if name in HostileAttributeError.protected:
            raise FatalFailure('custom exception setter must not run')
        return super().__setattr__(name, value)


def test_primary_annotation_override_cannot_replace_failure(monkeypatch):
    primary, _, before = _make_primary('explicit', HostileAnnotationError)
    secondary = OSError('cleanup annotation sentinel')
    with pytest.raises(HostileAnnotationError) as caught:
        _invoke(monkeypatch, 'dispatcher_cleanup', primary, secondary, [])
    _assert_primary(caught.value, primary, before)
    assert 'cleanup annotation sentinel' in '\n'.join(_notes(primary))
    assert len(_records(primary)) == 1


def test_hostile_secondary_text_still_keeps_type_and_cleanup_stack(monkeypatch):
    primary, _, before = _make_primary('explicit')
    secondary = HostileTextError('unrenderable secondary')
    with pytest.raises(KeyboardInterrupt) as caught:
        _invoke(monkeypatch, 'dispatcher_cleanup', primary, secondary, [])
    _assert_primary(caught.value, primary, before)
    records = _records(primary)
    assert len(records) == 1 and 'HostileTextError' in records[0]['exception_type']
    assert '_cleanup_origin' in records[0]['traceback']
    assert 'HostileTextError' in '\n'.join(_notes(primary))


@pytest.mark.parametrize('kind', ('tuple', 'integer', 'dictionary'))
def test_malformed_notes_are_preserved_with_text_only_fallback(monkeypatch, kind):
    primary, _, before = _make_primary('explicit')
    malformed = {'tuple': ('preexisting invalid notes',), 'integer': 17,
                 'dictionary': {'preexisting': 'invalid notes'}}[kind]
    storage = _DESCRIPTORS['__dict__'].__get__(primary)
    storage['__notes__'] = malformed
    secondary = OSError('fallback evidence sentinel')
    with pytest.raises(KeyboardInterrupt) as caught:
        _invoke(monkeypatch, 'dispatcher_cleanup', primary, secondary, [])
    _assert_primary(caught.value, primary, before, existing_note=False)
    assert storage['__notes__'] is malformed
    records = _records(primary)
    assert len(records) == 1
    assert 'fallback evidence sentinel' in records[0]['traceback']
    assert '_cleanup_origin' in records[0]['traceback']


def test_primary_attribute_hooks_are_not_required_for_preservation(monkeypatch):
    primary, _, before = _make_primary('explicit', HostileAttributeError)
    secondary = OSError('attribute fallback sentinel')
    with pytest.raises(HostileAttributeError) as caught:
        _invoke(monkeypatch, 'dispatcher_cleanup', primary, secondary, [])
    _assert_primary(caught.value, primary, before)
    records = _records(primary)
    assert len(records) == 1 and 'attribute fallback sentinel' in records[0]['traceback']


def test_secondary_explicit_chain_is_visible_without_becoming_primary_cause(monkeypatch):
    primary, _, before = _make_primary('explicit')
    secondary, cleanup_root = RuntimeError('secondary cleanup sentinel'), LookupError('cleanup root sentinel')
    try:
        try:
            _original_chain_origin(cleanup_root)
        except LookupError:
            raise secondary from cleanup_root
    except RuntimeError as caught_secondary:
        assert caught_secondary is secondary
    with pytest.raises(KeyboardInterrupt) as caught:
        _invoke(monkeypatch, 'dispatcher_cleanup', primary, secondary, [])
    _assert_primary(caught.value, primary, before)
    text = _records(primary)[0]['traceback']
    assert 'cleanup root sentinel' in text and 'secondary cleanup sentinel' in text
    assert 'original worker failure' not in text


def test_same_secondary_in_two_stages_keeps_two_phase_records(monkeypatch):
    primary, _, before = _make_primary('explicit')
    secondary = OSError('reused secondary sentinel')
    with pytest.raises(KeyboardInterrupt) as caught:
        _dispatch_injection(monkeypatch, primary, secondary, 'dispatcher_cleanup', [], extra=secondary)
    _assert_primary(caught.value, primary, before)
    records = _records(primary)
    assert len(records) == 2 and records[0]['phase'] != records[1]['phase']
    assert all('reused secondary sentinel' in record['traceback'] for record in records)
    assert all('original worker failure' not in record['traceback'] for record in records)


def test_cleanup_reraising_primary_cannot_create_self_cause(monkeypatch):
    primary, _, before = _make_primary('explicit')
    with pytest.raises(KeyboardInterrupt) as caught:
        _invoke(monkeypatch, 'dispatcher_cleanup', primary, primary, [])
    _assert_primary(caught.value, primary, before)
    assert _state(primary)['__cause__'] is not primary


def test_successful_cleanup_receipt_annotation_cannot_become_cleanup_failure(monkeypatch):
    primary, _, before = _make_primary('explicit', HostileAnnotationError)
    receipt = {'qualified': True, 'drained': True,
               'scope': 'Exact CPython successful-start wait window only'}
    with pytest.raises(HostileAnnotationError) as caught:
        _dispatch_injection(monkeypatch, primary, None, 'dispatcher_receipt', [], receipt=receipt)
    _assert_primary(caught.value, primary, before)
    assert _records(primary) == ()
    assert any('Interrupted-start recovery:' in note and "'drained': True" in note for note in _notes(primary))


def test_secondary_traceback_formatting_does_not_mutate_original_cause_graph():
    from dwd import _cleanup_diagnostics as diagnostics
    primary, original_root, _ = _make_primary('explicit')
    secondary, cleanup_root = OSError('secondary graph sentinel'), LookupError('cleanup root stack sentinel')
    try:
        _primary_origin(primary)
    except BaseException:
        try:
            try:
                _original_chain_origin(cleanup_root)
            except LookupError:
                raise secondary from cleanup_root
        except OSError:
            errors = (primary, original_root, secondary, cleanup_root)
            states = tuple(_state(error) for error in errors)
            diagnostics.note_cleanup_failure(primary, secondary, 'graph preservation phase')
            for error, before in zip(errors, states, strict=True):
                after = _state(error)
                assert all(after[field] is before[field] for field in before)
            record = _records(primary)[0]
            assert 'OSError: secondary graph sentinel' in record['traceback']
            assert 'LookupError: cleanup root stack sentinel' in record['traceback']
            assert '_original_chain_origin' in record['traceback']
            assert 'direct cause' in record['traceback']
            assert 'original worker failure' not in record['traceback']


def test_traceback_formatting_import_is_failure_only(monkeypatch):
    import builtins
    from dwd import _cleanup_diagnostics as diagnostics
    original_import, imports = builtins.__import__, []
    def observed_import(name, *args, **kwargs):
        if name == 'traceback':
            imports.append(name)
        return original_import(name, *args, **kwargs)
    def evaluate(K, x, rhs, shift, intercept, start, stop, *outputs):
        for output in outputs:
            output.fill(0.)
        return 0
    monkeypatch.setattr(compiled, '_ACCEL', SimpleNamespace(evaluate=evaluate))
    monkeypatch.setattr(compiled, '_worker_count', lambda n: 1)
    monkeypatch.setattr(compiled, 'retire_for_serial_call', lambda: None)
    monkeypatch.setattr(builtins, '__import__', observed_import)
    values = compiled.compiled_values(np.eye(8), np.ones(8), np.zeros(8), .125, .3)
    assert isinstance(values, tuple) and len(values) == 3
    assert imports == []
    primary = RuntimeError('primary failure-only import sentinel')
    diagnostics.note_cleanup_receipt(primary, {'qualified': True, 'drained': True})
    assert imports == []
    diagnostics.note_cleanup_failure(primary, OSError('secondary formatting sentinel'), 'lazy formatting phase')
    assert imports and _records(primary)[0]['phase'] == 'lazy formatting phase'


class ShadowedDictionaryError(RuntimeError):
    @property
    def __dict__(self):
        raise FatalFailure('shadowed exception dictionary must not run')


def test_shadowed_exception_dictionary_keeps_primary_and_fallback_evidence(monkeypatch):
    primary, _, before = _make_primary('explicit', ShadowedDictionaryError)
    secondary = OSError('shadowed dictionary cleanup sentinel')
    with pytest.raises(ShadowedDictionaryError) as caught:
        _invoke(monkeypatch, 'dispatcher_cleanup', primary, secondary, [])
    _assert_primary(caught.value, primary, before)
    records = _records(primary)
    assert len(records) == 1
    assert 'shadowed dictionary cleanup sentinel' in records[0]['traceback']
    assert records[0]['exception_type'] == 'builtins.OSError'


@pytest.mark.parametrize('primary_type', (RuntimeError, KeyboardInterrupt))
def test_group_containing_primary_preserves_graph_without_recursive_note_echo(primary_type):
    from dwd import _cleanup_diagnostics as diagnostics
    primary, primary_root, _ = _make_primary('explicit', primary_type)
    other = OSError('distinct cleanup group member')
    group = BaseExceptionGroup('secondary cleanup aggregate', [primary, other])
    try:
        _primary_origin(primary)
    except BaseException:
        try:
            _cleanup_origin(group)
        except BaseException as caught:
            assert caught is group
            errors = (primary, primary_root, group, other)
            states = tuple(_state(error) for error in errors)
            members = BaseExceptionGroup.exceptions.__get__(group)
            phases = ('first group cleanup phase', 'second group cleanup phase', 'third group cleanup phase')
            for phase in phases:
                diagnostics.note_cleanup_failure(primary, group, phase)
                for error, before in zip(errors, states, strict=True):
                    after = _state(error)
                    assert all(after[field] is before[field] for field in before)
                assert BaseExceptionGroup.exceptions.__get__(group) is members
                assert members[0] is primary and members[1] is other
            records = _records(primary)
            assert tuple(record['phase'] for record in records) == phases
            texts = [record['traceback'] for record in records]
            for text in texts:
                assert 'secondary cleanup aggregate' in text
                assert 'distinct cleanup group member' in text
                assert 'original worker failure' in text
                assert 'existing primary note' not in text
                assert all(phase not in text for phase in phases)
            # Formatting the same live graph must not grow with the diagnostic
            # notes appended to its primary member after each recorded stage.
            assert texts[0] == texts[1] == texts[2]
