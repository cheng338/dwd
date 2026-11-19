"""Accurate cleanup applicability without changing executor or error semantics."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import gc
import json
import threading
from types import SimpleNamespace
import weakref

import numpy as np
import psutil
import pytest

from dwd import _cleanup_diagnostics as diagnostics
from dwd import _compiled_residual as compiled
from dwd import _residual_startup_cleanup as startup
from dwd import _residual_worker_scope as scopes


_STARTUP_SCOPE = 'Exact CPython successful-start wait window only'


def _attributes(error):
    return BaseException.__dict__['__dict__'].__get__(error)


def _notes(error):
    notes = _attributes(error).get('__notes__', [])
    return list(notes) if type(notes) is list else []


def _chain(error):
    return (BaseException.__cause__.__get__(error),
            BaseException.__context__.__get__(error),
            BaseException.__suppress_context__.__get__(error))


def _primary(kind='plain'):
    if kind == 'plain':
        error = RuntimeError('ordinary future error')
    else:
        try:
            raise ValueError('original lower-level cause')
        except ValueError as cause:
            try:
                if kind == 'explicit':
                    raise RuntimeError('ordinary future error') from cause
                if kind == 'suppressed':
                    raise RuntimeError('ordinary future error') from None
                raise RuntimeError('ordinary future error')
            except RuntimeError as caught:
                error = caught
    error.add_note('preexisting caller note')
    return error


def _failure_origin(error):
    raise error


def _real_pool_error(monkeypatch, route, trigger, primary, record_property):
    """Run real Future/thread/context paths; arithmetic is a bounded stub."""
    if route != 'custom' and startup._qualified_startup() is None:
        pytest.skip('Owned completion tokens unavailable on this interpreter')
    process = psutil.Process()
    process_identity = (process.pid, process.create_time())
    state = {'active': False}
    pools, events, calls, context_events = [], [], [], []
    barrier = threading.Barrier(2)
    entered = threading.Event()
    original_chain = _chain(primary)
    original_args = primary.args

    def event_factory():
        event = threading.Event()
        events.append(event)
        return event

    def evaluate(K, x, rhs, shift, intercept, first, last,
                 scores, residual, maxima):
        calls.append((state['active'], first, last))
        scores.fill(0.0)
        residual.fill(0.0)
        maxima.fill(0.0)
        if not state['active']:
            barrier.wait(timeout=5)
            return 0
        entered.set()
        if trigger == 'future':
            barrier.wait(timeout=5)
            if first == 0:
                _failure_origin(primary)
        assert events[-1].wait(timeout=5), 'Dispatcher did not publish cancellation'
        return 0

    def instrument_pool(pool):
        pools.append(pool)
        original_submit = pool.submit
        count = 0

        def submit(*args, **kwargs):
            nonlocal count
            if state['active']:
                count += 1
                if trigger == 'submit' and count == 2:
                    assert entered.wait(timeout=5), 'First worker never entered'
                    _failure_origin(primary)
            return original_submit(*args, **kwargs)

        monkeypatch.setattr(pool, 'submit', submit)
        return pool

    original_own = scopes.own_default_executor

    def owned(pool):
        return instrument_pool(original_own(pool))

    class CustomPool(ThreadPoolExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instrument_pool(self)

        def __enter__(self):
            context_events.append('enter')
            return super().__enter__()

        def __exit__(self, *exc):
            context_events.append('exit-before-drain')
            assert events[-1].is_set()
            result = super().__exit__(*exc)
            context_events.append('exit-after-drain')
            return result

    owner = SimpleNamespace(evaluate=evaluate)
    monkeypatch.setattr(compiled, '_ACCEL', owner)
    monkeypatch.setattr(compiled, '_TRUSTED_ACCEL', owner)
    monkeypatch.setattr(compiled, '_TRUSTED_EVALUATE', evaluate)
    monkeypatch.setattr(compiled, '_worker_count', lambda n: 2)
    monkeypatch.setattr(compiled, 'Event', event_factory)
    monkeypatch.setattr(scopes, 'own_default_executor', owned)
    if route == 'custom':
        monkeypatch.setattr(compiled, 'ThreadPoolExecutor', CustomPool)
    if trigger == 'poll':
        def interrupted_wait(*args, **kwargs):
            assert entered.wait(timeout=5), 'Worker did not enter before polling'
            _failure_origin(primary)
        original_wait = compiled.wait

        def selected_wait(*args, **kwargs):
            return (interrupted_wait(*args, **kwargs) if state['active']
                    else original_wait(*args, **kwargs))
        monkeypatch.setattr(compiled, 'wait', selected_wait)

    args = (np.eye(8), np.ones(8), np.zeros(8), .125, .3)
    scope = scopes.ResidualWorkerScope() if route == 'reused' else None
    with scope if scope is not None else nullcontext():
        if scope is not None:
            assert compiled.compiled_values(*args) is not None
            assert scope.executor is pools[0]
        state['active'] = True
        with pytest.raises(type(primary)) as caught:
            compiled.compiled_values(*args)
        assert caught.value is primary
        assert primary.args == original_args and _chain(primary) == original_chain
        assert events[-1].is_set()
        assert scope is None or scope.executor is None
    assert len(pools) == 1
    pool = pools[0]
    assert pool._shutdown and pool._threads
    assert all(not thread.is_alive() for thread in pool._threads)
    assert scopes._SCOPE.get() is None
    notes = _notes(primary)
    assert notes[0] == 'preexisting caller note'
    assert not any(note.startswith('Interrupted-start recovery:') for note in notes)
    if route == 'custom':
        assert context_events == ['enter', 'exit-before-drain', 'exit-after-drain']
        assert any('not qualified' in note and 'context' in note.lower() for note in notes)
    else:
        assert any(note.startswith('Owned executor cleanup:') for note in notes)
    assert len(notes) == 2
    assert '_dwd_cleanup_failures' not in _attributes(primary)
    record_property('cleanup_diagnostic_evidence', json.dumps({
        'route': route, 'trigger': trigger, 'same_primary': caught.value is primary,
        'original_chain_preserved': _chain(primary) == original_chain,
        'cancelled': events[-1].is_set(), 'pool_retired': True,
        'registered_workers_completed': len(pool._threads),
        'calls': calls, 'context_events': context_events, 'notes': notes,
        'process_pid': process_identity[0], 'process_creation_time': process_identity[1]
    }, sort_keys=True))


def test_ordinary_future_error_is_not_interrupted_start_recovery(monkeypatch, record_property):
    """Baseline reproduction does not require a new helper signature."""
    _real_pool_error(monkeypatch, 'per-call', 'future', _primary(), record_property)


@pytest.mark.parametrize('route', ['per-call', 'reused', 'custom'])
@pytest.mark.parametrize('trigger', ['future', 'submit', 'poll'])
def test_real_pool_errors_keep_cleanup_contract_and_accurate_label(
        monkeypatch, route, trigger, record_property):
    _real_pool_error(monkeypatch, route, trigger, _primary(), record_property)


@pytest.mark.parametrize('kind', ['plain', 'explicit', 'implicit', 'suppressed'])
def test_real_future_error_preserves_existing_exception_chain(monkeypatch, kind, record_property):
    _real_pool_error(monkeypatch, 'per-call', 'future', _primary(kind), record_property)


@pytest.mark.parametrize('trigger,error_type', [
    ('poll', KeyboardInterrupt), ('future', KeyboardInterrupt), ('future', SystemExit)
], ids=['poll-interrupt', 'worker-interrupt', 'worker-system-exit'])
def test_base_exceptions_do_not_imply_startup_failure(monkeypatch, trigger, error_type, record_property):
    primary = error_type('ordinary task or poll interruption')
    primary.add_note('preexisting caller note')
    _real_pool_error(monkeypatch, 'per-call', trigger, primary, record_property)


def _assert_note_preserves_primary(receipt, source, expected_prefix):
    error = _primary('explicit')
    chain, args, traceback = _chain(error), error.args, error.__traceback__
    diagnostics.note_cleanup_receipt(error, receipt, source=source)
    assert _chain(error) == chain and error.args == args
    assert error.__traceback__ is traceback
    notes = _notes(error)
    assert notes[0] == 'preexisting caller note' and len(notes) == 2
    assert notes[1].startswith(expected_prefix)
    assert '_dwd_cleanup_failures' not in _attributes(error)
    return error, notes[1]


@pytest.mark.parametrize('qualified,drained,reason', [
    (True, True, None),
    (False, False, 'Completion capability or owned-worker readiness is unavailable'),
    (False, False, 'Possible omitted worker completion is unproved')
])
def test_owned_cleanup_label_keeps_success_and_unproved_receipts_distinct(
        qualified, drained, reason):
    receipt = {'qualified': qualified, 'drained': drained,
               'shutdown_requested': True, 'reason': reason,
               'scope': 'Exclusively owned default executor; admitted CPython completion tokens'}
    before = dict(receipt)
    _, note = _assert_note_preserves_primary(receipt, 'owned_executor', 'Owned executor cleanup: ')
    assert repr(receipt) in note and receipt == before
    assert 'Interrupted-start recovery:' not in note
    assert 'not an executor-context shutdown result' not in note


def test_positive_startup_only_receipt_requires_exact_scoped_proof():
    receipt = {'qualified': True, 'drained': True, 'scope': _STARTUP_SCOPE,
               'owned_workers': 1, 'all_captured_workers_completed': True,
               'registry_mutation': False}
    before = dict(receipt)
    _, note = _assert_note_preserves_primary(receipt, 'startup_only', 'Interrupted-start recovery: ')
    assert repr(receipt) in note and receipt == before


@pytest.mark.parametrize('receipt', [
    {'qualified': False, 'drained': False, 'reason': 'No exact post-launch owned-worker proof'},
    {'qualified': False, 'drained': False, 'reason': 'Unsupported completion capability'}
], ids=['ordinary-nonadmission', 'unsupported-runtime'])
def test_startup_nonadmission_does_not_classify_context_shutdown(receipt):
    before = dict(receipt)
    _, note = _assert_note_preserves_primary(
        receipt, 'startup_only',
        'Optional startup recovery not qualified (not an executor-context shutdown result): ')
    assert repr(receipt) in note and receipt == before
    assert not note.startswith('Interrupted-start recovery:')


@pytest.mark.parametrize('receipt', [
    {}, {'qualified': True}, {'qualified': True, 'drained': True},
    {'qualified': True, 'drained': True, 'scope': 'unrelated helper'},
    {'qualified': True, 'drained': True, 'scope': [_STARTUP_SCOPE]},
    {'qualified': 1, 'drained': True, 'scope': _STARTUP_SCOPE},
    {'qualified': True, 'drained': 1, 'scope': _STARTUP_SCOPE},
    {'qualified': 0, 'drained': False}, {'qualified': False, 'drained': 0},
    {'qualified': True, 'drained': False, 'scope': _STARTUP_SCOPE},
    {'qualified': False, 'drained': True, 'scope': _STARTUP_SCOPE},
    None, [], 'not a receipt'
], ids=['empty', 'missing-drained', 'missing-scope', 'wrong-scope', 'nonstr-scope',
        'integer-qualified', 'integer-drained', 'integer-declined-qualified',
        'integer-declined-drained', 'contradictory-qualified', 'contradictory-drained',
        'none', 'list', 'text'])
def test_malformed_startup_receipt_does_not_infer_recovery_or_context_drain(receipt):
    before = repr(receipt)
    _, note = _assert_note_preserves_primary(
        receipt, 'startup_only',
        'Optional startup-recovery diagnostic (qualification unclassified; '
        'not an executor-context shutdown result): ')
    assert before in note and repr(receipt) == before
    assert not note.startswith('Interrupted-start recovery:')


def test_receipt_dict_subclass_is_not_used_as_classification_authority():
    class Receipt(dict):
        def get(self, *args):
            raise AssertionError('Subclass lookup must not be used as proof')
    receipt = Receipt(qualified=True, drained=True, scope=_STARTUP_SCOPE)
    _, note = _assert_note_preserves_primary(
        receipt, 'startup_only',
        'Optional startup-recovery diagnostic (qualification unclassified; '
        'not an executor-context shutdown result): ')
    assert repr(receipt) in note


def test_hostile_colliding_key_cannot_create_cleanup_failure():
    class Collision:
        def __hash__(self):
            return hash('qualified')

        def __eq__(self, other):
            raise RuntimeError('Hostile receipt key comparison')

        def __repr__(self):
            return '<hostile-key>'
    receipt = {Collision(): False}
    _, note = _assert_note_preserves_primary(
        receipt, 'startup_only',
        'Optional startup-recovery diagnostic (qualification unclassified; '
        'not an executor-context shutdown result): ')
    assert repr(receipt) in note
    assert len(receipt) == 1 and next(iter(receipt.values())) is False


@pytest.mark.parametrize('source', ['unclassified', 'unknown', None, 1])
def test_unknown_diagnostic_source_does_not_infer_scope(source):
    receipt = {'qualified': True, 'drained': True, 'scope': _STARTUP_SCOPE}
    _, note = _assert_note_preserves_primary(receipt, source, 'Cleanup diagnostic receipt (unclassified): ')
    assert repr(receipt) in note


def test_omitted_source_is_neutral():
    error = _primary()
    receipt = {'qualified': True, 'drained': True, 'scope': _STARTUP_SCOPE}
    diagnostics.note_cleanup_receipt(error, receipt)
    assert _notes(error)[-1].startswith('Cleanup diagnostic receipt (unclassified): ')


@pytest.mark.parametrize('source,prefix', [
    ('owned_executor', 'Owned executor cleanup: '),
    ('startup_only', 'Optional startup-recovery diagnostic (qualification unclassified; '
                     'not an executor-context shutdown result): '),
    ('unclassified', 'Cleanup diagnostic receipt (unclassified): ')
])
def test_hostile_receipt_representation_cannot_replace_primary(source, prefix):
    class Receipt:
        def __repr__(self):
            raise KeyboardInterrupt('Receipt formatting interruption')
    _, note = _assert_note_preserves_primary(Receipt(), source, prefix)
    assert note.endswith('[receipt formatting unavailable]')


def test_receipt_annotation_does_not_retain_payload_objects():
    class Receipt:
        def __repr__(self):
            return '<unclassified receipt>'
    # Construct the deliberately retained traceback in this frame, before the
    # payload. The general assertion helper constructs it in a nested frame
    # whose f_back retains that helper's receipt argument independently of the
    # diagnostic implementation.
    error = _primary('explicit')
    chain, args, traceback = _chain(error), error.args, error.__traceback__
    receipt = Receipt()
    reference = weakref.ref(receipt)
    diagnostics.note_cleanup_receipt(error, receipt, source='startup_only')
    assert _chain(error) == chain and error.args == args
    assert error.__traceback__ is traceback
    notes = _notes(error)
    assert notes[0] == 'preexisting caller note' and len(notes) == 2
    note = notes[1]
    assert note.startswith(
        'Optional startup-recovery diagnostic (qualification unclassified; '
        'not an executor-context shutdown result): ')
    assert '_dwd_cleanup_failures' not in _attributes(error)
    del receipt
    gc.collect()
    assert reference() is None
    assert '<unclassified receipt>' in note and _notes(error)[-1] == note


def test_malformed_existing_notes_remain_unchanged():
    error = _primary('suppressed')
    original_notes = ('malformed caller notes',)
    _attributes(error)['__notes__'] = original_notes
    chain, traceback = _chain(error), error.__traceback__
    diagnostics.note_cleanup_receipt(error, {'qualified': False, 'drained': False},
                                     source='startup_only')
    assert _attributes(error)['__notes__'] is original_notes
    assert _chain(error) == chain and error.__traceback__ is traceback
    assert '_dwd_cleanup_failures' not in _attributes(error)


def test_owned_unproved_completion_keeps_primary_and_secondary_evidence(monkeypatch):
    error = _primary('explicit')
    chain, original_traceback = _chain(error), error.__traceback__
    receipt = {'qualified': False, 'drained': False, 'shutdown_requested': True,
               'reason': 'Possible omitted worker completion is unproved'}
    pool, calls = object(), []

    def drain(executor, error=None):
        calls.append((executor, error))
        return receipt, None

    monkeypatch.setattr(scopes, 'drain_owned_executor', drain)
    with pytest.raises(RuntimeError) as caught:
        scopes._drain_default_executor(pool, error)
    assert caught.value is error and calls == [(pool, error)]
    assert _chain(error) == chain
    tail = error.__traceback__
    while tail is not None and tail is not original_traceback:
        tail = tail.tb_next
    assert tail is original_traceback
    notes = _notes(error)
    assert any(note.startswith('Owned executor cleanup: ') and repr(receipt) in note for note in notes)
    records = _attributes(error)['_dwd_cleanup_failures']
    assert len(records) == 1 and 'completion is unproved' in records[0]['traceback']
    assert not any('not an executor-context shutdown result' in note for note in notes)
