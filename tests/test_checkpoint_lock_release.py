"""Interrupted public searches release their own checkpoint descriptors promptly."""
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

from dwd.gen_kern_dwd import KernGDWD


EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import _cv_checkpoint as checkpoint
import resumable_kernel_cv as example


@pytest.fixture
def search(tmp_path, monkeypatch):
    # Descriptor ownership is independent of the machine's installed libraries.
    monkeypatch.setattr(example, 'runtime_fingerprint', lambda **options: {'native_pools': []})
    root = tmp_path / 'run'
    X = np.random.default_rng(17).normal(size=(20, 4))
    y = np.tile([-1, 1], 10)
    estimator = KernGDWD(kernel='rbf', kernel_kws={'gamma': .1}, random_state=2,
                         max_iter=3, stopping='fixed')

    def run(resume=False, jobs=1):
        return example.run_resumable_cv(estimator, X, y, {'lambd': [.02, .2]},
                                        root, cv=2, jobs=jobs, refit_best=False, resume=resume)

    return root, run


@pytest.mark.parametrize('scope,interrupt_work', [
    ('coordinator', False), ('coordinator', True), ('fold', True)])
def test_interrupted_release_allows_immediate_resume(search, monkeypatch, scope, interrupt_work):
    root, run = search
    work_error = KeyboardInterrupt('interrupted after publishing a receipt')
    exit_error = KeyboardInterrupt('interrupted before closing the descriptor')
    publish = example.write_receipt
    owner = checkpoint.CheckpointRun if scope == 'coordinator' else checkpoint._Lease
    original_exit = owner.__exit__
    interrupted = []

    def publish_then_interrupt(path, payload):
        publish(path, payload)
        raise work_error

    def interrupt_exit(lease, *args):
        if not interrupted and (scope == 'coordinator' or lease.path.name == 'fold-0000.lock'):
            interrupted.append(lease)
            raise exit_error
        return original_exit(lease, *args)

    with monkeypatch.context() as fault:
        if interrupt_work:
            fault.setattr(example, 'write_receipt', publish_then_interrupt)
        fault.setattr(owner, '__exit__', interrupt_exit)
        with pytest.raises(KeyboardInterrupt) as caught:
            run()
    saved = {p: p.read_bytes() for p in root.glob('receipts/fold-*/*.json')}
    assert len(interrupted) == 1 and len(saved) == (1 if interrupt_work else 4)
    # Both the caught traceback and the actual owner remain live during resume.
    summary = run(resume=True)['work_summary']
    assert caught.value is (work_error if interrupt_work else exit_error)
    assert (summary['replayed_evaluations'], summary['executed_evaluations']) == (
        len(saved), 4 - len(saved))
    assert all(p.read_bytes() == contents for p, contents in saved.items())


@pytest.mark.parametrize('scope', ['coordinator', 'fold', 'old-worker-probe'])
def test_interrupted_acquisition_releases_registered_owner(search, monkeypatch, scope):
    root, run = search
    resume = scope != 'fold'
    if resume:
        run()
    # Interrupt just after the OS lock succeeds, before the caller can register
    # __exit__. Keep the acquiring frame reachable while the next search runs.
    original_enter = checkpoint._Lease.__enter__
    interruption = KeyboardInterrupt('acquired')
    lock_name = 'coordinator.lock' if scope == 'coordinator' else 'fold-0000.lock'

    def acquire_then_interrupt(lease):
        result = original_enter(lease)
        if lease.path.name == lock_name:
            raise interruption
        return result

    with monkeypatch.context() as fault:
        fault.setattr(checkpoint._Lease, '__enter__', acquire_then_interrupt)
        with pytest.raises(KeyboardInterrupt) as caught:
            run(resume=resume)
    summary = run(resume=True)['work_summary']
    assert caught.value is interruption and caught.value.__traceback__ is not None
    assert (summary['replayed_evaluations'], summary['executed_evaluations']) == (
        (4, 0) if resume else (0, 4))


def test_reentering_same_coordinator_keeps_live_owner(tmp_path):
    identity = {'data': 'fixed'}
    with checkpoint.CheckpointRun(tmp_path, identity):
        pass
    owner = checkpoint.CheckpointRun(tmp_path, identity, resume=True)
    with owner:
        generation = (tmp_path / 'generation.json').read_bytes()
        with pytest.raises(checkpoint.CheckpointBusyError), owner:
            pytest.fail('a live coordinator was entered twice')
        with pytest.raises(checkpoint.CheckpointBusyError):
            with checkpoint.CheckpointRun(tmp_path, identity, resume=True):
                pytest.fail('a concurrent coordinator took the live lock')
        assert (tmp_path / 'generation.json').read_bytes() == generation
        owner.assert_current()
    with owner:
        owner.assert_current()


@pytest.mark.parametrize('scope', ['coordinator', 'stale-fold'])
@pytest.mark.parametrize('fault', ['exit-call', 'unlock-interrupt', 'unlock-error'])
def test_failed_entry_preserves_primary_and_cause(search, monkeypatch, scope, fault):
    root, run = search
    if scope == 'coordinator':
        run()
    cause = ValueError('manifest diagnostic')
    first = KeyboardInterrupt('manifest interrupted')
    secondary = KeyboardInterrupt('interrupted during entry cleanup')
    owner = checkpoint.CheckpointRun if scope == 'coordinator' else checkpoint._Lease
    original_exit = owner.__exit__
    errors, faults = [], []
    if os.name == 'nt':
        import msvcrt as provider
        locking, unlock = 'locking', provider.LK_UNLCK
    else:
        import fcntl as provider
        locking, unlock = 'flock', provider.LOCK_UN
    os_lock = getattr(provider, locking)

    def fail_unlock(fd, mode, *args):
        if mode == unlock:
            faults.append(fault)
            if fault == 'unlock-error':
                raise OSError('test unlock refused')
            raise secondary
        return os_lock(fd, mode, *args)

    def interrupt_cleanup(lease, kind, error, traceback):
        if error is not None and not errors:
            errors.append(error)
            if fault == 'exit-call':
                faults.append(fault)
                raise secondary
            with patch.object(provider, locking, fail_unlock):
                return original_exit(lease, kind, error, traceback)
        return original_exit(lease, kind, error, traceback)

    def interrupt_read(*args, **kwargs):
        raise first from cause

    with monkeypatch.context() as inject:
        inject.setattr(owner, '__exit__', interrupt_cleanup)
        if scope == 'coordinator':
            inject.setattr(checkpoint, 'read_receipt', interrupt_read)
        else:
            inject.setattr(example, 'worker_lease', lambda directory, token, index:
                           checkpoint.worker_lease(directory, 'retired', index))
        with pytest.raises((checkpoint.CheckpointError, KeyboardInterrupt, OSError)) as caught:
            run(resume=scope == 'coordinator')
    assert len(errors) == 1 and faults == [fault]
    summary = run(resume=True)['work_summary']
    assert caught.value is errors[0]
    if scope == 'coordinator':
        assert caught.value is first and caught.value.__cause__ is cause
        assert summary['replayed_evaluations'] == 4
    else:
        assert 'no longer owns' in str(caught.value) and caught.value.__cause__ is None
        assert summary['executed_evaluations'] == 4
    if fault == 'unlock-error':
        assert any('test unlock refused' in note for note in caught.value.__notes__)


def test_busy_entry_interrupted_during_cleanup_keeps_owner_and_os_cause(search, monkeypatch):
    root, run = search
    run()
    identity = json.loads((root / 'identity.json').read_bytes())['payload']['identity']
    original_exit = checkpoint._Lease.__exit__
    interruptions = []

    def interrupt_cleanup(lease, kind, error, traceback):
        if error is not None and not interruptions:
            interruptions.append(True)
            raise KeyboardInterrupt('interrupt while refusing a genuine owner')
        return original_exit(lease, kind, error, traceback)

    with checkpoint.CheckpointRun(root, identity, resume=True) as active:
        before = (root / 'generation.json').read_bytes()
        with monkeypatch.context() as inject:
            inject.setattr(checkpoint._Lease, '__exit__', interrupt_cleanup)
            with pytest.raises((checkpoint.CheckpointBusyError, KeyboardInterrupt)) as caught:
                run(resume=True)
        assert isinstance(caught.value, checkpoint.CheckpointBusyError) and interruptions == [True]
        assert isinstance(caught.value.__cause__, OSError)
        with pytest.raises(checkpoint.CheckpointBusyError):
            run(resume=True)
        active.assert_current()
        assert (root / 'generation.json').read_bytes() == before
    assert run(resume=True)['work_summary']['replayed_evaluations'] == 4


def test_second_interrupt_while_accepting_entry_error_preserves_diagnostics(search, monkeypatch):
    _, run = search
    run()
    first, second = KeyboardInterrupt('manifest read'), KeyboardInterrupt('accepting failure')
    cause = ValueError('manifest diagnostic')
    handler = getattr(checkpoint.CheckpointRun, '_acquire_or_release', checkpoint.CheckpointRun.__enter__)
    events = []

    def interrupt_read(*args, **kwargs):
        raise first from cause

    def trace(frame, event, arg):
        if frame.f_code is not handler.__code__ or not isinstance(frame.f_locals['self'], checkpoint.CheckpointRun):
            return None
        if event == 'exception' and arg[1] is first:
            events.append('acquisition failed')
        elif event == 'line' and events == ['acquisition failed']:
            events.append('interrupted handler')
            raise second
        return trace

    try:
        raise LookupError('caller is handling another error')
    except LookupError as ambient:
        with monkeypatch.context() as inject:
            inject.setattr(checkpoint, 'read_receipt', interrupt_read)
            previous = sys.gettrace()
            try:
                sys.settrace(trace)
                with pytest.raises(KeyboardInterrupt) as caught:
                    run(resume=True)
            finally:
                sys.settrace(previous)
        assert events == ['acquisition failed', 'interrupted handler']
        assert caught.value is first and first.__cause__ is cause and first.__context__ is ambient
    assert run(resume=True)['work_summary']['replayed_evaluations'] == 4


@pytest.mark.parametrize('jobs,worker_first', [(1, False), (2, False), (1, True)])
def test_delayed_fold_registered_before_dispatch_and_generation_handoff(search, monkeypatch,
                                                                      jobs, worker_first):
    root, run = search
    queued = []
    real_fold = example._run_fold

    def stop_before_worker(*args):
        queued.append(args)
        raise KeyboardInterrupt('queued before the worker acquired its first lock')

    def stop_before_parallel_worker(arguments, workers, native_threads):
        stop_before_worker(*next(iter(arguments)))

    with monkeypatch.context() as queue:
        queue.setattr(example, '_run_fold', stop_before_worker)
        queue.setattr(example, '_parallel_folds', stop_before_parallel_worker)
        with pytest.raises(KeyboardInterrupt):
            run(jobs=jobs)
    assert len(queued) == 1
    args = queued[0]
    old_token = args[9]
    registered = sorted(path.name for path in root.glob('fold-*.lock'))
    # If the old fold starts first, its generation and receipts stay intact.
    if worker_first:
        generation = (root / 'generation.json').read_bytes()
        with checkpoint.worker_lease(root, old_token, 0):
            with pytest.raises(checkpoint.CheckpointBusyError):
                run(resume=True)
            assert (root / 'generation.json').read_bytes() == generation
            assert not list(root.glob('receipts/fold-*/*.json'))

    publish = checkpoint.write_receipt
    admissions = []

    def try_delayed_fold(path, payload):
        if path.name == 'generation.json':
            try:
                real_fold(*args)
            except checkpoint.CheckpointBusyError:
                admissions.append('refused')
            else:
                admissions.append('admitted')
        publish(path, payload)

    with monkeypatch.context() as handoff:
        handoff.setattr(checkpoint, 'write_receipt', try_delayed_fold)
        summary = run(resume=True)['work_summary']
    assert admissions == ['refused']
    assert registered == ['fold-0000.lock']
    assert summary['replayed_evaluations'] == 0 and summary['executed_evaluations'] == 4
    with pytest.raises(checkpoint.CheckpointError, match='no longer owns'):
        real_fold(*args)
