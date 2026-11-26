"""Public resumable CV under joblib's required shared-memory backend."""
import _thread
from contextlib import ExitStack
import errno
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
from time import monotonic
from unittest.mock import patch

import pytest


class _InterruptOnConsumption:
    """Let joblib start its real output generator, then interrupt before consuming it."""
    def __init__(self, outputs, interrupt):
        self.outputs = outputs
        self.interrupt = interrupt

    def __next__(self):
        return next(self.outputs)

    def __iter__(self):
        self.interrupt()
        return self.outputs

    def close(self):
        self.outputs.close()


def _lingering_threads(before):
    # The child runs one case; ordinary library initialization is complete.
    # Allow terminating idle threads to leave, but keep the exception alive.
    extra = set(threading.enumerate()) - before
    deadline = monotonic() + 1
    for thread in extra:
        thread.join(max(0, deadline - monotonic()))
    return [t.name for t in extra if t.is_alive()]


def _exercise(directory, case):
    import numpy as np
    from joblib import Parallel, parallel_config
    from joblib.parallel import ThreadingBackend, get_active_backend
    from scipy.linalg import eigh
    from threadpoolctl import ThreadpoolController
    from dwd.gen_kern_dwd import KernGDWD
    import resumable_kernel_cv as example

    # Initialize lazy libraries, then give each adjustable provider a caller
    # setting distinguishable from the one thread requested for each CV fold.
    eigh(np.diag([1., 2.]), driver='evd')
    pools = ThreadpoolController().lib_controllers
    for index, pool in enumerate(pools):
        pool.set_num_threads(3 + index % 3)

    def counts():
        return {pool.filepath: pool.get_num_threads() for pool in pools}

    caller = counts()
    X = np.random.default_rng(448713).normal(size=(24, 4))
    y = np.tile([-1, 1], 12)
    model = KernGDWD(kernel='rbf', kernel_kws={'gamma': .2}, random_state=23,
                     max_iter=3, stopping='fixed')
    path = directory / 'interrupted'
    fold_count = 6 if case == 'suspended-dispatch' else 3
    total = 3 * fold_count

    def search(path, resume=False):
        return example.run_resumable_cv(model, X, y, {'lambd': [.03, .3, .6]}, path,
                                        cv=fold_count, jobs=2, resume=resume, refit_best=True)

    def receipts():
        return {p.relative_to(path): p.read_bytes() for p in path.glob('receipts/fold-*/*.json')}

    main = threading.get_ident()
    first_pair = threading.Barrier(2, timeout=45)
    ordinary_workers = set()
    original_score = example._fit_and_score

    def observe_score(*args):
        candidate, fold = args[-2:]
        if fold < 2 and candidate['lambd'] == .03:
            ordinary_workers.add(threading.get_ident())
            first_pair.wait()
        return original_score(*args)

    primary = threading.Event()
    returned = threading.Event()
    parked = threading.Event()
    batch_finished = threading.Event()
    condition = threading.Condition()
    active = set()
    late_starts = []
    original_fold = example._run_fold

    def observe_fold(*args, **kwargs):
        fold = args[6]
        with condition:
            if returned.is_set():
                late_starts.append(fold)
            active.add(fold)
        try:
            return original_fold(*args, **kwargs)
        finally:
            with condition:
                active.remove(fold)
                condition.notify_all()

    def finish_in_flight():
        assert primary.wait(45), 'primary error was never triggered'
        # A broken search releases these threads immediately upon return. A
        # draining search can wait here for real work to finish independently.
        returned.wait(.5)

    original_submit = ThreadingBackend.submit

    def park_first_batch(backend, task, callback=None):
        if not parked.is_set():
            def run_late():
                parked.set()
                try:
                    finish_in_flight()
                    return task()
                finally:
                    batch_finished.set()
            result = original_submit(backend, run_late, callback)
            assert parked.wait(45), 'first batch did not reach its worker'
            return result
        return original_submit(backend, task, callback)

    both_ready = threading.Event()
    at_second_receipt = threading.Barrier(2, action=both_ready.set, timeout=45)
    original_write = example.write_receipt

    def write(path, record):
        fold = record['fold_index']
        if record['candidate_index'] == 1 and fold < 2:
            if case == 'late-batch':
                assert parked.is_set() and fold == 1
            else:
                at_second_receipt.wait()
            if case in ('sigint', 'repeated-sigint', 'suspended-dispatch'):
                if case != 'suspended-dispatch' and fold == 0:
                    _thread.interrupt_main(signal.SIGINT)
                finish_in_flight()
            elif fold == 1:
                primary.set()
                raise PermissionError(errno.EACCES, 'cannot commit second receipt', str(path))
            else:
                finish_in_flight()
        return original_write(path, record)

    first_interrupt = KeyboardInterrupt('interrupted with both folds active')
    second_interrupt = KeyboardInterrupt('interrupted again before cleanup')
    diagnostic = OSError('caller signal detail')
    signals = []

    def interrupt(signum, frame):
        signals.append(signum)
        if len(signals) == 2:
            raise second_interrupt
        primary.set()
        raise first_interrupt from diagnostic

    def at_handler_entry(frame, event, arg):
        if frame.f_globals is example.__dict__:
            # Interrupt the first example handler to accept the signal. Its
            # function, line number and presence of an inner retry may change.
            if event == 'line' and len(signals) == 1 and sys.exception() is first_interrupt:
                signal.raise_signal(signal.SIGINT)
            return at_handler_entry

    original_outputs = Parallel._get_outputs
    original_register = example.CheckpointRun.register_fold
    late_registrations = []

    def interrupted_outputs(parallel, *args, **kwargs):
        def before_consumption():
            assert both_ready.wait(45)
            _thread.interrupt_main(signal.SIGINT)
            # If joblib runs off the main thread, wait for the caller's handler.
            assert primary.wait(45)
        return _InterruptOnConsumption(original_outputs(parallel, *args, **kwargs),
                                       before_consumption)

    def register(checkpoint, index):
        if index == 4:
            # A real completion callback can be publishing the first lock beyond
            # the initial 2*n_jobs lookahead while the coordinator is unwinding.
            returned.wait(.5)
        result = original_register(checkpoint, index)
        if returned.is_set():
            late_registrations.append(index)
        return result

    with parallel_config(backend='threading', require='sharedmem', n_jobs=3) as config:
        with patch.object(example, '_fit_and_score', observe_score):
            ordinary = search(directory / 'ordinary')
        assert len(ordinary_workers) == 2 and main not in ordinary_workers
        assert counts() == caller
        assert get_active_backend() == (config['backend'], 3)

        with ExitStack() as hooks:
            hooks.enter_context(patch.object(example, '_run_fold', observe_fold))
            hooks.enter_context(patch.object(example, 'write_receipt', write))
            if case == 'late-batch':
                hooks.enter_context(patch.object(ThreadingBackend, 'submit', park_first_batch))
            if case == 'suspended-dispatch':
                hooks.enter_context(patch.object(Parallel, '_get_outputs', interrupted_outputs))
                hooks.enter_context(patch.object(example.CheckpointRun, 'register_fold', register))
            if case in ('sigint', 'repeated-sigint', 'suspended-dispatch'):
                previous = signal.signal(signal.SIGINT, interrupt)
                hooks.callback(signal.signal, signal.SIGINT, previous)
                expected = pytest.raises(KeyboardInterrupt, match='interrupted with both folds active')
            else:
                expected = pytest.raises(PermissionError, match='cannot commit second receipt')
            before = set(threading.enumerate())
            if case == 'repeated-sigint':
                hooks.callback(sys.settrace, sys.gettrace())
                sys.settrace(at_handler_entry)
            try:
                with expected as held:
                    search(path)
                with condition:
                    active_at_return = sorted(active)
                limits_at_return = counts()
                committed = receipts()
                registrations = set(path.glob('fold-*.lock'))
            finally:
                returned.set()
            if case == 'late-batch':
                assert batch_finished.wait(45)
            with condition:
                assert condition.wait_for(lambda: not active, timeout=45)
            lingering = _lingering_threads(before)
            after = counts()
            observed = (active_at_return, late_starts, limits_at_return, after)
            assert observed == ([], [], caller, caller), (observed, caller)
            assert receipts() == committed
            assert not (late_registrations or lingering), (late_registrations, lingering)
            assert set(path.glob('fold-*.lock')) == registrations
            assert isinstance(held.value, (KeyboardInterrupt, PermissionError))
            if case == 'repeated-sigint':
                assert signals == [signal.SIGINT, signal.SIGINT]
                assert held.value is first_interrupt and held.value.__cause__ is diagnostic
        assert get_active_backend() == (config['backend'], 3)

        # The interrupted evaluations may commit. Later candidates must wait
        # for the resumed call, which reuses each previously committed receipt.
        records = [json.loads(contents)['payload'] for contents in committed.values()]
        published = {(r['fold_index'], r['candidate_index']) for r in records}
        first_folds = {1} if case == 'late-batch' else {0, 1}
        assert all((fold, 0) in published and (fold, 2) not in published for fold in first_folds)
        assert 0 < len(committed) < total
        resumed = search(path, resume=True)
        assert resumed['work_summary']['replayed_evaluations'] == len(committed)
        assert resumed['work_summary']['executed_evaluations'] == total - len(committed)
        for relative, contents in committed.items():
            assert (path / relative).read_bytes() == contents
        assert resumed['best_params'] == ordinary['best_params']
        assert resumed['best_score'] == ordinary['best_score']
        for left, right in zip(resumed['all_cv_results'], ordinary['all_cv_results'], strict=True):
            for key in ('train_correct', 'test_correct'):
                assert left[key] == right[key]
        np.testing.assert_array_equal(resumed['best_clf'].decision_function(X),
                                      ordinary['best_clf'].decision_function(X))
        assert counts() == caller
        # Mutation distinguishes the original live mapping from a settings copy.
        config['n_jobs'] = 5
        assert get_active_backend() == (config['backend'], 5)


def _interrupted_teardown(case):
    from joblib import Parallel, parallel_config
    from joblib.parallel import ThreadingBackend
    import resumable_kernel_cv as example

    coordinator_first = case.endswith('-sigint')
    first = KeyboardInterrupt('coordinator interrupted') if coordinator_first else OSError('fold failed')
    diagnostic = ValueError('original diagnostic')
    together = threading.Barrier(2, timeout=45)
    primary_started = threading.Event()
    secondary_sent = threading.Event()
    secondary_received = threading.Event()

    def fold(index, **kwargs):
        together.wait()
        if index == 0:
            if coordinator_first:
                _thread.interrupt_main(signal.SIGINT)
            else:
                primary_started.set()
                raise first from diagnostic
        assert primary_started.wait(45)

    def interrupt(signum, frame):
        if secondary_sent.is_set():
            secondary_received.set()
            raise KeyboardInterrupt('interrupted joblib teardown')
        primary_started.set()
        raise first from diagnostic

    def interrupt_caller():
        assert primary_started.is_set()
        if not secondary_sent.is_set():
            secondary_sent.set()
            # A dispatcher in a different thread can finish even if the
            # caller processes this signal when its wait for shutdown ends.
            _thread.interrupt_main(signal.SIGINT)

    original_error = Parallel._raise_error_fast
    original_terminate = ThreadingBackend.terminate

    def dispatch_error(parallel):
        try:
            return original_error(parallel)
        except OSError:
            if case == 'teardown-entry-fold-error':
                interrupt_caller()
            raise

    def terminate(backend):
        # A dispatcher may capture the fold error itself and let joblib close
        # a healthy pool. Either way the user's signal goes to the caller.
        interrupt_caller()
        return original_terminate(backend)

    previous = signal.signal(signal.SIGINT, interrupt)
    try:
        before = set(threading.enumerate())
        try:
            raise LookupError('unrelated caller error')
        except LookupError as ambient:
            with parallel_config(backend='threading', require='sharedmem'), \
                    patch.object(example, '_run_fold', fold), \
                    patch.object(Parallel, '_raise_error_fast', dispatch_error), \
                    patch.object(ThreadingBackend, 'terminate', terminate), \
                    pytest.raises(type(first)) as caught:
                example._parallel_folds([(0,), (1,)], workers=2, native_threads=1)
            assert sys.exception() is ambient
        assert secondary_received.is_set()
        assert caught.value is first
        assert str(diagnostic) in str(first.__cause__)
        assert not (lingering := _lingering_threads(before)), lingering
    finally:
        primary_started.set()
        signal.signal(signal.SIGINT, previous)


def _late_worker_error():
    from joblib import parallel_config
    from joblib.parallel import ThreadingBackend
    import resumable_kernel_cv as example

    first = KeyboardInterrupt('caller stopped search')
    diagnostic = RuntimeError('caller diagnostic')
    entered = threading.Barrier(2, timeout=45)
    aborting = threading.Event()
    worker_done = threading.Event()
    original_override = example._resource_override
    original_retrieve = ThreadingBackend.retrieve_result_callback

    def fold(index, **kwargs):
        entered.wait()
        if index == 0:
            _thread.interrupt_main(signal.SIGINT)
            assert aborting.wait(45)
            raise PermissionError('receipt failed after caller entered cleanup')
        assert worker_done.wait(45)

    def retrieve(backend, result):
        try:
            return original_retrieve(backend, result)
        finally:
            # The sibling waits; only the failing fold can complete first.
            worker_done.set()

    def override(*args, abort, **kwargs):
        def wait_for_worker(*abort_args, **abort_kwargs):
            assert sys.exception() is first
            aborting.set()
            assert worker_done.wait(45)
            return abort(*abort_args, **abort_kwargs)
        return original_override(*args, abort=wait_for_worker, **kwargs)

    def interrupt(signum, frame):
        raise first from diagnostic

    before = set(threading.enumerate())
    previous = signal.signal(signal.SIGINT, interrupt)
    try:
        with parallel_config(backend='threading', require='sharedmem'), \
                patch.object(example, '_run_fold', fold), \
                patch.object(example, '_resource_override', override), \
                patch.object(ThreadingBackend, 'retrieve_result_callback', retrieve), \
                pytest.raises(KeyboardInterrupt) as caught:
            example._parallel_folds([(0,), (1,)], workers=2, native_threads=1)
        assert aborting.is_set() and worker_done.is_set()
        assert caught.value is first and caught.value.__cause__ is diagnostic
        assert not (lingering := _lingering_threads(before)), lingering
    finally:
        aborting.set()
        worker_done.set()
        signal.signal(signal.SIGINT, previous)


def _run_child(tmp_path, case):
    # Deliver SIGINT and set real native limits only in an isolated process.
    result = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()),
                             json.dumps(sys.path), str(tmp_path), case],
                            capture_output=True, text=True, errors='replace', timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('case', ['sigint', 'receipt-error', 'late-batch', 'suspended-dispatch',
                                  'repeated-sigint'])
def test_sharedmem_failure_restores_caller_before_return(tmp_path, case):
    _run_child(tmp_path, case)


@pytest.mark.parametrize('case', ['teardown-fold-error', 'teardown-sigint',
                                  'teardown-entry-fold-error'])
def test_interrupt_during_joblib_teardown_preserves_primary(tmp_path, case):
    _run_child(tmp_path, case)


def test_worker_error_after_caller_enters_cleanup_preserves_interrupt(tmp_path):
    _run_child(tmp_path, 'late-worker-error')


if __name__ == '__main__':
    sys.path[:] = json.loads(sys.argv[1])
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
    if sys.argv[3].startswith('teardown-'):
        _interrupted_teardown(sys.argv[3])
    elif sys.argv[3] == 'late-worker-error':
        _late_worker_error()
    else:
        _exercise(Path(sys.argv[2]), sys.argv[3])
