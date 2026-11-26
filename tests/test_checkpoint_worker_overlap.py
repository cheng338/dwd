"""Checkpoint fold-worker ownership, independent progress and failure cleanup."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from functools import partial
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from unittest.mock import patch
import warnings

import pytest


def _ordinary_task(index, root):
    if index < 2:
        (root / f'ordinary-{os.getpid()}').touch()
        deadline = time.monotonic() + 90
        while not any((root / 'cv' / 'receipts').glob('fold-*/*.json')):
            if (root / 'cv-failed').exists():
                raise RuntimeError('CV failed before the first fold receipt')
            if time.monotonic() > deadline:
                raise TimeoutError('No CV fold receipt while ordinary work was active')
            time.sleep(.02)
    return index, os.getpid()


def _observed_fold(root, coordinator_pid, clf, X, y, train, test, candidates, fold_index,
                   *rest):
    import numpy as np
    from resumable_kernel_cv import _run_fold

    pid = os.getpid()
    assert pid != coordinator_pid, 'CV fold ran in the coordinator'
    assert not (root / f'ordinary-{pid}').exists(), 'CV fold used an ordinary worker'
    assert isinstance(X, np.memmap), 'CV worker did not receive a memmapped feature array'
    backing = Path(X.filename)
    assert backing.is_file() and any(folder.samefile(root / 'maps') for folder in backing.parents)
    (root / f'cv-worker-{pid}').write_text(
        json.dumps({'fold': fold_index, 'map': str(backing)}), encoding='utf-8')
    deadline = time.monotonic() + 60
    while len(list(root.glob('cv-worker-*'))) < 2:
        if time.monotonic() > deadline:
            raise TimeoutError('Two distinct CV fold workers must enter together')
        time.sleep(.02)
    return _run_fold(clf, X, y, train, test, candidates, fold_index, *rest)


def _overlap(root):
    import numpy as np
    from joblib import Parallel, delayed, parallel_config
    from dwd.gen_kern_dwd import KernGDWD
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
    from _cv_checkpoint import io_path
    import resumable_kernel_cv as example

    def ordinary_job():
        with parallel_config(backend='loky', inner_max_num_threads=1):
            return Parallel(n_jobs=2, pre_dispatch=2, batch_size=1)(
                delayed(_ordinary_task)(index, root) for index in range(6))

    model = KernGDWD(kernel='rbf', kernel_kws={'gamma': .1}, random_state=2,
                     max_iter=3, stopping='fixed')
    X = np.random.default_rng(197).normal(size=(20, 4))
    y = np.tile([-1, 1], 10)
    grid = {'lambd': [.02, .2]}
    assert not {'cvxpy', 'scs'}.intersection(sys.modules)
    serial = example.run_resumable_cv(model, X, y, grid, root / 'serial',
                                      cv=2, jobs=1, refit_best=False)
    with ThreadPoolExecutor(max_workers=1) as thread:
        ordinary = thread.submit(ordinary_job)
        deadline = time.monotonic() + 60
        while len(list(root.glob('ordinary-*'))) < 2:
            if ordinary.done():
                ordinary.result()
            if time.monotonic() > deadline:
                raise TimeoutError('Both ordinary workers must start before CV')
            time.sleep(.02)
        # Force even this small dataset through the real memmap reducers.
        maps = root / 'maps'
        maps.mkdir()
        try:
            observe = partial(_observed_fold, root, os.getpid())
            with parallel_config(max_nbytes=1, temp_folder=str(io_path(maps))), \
                    patch.object(example, '_run_fold', observe):
                cv = example.run_resumable_cv(model, X, y, grid, root / 'cv',
                                              cv=2, jobs=2, refit_best=False)
        except BaseException:
            (root / 'cv-failed').touch()
            raise
        assert not list(maps.iterdir())
        actual = ordinary.result(timeout=30)
    assert [index for index, pid in actual] == list(range(6))
    # Later tasks were dispatched by the original pool's completion callbacks.
    assert {pid for index, pid in actual} == {pid for index, pid in actual[:2]}
    observed = [json.loads(p.read_text(encoding='utf-8')) for p in root.glob('cv-worker-*')]
    assert sorted(row['fold'] for row in observed) == [0, 1]
    assert len({row['map'] for row in observed}) == 1
    assert cv['work_summary']['executed_evaluations'] == 4
    assert cv['work_summary']['fold_workers'] == 2
    assert cv['best_params'] == serial['best_params']
    assert cv['best_score'] == serial['best_score']
    for parallel_fold, serial_fold in zip(cv['all_cv_results'], serial['all_cv_results'], strict=True):
        for key in ('train_correct', 'test_correct'):
            assert parallel_fold[key] == serial_fold[key]
    replay = example.run_resumable_cv(model, X, y, grid, root / 'cv', cv=2,
                                      jobs=1, resume=True, refit_best=False)
    assert replay['work_summary']['executed_evaluations'] == 0
    assert replay['work_summary']['replayed_evaluations'] == 4
    assert replay['all_cv_results'] == cv['all_cv_results']
    assert not {'cvxpy', 'scs'}.intersection(sys.modules)


def _partial_start(failure_point):
    from joblib.externals.loky.backend.context import get_context
    from resumable_kernel_cv import _parallel_folds

    process_type = get_context('loky').Process
    original_start = process_type.start
    started = []

    def interrupted_start(process):
        second = len(started) == 1
        if second and failure_point == 'before-second-start':
            raise OSError('injected second worker start failure')
        original_start(process)
        started.append(process)
        if second:
            # The child exists, but Loky has not added it to its PID table.
            raise OSError('injected second worker start failure')

    try:
        with patch.object(process_type, 'start', interrupted_start), \
                pytest.raises(OSError, match='injected second worker start failure'):
            _parallel_folds([(), ()], workers=2, native_threads=1)
        assert len(started) == (1 if failure_point == 'before-second-start' else 2)
        assert not any(process.is_alive() for process in started), (
            'A fold worker survived the failed pool startup')
    finally:
        # A failing candidate must not leave its live children with pytest.
        for process in started:
            if process.is_alive():
                process.terminate()
        for process in started:
            process.join(timeout=10)
            assert not process.is_alive()


def _mark_worker_ready(root, solver_modules):
    (root / f'ready-{os.getpid()}').touch()


def _interrupt_pool_start(root, case):
    import psutil
    from joblib.externals.loky.backend.context import get_context
    import resumable_kernel_cv as example

    process_type = get_context('loky').Process
    original_popen = process_type._Popen
    launched = []
    interrupt = KeyboardInterrupt('interrupted during worker startup')
    received = []
    backend = None
    before_abort = case in ('launch-before-abort-call', 'registered-at-abort-entry')
    checked = False

    def handle_interrupt(signum, frame):
        failure = KeyboardInterrupt('interrupted during shutdown') if received else interrupt
        received.append(failure)
        raise failure

    def interrupt_second_launch(worker):
        popen = original_popen(worker)
        launched.append(psutil.Process(popen.pid))
        if len(launched) == 2:
            # The OS child exists, but BaseProcess.start has not received popen.
            # Let the child finish bootstrapping before dropping the parent frame.
            deadline = time.monotonic() + 60
            while not (root / f'ready-{popen.pid}').exists():
                if time.monotonic() > deadline:
                    raise TimeoutError('Second worker never reached its initializer')
                time.sleep(.02)
            # Python delivers SIGINT to the coordinator's main thread even if
            # this launcher runs on a different thread.
            if case != 'registered-at-abort-entry':
                signal.raise_signal(signal.SIGINT)
        return popen

    def capture_backend(instance, *args, **kwargs):
        nonlocal backend
        backend = instance
        count = configure(instance, *args, **kwargs)
        if case == 'registered-at-abort-entry':
            signal.raise_signal(signal.SIGINT)
        return count

    def before_abort_call(instance):
        # The interrupt precedes the real method, including when the coordinator
        # obtained its bound method before starting the workers.
        if len(received) == 1:
            signal.raise_signal(signal.SIGINT)
        return abort_workers(instance)

    def at_abort_entry(frame, event, arg):
        if event == 'call' and frame.f_code is abort_workers.__code__ and len(received) == 1:
            signal.raise_signal(signal.SIGINT)

    old_handler = signal.signal(signal.SIGINT, handle_interrupt)
    try:
        with ExitStack() as controls, \
                patch.object(process_type, '_Popen', staticmethod(interrupt_second_launch)), \
                patch.object(example, '_initialize_fold_worker', partial(_mark_worker_ready, root)), \
                pytest.raises(KeyboardInterrupt) as caught:
            if before_abort:
                backend_type = example._CheckpointFoldBackend
                configure = backend_type.configure
                abort_workers = backend_type.abort_workers
                controls.enter_context(patch.object(backend_type, 'configure', capture_backend))
                if case == 'launch-before-abort-call':
                    controls.enter_context(patch.object(backend_type, 'abort_workers', before_abort_call))
                else:
                    controls.callback(sys.setprofile, sys.getprofile())
                    sys.setprofile(at_abort_entry)
            if case == 'interrupt-during-launch-and-shutdown':
                from joblib.externals.loky.process_executor import _ExecutorFlags

                original_flag = _ExecutorFlags.flag_as_shutting_down
                shutdown_signal_sent = False

                def interrupt_before_flag(flags, *args, **kwargs):
                    nonlocal shutdown_signal_sent
                    if not shutdown_signal_sent:
                        shutdown_signal_sent = True
                        signal.raise_signal(signal.SIGINT)
                    return original_flag(flags, *args, **kwargs)

                controls.enter_context(patch.object(
                    _ExecutorFlags, 'flag_as_shutting_down', interrupt_before_flag))
            submit = controls.enter_context(patch.object(
                example._CheckpointExecutor, 'submit', autospec=True,
                side_effect=AssertionError('Fold submitted after startup interruption')))
            try:
                example._parallel_folds([(), ()], workers=2, native_threads=1)
            finally:
                # Observe the children on return, before even restoring the mocks.
                alive = [child.pid for child in launched if child.is_running()]
        assert len(received) == (1 if case == 'interrupt-during-launch' else 2)
        submit.assert_not_called()
        assert caught.value is interrupt and not alive, (
            f'Caller received {caught.value!r}; surviving fold workers: {alive}')
        assert len(launched) == 2
        checked = True
    finally:
        signal.signal(signal.SIGINT, old_handler)
        if backend is not None and not checked:
            # Release the waiting owner on the failing version, after observing its children.
            backend.abort_workers()
        for child in launched:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        _, survivors = psutil.wait_procs(launched, timeout=10)
        assert not survivors


def _interrupted_config_entry():
    from joblib import parallel_config
    from joblib.parallel import get_active_backend
    import resumable_kernel_cv as example

    enter_config = parallel_config.__init__
    interrupt = KeyboardInterrupt('interrupted after activating the private backend')

    def interrupt_after_activation(config, *args, **kwargs):
        enter_config(config, *args, **kwargs)
        if isinstance(kwargs.get('backend'), example._CheckpointFoldBackend):
            raise interrupt

    with parallel_config(backend='threading', n_jobs=3):
        caller_backend, caller_jobs = get_active_backend()
        with patch.object(parallel_config, '__init__', interrupt_after_activation), \
                patch.object(example.Parallel, '__call__') as dispatch, \
                pytest.raises(KeyboardInterrupt) as caught:
            example._parallel_folds([(), ()], workers=2, native_threads=1)
        backend, jobs = get_active_backend()
        dispatch.assert_not_called()
        assert caught.value is interrupt
        assert backend is caller_backend and jobs == caller_jobs == 3


def _interrupted_config_restore():
    from joblib import parallel_config
    from joblib.parallel import get_active_backend
    import resumable_kernel_cv as example

    detail = LookupError('fold diagnostic')
    first = ArithmeticError('fold failed')
    unregister = parallel_config.unregister
    interrupted = False

    def fail_dispatch(*args, **kwargs):
        try:
            raise detail
        except LookupError:
            raise first

    def interrupt_once(config):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt('interrupted while restoring the caller backend')
        return unregister(config)

    with parallel_config(backend='threading', n_jobs=3):
        caller_backend, caller_jobs = get_active_backend()
        with patch.object(example.Parallel, '__call__', fail_dispatch), \
                patch.object(parallel_config, 'unregister', interrupt_once), \
                pytest.raises((ArithmeticError, KeyboardInterrupt)) as caught:
            example._parallel_folds([(), ()], workers=2, native_threads=1)
        backend, jobs = get_active_backend()
        assert interrupted and caught.value is first and first.__context__ is detail
        assert backend is caller_backend and jobs == caller_jobs == 3


def _small_config_study(root):
    import numpy as np
    from dwd.gen_kern_dwd import KernGDWD
    import resumable_kernel_cv as example

    X = np.random.default_rng(91).normal(size=(20, 4))
    y = np.tile([-1, 1], 10)
    model = KernGDWD(kernel='rbf', kernel_kws={'gamma': .1}, random_state=2,
                     max_iter=3, stopping='fixed')
    return partial(example.run_resumable_cv, model, X, y, {'lambd': [.02]}, root, cv=2), len(y)


def _interrupted_refit_config(root, case):
    from joblib import parallel_config
    from joblib._parallel_backends import SequentialBackend
    from joblib.parallel import get_active_backend
    from dwd.gen_kern_dwd import KernGDWD
    import resumable_kernel_cv as example

    root = root / 'cv'
    run, full_size = _small_config_study(root)
    fit = KernGDWD.fit
    refits = []
    detail = ValueError('underlying fit diagnostic')
    fit_error = ArithmeticError('full-data fit failed')
    interrupt = KeyboardInterrupt(case)
    signals = []

    def observe_fit(estimator, features, labels, *args, **kwargs):
        if len(labels) != full_size:
            return fit(estimator, features, labels, *args, **kwargs)
        backend, jobs = get_active_backend()
        assert isinstance(backend, SequentialBackend) and jobs == 1
        try:
            if case == 'refit-exit-after-fit-error' and not refits:
                raise fit_error from detail
            return fit(estimator, features, labels, *args, **kwargs)
        finally:
            refits.append(estimator)

    def handle_interrupt(signum, frame):
        signals.append(signum)
        raise interrupt

    entry = case == 'refit-config-entry'
    hook = parallel_config.__init__ if entry else parallel_config.unregister

    def interrupt_boundary(frame, event, arg):
        if event != ('return' if entry else 'call') or frame.f_code is not hook.__code__:
            return
        if signals or (not entry and not refits):
            return
        if isinstance(get_active_backend()[0], SequentialBackend):
            signal.raise_signal(signal.SIGINT)

    with parallel_config(backend='threading', n_jobs=3, max_nbytes=12345) as caller, \
            patch.object(KernGDWD, 'fit', observe_fit):
        expected = caller.copy()
        prior_profile = sys.getprofile()
        prior_handler = signal.signal(signal.SIGINT, handle_interrupt)
        try:
            with pytest.raises((KeyboardInterrupt, ArithmeticError)) as caught:
                sys.setprofile(interrupt_boundary)
                run()
        finally:
            sys.setprofile(prior_profile)
            signal.signal(signal.SIGINT, prior_handler)

        receipts = {p: p.read_bytes() for p in (root / 'receipts').glob('fold-*/*.json')}
        assert len(receipts) == 2 and signals == [signal.SIGINT]
        assert len(refits) == int(not entry)
        primary = fit_error if case == 'refit-exit-after-fit-error' else interrupt
        assert caught.value is primary, f'Received {caught.value!r} instead of {primary!r}'
        if primary is fit_error:
            assert fit_error.__cause__ is detail
        with parallel_config() as actual:
            assert actual == expected and actual['backend'] is expected['backend']

        with patch.object(example, '_run_fold', side_effect=AssertionError('Completed fold reran')):
            resumed = run(resume=True)
        assert len(refits) == int(not entry) + 1 and resumed['best_clf'] is refits[-1]
        assert resumed['work_summary']['executed_evaluations'] == 0
        assert resumed['work_summary']['replayed_evaluations'] == 2
        assert receipts == {p: p.read_bytes() for p in (root / 'receipts').glob('fold-*/*.json')}
        with parallel_config() as actual:
            assert actual == expected and actual['backend'] is expected['backend']


def _check_live_caller(caller):
    from joblib.parallel import get_active_backend

    backend, jobs = get_active_backend()
    caller['n_jobs'] = 2
    after_backend, after_jobs = get_active_backend()
    assert backend is after_backend is caller['backend'] and (jobs, after_jobs) == (3, 2), (
        f'Caller changed n_jobs from 3 to 2; active values were {jobs} and {after_jobs}')


def _caller_joblib_config(root, case):
    from joblib import parallel_config
    from joblib._parallel_backends import SequentialBackend
    from joblib.parallel import get_active_backend
    from dwd.gen_kern_dwd import KernGDWD
    import resumable_kernel_cv as example

    root = root / 'cv'
    run, full_size = _small_config_study(root)
    fold_route = case.startswith('fold-')
    options = {'jobs': 2, 'refit_best': False} if fold_route else {}
    enter_return = case.endswith('enter-return')
    hook = parallel_config.__enter__ if enter_return else parallel_config.__init__
    interrupt = KeyboardInterrupt(case)
    signals = []
    refits = []
    fit = KernGDWD.fit

    def observe_fit(model, features, labels, *args, **kwargs):
        if len(labels) == full_size:
            refits.append(model)
        return fit(model, features, labels, *args, **kwargs)

    def handle_interrupt(signum, frame):
        signals.append(signum)
        raise interrupt

    def interrupt_boundary(frame, event, arg):
        if signals or event != 'return' or frame.f_code is not hook.__code__:
            return
        if not fold_route and len(list((root / 'receipts').glob('fold-*/*.json'))) != 2:
            return
        if enter_return:
            active, _ = get_active_backend()
            target = example._CheckpointFoldBackend if fold_route else SequentialBackend
            if not isinstance(active, target):
                return
        signal.raise_signal(signal.SIGINT)

    with parallel_config(backend='threading', n_jobs=3, max_nbytes=12345) as caller, \
            patch.object(KernGDWD, 'fit', observe_fit):
        success = case.endswith('success')
        if success:
            result = run(**options)
            assert result['work_summary']['executed_evaluations'] == 2
            assert len(refits) == int(not fold_route)
        else:
            prior_profile = sys.getprofile()
            prior_handler = signal.signal(signal.SIGINT, handle_interrupt)
            try:
                with pytest.raises(KeyboardInterrupt) as caught:
                    sys.setprofile(interrupt_boundary)
                    run(**options)
            finally:
                sys.setprofile(prior_profile)
                signal.signal(signal.SIGINT, prior_handler)
            assert caught.value is interrupt and signals == [signal.SIGINT] and not refits
        assert len(list((root / 'receipts').glob('fold-*/*.json'))) == (2 if success or not fold_route else 0)
        _check_live_caller(caller)


def _repeated_refit_config_interrupt(root):
    from joblib import parallel_config
    from joblib._parallel_backends import SequentialBackend
    from joblib.parallel import get_active_backend
    from dwd.gen_kern_dwd import KernGDWD
    import resumable_kernel_cv as example

    root = root / 'cv'
    run, _ = _small_config_study(root)
    run(refit_best=False)
    receipts = {p: p.read_bytes() for p in (root / 'receipts').glob('fold-*/*.json')}
    assert len(receipts) == 2
    initialize = parallel_config.__init__
    first = KeyboardInterrupt('interrupted while activating the refit backend')
    second = KeyboardInterrupt('interrupted before recovering the caller backend')
    signals = []

    def handle_interrupt(signum, frame):
        signals.append(signum)
        raise first if len(signals) == 1 else second

    def interrupt_activation(config, *args, **kwargs):
        initialize(config, *args, **kwargs)
        if not signals and isinstance(get_active_backend()[0], SequentialBackend):
            signal.raise_signal(signal.SIGINT)

    def before_recovery(frame, event, arg):
        if event == 'line' and len(signals) == 1 and isinstance(get_active_backend()[0], SequentialBackend):
            signal.raise_signal(signal.SIGINT)
        return before_recovery

    def watch_first(frame, event, arg):
        if frame.f_code.co_filename != example.__file__:
            return None
        # Target the first handler reached while the override is still live.
        # Frames that simply propagate the exception have no intervening line.
        if event == 'exception' and arg[1] is first:
            return before_recovery
        return watch_first

    with parallel_config(backend='threading', n_jobs=3) as caller, \
            patch.object(parallel_config, '__init__', interrupt_activation), \
            patch.object(KernGDWD, 'fit', side_effect=AssertionError('Unexpected fit')) as fit:
        prior_trace = sys.gettrace()
        prior_handler = signal.signal(signal.SIGINT, handle_interrupt)
        try:
            with pytest.raises(KeyboardInterrupt) as caught:
                sys.settrace(watch_first)
                run(resume=True)
        finally:
            sys.settrace(prior_trace)
            signal.signal(signal.SIGINT, prior_handler)
        fit.assert_not_called()
        assert signals == [signal.SIGINT, signal.SIGINT]
        assert receipts == {p: p.read_bytes() for p in (root / 'receipts').glob('fold-*/*.json')}
        assert caught.value is first, f'Caller received {caught.value!r}'
        _check_live_caller(caller)


def _interrupted_native_config(root, case):
    import numpy as np
    from scipy.linalg import eigh
    from threadpoolctl import threadpool_info, threadpool_limits
    from dwd.gen_kern_dwd import KernGDWD
    import resumable_kernel_cv as example

    # Load lazily initialized pools before capturing the caller's settings.
    eigh(np.diag([1., 2.]), driver='evd')
    directory = root / 'cv'
    entry = case == 'native-config-entry'
    hook = threadpool_limits._set_threadpool_limits if entry else threadpool_limits.restore_original_limits
    interrupt = KeyboardInterrupt(case)
    first = ArithmeticError('runtime fingerprint failed')
    signals = []

    def sizes():
        return {p['filepath']: p['num_threads'] for p in threadpool_info()}

    def handle_interrupt(signum, frame):
        signals.append(signum)
        raise interrupt

    def interrupt_boundary(frame, event, arg):
        if event != ('return' if entry else 'call') or frame.f_code is not hook.__code__:
            return
        if not signals and sizes() != caller_sizes:
            signal.raise_signal(signal.SIGINT)

    with threadpool_limits(limits=3), patch.object(example, 'runtime_fingerprint', side_effect=first) as compute:
        caller_sizes = sizes()
        assert 3 in caller_sizes.values(), 'Expected an adjustable native pool'
        prior_profile = sys.getprofile()
        prior_handler = signal.signal(signal.SIGINT, handle_interrupt)
        try:
            with pytest.raises((KeyboardInterrupt, ArithmeticError)) as caught:
                sys.setprofile(interrupt_boundary)
                example.run_resumable_cv(KernGDWD(random_state=2), np.eye(4), [-1, 1, -1, 1],
                                         {'lambd': [.02]}, directory, cv=2)
        finally:
            sys.setprofile(prior_profile)
            signal.signal(signal.SIGINT, prior_handler)
        actual = sizes()
        assert signals == [signal.SIGINT] and compute.call_count == int(not entry)
        assert not directory.exists()
        primary = interrupt if entry else first
        assert caught.value is primary and actual == caller_sizes, (
            f'Caller received {caught.value!r}, expected {primary!r}; '
            f'native pool sizes {actual}, expected {caller_sizes}')


def _interrupted_starter():
    import resumable_kernel_cv as example

    release = threading.Event()
    starters = []
    original_start = threading.Thread.start
    interrupt = KeyboardInterrupt('interrupted inside Thread.start')

    def interrupted_start(thread):
        run = thread.run

        def delayed_run():
            release.wait(timeout=10)
            run()

        thread.run = delayed_run
        original_start(thread)
        starters.append(thread)
        # The native thread exists and has not claimed worker startup.
        raise interrupt

    with patch.object(example._CheckpointExecutor, '_ensure_executor_running') as launch_workers, \
            patch.object(threading.Thread, 'start', interrupted_start):
        try:
            with pytest.raises(KeyboardInterrupt) as caught:
                example._parallel_folds([(), ()], workers=2, native_threads=1)
            assert caught.value is interrupt
        finally:
            release.set()
            for thread in starters:
                thread.join(timeout=10)
                assert not thread.is_alive()
        assert len(starters) == 1
        launch_workers.assert_not_called()


def _repeated_startup_interrupt():
    import resumable_kernel_cv as example

    claimed = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    first = KeyboardInterrupt('initial startup wait')
    drain_interrupted = False

    def launch_workers():
        claimed.set()
        if not release.wait(timeout=10):
            raise TimeoutError('Startup owner was never released')
        finished.set()

    class InterruptedWait(example.Future):
        def result(self):
            assert claimed.wait(timeout=10)
            raise first

        def exception(self):
            nonlocal drain_interrupted
            if not drain_interrupted:
                drain_interrupted = True
                raise KeyboardInterrupt('again while waiting for startup cleanup')
            release.set()
            return super().exception(timeout=10)

    with patch.object(example._CheckpointExecutor, '_ensure_executor_running', side_effect=launch_workers), \
            patch.object(example, 'Future', InterruptedWait):
        try:
            with pytest.raises(KeyboardInterrupt) as caught:
                example._parallel_folds([(), ()], workers=2, native_threads=1)
            assert caught.value is first
            assert drain_interrupted and finished.is_set()
        finally:
            release.set()


def _failing_fold(root, fold):
    (root / f'entered-{fold}-{os.getpid()}').touch()
    if fold == 0:
        time.sleep(60)
        (root / 'hold-expired').touch()
        raise TimeoutError('The held fold outlived the failure check')
    deadline = time.monotonic() + 60
    while not any(root.glob('entered-0-*')):
        if time.monotonic() > deadline:
            raise TimeoutError('The other fold never entered')
        time.sleep(.02)
    raise ArithmeticError('injected failure inside a running fold')


def _managed_failure(root, condition):
    import psutil
    from joblib.externals.loky.backend import utils
    from joblib.externals.loky.backend.context import get_context
    import resumable_kernel_cv as example

    started = []
    process_type = get_context('loky').Process
    original_start = process_type.start
    original_check_output = subprocess.check_output

    def observe_start(process):
        original_start(process)
        started.append(process)

    def without_tree_tools(command, *args, **kwargs):
        if command[0] in ('pgrep', 'taskkill'):
            raise FileNotFoundError(command[0])
        return original_check_output(command, *args, **kwargs)

    try:
        with ExitStack() as controls, warnings.catch_warnings(), \
                patch.object(threading, 'excepthook') as thread_error, \
                patch.object(process_type, 'start', observe_start), \
                patch.object(example, '_run_fold', _failing_fold), \
                pytest.raises(ArithmeticError, match='^injected failure inside a running fold$'):
            if condition == 'access-denied':
                lookup = controls.enter_context(patch.object(
                    psutil.Process, 'children', side_effect=psutil.AccessDenied()))
            else:
                controls.enter_context(patch.object(utils, 'psutil', None))
                controls.enter_context(patch.object(subprocess, 'check_output', without_tree_tools))
                warnings.filterwarnings('error', message='Failed to kill subprocesses on this platform')
            example._parallel_folds([(root, 0), (root, 1)], workers=2, native_threads=1)
        thread_error.assert_not_called()
        if condition == 'access-denied':
            lookup.assert_called()
        markers = [next(root.glob(f'entered-{fold}-*')) for fold in (0, 1)]
        pids = {int(marker.name.rsplit('-', 1)[1]) for marker in markers}
        assert len(started) == len(pids) == 2 and pids == {p.pid for p in started}
        assert not (root / 'hold-expired').exists()
        if os.name == 'posix':
            # Query before is_alive(), which can reap a zombie as a side effect.
            assert not any(psutil.pid_exists(pid) for pid in pids)
        assert all(not p.is_alive() and p.exitcode is not None for p in started), (
            'A fold worker survived propagation of the original fold failure')
    finally:
        # Retain the liveness observation above before cleaning a failing version.
        for process in started:
            if process.is_alive():
                process.terminate()
        for process in started:
            process.join(timeout=10)
            assert not process.is_alive()


def _run_child(tmp_path, case):
    import psutil

    # A regression can deadlock the manager and coordinator. A file avoids
    # blocking on inherited pipe handles if a descendant has to be killed.
    log_path = tmp_path / (case + '.log')
    with log_path.open('wb') as log:
        child = psutil.Popen(
            [sys.executable, '-B', str(Path(__file__).resolve()),
             json.dumps(sys.path), str(tmp_path), case], stdout=log, stderr=subprocess.STDOUT)
        try:
            code = child.wait(timeout=180)
        finally:
            if child.poll() is None:
                descendants = child.children(recursive=True)
                for process in reversed(descendants):
                    try:
                        process.kill()
                    except psutil.NoSuchProcess:
                        pass
                child.kill()
                child.wait(timeout=10)
                psutil.wait_procs(descendants, timeout=10)
    assert code == 0, log_path.read_text(encoding='utf-8', errors='replace')


def test_parallel_cv_overlaps_an_ordinary_loky_pool(tmp_path):
    _run_child(tmp_path, 'overlap')


@pytest.mark.parametrize('failure_point', ['before-second-start', 'after-second-start'])
def test_partial_pool_start_leaves_no_fold_workers(tmp_path, failure_point):
    _run_child(tmp_path, failure_point)


def test_sigint_during_worker_launch_reaps_all_fold_workers(tmp_path):
    _run_child(tmp_path, 'interrupt-during-launch')


def test_sigint_during_startup_shutdown_reaps_workers_and_preserves_first_exception(tmp_path):
    _run_child(tmp_path, 'interrupt-during-launch-and-shutdown')


@pytest.mark.parametrize('case', ['launch-before-abort-call', 'registered-at-abort-entry'])
def test_repeated_sigint_at_abort_boundary_preserves_first_error_and_reaps_workers(tmp_path, case):
    _run_child(tmp_path, case)


def test_interrupted_starter_cannot_launch_workers_after_return(tmp_path):
    _run_child(tmp_path, 'interrupted-starter')


def test_interrupt_restoring_joblib_config_preserves_fold_error_and_caller_backend(tmp_path):
    _run_child(tmp_path, 'interrupted-config-restore')


def test_interrupt_activating_joblib_config_restores_caller_before_dispatch(tmp_path):
    _run_child(tmp_path, 'interrupted-config-entry')


@pytest.mark.parametrize('case', ['refit-config-entry', 'refit-config-exit', 'refit-exit-after-fit-error'])
def test_final_refit_interrupt_restores_caller_and_replays_committed_folds(tmp_path, case):
    _run_child(tmp_path, case)


@pytest.mark.parametrize('case', ['fold-success', 'refit-success', 'fold-config-construction',
                                  'refit-config-construction', 'fold-config-enter-return',
                                  'refit-config-enter-return'])
def test_public_cv_keeps_caller_joblib_config_live(tmp_path, case):
    _run_child(tmp_path, case)


def test_repeated_sigint_during_refit_config_activation_restores_live_caller(tmp_path):
    _run_child(tmp_path, 'refit-repeated-config-interrupt')


@pytest.mark.parametrize('case', ['native-config-entry', 'native-config-exit'])
def test_native_limit_interrupt_restores_caller_and_preserves_first_error(tmp_path, case):
    _run_child(tmp_path, case)


def test_repeated_startup_interrupt_preserves_first_exception(tmp_path):
    _run_child(tmp_path, 'repeated-startup-interrupt')


def test_fold_failure_reaps_workers_without_process_tree_tools(tmp_path):
    _run_child(tmp_path, 'managed-failure')


def test_fold_failure_reaps_workers_when_child_lookup_is_denied(tmp_path):
    _run_child(tmp_path, 'access-denied')


if __name__ == '__main__':
    sys.path[:] = json.loads(sys.argv[1])
    if sys.argv[3] == 'overlap':
        _overlap(Path(sys.argv[2]))
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
        if sys.argv[3] in ('managed-failure', 'access-denied'):
            _managed_failure(Path(sys.argv[2]), sys.argv[3])
        elif sys.argv[3] in ('interrupt-during-launch', 'interrupt-during-launch-and-shutdown',
                             'launch-before-abort-call', 'registered-at-abort-entry'):
            _interrupt_pool_start(Path(sys.argv[2]), sys.argv[3])
        elif sys.argv[3] == 'interrupted-starter':
            _interrupted_starter()
        elif sys.argv[3] == 'interrupted-config-restore':
            _interrupted_config_restore()
        elif sys.argv[3] == 'interrupted-config-entry':
            _interrupted_config_entry()
        elif sys.argv[3] in ('refit-config-entry', 'refit-config-exit', 'refit-exit-after-fit-error'):
            _interrupted_refit_config(Path(sys.argv[2]), sys.argv[3])
        elif sys.argv[3] in ('fold-success', 'refit-success', 'fold-config-construction',
                             'refit-config-construction', 'fold-config-enter-return', 'refit-config-enter-return'):
            _caller_joblib_config(Path(sys.argv[2]), sys.argv[3])
        elif sys.argv[3] == 'refit-repeated-config-interrupt':
            _repeated_refit_config_interrupt(Path(sys.argv[2]))
        elif sys.argv[3] in ('native-config-entry', 'native-config-exit'):
            _interrupted_native_config(Path(sys.argv[2]), sys.argv[3])
        elif sys.argv[3] == 'repeated-startup-interrupt':
            _repeated_startup_interrupt()
        else:
            _partial_start(sys.argv[3])
