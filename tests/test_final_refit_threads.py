"""Final-refit thread limits in the standalone kernel CV example."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
import inspect
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from joblib import effective_n_jobs, parallel_config
from sklearn.metrics import pairwise
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import ThreadpoolController, threadpool_info, threadpool_limits

from dwd.gen_kern_dwd import KernGDWD

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import resumable_kernel_cv as example
import _cv_checkpoint as checkpoint
import tune_kernel_dwd as command

from _isolated_cv_test_support import (
    _fixture_oracle_is_loaded, run_isolated_test,
)

_ISOLATED_CHILD = False
def counts():
    return {str(pool['filepath']): pool['num_threads'] for pool in threadpool_info()}


def assert_counts(test, expected):
    observed = counts()
    test.assertTrue(observed)
    test.assertEqual(set(observed.values()), {expected})


def requires_adjustable_pools(method):
    method.requires_adjustable_pools = True
    return method


rng = np.random.RandomState(6100604)
X = rng.normal(size=(24, 4)) * [1., 3., .5, 2.]
y = np.tile([-1, 1], 12)
Q = rng.normal(size=(7, 4))
FOLDS = list(StratifiedKFold(2, shuffle=True, random_state=19).split(X, y))
PARAMS = {'lambd': [.05, .2]}


def model():
    return KernGDWD(kernel='rbf', kernel_kws={'gamma': .12}, random_state=31,
                    stopping='fixed', max_iter=3)


def run_example(path, **options):
    estimator = options.pop('clf', model())
    settings = dict(cv=FOLDS, jobs=1, scale=True)
    settings.update(options)
    return example.run_resumable_cv(estimator, X, y, PARAMS, path, **settings)


def receipts(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in sorted(path.glob('receipts/fold-*/*.json'))}


class FinalRefitThreadTests(unittest.TestCase):
    def run(self, result=None):
        if not _ISOLATED_CHILD:
            method = getattr(self, self._testMethodName, None)
            # Some SCS wheels load a single-threaded OpenBLAS at collection.
            # Exercise the adjustable-pool tests on the actual clean runtime.
            fixed_pool_loaded = (getattr(method, 'requires_adjustable_pools', False)
                                 and sys.modules.get('scs') is not None
                                 and any(p.get('threading_layer') == 'disabled'
                                         for p in threadpool_info()))
            if fixed_pool_loaded or _fixture_oracle_is_loaded():
                return run_isolated_test(self, result, module_path=__file__)
        return super().run(result)

    def setUp(self):
        # Load lazily initialized native pools before capturing thread limits.
        # Limits can only restore libraries present when their scope starts.
        from scipy.linalg import eigh
        eigh(np.diag([1., 2.]), driver='evd')
        resources = ExitStack()
        self.addCleanup(resources.close)
        resources.enter_context(threadpool_limits(1))
        self.path = Path(resources.enter_context(tempfile.TemporaryDirectory())) / 'run'

    def check_provenance(self, result, cv_threads=1, performed=True):
        final = result['work_summary']['final_refit']
        self.assertEqual(final['native_threads_requested'], 1)
        self.assertEqual(final['estimator_workers'], 1 if performed else 0)
        self.assertEqual(final['performed'], performed)
        self.assertEqual(final['fresh_model'], performed)
        if performed:
            self.assertEqual({p['num_threads'] for p in final['runtime']['native_pools']}, {1})
            self.assertIn('accelerator', final['runtime'])
            self.assertIn('files_sha256', final['runtime'])
        else:
            self.assertIsNone(final['runtime'])
        if 'native_threads_per_worker' in result['work_summary']:
            self.assertEqual(result['work_summary']['native_threads_per_worker'], cv_threads)

    @requires_adjustable_pools
    def test_persistent_native_failure_restores_other_libraries(self):
        controller = ThreadpoolController()
        pools = controller.lib_controllers
        for index, pool in enumerate(pools):
            pool.set_num_threads(3 + index % 2)
        original = [pool.get_num_threads() for pool in pools]
        adjustable = [index for index, count in enumerate(original) if count is not None and count > 1]
        if len(adjustable) < 2:
            self.skipTest('Requires two independently adjustable native libraries')
        first, middle = adjustable[0], adjustable[len(adjustable) // 2]
        for blocked in ((first,), (middle,), (first, middle)):
            with self.subTest(blocked=blocked), ExitStack() as patches:
                for pool, count in zip(pools, original):
                    pool.set_num_threads(count)
                errors = []
                for index in blocked:
                    def reject(count, setter=pools[index].set_num_threads, index=index):
                        if count == original[index]:
                            error = RuntimeError(f'Persistent failure in {pools[index].filepath}')
                            errors.append((index, error))
                            raise error
                        setter(count)

                    patches.enter_context(patch.object(pools[index], 'set_num_threads', reject))
                patches.enter_context(patch.object(example, 'ThreadpoolController', return_value=controller))
                interrupted = KeyboardInterrupt('operation interrupted')
                with patch.object(example, 'runtime_fingerprint', side_effect=interrupted) as operation:
                    with self.assertRaises(RuntimeError) as caught:
                        run_example(self.path)
                operation.assert_called_once()
                actual = [pool.get_num_threads() for pool in pools]
                for index, count in enumerate(original):
                    if index not in blocked:
                        self.assertEqual(actual[index], count, pools[index].filepath)
                chain = []
                error = caught.exception
                while error is not None:
                    self.assertNotIn(error, chain)
                    chain.append(error)
                    error = error.__context__
                self.assertIn(interrupted, chain)
                self.assertIs(caught.exception, errors[-1][1])
                for index in blocked:
                    failures = [error for i, error in errors if i == index]
                    self.assertGreater(len(failures), 0)
                    self.assertLessEqual(len(failures), 4)
                    self.assertIn(failures[-1], chain)

    def test_default_and_explicit_one_keep_original_runtime_scope(self):
        self.assertEqual(inspect.signature(example.run_resumable_cv)
                         .parameters['final_native_threads'].default, 1)
        records = []
        for name, options in (('omitted', {}), ('explicit', {'final_native_threads': 1})):
            limits, guards = [], []
            original_limits, original_guard = example._native_thread_limit, example._check_runtime

            def observe_limits(*args, **kwargs):
                limits.append(kwargs.get('limits', args[0] if args else None))
                return original_limits(*args, **kwargs)

            def observe_guard(*args, **kwargs):
                guards.append(dict(kwargs))
                self.assertNotIn('native_threads', kwargs)
                return original_guard(*args, **kwargs)

            with patch.object(example, '_native_thread_limit', observe_limits), \
                    patch.object(example, '_check_runtime', observe_guard):
                result = run_example(self.path / name, **options)
            self.check_provenance(result, 1)
            self.assertTrue(limits)
            self.assertEqual(set(limits), {1})
            # Preserve the three original parent guard calls around final fit.
            self.assertEqual(guards[-3:], [{}, {'check_inventory': False}, {}])
            records.append((limits, guards))
        self.assertEqual(records[0], records[1])

    @requires_adjustable_pools
    def test_parallel_cv_serial_final_fit_and_caller_restored(self):
        observed = []
        pairwise_workers = []
        original_pairwise = pairwise._parallel_pairwise
        original_fit, original_init = KernGDWD.fit, KernGDWD.cv_init
        original_scale, original_score = StandardScaler.fit_transform, example._fit_and_score
        original_publish = example._publish_summaries

        def fit(estimator, data, labels, *a, **k):
            observed.append(('fit', len(data), counts()))
            return original_fit(estimator, data, labels, *a, **k)

        def initialize(estimator, data, *a, **k):
            observed.append(('prepare', len(data), counts()))
            return original_init(estimator, data, *a, **k)

        def scale(scaler, data, *a, **k):
            observed.append(('scale', len(data), counts()))
            return original_scale(scaler, data, *a, **k)

        def score(*a, **k):
            observed.append(('cv-fit-score', 12, counts()))
            return original_score(*a, **k)

        def publish(*a, **k):
            observed.append(('publish', 0, counts()))
            return original_publish(*a, **k)

        def kernel(X, Y, func, n_jobs, **kwargs):
            pairwise_workers.append((len(X), n_jobs, effective_n_jobs(n_jobs)))
            return original_pairwise(X, Y, func, n_jobs, **kwargs)

        estimator = model().set_params(kernel_kws={'gamma': .12, 'n_jobs': 2})
        with threadpool_limits(3), parallel_config(backend='threading', n_jobs=3), \
                patch.object(pairwise, '_parallel_pairwise', kernel), \
                patch.object(KernGDWD, 'fit', fit), \
                patch.object(KernGDWD, 'cv_init', initialize), \
                patch.object(StandardScaler, 'fit_transform', scale), \
                patch.object(example, '_fit_and_score', score), \
                patch.object(example, '_publish_summaries', publish):
            before = counts()
            result = run_example(self.path, clf=estimator, cv_native_threads=2,
                                 final_native_threads=np.int64(1))
            self.assertEqual(counts(), before)
            self.assertEqual(effective_n_jobs(None), 3)
            training_workers = pairwise_workers.copy()
            pairwise_workers.clear()
            query = result['scaler'].transform(Q)
            fitted = result['best_clf']
            parallel_labels = fitted.predict(query)
            self.assertEqual(pairwise_workers, [(len(y), 2, 2)])
            pairwise_workers.clear()
            with parallel_config(backend='sequential', n_jobs=1), threadpool_limits(1):
                serial_labels = fitted.predict(query)
            self.assertEqual(pairwise_workers, [(len(y), 2, 1)])
            np.testing.assert_array_equal(parallel_labels, serial_labels)
        for stage, rows, pools in observed:
            self.assertEqual(set(pools.values()), {1 if rows == len(y) else 2}, (stage, rows, pools))
        self.assertIn((len(y), 2, 1), training_workers)
        self.assertIn((len(y) // 2, 2, 2), training_workers)
        self.assertTrue(all(workers == 1 for rows, _, workers in training_workers if rows == len(y)))
        self.assertEqual(result['best_clf'].kernel_kws_, estimator.kernel_kws)
        self.assertTrue(any(stage == 'publish' for stage, _, _ in observed))
        self.check_provenance(result, 2)
        stored = json.loads((self.path / 'cv-results.json').read_text())
        self.assertEqual(stored['final_refit'], result['work_summary']['final_refit'])

    def test_resume_fits_fresh_serial_model_and_preserves_cv_receipts(self):
        first = run_example(self.path)
        saved = receipts(self.path)
        identity = (self.path / 'identity.json').read_bytes()
        with threadpool_limits(3), patch.object(
                example, '_fit_and_score', side_effect=AssertionError('CV must replay')):
            result = run_example(self.path, resume=True)
        self.assertIsNot(first['best_clf'], result['best_clf'])
        self.assertEqual(receipts(self.path), saved)
        self.assertEqual((self.path / 'identity.json').read_bytes(), identity)
        self.assertEqual(result['work_summary']['executed_evaluations'], 0)
        self.assertEqual(result['work_summary']['replayed_evaluations'], 4)
        self.check_provenance(result)

    @requires_adjustable_pools
    def test_cv_only_does_not_enter_final_scope(self):
        limits = []
        original = example._native_thread_limit

        def observed(*a, **k):
            limits.append(k.get('limits', a[0] if a else None))
            return original(*a, **k)

        with patch.object(example, '_native_thread_limit', observed):
            result = run_example(self.path, refit_best=False, cv_native_threads=2)
        self.assertTrue(limits)
        self.assertEqual(set(limits), {2})
        self.assertIsNone(result['best_clf'])
        self.check_provenance(result, 2, performed=False)

    def test_two_worker_cv_keeps_one_thread_identity_with_serial_final(self):
        with threadpool_limits(3):
            before = counts()
            result = run_example(self.path, jobs=2)
            self.assertEqual(counts(), before)
        self.assertEqual(result['work_summary']['fold_workers'], 2)
        identity = json.loads((self.path / 'identity.json').read_text())['payload']['identity']
        self.assertEqual(identity['native_threads'], 1)
        self.assertEqual({p['num_threads'] for p in identity['runtime']['native_pools']}, {1})
        self.check_provenance(result)
        saved = receipts(self.path)
        with patch.object(example, '_fit_and_score', side_effect=AssertionError('CV must replay')):
            replayed = run_example(self.path, jobs=1, resume=True, final_native_threads=1)
        self.assertEqual(receipts(self.path), saved)
        self.assertEqual(replayed['work_summary']['fold_workers'], 0)
        self.check_provenance(replayed, 1)

    def test_invalid_values_fail_before_directory_or_fit(self):
        invalid = (True, np.bool_(True), 0, -1, None, 1.5, '2')
        for value in invalid:
            with self.subTest(value=repr(value)), patch.object(
                    KernGDWD, 'fit', side_effect=AssertionError('unexpected fit')):
                with self.assertRaises(ValueError):
                    run_example(self.path, final_native_threads=value)
                self.assertFalse(self.path.exists())
                with self.assertRaises(ValueError):
                    checkpoint.runtime_fingerprint(expected_native_threads=value)
        with self.assertRaisesRegex(ValueError, 'final_native_threads must be 1'):
            run_example(self.path, final_native_threads=2)
        self.assertFalse(self.path.exists())

    @requires_adjustable_pools
    def test_final_failures_restore_caller_and_resume_completed_cv(self):
        run_example(self.path, refit_best=False, cv_native_threads=2)
        saved = receipts(self.path)
        for stage in ('scale', 'fit', 'runtime'):
            for error_type in (RuntimeError, KeyboardInterrupt):
                original = {'scale': StandardScaler.fit_transform, 'fit': KernGDWD.fit,
                            'runtime': example._check_runtime}[stage]

                def fail(*a, **k):
                    final = (k.get('check_inventory') is False and 'native_threads' not in k
                             if stage == 'runtime' else len(a[1]) == len(y))
                    if final:
                        assert_counts(self, 1)
                        self.assertEqual(effective_n_jobs(3), 1)
                        raise error_type('manufactured final failure')
                    return original(*a, **k)

                target, name = {'scale': (StandardScaler, 'fit_transform'),
                                'fit': (KernGDWD, 'fit'), 'runtime': (example, '_check_runtime')}[stage]
                with self.subTest(stage=stage, error=error_type.__name__), threadpool_limits(3), \
                        parallel_config(backend='threading', n_jobs=3):
                    before = counts()
                    with patch.object(target, name, fail), patch.object(
                            example, '_fit_and_score', side_effect=AssertionError('CV must replay')):
                        with self.assertRaisesRegex(error_type, 'manufactured final failure'):
                            run_example(self.path, resume=True, cv_native_threads=2)
                    self.assertEqual(counts(), before)
                    self.assertEqual(effective_n_jobs(None), 3)
                    self.assertEqual(receipts(self.path), saved)
                    result = run_example(self.path, resume=True, cv_native_threads=2)
                    self.assertEqual(counts(), before)
                    self.check_provenance(result, 2)

    @requires_adjustable_pools
    def test_diagnostic_and_summary_failures_keep_final_runtime(self):
        for stage in ('diagnostics', 'summary'):
            original = example._diagnostics

            def fail_diagnostics(estimator):
                if estimator.dual_coef_.shape[1] == len(y):
                    raise RuntimeError('manufactured reporting failure')
                return original(estimator)

            context = (patch.object(example, '_diagnostics', fail_diagnostics) if stage == 'diagnostics'
                       else patch.object(example, '_work_summary', side_effect=RuntimeError('manufactured reporting failure')))
            with self.subTest(stage=stage), warnings.catch_warnings(), context, threadpool_limits(3):
                warnings.simplefilter('error', RuntimeWarning)
                result = run_example(self.path / stage, cv_native_threads=2)
                assert_counts(self, 3)
                self.check_provenance(result, 2)
                self.assertIsNotNone(result['best_clf'])
                if stage == 'summary':
                    self.assertEqual(result['work_summary']['coverage'], 'unavailable')

    @requires_adjustable_pools
    def test_runtime_count_guard_default_and_phase_identity_remain_strict(self):
        with threadpool_limits(2):
            with self.assertRaises(checkpoint.CheckpointError):
                checkpoint.runtime_fingerprint()
            observed = checkpoint.runtime_fingerprint(expected_native_threads=2)
            self.assertEqual({p['num_threads'] for p in observed['native_pools']}, {2})
            with self.assertRaises(checkpoint.CheckpointError):
                checkpoint.runtime_fingerprint(expected_native_threads=3)
            for field in ('files_sha256', 'accelerator', 'native_pools'):
                different = copy.deepcopy(observed)
                different[field] = 'different'
                with self.subTest(field=field), self.assertRaises(checkpoint.CheckpointError):
                    example._check_runtime(different, native_threads=2)

    def test_two_thread_cv_refuses_one_fixed_pool_before_opening_run(self):
        fixed = self.path.parent / 'fixed-native-pool'
        fixed.write_bytes(b'synthetic single-threaded library')
        original_info = checkpoint.ThreadpoolController.info

        def mixed_counts(controller):
            adjustable = [dict(pool, num_threads=2) for pool in original_info(controller)]
            return [*adjustable, dict(filepath=str(fixed), user_api='blas', num_threads=1)]

        with patch.object(checkpoint.ThreadpoolController, 'info', mixed_counts), patch.object(
                KernGDWD, 'fit', side_effect=AssertionError('unexpected fit')):
            with self.assertRaisesRegex(checkpoint.CheckpointError, 'thread count differs'):
                run_example(self.path, cv_native_threads=2)
        self.assertFalse(self.path.exists())

    def test_cli_validates_and_forwards_final_threads(self):
        common = ['--data', 'missing.npz', '--run-dir', 'unused', '--lambd', '.1', '--gamma', '.2']
        with redirect_stderr(io.StringIO()):
            for value in ('0', '2', '-1', '1.5', 'True'):
                with self.subTest(invalid=value), self.assertRaises(SystemExit) as caught:
                    command.main(common + ['--final-native-threads', value])
                self.assertEqual(caught.exception.code, 2)
        self.path.mkdir(parents=True)
        np.savez(self.path / 'data.npz', X=X, y=y)
        common[1] = str(self.path / 'data.npz')
        for extra in ([], ['--final-native-threads', '1', '--cv-only']):
            with self.subTest(extra=extra), patch.object(command, 'run_resumable_cv', return_value={
                    'best_score': .5, 'best_params': {}, 'work_summary': {},
                    'finite_grid_report': {
                        'status': 'unavailable',
                        'scope': 'best_among_supplied_candidates',
                        'reporting_error': 'RuntimeError'}}) as called, redirect_stdout(io.StringIO()):
                command.main(common + extra)
            self.assertEqual(called.call_args.kwargs['final_native_threads'], 1)


if __name__ == '__main__':
    unittest.main()
