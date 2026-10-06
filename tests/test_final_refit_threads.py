"""Final-refit thread limits in the standalone kernel CV example."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
import inspect
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info, threadpool_limits

from dwd.gen_kern_dwd import KernGDWD

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import resumable_kernel_cv as example
import _cv_checkpoint as checkpoint
import tune_kernel_dwd as command

_ISOLATED_CHILD = False
_ISOLATED_TEST_SCRIPT = r'''
import importlib.util
import json
from pathlib import Path
import sys
import unittest

sys.path[:] = json.loads(sys.argv[5])
spec = importlib.util.spec_from_file_location('_isolated_resumable_cv_tests', sys.argv[2])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module._ISOLATED_CHILD = True
case = getattr(module, sys.argv[3])(sys.argv[4])
result = unittest.TestResult()
case.run(result)
payload = {
    'tests_run': result.testsRun,
    'successful': result.wasSuccessful(),
    'failures': [(str(test), detail) for test, detail in result.failures],
    'errors': [(str(test), detail) for test, detail in result.errors],
    'skipped': [(str(test), reason) for test, reason in result.skipped],
    'expected_failures': [(str(test), detail) for test, detail in result.expectedFailures],
    'unexpected_successes': [str(test) for test in result.unexpectedSuccesses],
}
Path(sys.argv[1]).write_text(json.dumps(payload, allow_nan=False), encoding='utf-8')
raise SystemExit(0 if result.wasSuccessful() else 1)
'''


def _fixture_oracle_is_loaded():
    # Other numerical tests intentionally import historical implementations
    # under dwd.* aliases. A resumable run must reject that process; exercise
    # this example in a fresh interpreter without relaxing its runtime guard.
    for name, module in tuple(sys.modules.items()):
        if not name.startswith('dwd.'):
            continue
        source = str(getattr(module, '__file__', '')).replace('\\', '/').lower()
        if '/tests/fixtures/' in source:
            return True
    return False


def counts():
    return {str(pool['filepath']): pool['num_threads'] for pool in threadpool_info()}


def assert_counts(test, expected):
    observed = counts()
    test.assertTrue(observed)
    test.assertEqual(set(observed.values()), {expected})


rng = np.random.RandomState(6100604)
X = rng.normal(size=(24, 4)) * [1., 3., .5, 2.]
y = np.tile([-1, 1], 12)
Q = rng.normal(size=(7, 4))
FOLDS = list(StratifiedKFold(2, shuffle=True, random_state=19).split(X, y))
PARAMS = {'lambd': [.05, .2]}


def model(implementation='optimized'):
    return KernGDWD(kernel='rbf', kernel_kws={'gamma': .12}, random_state=31,
                    stopping='fixed', max_iter=3, implementation=implementation)


def run_example(path, **options):
    estimator = options.pop('clf', model())
    settings = dict(cv=FOLDS, jobs=1, scale=True)
    settings.update(options)
    return example.run_resumable_cv(estimator, X, y, PARAMS, path, **settings)


def receipts(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in sorted(path.glob('receipts/fold-*/*.json'))}


class FinalRefitThreadTests(unittest.TestCase):
    def run(self, result=None):
        if _ISOLATED_CHILD or not _fixture_oracle_is_loaded():
            return super().run(result)
        own_result = result is None
        if own_result:
            result = self.defaultTestResult()
            result.startTestRun()
        result.startTest(self)
        try:
            with tempfile.TemporaryDirectory(prefix='dwd-cv-isolated-') as directory:
                receipt = Path(directory) / 'result.json'
                process = subprocess.run(
                    [sys.executable, '-B', '-X', 'utf8', '-c', _ISOLATED_TEST_SCRIPT,
                     str(receipt), str(Path(__file__).resolve()),
                     type(self).__name__, self._testMethodName, json.dumps(sys.path)],
                    capture_output=True, text=True, encoding='utf-8', errors='replace',
                    timeout=300, check=False)
                if process.stdout:
                    sys.stdout.write(process.stdout)
                if process.stderr:
                    sys.stderr.write(process.stderr)
                output = '\n'.join(part for part in (process.stdout, process.stderr) if part)
                if not receipt.is_file():
                    raise RuntimeError('Isolated test produced no result receipt; exit '
                                       + str(process.returncode) + '\n' + output)
                child = json.loads(receipt.read_text(encoding='utf-8'))
                if (child['tests_run'] != 1 or type(child['successful']) is not bool
                        or process.returncode != (0 if child['successful'] else 1)):
                    raise RuntimeError('Isolated test count or exit status is inconsistent: '
                                       + repr(child) + '\n' + output)
                bad = bool(child['failures'] or child['errors']
                           or child['unexpected_successes'])
                if child['successful'] == bad:
                    raise RuntimeError('Isolated test outcome is inconsistent: '
                                       + repr(child) + '\n' + output)
                reported = False
                for key, exception_type, report in (
                        ('failures', AssertionError, result.addFailure),
                        ('errors', RuntimeError, result.addError),
                        ('expected_failures', AssertionError, result.addExpectedFailure)):
                    for description, detail in child[key]:
                        error = exception_type('Isolated ' + description + '\n' + detail
                                               + ('\n' + output if output else ''))
                        report(self, (exception_type, error, None))
                        reported = True
                for description, reason in child['skipped']:
                    result.addSkip(self, description + ': ' + reason)
                    reported = True
                for description in child['unexpected_successes']:
                    result.addUnexpectedSuccess(self)
                    reported = True
                if not reported:
                    result.addSuccess(self)
        except Exception:
            result.addError(self, sys.exc_info())
        finally:
            result.stopTest(self)
            if own_result:
                result.stopTestRun()
        return result

    def setUp(self):
        resources = ExitStack()
        self.addCleanup(resources.close)
        resources.enter_context(threadpool_limits(1))
        self.path = Path(resources.enter_context(tempfile.TemporaryDirectory())) / 'run'

    def check_provenance(self, result, requested, performed=True):
        final = result['work_summary']['final_refit']
        self.assertEqual(final['native_threads_requested'], requested)
        self.assertEqual(final['performed'], performed)
        self.assertEqual(final['fresh_model'], performed)
        if performed:
            self.assertEqual({p['num_threads'] for p in final['runtime']['native_pools']}, {requested})
            self.assertIn('accelerator', final['runtime'])
            self.assertIn('files_sha256', final['runtime'])
        else:
            self.assertIsNone(final['runtime'])
        if 'native_threads_per_worker' in result['work_summary']:
            self.assertEqual(result['work_summary']['native_threads_per_worker'], 1)

    def test_default_and_explicit_one_keep_original_runtime_scope(self):
        self.assertEqual(inspect.signature(example.run_resumable_cv)
                         .parameters['final_native_threads'].default, 1)
        records = []
        for name, options in (('omitted', {}), ('explicit', {'final_native_threads': 1})):
            limits, guards = [], []
            original_limits, original_guard = example.threadpool_limits, example._check_runtime

            def observe_limits(*args, **kwargs):
                limits.append(kwargs.get('limits', args[0] if args else None))
                return original_limits(*args, **kwargs)

            def observe_guard(*args, **kwargs):
                guards.append(dict(kwargs))
                self.assertNotIn('native_threads', kwargs)
                return original_guard(*args, **kwargs)

            with patch.object(example, 'threadpool_limits', observe_limits), \
                    patch.object(example, '_check_runtime', observe_guard):
                result = run_example(self.path / name, **options)
            self.check_provenance(result, 1)
            self.assertTrue(limits)
            self.assertEqual(set(limits), {1})
            # Preserve the three original parent guard calls around final fit.
            self.assertEqual(guards[-3:], [{}, {'check_inventory': False}, {}])
            records.append((limits, guards))
        self.assertEqual(records[0], records[1])

    def test_two_threads_only_in_final_phase_and_caller_restored(self):
        observed = []
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

        with threadpool_limits(3), patch.object(KernGDWD, 'fit', fit), \
                patch.object(KernGDWD, 'cv_init', initialize), \
                patch.object(StandardScaler, 'fit_transform', scale), \
                patch.object(example, '_fit_and_score', score), \
                patch.object(example, '_publish_summaries', publish):
            before = counts()
            result = run_example(self.path, final_native_threads=np.int64(2))
            self.assertEqual(counts(), before)
        for stage, rows, pools in observed:
            self.assertEqual(set(pools.values()), {2 if rows == len(y) else 1}, (stage, rows, pools))
        self.assertTrue(any(stage == 'publish' for stage, _, _ in observed))
        self.check_provenance(result, 2)
        stored = json.loads((self.path / 'cv-results.json').read_text())
        self.assertEqual(stored['final_refit'], result['work_summary']['final_refit'])

    def test_resume_changes_only_fresh_final_threads_and_preserves_cv_receipts(self):
        with threadpool_limits(1):
            first = run_example(self.path, final_native_threads=1)
        saved = receipts(self.path)
        identity = (self.path / 'identity.json').read_bytes()
        prior_model = first['best_clf']
        for threads in (2, 1):
            with self.subTest(final_threads=threads), threadpool_limits(1), patch.object(
                    example, '_fit_and_score', side_effect=AssertionError('CV must replay')):
                result = run_example(self.path, resume=True, final_native_threads=threads)
                self.assertIsNot(prior_model, result['best_clf'])
                self.assertEqual(receipts(self.path), saved)
                self.assertEqual((self.path / 'identity.json').read_bytes(), identity)
                self.assertEqual(result['work_summary']['executed_evaluations'], 0)
                self.assertEqual(result['work_summary']['replayed_evaluations'], 4)
                self.check_provenance(result, threads)
                prior_model = result['best_clf']

    def test_cv_only_does_not_enter_final_scope(self):
        limits = []
        original = example.threadpool_limits

        def observed(*a, **k):
            limits.append(k.get('limits', a[0] if a else None))
            return original(*a, **k)

        with patch.object(example, 'threadpool_limits', observed):
            result = run_example(self.path, refit_best=False, final_native_threads=2)
        self.assertTrue(limits)
        self.assertEqual(set(limits), {1})
        self.assertIsNone(result['best_clf'])
        self.check_provenance(result, 2, performed=False)

    def test_two_worker_cv_keeps_one_thread_identity_with_two_thread_final(self):
        with threadpool_limits(3):
            before = counts()
            result = run_example(self.path, jobs=2, final_native_threads=2)
            self.assertEqual(counts(), before)
        self.assertEqual(result['work_summary']['fold_workers'], 2)
        identity = json.loads((self.path / 'identity.json').read_text())['payload']['identity']
        self.assertEqual(identity['native_threads'], 1)
        self.assertEqual({p['num_threads'] for p in identity['runtime']['native_pools']}, {1})
        self.check_provenance(result, 2)
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

    def test_final_failures_restore_caller_and_resume_completed_cv(self):
        run_example(self.path, refit_best=False)
        saved = receipts(self.path)
        for stage in ('scale', 'fit', 'runtime'):
            for error_type in (RuntimeError, KeyboardInterrupt):
                original = {'scale': StandardScaler.fit_transform, 'fit': KernGDWD.fit,
                            'runtime': example._check_runtime}[stage]

                def fail(*a, **k):
                    final = (k.get('native_threads') == 2 if stage == 'runtime' else len(a[1]) == len(y))
                    if final:
                        assert_counts(self, 2)
                        raise error_type('manufactured final failure')
                    return original(*a, **k)

                target, name = {'scale': (StandardScaler, 'fit_transform'),
                                'fit': (KernGDWD, 'fit'), 'runtime': (example, '_check_runtime')}[stage]
                with self.subTest(stage=stage, error=error_type.__name__), threadpool_limits(3):
                    before = counts()
                    with patch.object(target, name, fail), patch.object(
                            example, '_fit_and_score', side_effect=AssertionError('CV must replay')):
                        with self.assertRaisesRegex(error_type, 'manufactured final failure'):
                            run_example(self.path, resume=True, final_native_threads=2)
                    self.assertEqual(counts(), before)
                    self.assertEqual(receipts(self.path), saved)
                    result = run_example(self.path, resume=True, final_native_threads=2)
                    self.assertEqual(counts(), before)
                    self.check_provenance(result, 2)

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
                result = run_example(self.path / stage, final_native_threads=2)
                assert_counts(self, 3)
                self.check_provenance(result, 2)
                self.assertIsNotNone(result['best_clf'])
                if stage == 'summary':
                    self.assertEqual(result['work_summary']['coverage'], 'unavailable')

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

    def test_cli_validates_and_forwards_final_threads(self):
        common = ['--data', 'missing.npz', '--run-dir', 'unused', '--lambd', '.1', '--gamma', '.2']
        with redirect_stderr(io.StringIO()):
            for value in ('0', '-1', '1.5', 'True'):
                with self.subTest(invalid=value), self.assertRaises(SystemExit) as caught:
                    command.main(common + ['--final-native-threads', value])
                self.assertEqual(caught.exception.code, 2)
        self.path.mkdir(parents=True)
        np.savez(self.path / 'data.npz', X=X, y=y)
        common[1] = str(self.path / 'data.npz')
        for extra, expected in (([], 1), (['--final-native-threads', '2', '--cv-only'], 2)):
            with self.subTest(extra=extra), patch.object(command, 'run_resumable_cv', return_value={
                    'best_score': .5, 'best_params': {}, 'work_summary': {}}) as called, redirect_stdout(io.StringIO()):
                command.main(common + extra)
            self.assertEqual(called.call_args.kwargs['final_native_threads'], expected)


if __name__ == '__main__':
    unittest.main()
