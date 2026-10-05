"""Small numerical and recovery checks for the standalone tuning example."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csr_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from dwd.cv import DoL2LoD, run_cv
from dwd.gen_kern_dwd import KernGDWD


EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import resumable_kernel_cv as example
from _cv_checkpoint import CheckpointError, stable, write_receipt


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


class ResumableKernelCVTests(unittest.TestCase):
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
        self.resources = ExitStack()
        self.addCleanup(self.resources.close)
        self.resources.enter_context(threadpool_limits(limits=1))
        self.root = Path(self.resources.enter_context(tempfile.TemporaryDirectory()))
        rng = np.random.default_rng(817)
        self.X = rng.normal(size=(24, 4)) * np.array([1., 4., .3, 2.])
        self.X += np.array([3., -2., 8., 1.])
        self.y = np.tile([-7, 9], 12)
        self.query = rng.normal(size=(5, 4))
        self.params = {'lambd': [.08, .2]}
        self.folds = list(StratifiedKFold(2, shuffle=True, random_state=71).split(
            self.X, self.y))

    def model(self, implementation='optimized', **kwargs):
        values = dict(kernel='rbf', kernel_kws={'gamma': .12},
                      implementation=implementation, initialization='zero',
                      stopping='fixed', max_iter=3, random_state=23)
        values.update(kwargs)
        return KernGDWD(**values)

    def run_example(self, name='run', **kwargs):
        options = dict(cv=self.folds, scoring='accuracy', jobs=1,
                       refit_best=True, scale=False)
        options.update(kwargs)
        model = options.pop('clf', self.model())
        X = options.pop('X', self.X)
        y = options.pop('y', self.y)
        params = options.pop('params', self.params)
        return example.run_resumable_cv(
            model, X, y, params, self.root / name, **options)

    def assert_scores_equal(self, actual, expected):
        self.assertEqual(actual['best_params'], expected[0])
        self.assertEqual(actual['best_score'], expected[1])
        self.assertEqual(actual['agg_results']['params'], expected[3]['params'])
        for key in ('mean_train_score', 'mean_test_score',
                    'std_train_score', 'std_test_score'):
            assert_array_equal(actual['agg_results'][key], expected[3][key])
        for actual_fold, expected_fold in zip(actual['all_cv_results'], expected[4]):
            self.assertEqual(actual_fold['params'], expected_fold['params'])
            assert_array_equal(actual_fold['train_score'], expected_fold['train_score'])
            assert_array_equal(actual_fold['test_score'], expected_fold['test_score'])

    def receipts(self, name='run'):
        return sorted((self.root / name / 'receipts').glob(
            'fold-*/candidate-*.json'))

    def assert_work(self, result, executed, replayed, refit):
        summary = result['work_summary']
        self.assertEqual(summary['executed_evaluations'], executed)
        self.assertEqual(summary['replayed_evaluations'], replayed)
        self.assertEqual(summary['final_refit']['performed'], refit)
        self.assertEqual(summary['final_refit']['fresh_model'], refit)
        if refit:
            self.assertEqual(summary['final_refit']['diagnostics']['max_iter'],
                             result['best_clf'].max_iter)
            self.assertEqual(summary['final_refit']['diagnostics']['n_iter'],
                             result['best_clf'].n_iter_)
        else:
            self.assertIsNone(summary['final_refit']['diagnostics'])
        self.assertEqual(summary['fold_count'], len(self.folds))
        self.assertEqual(summary['candidate_count'], len(DoL2LoD(self.params)))
        for section in ('current_seconds', 'historical_seconds'):
            for key, value in summary[section].items():
                with self.subTest(section=section, duration=key):
                    self.assertTrue(np.isfinite(value))
                    self.assertGreaterEqual(value, 0.)

    def test_serial_matches_original_cv_for_both_implementations(self):
        for implementation in ('reference', 'optimized'):
            with self.subTest(implementation=implementation):
                model = self.model(implementation)
                original_params = copy.deepcopy(model.get_params())
                expected = run_cv(model, self.X, self.y, self.params,
                                  cv=self.folds, refit_best=True)
                actual = self.run_example(implementation, clf=model)
                self.assert_scores_equal(actual, expected)
                assert_allclose(actual['best_clf'].dual_coef_,
                                expected[2].dual_coef_, rtol=0, atol=0)
                assert_allclose(actual['best_clf'].intercept_,
                                expected[2].intercept_, rtol=0, atol=0)
                assert_array_equal(actual['best_clf'].predict(self.query),
                                   expected[2].predict(self.query))
                self.assertIsNone(actual['scaler'])
                self.assertEqual(model.get_params(), original_params)
                self.assertFalse(hasattr(model, 'dual_coef_'))
                self.assert_work(actual, 4, 0, True)

    def test_scaling_matches_training_fold_only_manual_loop_and_final_refit(self):
        candidates = DoL2LoD(self.params)
        expected_train = []
        expected_test = []
        for train, test in self.folds:
            scaler = StandardScaler().fit(self.X[train])
            train_X = scaler.transform(self.X[train])
            test_X = scaler.transform(self.X[test])
            train_scores, test_scores = [], []
            for candidate in candidates:
                fitted = self.model().set_params(**candidate).fit(train_X, self.y[train])
                train_scores.append(fitted.score(train_X, self.y[train]))
                test_scores.append(fitted.score(test_X, self.y[test]))
            expected_train.append(train_scores)
            expected_test.append(test_scores)
        original_scale_fit = StandardScaler.fit
        scaling_inputs = []

        def observe_scaling(scaler, X, y=None, sample_weight=None):
            scaling_inputs.append(X.copy())
            return original_scale_fit(scaler, X, y, sample_weight=sample_weight)

        with patch.object(StandardScaler, 'fit', observe_scaling):
            actual = self.run_example(scale=True)
        self.assertEqual(len(scaling_inputs), 3)
        for observed, (train, _) in zip(scaling_inputs, self.folds):
            assert_array_equal(observed, self.X[train])
        assert_array_equal(scaling_inputs[-1], self.X)
        assert_array_equal([fold['train_score'] for fold in actual['all_cv_results']],
                           expected_train)
        assert_array_equal([fold['test_score'] for fold in actual['all_cv_results']],
                           expected_test)
        means = np.mean(expected_test, axis=0)
        winner = candidates[np.argmax(means)]
        self.assertEqual(actual['best_params'], winner)
        self.assertEqual(actual['best_score'], means[np.argmax(means)])
        scaler = StandardScaler().fit(self.X)
        expected = self.model().set_params(**winner).fit(scaler.transform(self.X), self.y)
        assert_array_equal(actual['scaler'].mean_, scaler.mean_)
        assert_array_equal(actual['scaler'].scale_, scaler.scale_)
        assert_allclose(actual['best_clf'].dual_coef_, expected.dual_coef_, rtol=0, atol=0)
        assert_allclose(actual['best_clf'].decision_function(actual['scaler'].transform(self.query)),
                        expected.decision_function(scaler.transform(self.query)), rtol=0, atol=0)

    def test_compatible_candidates_prepare_once_per_fold(self):
        original = KernGDWD.cv_init
        calls = []

        def observed(model, data):
            calls.append(data.copy())
            return original(model, data)

        with patch.object(KernGDWD, 'cv_init', observed):
            result = self.run_example(refit_best=False)
        self.assertEqual(len(calls), len(self.folds))
        for prepared, (train, _) in zip(calls, self.folds):
            assert_array_equal(prepared, self.X[train])
        self.assertIsNone(result['best_clf'])
        self.assertIsNone(result['scaler'])
        self.assert_work(result, 4, 0, False)

    def test_mutating_caller_arrays_during_scoring_cannot_change_owned_data(self):
        caller_X, caller_y = self.X.copy(), self.y.copy()
        baseline = self.run_example('baseline', X=caller_X, y=caller_y)
        original = example.check_scoring
        mutated = []

        def observed_scorer(*args, **kwargs):
            scorer = original(*args, **kwargs)

            def mutate_caller(model, X, y):
                if not mutated:
                    caller_X[:] *= 3.
                    caller_y[:] = caller_y[::-1]
                    mutated.append(True)
                return scorer(model, X, y)

            return mutate_caller

        with patch.object(example, 'check_scoring', observed_scorer):
            result = self.run_example(X=caller_X, y=caller_y)
        self.assertTrue(mutated)
        self.assertFalse(np.array_equal(caller_y, self.y))
        self.assertEqual(result['best_params'], baseline['best_params'])
        self.assertEqual(result['best_score'], baseline['best_score'])
        for key in ('mean_train_score', 'mean_test_score',
                    'std_train_score', 'std_test_score'):
            assert_array_equal(result['agg_results'][key], baseline['agg_results'][key])
        assert_array_equal(result['best_clf'].dual_coef_, baseline['best_clf'].dual_coef_)
        assert_array_equal(result['best_clf'].intercept_, baseline['best_clf'].intercept_)

    def test_auxiliary_report_failure_does_not_undo_fit_or_completed_receipts(self):
        baseline = self.run_example('baseline')
        with warnings.catch_warnings(), patch.object(
                example, 'write_json_atomic', side_effect=OSError('manufactured report failure')) as publish:
            warnings.simplefilter('error', RuntimeWarning)
            result = self.run_example()
        self.assertGreaterEqual(publish.call_count, 1)
        self.assertEqual(len(self.receipts()), 4)
        self.assertEqual(result['best_params'], baseline['best_params'])
        self.assertEqual(result['best_score'], baseline['best_score'])
        assert_array_equal(result['best_clf'].dual_coef_, baseline['best_clf'].dual_coef_)
        self.assert_work(result, 4, 0, True)
        with patch.object(example, '_fit_and_score', side_effect=AssertionError('CV rerun')):
            resumed = self.run_example(resume=True)
        self.assert_work(resumed, 0, 4, True)

    def test_persisted_results_match_returned_scores_timings_and_refit_diagnostics(self):
        for resume in (False, True):
            with self.subTest(resume=resume):
                result = self.run_example(scale=True, resume=resume)
                saved = json.loads((self.root / 'run' / 'cv-results.json').read_text(
                    encoding='utf-8'))
                work = json.loads((self.root / 'run' / 'work-summary.json').read_text(
                    encoding='utf-8'))
                manifest = json.loads((self.root / 'run' / 'identity.json').read_text(
                    encoding='utf-8'))['payload']
                self.assertEqual(saved['schema'], 1)
                self.assertEqual(saved['identity_sha256'], manifest['identity_sha256'])
                self.assertEqual(saved['best_params'], result['best_params'])
                self.assertEqual(saved['best_params_identity'], stable(result['best_params']))
                self.assertEqual(saved['candidate_parameters_identity'], stable(DoL2LoD(self.params)))
                self.assertEqual(saved['best_score'], result['best_score'])
                self.assertEqual(saved['agg_results'], result['agg_results'])
                self.assertEqual(saved['all_cv_results'], result['all_cv_results'])
                self.assertEqual(saved['final_refit'], result['work_summary']['final_refit'])
                self.assertEqual(work, result['work_summary'])
                self.assertNotIn('best_clf', saved)
                self.assertNotIn('scaler', saved)
                self.assert_work(result, 0 if resume else 4, 4 if resume else 0, True)

    def test_summary_assembly_failure_preserves_selected_scores_and_model(self):
        baseline = self.run_example('baseline')
        with warnings.catch_warnings(), patch.object(
                example, '_work_summary', side_effect=RuntimeError('manufactured summary failure')):
            warnings.simplefilter('error', RuntimeWarning)
            result = self.run_example()
        self.assertEqual(result['work_summary']['coverage'], 'unavailable')
        self.assertTrue(result['work_summary']['final_refit']['performed'])
        self.assertTrue(result['work_summary']['final_refit']['fresh_model'])
        self.assertEqual(result['best_params'], baseline['best_params'])
        self.assertEqual(result['best_score'], baseline['best_score'])
        assert_array_equal(result['best_clf'].dual_coef_, baseline['best_clf'].dual_coef_)
        self.assertEqual(len(self.receipts()), 4)

    def test_final_diagnostic_failure_does_not_discard_completed_model(self):
        baseline = self.run_example('baseline')
        original = example._diagnostics

        def fail_final_only(model):
            if model.dual_coef_.shape[1] == len(self.y):
                raise RuntimeError('manufactured final diagnostic failure')
            return original(model)

        with warnings.catch_warnings(), patch.object(example, '_diagnostics', fail_final_only):
            warnings.simplefilter('error', RuntimeWarning)
            result = self.run_example()
        self.assertIsNone(result['work_summary']['final_refit']['diagnostics'])
        self.assertTrue(result['work_summary']['final_refit']['performed'])
        self.assertEqual(result['best_params'], baseline['best_params'])
        self.assertEqual(result['best_score'], baseline['best_score'])
        assert_array_equal(result['best_clf'].dual_coef_, baseline['best_clf'].dual_coef_)
        self.assertEqual(len(self.receipts()), 4)

    def test_kernel_changes_refresh_preparation_inside_each_fold(self):
        original = KernGDWD.cv_init
        calls = []

        def observed(model, data):
            calls.append((len(data), model.kernel_kws['gamma']))
            return original(model, data)

        params = {'kernel_kws': [{'gamma': .12}, {'gamma': .3}], 'lambd': [.08, .2]}
        with patch.object(KernGDWD, 'cv_init', observed):
            result = self.run_example(params=params, refit_best=False)
        self.assertEqual(calls, [(12, .12), (12, .3), (12, .12), (12, .3)])
        self.assertEqual(result['agg_results']['params'], DoL2LoD(params))

    def test_completed_replay_preserves_cv_receipts_but_refits_fresh_model(self):
        first = self.run_example(scale=True)
        before = {path: path.read_bytes() for path in self.receipts()}
        self.assertEqual(len(before), 4)
        original_fit = KernGDWD.fit
        fit_sizes = []

        def observed(model, X, y, *args, **kwargs):
            fit_sizes.append(len(y))
            return original_fit(model, X, y, *args, **kwargs)

        with patch.object(KernGDWD, 'fit', observed), patch.object(
                example, '_fit_and_score', side_effect=AssertionError('replayed a fit')):
            second = self.run_example(scale=True, resume=True)
        self.assertEqual(fit_sizes, [len(self.y)])
        self.assertIsNot(first['best_clf'], second['best_clf'])
        assert_array_equal(first['best_clf'].dual_coef_, second['best_clf'].dual_coef_)
        self.assertEqual(first['all_cv_results'], second['all_cv_results'])
        self.assertEqual(before, {path: path.read_bytes() for path in self.receipts()})
        self.assert_work(second, 0, 4, True)
        for key in ('preparation', 'fit', 'score'):
            self.assertEqual(second['work_summary']['current_seconds'][key], 0.)
            self.assertEqual(second['work_summary']['historical_seconds'][key],
                             first['work_summary']['current_seconds'][key])

    def test_interrupted_candidate_is_rerun_and_completed_candidates_are_replayed(self):
        original = example._fit_and_score
        attempted = []
        stop = KeyboardInterrupt('manufactured interruption during second candidate')

        def interrupted(*args, **kwargs):
            attempted.append(True)
            if len(attempted) == 2:
                raise stop
            return original(*args, **kwargs)

        with patch.object(example, '_fit_and_score', interrupted):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.run_example(refit_best=False)
        self.assertIs(caught.exception, stop)
        saved = {path: path.read_bytes() for path in self.receipts()}
        self.assertEqual(len(saved), 1)
        with patch.object(example, '_fit_and_score', wraps=original) as fitted:
            resumed = self.run_example(refit_best=False, resume=True)
        self.assertEqual(fitted.call_count, 3)
        self.assert_work(resumed, 3, 1, False)
        self.assertTrue(all(path.read_bytes() == data for path, data in saved.items()))
        fresh = self.run_example('fresh', refit_best=False)
        self.assertEqual(resumed['best_params'], fresh['best_params'])
        assert_array_equal(resumed['agg_results']['mean_test_score'],
                           fresh['agg_results']['mean_test_score'])

    def test_interrupted_final_refit_replays_all_cv_then_refits_again(self):
        original_fit = KernGDWD.fit
        stop = KeyboardInterrupt('manufactured final refit interruption')

        def interrupted(model, X, y, *args, **kwargs):
            if len(y) == len(self.y):
                raise stop
            return original_fit(model, X, y, *args, **kwargs)

        with patch.object(KernGDWD, 'fit', interrupted):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.run_example()
        self.assertIs(caught.exception, stop)
        self.assertEqual(len(self.receipts()), 4)
        with patch.object(example, '_fit_and_score', side_effect=AssertionError('CV rerun')):
            result = self.run_example(resume=True)
        self.assertIsNotNone(result['best_clf'])
        self.assert_work(result, 0, 4, True)

    def test_reference_random_initialization_resumes_exactly_after_unpublished_fit(self):
        model = self.model('reference', initialization='auto', random_state=83)
        expected = run_cv(model, self.X, self.y, self.params,
                          cv=self.folds, refit_best=True)
        original = example._fit_and_score
        attempted = []
        stop = KeyboardInterrupt('manufactured interruption after an unsaved random fit')

        def interrupted(*args, **kwargs):
            result = original(*args, **kwargs)
            attempted.append(True)
            if len(attempted) == 2:
                raise stop
            return result

        with patch.object(example, '_fit_and_score', interrupted):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.run_example(clf=model)
        self.assertIs(caught.exception, stop)
        self.assertEqual(len(self.receipts()), 1)
        with patch.object(example, '_fit_and_score', wraps=original) as fitted:
            resumed = self.run_example(clf=model, resume=True)
        self.assertEqual(fitted.call_count, 3)
        self.assert_scores_equal(resumed, expected)
        assert_array_equal(resumed['best_clf'].dual_coef_, expected[2].dual_coef_)
        assert_array_equal(resumed['best_clf'].intercept_, expected[2].intercept_)
        assert_array_equal(resumed['best_clf'].decision_function(self.query),
                           expected[2].decision_function(self.query))
        self.assert_work(resumed, 3, 1, True)

    def test_non_resume_cannot_overwrite_existing_completed_run(self):
        self.run_example(refit_best=False)
        before = {path: path.read_bytes() for path in self.receipts()}
        with patch.object(KernGDWD, 'fit', side_effect=AssertionError('unexpected fit')):
            with self.assertRaises(CheckpointError):
                self.run_example(refit_best=False)
        self.assertEqual(before, {path: path.read_bytes() for path in self.receipts()})

    def test_identity_changes_reject_before_fitting(self):
        self.run_example(refit_best=False)
        changed_X = self.X.copy()
        changed_X[0, 0] = np.nextafter(changed_X[0, 0], np.inf)
        changed_y = self.y.copy()
        changed_y[[0, 1]] = changed_y[[1, 0]]
        mutations = [dict(X=changed_X), dict(y=changed_y),
                     dict(params={'lambd': [.2, .08]}),
                     dict(clf=self.model(max_iter=4)),
                     dict(clf=self.model(random_state=24)), dict(scale=True),
                     dict(cv=self.folds[::-1])]
        before = {path: path.read_bytes() for path in self.receipts()}
        for mutation in mutations:
            with self.subTest(mutation=repr(mutation)):
                with patch.object(KernGDWD, 'fit', side_effect=AssertionError('unexpected fit')):
                    with self.assertRaises(CheckpointError):
                        self.run_example(resume=True, refit_best=False, **mutation)
        self.assertEqual(before, {path: path.read_bytes() for path in self.receipts()})

    def test_corrupted_late_receipt_is_rejected_before_filling_missing_early_one(self):
        self.run_example(refit_best=False)
        receipts = self.receipts()
        receipts[0].unlink()
        receipts[-1].write_text('{"broken": true}', encoding='utf-8')
        with patch.object(KernGDWD, 'fit', side_effect=AssertionError('unexpected fit')):
            with self.assertRaises(CheckpointError):
                self.run_example(refit_best=False, resume=True)
        self.assertFalse(receipts[0].exists())

    def test_changed_runtime_identity_rejects_reuse_before_fitting(self):
        self.run_example(refit_best=False)
        original = example.runtime_fingerprint

        def changed_runtime(*args, **kwargs):
            result = copy.deepcopy(original(*args, **kwargs))
            result['machine'] += '-different-runtime'
            return result

        with patch.object(example, 'runtime_fingerprint', changed_runtime), patch.object(
                KernGDWD, 'fit', side_effect=AssertionError('unexpected fit')):
            with self.assertRaises(CheckpointError):
                self.run_example(refit_best=False, resume=True)

    def test_checksummed_receipts_still_require_valid_evaluation_metadata(self):
        self.run_example(refit_best=False)
        receipt = self.receipts()[-1]
        original = receipt.read_bytes()
        payload = json.loads(original)['payload']
        mutations = [dict(schema=True), dict(fold_index=99),
                     dict(candidate_index=99), dict(train_count=999),
                     dict(parameter_sha256='0' * 64),
                     dict(train_score=1.01), dict(test_score=True),
                     dict(runtime=-1.), dict(diagnostics={})]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                modified = copy.deepcopy(payload)
                modified.update(mutation)
                write_receipt(receipt, modified)
                try:
                    with patch.object(KernGDWD, 'fit', side_effect=AssertionError('unexpected fit')):
                        with self.assertRaises(CheckpointError):
                            self.run_example(refit_best=False, resume=True)
                finally:
                    receipt.write_bytes(original)

    def test_invalid_fold_indices_reject_before_fitting(self):
        valid_train, valid_test = self.folds[0]
        invalid = [
            (valid_train[:-1].tolist() + [valid_train[0]], valid_test),
            (valid_train, valid_test[:-1].tolist() + [valid_test[0]]),
            (valid_train, np.r_[valid_test, valid_train[0]]),
            (np.array([], dtype=int), valid_test),
            (valid_train, np.array([], dtype=int)),
            (np.r_[valid_train, -1], valid_test),
            (valid_train, np.r_[valid_test, len(self.y)]),
            (valid_train.astype(float), valid_test),
            (valid_train.reshape(2, -1), valid_test),
            (np.array([True, False]), valid_test),
        ]
        for index, fold in enumerate(invalid):
            with self.subTest(index=index):
                with patch.object(KernGDWD, 'fit', side_effect=AssertionError('unexpected fit')):
                    with self.assertRaises((TypeError, ValueError)):
                        self.run_example('fold-' + str(index), cv=[fold, self.folds[1]])

    def test_candidate_order_and_first_tie_match_original_selection(self):
        params = {'lambd': [.2, .08], 'q': [2., 1.]}
        with patch.object(example, 'check_scoring', return_value=lambda model, X, y: .5):
            result = self.run_example(params=params)
        candidates = DoL2LoD(params)
        self.assertEqual(result['agg_results']['params'], candidates)
        self.assertEqual(result['best_params'], candidates[0])
        self.assertEqual(result['best_score'], .5)
        self.assertEqual(result['best_clf'].lambd, .2)
        self.assertEqual(result['best_clf'].q, 2.)

    def test_nonfinite_and_nonscalar_scores_are_not_checkpointed(self):
        for index, value in enumerate((float('nan'), float('inf'),
                                       -float('inf'), np.array([.5]), .5 + 0j)):
            with self.subTest(value=repr(value)):
                name = 'invalid-' + str(index)
                with patch.object(example, 'check_scoring', return_value=lambda model, X, y: value):
                    with self.assertRaisesRegex(ValueError, 'finite real scalar'):
                        self.run_example(name)
                self.assertEqual(self.receipts(name), [])

    def test_out_of_range_accuracy_scores_cannot_be_saved_or_refitted(self):
        original_fit = KernGDWD.fit
        fitted_sizes = []

        def observed(model, X, y, *args, **kwargs):
            fitted_sizes.append(len(y))
            return original_fit(model, X, y, *args, **kwargs)

        for index, score in enumerate((-0.01, 1.01, np.finfo(float).max)):
            with self.subTest(score=score):
                fitted_sizes.clear()
                name = 'range-' + str(index)
                with patch.object(KernGDWD, 'fit', observed), patch.object(
                        example, 'check_scoring', return_value=lambda model, X, y: score):
                    with self.assertRaises(ValueError):
                        self.run_example(name)
                self.assertEqual(fitted_sizes, [12])
                self.assertEqual(self.receipts(name), [])

    def test_real_two_worker_execution_matches_serial_and_replays(self):
        serial = self.run_example('serial', scale=True)
        parallel = self.run_example('parallel', scale=True, jobs=2)
        self.assertEqual(serial['best_params'], parallel['best_params'])
        for key in ('mean_train_score', 'mean_test_score',
                    'std_train_score', 'std_test_score'):
            assert_array_equal(serial['agg_results'][key], parallel['agg_results'][key])
        assert_allclose(serial['best_clf'].dual_coef_, parallel['best_clf'].dual_coef_,
                        rtol=0, atol=0)
        with patch.object(example, '_fit_and_score', side_effect=AssertionError('CV rerun')):
            replayed = self.run_example('parallel', scale=True, jobs=1, resume=True)
        self.assert_work(replayed, 0, 4, True)

    def test_restricted_example_rejects_unsupported_inputs_before_fitting(self):
        class DerivedKernGDWD(KernGDWD):
            pass

        modifications = [
            dict(clf=self.model(random_state=None)),
            dict(clf=self.model(random_state=np.random.RandomState(1))),
            dict(clf=self.model(kernel='precomputed')),
            dict(clf=self.model(kernel=lambda X, Y: X @ Y.T)),
            dict(clf=self.model(callback=lambda state: False)),
            dict(clf=self.model(stopping='validation')),
            dict(clf=DerivedKernGDWD(random_state=23)),
            dict(params={'random_state': [1, 2]}),
            dict(params={'kernel': ['precomputed']}),
            dict(params={'callback': [lambda state: False]}),
            dict(scoring=lambda model, X, y: .5),
            dict(X=csr_matrix(self.X)),
            dict(X=self.X.astype(complex)),
            dict(X=np.where(np.arange(self.X.size).reshape(self.X.shape) == 0,
                            np.nan, self.X)),
            dict(y=np.arange(len(self.y)) % 3),
            dict(params={'lambd': []}),
            dict(jobs=0), dict(jobs=True), dict(jobs=1.5),
        ]
        for index, modification in enumerate(modifications):
            with self.subTest(index=index, modification=repr(modification)):
                with patch.object(KernGDWD, 'fit', side_effect=AssertionError('unexpected fit')):
                    with self.assertRaises((TypeError, ValueError, CheckpointError)):
                        self.run_example('unsupported-' + str(index), **modification)


if __name__ == '__main__':
    unittest.main(verbosity=2)
