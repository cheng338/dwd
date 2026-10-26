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
import dwd.gen_kern_dwd as estimator_module
import dwd._kernel_solver as solver_module
from numerical_test_support import AcceptedNativeReplay, objective_oracle
from reference_mm_oracle import reference_mm_oracle


EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import resumable_kernel_cv as example
import _grid_report as grid_report
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


class _ReferenceResumeObserver:
    """Observe actual initialized solves and fitted states without replacing them."""

    def __init__(self, case):
        self.case = case
        self.phase, self.active = None, None
        self.events, self.models, self.oracles = [], [], {}
        self.original_fit = KernGDWD.fit
        self.original_solve = solver_module.solve_kernel

    def task(self, X, y, penalty):
        case = self.case
        for index, (train, test) in enumerate(case.folds):
            if np.array_equal(X, case.X[train]) and np.array_equal(y, case.y[train]):
                return ('fold', index, float(penalty)), case.X[test]
        if np.array_equal(X, case.X) and np.array_equal(y, case.y):
            return ('full', -1, float(penalty)), case.query
        raise AssertionError('task/data identity guard')

    @staticmethod
    def assert_continuous(actual, expected, description):
        # Tight regression bound for this 24-row fixture, not a general error
        # certificate or a change to the package's numerical acceptance.
        scale = max(1., float(np.max(np.abs(expected))))
        tolerance = 256 * 24 * np.finfo(float).eps * scale
        assert_allclose(actual, expected, rtol=0, atol=tolerance, err_msg=description)

    def solve(self, K, y, penalty, **options):
        case, event, model = self.case, self.active, self.models[-1]
        case.assertIsNotNone(event, 'solver must belong to an observed fit')
        case.assertEqual(model.random_state, 83, 'seed parameter guard')
        case.assertEqual(model.initialization, 'auto')
        case.assertEqual(model.implementation, 'reference')
        initial = np.random.RandomState(83).normal(size=len(y))
        initial /= np.linalg.norm(initial)
        assert_array_equal(options['alpha_init'], initial, err_msg='initialization reset guard')
        case.assertIsNone(options['offset_init'], 'initial offset guard')
        assert_array_equal(y, np.where(event['y'] == 9, 1., -1.))
        case.assertEqual(float(penalty), event['task'][2], 'solver candidate guard')
        case.assertEqual(options['max_iter'], 3)
        case.assertEqual(options['stopping'], 'fixed')
        case.assertEqual(options['implementation'], 'reference')
        event.update(K=np.array(K, copy=True), signed_y=np.array(y, copy=True),
                     alpha_init=initial.copy(), offset_init=options['offset_init'])
        return self.original_solve(K, y, penalty, **options)

    def fit(self, model, X, y, *args, **kwargs):
        self.case.assertIsNone(self.active)
        task, held = self.task(X, y, model.lambd)
        event = {'phase': self.phase, 'task': task, 'X': np.array(X, copy=True),
                 'y': np.array(y, copy=True), 'held': held}
        self.events.append(event)
        # Retaining references prevents object-id reuse from confusing fresh-fit checks.
        self.models.append(model)
        self.active = event
        try:
            result = self.original_fit(model, X, y, *args, **kwargs)
            self.case.assertIs(result, model)
            self.inspect_model(model, event)
            return result
        finally:
            self.active = None

    def inspect_model(self, model, event):
        case = self.case
        case.assertEqual(model.n_iter_, 3)
        case.assertEqual(model.returned_iteration_, 3)
        case.assertEqual(model.termination_reason_, 'max_iter')
        case.assertEqual(model.backend_, 'spectral')
        case.assertEqual(model.diagnostics_['implementation'], 'reference')
        case.assertEqual(model.diagnostics_['positive_eigenvalues_discarded'], 0)
        case.assertFalse(model.diagnostics_['kernel_approximation_used'])
        key = (event['task'], event['K'].tobytes(order='C'))
        if key not in self.oracles:
            self.oracles[key] = reference_mm_oracle(
                event['K'], event['signed_y'], event['task'][2], event['alpha_init'])
        oracle_alpha, oracle_b = self.oracles[key]
        alpha, b = model.dual_coef_.reshape(-1), float(model.intercept_[0])
        self.assert_continuous(alpha, oracle_alpha, 'Decimal MM coefficient guard')
        self.assert_continuous(b, oracle_b, 'Decimal MM intercept guard')
        objective = objective_oracle(event['K'], event['signed_y'], alpha, b, event['task'][2])
        self.assert_continuous(model.final_objective_, objective, 'original-kernel objective guard')
        event.update(alpha=alpha.copy(), b=b, history=model.objective_history_.copy(),
                     train_labels=model.predict(event['X']).copy(),
                     held_labels=model.predict(event['held']).copy(),
                     query_labels=model.predict(case.query).copy(),
                     decisions=model.decision_function(case.query).copy(), model=model)


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
                self.assertEqual(saved['schema'], 2)
                self.assertEqual(saved['identity_sha256'], manifest['identity_sha256'])
                self.assertEqual(saved['best_params'], result['best_params'])
                self.assertEqual(saved['best_params_identity'], stable(result['best_params']))
                self.assertEqual(saved['candidate_parameters_identity'], stable(DoL2LoD(self.params)))
                self.assertEqual(saved['best_score'], result['best_score'])
                self.assertEqual(saved['agg_results'], result['agg_results'])
                self.assertEqual(saved['all_cv_results'], result['all_cv_results'])
                self.assertEqual(saved['finite_grid_report'], result['finite_grid_report'])
                self.assertEqual(result['finite_grid_report']['status'], 'available')
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
        self.assertEqual(first['finite_grid_report'], second['finite_grid_report'])
        self.assertEqual(second['finite_grid_report']['candidate_count'], 2)
        self.assertEqual(second['finite_grid_report']['fold_count'], 2)
        self.assertEqual(before, {path: path.read_bytes() for path in self.receipts()})
        self.assert_work(second, 0, 4, True)
        for key in ('preparation', 'fit', 'score'):
            self.assertEqual(second['work_summary']['current_seconds'][key], 0.)
            self.assertEqual(second['work_summary']['historical_seconds'][key],
                             first['work_summary']['current_seconds'][key])

    def test_finite_grid_report_failure_preserves_fit_receipts_and_later_replay(self):
        first = self.run_example(scale=True)
        before = {path: path.read_bytes() for path in self.receipts()}
        identity_path = self.root / 'run' / 'identity.json'
        identity_before = identity_path.read_bytes()
        first_query = first['scaler'].transform(self.query)
        first_scores = first['best_clf'].decision_function(first_query)
        first_predictions = first['best_clf'].predict(first_query)
        original_fit = KernGDWD.fit
        fit_sizes = []

        def observed_fit(model, X, y, *args, **kwargs):
            fit_sizes.append(len(y))
            return original_fit(model, X, y, *args, **kwargs)

        with patch.object(grid_report, 'summarize_grid',
                          side_effect=RuntimeError('private reporting sentinel')) as failed, \
                patch.object(example, '_fit_and_score', side_effect=AssertionError('CV rerun')), \
                patch.object(KernGDWD, 'fit', observed_fit):
            result = self.run_example(scale=True, resume=True)
        self.assertEqual(failed.call_count, 1)
        self.assertEqual(fit_sizes, [len(self.y)])
        self.assertIsNot(result['best_clf'], first['best_clf'])
        self.assertEqual(result['finite_grid_report'], {
            'status': 'unavailable', 'scope': 'best_among_supplied_candidates',
            'reporting_error': 'RuntimeError'})
        self.assertEqual(result['best_params'], first['best_params'])
        self.assertEqual(result['best_score'], first['best_score'])
        self.assertEqual(result['all_cv_results'], first['all_cv_results'])
        self.assertEqual(result['agg_results'], first['agg_results'])
        assert_array_equal(result['best_clf'].dual_coef_, first['best_clf'].dual_coef_)
        assert_array_equal(result['best_clf'].intercept_, first['best_clf'].intercept_)
        result_query = result['scaler'].transform(self.query)
        assert_array_equal(result['best_clf'].decision_function(result_query), first_scores)
        assert_array_equal(result['best_clf'].predict(result_query), first_predictions)
        self.assertEqual(before, {path: path.read_bytes() for path in self.receipts()})
        self.assertEqual(identity_before, identity_path.read_bytes())
        self.assert_work(result, 0, 4, True)
        saved_path = self.root / 'run' / 'cv-results.json'
        saved = json.loads(saved_path.read_text(encoding='utf-8'))
        self.assertEqual(saved['finite_grid_report'], result['finite_grid_report'])
        self.assertNotIn('private reporting sentinel', saved_path.read_text(encoding='utf-8'))

        # An unavailable convenience report never prevents receipt reuse. A
        # later report can be regenerated with every estimator fit prohibited.
        with patch.object(KernGDWD, 'fit', side_effect=AssertionError('unexpected fit')), \
                patch.object(example, '_run_fold', side_effect=AssertionError('unexpected CV work')):
            replayed = self.run_example(scale=True, resume=True, refit_best=False)
        self.assertEqual(replayed['finite_grid_report'], first['finite_grid_report'])
        self.assertEqual(replayed['best_params'], first['best_params'])
        self.assertEqual(replayed['all_cv_results'], first['all_cv_results'])
        self.assertEqual(before, {path: path.read_bytes() for path in self.receipts()})
        self.assertEqual(identity_before, identity_path.read_bytes())
        self.assert_work(replayed, 0, 4, False)
        saved = json.loads(saved_path.read_text(encoding='utf-8'))
        self.assertEqual(saved['finite_grid_report'], first['finite_grid_report'])

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

    def test_joint_affine_interruption_replays_only_completed_folds(self):
        model = self.model(affine_computation='joint')
        original = example._fit_and_score
        calls = []

        def interrupted(*args, **kwargs):
            calls.append(True)
            if len(calls) == 2:
                raise KeyboardInterrupt('interrupted joint affine candidate')
            return original(*args, **kwargs)

        with patch.object(example, '_fit_and_score', interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.run_example(clf=model)
        saved = {path: path.read_bytes() for path in self.receipts()}
        self.assertEqual(len(saved), 1)
        resumed = self.run_example(clf=model, resume=True)
        fresh = self.run_example('fresh-joint', clf=model)
        self.assert_work(resumed, 3, 1, True)
        self.assertTrue(all(path.read_bytes() == value for path, value in saved.items()))
        self.assertEqual(resumed['best_clf'].affine_computation_, 'joint')
        self.assertEqual(resumed['best_params'], fresh['best_params'])
        assert_array_equal(resumed['agg_results']['mean_test_score'],
                           fresh['agg_results']['mean_test_score'])
        assert_array_equal(resumed['best_clf'].decision_function(self.query),
                           fresh['best_clf'].decision_function(self.query))

    def test_changed_affine_policy_rejects_checkpoint_reuse_before_fitting(self):
        for original_policy, changed_policy in (('standard', 'joint'), ('joint', 'standard')):
            with self.subTest(original_policy=original_policy):
                name = 'affine-' + original_policy
                self.run_example(name, clf=self.model(affine_computation=original_policy),
                                 refit_best=False)
                with patch.object(KernGDWD, 'fit', side_effect=AssertionError('unexpected fit')):
                    with self.assertRaises(CheckpointError):
                        self.run_example(name, clf=self.model(affine_computation=changed_policy),
                                         refit_best=False, resume=True)

    def test_extended_exact_recovery_survives_interruption_and_refit(self):
        model = self.model(exact_recovery='extended')
        original = example._fit_and_score
        calls = []

        def interrupted(*args, **kwargs):
            calls.append(True)
            if len(calls) == 2:
                raise KeyboardInterrupt('interrupted extended-recovery candidate')
            return original(*args, **kwargs)

        with patch.object(example, '_fit_and_score', interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.run_example(clf=model)
        saved = {path: path.read_bytes() for path in self.receipts()}
        self.assertEqual(len(saved), 1)
        resumed = self.run_example(clf=model, resume=True)
        fresh = self.run_example('fresh-extended', clf=model)
        self.assert_work(resumed, 3, 1, True)
        self.assertTrue(all(path.read_bytes() == value for path, value in saved.items()))
        self.assertEqual(resumed['best_clf'].exact_recovery_, 'extended')
        self.assertEqual(resumed['best_params'], fresh['best_params'])
        assert_array_equal(resumed['agg_results']['mean_test_score'],
                           fresh['agg_results']['mean_test_score'])
        assert_array_equal(resumed['best_clf'].decision_function(self.query),
                           fresh['best_clf'].decision_function(self.query))

    def test_changed_exact_recovery_rejects_checkpoint_reuse_before_fitting(self):
        for original_policy, changed_policy in (('standard', 'extended'), ('extended', 'standard')):
            with self.subTest(original_policy=original_policy):
                name = 'recovery-' + original_policy
                self.run_example(name, clf=self.model(exact_recovery=original_policy),
                                 refit_best=False)
                with patch.object(KernGDWD, 'fit', side_effect=AssertionError('unexpected fit')):
                    with self.assertRaises(CheckpointError):
                        self.run_example(name, clf=self.model(exact_recovery=changed_policy),
                                         refit_best=False, resume=True)

    def test_reference_random_initialization_resumes_exactly_after_unpublished_fit(self):
        # Control native basis variation only for the exact-output assertion.
        # The companion below still performs every independent native solve.
        self._check_reference_resume(fixed_basis=True)

    def test_reference_random_initialization_resume_with_independent_native_bases(self):
        self._check_reference_resume(fixed_basis=False)

    def test_reference_resume_detects_changed_initialization_seed(self):
        with self.assertRaisesRegex(AssertionError, 'initialization reset guard'):
            self._check_reference_resume(fixed_basis=True, mutation='seed')

    def test_reference_resume_detects_advancing_initialization_generator(self):
        with self.assertRaisesRegex(AssertionError, 'initialization reset guard'):
            self._check_reference_resume(fixed_basis=True, mutation='reset')

    def test_reference_resume_detects_incorrect_work_accounting(self):
        with self.assertRaisesRegex(AssertionError, 'work accounting guard'):
            self._check_reference_resume(fixed_basis=False, mutation='accounting')

    def _check_reference_resume(self, *, fixed_basis, mutation=None):
        observed = _ReferenceResumeObserver(self)
        random_state = estimator_module.check_random_state
        shared_rng = np.random.RandomState(83)
        memo = None

        def random_state_control(seed):
            if observed.phase == 'resumed':
                if mutation == 'seed':
                    return random_state(84)
                if mutation == 'reset':
                    return shared_rng
            return random_state(seed)

        # A plain function preserves method binding when installed on the class.
        def observe_fit(model, X, y, *args, **kwargs):
            return observed.fit(model, X, y, *args, **kwargs)

        with ExitStack() as stack:
            if fixed_basis:
                memo = stack.enter_context(AcceptedNativeReplay())
            stack.enter_context(patch.object(KernGDWD, 'fit', observe_fit))
            stack.enter_context(patch.object(solver_module, 'solve_kernel', observed.solve))
            stack.enter_context(patch.object(estimator_module, 'check_random_state', random_state_control))
            model = self.model('reference', initialization='auto', random_state=83)
            observed.phase = 'baseline'
            expected = run_cv(model, self.X, self.y, self.params, cv=self.folds, refit_best=True)
            observed.phase = 'uninterrupted'
            uninterrupted = self.run_example(name='uninterrupted', clf=model)
            fit_and_score = example._fit_and_score
            attempted = []
            stop = KeyboardInterrupt('manufactured interruption after unpublished random fit')

            def interrupted(*args, **kwargs):
                result = fit_and_score(*args, **kwargs)
                attempted.append(True)
                if len(attempted) == 2:
                    raise stop
                return result

            observed.phase = 'interrupted'
            with patch.object(example, '_fit_and_score', interrupted):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    self.run_example(clf=model)
            self.assertIs(caught.exception, stop)
            self.assertEqual(len(attempted), 2)
            self.assertEqual(len(self.receipts()), 1)
            receipts_before = {path: path.read_bytes() for path in self.receipts()}
            identity_path = self.root / 'run' / 'identity.json'
            identity_before = identity_path.read_bytes()
            observed.phase = 'resumed'
            with patch.object(example, '_fit_and_score', wraps=fit_and_score) as called:
                resumed = self.run_example(clf=model, resume=True)
            self.assertEqual(called.call_count, 3, 'CV execution count guard')

        self.assertEqual(identity_path.read_bytes(), identity_before, 'checkpoint identity preservation guard')
        self.assertEqual({path: path.read_bytes() for path in receipts_before}, receipts_before,
                         'completed receipt preservation guard')
        self.assertEqual(len(self.receipts()), 4)
        self.assert_scores_equal(uninterrupted, expected)
        self.assert_scores_equal(resumed, expected)
        if mutation == 'accounting':
            resumed['work_summary']['executed_evaluations'] += 1
        self.assertEqual(resumed['work_summary']['executed_evaluations'], 3, 'work accounting guard')
        self.assert_work(uninterrupted, 4, 0, True)
        self.assert_work(resumed, 3, 1, True)
        selected = float(expected[0]['lambd'])
        cv_tasks = [('fold', fold, penalty) for fold in (0, 1) for penalty in (.08, .2)]
        expected_tasks = {'baseline': cv_tasks + [('full', -1, selected)],
                          'uninterrupted': cv_tasks + [('full', -1, selected)],
                          'interrupted': cv_tasks[:2],
                          'resumed': cv_tasks[1:] + [('full', -1, selected)]}
        for phase, tasks in expected_tasks.items():
            self.assertEqual([event['task'] for event in observed.events if event['phase'] == phase],
                             tasks, 'task/workflow guard')
        anchors = {event['task']: event for event in observed.events if event['phase'] == 'baseline'}
        for event in observed.events:
            anchor = anchors[event['task']]
            assert_array_equal(event['alpha_init'], anchor['alpha_init'])
            self.assertEqual(event['offset_init'], anchor['offset_init'])
            for field in ('train_labels', 'held_labels', 'query_labels'):
                assert_array_equal(event[field], anchor[field], err_msg='individual label guard: ' + field)
            for field in ('alpha', 'b', 'history', 'decisions'):
                if fixed_basis:
                    assert_array_equal(event[field], anchor[field], err_msg='shared-basis exact guard: ' + field)
                else:
                    observed.assert_continuous(event[field], anchor[field], 'native companion: ' + field)
        for phase in ('baseline', 'uninterrupted', 'resumed'):
            rows = [event for event in observed.events if event['phase'] == phase]
            self.assertFalse(any(rows[-1]['model'] is event['model'] for event in rows[:-1]),
                             'fresh final model guard')
        if memo is not None:
            self.assertEqual(len(memo.cache), 3, 'distinct fold/full basis guard')
            self.assertEqual(memo.native_calls, 3)
            self.assertGreater(memo.replays, 0)

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
        mutations = [dict(schema=True), dict(schema=1), dict(fold_index=99),
                     dict(candidate_index=99), dict(train_count=999),
                     dict(parameter_sha256='0' * 64),
                     dict(train_score=1.01), dict(test_score=True),
                     dict(test_correct=True), dict(train_correct=-1),
                     dict(test_correct=payload['test_correct'] + 1),
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
        report = result['finite_grid_report']
        self.assertEqual(report['status'], 'available')
        self.assertEqual(report['scope'], 'best_among_supplied_candidates')
        self.assertEqual(report['best_index'], 0)
        self.assertEqual(report['best_params'], candidates[0])
        self.assertEqual(report['candidate_count'], len(candidates))
        self.assertEqual(report['tied_best_count'], len(candidates))
        self.assertEqual(report['tied_best_indices'], list(range(len(candidates))))
        self.assertTrue(report['all_flat'])
        self.assertEqual(report['best_score_exact'], {'numerator': 1, 'denominator': 2})
        for fold in result['all_cv_results']:
            self.assertEqual(fold['params'], candidates)

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
