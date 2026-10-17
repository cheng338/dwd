"""Explicit resource planning and identity for the resumable CV example."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
sys.path.insert(0, str(EXAMPLES)) if str(EXAMPLES) not in sys.path else None
import resumable_kernel_cv as example
import tune_kernel_dwd as command


class ResourcePlanTests(unittest.TestCase):
    def test_allowances_require_a_complete_pair(self):
        for values in ((100, None), (None, 100)):
            with self.assertRaisesRegex(ValueError, 'together'):
                example._memory_allowances(*values)

    def test_invalid_allowances_fail_before_fitting(self):
        for value in (0, -1, True, np.bool_(True), float('nan'), float('inf'), '2'):
            for values in ((value, 1), (2, value)):
                with self.subTest(values=values), self.assertRaises(ValueError):
                    example._memory_allowances(*values)
        with self.assertRaisesRegex(ValueError, 'one worker'):
            example._memory_allowances(1, 2)

    def test_worker_cap_respects_fold_count_and_user_allowance(self):
        plan = example._worker_plan(8, 5, 4, 10000., 4000.)
        self.assertEqual(plan['workers'], 2)
        self.assertEqual(plan['nominal_compute_threads'], 8)
        self.assertEqual(plan['memory_basis'], 'caller_supplied_allowances')
        self.assertEqual(example._worker_plan(4, 5, 4, None, None)['workers'], 4)
        self.assertEqual(example._worker_plan(4, 0, 4, 10000., 4000.)['workers'], 0)

    def test_memory_division_never_rounds_up_a_boundary(self):
        self.assertEqual(example._worker_plan(8, 5, 1, .3, .1)['workers'], 2)

    def test_all_cpu_request_is_still_capped_by_folds(self):
        with patch.object(example.os, 'cpu_count', return_value=32):
            self.assertEqual(example._worker_plan(-1, 5, 1, None, None)['workers'], 5)
            divided = example._worker_plan(-1, 5, 16, None, None)
            self.assertEqual(divided['workers'], 2)
            self.assertEqual(divided['nominal_compute_threads'], 32)
            self.assertEqual(divided['automatic_cpu_budget'], 32)
            self.assertEqual(example._worker_plan(4, 5, 16, None, None)['workers'], 4)
        with patch.object(example.os, 'cpu_count', return_value=None):
            self.assertEqual(example._worker_plan(-1, 5, 4, None, None)['workers'], 1)

    def test_cli_forwards_budgets(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'train.npz'
            np.savez(data, X=np.ones((10, 2)), y=np.arange(10) % 2)
            with patch.object(command, 'run_resumable_cv', return_value={
                    'best_score': 1., 'best_params': {}, 'work_summary': {},
                    'finite_grid_report': {
                        'status': 'unavailable',
                        'scope': 'best_among_supplied_candidates',
                        'reporting_error': 'RuntimeError'}}) as run, redirect_stdout(io.StringIO()):
                command.main(['--data', str(data), '--run-dir', str(Path(directory) / 'run'),
                              '--lambd', '.1', '--gamma', '.1', '--jobs', '4',
                              '--cv-native-threads', '4', '--cv-memory-budget-mib', '8000',
                              '--cv-worker-memory-mib', '2000'])
            self.assertEqual(run.call_args.kwargs['cv_native_threads'], 4)
            self.assertEqual(run.call_args.kwargs['cv_memory_budget_mib'], 8000.)
            self.assertEqual(run.call_args.kwargs['cv_worker_memory_mib'], 2000.)

    def test_real_scopes_resume_and_parallel_workers_in_fresh_process(self):
        # Runtime source guards intentionally reject historical fixtures imported
        # by other tests. Keep this check independent of their import order.
        script = r'''
import json, sys
from pathlib import Path
sys.path[:] = json.loads(sys.argv[2])
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_info, threadpool_limits
from unittest.mock import patch
from dwd.gen_kern_dwd import KernGDWD
import resumable_kernel_cv as cv
from _cv_checkpoint import CheckpointError
root = Path(sys.argv[1])
rng = np.random.RandomState(12)
X = rng.normal(size=(24, 4))
y = np.tile([-1, 1], 12)
folds = list(StratifiedKFold(2, shuffle=True, random_state=17).split(X, y))
def run(path, **options):
    model = KernGDWD(kernel='rbf', kernel_kws={'gamma': .12}, random_state=31,
                    stopping='fixed', max_iter=3)
    return cv.run_resumable_cv(model, X, y, {'lambd': [.05, .2]}, path,
                              cv=folds, scale=True, **options)
# MKL can load its OpenMP runtime on the first numerical call. Initialize it
# before the outer scope snapshots the pools whose limits it must restore.
np.dot(np.ones((2, 2)), np.ones((2, 2)))
with threadpool_limits(3):
    assert all(pool['num_threads'] == 3 for pool in threadpool_info())
    serial = run(root / 'one', cv_native_threads=1)
    parallel = run(root / 'two', jobs=2, cv_native_threads=2,
                   cv_memory_budget_mib=4000, cv_worker_memory_mib=2000,
                   final_native_threads=1)
    assert all(pool['num_threads'] == 3 for pool in threadpool_info())
    assert parallel['work_summary']['fold_workers'] == 2
    assert parallel['work_summary']['native_threads_per_worker'] == 2
    assert all(p['num_threads'] == 1 for p in parallel['work_summary']['final_refit']['runtime']['native_pools'])
    assert serial['best_params'] == parallel['best_params']
    np.testing.assert_array_equal(serial['agg_results']['mean_test_score'], parallel['agg_results']['mean_test_score'])
    np.testing.assert_array_equal(serial['best_clf'].predict(serial['scaler'].transform(X)),
                                  parallel['best_clf'].predict(parallel['scaler'].transform(X)))
    identity = json.loads((root / 'two' / 'identity.json').read_text())
    old_receipts = {str(p): p.read_bytes() for p in (root / 'two' / 'receipts').rglob('*.json')}
    replay = run(root / 'two', resume=True, jobs=1, cv_native_threads=2,
                 cv_memory_budget_mib=1000, cv_worker_memory_mib=1000,
                 final_native_threads=2)
    assert replay['work_summary']['fold_workers'] == 0
    assert replay['work_summary']['replayed_evaluations'] == 4
    assert all(p['num_threads'] == 2 for p in replay['work_summary']['final_refit']['runtime']['native_pools'])
    assert old_receipts == {str(p): p.read_bytes() for p in (root / 'two' / 'receipts').rglob('*.json')}
    try:
        run(root / 'two', resume=True, cv_native_threads=1)
    except CheckpointError:
        pass
    else:
        raise AssertionError('changed CV thread count was allowed to reuse receipts')
    assert all(pool['num_threads'] == 3 for pool in threadpool_info())
    original_fit = cv._fit_and_score
    calls = []
    def interrupt(*args, **kwargs):
        calls.append(1)
        assert all(pool['num_threads'] == 2 for pool in threadpool_info())
        if len(calls) == 2:
            raise RuntimeError('manufactured interruption')
        return original_fit(*args, **kwargs)
    with patch.object(cv, '_fit_and_score', interrupt):
        try:
            run(root / 'interrupted', cv_native_threads=2, refit_best=False)
        except RuntimeError as error:
            assert str(error) == 'manufactured interruption'
        else:
            raise AssertionError('interruption not exercised')
    assert all(pool['num_threads'] == 3 for pool in threadpool_info())
    continued = run(root / 'interrupted', cv_native_threads=2, resume=True, refit_best=False)
    assert continued['work_summary']['executed_evaluations'] == 3
    assert continued['work_summary']['replayed_evaluations'] == 1
    assert all(pool['num_threads'] == 3 for pool in threadpool_info())
    original_model_fit = KernGDWD.fit
    def changed_final_threads(model, data, labels, *args, **kwargs):
        fitted = original_model_fit(model, data, labels, *args, **kwargs)
        if len(data) == len(X):
            assert all(pool['num_threads'] == 1 for pool in threadpool_info())
            threadpool_limits(2)
        return fitted
    with patch.object(KernGDWD, 'fit', changed_final_threads):
        try:
            run(root / 'bad-final-runtime', cv_native_threads=4, final_native_threads=1)
        except CheckpointError:
            pass
        else:
            raise AssertionError('final runtime change hidden by restoring CV limits')
    assert all(pool['num_threads'] == 3 for pool in threadpool_info())
    assert not (root / 'bad-final-runtime' / 'cv-results.json').exists()
    assert len(list((root / 'bad-final-runtime' / 'receipts').rglob('*.json'))) == 4
print('thread scopes, numerical comparison, workers, replay and identity passed')
'''
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-B', '-c', script, directory,
                                     json.dumps(sys.path)], capture_output=True,
                                    text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
