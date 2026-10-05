"""Receipt identity, ownership and CLI boundaries for the standalone example."""
import contextlib
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
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import _cv_checkpoint as checkpoint
import tune_kernel_dwd as command


class StandaloneCheckpointTests(unittest.TestCase):
    def test_typed_identity_preserves_scalar_and_container_distinctions(self):
        values = [1, True, 1., np.float32(1), np.float64(1), [1], (1,),
                  {'value': 1}, ['dict', [['value', ['int', 1]]]], -0., 0.]
        identities = [checkpoint.digest(checkpoint.stable(value)) for value in values]
        self.assertEqual(len(set(identities)), len(values))
        with self.assertRaises(checkpoint.CheckpointError):
            checkpoint.stable(float('nan'))
        with self.assertRaises(checkpoint.CheckpointError):
            checkpoint.stable(lambda x: x)

    def test_array_identity_includes_shape_dtype_and_values(self):
        original = np.arange(8, dtype=np.float64)
        self.assertEqual(checkpoint.array_fingerprint(original),
                         checkpoint.array_fingerprint(original.copy()))
        for other in (original.reshape(2, 4), original.astype(np.float32), original[::-1]):
            self.assertNotEqual(checkpoint.array_fingerprint(original),
                                checkpoint.array_fingerprint(other))

    def test_receipt_rejects_checksum_identity_duplicate_keys_and_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'receipt.json'
            payload = {'identity_sha256': 'expected', 'value': .625}
            checkpoint.write_receipt(path, payload)
            self.assertEqual(checkpoint.read_receipt(path, 'expected'), payload)
            with self.assertRaises(checkpoint.CheckpointError):
                checkpoint.read_receipt(path, 'other')
            record = json.loads(path.read_text()); record['payload']['value'] = .5
            path.write_text(json.dumps(record))
            with self.assertRaises(checkpoint.CheckpointError):
                checkpoint.read_receipt(path, 'expected')
            for raw in ('{"payload":{},"payload":{},"sha256":"x"}', '{"payload":NaN}', '[]'):
                path.write_text(raw)
                with self.assertRaises(checkpoint.CheckpointError):
                    checkpoint.read_receipt(path, 'expected')

    def test_atomic_write_rejects_nonfinite_without_replacing_existing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'result.json'
            checkpoint.write_json_atomic(path, {'ok': 1})
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                checkpoint.write_json_atomic(path, {'bad': float('inf')})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(folder).glob('*.tmp')), [])
            with self.assertRaises(FileExistsError):
                checkpoint.write_json_atomic(path, {'ok': 2}, replace=False)
            self.assertEqual(path.read_bytes(), original)

    def test_run_identity_and_generation_reject_stale_workers(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'run'; identity = {'data': 'fixed'}
            with checkpoint.CheckpointRun(root, identity) as first:
                old = first.token
                first.assert_current()
                with checkpoint.worker_lease(root, old, 0):
                    checkpoint.assert_generation(root, old)
            with checkpoint.CheckpointRun(root, identity, resume=True) as second:
                self.assertNotEqual(old, second.token)
                with self.assertRaises(checkpoint.CheckpointError):
                    checkpoint.assert_generation(root, old)
                with self.assertRaises(checkpoint.CheckpointError):
                    with checkpoint.worker_lease(root, old, 0):
                        self.fail('stale worker entered')
            before = (root / 'identity.json').read_bytes()
            with self.assertRaises(checkpoint.CheckpointError):
                with checkpoint.CheckpointRun(root, {'data': 'changed'}, resume=True):
                    pass
            self.assertEqual((root / 'identity.json').read_bytes(), before)

    def test_live_worker_blocks_resume_without_invalidating_its_generation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'run'; identity = {'data': 'fixed'}
            with checkpoint.CheckpointRun(root, identity) as run:
                token = run.token
            code = ('import sys; sys.path.insert(0,sys.argv[1]); '
                    'from _cv_checkpoint import worker_lease; '
                    'lease=worker_lease(sys.argv[2],sys.argv[3],0); lease.__enter__(); '
                    'print("locked",flush=True); sys.stdin.readline(); lease.__exit__(None,None,None)')
            child = subprocess.Popen([sys.executable, '-B', '-c', code, str(EXAMPLES), str(root), token],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True)
            try:
                self.assertEqual(child.stdout.readline().strip(), 'locked')
                with self.assertRaises(checkpoint.CheckpointBusyError):
                    with checkpoint.CheckpointRun(root, identity, resume=True):
                        pass
                checkpoint.assert_generation(root, token)
            finally:
                child.communicate('\n', timeout=30)
            self.assertEqual(child.returncode, 0)
            with checkpoint.CheckpointRun(root, identity, resume=True):
                pass

    def test_runtime_hash_refuses_in_process_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'module.py'; path.write_text('value=1\n')
            checkpoint._file_hash(path)
            path.write_text('value=222\n')
            with self.assertRaises(checkpoint.CheckpointError):
                checkpoint._file_hash(path)

    def test_cli_validation_is_before_data_access(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for extra in (['--jobs', '0'], ['--folds', '1'], ['--gamma', 'nan'],
                          ['--seed', '-1'], ['--max-iter', '0']):
                with self.subTest(extra=extra), self.assertRaises(SystemExit) as raised:
                    command.main(['--data', 'absent.npz', '--run-dir', 'unused',
                                  '--lambd', '.1', '--gamma', '.2'] + extra)
                self.assertEqual(raised.exception.code, 2)

    def test_cli_uses_declared_grid_folds_and_seed_without_test_data(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'train.npz'
            X = np.arange(48).reshape(12, 4); y = np.tile([0, 1], 6)
            np.savez(path, X=X, y=y)
            with patch.object(command, 'run_resumable_cv', return_value={
                    'best_score': .5, 'best_params': {}, 'work_summary': {}}) as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    command.main(['--data', str(path), '--run-dir', str(Path(folder)/'run'),
                                  '--lambd', '.1', '.2', '--gamma', '.3', '.4',
                                  '--folds', '2', '--seed', '17', '--jobs', '2',
                                  '--scale', '--resume', '--cv-only'])
                args, kwargs = run.call_args
                self.assertEqual(args[0].random_state, 17)
                self.assertEqual(args[3], {'lambd': [.1, .2],
                                          'kernel_kws': [{'gamma': .3}, {'gamma': .4}]})
                self.assertTrue(kwargs['cv'].shuffle)
                self.assertEqual(kwargs['cv'].random_state, 17)
                self.assertEqual(kwargs['jobs'], 2)
                self.assertTrue(kwargs['scale']); self.assertTrue(kwargs['resume'])
                self.assertFalse(kwargs['refit_best'])
                np.testing.assert_array_equal(args[1], X)
                np.testing.assert_array_equal(args[2], y)


if __name__ == '__main__':
    unittest.main(verbosity=2)
