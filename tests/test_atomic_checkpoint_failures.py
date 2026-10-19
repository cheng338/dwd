"""Atomic receipts retain their primary failure and clean only owned files."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

EXAMPLES = str(Path(__file__).resolve().parents[1] / 'examples')
if EXAMPLES not in sys.path:
    sys.path.insert(0, EXAMPLES)
import _cv_checkpoint as checkpoint


class AtomicCheckpointFailureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / 'result.json'
        self.path.write_bytes(b'{"previous": true}\n')
        self.previous = self.path.read_bytes()

    def test_serialization_precedes_temp_creation_and_cleanup(self):
        failure = ValueError('serialization failed')
        with patch.object(checkpoint.json, 'dumps', side_effect=failure), \
                patch.object(Path, 'open') as opened, patch.object(Path, 'unlink') as unlinked:
            with self.assertRaises(ValueError) as caught:
                checkpoint.write_json_atomic(self.path, {'next': True})
        self.assertIs(caught.exception, failure)
        opened.assert_not_called()
        unlinked.assert_not_called()
        self.assertEqual(self.path.read_bytes(), self.previous)

    def test_replace_and_cleanup_failure_preserves_original_base_exception(self):
        for primary in (OSError('replace failed'), KeyboardInterrupt(), SystemExit(17)):
            with self.subTest(primary=type(primary).__name__), \
                    patch.object(checkpoint.os, 'replace', side_effect=primary), \
                    patch.object(Path, 'unlink', side_effect=PermissionError('cleanup failed')):
                with self.assertRaises(type(primary)) as caught:
                    checkpoint.write_json_atomic(self.path, {'next': True})
            self.assertIs(caught.exception, primary)
            self.assertEqual(self.path.read_bytes(), self.previous)
        self.assertEqual(len(list(self.root.glob('*.tmp'))), 3)

    def test_fsync_and_cleanup_failure_preserves_write_failure(self):
        primary = OSError('sync failed')
        with patch.object(checkpoint.os, 'fsync', side_effect=primary), \
                patch.object(Path, 'unlink', side_effect=PermissionError('cleanup failed')):
            with self.assertRaises(OSError) as caught:
                checkpoint.write_json_atomic(self.path, {'next': True})
        self.assertIs(caught.exception, primary)
        self.assertEqual(self.path.read_bytes(), self.previous)

    def test_exclusive_open_collision_does_not_clean_an_unowned_file(self):
        collision = self.path.with_name(self.path.name + '.collision.tmp')
        collision.write_bytes(b'belongs to someone else')
        with patch.object(checkpoint.uuid, 'uuid4', return_value=SimpleNamespace(hex='collision')), \
                patch.object(Path, 'unlink') as unlinked:
            with self.assertRaises(FileExistsError):
                checkpoint.write_json_atomic(self.path, {'next': True})
        unlinked.assert_not_called()
        self.assertEqual(collision.read_bytes(), b'belongs to someone else')
        self.assertEqual(self.path.read_bytes(), self.previous)

    def test_required_write_failure_is_strict_and_normal_cleanup_succeeds(self):
        failure = PermissionError('cannot publish')
        with patch.object(checkpoint.os, 'replace', side_effect=failure):
            with self.assertRaises(PermissionError) as caught:
                checkpoint.write_json_atomic(self.path, {'next': True})
        self.assertIs(caught.exception, failure)
        self.assertEqual(self.path.read_bytes(), self.previous)
        self.assertEqual(list(self.root.glob('*.tmp')), [])

    def test_exclusive_publication_failure_keeps_existing_destination(self):
        with patch.object(Path, 'unlink', side_effect=PermissionError('cleanup failed')):
            with self.assertRaises(FileExistsError):
                checkpoint.write_json_atomic(self.path, {'next': True}, replace=False)
        self.assertEqual(self.path.read_bytes(), self.previous)

    def test_cleanup_failure_after_successful_link_is_not_suppressed(self):
        fresh = self.root / 'fresh.json'
        failure = PermissionError('cleanup failed')
        with patch.object(Path, 'unlink', side_effect=failure):
            with self.assertRaises(PermissionError) as caught:
                checkpoint.write_json_atomic(fresh, {'next': True}, replace=False)
        self.assertIs(caught.exception, failure)
        self.assertEqual(json.loads(fresh.read_text()), {'next': True})

    def test_success_preserves_format_and_leaves_no_temporary_file(self):
        payload = {'b': [1, 'é'], 'a': True}
        checkpoint.write_json_atomic(self.path, payload)
        self.assertEqual(self.path.read_bytes(),
                         (json.dumps(payload, sort_keys=True, allow_nan=False) + '\n').encode())
        self.assertEqual(list(self.root.glob('*.tmp')), [])


if __name__ == '__main__':
    unittest.main()
