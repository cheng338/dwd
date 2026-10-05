"""Accelerator identity and resume checks for the checkpointed CV example."""
from contextlib import contextmanager, ExitStack
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import _cv_checkpoint as checkpoint


class AcceleratorIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='dwd-native-identity-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'dwd'
        self.root.mkdir()
        self.file = self.root / '_native_accel.pyd'
        self.file.write_bytes(b'native-artifact-one')
        self.sha = hashlib.sha256(self.file.read_bytes()).hexdigest()
        self.files = {str(self.file): self.sha}

    def module(self, origin=None):
        result = types.ModuleType('test_accelerator')
        result.__file__ = str(self.file if origin is None else origin)
        result.evaluate = lambda: None
        return result

    def identify(self, module, native=False, files=None):
        return checkpoint._accelerator_identity(
            module, native, self.files if files is None else files, self.root)

    def test_unavailable_flags(self):
        left = self.identify(None, False)
        right = self.identify(None, True)
        self.assertEqual(left, dict(available=False, native_screen_supported=False,
                                    artifact=None, sha256=None))
        self.assertEqual(right, dict(available=False, native_screen_supported=True,
                                     artifact=None, sha256=None))
        self.assertNotEqual(checkpoint.digest(left), checkpoint.digest(right))

    def test_valid_binding_origin_hash_and_flags(self):
        for flag in (False, True):
            with self.subTest(flag=flag):
                self.assertEqual(self.identify(self.module(), flag), dict(
                    available=True, native_screen_supported=flag,
                    artifact=str(self.file), sha256=self.sha))

    def test_two_inventoried_bindings_distinguish(self):
        other = self.root / 'second.pyd'
        other.write_bytes(b'native-artifact-two')
        files = {**self.files, str(other): hashlib.sha256(other.read_bytes()).hexdigest()}
        first = self.identify(self.module(), files=files)
        second = self.identify(self.module(other), files=files)
        self.assertNotEqual(first['artifact'], second['artifact'])
        self.assertNotEqual(first['sha256'], second['sha256'])
        self.assertNotEqual(checkpoint.digest(first), checkpoint.digest(second))

    def test_reject_nonmodule(self):
        with self.assertRaises(checkpoint.CheckpointError):
            self.identify(types.SimpleNamespace(
                __file__=str(self.file), evaluate=lambda: None))

    def test_reject_no_evaluate(self):
        module = self.module()
        del module.evaluate
        with self.assertRaises(checkpoint.CheckpointError):
            self.identify(module)

    def test_reject_noncallable_evaluate(self):
        module = self.module()
        module.evaluate = 1
        with self.assertRaises(checkpoint.CheckpointError):
            self.identify(module)

    def test_reject_missing_or_invalid_origin(self):
        for origin in (None, '', 7):
            module = self.module()
            module.__file__ = origin
            with self.subTest(origin=origin), self.assertRaises(checkpoint.CheckpointError):
                self.identify(module)

    def test_reject_missing_file(self):
        with self.assertRaises(checkpoint.CheckpointError):
            self.identify(self.module(self.root / 'missing.pyd'))

    def test_reject_directory_origin_not_in_inventory(self):
        directory = self.root / 'directory.pyd'
        directory.mkdir()
        with self.assertRaises(checkpoint.CheckpointError):
            self.identify(self.module(directory))

    def test_reject_non_native_extension(self):
        source = self.root / 'helper.py'
        source.write_text('# source', encoding='utf-8')
        files = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
        with self.assertRaises(checkpoint.CheckpointError):
            self.identify(self.module(source), files=files)

    def test_reject_outside_dwd(self):
        outside = Path(self.temp.name) / 'foreign.pyd'
        outside.write_bytes(b'outside')
        files = {**self.files, str(outside): hashlib.sha256(outside.read_bytes()).hexdigest()}
        with self.assertRaises(checkpoint.CheckpointError):
            self.identify(self.module(outside), files=files)

    def test_reject_ambiguous_inventory_aliases(self):
        inner = self.root / 'sub'
        inner.mkdir()
        alias = inner / '..' / self.file.name
        with self.assertRaises(checkpoint.CheckpointError):
            self.identify(self.module(), files={**self.files, str(alias): self.sha})

    def test_origin_alias_matches_same_inventory_file(self):
        inner = self.root / 'sub'
        inner.mkdir()
        alias = inner / '..' / self.file.name
        self.assertEqual(self.identify(self.module(alias)), self.identify(self.module()))

    def test_output_field_mutations_do_not_change_later_identity(self):
        expected = self.identify(self.module(), True)
        changed = self.identify(self.module(), True)
        changed.update(available=False, native_screen_supported=False,
                       artifact='changed', sha256='changed')
        self.assertEqual(self.identify(self.module(), True), expected)
        self.assertEqual(self.files, {str(self.file): self.sha})


class RuntimeAcceleratorIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='dwd-runtime-identity-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'dwd'
        self.root.mkdir()
        self.origin = self.root / '__init__.py'
        self.origin.write_text('__version__ = "fixture"\n', encoding='utf-8')
        self.native = self.root / '_native_accel.pyd'
        self.native.write_bytes(b'fixed-native-artifact')
        self.accelerator = types.ModuleType('test_accelerator')
        self.accelerator.__file__ = str(self.native)
        self.accelerator.evaluate = lambda: None

    @contextmanager
    def runtime(self, available=True, native_supported=True):
        # A fresh helper owns its caches. The inventory stays fixed across
        # capability states, without scanning the installed numerical packages.
        from dwd import _compiled_residual, _native_residual
        spec = importlib.util.spec_from_file_location(
            '_checkpoint_identity_fixture', EXAMPLES / '_cv_checkpoint.py')
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        package = types.ModuleType('dwd')
        package.__file__ = str(self.origin)
        package.__path__ = [str(self.root)]
        package.__version__ = 'fixture'
        import_module = helper.importlib.import_module

        def import_fixture(name):
            return package if name == 'dwd' else import_module(name)

        pool = dict(user_api='blas', num_threads=1, filepath=str(self.native))
        controller = types.SimpleNamespace(info=lambda: [dict(pool)])
        with ExitStack() as stack:
            stack.enter_context(patch.object(
                _compiled_residual, '_ACCEL', self.accelerator if available else None))
            stack.enter_context(patch.object(
                _native_residual, '_native_supported', return_value=native_supported))
            stack.enter_context(patch.object(
                helper, '_runtime_package_names', return_value=[('dwd', 'dwd')]))
            stack.enter_context(patch.object(
                helper, '_metadata_identity', return_value='fixture'))
            stack.enter_context(patch.object(
                helper.importlib, 'import_module', side_effect=import_fixture))
            stack.enter_context(patch.object(helper, '_loaded_source_guards'))
            stack.enter_context(patch.object(
                helper, 'ThreadpoolController', return_value=controller))
            yield helper

    def test_runtime_serialization_distinguishes_capabilities_with_identical_files(self):
        fingerprints = []
        for available in (False, True):
            for native_supported in (False, True):
                with self.subTest(available=available, native_supported=native_supported):
                    with self.runtime(available, native_supported) as helper:
                        fingerprint = helper.runtime_fingerprint()
                    encoded = json.dumps(fingerprint, allow_nan=False, sort_keys=True)
                    self.assertEqual(json.loads(encoded), fingerprint)
                    self.assertEqual(fingerprint['accelerator'], dict(
                        available=available, native_screen_supported=native_supported,
                        artifact=str(self.native) if available else None,
                        sha256=hashlib.sha256(self.native.read_bytes()).hexdigest()
                        if available else None))
                    fingerprints.append(fingerprint)
        self.assertEqual(len({item['files_sha256'] for item in fingerprints}), 1)
        self.assertEqual(len({checkpoint.digest(item) for item in fingerprints}), 4)

    def test_returned_accelerator_fields_do_not_mutate_cached_runtime(self):
        with self.runtime() as helper:
            first = helper.runtime_fingerprint()
            expected = copy.deepcopy(first['accelerator'])
            first['accelerator'].update(available=False, artifact='changed', sha256='changed')
            first['accelerator']['extra'] = True
            for deep in (False, True):
                with self.subTest(check_inventory=deep):
                    again = helper.runtime_fingerprint(check_inventory=deep)
                    self.assertEqual(again['accelerator'], expected)
                    self.assertIsNot(again['accelerator'], first['accelerator'])

    def test_runtime_keeps_in_process_availability_and_native_guards(self):
        from dwd import _compiled_residual, _native_residual
        with self.runtime() as helper:
            helper.runtime_fingerprint()
            with patch.object(_compiled_residual, '_ACCEL', None):
                with self.assertRaises(helper.CheckpointError):
                    helper.runtime_fingerprint(check_inventory=False)
            with patch.object(_native_residual, '_native_supported', return_value=False):
                with self.assertRaises(helper.CheckpointError):
                    helper.runtime_fingerprint(check_inventory=False)

    def test_resume_rejects_stale_or_missing_capabilities_before_new_generation(self):
        with self.runtime() as helper:
            current = {'runtime': helper.runtime_fingerprint()}
            missing = copy.deepcopy(current)
            del missing['runtime']['accelerator']
            stale_availability = copy.deepcopy(current)
            stale_availability['runtime']['accelerator'].update(
                available=False, artifact=None, sha256=None)
            stale_native = copy.deepcopy(current)
            stale_native['runtime']['accelerator']['native_screen_supported'] = False
            stale_binding = copy.deepcopy(current)
            stale_binding['runtime']['accelerator']['sha256'] = '0' * 64
            for index, old in enumerate((missing, stale_availability, stale_native, stale_binding)):
                with self.subTest(previous_identity=index):
                    directory = Path(self.temp.name) / ('run-' + str(index))
                    with helper.CheckpointRun(directory, old) as run:
                        token = run.token
                    before = {path.name: path.read_bytes()
                              for path in directory.iterdir() if path.suffix == '.json'}
                    with self.assertRaises(helper.CheckpointError):
                        with helper.CheckpointRun(directory, current, resume=True):
                            self.fail('incompatible receipt was reused')
                    after = {path.name: path.read_bytes()
                             for path in directory.iterdir() if path.suffix == '.json'}
                    self.assertEqual(after, before)
                    helper.assert_generation(directory, token)
            directory = Path(self.temp.name) / 'unchanged'
            with helper.CheckpointRun(directory, current) as first:
                previous = first.token
            with helper.CheckpointRun(directory, current, resume=True) as resumed:
                self.assertNotEqual(resumed.token, previous)


if __name__ == '__main__':
    unittest.main(verbosity=2)
