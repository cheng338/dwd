"""Explicit Zig builds must not inherit the build machine's instruction set."""
import os
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

from setuptools import Distribution, Extension


class NativeBuildPolicyTests(unittest.TestCase):
    def build_command(self, platform):
        with patch('setuptools.setup'):
            namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'setup.py'))
        command = namespace['StrictBuildExt'](Distribution())
        command.extensions = [Extension('dwd._residual_accel', ['residual_core.c'])]
        return command

    def test_zig_uses_baseline_x86_64_and_strict_arithmetic(self):
        command = self.build_command('win-amd64')
        with tempfile.TemporaryDirectory() as temporary, \
                patch.dict(os.environ, {'DWD_BUILD_ZIG': 'zig.exe'}), \
                patch('sys.platform', 'win32'), \
                patch('sysconfig.get_platform', return_value='win-amd64'), \
                patch.object(command, 'get_ext_fullpath', return_value=str(Path(temporary) / 'module.pyd')), \
                patch('subprocess.run') as run:
            command.build_extensions()
        args = run.call_args.args[0]
        self.assertEqual([arg for arg in args if arg.startswith('-march=')], ['-march=x86_64'])
        for flag in ['-ffp-contract=off', '-fno-fast-math', '-fno-associative-math',
                     '-fno-unsafe-math-optimizations', '-fno-finite-math-only']:
            self.assertIn(flag, args)
        self.assertTrue(run.call_args.kwargs['check'])

    def test_zig_rejects_non_amd64_python_before_compiling(self):
        for platform in ['win32', 'win-arm64']:
            with self.subTest(platform=platform):
                command = self.build_command(platform)
                with patch.dict(os.environ, {'DWD_BUILD_ZIG': 'zig.exe'}), \
                        patch('sys.platform', 'win32'), \
                        patch('sysconfig.get_platform', return_value=platform), \
                        patch('subprocess.run') as run:
                    with self.assertRaisesRegex(RuntimeError, 'Windows AMD64'):
                        command.build_extensions()
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
