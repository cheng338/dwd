"""Compiled eligibility is independent of the scalar sumprod fallback."""
from fractions import Fraction
import math
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dwd import _compiled_residual as compiled
from dwd import _compiled_scores as scores
from dwd import _native_residual as native
from dwd.profiling import residual_profile


def runtime(version, platform='win32', implementation='cpython'):
    return SimpleNamespace(version_info=version, platform=platform,
        implementation=SimpleNamespace(name=implementation), maxsize=2**63 - 1,
        float_info=sys.float_info)


class NativeRuntimeEligibilityTests(unittest.TestCase):
    def test_scalar_runtime_matrix_accepts_later_python3_versions(self):
        # A substitute builtin checks eligibility only, never its arithmetic.
        for version, os_name, machine, gil_disabled, expected in (
                ((3, 11), 'win32', 'AMD64', 0, False),
                ((3, 12), 'win32', 'AMD64', 0, True),
                ((3, 12), 'linux', 'aarch64', None, True),
                ((3, 12), 'win32', 'AMD64', 1, False),
                ((3, 13), 'win32', 'AMD64', 0, True),
                ((3, 13), 'win32', 'x86_64', None, True),
                ((3, 13), 'win32', 'AMD64', 1, False),
                ((3, 13), 'win32', 'AMD64', 'unknown', False),
                ((3, 13), 'linux', 'x86_64', 0, True),
                ((3, 13), 'darwin', 'x86_64', 0, True),
                ((3, 13), 'win32', 'ARM64', 0, True),
                ((3, 14), 'win32', 'AMD64', 0, True),
                ((3, 15), 'linux', 'aarch64', 0, True),
                ((3, 99), 'darwin', 'arm64', None, True),
                ((3, 99), 'win32', 'AMD64', 1, False),
                ((3, 99), 'win32', 'AMD64', 'unknown', False),
                ((4, 0), 'win32', 'AMD64', 0, False)):
            with self.subTest(version=version, platform=os_name, machine=machine,
                              gil_disabled=gil_disabled), \
                    patch.object(native, 'sys', runtime(version, os_name)), \
                    patch.object(native.platform, 'machine', return_value=machine), \
                    patch.object(native, '_GIL_DISABLED', gil_disabled), \
                    patch.object(native.math, 'sumprod', math.fsum, create=True):
                self.assertEqual(native._scalar_supported(), expected)
        for implementation, maxsize, expected in (
                ('pypy', 2**63 - 1, False), ('cpython', 2**31 - 1, True)):
            identity = runtime((3, 13), implementation=implementation)
            identity.maxsize = maxsize
            with patch.object(native, 'sys', identity), \
                    patch.object(native.platform, 'machine', return_value='AMD64'), \
                    patch.object(native, '_GIL_DISABLED', 0), \
                    patch.object(native.math, 'sumprod', math.fsum, create=True):
                self.assertEqual(native._scalar_supported(), expected)

    def test_compiled_runtime_matrix_preserves_portable_exclusions(self):
        # Identity substitution tests control flow, not another runtime's math.
        for version in ((3, 10), (3, 11), (3, 12), (3, 13), (3, 14),
                        (3, 15), (3, 99), (4, 0)):
            with patch.object(native, 'sys', runtime(version)), \
                    patch.object(scores, 'sys', runtime(version)), \
                    patch.object(native.platform, 'machine', return_value='AMD64'), \
                    patch.object(native, '_GIL_DISABLED', 0), \
                    patch.object(scores, '_GIL_DISABLED', 0):
                expected = version[0] == 3 and version >= (3, 11)
                self.assertEqual(native._compiled_runtime_supported(), expected)
                self.assertEqual(scores._supported(), expected)
        for os_name, machine, implementation in (
                ('linux', 'x86_64', 'cpython'), ('darwin', 'x86_64', 'cpython'),
                ('win32', 'ARM64', 'cpython'), ('win32', 'AMD64', 'pypy')):
            with patch.object(native, 'sys', runtime((3, 13), os_name, implementation)), \
                    patch.object(scores, 'sys', runtime((3, 13), os_name, implementation)), \
                    patch.object(native.platform, 'machine', return_value=machine):
                self.assertFalse(native._compiled_runtime_supported())
                self.assertFalse(scores._supported())
        for version in ((3, 11), (3, 12), (3, 13), (3, 99)):
            for disabled in (None, 0, 1, 'unknown'):
                with self.subTest(version=version, gil_disabled=disabled), \
                        patch.object(native, 'sys', runtime(version)), \
                        patch.object(scores, 'sys', runtime(version)), \
                        patch.object(native.platform, 'machine', return_value='AMD64'), \
                        patch.object(native, '_GIL_DISABLED', disabled), \
                        patch.object(scores, '_GIL_DISABLED', disabled):
                    expected = disabled in (None, 0)
                    self.assertEqual(native._compiled_runtime_supported(), expected)
                    self.assertEqual(scores._supported(), expected)

    def test_later_versions_retain_builtin_and_binary64_guards(self):
        with patch.object(native, 'sys', runtime((3, 99))), \
                patch.object(native, '_GIL_DISABLED', 0):
            for replacement in (None, lambda a, b: 0.):
                with patch.object(native.math, 'sumprod', replacement, create=True):
                    self.assertFalse(native._scalar_supported())
        for field, bad in (('radix', 10), ('mant_dig', 24), ('max_exp', 128)):
            identity = runtime((3, 99))
            fields = dict(radix=2, mant_dig=53, max_exp=1024)
            fields[field] = bad
            identity.float_info = SimpleNamespace(**fields)
            with self.subTest(field=field), patch.object(native, 'sys', identity), \
                    patch.object(scores, 'sys', identity), \
                    patch.object(native.platform, 'machine', return_value='AMD64'), \
                    patch.object(native, '_GIL_DISABLED', 0), \
                    patch.object(scores, '_GIL_DISABLED', 0), \
                    patch.object(native.math, 'sumprod', math.fsum, create=True):
                self.assertFalse(native._scalar_supported())
                self.assertFalse(native._compiled_runtime_supported())
                self.assertFalse(scores._supported())

    def test_later_versions_keep_compiled_64bit_requirement(self):
        for version in ((3, 14), (3, 99)):
            identity = runtime(version)
            identity.maxsize = 2**31 - 1
            with self.subTest(version=version), \
                    patch.object(native, 'sys', identity), \
                    patch.object(scores, 'sys', identity), \
                    patch.object(native.platform, 'machine', return_value='AMD64'), \
                    patch.object(native, '_GIL_DISABLED', 0), \
                    patch.object(scores, '_GIL_DISABLED', 0):
                self.assertFalse(native._compiled_runtime_supported())
                self.assertFalse(scores._supported())

    def test_later_nonwindows_scalar_runtime_can_try_local_extension(self):
        # Preserve the existing 3.12 dispatch order on later eligible runtimes.
        # Substituted identities exercise control flow, not other platforms' math.
        inputs = (np.eye(8), .5, np.zeros(8), np.ones(8), .25, 0.)
        for version in ((3, 13), (3, 14), (3, 99)):
            with self.subTest(version=version), \
                    patch.object(native, 'sys', runtime(version, 'linux')), \
                    patch.object(native.platform, 'machine', return_value='aarch64'), \
                    patch.object(native, '_GIL_DISABLED', 0), \
                    patch.object(native.math, 'sumprod', math.fsum, create=True), \
                    patch.object(compiled, '_ACCEL', SimpleNamespace(evaluate=lambda *a: 2)), \
                    patch.object(native, 'compiled_values',
                                 return_value=compiled._COMPILED_DOMAIN_REFUSAL) as entry:
                self.assertTrue(native._scalar_supported())
                self.assertFalse(native._compiled_runtime_supported())
                self.assertIsNone(native.native_compensated_residual(*inputs))
                entry.assert_called_once()

    def test_compiled_refusal_never_uses_unqualified_scalar_builtin(self):
        inputs = (np.eye(8), .5, np.zeros(8), np.ones(8), .25, 0.)
        for version in ((3, 11), (3, 13)):
            with self.subTest(version=version), \
                    patch.object(native, 'sys', runtime(version)), \
                    patch.object(native.platform, 'machine', return_value='AMD64'), \
                    patch.object(compiled, '_ACCEL', SimpleNamespace(evaluate=lambda *a: 2)), \
                    patch.object(native, 'compiled_values', return_value=None), \
                    patch.object(native.math, 'sumprod', side_effect=AssertionError(
                        'Unqualified scalar builtin called'), create=True), residual_profile() as profile:
                self.assertTrue(native._native_supported())
                self.assertFalse(native._scalar_supported())
                self.assertIsNone(native.native_compensated_residual(*inputs))
            observed = profile.as_dict()
            self.assertEqual(observed['routes']['compiled_dispatch']['declined'], 1)
            self.assertEqual(observed['decline_reasons'], {'scalar_runtime_unqualified': 1})
            self.assertEqual(observed['counts']['scalar_sumprod_calls_started'], 0)

    def test_missing_extension_returns_before_dispatch_when_scalar_unqualified(self):
        for version, gil_disabled in (((3, 11), 0), ((3, 13), 1),
                                      ((3, 99), 1), ((4, 0), 0)):
            with patch.object(native, 'sys', runtime(version)), \
                    patch.object(native, '_GIL_DISABLED', gil_disabled), \
                    patch.object(native.platform, 'machine', return_value='AMD64'), \
                    patch.object(compiled, '_ACCEL', None), \
                    patch.object(native, 'compiled_values', side_effect=AssertionError('No extension')):
                self.assertFalse(native._native_supported())
                self.assertIsNone(native.native_compensated_residual(
                    np.eye(8), .5, np.zeros(8), np.ones(8), .25, 0.))

    @unittest.skipUnless(native._scalar_supported(), 'Requires qualified scalar sumprod.')
    def test_qualified_scalar_fallback_remains_available_without_extension(self):
        with patch.object(compiled, '_ACCEL', None), residual_profile() as profile:
            self.assertTrue(native._native_supported())
            self.assertIsNotNone(native.native_compensated_residual(
                np.eye(2), .5, np.zeros(2), np.ones(2), .25, 0.))
        self.assertEqual(profile.as_dict()['routes']['scalar_fallback']['bounded_return'], 1)

    @unittest.skipUnless(native._scalar_supported(), 'Requires supported scalar sumprod.')
    def test_future_identity_keeps_current_scalar_arithmetic_and_fallback(self):
        # Actual arithmetic is the running interpreter's implementation. This
        # confirms dispatch and parity, not arithmetic on a future interpreter.
        inputs = (np.eye(2), .5, np.zeros(2), np.ones(2), .25, 0.)
        with patch.object(compiled, '_ACCEL', None):
            expected = native.native_compensated_residual(*inputs)
            for version in ((3, 14), (3, 15), (3, 99)):
                with self.subTest(version=version), \
                        patch.object(native, 'sys', runtime(version)), \
                        patch.object(native, '_GIL_DISABLED', 0):
                    actual = native.native_compensated_residual(*inputs)
                    self.assertIsNotNone(actual)
                    for observed, reference in zip(actual, expected):
                        self.assertEqual(np.asarray(observed).tobytes(),
                                         np.asarray(reference).tobytes())

    @unittest.skipUnless(compiled._ACCEL is not None and native._compiled_runtime_supported(),
                         'Requires the qualified compiled runtime and extension.')
    def test_actual_compiled_path_activates_with_exact_rational_bounds(self):
        rng = np.random.RandomState(182611)
        K = rng.uniform(.5, 1., (8, 8))
        x, rhs = rng.uniform(.5, 1., 8), rng.uniform(.5, 1., 8)
        with patch.object(compiled._ACCEL, 'evaluate', wraps=compiled._ACCEL.evaluate) as entry:
            result = native.native_compensated_residual(K, .125, rhs, x, -.25, .5)
        self.assertGreater(entry.call_count, 0)
        self.assertIsNotNone(result)
        residual, constraint, values, residual_bounds, constraint_bound, score_bounds = result
        exact = lambda value: Fraction.from_float(float(value))
        for i, row in enumerate(K):
            score = sum((exact(a) * exact(b) for a, b in zip(row, x)), Fraction())
            r = exact(rhs[i]) + exact(.25) - exact(.125) * exact(x[i]) - score
            self.assertLessEqual(abs(exact(values[i]) - score), exact(score_bounds[i]))
            self.assertLessEqual(abs(exact(residual[i]) - r), exact(residual_bounds[i]))
        c = exact(.5) - sum(map(exact, x), Fraction())
        self.assertLessEqual(abs(exact(constraint) - c), exact(constraint_bound))
        for version in ((3, 14), (3, 15), (3, 99)):
            with self.subTest(version=version), \
                    patch.object(native, 'sys', runtime(version)), \
                    patch.object(native.platform, 'machine', return_value='AMD64'), \
                    patch.object(native, '_GIL_DISABLED', 0), \
                    patch.object(compiled._ACCEL, 'evaluate', wraps=compiled._ACCEL.evaluate) as entry:
                actual = native.native_compensated_residual(K, .125, rhs, x, -.25, .5)
                self.assertGreater(entry.call_count, 0)
                for observed, reference in zip(actual, result):
                    self.assertEqual(np.asarray(observed).tobytes(),
                                     np.asarray(reference).tobytes())


if __name__ == '__main__':
    unittest.main()
