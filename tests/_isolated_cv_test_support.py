"""Single-case subprocess isolation for resumable-CV tests only."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


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


def run_isolated_test(case, result, *, module_path):
    own_result = result is None
    if own_result:
        result = case.defaultTestResult()
        result.startTestRun()
    result.startTest(case)
    try:
        with tempfile.TemporaryDirectory(prefix='dwd-cv-isolated-') as directory:
            receipt = Path(directory) / 'result.json'
            process = subprocess.run(
                [sys.executable, '-B', '-X', 'utf8', '-c', _ISOLATED_TEST_SCRIPT,
                 str(receipt), str(Path(module_path).resolve()),
                 type(case).__name__, case._testMethodName, json.dumps(sys.path)],
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
                    report(case, (exception_type, error, None))
                    reported = True
            for description, reason in child['skipped']:
                result.addSkip(case, description + ': ' + reason)
                reported = True
            for description in child['unexpected_successes']:
                result.addUnexpectedSuccess(case)
                reported = True
            if not reported:
                result.addSuccess(case)
    except Exception:
        result.addError(case, sys.exc_info())
    finally:
        result.stopTest(case)
        if own_result:
            result.stopTestRun()
    return result
