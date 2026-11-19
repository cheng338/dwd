"""Check startup-test coverage without claiming future-interpreter execution."""
from types import SimpleNamespace

import pytest

import test_residual_startup_cleanup as startup_tests
from dwd import _residual_startup_cleanup as cleanup


@pytest.mark.parametrize("implementation,version,required", [
    pytest.param("cpython", (3, 11), True, id="cpython311"),
    pytest.param("cpython", (3, 12), True, id="cpython312"),
    pytest.param("cpython", (3, 15), True, id="cpython315"),
    pytest.param("cpython", (3, 10), False, id="cpython310"),
    pytest.param("cpython", (4, 0), False, id="cpython400"),
    pytest.param("pypy", (3, 12), False, id="pypy312"),
])
def test_startup_capability_assertion_policy(
        implementation, version, required, monkeypatch):
    real_identity = cleanup.sys
    fake_identity = SimpleNamespace(
        implementation=SimpleNamespace(name=implementation), version_info=version)
    for result in (None, 123):
        calls = []

        def probe():
            calls.append(True)
            return result

        with monkeypatch.context() as scoped:
            # Change the startup test module's identity, never the identity
            # used by the production capability detector.
            scoped.setattr(startup_tests, "sys", fake_identity)
            scoped.setattr(cleanup, "_qualified_wait_line", probe)
            assert cleanup.sys is real_identity
            if required and result is None:
                with pytest.raises(AssertionError):
                    startup_tests.test_supported_cpython_startup_capability()
            else:
                startup_tests.test_supported_cpython_startup_capability()
            assert len(calls) == int(required)
    assert cleanup.sys is real_identity
