"""Required native-build policy checks without invoking native compilers.

These tests exercise setuptools' real error boundary with injected compiler
failures and check the final output contract. Separate artifact validation
checks that built native and portable wheels load and use their intended routes.
"""

from pathlib import Path
import importlib
import inspect
import runpy
import subprocess
import platform
import sys
import sysconfig
from types import SimpleNamespace

import pytest
import setuptools
from setuptools import Distribution
from setuptools.command.build_ext import build_ext as SetuptoolsBuildExt


STRICT_UNIX_FLAGS = [
    "-O3", "-g0", "-std=c11", "-ffp-contract=off", "-fno-fast-math",
    "-fno-associative-math", "-fno-unsafe-math-optimizations",
    "-fno-finite-math-only",
]
STRICT_MSVC_FLAGS = ["/O2", "/std:c11", "/fp:strict"]
EXPECTED_SOURCES = {
    "dwd._residual_accel": [
        "dwd/residual_bridge.c", "dwd/residual_core.c", "dwd/dense_score_core.c",
    ],
    "dwd._ordered_reduction_accel": ["dwd/ordered_reduction_bridge.c"],
    "dwd._feature_expansion_accel": ["dwd/feature_expansion_bridge.c"],
}

FEATURE_EXTENSION = 'dwd._feature_expansion_accel'

def expected_names(loaded):
    return list(EXPECTED_SOURCES)[:3 if loaded.namespace['feature_expansion'] else 2]


def expected_flags(extension, compiler_type):
    base = STRICT_MSVC_FLAGS if compiler_type == 'msvc' else STRICT_UNIX_FLAGS
    return base + (['-march=x86-64'] if extension.name == FEATURE_EXTENSION and compiler_type == 'unix' else [])


def require_index(loaded, index):
    if index >= len(loaded.extensions):
        pytest.skip('This interpreter build intentionally omits the full-ABI extension.')


@pytest.fixture
def load_setup(monkeypatch):
    source = Path(__file__).resolve().parents[1] / "setup.py"
    assert source.is_file(), f"Build-policy source does not exist: {source}"

    def load(mode=None):
        if mode is None:
            monkeypatch.delenv("DWD_BUILD_ACCEL", raising=False)
        else:
            monkeypatch.setenv("DWD_BUILD_ACCEL", mode)
        monkeypatch.delenv("DWD_BUILD_ZIG", raising=False)
        calls = []

        def capture_setup(*args, **kwargs):
            assert not args
            calls.append(kwargs)

        monkeypatch.setattr(setuptools, "setup", capture_setup)
        namespace = runpy.run_path(str(source))
        assert len(calls) == 1
        return SimpleNamespace(
            namespace=namespace,
            options=calls[0],
            command_class=calls[0]["cmdclass"]["build_ext"],
            extensions=calls[0]["ext_modules"],
        )

    return load


def build_command(loaded, compiler_type="unix"):
    command = loaded.command_class(Distribution({"ext_modules": loaded.extensions}))
    command.extensions = loaded.extensions
    command.compiler = SimpleNamespace(compiler_type=compiler_type)
    command.parallel = None
    return command


def compiler_error_classes(command):
    """Use the classes actually consumed by this loaded setuptools command.

    Importing setuptools._distutils directly can load a second copy alongside
    its distutils alias. Identically named exceptions from those copies are not
    interchangeable, so derive the exact classes from the real filter globals.
    """
    namespace = inspect.unwrap(type(command)._filter_build_errors).__globals__
    compile_error = namespace["CompileError"]
    compiler_base = namespace["CCompilerError"]
    distutils_base = namespace["DistutilsError"]
    c_errors = importlib.import_module(compile_error.__module__)
    compiler_errors = importlib.import_module(compiler_base.__module__)
    link_error = c_errors.LinkError
    platform_error = getattr(compiler_errors, "PlatformError", None)
    if platform_error is None:
        # Older setuptools defines this under the general distutils errors.
        platform_error = importlib.import_module(distutils_base.__module__).DistutilsPlatformError
    assert c_errors.CompileError is compile_error
    assert issubclass(compile_error, compiler_base)
    assert issubclass(link_error, compiler_base)
    assert issubclass(platform_error, (compiler_base, distutils_base))
    return {"compile": compile_error, "link": link_error, "msvc-platform": platform_error}


@pytest.mark.parametrize("mode", [None, "1", "2", "-1", "false", ""])
def test_nonzero_modes_request_all_qualified_strict_extensions(load_setup, mode):
    loaded = load_setup(mode)
    assert loaded.namespace["portable"] is False
    qualified = (sys.implementation.name == 'cpython' and sys.version_info[:2] == (3, 12)
                 and sys.platform in ('win32', 'linux') and sys.maxsize > 2**32
                 and platform.machine().lower() in ('amd64', 'x86_64')
                 and sysconfig.get_config_var('Py_GIL_DISABLED') in (None, 0))
    assert loaded.namespace['feature_expansion'] is qualified
    assert [extension.name for extension in loaded.extensions] == list(EXPECTED_SOURCES)[:3 if qualified else 2]
    for extension in loaded.extensions:
        fused = extension.name == FEATURE_EXTENSION
        assert extension.sources == EXPECTED_SOURCES[extension.name]
        assert extension.define_macros == ([] if fused else [("Py_LIMITED_API", "0x030B0000")])
        assert extension.py_limited_api is (not fused)
        assert extension.optional is False
    assert loaded.extensions[0].depends == []
    assert loaded.extensions[1].depends == ["dwd/dense_score_core.c"]
    if qualified:
        assert loaded.extensions[2].depends == ["dwd/feature_expansion_core.c", "dwd/dense_score_core.c"]
    assert loaded.options["options"] == ({} if qualified else {"bdist_wheel": {"py_limited_api": "cp311"}})


def test_zero_requests_portable_without_native_wheel_options(load_setup):
    loaded = load_setup("0")
    assert loaded.namespace["portable"] is True
    assert loaded.extensions == []
    assert loaded.options["options"] == {}


def test_portable_run_does_not_invoke_compilation_even_with_explicit_zig(
    load_setup, monkeypatch,
):
    loaded = load_setup("0")
    command = build_command(loaded)
    monkeypatch.setenv("DWD_BUILD_ZIG", "must-not-be-invoked.exe")

    def unexpected(*args, **kwargs):
        pytest.fail("Portable build invoked the native compilation path")

    monkeypatch.setattr(command, "build_extensions", unexpected)
    monkeypatch.setattr(subprocess, "run", unexpected)
    command.run()


@pytest.mark.parametrize("mode", [None, "1"])
@pytest.mark.parametrize("failed_index", [0, 1, 2], ids=["residual", "ordered", "feature"])
@pytest.mark.parametrize(
    "error_kind,compiler_type",
    [("compile", "unix"), ("link", "unix"), ("msvc-platform", "msvc")],
    ids=["compile", "link", "msvc-discovery"],
)
def test_standard_setuptools_failure_boundary(
    load_setup, monkeypatch, mode, failed_index, error_kind, compiler_type,
):
    loaded = load_setup(mode)
    require_index(loaded, failed_index)
    command = build_command(loaded, compiler_type)
    error_class = compiler_error_classes(command)[error_kind]
    failure = error_class("injected compiler boundary failure")
    attempts = []
    warnings = []

    def fail_extension(extension):
        attempts.append(extension.name)
        if extension is loaded.extensions[failed_index]:
            raise failure

    monkeypatch.setattr(command, "build_extension", fail_extension)
    monkeypatch.setattr(command, "warn", warnings.append)
    with pytest.raises(error_class) as caught:
        command.build_extensions()
    assert caught.value is failure
    assert warnings == []
    assert attempts == expected_names(loaded)[:failed_index + 1]
    assert all(extension.extra_compile_args == expected_flags(extension, compiler_type)
               for extension in loaded.extensions)


@pytest.mark.parametrize("compiler_type", ["unix", "msvc"])
def test_standard_success_retains_strict_arithmetic_flags(
    load_setup, monkeypatch, compiler_type,
):
    loaded = load_setup()
    command = build_command(loaded, compiler_type)
    compiled = []
    monkeypatch.setattr(command, "build_extension", compiled.append)
    command.build_extensions()
    assert compiled == loaded.extensions
    assert all(extension.extra_compile_args == expected_flags(extension, compiler_type)
               for extension in compiled)


def configure_zig(loaded, monkeypatch, tmp_path, platform="win-amd64"):
    command = build_command(loaded)
    target = tmp_path / "dwd" / "_residual_accel.pyd"
    monkeypatch.setenv("DWD_BUILD_ZIG", "isolated-zig.exe")
    monkeypatch.setattr(loaded.namespace["sys"], "platform", "win32")
    monkeypatch.setattr(loaded.namespace["sysconfig"], "get_platform", lambda: platform)
    monkeypatch.setattr(command, "get_ext_fullpath", lambda name: str(
        target.with_name(name.rsplit('.', 1)[-1] + '.pyd')))
    return command, target


def test_zig_success_keeps_abi_baseline_cpu_and_strict_flags(
    load_setup, monkeypatch, tmp_path,
):
    loaded = load_setup()
    command, target = configure_zig(loaded, monkeypatch, tmp_path)
    calls = []

    def fake_compile(argv, *, check):
        calls.append((argv, check))
        current_target = Path(argv[-1])
        current_target.write_bytes(b"isolated test output, not a loadable extension")
        for stem in (current_target.stem, *(Path(arg).stem for arg in argv if arg.endswith('.c'))):
            for suffix in (".pdb", ".lib", ".exp"):
                (current_target.parent / (stem + suffix)).write_bytes(b"link product")

    monkeypatch.setattr(subprocess, "run", fake_compile)
    command.build_extensions()
    assert len(calls) == len(loaded.extensions)
    targets = []
    for extension, (argv, check) in zip(loaded.extensions, calls):
        current_target = target.with_name(extension.name.rsplit('.', 1)[-1] + '.pyd')
        targets.append(current_target)
        assert check is True
        assert argv[:2] == ["isolated-zig.exe", "cc"]
        assert argv[2:2 + len(STRICT_UNIX_FLAGS)] == STRICT_UNIX_FLAGS
        assert [arg for arg in argv if arg.startswith("-march=")] == ["-march=x86_64"]
        assert ("-DPy_LIMITED_API=0x030B0000" in argv) is extension.py_limited_api
        sources = EXPECTED_SOURCES[extension.name]
        assert argv[-len(sources)-2:] == [*sources, "-o", str(current_target)]
        library = ("python3.lib" if extension.py_limited_api else
                   f"python{sys.version_info.major}{sys.version_info.minor}.lib")
        assert any(arg.endswith(library) for arg in argv)
        assert current_target.is_file()
    assert set(target.parent.iterdir()) == set(targets)


@pytest.mark.parametrize("mode", [None, "1"])
@pytest.mark.parametrize("failed_index", [0, 1, 2], ids=["residual", "ordered", "feature"])
@pytest.mark.parametrize("kind", ["missing-executable", "nonzero-exit"])
def test_explicit_zig_failure_remains_fatal(
    load_setup, monkeypatch, tmp_path, mode, failed_index, kind,
):
    loaded = load_setup(mode)
    require_index(loaded, failed_index)
    command, target = configure_zig(loaded, monkeypatch, tmp_path)
    failure = (
        FileNotFoundError("isolated Zig path is absent")
        if kind == "missing-executable"
        else subprocess.CalledProcessError(73, ["isolated-zig.exe", "cc"])
    )

    attempts = []
    def fail_compile(argv, **kwargs):
        attempts.append(argv)
        if len(attempts) == failed_index + 1:
            raise failure
        Path(argv[-1]).write_bytes(b"completed first extension")

    monkeypatch.setattr(subprocess, "run", fail_compile)
    with pytest.raises(type(failure)) as caught:
        command.build_extensions()
    assert caught.value is failure
    assert len(attempts) == failed_index + 1
    failed_target = target.with_name(loaded.extensions[failed_index].name.rsplit('.', 1)[-1] + '.pyd')
    assert not failed_target.exists()
    assert target.exists() is (failed_index > 0)


@pytest.mark.parametrize("platform", ["win32", "win-arm64"])
def test_zig_rejects_non_amd64_before_compilation(load_setup, monkeypatch, tmp_path, platform):
    loaded = load_setup()
    command, target = configure_zig(loaded, monkeypatch, tmp_path, platform)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Unexpected compiler call"))
    with pytest.raises(RuntimeError, match="Windows AMD64"):
        command.build_extensions()
    assert not target.exists()


def test_zig_rejects_non_windows_before_compilation(load_setup, monkeypatch, tmp_path):
    loaded = load_setup()
    command, target = configure_zig(loaded, monkeypatch, tmp_path)
    monkeypatch.setattr(loaded.namespace["sys"], "platform", "linux")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Unexpected compiler call"))
    with pytest.raises(RuntimeError, match="supports Windows builds"):
        command.build_extensions()
    assert not target.exists()


def finalized_command(loaded, tmp_path, inplace=False):
    """Use setuptools' actual output-path resolver, without running a compiler."""
    distribution = Distribution({
        "ext_modules": loaded.extensions,
        "packages": ["dwd"],
        "package_dir": {"": str(tmp_path / "source")},
    })
    command = loaded.command_class(distribution)
    command.build_lib = str(tmp_path / "build")
    command.inplace = inplace
    command.ensure_finalized()
    return command


@pytest.mark.parametrize("inplace", [False, True], ids=["build-tree", "in-place"])
@pytest.mark.parametrize("selected_index", [0, 1, 2], ids=["residual", "ordered", "feature"])
@pytest.mark.parametrize("state", ["missing", "directory", "empty", "nonempty"])
def test_final_output_guard_uses_selected_nonempty_artifact(
    load_setup, monkeypatch, tmp_path, inplace, selected_index, state,
):
    loaded = load_setup()
    require_index(loaded, selected_index)
    command = finalized_command(loaded, tmp_path, inplace)
    target = Path(command.get_ext_fullpath(loaded.extensions[selected_index].name)).resolve()
    selected_parent = tmp_path / ("source" if inplace else "build") / "dwd"
    other_parent = tmp_path / ("build" if inplace else "source") / "dwd"
    assert target.parent == selected_parent.resolve()
    selected_parent.mkdir(parents=True)
    other_parent.mkdir(parents=True)
    for index, extension in enumerate(loaded.extensions):
        if index != selected_index:
            Path(command.get_ext_fullpath(extension.name)).write_bytes(b"other required extension")
    # A good artifact at the unselected location must not mask a bad final one.
    (other_parent / target.name).write_bytes(b"unselected artifact")
    if state == "directory":
        target.mkdir()
    elif state == "empty":
        target.write_bytes(b"")
    elif state == "nonempty":
        target.write_bytes(b"nonempty output, not a loadable test extension")
    calls = []

    def successful_super_run(instance):
        assert instance is command
        calls.append(instance.inplace)

    monkeypatch.setattr(SetuptoolsBuildExt, "run", successful_super_run)
    if state != "nonempty":
        with pytest.raises(RuntimeError, match="Required extension was not produced"):
            command.run()
    else:
        command.run()
    assert calls == [inplace]


def test_final_output_guard_does_not_require_outputs_during_dry_run(
    load_setup, monkeypatch, tmp_path,
):
    loaded = load_setup()
    command = finalized_command(loaded, tmp_path)
    command.dry_run = True
    calls = []
    monkeypatch.setattr(SetuptoolsBuildExt, "run", lambda instance: calls.append(instance))

    def unexpected_output_lookup(name):
        pytest.fail("Dry-run attempted the post-build output check")

    monkeypatch.setattr(command, "get_ext_fullpath", unexpected_output_lookup)
    command.run()
    assert calls == [command]


@pytest.mark.parametrize("kind", ["compile", "link", "msvc-platform", "missing-zig", "zig-exit"])
def test_final_output_guard_preserves_the_original_build_exception(
    load_setup, monkeypatch, tmp_path, kind,
):
    loaded = load_setup()
    command = finalized_command(loaded, tmp_path)
    error_classes = compiler_error_classes(command)
    failures = {
        "compile": error_classes["compile"]("original compile failure"),
        "link": error_classes["link"]("original link failure"),
        "msvc-platform": error_classes["msvc-platform"]("original MSVC discovery failure"),
        "missing-zig": FileNotFoundError("original missing Zig executable"),
        "zig-exit": subprocess.CalledProcessError(73, ["isolated-zig.exe", "cc"]),
    }
    failure = failures[kind]

    def failed_super_run(instance):
        assert instance is command
        raise failure

    def unexpected_output_lookup(name):
        pytest.fail("Output guard replaced an existing build failure")

    monkeypatch.setattr(SetuptoolsBuildExt, "run", failed_super_run)
    monkeypatch.setattr(command, "get_ext_fullpath", unexpected_output_lookup)
    with pytest.raises(type(failure)) as caught:
        command.run()
    assert caught.value is failure


@pytest.mark.parametrize('version', [(3, 11), (3, 13)])
def test_other_python_versions_retain_only_limited_abi(load_setup, monkeypatch, version):
    monkeypatch.setattr(sys, 'version_info', version)
    loaded = load_setup()
    assert loaded.namespace['feature_expansion'] is False
    assert [extension.name for extension in loaded.extensions] == list(EXPECTED_SOURCES)[:2]
    assert all(extension.py_limited_api for extension in loaded.extensions)
    assert loaded.options['options'] == {'bdist_wheel': {'py_limited_api': 'cp311'}}


@pytest.mark.parametrize('boundary', ['implementation', 'platform', 'machine', 'pointer', 'gil'])
def test_unqualified_build_omits_full_abi_only(load_setup, monkeypatch, boundary):
    if boundary == 'implementation':
        implementation = dict(vars(sys.implementation))
        implementation['name'] = 'pypy'
        monkeypatch.setattr(sys, 'implementation', SimpleNamespace(**implementation))
    elif boundary == 'platform': monkeypatch.setattr(sys, 'platform', 'darwin')
    elif boundary == 'machine': monkeypatch.setattr(platform, 'machine', lambda: 'arm64')
    elif boundary == 'pointer': monkeypatch.setattr(sys, 'maxsize', 2**31-1)
    else:
        original = sysconfig.get_config_var
        monkeypatch.setattr(sysconfig, 'get_config_var', lambda name: 1 if name == 'Py_GIL_DISABLED' else original(name))
    loaded = load_setup()
    assert loaded.namespace['feature_expansion'] is False
    assert [extension.name for extension in loaded.extensions] == list(EXPECTED_SOURCES)[:2]
    assert loaded.options['options'] == {'bdist_wheel': {'py_limited_api': 'cp311'}}


def test_full_abi_rejects_unreviewed_compiler_family(load_setup, monkeypatch):
    loaded = load_setup()
    require_index(loaded, 2)
    command = build_command(loaded, 'mingw32')
    monkeypatch.setattr(command, 'build_extension', lambda extension: pytest.fail('Unexpected compilation'))
    with pytest.raises(RuntimeError, match='Unreviewed feature-expansion compiler family'):
        command.build_extensions()
