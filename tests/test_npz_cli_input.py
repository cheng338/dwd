"""Real CLI input files, closed load resources and no premature CV work."""
from copy import deepcopy
import importlib.util
import io
import os
from pathlib import Path
import pickle
import struct
import zipfile
import zlib

import numpy as np
import pytest


@pytest.fixture
def command(monkeypatch):
    repository = Path(os.environ.get("IR55_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
    examples = repository / "examples"
    path = examples / "tune_kernel_dwd.py"
    assert path.is_file(), "IR55_REPO_ROOT must select the isolated DWD repository"
    monkeypatch.syspath_prepend(str(repository))
    monkeypatch.syspath_prepend(str(examples))
    spec = importlib.util.spec_from_file_location("ir55_tune_kernel_dwd", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve() == path
    return module


def arrays():
    return np.arange(24, dtype=np.float32).reshape(8, 3), np.array([-1, 1] * 4, dtype=np.int8)


def archive_bytes(**members):
    stream = io.BytesIO()
    np.savez(stream, **members)
    return stream.getvalue()


def make_invalid(path, kind):
    X, y = arrays()
    if kind == "missing_file":
        return
    if kind == "directory":
        path.mkdir()
        return
    if kind in ("npy_lower", "npy_upper", "truncated_npy"):
        stream = io.BytesIO()
        np.save(stream, X, allow_pickle=False)
        data = stream.getvalue()
        if kind == "truncated_npy":
            data = data[:-4]
    elif kind == "empty":
        data = b""
    elif kind == "malformed_npy_header":
        data = b"\x93NUMPY\x01\x00" + struct.pack("<H", 6) + b"oops!\n"
    elif kind == "truncated_zip":
        data = archive_bytes(X=X, y=y)[:-22]
    elif kind == "malformed_zip":
        data = b"PK\x03\x04not-a-complete-zip-file"
    elif kind == "corrupt_member_crc":
        data = bytearray(archive_bytes(X=X, y=y))
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            member = archive.getinfo("X.npy")
            assert member.compress_type == zipfile.ZIP_STORED
        name_length, extra_length = struct.unpack_from("<HH", data, member.header_offset + 26)
        start = member.header_offset + 30 + name_length + extra_length
        data[start + member.compress_size - 1] ^= 1
    elif kind == "corrupt_deflate":
        stream = io.BytesIO()
        np.savez_compressed(stream, X=X, y=y)
        data = bytearray(stream.getvalue())
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            member = archive.getinfo("X.npy")
            assert member.compress_type == zipfile.ZIP_DEFLATED and member.compress_size > 0
        name_length, extra_length = struct.unpack_from("<HH", data, member.header_offset + 26)
        start = member.header_offset + 30 + name_length + extra_length
        # Keep BFINAL and the other bits; set the first DEFLATE block's BTYPE
        # bits to the reserved value 3, an actual decompressor format failure.
        data[start] = (data[start] & 0xF9) | 0x06
    elif kind == "missing_X":
        data = archive_bytes(y=y)
    elif kind == "missing_y":
        data = archive_bytes(X=X)
    elif kind == "object_X":
        data = archive_bytes(X=np.array([{"blocked": True}], dtype=object), y=y)
    elif kind == "object_y":
        data = archive_bytes(X=X, y=np.array([{"blocked": True}], dtype=object))
    elif kind == "raw_pickle":
        data = pickle.dumps({"X": X.tolist(), "y": y.tolist()}, protocol=4)
    else:
        raise AssertionError(kind)
    path.write_bytes(data)


def arguments(path, run):
    return ["--data", str(path), "--run-dir", str(run), "--lambd", ".1", ".2",
            "--gamma", ".3", ".4", "--folds", "2", "--seed", "17", "--max-iter", "3",
            "--jobs", "2", "--cv-native-threads", "1", "--final-native-threads", "2",
            "--cv-memory-budget-mib", "8", "--cv-worker-memory-mib", "4"]


def observe_load(command, monkeypatch, *, load_error=None):
    original = command.np.load
    records = []

    def load(file, *args, **kwargs):
        record = {"file": file, "args": args, "kwargs": dict(kwargs), "archive": None,
                  "zip": None, "archive_file": None}
        records.append(record)
        assert kwargs.get("allow_pickle") is False
        if load_error is not None:
            raise load_error
        result = original(file, *args, **kwargs)
        if isinstance(result, np.lib.npyio.NpzFile):
            record.update(archive=result, zip=result.zip, archive_file=result.zip.fp)
        return result

    monkeypatch.setattr(command.np, "load", load)
    return records


def assert_closed(records, *, archive_expected=None):
    assert len(records) == 1
    record = records[0]
    assert record["kwargs"]["allow_pickle"] is False
    # An actual owned reader lets failure inside np.load be checked too, before
    # an NpzFile object exists. Do not proxy or weaken the returned archive type.
    assert isinstance(record["file"], io.IOBase)
    assert record["file"].closed
    if archive_expected is not None:
        assert (record["archive"] is not None) is archive_expected
    if record["archive"] is not None:
        assert record["archive"].zip is None
        assert record["zip"].fp is None
        assert record["archive_file"].closed


def forbid_work(command, monkeypatch):
    calls = []

    def fail(stage):
        def invoked(*args, **kwargs):
            calls.append(stage)
            raise AssertionError("Invalid CLI input reached " + stage)
        return invoked

    monkeypatch.setattr(command, "KernGDWD", fail("estimator construction"))
    monkeypatch.setattr(command, "run_resumable_cv", fail("CV"))
    return calls


@pytest.mark.parametrize("kind", [
    "npy_lower", "npy_upper", "empty", "truncated_npy", "malformed_npy_header",
    "truncated_zip", "malformed_zip", "corrupt_member_crc", "missing_X", "missing_y",
    "object_X", "object_y", "raw_pickle", "corrupt_deflate", "missing_file", "directory",
])
def test_bad_inputs_are_usage_errors_before_estimator_or_cv(command, monkeypatch, tmp_path, capsys, kind):
    path = tmp_path / ("input.NPZ" if kind == "npy_upper" else "input.npz")
    make_invalid(path, kind)
    original = path.read_bytes() if path.is_file() else None
    if kind == "corrupt_deflate":
        with zipfile.ZipFile(path) as archive, pytest.raises(zlib.error):
            archive.read("X.npy")
    run = tmp_path / "not-started-run"
    records = observe_load(command, monkeypatch)
    calls = forbid_work(command, monkeypatch)
    with pytest.raises(SystemExit) as caught:
        command.main(arguments(path, run))
    assert caught.value.code == 2
    stderr = capsys.readouterr().err
    assert "Could not read training arrays X and y:" in stderr
    assert "Traceback (most recent call last)" not in stderr
    assert calls == [] and not run.exists()
    if kind in ("missing_file", "directory"):
        assert records == [], "File opening must fail before np.load is called"
        assert path.is_dir() is (kind == "directory")
        if kind == "directory":
            assert list(path.iterdir()) == []
        else:
            assert not path.exists()
    else:
        assert_closed(records, archive_expected=kind in {
            "corrupt_member_crc", "corrupt_deflate", "missing_X", "missing_y", "object_X", "object_y"})
        assert path.read_bytes() == original


@pytest.mark.parametrize("compressed,scale", [(False, False), (True, True)],
                         ids=["plain-unscaled", "compressed-uppercase-scaled"])
def test_valid_npz_forwards_arrays_options_after_resources_close(
        command, monkeypatch, tmp_path, capsys, compressed, scale):
    X, y = arrays()
    path = tmp_path / ("input.NPZ" if compressed else "input.npz")
    # A stream prevents np.savez from silently appending another suffix.
    with path.open("wb") as stream:
        (np.savez_compressed if compressed else np.savez)(stream, X=X, y=y)
    original = path.read_bytes()
    run = tmp_path / "run"
    records = observe_load(command, monkeypatch)
    constructed, received = [], []
    constructor = command.KernGDWD
    result = {"best_params": {"lambd": 0.1}, "work_summary": {"fixture": True},
              "finite_grid_report": {"status": "unavailable", "scope": "best_among_supplied_candidates",
                                     "reporting_error": "fixture-no-CV"}}

    def make_estimator(**kwargs):
        assert_closed(records, archive_expected=True)
        assert not run.exists()
        model = constructor(**kwargs)
        constructed.append((model, kwargs))
        return model

    def cv(model, data, labels, grid, directory, **kwargs):
        assert_closed(records, archive_expected=True)
        assert not run.exists()
        received.append((model, data, labels, grid, directory, kwargs))
        return result

    monkeypatch.setattr(command, "KernGDWD", make_estimator)
    monkeypatch.setattr(command, "run_resumable_cv", cv)
    argv = arguments(path, run)
    if scale:
        argv += ["--scale", "--resume", "--cv-only"]
    returned = command.main(argv)
    capsys.readouterr()
    assert returned is result and len(constructed) == len(received) == 1
    model, data, labels, grid, directory, options = received[0]
    assert model is constructed[0][0]
    assert constructed[0][1] == {"kernel": "rbf", "random_state": 17, "max_iter": 3}
    np.testing.assert_array_equal(data, X)
    np.testing.assert_array_equal(labels, y)
    assert data.dtype == X.dtype and labels.dtype == y.dtype
    assert grid == {"lambd": [0.1, 0.2], "kernel_kws": [{"gamma": 0.3}, {"gamma": 0.4}]}
    assert directory == run
    folds = options.pop("cv")
    assert folds.n_splits == 2 and folds.shuffle is True and folds.random_state == 17
    assert options == {"jobs": 2, "resume": scale, "scale": scale, "refit_best": not scale,
                       "final_native_threads": 2, "cv_native_threads": 1,
                       "cv_memory_budget_mib": 8.0, "cv_worker_memory_mib": 4.0}
    assert_closed(records, archive_expected=True)
    assert not run.exists() and path.read_bytes() == original


@pytest.mark.parametrize("phase", ["load", "extract"])
@pytest.mark.parametrize("error_type", [TypeError, RuntimeError, KeyboardInterrupt])
def test_unrelated_failures_propagate_unchanged_and_close_resources(
        command, monkeypatch, tmp_path, phase, error_type):
    X, y = arrays()
    path = tmp_path / "input.npz"
    path.write_bytes(archive_bytes(X=X, y=y))
    run = tmp_path / "not-started-run"
    primary = error_type("unrelated failure must not become an argparse format error")
    cause = LookupError("retained original cause")
    primary.__cause__ = cause
    primary.add_note("retained original note")
    original_notes = deepcopy(primary.__notes__)
    records = observe_load(command, monkeypatch, load_error=primary if phase == "load" else None)
    if phase == "extract":
        original_getitem = np.lib.npyio.NpzFile.__getitem__

        def getitem(archive, key):
            if any(record["archive"] is archive for record in records):
                raise primary
            return original_getitem(archive, key)

        monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", getitem)
    calls = forbid_work(command, monkeypatch)
    with pytest.raises(error_type) as caught:
        command.main(arguments(path, run))
    assert caught.value is primary
    assert primary.__cause__ is cause
    assert primary.__notes__ == original_notes
    assert calls == [] and not run.exists()
    assert_closed(records, archive_expected=phase == "extract")


def test_downstream_cv_failure_is_not_reported_as_bad_input(command, monkeypatch, tmp_path, capsys):
    X, y = arrays()
    path = tmp_path / "input.npz"
    path.write_bytes(archive_bytes(X=X, y=y))
    run = tmp_path / "not-started-run"
    records = observe_load(command, monkeypatch)
    primary = TypeError("unrelated downstream CV failure")
    calls = []

    def cv(*args, **kwargs):
        assert_closed(records, archive_expected=True)
        calls.append((args, kwargs))
        raise primary

    monkeypatch.setattr(command, "run_resumable_cv", cv)
    with pytest.raises(TypeError) as caught:
        command.main(arguments(path, run))
    assert caught.value is primary and len(calls) == 1
    assert "Could not read training arrays" not in capsys.readouterr().err
    assert not run.exists()
    assert_closed(records, archive_expected=True)
