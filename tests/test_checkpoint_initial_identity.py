"""Unpublished initial identity scratch permits only a new, explicit fresh run."""
import importlib.util
import json
from pathlib import Path

import pytest


IDENTITY = {"purpose": "initial-identity-regression", "value": 1}
TEMP = "identity.json." + "a" * 32 + ".tmp"


@pytest.fixture(scope="module")
def checkpoint():
    path = Path(__file__).resolve().parents[1] / "examples" / "_cv_checkpoint.py"
    spec = importlib.util.spec_from_file_location(
        "_test_checkpoint_initial_identity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("contents", [b"", b'{"incomplete":', None])
def test_initial_scratch_is_preserved_and_never_adopted(checkpoint, tmp_path, contents):
    run = tmp_path / "run"
    run.mkdir()
    if contents is None:
        other_identity = {"purpose": "another-invocation", "value": 99}
        payload = {"schema": 1,
                   "identity_sha256": checkpoint.digest(other_identity),
                   "identity": other_identity}
        contents = (json.dumps({"payload": payload, "sha256": checkpoint.digest(payload)},
                               sort_keys=True) + "\n").encode("utf-8")
    orphan = run / TEMP
    orphan.write_bytes(contents)
    with pytest.raises(checkpoint.CheckpointError):
        with checkpoint.CheckpointRun(run, IDENTITY, resume=True):
            pass
    with checkpoint.CheckpointRun(run, IDENTITY) as context:
        context.assert_current()
        first_token = context.token
    manifest = (run / "identity.json").read_bytes()
    saved = checkpoint.read_receipt(run / "identity.json", checkpoint.digest(IDENTITY))
    assert saved == {"schema": 1, "identity_sha256": checkpoint.digest(IDENTITY),
                     "identity": IDENTITY}
    assert orphan.read_bytes() == contents
    with pytest.raises(checkpoint.CheckpointError, match="not fresh"):
        with checkpoint.CheckpointRun(run, IDENTITY):
            pass
    with checkpoint.CheckpointRun(run, IDENTITY, resume=True) as context:
        context.assert_current()
        assert context.token != first_token
    assert (run / "identity.json").read_bytes() == manifest
    assert orphan.read_bytes() == contents


@pytest.mark.parametrize("name,is_directory", [
    ("foreign.tmp", False),
    ("identity.json.owned-crash.tmp", False),
    ("identity.json." + "A" * 32 + ".tmp", False),
    ("identity.json", False),
    ("generation.json", False),
    ("fold-0000.lock", False),
    ("receipts", True),
    (TEMP, True),
])
def test_unknown_or_later_stage_evidence_remains_nonfresh(
        checkpoint, tmp_path, name, is_directory):
    run = tmp_path / "run"
    run.mkdir()
    allowed = run / ("identity.json." + "b" * 32 + ".tmp")
    allowed.write_bytes(b"unpublished")
    rejected = run / name
    if is_directory:
        rejected.mkdir()
        evidence = rejected / "untouched"
    else:
        evidence = rejected
    evidence.write_bytes(b"preserve-existing-evidence")
    with pytest.raises(checkpoint.CheckpointError, match="not fresh"):
        with checkpoint.CheckpointRun(run, IDENTITY):
            pass
    assert allowed.read_bytes() == b"unpublished"
    assert evidence.read_bytes() == b"preserve-existing-evidence"
    if name != "identity.json":
        assert not (run / "identity.json").exists()


@pytest.mark.parametrize("existing_lock", [False, True])
def test_ordinary_startup_and_resume_skip_orphan_metadata(
        checkpoint, tmp_path, monkeypatch, existing_lock):
    run = tmp_path / "run"
    run.mkdir()
    if existing_lock:
        (run / "coordinator.lock").write_bytes(b"\x00")

    def unexpected(*args, **kwargs):
        raise AssertionError("Ordinary startup or resume classified orphan metadata")

    # Patch only this test module's isolated checkpoint instance, not pathlib.
    monkeypatch.setattr(checkpoint, "_is_unpublished_identity_temporary", unexpected)
    with checkpoint.CheckpointRun(run, IDENTITY) as context:
        context.assert_current()
        first_token = context.token
    manifest = (run / "identity.json").read_bytes()
    with checkpoint.CheckpointRun(run, IDENTITY, resume=True) as context:
        context.assert_current()
        assert context.token != first_token
    assert (run / "identity.json").read_bytes() == manifest
