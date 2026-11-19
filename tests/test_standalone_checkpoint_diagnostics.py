"""Standalone checkpoint diagnostics distinguish integrity from compatibility."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import _cv_checkpoint as checkpoint


def saved_json(folder):
    return {path.name: path.read_bytes() for path in folder.glob('*.json')}


def test_healthy_receipt_and_resume_keep_existing_read_and_hash_counts(tmp_path, monkeypatch):
    folder = checkpoint.io_path(tmp_path) / 'run'
    identity = {'data': {'sha256': 'fixed'}}
    with checkpoint.CheckpointRun(folder, identity):
        pass
    manifest = folder / 'identity.json'
    before = manifest.read_bytes()
    resumed = checkpoint.CheckpointRun(folder, identity, resume=True)
    original_read, original_digest = checkpoint._read_json, checkpoint.digest
    calls = {'read': 0, 'digest': 0}

    def read(path):
        calls['read'] += 1
        return original_read(path)

    def digest(value):
        calls['digest'] += 1
        return original_digest(value)

    monkeypatch.setattr(checkpoint, '_read_json', read)
    monkeypatch.setattr(checkpoint, 'digest', digest)
    monkeypatch.setattr(checkpoint, '_identity_mismatch_details',
                        lambda *args: pytest.fail('Healthy paths must not format differences'))
    assert checkpoint.read_receipt(manifest, resumed.identity_hash)['identity'] == identity
    assert calls == {'read': 1, 'digest': 1}
    calls.update(read=0, digest=0)
    with resumed:
        pass
    assert calls == {'read': 1, 'digest': 2}  # Existing read and generation publication.
    assert manifest.read_bytes() == before


@pytest.mark.parametrize('damage', ['checksum', 'foreign', 'both'])
def test_generic_receipt_integrity_precedes_neutral_identity_error(tmp_path, damage):
    path = checkpoint.io_path(tmp_path) / 'receipt.json'
    payload = {'identity_sha256': 'a' * 64, 'value': 'SECRET_VALUE'}
    checkpoint.write_receipt(path, payload)
    record = json.loads(path.read_text(encoding='utf-8'))
    if damage in ('foreign', 'both'):
        record['payload']['identity_sha256'] = 'b' * 64
    if damage == 'foreign':
        record['sha256'] = checkpoint.digest(record['payload'])
    else:
        record['sha256'] = '0' * 64
    path.write_text(json.dumps(record), encoding='utf-8')
    before = path.read_bytes()
    with pytest.raises(checkpoint.CheckpointError) as caught:
        checkpoint.read_receipt(path, 'a' * 64)
    category = 'identity' if damage == 'foreign' else 'checksum'
    assert str(caught.value).startswith('Checkpoint ' + category + ' mismatch:')
    assert 'Run identity' not in str(caught.value) and 'SECRET_VALUE' not in str(caught.value)
    assert path.read_bytes() == before


@pytest.mark.parametrize('mode', ['changed-run', 'corrupt-and-changed', 'same-hash-changed-body'])
def test_run_diagnostics_preserve_manifest_generation_and_ownership(tmp_path, mode):
    folder = checkpoint.io_path(tmp_path) / 'run'
    identity = {'data': {'sha256': 'SECRET_OLD'}, 'parameters': {'count': 1}}
    with checkpoint.CheckpointRun(folder, identity):
        pass
    expected = deepcopy(identity)
    if mode != 'same-hash-changed-body':
        expected['data']['sha256'] = 'SECRET_NEW'
    manifest = folder / 'identity.json'
    record = json.loads(manifest.read_text(encoding='utf-8'))
    if mode == 'corrupt-and-changed':
        record['sha256'] = '0' * 64
    elif mode == 'same-hash-changed-body':
        record['payload']['identity']['data']['sha256'] = 'SECRET_INCONSISTENT'
        record['sha256'] = checkpoint.digest(record['payload'])
    if mode != 'changed-run':
        manifest.write_text(json.dumps(record), encoding='utf-8')
    before = saved_json(folder)
    rejected = checkpoint.CheckpointRun(folder, expected, resume=True)
    with pytest.raises(checkpoint.CheckpointError) as caught:
        with rejected:
            pytest.fail('Incompatible or corrupt run was admitted')
    message = str(caught.value)
    prefix = {'changed-run': 'Run identity changed; use a new run directory',
              'corrupt-and-changed': 'Checkpoint checksum mismatch:',
              'same-hash-changed-body': 'Checkpoint identity metadata inconsistent:'}[mode]
    assert message.startswith(prefix), message
    if mode == 'changed-run':
        assert '"data"' in message
    assert 'SECRET_' not in message and saved_json(folder) == before
    assert rejected._lease is None
    with checkpoint._Lease(folder / 'coordinator.lock'):
        assert saved_json(folder) == before


@pytest.mark.parametrize('identity', [None, 7, ['fixed', 2]])
def test_nondict_identity_roundtrip_remains_supported(tmp_path, identity):
    folder = checkpoint.io_path(tmp_path) / 'run'
    with checkpoint.CheckpointRun(folder, identity):
        pass
    before = (folder / 'identity.json').read_bytes()
    with checkpoint.CheckpointRun(folder, deepcopy(identity), resume=True):
        pass
    assert (folder / 'identity.json').read_bytes() == before


@pytest.mark.parametrize('old,new', [(True, 1), (1, 1.0), (-0.0, 0.0)])
def test_field_diagnostics_preserve_canonical_types(old, new):
    assert checkpoint._identity_mismatch_details({'typed': old}, {'typed': new}) == \
           ' (differing identity fields: "typed")'


def test_field_details_are_bounded_and_escape_names_without_values():
    keys = [chr(0x10000 + index) * 40 for index in range(6)]
    detail = checkpoint._identity_mismatch_details(
        {key: 'SECRET_OLD' for key in keys}, {key: 'SECRET_NEW' for key in keys})
    assert len(detail) <= 512 and 'additional details truncated' in detail
    assert 'SECRET_' not in detail and not any(char in detail for char in '\r\n\t')
    assert checkpoint._identity_mismatch_details({'bad': object()}, {'bad': 1}) == \
           ' (identity details unavailable)'
