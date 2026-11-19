"""Fault injections for the actual source-archive CI boundary."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile

import pytest


_SPEC = importlib.util.spec_from_file_location(
    '_ci_sdist_under_test', Path(__file__).with_name('ci_sdist.py'))
ci_sdist = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ci_sdist)


@pytest.fixture
def source_files():
    return {
        'tests/test_sample.py': b'def test_sample(): assert True\n',
        'tests/fixtures/cached_evd.npz': b'\x00\x01binary fixture\xff',
        'tests/fixtures/future.resource': b'a new fixture format',
        'examples/checkpoint.py': b'def resume(): return 1\n',
        'doc/example_notebooks/Basic functionality.ipynb': b'{"cells": []}\n',
        'doc/example_notebooks/2D visualization.ipynb': b'{"cells": [1]}\n',
        'doc/figures/make_figures.py': b'def draw(): pass\n',
        'pyproject.toml': b'[build-system]\nrequires = ["setuptools"]\n',
        'MANIFEST.in': b'recursive-include tests *.py *.npz\n',
        'README.md': b'Source metadata\n',
        'LICENSE.txt': b'License\n',
        '.gitattributes': b'* text=auto\n',
        'dwd/__init__.py': b'raise AssertionError("do not import the package")\n',
    }


@pytest.fixture
def source(tmp_path, source_files):
    source = tmp_path / 'checkout'
    for name, content in source_files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return source


def _write_sdist(directory, files, *, extra=(), filename='dwd-1.0.tar.gz'):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    with tarfile.open(path, 'w:gz') as archive:
        root = tarfile.TarInfo('dwd-1.0')
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        for name, content in {'PKG-INFO': b'Name: dwd\nVersion: 1.0\n', **files}.items():
            member = tarfile.TarInfo('dwd-1.0/' + name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        for member, content in extra:
            archive.addfile(member, io.BytesIO(content) if content is not None else None)
    return path


def _extra_file(name, content=b'x'):
    member = tarfile.TarInfo(name)
    member.size = len(content)
    return member, content


def _reject_before_extraction(source, archive_dir, destination, match):
    with pytest.raises(RuntimeError, match=match):
        ci_sdist.verify_and_extract(source, archive_dir, destination)
    assert not destination.exists()


def test_verified_archive_extracts_exact_bytes_without_importing_package(tmp_path, source, source_files):
    archive_dir = tmp_path / 'sdists'
    archive = _write_sdist(archive_dir, source_files)
    destination = tmp_path / 'extracted'
    result = ci_sdist.verify_and_extract(source, archive_dir, destination)
    extracted = Path(result['source'])
    assert extracted == destination / 'dwd-1.0'
    assert result['archive_sha256'] == hashlib.sha256(archive.read_bytes()).hexdigest()
    for name, content in source_files.items():
        assert (extracted / name).read_bytes() == content
    assert result['required_files'] == len(source_files) - 1  # package source is not a test resource
    assert result['extracted_files'] == len(source_files) + 1  # generated PKG-INFO


@pytest.mark.parametrize('missing', [
    'tests/fixtures/cached_evd.npz',
    'tests/fixtures/future.resource',
    'tests/test_sample.py',
    'examples/checkpoint.py',
    'doc/example_notebooks/Basic functionality.ipynb',
    'doc/figures/make_figures.py',
    'README.md',
    'pyproject.toml',
])
def test_omitted_required_file_is_detected(tmp_path, source, source_files, missing):
    files = {name: content for name, content in source_files.items() if name != missing}
    _write_sdist(tmp_path / 'sdists', files)
    _reject_before_extraction(source, tmp_path / 'sdists', tmp_path / 'out', 'inventory|omitted')


def test_changed_fixture_bytes_are_detected(tmp_path, source, source_files):
    files = dict(source_files)
    files['tests/fixtures/cached_evd.npz'] += b'changed'
    _write_sdist(tmp_path / 'sdists', files)
    _reject_before_extraction(source, tmp_path / 'sdists', tmp_path / 'out', 'changed required resource')


def test_unknown_new_fixture_is_discovered_without_updating_allowlist(tmp_path, source, source_files):
    _write_sdist(tmp_path / 'sdists', source_files)
    (source / 'tests' / 'fixtures' / 'future.custom-format').write_bytes(b'new resource')
    _reject_before_extraction(source, tmp_path / 'sdists', tmp_path / 'out', 'future.custom-format')


def test_unexpected_archive_test_is_rejected(tmp_path, source, source_files):
    files = {**source_files, 'tests/test_untracked.py': b'assert False\n'}
    _write_sdist(tmp_path / 'sdists', files)
    _reject_before_extraction(source, tmp_path / 'sdists', tmp_path / 'out', 'unexpected')


def test_generated_checkout_caches_do_not_become_required(tmp_path, source, source_files):
    for name in ('tests/__pycache__/sample.pyc', 'tests/.pytest_cache/README.md',
                 'tests/old.pyc', 'tests/old.pyo', 'examples/__pycache__/checkpoint.pyc'):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'generated')
    _write_sdist(tmp_path / 'sdists', source_files)
    result = ci_sdist.verify_and_extract(source, tmp_path / 'sdists', tmp_path / 'out')
    assert result['required_files'] == len(source_files) - 1


def test_unreadable_checkout_directory_does_not_silently_shrink_inventory(
        tmp_path, source, source_files, monkeypatch):
    _write_sdist(tmp_path / 'sdists', source_files)

    def unreadable_walk(directory, *, followlinks, onerror):
        onerror(PermissionError(13, 'Access denied', str(Path(directory) / 'fixtures')))

    monkeypatch.setattr(ci_sdist.os, 'walk', unreadable_walk)
    _reject_before_extraction(source, tmp_path / 'sdists', tmp_path / 'out',
                              'Cannot inventory test resources')


@pytest.mark.parametrize('name', [
    '../escape', '/absolute', 'C:/escape', 'C:relative',
    'dwd-1.0\\tests\\escape.py', 'dwd-1.0/../escape', 'dwd-1.0/./escape',
    'dwd-1.0/tests//escape', 'dwd-1.0/tests/NUL.txt',
    'dwd-1.0/tests/trailing.', 'dwd-1.0/tests/trailing ', 'dwd-1.0/line\nbreak',
])
def test_unsafe_archive_names_are_rejected(tmp_path, source, source_files, name):
    _write_sdist(tmp_path / 'sdists', source_files, extra=[_extra_file(name)])
    _reject_before_extraction(source, tmp_path / 'sdists', tmp_path / 'out', 'Unsafe')


@pytest.mark.parametrize('member_type', [
    tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE,
])
def test_links_and_special_members_are_rejected(tmp_path, source, source_files, member_type):
    member = tarfile.TarInfo('dwd-1.0/not-regular')
    member.type = member_type
    member.linkname = 'tests/test_sample.py'
    _write_sdist(tmp_path / 'sdists', source_files, extra=[(member, None)])
    _reject_before_extraction(source, tmp_path / 'sdists', tmp_path / 'out', 'link or special')


@pytest.mark.parametrize('name', [
    'dwd-1.0/tests/test_sample.py',
    'dwd-1.0/tests/TEST_SAMPLE.PY',
    'dwd-1.0/Tests/different.py',
])
def test_duplicate_or_case_aliased_members_are_rejected(tmp_path, source, source_files, name):
    _write_sdist(tmp_path / 'sdists', source_files, extra=[_extra_file(name)])
    _reject_before_extraction(source, tmp_path / 'sdists', tmp_path / 'out', 'Duplicate|Case-aliased')


def test_multiple_roots_are_rejected(tmp_path, source, source_files):
    _write_sdist(tmp_path / 'sdists', source_files, extra=[_extra_file('another-root/file')])
    _reject_before_extraction(source, tmp_path / 'sdists', tmp_path / 'out', 'one top-level')


def test_file_as_directory_is_rejected(tmp_path, source, source_files):
    _write_sdist(tmp_path / 'sdists', source_files,
                 extra=[_extra_file('dwd-1.0/parent'), _extra_file('dwd-1.0/parent/child')])
    _reject_before_extraction(source, tmp_path / 'sdists', tmp_path / 'out', 'also used as a directory')


@pytest.mark.parametrize('count', [0, 2])
def test_ambiguous_archive_selection_is_rejected(tmp_path, source, source_files, count):
    directory = tmp_path / 'sdists'
    directory.mkdir()
    for number in range(count):
        _write_sdist(directory, source_files, filename=f'dwd-{number}.tar.gz')
    _reject_before_extraction(source, directory, tmp_path / 'out', 'exactly one source archive')


def test_existing_destination_is_untouched(tmp_path, source, source_files):
    _write_sdist(tmp_path / 'sdists', source_files)
    destination = tmp_path / 'out'
    destination.mkdir()
    sentinel = destination / 'sentinel'
    sentinel.write_bytes(b'preserve')
    with pytest.raises(RuntimeError, match='already exists'):
        ci_sdist.verify_and_extract(source, tmp_path / 'sdists', destination)
    assert list(destination.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b'preserve'


def test_destination_inside_checkout_is_rejected(tmp_path, source, source_files):
    _write_sdist(tmp_path / 'sdists', source_files)
    _reject_before_extraction(source, tmp_path / 'sdists', source / 'out', 'outside the source')


def test_invalid_archive_does_not_create_destination(tmp_path, source):
    directory = tmp_path / 'sdists'
    directory.mkdir()
    (directory / 'dwd-1.0.tar.gz').write_bytes(b'not a tar archive')
    _reject_before_extraction(source, directory, tmp_path / 'out', 'Cannot read complete')


def test_source_fixture_symlink_is_rejected(tmp_path, source, source_files):
    _write_sdist(tmp_path / 'sdists', source_files)
    target = tmp_path / 'external-fixture'
    target.write_bytes(b'outside')
    try:
        (source / 'tests' / 'fixtures' / 'linked').symlink_to(target)
    except OSError as error:
        pytest.skip(f'This host does not permit creating test symlinks: {error}')
    _reject_before_extraction(source, tmp_path / 'sdists', tmp_path / 'out', 'regular files')
