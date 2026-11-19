"""Verify checkout test resources in an sdist before extracting it for CI."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tarfile


# These source-only inputs are used by test_documentation_examples.py. Test
# files and their in-tree fixtures are discovered recursively, not enumerated.
EXTERNAL_TEST_RESOURCES = (
    'doc/example_notebooks/Basic functionality.ipynb',
    'doc/example_notebooks/2D visualization.ipynb',
    'doc/figures/make_figures.py',
)
GENERATED_DIRECTORIES = frozenset({'__pycache__', '.pytest_cache'})
METADATA_SUFFIXES = frozenset({'.md', '.toml', '.txt', '.cff'})
METADATA_NAMES = frozenset({'setup.py', 'MANIFEST.in', 'LICENSE', '.gitattributes'})


def _regular_bytes(path):
    if not stat.S_ISREG(path.lstat().st_mode):
        raise RuntimeError(f'Expected a regular file, not a link or special file: {path}')
    return path.read_bytes()


def _inventory_error(error):
    raise RuntimeError(f'Cannot inventory test resources: {error.filename}') from error


def source_inventory(source):
    """Capture expected bytes independently of the manifest and built archive."""
    source = Path(source)
    if source.is_symlink():
        raise RuntimeError('Source directory must not be a symlink.')
    source = source.resolve(strict=True)
    required = {}
    for resource_directory in ('tests', 'examples'):
        resource_root = source / resource_directory
        if not stat.S_ISDIR(resource_root.lstat().st_mode):
            raise RuntimeError(f'Source {resource_directory} must be a regular directory.')
        for directory, directories, filenames in os.walk(
                resource_root, followlinks=False, onerror=_inventory_error):
            directory = Path(directory)
            for name in list(directories):
                path = directory / name
                if not stat.S_ISDIR(path.lstat().st_mode):
                    raise RuntimeError(f'Test resources must not contain linked directories: {path}')
                if name in GENERATED_DIRECTORIES:
                    directories.remove(name)
            for name in filenames:
                path = directory / name
                if not stat.S_ISREG(path.lstat().st_mode):
                    raise RuntimeError(f'Test resources must be regular files: {path}')
                if path.suffix not in {'.pyc', '.pyo'}:
                    required[path.relative_to(source).as_posix()] = path.read_bytes()
    if not any(name.startswith('tests/') for name in required):
        raise RuntimeError('Source tests inventory is empty.')
    for name in EXTERNAL_TEST_RESOURCES:
        path = source / name
        # Refuse linked ancestors too; a file must really belong to this source.
        if any(parent.is_symlink() for parent in path.parents if parent != source
               and parent.is_relative_to(source)):
            raise RuntimeError(f'External test resource has a linked parent: {path}')
        required[name] = _regular_bytes(path)
    for path in source.iterdir():
        if path.suffix in METADATA_SUFFIXES or path.name in METADATA_NAMES:
            required[path.name] = _regular_bytes(path)
    return required


def single_sdist(directory):
    archives = sorted(Path(directory).resolve().glob('*.tar.gz'))
    if len(archives) != 1:
        raise RuntimeError(f'Expected exactly one source archive in {directory}: {archives}')
    if not stat.S_ISREG(archives[0].lstat().st_mode):
        raise RuntimeError('Source archive must be a regular file, not a link.')
    return archives[0]


def _member_parts(member):
    name = member.name
    # Tar uses POSIX paths. Also reject names that Windows would interpret or
    # normalize differently, because the same archive is tested on both OSes.
    if (not name or name.startswith('/') or '\\' in name or ':' in name
            or any(ord(character) < 32 for character in name)):
        raise RuntimeError(f'Unsafe source archive member: {name!r}')
    if member.isdir() and name.endswith('/'):
        name = name[:-1]
    parts = tuple(name.split('/'))
    reserved = {'CON', 'PRN', 'AUX', 'NUL'} | {
        prefix + str(number) for prefix in ('COM', 'LPT') for number in range(1, 10)
    }
    for part in parts:
        if (not part or part in {'.', '..'} or part != part.rstrip(' .')
                or any(character in '<>"|?*' for character in part)
                or part.split('.', 1)[0].upper() in reserved):
            raise RuntimeError(f'Unsafe source archive member: {member.name!r}')
    return parts


def _archive_payload(payload, required):
    """Preflight every member and payload before creating an extraction tree."""
    entries = {}
    prefixes = {}
    files = {}
    roots = set()
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
            for member in archive:
                if not (member.isdir() or member.isreg()):
                    raise RuntimeError(f'Source archive contains a link or special file: {member.name}')
                parts = _member_parts(member)
                roots.add(parts[0])
                if parts in entries:
                    raise RuntimeError(f'Duplicate source archive member: {member.name}')
                for length in range(1, len(parts) + 1):
                    prefix = parts[:length]
                    key = tuple(part.casefold() for part in prefix)
                    if key in prefixes and prefixes[key] != prefix:
                        raise RuntimeError(f'Case-aliased source archive member: {member.name}')
                    prefixes[key] = prefix
                entries[parts] = member.isdir()
                if member.isreg():
                    if len(parts) == 1:
                        raise RuntimeError('Source archive root must be a directory.')
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise RuntimeError(f'Cannot read source archive member: {member.name}')
                    with stream:
                        content = stream.read()
                    if len(content) != member.size:
                        raise RuntimeError(f'Truncated source archive member: {member.name}')
                    files[parts] = content
    except (tarfile.TarError, EOFError, OSError) as error:
        raise RuntimeError('Cannot read complete source archive.') from error
    if len(roots) != 1:
        raise RuntimeError('Source archive must contain exactly one top-level directory.')
    root = roots.pop()
    if (root, 'PKG-INFO') not in files:
        raise RuntimeError('Source archive must contain top-level PKG-INFO.')
    for parts in entries:
        if any(entries.get(parts[:length]) is False for length in range(1, len(parts))):
            raise RuntimeError('Source archive file is also used as a directory: ' + '/'.join(parts))
    relative_files = {'/'.join(parts[1:]): content for parts, content in files.items()}
    for resource_directory in ('tests/', 'examples/'):
        expected = {name for name in required if name.startswith(resource_directory)}
        archived = {name for name in relative_files if name.startswith(resource_directory)}
        if expected != archived:
            raise RuntimeError('Source archive ' + resource_directory
                               + ' inventory differs from checkout: missing='
                               + repr(sorted(expected - archived)) + ', unexpected='
                               + repr(sorted(archived - expected)))
    for name, content in required.items():
        if name not in relative_files:
            raise RuntimeError('Source archive omitted required resource: ' + name)
        if relative_files[name] != content:
            raise RuntimeError('Source archive changed required resource: ' + name)
    return root, entries, files


def verify_and_extract(source, sdist_directory, destination):
    """Extract one verified archive exclusively; never write during preflight."""
    source = Path(source)
    destination = Path(destination)
    if os.path.lexists(destination):
        raise RuntimeError('Extraction destination already exists.')
    resolved_destination = destination.resolve()
    if resolved_destination.is_relative_to(source.resolve()):
        raise RuntimeError('Extraction destination must be outside the source checkout.')
    required = source_inventory(source)
    archive = single_sdist(sdist_directory)
    payload = _regular_bytes(archive)
    root, entries, files = _archive_payload(payload, required)
    # All paths and bytes have passed before the exclusive directory creation.
    # No archive permissions, owners, links or newer tar extraction API apply.
    resolved_destination.mkdir(parents=True, exist_ok=False)
    for parts, is_directory in sorted(entries.items(), key=lambda item: (len(item[0]), item[0])):
        path = resolved_destination.joinpath(*parts)
        if is_directory:
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(files[parts])
    return {
        'archive': str(archive),
        'archive_sha256': hashlib.sha256(payload).hexdigest(),
        'required_files': len(required),
        'extracted_files': len(files),
        'source': str(resolved_destination / root),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='.')
    parser.add_argument('--sdist-dir', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--github-output', action='store_true')
    args = parser.parse_args()
    output = os.environ.get('GITHUB_OUTPUT') if args.github_output else None
    if args.github_output and not output:
        parser.error('--github-output requires GITHUB_OUTPUT.')
    result = verify_and_extract(args.source, args.sdist_dir, args.destination)
    if output:
        with open(output, 'a', encoding='utf-8') as stream:
            stream.write('source=' + result['source'] + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
