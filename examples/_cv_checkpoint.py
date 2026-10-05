"""Strict receipt identity, runtime checks and process ownership for the example.

Runtime fingerprints require an immutable installation during a process's
lifetime. They describe disk artifacts and observed loaded libraries, not an
attestation of machine code mapped before this module was imported.
"""
from contextlib import contextmanager, ExitStack
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import sys
import threading
import uuid

import numpy as np
from sklearn.base import BaseEstimator
from threadpoolctl import ThreadpoolController

_HASHES = {}
_HASH_LOCK = threading.RLock()
_RUNTIME_FILES = _RUNTIME_PACKAGE_FILES = _RUNTIME_PACKAGES = None
_RUNTIME_MODULES = _RUNTIME_DIGEST = _RUNTIME_NATIVE = None
_RUNTIME_ACCELERATOR = _RUNTIME_CONTROLLER = _RUNTIME_MODULE_COUNT = None
_RUNTIME_GUARDS = set()
_RESOLVED_PATHS = {}


def array_fingerprint(value):
    """Identify logical array bytes, shape and dtype without numeric conversion."""
    array = np.asarray(value)
    if (array.dtype.hasobject or array.dtype.fields is not None
            or array.dtype.subdtype is not None or array.dtype.metadata is not None):
        raise CheckpointError('Unsupported array dtype for checkpoint identity')
    return {'dtype': array.dtype.str, 'shape': list(array.shape),
            'sha256': hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}


def write_json_atomic(path, payload, *, replace=True):
    path = io_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(payload, stream, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            # An exclusive hard link publishes the complete file atomically.
            os.link(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class CheckpointRun:
    """Own a run; refuse live orphan workers before issuing a new generation."""
    def __init__(self, run_dir, identity, *, resume=False):
        self.directory = io_path(run_dir)
        self.identity, self.identity_hash = identity, digest(identity)
        self.resume, self.token, self._lease = resume, None, None

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lease = _Lease(self.directory / 'coordinator.lock')
        self._lease.__enter__()
        try:
            manifest = self.directory / 'identity.json'
            if self.resume:
                payload = read_receipt(manifest, self.identity_hash)
                if payload != {'schema': 1, 'identity_sha256': self.identity_hash,
                               'identity': self.identity}:
                    raise CheckpointError('Run identity does not match this invocation')
            elif any(p.name != 'coordinator.lock' for p in self.directory.iterdir()):
                raise CheckpointError('Run directory is not fresh; use resume or a new directory')
            # Pending old workers must pass the token check after taking their
            # own lock. Active old workers prevent a replacement coordinator.
            with ExitStack() as workers:
                for path in sorted(self.directory.glob('fold-*.lock')):
                    workers.enter_context(_Lease(path))
                if not self.resume:
                    write_receipt(manifest, {'schema': 1, 'identity_sha256': self.identity_hash,
                                             'identity': self.identity})
                self.token = uuid.uuid4().hex
                write_receipt(self.directory / 'generation.json',
                              {'identity_sha256': self.identity_hash, 'token': self.token})
            return self
        except BaseException:
            self._lease.__exit__(None, None, None)
            self._lease = None
            raise

    def assert_current(self):
        assert_generation(self.directory, self.token)

    def __exit__(self, *args):
        if self._lease is not None:
            self._lease.__exit__(*args)
            self._lease = None


def assert_generation(run_dir, token):
    path = io_path(run_dir) / 'generation.json'
    record = _read_json(path)
    if not isinstance(record, dict) or set(record) != {'payload', 'sha256'}:
        raise CheckpointError('Malformed worker generation record')
    payload = record['payload']
    if (not isinstance(payload, dict) or set(payload) != {'identity_sha256', 'token'}
            or record['sha256'] != digest(payload) or payload['token'] != token):
        raise CheckpointError('This worker no longer owns the run generation')


@contextmanager
def worker_lease(run_dir, token, fold_index):
    with _Lease(io_path(run_dir) / ('fold-%04d.lock' % fold_index)):
        assert_generation(run_dir, token)
        yield

class CheckpointError(RuntimeError):
    """A saved identity or receipt cannot safely be reused."""


class CheckpointBusyError(CheckpointError):
    """Another coordinator or surviving fold worker still owns this run."""


def io_path(path):
    """Use extended Windows paths for all internal receipt and lock I/O."""
    absolute = os.path.abspath(os.fspath(path))
    if os.name == 'nt' and not absolute.startswith('\\\\?\\'):
        absolute = ('\\\\?\\UNC\\' + absolute[2:] if absolute.startswith('\\\\')
                    else '\\\\?\\' + absolute)
    return Path(absolute)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(_canonical(value).encode('utf-8')).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError('Duplicate JSON key in checkpoint')
        result[key] = value
    return result


def _read_json(path):
    try:
        return json.loads(io_path(path).read_text(encoding='utf-8'), object_pairs_hook=_unique_object,
                          parse_constant=lambda value: (_ for _ in ()).throw(CheckpointError('Nonfinite JSON token')))
    except (ValueError, OSError) as error:
        raise CheckpointError('Unreadable checkpoint: ' + str(path)) from error


def write_receipt(path, payload):
    write_json_atomic(path, {'payload': payload, 'sha256': digest(payload)}, replace=True)


def read_receipt(path, identity_sha256):
    record = _read_json(path)
    if not isinstance(record, dict) or set(record) != {'payload', 'sha256'}:
        raise CheckpointError('Malformed checkpoint envelope: ' + str(path))
    payload = record['payload']
    if (not isinstance(payload, dict) or record['sha256'] != digest(payload)
            or payload.get('identity_sha256') != identity_sha256):
        raise CheckpointError('Checkpoint checksum or identity mismatch: ' + str(path))
    return payload


class _Lease:
    """Non-inherited advisory lock; lockfiles remain as harmless stable names."""
    def __init__(self, path):
        self.path, self.stream = io_path(path), None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = None
        try:
            descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
            os.set_inheritable(descriptor, False)
            self.stream = os.fdopen(descriptor, 'r+b', buffering=0)
            descriptor = None  # The stream owns the descriptor from here.
            if os.fstat(self.stream.fileno()).st_size == 0:
                self.stream.write(b'\0')
            self.stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException as error:
            if self.stream is not None:
                self.stream.close()
                self.stream = None
            elif descriptor is not None:
                os.close(descriptor)
            if isinstance(error, OSError):
                raise CheckpointBusyError('Live or inaccessible checkpoint owner: ' + str(self.path)) from error
            raise
        return self

    def __exit__(self, *args):
        if self.stream is not None:
            try:
                self.stream.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            finally:
                self.stream.close()
                self.stream = None


def _file_hash(path):
    # resolve() walks path components on Windows; do that only once per name.
    raw = os.fspath(path)
    if raw not in _RESOLVED_PATHS:
        _RESOLVED_PATHS[raw] = Path(path).absolute()
    path = _RESOLVED_PATHS[raw]
    info = path.stat()
    stamp = (info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino)
    with _HASH_LOCK:
        if path in _HASHES:
            previous, digest = _HASHES[path]
            if previous != stamp:
                raise CheckpointError('Runtime artifact changed in this process: ' + str(path))
            return digest
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        after = path.stat()
        if stamp != (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino):
            raise CheckpointError('Runtime artifact changed while fingerprinting: ' + str(path))
        result = digest.hexdigest()
        _HASHES[path] = (stamp, result)
        return result


def _class_identity(value):
    cls = type(value)
    source = inspect.getsourcefile(cls)
    if not source:
        raise CheckpointError('Estimator class source is unavailable: ' + cls.__qualname__)
    return {'class': cls.__module__ + '.' + cls.__qualname__, 'source_sha256': _file_hash(source)}


def stable(value):
    """Encode values and types without changing their numerical semantics.

    Every node is tagged, including ordinary containers, so user parameters
    cannot impersonate an array, estimator or RNG encoding. Unsupported
    subclasses are rejected rather than identified only by their base value.
    """
    cls = type(value)
    if value is None:
        return ['none']
    if cls in (str, bool, int):
        return [cls.__name__, value]
    if cls is float:
        if not math.isfinite(value):
            raise CheckpointError('Nonfinite parameter identity')
        return ['float', value.hex()]
    if isinstance(value, np.generic) and cls is value.dtype.type:
        if isinstance(value, np.floating):
            if not np.isfinite(value):
                raise CheckpointError('Nonfinite parameter identity')
            # A round-trippable value avoids longdouble's platform padding bytes
            # and the precision loss of conversion to a Python float.
            scalar = np.format_float_scientific(value, unique=True, trim='k')
        elif isinstance(value, np.integer):
            scalar = str(int(value))
        elif isinstance(value, np.bool_):
            scalar = bool(value)
        elif isinstance(value, np.str_):
            scalar = str(value)
        else:
            raise CheckpointError('Unsupported NumPy scalar parameter identity: ' + cls.__qualname__)
        return ['numpy_scalar', cls.__module__ + '.' + cls.__qualname__, value.dtype.str, scalar]
    if cls is type(Path()):
        return ['path', cls.__module__ + '.' + cls.__qualname__, str(value)]
    if cls is np.ndarray:
        if (value.dtype.hasobject or value.dtype.fields is not None
                or value.dtype.subdtype is not None or value.dtype.metadata is not None):
            raise CheckpointError('Object, structured and metadata dtypes lack a supported parameter identity')
        return ['array', value.dtype.type.__module__ + '.' + value.dtype.type.__qualname__,
                array_fingerprint(value), list(value.strides),
                [bool(value.flags[name]) for name in ('C_CONTIGUOUS', 'F_CONTIGUOUS', 'WRITEABLE', 'ALIGNED')]]
    if cls is np.random.RandomState:
        return ['RandomState', stable(value.get_state())]
    if cls is np.random.Generator:
        bit_cls = type(value.bit_generator)
        if bit_cls is not getattr(np.random, bit_cls.__name__, None):
            raise CheckpointError('Unsupported random bit-generator identity')
        return ['Generator', bit_cls.__module__ + '.' + bit_cls.__qualname__,
                stable(value.bit_generator.state)]
    if isinstance(value, BaseEstimator):
        parameters = value.get_params(deep=False)
        if parameters.get('random_state', 0) is None:
            raise CheckpointError('Checkpointed estimators require an explicit random_state; None cannot safely resume stochastic work')
        return ['estimator', _class_identity(value), stable(parameters)]
    if cls is dict:
        if any(type(key) is not str for key in value):
            raise CheckpointError('Parameter dictionaries require string keys')
        return ['dict', [[key, stable(item)] for key, item in value.items()]]
    if cls in (tuple, list):
        return [cls.__name__, [stable(item) for item in value]]
    raise CheckpointError('Unsupported reusable parameter identity: ' + type(value).__qualname__)


def _runtime_package_names(modules=None):
    names = [('dwd', 'dwd'),
             ('numpy', 'numpy'), ('scipy', 'scipy'), ('sklearn', 'scikit-learn'),
             ('joblib', 'joblib'), ('threadpoolctl', 'threadpoolctl')]
    modules = sys.modules if modules is None else modules
    # Fingerprint an already active bridge, without activating an absent backend.
    if modules.get('aocl_runtime') is not None:
        names.append(('aocl_runtime', 'aocl-runtime'))
    return names


def _metadata_identity(distribution):
    try:
        installed = importlib.metadata.distribution(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None
    for member in installed.files or ():
        if str(member).replace('\\', '/').endswith('.dist-info/METADATA'):
            path = Path(installed.locate_file(member)).absolute()
            _file_hash(path)
            _RUNTIME_GUARDS.add(path)
    return installed.version


def _loaded_source_guards():
    """Only executed module sources need a per-fit stat; inventory checks are deeper."""
    prefixes = tuple(name for name, _ in _runtime_package_names())
    for name, module in tuple(sys.modules.items()):
        if not any(name == prefix or name.startswith(prefix + '.') for prefix in prefixes):
            continue
        path = getattr(module, '__file__', None)
        if path is None:
            continue
        if path.endswith('.pyc'):
            path = importlib.util.source_from_cache(path)
        path = Path(path).absolute()
        if path.suffix.lower() in ('.py', '.pyd', '.dll', '.so', '.dylib') or '.so.' in path.name:
            # A newly imported file must be one of the initially identified artifacts.
            if str(path) not in _RUNTIME_PACKAGE_FILES:
                raise CheckpointError('A numerical module loaded outside the frozen inventory: ' + str(path))
            _RUNTIME_GUARDS.add(path)


def runtime_fingerprint(*, check_inventory=True):
    """Hash once, check complete inventory per stage and core/native guards per fit.

    Workers avoid traversing large numerical package trees on every fast fit.
    Their first call still identifies the full runtime. The coordinator checks
    the full inventory before and after each search, while every worker checks
    DWD source, loaded DLLs and example source around each fit.
    """
    global _RUNTIME_FILES, _RUNTIME_PACKAGE_FILES, _RUNTIME_PACKAGES, _RUNTIME_MODULES
    global _RUNTIME_DIGEST, _RUNTIME_NATIVE, _RUNTIME_ACCELERATOR
    global _RUNTIME_CONTROLLER, _RUNTIME_MODULE_COUNT
    from dwd import _compiled_residual, _native_residual
    import scipy.linalg  # Load numerical dependencies before observing thread pools.
    deep = check_inventory or _RUNTIME_PACKAGE_FILES is None
    files, packages = ({} if deep else dict(_RUNTIME_PACKAGE_FILES)), {}
    module_guards = {}
    if _RUNTIME_NATIVE is None:
        np.dot(np.ones((2, 2)), np.ones((2, 2)))
        _RUNTIME_NATIVE = True
        _RUNTIME_ACCELERATOR = (_compiled_residual._ACCEL, bool(_native_residual._native_supported()))
    elif (_compiled_residual._ACCEL is not _RUNTIME_ACCELERATOR[0]
          or bool(_native_residual._native_supported()) != _RUNTIME_ACCELERATOR[1]):
        raise CheckpointError('Loaded accelerator availability or identity changed')
    for name, distribution in _runtime_package_names():
        module = importlib.import_module(name)
        origin = Path(module.__file__).absolute()
        root = origin.parent
        if _RUNTIME_PACKAGES is None:
            installed = _metadata_identity(distribution)
        else:
            if name not in _RUNTIME_PACKAGES:
                raise CheckpointError('An optional numerical backend activated after runtime initialization')
            installed = _RUNTIME_PACKAGES[name]['distribution_version']
        packages[name] = {'version': str(getattr(module, '__version__', installed)),
                          'distribution_version': installed, 'origin': str(origin)}
        module_guards[name] = (module, str(origin), packages[name]['version'])
        if deep:
            candidates = [origin] if not hasattr(module, '__path__') else root.rglob('*')
            for path in candidates:
                if not path.is_file() or '__pycache__' in path.parts:
                    continue
                suffixes = ('.py', '.c', '.h', '.pyd', '.dll', '.so', '.dylib')
                if path.suffix.lower() in suffixes or '.so.' in path.name or (name == 'aocl_runtime' and path.suffix.lower() == '.json'):
                    path = path.absolute()
                    files[str(path)] = _file_hash(path)
                    if name in ('dwd', 'aocl_runtime'):
                        _RUNTIME_GUARDS.add(path)
    if _RUNTIME_PACKAGES is None:
        _RUNTIME_PACKAGES, _RUNTIME_MODULES = packages, module_guards
    elif packages != _RUNTIME_PACKAGES or module_guards != _RUNTIME_MODULES:
        raise CheckpointError('Loaded numerical package identity changed')
    if deep:
        if _RUNTIME_PACKAGE_FILES is not None and files != _RUNTIME_PACKAGE_FILES:
            raise CheckpointError('Numerical package inventory changed; restart before resuming')
        _RUNTIME_PACKAGE_FILES = dict(files)
    _loaded_source_guards()
    for path in _RUNTIME_GUARDS:
        _file_hash(path)
    # Enumerate newly loaded libraries only at a stage boundary or after a
    # Python extension/module import; the controller queries live thread counts.
    if deep or _RUNTIME_CONTROLLER is None or _RUNTIME_MODULE_COUNT != len(sys.modules):
        _RUNTIME_CONTROLLER = ThreadpoolController()
        _RUNTIME_MODULE_COUNT = len(sys.modules)
    pools = []
    for pool in _RUNTIME_CONTROLLER.info():
        record = {key: value for key, value in pool.items() if key != 'filepath'}
        path = Path(pool['filepath']).absolute()
        files[str(path)] = _file_hash(path)
        record.update(filepath=str(path), sha256=files[str(path)])
        if pool.get('num_threads') != 1:
            raise CheckpointError('Checkpointed CV requires one active native thread per worker')
        pools.append(record)
    if not any(pool.get('user_api') == 'blas' for pool in pools):
        raise CheckpointError('Loaded BLAS identity is unavailable')
    for path in (Path(sys.executable), Path(__file__), Path(__file__).with_name('resumable_kernel_cv.py')):
        files[str(path.absolute())] = _file_hash(path)
    for path in _RUNTIME_GUARDS:
        if path.name == 'METADATA':
            files[str(path)] = _file_hash(path)
    inventory = sorted(files)
    if _RUNTIME_FILES is None:
        _RUNTIME_FILES = inventory
        _RUNTIME_DIGEST = digest(files)
    elif inventory != _RUNTIME_FILES:
        raise CheckpointError('Loaded runtime inventory changed in this process; restart before resuming')
    return {'python': sys.version, 'implementation': platform.python_implementation(),
            'machine': platform.machine(), 'packages': packages,
            'files_sha256': _RUNTIME_DIGEST, 'file_count': len(files),
            'native_pools': sorted(pools, key=lambda row: row['filepath'])}


