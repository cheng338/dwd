"""Observe optional-import exceptions and logging in bounded fresh processes.

This module is an investigation/test helper, not package runtime code. Each
child deliberately configures only its own logging/import state. Results are
observations, not an assertion that a particular implementation is correct.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def io_path(path):
    value = os.path.abspath(path)
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        return "\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value
    return value


def save(path, payload):
    with open(io_path(path), "x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def source_roots(snapshot):
    snapshot = Path(snapshot).resolve()
    if (snapshot / "dwd" / "dwd" / "__init__.py").is_file():
        dwd_root = snapshot / "dwd"
        ensemble_root = snapshot / "ensemble-dwd" / "src"
    else:
        dwd_root = snapshot
        ensemble_root = None
    if not (dwd_root / "dwd" / "socp_dwd.py").is_file():
        raise ValueError("snapshot must contain the selected DWD package")
    paths = [dwd_root]
    if ensemble_root is not None and ensemble_root.is_dir():
        paths.append(ensemble_root)
    return paths


def cases():
    rows = []
    for error in ("missing", "transitive", "generic", "runtime"):
        for handler in ("empty", "configured"):
            rows.append(dict(phase="preloaded", error=error, handler=handler, later="stream"))
    for handler in ("empty", "configured"):
        rows.append(dict(phase="cold", error="missing", handler=handler, later="stream"))
    for phase in ("cold", "preloaded"):
        rows.append(dict(phase=phase, error="missing", handler="empty", later="file"))
    rows.append(dict(phase="cold", error="none", handler="empty", later="stream"))
    for row in rows:
        row["id"] = "-".join(row[key] for key in ("phase", "error", "handler", "later"))
    return rows


def observe(snapshot, case, output):
    import builtins
    import contextlib
    import io
    import logging
    import traceback

    paths = source_roots(snapshot)
    sys.path[:0] = [str(path) for path in paths]
    package_dir = paths[0] / "dwd"
    record = {"case": case, "pid": os.getpid(), "python": sys.version,
              "source_paths": [str(path) for path in paths],
              "child_actual_exit": None, "status": "preparing"}
    # Prewarm unrelated numerical dependencies before the logging experiment.
    # Neither DWD target module is imported by this explicit warmup.
    import numpy
    import sklearn
    from threadpoolctl import threadpool_info, threadpool_limits

    limiter = threadpool_limits(limits=1)
    record["prewarm_versions"] = {"numpy": numpy.__version__, "sklearn": sklearn.__version__}
    record["prewarm_target_present"] = "dwd.socp_dwd" in sys.modules
    record["prewarm_persistence_present"] = "dwd._conic_persistence" in sys.modules
    if record["prewarm_target_present"] or record["prewarm_persistence_present"]:
        raise RuntimeError("Fresh dependency warmup unexpectedly imported a DWD target")
    if case["phase"] == "preloaded":
        try:
            import dwd._conic_persistence
        except BaseException as error:
            record.update(status="preload_failed", preload_error=repr(error),
                          preload_traceback=traceback.format_exc())
            return record
    record["before_target_modules"] = {
        name: name in sys.modules for name in ("dwd.socp_dwd", "dwd._conic_persistence", "cvxpy")}

    root = logging.getLogger()
    record["startup_root_handler_count"] = len(root.handlers)
    root.handlers.clear()
    root.disabled = False
    logging.disable(logging.NOTSET)
    custom_stream = io.StringIO()
    custom = None
    if case["handler"] == "configured":
        custom = logging.StreamHandler(custom_stream)
        custom.setLevel(logging.DEBUG)
        custom.setFormatter(logging.Formatter("CUSTOM:%(levelname)s:%(message)s"))
        custom.addFilter(logging.Filter())
        root.addHandler(custom)
        root.setLevel(logging.DEBUG)
    else:
        root.setLevel(logging.WARNING)

    def root_state():
        return {"level": root.level, "disabled": root.disabled,
                "manager_disable": root.manager.disable,
                "filters": [id(item) for item in root.filters],
                "handlers": [{"id": id(item), "type": type(item).__name__,
                              "level": item.level, "formatter_id": id(item.formatter),
                              "format": getattr(item.formatter, "_fmt", None),
                              "stream_id": id(getattr(item, "stream", None)),
                              "filters": [id(value) for value in item.filters]}
                             for item in root.handlers]}

    def error_state(error):
        if error is None:
            return None
        return {"id": id(error), "type": type(error).__name__, "name": getattr(error, "name", None),
                "message": str(error), "cause_id": id(error.__cause__),
                "cause_type": type(error.__cause__).__name__,
                "cause_message": str(error.__cause__) if error.__cause__ is not None else None,
                "suppress_context": error.__suppress_context__,
                "notes": list(getattr(error, "__notes__", []))}

    injected = {
        "missing": lambda: ModuleNotFoundError("probe: optional cvxpy absent", name="cvxpy"),
        "transitive": lambda: ModuleNotFoundError("probe: transitive scs absent", name="scs"),
        "generic": lambda: ImportError("probe: CVXPY import is broken"),
        "runtime": lambda: RuntimeError("probe: CVXPY initialization failed"),
        "none": lambda: None,
    }[case["error"]]()
    if injected is not None:
        injected.__cause__ = LookupError("probe: existing import cause")
        injected.__suppress_context__ = True
        injected.add_note("probe: existing exception note")
    record["injected_before"] = error_state(injected)
    record["root_before"] = root_state()
    calls = []
    log_records = []
    original_import = builtins.__import__
    original_handle = logging.Logger.handle

    def controlled_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cvxpy" and injected is not None:
            calls.append({"name": name, "caller_module": (globals or {}).get("__name__"),
                          "caller_file": (globals or {}).get("__file__"),
                          "fromlist": list(fromlist or ())})
            raise injected
        return original_import(name, globals, locals, fromlist, level)

    def observed_handle(logger, event):
        log_records.append({"logger": event.name, "level": event.levelname,
                            "message": event.getMessage()})
        return original_handle(logger, event)

    caught = None
    target = None
    captured_stderr = io.StringIO()
    builtins.__import__ = controlled_import
    logging.Logger.handle = observed_handle
    try:
        with contextlib.redirect_stderr(captured_stderr):
            try:
                target = __import__("dwd.socp_dwd", fromlist=["DWD"])
            except BaseException as error:
                caught = error
                record["traceback"] = traceback.format_exception(error)
                record["traceback_frames"] = [dict(filename=f.filename, lineno=f.lineno, name=f.name)
                                               for f in traceback.extract_tb(error.__traceback__)]
    finally:
        builtins.__import__ = original_import
        logging.Logger.handle = original_handle
    record.update(status="observed", import_calls=calls, log_records=log_records,
                  import_stderr=captured_stderr.getvalue(), custom_import_text=custom_stream.getvalue(),
                  caught_is_injected=(caught is injected if injected is not None else None),
                  caught=error_state(caught), root_after=root_state())
    record["root_state_unchanged"] = record["root_before"] == record["root_after"]

    # Do not clear or force reconfigure the root here: doing so would conceal
    # an import-created handler that prevents the application's basicConfig.
    later_stream = io.StringIO()
    later_file = Path(output).with_suffix(".application.log")
    config = {"level": logging.INFO, "format": "APP:%(levelname)s:%(message)s"}
    if case["later"] == "file":
        config["filename"] = io_path(later_file)
    else:
        config["stream"] = later_stream
    logging.basicConfig(**config)
    marker = "probe: subsequent application INFO"
    root.info(marker)
    for handler in root.handlers:
        handler.flush()
    file_text = ""
    if os.path.exists(io_path(later_file)):
        with open(io_path(later_file), encoding="utf-8") as stream:
            file_text = stream.read()
    record["later"] = {"mode": case["later"], "root_state": root_state(),
                       "stream_text": later_stream.getvalue(), "file_text": file_text,
                       "custom_text": custom_stream.getvalue(),
                       "requested_destination_received_info": marker in (file_text + later_stream.getvalue()),
                       "configured_handler_received_info": marker in custom_stream.getvalue()}
    origins = {name: str(Path(module.__file__).resolve())
               for name, module in list(sys.modules.items())
               if (name == "dwd" or name.startswith("dwd.")) and getattr(module, "__file__", None)}
    record["dwd_origins"] = origins
    record["dwd_origins_selected"] = bool(origins) and all(
        Path(path).is_relative_to(package_dir.resolve()) for path in origins.values())
    record["source_sha256"] = {}
    for name in ("socp_dwd.py", "_conic_persistence.py"):
        with open(io_path(package_dir / name), "rb") as stream:
            record["source_sha256"][name] = hashlib.sha256(stream.read()).hexdigest()
    record["native_pools_after"] = threadpool_info()
    record["target_imported"] = target is not None
    if target is not None:
        record["target_exports"] = {"DWD": hasattr(target, "DWD"),
                                    "PersistentProblem": hasattr(target, "PersistentProblem"),
                                    "cvxpy_binding_is_loaded_module": target.cp is sys.modules.get("cvxpy")}
    limiter.restore_original_limits()
    return record


def run_panel(snapshot, output, workers=4, case_ids=None):
    if not 1 <= workers <= 4:
        raise ValueError("workers must be between 1 and 4")
    matrix = cases()
    if case_ids is not None:
        requested = list(case_ids)
        known = {row["id"] for row in matrix}
        if not requested or len(set(requested)) != len(requested) or not set(requested) <= known:
            raise ValueError("case_ids must be a nonempty, unique subset of known cases")
        matrix = [row for row in matrix if row["id"] in requested]
    source_roots(snapshot)
    output = Path(output).resolve()
    Path(io_path(output.parent)).mkdir(parents=True, exist_ok=True)
    directory = output.with_name(output.stem + "-cases")
    Path(io_path(directory)).mkdir()
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    script = Path(__file__).resolve()

    def run_one(item):
        index, case = item
        child_output = directory / (str(index).zfill(2) + ".json")
        stdout_path = child_output.with_suffix(".stdout.txt")
        stderr_path = child_output.with_suffix(".stderr.txt")
        command = [sys.executable, "-I", "-B", str(script), "--snapshot", str(Path(snapshot).resolve()),
                   "--output", str(child_output), "--child", case["id"]]
        with open(io_path(stdout_path), "xb") as stdout, open(io_path(stderr_path), "xb") as stderr:
            try:
                completed = subprocess.run(command, stdout=stdout, stderr=stderr, env=environment,
                                           cwd=io_path(directory), timeout=90, check=False)
                result = {"case": case, "actual_exit": completed.returncode, "timed_out": False}
            except subprocess.TimeoutExpired:
                result = {"case": case, "actual_exit": None, "timed_out": True}
            except OSError as error:
                result = {"case": case, "actual_exit": None, "timed_out": False,
                          "launch_error": repr(error)}
        result.update(stdout_path=str(stdout_path), stderr_path=str(stderr_path), output_path=str(child_output))
        if os.path.exists(io_path(child_output)):
            with open(io_path(child_output), "r", encoding="utf-8") as stream:
                result["observation"] = json.load(stream)
        else:
            result["observation"] = None
        return result

    # The final unblocked smoke is a separate fresh process after the matrix.
    parallel = [(index, case) for index, case in enumerate(matrix) if case["error"] != "none"]
    final_smoke = [(index, case) for index, case in enumerate(matrix) if case["error"] == "none"]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(run_one, parallel))
    results.extend(run_one(item) for item in final_smoke)
    with open(io_path(script), "rb") as stream:
        script_hash = hashlib.sha256(stream.read()).hexdigest()
    panel = {"schema": "ir44-optional-import-observations-v1", "snapshot": str(Path(snapshot).resolve()),
             "workers": workers, "per_child_timeout_seconds": 90,
             "selected_case_ids": [row["id"] for row in matrix],
             "helper_sha256": script_hash, "cases": results,
             "all_children_completed": all(row["actual_exit"] == 0 and not row["timed_out"] for row in results),
             "scope": "import/logging metadata observations; no fits, solver runs or timing claim"}
    save(output, panel)
    return panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--case-id", action="append", dest="case_ids",
                        help="Select a case; repeat for a subset. Omit to run all 13.")
    parser.add_argument("--child", choices=[row["id"] for row in cases()])
    args = parser.parse_args()
    if args.child:
        case = next(row for row in cases() if row["id"] == args.child)
        try:
            observation = observe(args.snapshot, case, args.output)
        except BaseException as error:
            import traceback
            observation = {"case": case, "status": "probe_failed", "error_type": type(error).__name__,
                           "error": str(error), "traceback": traceback.format_exc(), "child_actual_exit": None}
        save(args.output, observation)
        return 0 if observation["status"] == "observed" else 1
    return 0 if run_panel(args.snapshot, args.output, args.workers, args.case_ids)["all_children_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
