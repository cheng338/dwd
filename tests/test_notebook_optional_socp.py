"""Execute selected example cells in isolated Python children with Agg.

Read notebook resources from the checkout or detached test bundle. Each child
uses the selected DWD package and reports its module origins.
"""
import ast
import builtins
import contextlib
import importlib.abc
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = Path("doc/example_notebooks/2D visualization.ipynb")
SKIP_MESSAGE = "SOCP example skipped: CVXPY is not installed; see README for installation instructions."
RESULT_PREFIX = "IR33_RESULT "


def notebook(root):
    return json.loads((root / NOTEBOOK_PATH).read_text(encoding="utf-8"))


def python_cell(cell):
    lines = "".join(cell["source"]).splitlines()
    # Replace only the notebook's display magic with the child-selected Agg
    # backend. No import, estimator argument, data size or solver is replaced.
    return "\n".join(line for line in lines
                     if not line.lstrip().startswith("%matplotlib "))


def execute_cell(doc, index, namespace):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(python_cell(doc["cells"][index]), f"<2D notebook cell {index}>", "exec"), namespace)
    return output.getvalue()


def origin_proof(package_root):
    import dwd

    package = (package_root / "dwd").resolve()
    assert Path(dwd.__file__).resolve() == package / "__init__.py", (dwd.__file__, package)
    files = {}
    for name, module in tuple(sys.modules.items()):
        if name == "dwd" or name.startswith("dwd."):
            path = getattr(module, "__file__", None)
            if path is not None:
                resolved = Path(path).resolve()
                assert resolved.is_relative_to(package), (name, str(resolved), str(package))
                files[name] = str(resolved)
    assert files.get("dwd") == str(Path(dwd.__file__).resolve())
    return files


def science_record(namespace, model_name):
    import numpy as np

    model, X, y = namespace[model_name], namespace["X"], namespace["y"]
    assert X.shape == (200, 2) and y.shape == (200,)
    scores = model.decision_function(X)
    prediction = model.predict(X)
    assert scores.shape == prediction.shape == (200,)
    assert np.isfinite(scores).all()
    assert np.isin(prediction, model.classes_).all()
    np.testing.assert_array_equal(prediction, model.classes_[(scores > 0).astype(int)])
    accuracy = float(np.mean(prediction == y))
    assert 0.0 <= accuracy <= 1.0
    result = dict(rows=200, features=2, finite_scores=True, accuracy=accuracy)
    if hasattr(model, "solver_status_"):
        assert model.solver_status_ in ("optimal", "optimal_inaccurate")
        result["solver_status"] = model.solver_status_
    if hasattr(model, "converged_"):
        result["converged"] = bool(model.converged_)
    return result


def missing_dependency(doc):
    class BlockCVXPY(importlib.abc.MetaPathFinder):
        def __init__(self):
            self.requests = []

        def find_spec(self, fullname, path=None, target=None):
            if fullname == "cvxpy" or fullname.startswith("cvxpy."):
                self.requests.append(fullname)
                raise ModuleNotFoundError("CVXPY intentionally absent in this test child", name="cvxpy")
            return None

    # This is a fresh, disposable child. Remove any startup-cached optional
    # module so an import-cache hit cannot bypass the missing-dependency proof.
    for name in tuple(sys.modules):
        if name == "dwd.socp_dwd" or name == "cvxpy" or name.startswith("cvxpy."):
            del sys.modules[name]
    blocker = BlockCVXPY()
    sys.meta_path.insert(0, blocker)
    try:
        namespace = {"__name__": "__notebook_shared__"}
        execute_cell(doc, 1, namespace)
        assert blocker.requests == []
        assert "DWD" not in namespace
        output = execute_cell(doc, 2, namespace)
        assert output.strip() == SKIP_MESSAGE
        assert not {"DWD", "dwd", "X", "y"}.intersection(namespace)
        assert blocker.requests == ["cvxpy"]
        execute_cell(doc, 3, namespace)
        assert blocker.requests == ["cvxpy"]
        plotted = science_record(namespace, "kdwd")
        import numpy as np
        assert namespace["f"].shape == namespace["xx"].shape == namespace["yy"].shape
        assert np.isfinite(namespace["f"]).all()
        namespace["plt"].gcf().canvas.draw()
        namespace["plt"].close("all")

        # Each README example starts without the shared cell or the other
        # example's namespace. Neither can borrow a previous import/data value.
        linear = {"__name__": "__readme_linear__"}
        assert execute_cell(doc, 5, linear).strip() == SKIP_MESSAGE
        assert not {"DWD", "dwd", "X", "y", "KernGDWD"}.intersection(linear)
        assert blocker.requests == ["cvxpy", "cvxpy"]
        kernel = {"__name__": "__readme_kernel__"}
        execute_cell(doc, 6, kernel)
        assert blocker.requests == ["cvxpy", "cvxpy"]
        independent = science_record(kernel, "kdwd")
        assert "DWD" not in kernel
        assert not any(name == "cvxpy" or name.startswith("cvxpy.") for name in sys.modules)
        assert "dwd.socp_dwd" not in sys.modules
        return dict(blocked_imports=blocker.requests, partial_socp_fits=0,
                    kernel_plot=plotted, independent_kernel=independent)
    finally:
        sys.meta_path.remove(blocker)


def present_dependency(doc):
    import numpy as np
    import matplotlib.pyplot as plt

    # The examples deliberately leave make_blobs' random_state unspecified.
    # Seed only this disposable child's RNG for a repeatable validation input.
    np.random.seed(62116)
    shared = {"__name__": "__notebook_shared__"}
    execute_cell(doc, 1, shared)
    output = execute_cell(doc, 2, shared)
    assert SKIP_MESSAGE not in output
    plotted = science_record(shared, "dwd")
    plt.gcf().canvas.draw()
    plt.close("all")
    independent = {"__name__": "__readme_linear__"}
    output = execute_cell(doc, 5, independent)
    assert SKIP_MESSAGE not in output
    result = science_record(independent, "dwd")
    assert float(output.strip()) == result["accuracy"]
    assert "KernGDWD" not in independent
    return dict(linear_plot=plotted, independent_linear=result, child_rng_seed=62116)


def negative_flow(doc, mode):
    original_import = builtins.__import__
    records = []
    for index in (2, 5):
        namespace = {"__name__": "__negative_flow__"}
        if index == 2:
            execute_cell(doc, 1, namespace)
        calls = []
        if mode == "unrelated_import":
            error = ModuleNotFoundError("Unrelated dependency failure", name="scs")
        elif mode == "fit_missing_cvxpy":
            error = ModuleNotFoundError("Failure inside fit, not its import", name="cvxpy")
        else:
            assert mode == "fit_runtime"
            error = RuntimeError("Failure inside an available SOCP estimator")

        class FakeDWD:
            def __init__(self, C):
                assert C == "auto"
                calls.append("construct")

            def fit(self, X, y):
                assert X.shape == (200, 2) and y.shape == (200,)
                calls.append("fit")
                raise error

        def controlled_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "dwd.socp_dwd":
                calls.append("import")
                if mode == "unrelated_import":
                    raise error
                return SimpleNamespace(DWD=FakeDWD)
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = controlled_import
        try:
            try:
                execute_cell(doc, index, namespace)
            except BaseException as actual:
                assert actual is error
            else:
                raise AssertionError("A non-optional failure was silently skipped")
        finally:
            builtins.__import__ = original_import
        assert calls == (["import"] if mode == "unrelated_import" else ["import", "construct", "fit"])
        assert "dwd" not in namespace
        records.append(dict(cell=index, error_type=type(error).__name__,
                            error_name=getattr(error, "name", None), calls=calls,
                            original_error_preserved=True))
    return dict(negative_controls=records)


def child_main(package_root, scenario):
    sys.path.insert(0, str(package_root))
    import matplotlib
    matplotlib.use("Agg", force=True)
    try:
        doc = notebook(ROOT)
        if scenario == "missing":
            result = missing_dependency(doc)
        elif scenario == "present":
            result = present_dependency(doc)
        else:
            result = negative_flow(doc, scenario)
        result.update(scenario=scenario, package_origins=origin_proof(package_root), backend=matplotlib.get_backend())
        assert result["backend"].lower() == "agg"
        print(RESULT_PREFIX + json.dumps(result, sort_keys=True))
    finally:
        import matplotlib.pyplot as plt
        plt.close("all")


def run_child(scenario, tmp_path, record_property):
    package_root = ROOT
    if not (ROOT / "dwd").is_dir():
        # A detached bundle has the notebook and tests; DWD comes from the selected install.
        spec = importlib.util.find_spec("dwd")
        assert spec is not None and spec.origin is not None, "DWD must be available to run the notebook tests"
        package_root = Path(spec.origin).resolve().parent.parent
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(Path(__file__).resolve()),
         "--child", str(package_root), scenario],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=240)
    stdout = tmp_path / f"{scenario}.stdout.txt"
    stderr = tmp_path / f"{scenario}.stderr.txt"
    stdout.write_text(completed.stdout, encoding="utf-8")
    stderr.write_text(completed.stderr, encoding="utf-8")
    record_property("child_actual_exit", completed.returncode)
    record_property("child_stdout", str(stdout))
    record_property("child_stderr", str(stderr))
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    lines = [line[len(RESULT_PREFIX):] for line in completed.stdout.splitlines()
             if line.startswith(RESULT_PREFIX)]
    assert len(lines) == 1
    result = json.loads(lines[0])
    assert result["scenario"] == scenario and result["package_origins"]
    record_property("child_result", json.dumps(result, sort_keys=True))
    return result


def test_missing_cvxpy_skips_only_socp_and_runs_actual_kernel_cells(tmp_path, record_property):
    run_child("missing", tmp_path, record_property)


@pytest.mark.skipif(importlib.util.find_spec("cvxpy") is None, reason="Optional SOCP examples require CVXPY")
def test_present_cvxpy_runs_actual_socp_cells_independently(tmp_path, record_property):
    run_child("present", tmp_path, record_property)


@pytest.mark.parametrize("scenario", ["unrelated_import", "fit_runtime", "fit_missing_cvxpy"])
def test_optional_cell_does_not_swallow_other_failures(scenario, tmp_path, record_property):
    run_child(scenario, tmp_path, record_property)


def test_notebook_import_scope_changed_history_and_explicit_figure_requirement():
    doc = notebook(ROOT)
    assert len(doc["cells"]) == 8
    for index in (1, 2, 5, 6):
        assert doc["cells"][index]["outputs"] == []
        assert doc["cells"][index]["execution_count"] is None
    imports = {}
    for index in (1, 2, 5, 6):
        tree = ast.parse(python_cell(doc["cells"][index]))
        imports[index] = {(node.module, alias.name)
                          for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                          for alias in node.names}
    assert ("dwd.socp_dwd", "DWD") not in imports[1]
    assert ("dwd.socp_dwd", "DWD") in imports[2]
    assert imports[5] == {("sklearn.datasets", "make_blobs"), ("dwd.socp_dwd", "DWD")}
    assert imports[6] == {("sklearn.datasets", "make_circles"), ("dwd.gen_kern_dwd", "KernGDWD")}
    notice = "".join(doc["cells"][0]["source"])
    assert "retained as historical results" in notice
    assert "Full notebook execution has not been rerun" in notice
    # Complete equality of unchanged cells3/4/7 is a root snapshot check,
    # not a dependency on an external historical file in this permanent test.
    assert doc["cells"][3]["outputs"] and doc["cells"][3]["execution_count"] is not None
    figure_source = (ROOT / "doc/figures/make_figures.py").read_text(encoding="utf-8")
    figure_notice = ast.get_docstring(ast.parse(figure_source))
    assert figure_notice is not None
    assert "CVXPY" in figure_notice and "dwd[socp]" in figure_notice
    assert "both" in figure_notice and "kernel" in figure_notice and "linear" in figure_notice


if __name__ == "__main__":
    assert len(sys.argv) == 4 and sys.argv[1] == "--child"
    child_main(Path(sys.argv[2]).resolve(), sys.argv[3])
