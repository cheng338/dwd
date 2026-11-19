"""Capture one additional genuine legacy RBF state from the immutable baseline.

The original 11-case archive is never read for fitting, rewritten or regenerated.
This root-run script shares its source-pin validation definitions only.
"""

import argparse
import hashlib
import importlib.util
import json
import pickle
import platform
import sys
import warnings
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    metadata_path = output.with_suffix(".json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError("Supplemental capture never overwrites existing files")
    if output.suffix != ".npz":
        raise ValueError("Expected a .npz output")
    if any(name == "dwd" or name.startswith("dwd.") for name in sys.modules):
        raise RuntimeError("Capture requires a fresh process")
    definitions = Path(__file__).with_name("capture_legacy_solver_models.py")
    spec = importlib.util.spec_from_file_location("_legacy_capture_definitions", definitions)
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    for relative, expected in shared.PINS.items():
        if shared.digest(source / relative) != expected:
            raise RuntimeError("Baseline source mismatch: " + relative)
    before = shared.source_manifest(source)
    sys.path.insert(0, str(source))

    import numpy as np
    import scipy
    import sklearn
    import dwd
    from dwd.gen_kern_dwd import KernGDWD

    if Path(dwd.__file__).resolve() != source / "dwd" / "__init__.py":
        raise RuntimeError("Imported DWD outside the pinned baseline")
    X = np.array([[0., 0.], [.5, 1.], [1., 0.], [2., 1.]])
    y = np.array([0, 0, 1, 1])
    query = np.array([[1., 0.], [-1., .5], [0., 0.], [0., 0.], [.5, 1.]])
    model = KernGDWD(kernel="rbf", kernel_kws={"gamma": 1.}, solver_mode="legacy",
                    lambd=.2, max_iter=2, stopping="fixed", initialization="zero",
                    random_state=7, acceleration=None, affine_computation="standard",
                    residual_check_order="refinement_first")
    with np.errstate(divide="warn", over="warn", invalid="warn", under="ignore"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X, y)
            scores, labels = model.decision_function(query), model.predict(query)
    if model.solver_mode != "legacy" or model.n_iter_ != 2 or not np.isfinite(scores).all():
        raise AssertionError("Supplemental baseline fit did not produce the specified state")
    model_bytes = pickle.dumps(model, protocol=4)
    if shared.source_manifest(source) != before:
        raise RuntimeError("Baseline changed during supplemental capture")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, X=X, y=y, query=query, scores=scores, labels=labels,
                            pickle=np.frombuffer(model_bytes, dtype=np.uint8).copy())
    metadata = {
        "schema": 1, "case": "sparse_query_rbf", "fitted_solver_mode": model.solver_mode,
        "n_iter": int(model.n_iter_), "parameters": model.get_params(deep=False),
        "fixture_sha256": shared.digest(output),
        "pickle_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "source_root": str(source), "source_sha256": before,
        "required_source_pins": shared.PINS,
        "capture_script_sha256": shared.digest(Path(__file__).resolve()),
        "validation_definitions_sha256": shared.digest(definitions),
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "numpy": np.__version__, "scipy": scipy.__version__,
                        "sklearn": sklearn.__version__, "dwd": dwd.__version__},
        "warnings": [{"category": item.category.__name__, "message": str(item.message)}
                     for item in caught],
    }
    with metadata_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(metadata, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"fixture": str(output), "sha256": metadata["fixture_sha256"],
                      "manifest": str(metadata_path), "cases": 1}, sort_keys=True))


if __name__ == "__main__":
    main()
