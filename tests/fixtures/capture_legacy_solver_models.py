"""One-time capture from the immutable pre-retirement source, run by the owner.

This script never imports the candidate solver. All fitting occurs in the
explicit, hash-checked --source baseline. Outputs are created exclusively and
contain synthetic data only. It is not part of pytest collection.
"""

import argparse
import hashlib
import json
import pickle
import platform
import sys
import warnings
from pathlib import Path


PINS = {
    "dwd/gen_dwd.py": "54d83898b36bbbe55601dc2f27a1c889bc84aec286ff4da4db54a917e991142c",
    "dwd/gen_kern_dwd.py": "d1513a32453d39a980e12e1d9f9c78ea4bef2cbde69f6ddaaa135a906c867eb7",
    "dwd/cv.py": "409adf9ba23e8a4586f861ecb65d5ed3fc83c33cb3d3a081db811119be82fbc9",
    "dwd/kernel_utils.py": "429c0ed68412b6dbe1291a858a9594d1953bc58d015f6b280ed82a1ca1244fba",
    "dwd/linear_model.py": "155c76ea8804b9ea8dbcbcdeeedc1511e9c47f07531275c37c57661bc20235c7",
    "dwd/_fit_state.py": "f0a28ff82c8ec9311feab2a9d2e29fdbd8b161d92649e11c96650ed39e7b76e3",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest(source):
    return {path.relative_to(source).as_posix(): digest(path)
            for path in sorted((source / "dwd").rglob("*"))
            if path.is_file() and path.suffix.lower() in (".py", ".pyd", ".so")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    metadata_path = output.with_suffix(".json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError("Capture never overwrites an existing fixture or manifest")
    if output.suffix != ".npz":
        raise ValueError("The capture output must have .npz suffix")
    if any(name == "dwd" or name.startswith("dwd.") for name in sys.modules):
        raise RuntimeError("Capture requires a fresh process with no imported DWD")
    for relative, expected in PINS.items():
        if digest(source / relative) != expected:
            raise RuntimeError("Baseline source mismatch: " + relative)
    before = source_manifest(source)
    sys.path.insert(0, str(source))

    import numpy as np
    import scipy
    import sklearn
    import dwd
    from dwd.gen_dwd import GenDWD, GenDWDCV
    from dwd.gen_kern_dwd import KernGDWD, KernGDWDCV
    from sklearn.metrics.pairwise import rbf_kernel

    if Path(dwd.__file__).resolve() != source / "dwd" / "__init__.py":
        raise RuntimeError("Capture imported DWD outside the requested baseline")
    rng = np.random.RandomState(813)
    X = rng.normal(size=(16, 3))
    y = np.tile(np.array(["negative", "positive"]), 8)
    query = rng.normal(size=(7, 3))
    common = dict(solver_mode="legacy", lambd=.2, q=1., max_iter=3,
                  stopping="fixed", initialization="zero", random_state=7)
    cv_common = dict(solver_mode="legacy", lambd_vals=[.2], q_vals=[1.],
                     cv=2, max_iter=3, stopping="fixed", initialization="zero",
                     random_state=7)
    cases = [
        ("linear_implicit", GenDWD(implicit_P=True, **common), X, y, query, True),
        ("linear_explicit", GenDWD(implicit_P=False, **common), X, y, query, False),
        ("linear_cv", GenDWDCV(**cv_common), X, y, query, False),
        ("kernel_rbf", KernGDWD(kernel="rbf", kernel_kws={"gamma": .2}, **common),
         X, y, query, True),
        ("kernel_precomputed", KernGDWD(kernel="precomputed", **common),
         rbf_kernel(X, X, gamma=.2), y, rbf_kernel(query, X, gamma=.2), False),
        ("kernel_cv", KernGDWDCV(kernel="rbf", kernel_kws_vals=[{"gamma": .2}],
                                **cv_common), X, y, query, False),
    ]
    # These exact settings mirror the existing sparse/prediction regression
    # cases. Capture once rather than silently relabel a corrected fitted state.
    laplacian_X = np.array([[0., .25, -.5], [.5, 1., .25], [1., -.25, 0.],
                           [2., 1., .5], [-1., .5, 1.], [.25, -1., 2.],
                           [1.5, .25, -1.], [-.5, 2., .75]])
    laplacian_y = np.array([-1, -1, 1, 1, -1, 1, 1, -1])
    laplacian_query = np.array([[1., -.25, .5], [-.5, .75, 0.], [0., 0., 0.],
                               [0., 0., 0.], [.25, 1., -.5]])
    laplacian = KernGDWD(kernel="laplacian", kernel_kws={"gamma": .375},
                        backend="spectral", acceleration=None,
                        affine_computation="standard", residual_check_order="refinement_first",
                        **common)
    cases.append(("laplacian", laplacian, laplacian_X, laplacian_y,
                  laplacian_query, False))
    cosine_options = dict(kernel="cosine", solver_mode="legacy", lambd=.13,
                          max_iter=3, stopping="fixed", initialization="zero",
                          random_state=7, acceleration=None,
                          residual_check_order="refinement_first")
    cosine_X = np.array([[1., 0.], [0., 1.], [1., 1.], [-1., 1.]])
    cases.append(("cosine_mixed", KernGDWD(**cosine_options), cosine_X,
                  np.array([1, -1, 1, -1]), cosine_X.copy(), False))
    cosine_float32 = np.eye(4, dtype=np.float32)
    cases.append(("cosine_float32", KernGDWD(**cosine_options), cosine_float32,
                  np.array([-1, 1, -1, 1]), cosine_float32.copy(), False))
    poly_X = np.random.RandomState(13).normal(size=(8, 2))
    cases.append(("polynomial", KernGDWD(kernel="poly", kernel_kws={"degree": 4},
                  solver_mode="legacy", lambd=.3, max_iter=8, random_state=0),
                  poly_X, np.tile([0, 1], 4), poly_X.copy(), False))
    cases.append(("precomputed_error_policy", KernGDWD(kernel="precomputed",
                  implementation="optimized", affine_computation="standard", lambd=.1,
                  q=1., max_iter=8, stopping="fixed", initialization="zero", acceleration=None,
                  random_state=13, solver_mode="legacy", backend="spectral"),
                  np.eye(6), np.array([-1, 1, -1, 1, -1, 1]), np.eye(6), False))

    payload, records = {}, {}
    for name, model, train, labels, test, prepare in cases:
        with np.errstate(divide="warn", over="warn", invalid="warn", under="ignore"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if prepare:
                    model.cv_init(train)
                model.fit(train, labels)
                scores = model.decision_function(test)
                predictions = model.predict(test)
        if model.solver_mode != "legacy":
            raise AssertionError("Capture unexpectedly changed solver mode")
        if not np.isfinite(scores).all():
            raise AssertionError("Historical prediction is nonfinite: " + name)
        nested = getattr(model, "best_estimator_", None)
        if nested is not None and nested.solver_mode != "legacy":
            raise AssertionError("Historical CV used a different solver mode")
        encoded = pickle.dumps(model, protocol=4)
        payload[name + "__pickle"] = np.frombuffer(encoded, dtype=np.uint8).copy()
        for key, value in (("X", train), ("y", labels), ("query", test),
                           ("scores", scores), ("labels", predictions)):
            payload[name + "__" + key] = np.asarray(value).copy()
        records[name] = {
            "class": type(model).__module__ + "." + type(model).__name__,
            "fitted_solver_mode": model.solver_mode,
            "nested_solver_mode": None if nested is None else nested.solver_mode,
            "prepared_cache": prepare,
            "n_iter": int(getattr(nested or model, "n_iter_", -1)),
            "pickle_sha256": hashlib.sha256(encoded).hexdigest(),
            "warnings": [{"category": item.category.__name__, "message": str(item.message)}
                         for item in caught],
        }
    if source_manifest(source) != before:
        raise RuntimeError("Baseline changed during fixture capture")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, **payload)
    metadata = {
        "schema": 1, "purpose": "Genuine pre-retirement fitted prediction states; not new correctness claims",
        "case_names": [case[0] for case in cases], "cases": records,
        "fixture_sha256": digest(output), "source_root": str(source),
        "source_sha256": before, "required_source_pins": PINS,
        "capture_script_sha256": digest(Path(__file__).resolve()),
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "numpy": np.__version__, "scipy": scipy.__version__,
                        "sklearn": sklearn.__version__, "dwd": dwd.__version__},
    }
    with metadata_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"fixture": str(output), "sha256": metadata["fixture_sha256"],
                      "manifest": str(metadata_path), "cases": len(cases)}, sort_keys=True))


if __name__ == "__main__":
    main()
