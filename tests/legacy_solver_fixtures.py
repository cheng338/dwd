"""Load trusted, genuinely legacy-fitted regression states; never fit a model.

The adjacent manifest records the immutable source and capture environment.
Only repository-controlled fixture bytes are unpickled here. This helper is
test support, not a public compatibility path or historical runtime emulator.
"""

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "legacy_solver_retirement.npz"
MANIFEST = FIXTURE.with_suffix(".json")
CASE_NAMES = (
    "linear_implicit", "linear_explicit", "linear_cv", "kernel_rbf",
    "kernel_precomputed", "kernel_cv", "laplacian", "cosine_mixed",
    "cosine_float32", "polynomial", "precomputed_error_policy",
)


def load_legacy_case(name):
    """Return a fresh model and fresh arrays from an immutable baseline capture."""
    if name not in CASE_NAMES:
        raise KeyError(name)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    raw = FIXTURE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["fixture_sha256"]:
        raise AssertionError("Legacy fixture bytes differ from the capture manifest")
    if manifest["schema"] != 1 or tuple(manifest["case_names"]) != CASE_NAMES:
        raise AssertionError("Unexpected legacy fixture manifest")
    case_manifest = manifest["cases"][name]
    if case_manifest["fitted_solver_mode"] != "legacy":
        raise AssertionError("The fixture was not genuinely fitted in legacy mode")
    with np.load(FIXTURE, allow_pickle=False) as saved:
        model_bytes = saved[name + "__pickle"].tobytes()
        if hashlib.sha256(model_bytes).hexdigest() != case_manifest["pickle_sha256"]:
            raise AssertionError("Legacy model differs from its capture receipt")
        result = {key: saved[name + "__" + key].copy()
                  for key in ("X", "y", "query", "scores", "labels")}
    model = pickle.loads(model_bytes)
    if model.solver_mode != "legacy":
        raise AssertionError("Loaded model lost its historical solver attribution")
    result["model"] = model
    return result
