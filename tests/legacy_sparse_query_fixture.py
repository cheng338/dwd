"""Trusted supplemental genuine historical state for sparse query regressions."""

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "legacy_sparse_query_retirement.npz"


def load_legacy_sparse_query_case():
    """Return fresh saved arrays/model, without fitting any historical solver."""
    manifest = json.loads(FIXTURE.with_suffix(".json").read_text(encoding="utf-8"))
    if manifest["schema"] != 1 or manifest["case"] != "sparse_query_rbf":
        raise AssertionError("Unexpected historical sparse-query fixture schema")
    if hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != manifest["fixture_sha256"]:
        raise AssertionError("Historical sparse-query fixture hash mismatch")
    if manifest["fitted_solver_mode"] != "legacy":
        raise AssertionError("Supplemental state was not genuinely fitted in legacy mode")
    with np.load(FIXTURE, allow_pickle=False) as saved:
        model_bytes = saved["pickle"].tobytes()
        if hashlib.sha256(model_bytes).hexdigest() != manifest["pickle_sha256"]:
            raise AssertionError("Historical sparse-query pickle hash mismatch")
        result = {key: saved[key].copy() for key in ("X", "y", "query", "scores", "labels")}
    result["model"] = pickle.loads(model_bytes)
    if result["model"].solver_mode != "legacy":
        raise AssertionError("Historical sparse-query solver attribution changed")
    return result
