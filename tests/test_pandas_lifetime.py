"""Check when process searches release their detached worker feature frames."""
import gc
import sys
import weakref

import numpy as np
import pytest
from joblib import parallel_config
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GridSearchCV

from dwd import model_selection


pytestmark = pytest.mark.usefixtures("search_parent_native_thread")


pd = pytest.importorskip("pandas")


class _EvaluateTwice(GridSearchCV):
    def _run_search(self, evaluate_candidates):
        evaluate_candidates([{}])
        evaluate_candidates([{}])


class _FailDuringResults(GridSearchCV):
    def _format_results(self, *args, **kwargs):
        raise RuntimeError("parent failed while formatting candidate results")


@pytest.mark.skipif(sys.implementation.name != "cpython",
                    reason="checks immediate release by CPython reference counting")
@pytest.mark.parametrize("search_type", [GridSearchCV, _EvaluateTwice, _FailDuringResults],
                         ids=["ordinary-fit", "repeated-evaluation", "evaluation-error"])
def test_process_search_releases_detached_frame(tmp_path, monkeypatch, search_type):
    path = tmp_path / "features.npy"
    np.save(path, np.arange(32.).reshape(8, 4))
    mapped = np.load(path, mmap_mode="r")
    frame = pd.DataFrame(mapped, copy=False)
    assert np.shares_memory(frame.to_numpy(copy=False), mapped)
    search = search_type(DummyRegressor(), {}, cv=2, n_jobs=2, error_score="raise")

    snapshots = []
    alive_before_next_copy = []
    prepare = model_selection._parallel_search_features

    def observe(value, *args, **kwargs):
        prepared = prepare(value, *args, **kwargs)
        if prepared is not value:
            alive_before_next_copy.append(sum(ref() is not None for ref in snapshots))
            snapshots.append(weakref.ref(prepared))
        return prepared

    monkeypatch.setattr(model_selection, "_parallel_search_features", observe)
    # A collection during dispatch can hide a reference cycle. Collection is
    # disabled only for this short fit, leaving reference counting in effect.
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        with parallel_config(backend="loky", inner_max_num_threads=1):
            if search_type is _FailDuringResults:
                with pytest.raises(RuntimeError, match="parent failed while formatting"):
                    model_selection.fit_shared_fold_search(search, frame, np.arange(8))
            else:
                assert model_selection.fit_shared_fold_search(search, frame, np.arange(8)) is search
        count = 2 if search_type is _EvaluateTwice else 1
        assert len(snapshots) == count
        assert alive_before_next_copy == [0] * count
        assert all(ref() is None for ref in snapshots)
    finally:
        if was_enabled:
            gc.enable()
