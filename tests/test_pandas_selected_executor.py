"""The process executor can be selected when sklearn counts the folds."""
from functools import partial
import os

import numpy as np
import pytest
from joblib import parallel_config
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GridSearchCV, KFold

from dwd.model_selection import fit_shared_fold_search


pytestmark = pytest.mark.usefixtures("search_parent_native_thread")


pd = pytest.importorskip("pandas")


class _ChooseWorkersDuringSplitCount:
    def get_n_splits(self, X=None, y=None, groups=None):
        self.search.n_jobs = 2
        return 2

    def split(self, X, y=None, groups=None):
        yield from KFold(2).split(X, y)


def _correct_first_column(estimator, X, y, *, parent_pid):
    assert os.getpid() != parent_pid
    return float(np.mean(X.iloc[:, 0].to_numpy() == 4 * y))


@pytest.mark.parametrize("entry_jobs", [1, None], ids=["serial-on-entry", "default-on-entry"])
def test_worker_frame_follows_jobs_selected_by_splitter(tmp_path, entry_jobs):
    path = tmp_path / "features.npy"
    np.save(path, np.arange(32.).reshape(8, 4))
    mapped = np.load(path, mmap_mode="r")
    frame = pd.DataFrame(mapped, copy=False)
    assert np.shares_memory(frame.to_numpy(copy=False), mapped)

    cv = _ChooseWorkersDuringSplitCount()
    score = partial(_correct_first_column, parent_pid=os.getpid())
    search = GridSearchCV(DummyRegressor(), {}, cv=cv, n_jobs=entry_jobs,
                          scoring=score, error_score="raise")
    cv.search = search
    # No callback or ambient parallelism. Sklearn asks for the split count
    # before creating its executor, so the splitter can select two processes.
    with parallel_config(backend="loky", n_jobs=1, inner_max_num_threads=1):
        assert fit_shared_fold_search(search, frame, np.arange(8)) is search

    assert search.n_jobs == 2
    for fold in range(2):
        np.testing.assert_array_equal(search.cv_results_[f"split{fold}_test_score"], [1.])


@pytest.fixture
def empty_mapped_level(tmp_path):
    path = tmp_path / "empty-level.bin"
    path.write_bytes(bytes(np.dtype(object).itemsize))
    mapped = np.memmap(path, mode="r", dtype=object, shape=(0,))
    level = pd.Index(np.asarray(mapped), dtype=object, copy=False)
    index = pd.MultiIndex(levels=[level], codes=[np.full(8, -1)], verify_integrity=False)
    # Zero object slots and eight usable rows. Verify the retained mapping
    # independently of DWD, since shares_memory is false for empty arrays.
    view = index.levels[0].to_numpy(copy=False)
    assert view.size == 0 and view.dtype == np.dtype(object)
    while isinstance(view, np.ndarray) and view is not mapped:
        view = view.base
    assert view is mapped
    return index


@pytest.mark.parametrize("entry_jobs,selected_jobs,expected_events", [
    (1, 2, ["count"]), (2, 2, []), (1, 1, ["count", "split"]),
], ids=["select-process-during-count", "process-before-count", "serial"])
def test_mapped_object_guard_follows_selected_executor(
        empty_mapped_level, entry_jobs, selected_jobs, expected_events):
    events = []
    rows = np.arange(8)
    frame = pd.DataFrame({"value": rows}, index=empty_mapped_level)

    class Splitter:
        def get_n_splits(self, X=None, y=None, groups=None):
            events.append("count")
            search.n_jobs = selected_jobs
            return 2

        def split(self, X, y=None, groups=None):
            events.append("split")
            if selected_jobs == 2:
                pytest.fail("parent splitter received the mapped object level")
            return KFold(2).split(X, y)

    search = GridSearchCV(DummyRegressor(), {}, cv=Splitter(), n_jobs=entry_jobs,
                          error_score="raise")
    with parallel_config(backend="loky", inner_max_num_threads=1):
        if selected_jobs == 2:
            with pytest.raises(ValueError, match="file-backed object arrays"):
                fit_shared_fold_search(search, frame, rows)
        else:
            assert fit_shared_fold_search(search, frame, rows) is search
    assert events == expected_events


def test_candidate_iteration_can_replace_a_checked_frame_axis(empty_mapped_level):
    rows = np.arange(8)
    frame = pd.DataFrame({"value": rows})
    events = []

    class Splitter:
        def get_n_splits(self, X=None, y=None, groups=None):
            events.append("count")
            return 2

        def split(self, X, y=None, groups=None):
            pytest.fail("parent splitter received the replacement mapped object level")

    class Search(GridSearchCV):
        def _run_search(self, evaluate_candidates):
            def candidates():
                frame.index = empty_mapped_level
                events.append("replace index")
                yield {}

            evaluate_candidates(candidates())

    search = Search(DummyRegressor(), {}, cv=Splitter(), n_jobs=2, error_score="raise")
    with parallel_config(backend="loky", inner_max_num_threads=1):
        with pytest.raises(ValueError, match="file-backed object arrays"):
            fit_shared_fold_search(search, frame, rows)
    assert events == ["count", "replace index"]
