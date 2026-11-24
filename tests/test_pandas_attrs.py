"""When preparing feature storage, preserve metadata accepted by stock loky."""
import os

import numpy as np
import pytest
from joblib import parallel_config
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV

from dwd.model_selection import fit_shared_fold_search


pytestmark = pytest.mark.usefixtures("search_parent_native_thread")


pd = pytest.importorskip("pandas")


class _ReadProvenance(RegressorMixin, BaseEstimator):
    def __init__(self, parent_pid):
        self.parent_pid = parent_pid

    def fit(self, X, y):
        self._check(X, y)
        if os.getpid() == self.parent_pid:
            self.refit_frame_ = X
        return self

    def score(self, X, y):
        assert os.getpid() != self.parent_pid
        self._check(X, y)
        return 1.0

    def _check(self, X, y):
        np.testing.assert_array_equal(X.to_numpy()[:, 0], y)
        assert not X.flags.allows_duplicate_labels
        source = X.attrs["source"]
        assert X.attrs["also_source"] is source
        assert source.tell() == 0
        assert source.readline() == "field observations\n"
        source.seek(0)


def test_process_preparation_preserves_serializable_attrs(tmp_path):
    # loky serializes an open text reader as an in-memory stream. Pandas can
    # deepcopy that stream when dividing the frame into folds in workers.
    rows = np.arange(8)
    path = tmp_path / "values.npy"
    np.save(path, rows)
    mapped = np.load(path, mmap_mode="c")
    frame = pd.DataFrame({"value": np.asarray(mapped)}, copy=False)
    assert np.shares_memory(frame.to_numpy(), mapped)
    frame.flags.allows_duplicate_labels = False
    provenance = tmp_path / "source.txt"
    provenance.write_text("field observations\n", encoding="utf-8")
    with provenance.open(encoding="utf-8") as source:
        frame.attrs.update(source=source, also_source=source)
        for use_helper in (False, True):
            search = GridSearchCV(_ReadProvenance(os.getpid()), {}, cv=2, n_jobs=2,
                                  error_score="raise")
            with parallel_config(backend="loky", inner_max_num_threads=1):
                if use_helper:
                    fit_shared_fold_search(search, frame, rows)
                else:
                    search.fit(frame, rows)
            assert search.best_estimator_.refit_frame_ is frame
            assert frame.attrs["source"] is source and source.tell() == 0
            np.testing.assert_array_equal(search.cv_results_["mean_test_score"], [1.])
