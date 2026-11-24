"""Check temporal axis metadata through public process search."""
import os

import numpy as np
import pytest
from joblib import parallel_config
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GridSearchCV

from dwd.model_selection import fit_shared_fold_search


pytestmark = pytest.mark.usefixtures("search_parent_native_thread")


pd = pytest.importorskip("pandas")


def _private_axis(path, axis):
    np.zeros(len(axis), dtype="int64").tofile(path)
    mapped = np.memmap(path, mode="c", dtype="int64")
    mapped[:] = axis.asi8
    result = type(axis)(np.asarray(mapped).view(axis.dtype), freq=axis.freq,
                        name=axis.name, copy=False)
    assert np.shares_memory(result.asi8, mapped)
    return result


@pytest.mark.parametrize("swap_axes,clear_freq", [(False, False), (True, False), (False, True)])
def test_temporal_axis_frequency_in_process_folds(tmp_path, swap_axes, clear_freq):
    date_freq = "D" if clear_freq else pd.offsets.CustomBusinessDay(weekmask="Mon Wed Thu")
    dates = pd.date_range("2025-01-01", periods=8, name="dates", unit="ns", freq=date_freq)
    deltas = pd.timedelta_range("1h", periods=8, name="elapsed", unit="ns", freq="2h")
    if clear_freq:
        dates.freq = deltas.freq = None
    index, columns = (deltas, dates) if swap_axes else (dates, deltas)
    data = np.arange(64).reshape(8, 8)
    expected = pd.DataFrame(data, index=index, columns=columns)
    frame = pd.DataFrame(data, index=_private_axis(tmp_path / "rows.bin", index),
                         columns=_private_axis(tmp_path / "columns.bin", columns))
    parent = os.getpid()

    def score(estimator, X, y):
        assert os.getpid() != parent
        fold = expected.iloc[y]
        pd.testing.assert_frame_equal(X, fold, check_exact=True)
        assert X.index.freq == fold.index.freq
        assert X.columns.freq == fold.columns.freq
        return 1.0

    search = GridSearchCV(DummyRegressor(), {}, cv=2, n_jobs=2, scoring=score,
                          error_score="raise")
    with parallel_config(backend="loky", inner_max_num_threads=1):
        fit_shared_fold_search(search, frame, np.arange(8))
    assert search.cv_results_["mean_test_score"].item() == 1.0
