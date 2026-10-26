"""Optional conic boundaries share promotion without changing auto-C policy."""
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy import sparse

cp = pytest.importorskip("cvxpy")

from dwd.socp_dwd import DWD, _prepare_auto_C_features, auto_dwd_C, solve_dwd_socp
from dwd.svm import SVM, solve_svm
from test_sparse_promotion import (DTYPES, LAYOUTS, Y, assert_unchanged,
                                   raw_features, snapshot)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_auto_C_promotes_duplicates_but_preserves_distance_dtype(layout, dtype):
    raw, expected = raw_features(dtype, layout, stale=True)
    before = snapshot(raw)
    prepared = _prepare_auto_C_features(raw)
    target = np.float32 if dtype == np.float32 else np.float64
    assert prepared.dtype == target
    assert_array_equal(prepared.toarray(), expected)
    assert auto_dwd_C(raw, Y) == auto_dwd_C(expected.astype(target), Y)
    assert_unchanged(raw, before)


def test_auto_C_normal_float32_fast_path():
    dense = np.arange(24., dtype=np.float32).reshape(8, 3)
    assert _prepare_auto_C_features(dense) is dense
    for cls in (sparse.csr_matrix, sparse.csr_array):
        raw = cls(dense)
        before = snapshot(raw)
        assert _prepare_auto_C_features(raw) is raw
        assert_unchanged(raw, before)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_auto_C_float32_overflow_refusal(layout):
    raw, _ = raw_features(np.float32, layout, readonly=False, stale=True)
    raw.data[:3] = np.array([2e38, 2e38, 1.], dtype=np.float32)
    before = snapshot(raw)
    with np.errstate(all="raise"), pytest.raises(ValueError):
        auto_dwd_C(raw, Y)
    assert_unchanged(raw, before)


def score(result, dense):
    if hasattr(result, "coef_"):
        return dense @ result.coef_.reshape(-1) + result.intercept_
    return dense @ result[0].reshape(-1) + result[1]


@pytest.mark.parametrize("layout", ("coo_array", "csr_matrix", "csc_array"))
@pytest.mark.parametrize("dtype", DTYPES)
def test_conic_public_fit_and_lowlevel_paths(layout, dtype):
    if "CLARABEL" not in cp.installed_solvers():
        pytest.skip("CLARABEL is needed for the optional conic numerical check")
    raw, dense = raw_features(dtype, layout, stale=True)
    before = snapshot(raw)
    solver = {"solver": "CLARABEL"}
    apis = (
        lambda x: DWD(C=.5, solver_kws=solver).fit(x, Y),
        lambda x: DWD(C="auto", solver_kws=solver).fit(x, Y),
        lambda x: solve_dwd_socp(x, Y, C=.5, solver_kws=solver),
        lambda x: SVM(C=.5, solver_kws=solver).fit(x, Y),
        lambda x: solve_svm(x, Y, C=.5, solver_kws=solver),
    )
    for call in apis:
        # Solvers may take slightly different stopping paths for sparse data;
        # compare the actual fitted decision function at solver accuracy.
        assert_allclose(score(call(raw), dense), score(call(dense), dense),
                        rtol=2e-5, atol=2e-6)
    assert_unchanged(raw, before)


@pytest.mark.parametrize("bad", (complex(1, 0), complex(1, 1), np.inf, np.nan))
def test_conic_bad_input_refused_before_solver(bad):
    raw, _ = raw_features(np.float32, "coo_array", readonly=False)
    if isinstance(bad, complex):
        raw = raw.astype(np.complex128)
    raw.data[0] = bad
    before = snapshot(raw)
    for call in (
        lambda: DWD(C=.5).fit(raw, Y), lambda: DWD(C="auto").fit(raw, Y),
        lambda: solve_dwd_socp(raw, Y, C=.5), lambda: SVM(C=.5).fit(raw, Y),
        lambda: solve_svm(raw, Y, C=.5), lambda: auto_dwd_C(raw, Y),
    ):
        with pytest.raises(ValueError):
            call()
    assert_unchanged(raw, before)
