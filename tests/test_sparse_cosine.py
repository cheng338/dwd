"""Sparse cosine geometry, tested independently of candidate implementations.

The runner selects one normal dwd import in each fresh process. These tests do
not import other candidates. Historical states below are explicitly constructed
fixtures, not claims that a released wheel generated a particular pickle.
"""
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from dwd.gen_kern_dwd import KernGDWD, KernGDWDCV
from dwd.kern_md import KernMD


LAYOUTS = ("csr_matrix", "csr_array", "csc_matrix", "csc_array", "coo_matrix", "coo_array")
CSR = ("csr_matrix", "csr_array")
X = np.array([[0,0,0,0], [1,0,0,1], [0,2,1,0], [-2,1,0,0],
              [2,0,2,0], [-1,-2,0,0], [1,1,-2,0], [0,-1,2,0]], dtype=float)
Y = np.array([-1,1,-1,1,-1,1,-1,1])
QUERY = X[[3,0,5,7,1]].copy()
OPTIONS = dict(lambd=.13, max_iter=3, stopping="fixed", initialization="zero",
               random_state=7, acceleration=None, residual_check_order="refinement_first")


@pytest.fixture(autouse=True)
def _local_numpy_policy():
    # Strict-underflow nextafter behavior is a separate existing finding.
    with np.errstate(divide="warn", over="warn", invalid="warn", under="ignore"):
        yield


def _buffers(value):
    if not sparse.issparse(value):
        return (value,)
    return ((value.data,value.row,value.col) if value.format == "coo"
            else (value.data,value.indices,value.indptr))


def _snapshot(value):
    metadata = ({name:item for name,item in value.__dict__.items()
                 if name in ("has_canonical_format","_has_canonical_format","_has_sorted_indices")}
                if sparse.issparse(value) else {})
    return (value.shape, tuple((str(a.dtype),a.shape,a.strides,a.flags.writeable,
                              a.__array_interface__["data"][0],a.tobytes(order="A"))
                             for a in _buffers(value)), metadata)


def _readonly(value):
    for buffer in _buffers(value):
        buffer.flags.writeable = False
    return value


def _from_entries(entries, shape, layout, dtype=np.float64, *, unsorted=False,
                  stale=False, readonly=True):
    entries = list(entries)
    if unsorted:
        entries.reverse()
    rows = np.array([e[0] for e in entries],dtype=np.int32)
    columns = np.array([e[1] for e in entries],dtype=np.int32)
    values = np.array([e[2] for e in entries],dtype=dtype)
    cls = getattr(sparse,layout)
    if layout.startswith("coo"):
        result = cls((values,(rows,columns)),shape=shape)
    else:
        major, minor, size = (rows,columns,shape[0]) if layout.startswith("csr") else (columns,rows,shape[1])
        order = np.argsort(major,kind="stable")
        indptr = np.asarray(np.r_[0,np.cumsum(np.bincount(major,minlength=size))],dtype=np.int32)
        result = cls((values[order],minor[order],indptr),shape=shape)
    if stale:
        result.has_canonical_format = True
        if result.format != "coo":
            result.has_sorted_indices = True
    return _readonly(result) if readonly else result


def _features(dense, layout, dtype=np.float64, *, unsorted=False, stale=False,
              readonly=True, duplicates=True):
    entries = []
    for row, vector in enumerate(dense):
        for col, value in enumerate(vector):
            if value:
                if duplicates:
                    first = (2+(row+col)%2)*float(value)
                    entries.extend(((row,col,first),(row,col,float(value)-first)))
                else:
                    entries.append((row,col,float(value)))
        # Keep row zero genuinely empty; other rows have explicit stored zeros.
        if np.any(vector) and (duplicates or not vector[-1]):
            entries.append((row,len(vector)-1,0.))
    return _from_entries(entries,dense.shape,layout,dtype,unsorted=unsorted,
                         stale=stale,readonly=readonly)


def _coordinates(value):
    """Aggregate original stored values as rationals without sparse conversion."""
    answer = [[Fraction() for _ in range(value.shape[1])] for _ in range(value.shape[0])]
    if not sparse.issparse(value):
        for row in range(value.shape[0]):
            for col in range(value.shape[1]):
                answer[row][col] = Fraction(float(value[row,col]))
    elif value.format == "coo":
        for row,col,item in zip(value.row,value.col,value.data):
            answer[int(row)][int(col)] += Fraction(float(item))
    elif value.format == "csr":
        for row in range(value.shape[0]):
            for pos in range(value.indptr[row],value.indptr[row+1]):
                answer[row][int(value.indices[pos])] += Fraction(float(value.data[pos]))
    elif value.format == "csc":
        for col in range(value.shape[1]):
            for pos in range(value.indptr[col],value.indptr[col+1]):
                answer[int(value.indices[pos])][col] += Fraction(float(value.data[pos]))
    else:
        raise AssertionError("Unexpected oracle storage format")
    return answer


def _dense_oracle(value):
    return np.asarray([[float(item) for item in row] for row in _coordinates(value)])


def _cosine_oracle(left, right):
    a, b = _coordinates(left), _coordinates(right)
    result = np.empty((len(a),len(b)))
    with localcontext() as context:
        context.prec = 80
        def dec(value):
            return Decimal(value.numerator)/Decimal(value.denominator)
        an = [dec(sum((item*item for item in row),Fraction())).sqrt() for row in a]
        bn = [dec(sum((item*item for item in row),Fraction())).sqrt() for row in b]
        for i,row in enumerate(a):
            for j,other in enumerate(b):
                dot = sum((x*y for x,y in zip(row,other)),Fraction())
                result[i,j] = float(dec(dot)/(an[i]*bn[j])) if an[i] and bn[j] else 0.
    return result


def _model(**extra):
    return KernGDWD(**dict(OPTIONS,kernel="cosine",**extra))


def _fixed_state(basis, *, historical_missing=False):
    """Construct only prediction state; do not pretend a fit generated it."""
    model = _model()
    model._Xfit = basis
    model.n_features_in_ = basis.shape[1]
    model.dual_coef_ = (np.arange(basis.shape[0],dtype=float)%3-1).reshape(1,-1)/4.
    model.intercept_ = np.array([.125])
    model.classes_ = np.array([-1,1])
    model.prediction_precision_ = "ordinary"
    model.affine_computation_ = "standard"
    if not historical_missing:
        model.kernel_ = "cosine"
        model.kernel_kws_ = None
        model.kernel_computation_ = "sklearn"
    return model


@pytest.mark.parametrize("layout",LAYOUTS)
@pytest.mark.parametrize("dtype",[np.int8,np.float32,np.float64])
@pytest.mark.parametrize("storage",["sorted","unsorted_stale"])
def test_fit_preparation_kernel_matches_coordinate_oracle(layout,dtype,storage):
    raw = _features(X,layout,dtype,unsorted=storage!="sorted",stale=storage!="sorted")
    query = _features(QUERY,layout,dtype,unsorted=storage!="sorted",stale=storage!="sorted")
    before, query_before = _snapshot(raw), _snapshot(query)
    expected = _cosine_oracle(raw,raw)
    model = _model().cv_init(raw)
    assert_allclose(model._cv_K,expected,rtol=2e-13,atol=2e-13)
    # cv_init has no labels. Supply only validation metadata, without claiming
    # a fitted predictor; the validation boundary performs public conversion.
    model.classes_ = np.array([-1,1])
    model.n_features_in_ = raw.shape[1]
    validation, labels = model._prepare_validation((query,Y[[3,0,5,7,1]]))
    assert_allclose(validation,_cosine_oracle(query,raw),rtol=2e-13,atol=2e-13)
    assert_array_equal(labels,Y[[3,0,5,7,1]])
    assert model._cv_cache_matches(model._Xfit)
    assert_allclose(np.diag(expected),np.where(np.any(X,axis=1),1.,0.),rtol=2e-15,atol=2e-15)
    assert _snapshot(raw)==before
    assert _snapshot(query)==query_before


@pytest.mark.parametrize("layout",CSR)
def test_analytic_duplicate_and_cancellation_directions(layout):
    raw = _from_entries([(0,0,.5),(0,0,.5),(1,0,1.),(1,0,-1.),(1,1,1.),
                         (2,0,2.**40),(2,0,1.),(2,0,-2.**40),(3,1,0.)],(5,2),layout)
    before = _snapshot(raw)
    model = KernMD(kernel="cosine").fit(np.eye(2),np.array([-1,1]))
    expected = np.array([[1.,0.,1.,0.,0.],[0.,1.,0.,0.,0.]])
    assert_allclose(_cosine_oracle(np.eye(2),raw),expected,rtol=0,atol=0)
    assert_allclose(model._compute_kernel(raw),expected,rtol=2e-13,atol=2e-13)
    assert _snapshot(raw)==before


@pytest.mark.parametrize("layout",CSR)
@pytest.mark.parametrize("placement",["training","query","both"])
def test_real_fit_query_batches_and_validation(layout,placement):
    raw = _features(X,layout,unsorted=True,readonly=False)
    query = _features(QUERY,layout,unsorted=True)
    train = raw if placement in ("training","both") else X.copy()
    test = query if placement in ("query","both") else QUERY.copy()
    before, query_before = _snapshot(train), _snapshot(test)
    model, control = _model().fit(train,Y), _model().fit(X,Y)
    assert_allclose(model._compute_kernel(model._Xfit),_cosine_oracle(X,X),rtol=2e-13,atol=2e-13)
    expected = control.decision_function(QUERY)
    for batch in (None,1,3):
        assert_allclose(model.decision_function(test,prediction_batch_size=batch),expected,rtol=2e-8,atol=2e-10)
    assert_array_equal(model.predict(test),control.predict(QUERY))
    assert_allclose(model.objective_history_,control.objective_history_,rtol=2e-9,atol=2e-11)
    gram, labels = model._prepare_validation((test,Y[[3,0,5,7,1]]))
    assert_allclose(gram,_cosine_oracle(QUERY,X),rtol=2e-13,atol=2e-13)
    assert_array_equal(labels,Y[[3,0,5,7,1]])
    assert _snapshot(train)==before
    assert _snapshot(test)==query_before


@pytest.mark.parametrize("layout",CSR)
@pytest.mark.parametrize("missing",[False,True])
def test_historical_duplicate_basis_requires_refit_without_mutation(layout,missing):
    raw = _features(X,layout,unsorted=True,stale=True)
    model = _fixed_state(raw,historical_missing=missing)
    before, coefficients = _snapshot(raw), _snapshot(model.dual_coef_)
    for batch in (None,1,3):
        with pytest.raises(ValueError,match="(?i)refit"):
            model.decision_function(QUERY,prediction_batch_size=batch)
    with pytest.raises(ValueError,match="(?i)refit"):
        model._prepare_validation((QUERY,Y[[3,0,5,7,1]]))
    assert _snapshot(raw)==before
    assert _snapshot(model.dual_coef_)==coefficients


@pytest.mark.parametrize("layout",CSR)
def test_unsorted_unique_historical_basis_remains_usable(layout):
    raw = _features(X,layout,unsorted=True,stale=True,duplicates=False)
    model = _fixed_state(raw,historical_missing=True)
    before = _snapshot(raw)
    expected = (_cosine_oracle(QUERY,X)@model.dual_coef_.T+model.intercept_).ravel()
    for batch in (None,1,3):
        assert_allclose(model.decision_function(QUERY,prediction_batch_size=batch),expected,rtol=2e-13,atol=2e-13)
    assert _snapshot(raw)==before


@pytest.mark.parametrize("layout",CSR)
def test_literal_historical_wrong_gram_cache_is_invalidated(layout):
    dense = np.eye(4)
    entries = [(row,row,.5) for row in range(4) for _ in range(2)]
    raw = _from_entries(entries,(4,4),layout,stale=True)
    model = _model().cv_init(dense)
    # Slotwise normalization of two .5 values gives the literal old 2*I Gram.
    model._cv_X = raw
    model._cv_K = 2*np.eye(4)
    before, old_gram = _snapshot(raw), model._cv_K.copy()
    assert not model._cv_cache_matches(getattr(sparse,layout)(dense))
    assert not model._cv_cache_matches(raw)
    fitted = model.fit(raw,np.array([-1,1,-1,1]))
    control = _model().fit(dense,np.array([-1,1,-1,1]))
    assert_allclose(fitted.decision_function(dense),control.decision_function(dense),rtol=2e-9,atol=2e-11)
    assert _snapshot(raw)==before
    assert_array_equal(model._cv_K,old_gram)


@pytest.mark.parametrize("layout",LAYOUTS)
@pytest.mark.parametrize("estimator",["mean_difference","kernel_dwd"])
def test_mixed_dtype_query_promotes_before_duplicate_sum(layout,estimator):
    dense = np.array([[1.,0.],[0.,1.],[1.,1.],[-1.,1.]])
    labels = np.array([1,-1,1,-1])
    raw = _from_entries([(0,0,2.**24),(0,0,1.),(0,0,-2.**24),(1,1,1.),(2,1,0.)],
                        (3,2),layout,np.float32)
    before = _snapshot(raw)
    model = (KernMD(kernel="cosine").fit(dense,labels) if estimator == "mean_difference"
             else _model(affine_computation="standard").fit(dense, labels))
    expected = (_cosine_oracle(raw,dense)@model.dual_coef_.T+model.intercept_).ravel()
    for batch in (None,1,2):
        assert_allclose(model.decision_function(raw,prediction_batch_size=batch),expected,rtol=2e-12,atol=2e-12)
    assert _snapshot(raw)==before


@pytest.mark.parametrize("layout",CSR)
@pytest.mark.parametrize("estimator",["mean_difference","kernel_dwd"])
def test_both_float32_moderate_query_policy(layout,estimator):
    # Full-rank training isolates query normalization from singular geometry.
    dense = np.eye(4,dtype=np.float32)
    labels = np.array([-1,1,-1,1])
    query = np.array([[1,0,0,0],[1,1,0,0],[0,0,0,0],[0,0,-2,1]],dtype=np.float32)
    raw = _features(query,layout,np.float32)
    before = _snapshot(raw)
    model = (KernMD(kernel="cosine").fit(dense,labels) if estimator == "mean_difference"
             else _model(affine_computation="standard").fit(dense, labels))
    expected = (_cosine_oracle(raw,dense)@model.dual_coef_.T+model.intercept_).ravel()
    assert_allclose(model.decision_function(raw),expected,rtol=2e-6,atol=2e-7)
    assert _snapshot(raw)==before


@pytest.mark.parametrize("layout",CSR)
def test_dense_mean_difference_fit_sparse_query(layout):
    raw = _features(QUERY,layout,unsorted=True,stale=True)
    before = _snapshot(raw)
    model = KernMD(kernel="cosine").fit(X,Y)
    expected = (_cosine_oracle(raw,X)@model.dual_coef_.T+model.intercept_).ravel()
    assert_allclose(model.decision_function(raw),expected,rtol=2e-13,atol=2e-13)
    assert _snapshot(raw)==before


@pytest.mark.parametrize("route",["wrapper","run_cv"])
def test_small_real_cross_validation_parity(route):
    raw = _features(X,"csr_matrix",unsorted=True)
    before = _snapshot(raw)
    if route == "wrapper":
        options = dict(OPTIONS)
        options.pop("lambd")
        options.update(kernel="cosine",lambd_vals=[.13],q_vals=[1.],cv=2)
        model, control = KernGDWDCV(**options).fit(raw,Y), KernGDWDCV(**options).fit(X,Y)
    else:
        from dwd.cv import run_cv
        first = run_cv(_model(),raw,Y,{"lambd":[.13]},cv=2)
        second = run_cv(_model(),X,Y,{"lambd":[.13]},cv=2)
        assert first[0]==second[0]
        model, control = first[2], second[2]
    assert_allclose(model.decision_function(QUERY),control.decision_function(QUERY),rtol=2e-8,atol=2e-10)
    assert _snapshot(raw)==before


@pytest.mark.parametrize("kernel",["callable","precomputed","rbf"])
def test_other_kernel_contracts_remain_unchanged(kernel):
    raw = _features(X,"csr_matrix",unsorted=True)
    query = _features(QUERY,"csr_matrix",unsorted=True)
    before, query_before = _snapshot(raw), _snapshot(query)
    if kernel == "callable":
        observed_storage = []
        def literal_linear(left,right):
            observed_storage.append((left.nnz,right.nnz))
            return _dense_oracle(left)@_dense_oracle(right).T
        model = KernGDWD(**dict(OPTIONS,kernel=literal_linear)).fit(raw,Y)
        expected = (_dense_oracle(query)@X.T@model.dual_coef_.T+model.intercept_).ravel()
        assert_allclose(model.decision_function(query),expected,rtol=2e-12,atol=2e-12)
        assert all(left_nnz==raw.nnz for left_nnz,_ in observed_storage)
        assert any(right_nnz==query.nnz for _,right_nnz in observed_storage)
    elif kernel == "precomputed":
        gram = X@X.T
        model = KernGDWD(**dict(OPTIONS,kernel="precomputed")).fit(gram,Y)
        query_gram = _features(QUERY@X.T,"csr_matrix",unsorted=True)
        saved = _snapshot(query_gram)
        expected = (QUERY@X.T@model.dual_coef_.T+model.intercept_).ravel()
        assert_allclose(model.decision_function(query_gram),expected,rtol=2e-12,atol=2e-12)
        assert _snapshot(query_gram)==saved
    else:
        options = dict(OPTIONS,kernel="rbf",kernel_kws={"gamma":.5})
        model, control = KernGDWD(**options).fit(raw,Y), KernGDWD(**options).fit(X,Y)
        assert_allclose(model.decision_function(query),control.decision_function(QUERY),rtol=2e-10,atol=2e-12)
    assert _snapshot(raw)==before
    assert _snapshot(query)==query_before


@pytest.mark.parametrize("layout",CSR)
@pytest.mark.parametrize("bad",[np.nan,np.inf,1.j])
def test_nonfinite_and_complex_cosine_query_refusals(layout,bad):
    raw = _from_entries([(0,0,bad)],(1,4),layout,np.complex128 if isinstance(bad,complex) else np.float64)
    if np.iscomplexobj(raw.data):
        # Existing complex-error formatting populates these lazy SciPy caches.
        # Prepopulate them so this test checks refusal and unchanged buffers/
        # existing flags, without asserting a new invalid-input metadata rule.
        bool(raw.has_canonical_format)
        bool(raw.has_sorted_indices)
    before = _snapshot(raw)
    model = _fixed_state(X.copy())
    with pytest.raises((TypeError,ValueError,FloatingPointError)):
        model.decision_function(raw)
    assert _snapshot(raw)==before


@pytest.mark.parametrize("layout",CSR)
def test_nonfinite_duplicate_coordinate_sum_refuses(layout):
    raw = _from_entries([(0,0,1e308),(0,0,1e308)],(1,4),layout)
    before = _snapshot(raw)
    model = _fixed_state(X.copy())
    with pytest.raises((ValueError,FloatingPointError,OverflowError)):
        model.decision_function(raw)
    assert _snapshot(raw)==before
