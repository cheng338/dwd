"""Exact integer-coordinate policy through actual public input boundaries."""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from dwd.gen_dwd import GenDWD, GenDWDCV, _augmented_gram, solve_gen_dwd
from dwd.gen_kern_dwd import KernGDWD, KernGDWDCV
from dwd.cv import run_cv
from test_sparse_promotion import KERNELS, literal_kernel

OPTIONS=dict(lambd=.1,max_iter=2,stopping='fixed',initialization='zero',random_state=7)
Y=np.tile([-1,1],4)


def raw_features(layout='csr_array', three=False):
    prefix=(2**53,1,-2**53) if three else (2**53+1,-2**53)
    entries=[(0,0,value) for value in prefix]
    entries += [(row,(row+1)%3,row) for row in range(1,8)]
    entries += [(0,2,0)]
    rows=np.asarray([row for row,_,_ in entries],dtype=np.int32)
    cols=np.asarray([col for _,col,_ in entries],dtype=np.int32)
    data=np.asarray([value for _,_,value in entries],dtype=np.int64)
    # Exact Python integer sums from the original triplets, followed by one
    # float conversion per coordinate. No SciPy reduction builds the oracle.
    totals={}
    for row,col,value in entries:
        totals[row,col]=totals.get((row,col),0)+int(value)
    dense=np.zeros((8,3),dtype=np.float64)
    for index,total in totals.items():
        dense[index]=float(total)
    cls=getattr(sparse,layout)
    if layout.startswith('coo'):
        raw=cls((data,(rows,cols)),shape=dense.shape)
    else:
        major,minor,count=(rows,cols,8) if layout.startswith('csr') else (cols,rows,3)
        order=np.argsort(major,kind='stable')
        indptr=np.asarray(np.r_[0,np.cumsum(np.bincount(major,minlength=count))],dtype=np.int32)
        raw=cls((data[order],minor[order],indptr),shape=dense.shape)
    raw.has_canonical_format=True
    if raw.format != 'coo':
        raw.has_sorted_indices=True
    for array in ((raw.data,raw.row,raw.col) if raw.format=='coo'
                  else (raw.data,raw.indices,raw.indptr)):
        array.flags.writeable=False
    assert dense[0,0]==1.
    return raw,dense


def snapshot(raw):
    buffers=((raw.data,raw.row,raw.col) if raw.format=='coo'
             else (raw.data,raw.indices,raw.indptr))
    flags={key:value for key,value in raw.__dict__.items()
           if key in ('has_canonical_format','_has_canonical_format','_has_sorted_indices')}
    return (raw.shape,tuple((array.dtype.str,array.shape,array.strides,
                             array.flags.writeable,array.__array_interface__['data'][0],
                             array.tobytes(order='A')) for array in buffers),flags)


@pytest.mark.parametrize('layout',['csr_matrix','coo_array','csc_matrix'])
@pytest.mark.parametrize('three',[False,True])
def test_linear_fit_query_preparation_and_lowlevel(layout,three):
    raw,dense=raw_features(layout,three)
    before=snapshot(raw)
    first,second=GenDWD(**OPTIONS).fit(raw,Y),GenDWD(**OPTIONS).fit(dense,Y)
    assert_allclose(first.coef_,second.coef_,rtol=1e-11,atol=1e-13)
    assert_allclose(first.intercept_,second.intercept_,rtol=1e-11,atol=1e-13)
    assert_allclose(first.decision_function(raw),second.decision_function(dense),rtol=1e-11,atol=1e-13)
    assert_array_equal(_augmented_gram(raw),_augmented_gram(dense))
    assert_array_equal(GenDWD(**OPTIONS).cv_init(raw)._P0_X.toarray(),dense)
    for a,b in zip(solve_gen_dwd(raw,Y,**OPTIONS)[:3],solve_gen_dwd(dense,Y,**OPTIONS)[:3]):
        assert_allclose(a,b,rtol=1e-11,atol=1e-13)
    assert snapshot(raw)==before


@pytest.mark.parametrize('layout',['csr_array','coo_matrix'])
@pytest.mark.parametrize('kernel',KERNELS)
def test_named_kernel_fit_query_validation_and_cv_init(layout,kernel):
    raw,dense=raw_features(layout)
    before=snapshot(raw)
    opts=dict(OPTIONS,kernel=kernel,kernel_kws=KERNELS[kernel])
    first,second=KernGDWD(**opts).fit(raw,Y),KernGDWD(**opts).fit(dense,Y)
    expected=literal_kernel(dense,dense,kernel,KERNELS[kernel])
    assert_array_equal(first._Xfit.toarray(),dense)
    assert_allclose(first._compute_kernel(first._Xfit),expected,rtol=2e-13,atol=2e-13)
    assert_allclose(first.decision_function(dense),second.decision_function(dense),rtol=2e-8,atol=2e-10)
    for batch in (None,3):
        assert_allclose(second.decision_function(raw,prediction_batch_size=batch),second.decision_function(dense),rtol=2e-12,atol=2e-12)
    gram,labels=second._prepare_validation((raw,Y))
    assert_allclose(gram,expected.T,rtol=2e-13,atol=2e-13)
    assert_array_equal(labels,Y)
    assert_array_equal(KernGDWD(**opts).cv_init(raw)._Xfit.toarray(),dense)
    assert snapshot(raw)==before


@pytest.mark.parametrize('factory,estimator',[(GenDWDCV,GenDWD),(KernGDWDCV,KernGDWD)])
def test_cv_boundaries(factory,estimator):
    raw,dense=raw_features('coo_array')
    before=snapshot(raw)
    opts=dict(lambd_vals=[.1],q_vals=[1.],cv=2,max_iter=2,stopping='fixed',initialization='zero',random_state=7)
    first,second=factory(**opts).fit(raw,Y),factory(**opts).fit(dense,Y)
    assert_allclose(first.decision_function(raw),second.decision_function(dense),rtol=2e-8,atol=2e-10)
    first=run_cv(estimator(**OPTIONS),raw,Y,{'lambd':[.1]},cv=2)
    second=run_cv(estimator(**OPTIONS),dense,Y,{'lambd':[.1]},cv=2)
    assert first[0]==second[0]
    assert_allclose(first[2].decision_function(raw),second[2].decision_function(dense),rtol=2e-8,atol=2e-10)
    assert snapshot(raw)==before


def test_callable_input_normalized_but_kernel_return_contract_separate():
    raw,dense=raw_features('coo_matrix')
    before=snapshot(raw)
    seen=[]
    def kernel(left,right):
        left=left.toarray() if sparse.issparse(left) else np.asarray(left)
        right=right.toarray() if sparse.issparse(right) else np.asarray(right)
        seen.append((left.copy(),right.copy()))
        return left @ right.T
    model=KernGDWD(kernel=kernel,**OPTIONS).fit(raw,Y)
    model.decision_function(raw)
    assert len(seen)>=2
    for left,right in seen:
        assert_array_equal(left,dense)
        assert_array_equal(right,dense)
    assert snapshot(raw)==before


def test_precomputed_dense_training_sparse_query_boundary():
    # Sparse precomputed training remains unsupported. A fitted dense kernel
    # still accepts sparse queries through the integer promotion boundary.
    dense=np.eye(8)
    rows=np.r_[0,0,np.arange(1,8)].astype(np.int32)
    cols=rows.copy()
    values=np.asarray([2**53+1,-2**53]+[1]*7,dtype=np.int64)
    raw=sparse.coo_array((values,(rows,cols)),shape=(8,8))
    for array in (raw.data,raw.row,raw.col):array.flags.writeable=False
    before=snapshot(raw)
    for training in (raw,sparse.csr_array(dense)):
        with pytest.raises(TypeError,match='dense data is required'):
            KernGDWD(kernel='precomputed',**OPTIONS).fit(training,Y)
    model=KernGDWD(kernel='precomputed',**OPTIONS).fit(dense,Y)
    assert_allclose(model.decision_function(raw),model.decision_function(dense),rtol=2e-8,atol=2e-10)
    assert_allclose(model._prepare_validation((raw,Y))[0],dense.T,rtol=0,atol=0)
    assert snapshot(raw)==before


def test_conic_inputs_and_linear_prediction():
    from importlib.util import find_spec
    if find_spec('cvxpy') is None:
        pytest.skip('This optional conic regression requires CVXPY.')
    import cvxpy as cp
    from dwd.socp_dwd import DWD,auto_dwd_C,solve_dwd_socp
    from dwd.svm import SVM,solve_svm
    if 'CLARABEL' not in cp.installed_solvers():
        pytest.skip('This optional conic regression requires CLARABEL.')
    raw,dense=raw_features('csr_matrix')
    before=snapshot(raw)
    solver={'solver':'CLARABEL'}
    assert auto_dwd_C(raw,Y)==auto_dwd_C(dense,Y)
    for factory in (DWD,SVM):
        first,second=factory(C=.5,solver_kws=solver).fit(raw,Y),factory(C=.5,solver_kws=solver).fit(dense,Y)
        assert_allclose(first.decision_function(raw),second.decision_function(dense),rtol=2e-5,atol=2e-6)
    for fun in (solve_dwd_socp,solve_svm):
        first,second=fun(raw,Y,C=.5,solver_kws=solver),fun(dense,Y,C=.5,solver_kws=solver)
        assert_allclose(dense @ first[0].reshape(-1)+first[1],dense @ second[0].reshape(-1)+second[1],rtol=2e-5,atol=2e-6)
    assert snapshot(raw)==before
