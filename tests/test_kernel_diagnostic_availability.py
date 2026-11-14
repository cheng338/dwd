"""Portable public regressions for unavailable dual diagnostics and earlier guards."""
import pickle
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from dwd import _kernel_solver as solver
from dwd.gen_kern_dwd import KernGDWD, KernGDWDCV


def two_by_two():
    t = 1. + 2.**-45
    return np.array([[1., t], [t, 1.]]), np.array([-1., 1.])


def unavailable(info):
    assert info['dual_objective_is_lower_bound'] is False
    assert info['fallback_reason'] in ('certified_negative_dual_quadratic', 'negative_constant_kernel')
    assert info['kernel_assumption'] == 'symmetric positive-semidefinite represented kernel (contradicted)'


@pytest.mark.parametrize('backend',['auto','cholesky','spectral','lbfgs'])
def test_public_fitted_state_pickle_and_failed_refit(backend):
    K,y=two_by_two();before=K.tobytes()
    penalty=.1 if backend=='lbfgs' else 1.
    model=KernGDWD(kernel='precomputed',backend=backend,lambd=penalty,max_iter=3,
                  stopping='fixed',initialization='zero',random_state=7).fit(K,y)
    check_is_fitted(model)
    assert model.dual_gap_ is None
    unavailable(model.diagnostics_['dual_certificate'])
    assert model.diagnostics_['dual_certificate']['dual_objective_lower_bound'] is None
    assert np.isfinite(model.objective_history_).all()
    restored=pickle.loads(pickle.dumps(model))
    assert restored.dual_gap_ is None
    assert restored.diagnostics_['dual_certificate']==model.diagnostics_['dual_certificate']
    np.testing.assert_array_equal(restored.predict(K),model.predict(K))
    np.testing.assert_array_equal(restored.decision_function(K),model.decision_function(K))
    assert clone(model).get_params()==model.get_params()
    assert K.tobytes()==before
    with pytest.raises(ValueError,match='positive semidefinite'):
        model.fit(-np.eye(2),y)
    with pytest.raises(NotFittedError):check_is_fitted(model)


def test_cv_scores_and_final_refit_do_not_treat_missing_gap_as_failure(monkeypatch):
    y=np.array([-1.,1.,-1.,1.]);K=np.ones((4,4))-2.**-47*np.outer(y,y)
    folds=[(np.array([0,1]),np.array([2,3])),(np.array([2,3]),np.array([0,1]))]
    observed=[];fit=KernGDWD.fit
    def track(self,*args,**kwargs):
        result=fit(self,*args,**kwargs)
        observed.append((self.dual_gap_,dict(self.diagnostics_['dual_certificate'])))
        return result
    with monkeypatch.context() as patch:
        patch.setattr(KernGDWD,'fit',track)
        cv=KernGDWDCV(kernel='precomputed',lambd_vals=[1.,2.],q_vals=[1.],cv=folds,
                     max_iter=3,stopping='fixed',initialization='zero').fit(K,y)
    assert len(observed)==5
    assert all(gap is None for gap,_ in observed)
    for _,info in observed:unavailable(info)
    assert np.isfinite(cv.best_score_)
    assert cv.best_estimator_.dual_gap_ is None
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(cv)).predict(K),cv.predict(K))


CASES=[pytest.param('original_two',backend,implementation,id='original_two-'+backend+'-'+implementation)
       for backend,implementation in [('auto','optimized'),('cholesky','optimized'),
           ('spectral','optimized'),('spectral','reference')]]
CASES += [pytest.param('original_four',backend,implementation,id='original_four-'+backend+'-'+implementation)
          for backend,implementation in [('auto','optimized'),('cholesky','optimized'),
              ('spectral','optimized'),('lbfgs','optimized'),('spectral','reference')]]



@pytest.mark.parametrize('case,backend,implementation',CASES)
def test_original_earlier_numerical_refusals_are_preserved(monkeypatch,case,backend,implementation):
    K,y=two_by_two()
    if case=='original_four':
        y=np.array([-1.,1.,-1.,1.]);K=np.ones((4,4))-2.**-45*np.outer(y,y)
    before=K.tobytes();calls=[];observed=[]
    options=dict(backend=backend,implementation=implementation,max_iter=3,stopping='fixed',
                 alpha_init=np.zeros(len(y)),offset_init=0.,acceleration=None)
    def forbidden(*args):
        calls.append(True)
        raise AssertionError('The earlier numerical guard must precede the final certificate')
    def callback(state):
        observed.append({key:value.copy() if isinstance(value,np.ndarray) else value
                         for key,value in state.items()})
    with monkeypatch.context() as patch:
        patch.setattr(solver,'kernel_dual_lower_bound',forbidden)
        with pytest.raises(FloatingPointError,match='Negative (original )?RKHS'):
            solver.solve_kernel(K,y,.1,callback=callback,**options)
    assert not calls
    assert K.tobytes()==before
