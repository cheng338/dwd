"""Independent tests of diagnostic availability, not PSD acceptance relaxation."""
from copy import deepcopy
import numpy as np
import pytest
from dwd import _dual_certificate as certificate
from dwd import _kernel_solver as solver
from fractions import Fraction


def two_by_two():
    t = 1. + 2.**-45
    return np.array([[1., t], [t, 1.]]), np.array([-1., 1.])


def unavailable(info):
    assert info['dual_objective_is_lower_bound'] is False
    assert info['fallback_reason'] in ('certified_negative_dual_quadratic', 'negative_constant_kernel')
    assert info['kernel_assumption'] == 'symmetric positive-semidefinite represented kernel (contradicted)'


def exact_quadratic(K, v):
    return sum((Fraction(float(v[i])) * Fraction(float(K[i,j])) * Fraction(float(v[j]))
                for i in range(len(v)) for j in range(len(v))), Fraction(0))



def cases():
    K, y = two_by_two()
    yield pytest.param(K, y, id='original-IR15-two-by-two')
    # New explicit analogue; the review document did not specify its E matrix.
    y = np.array([-1., 1., -1., 1.])
    for exponent in (-47, -46, -45):
        yield pytest.param(np.ones((4,4)) - 2.**exponent*np.outer(y,y), y,
                           id='new-J4-minus-dyy-'+str(exponent))


@pytest.mark.parametrize('K,y', list(cases()))
def test_exact_negative_direction_has_no_dual_endpoint(K, y):
    assert exact_quadratic(K, y) < 0
    before = K.tobytes()
    lower, info = certificate.kernel_dual_lower_bound(K, y, np.ones(len(y)), 1e-20, 1.)
    assert lower is None
    unavailable(info)
    assert info['fallback_reason'] == 'certified_negative_dual_quadratic'
    assert K.tobytes() == before


def test_negative_constant_never_claims_zero_lower_bound():
    lower, info = certificate.kernel_dual_lower_bound(-np.ones((2,2)), np.array([-1.,1.]),
                                                     np.ones(2), .1, 1.)
    assert lower is None
    unavailable(info)
    assert info['fallback_reason'] == 'negative_constant_kernel'


@pytest.mark.parametrize('error',[ValueError('unrelated'),MemoryError('resource'),
                                  RuntimeError('unexpected'),KeyboardInterrupt('stop')])
def test_unexpected_certificate_errors_keep_identity(monkeypatch,error):
    def fail(*args):raise error
    monkeypatch.setattr(certificate,'_quadratic_upper',fail)
    with pytest.raises(type(error)) as caught:
        certificate.kernel_dual_lower_bound(np.eye(2),np.array([-1.,1.]),np.ones(2),.1,1.)
    assert caught.value is error


@pytest.mark.parametrize('reason',[None,'',[], 'unspecified_failure'])
def test_contradiction_type_rejects_unknown_reason(reason):
    with pytest.raises(ValueError):certificate._PSDContradiction(reason)


def test_mutated_contradiction_is_not_swallowed(monkeypatch):
    error=certificate._PSDContradiction('certified_negative_dual_quadratic')
    error.reason='unspecified_failure'
    def fail(*args):raise error
    monkeypatch.setattr(certificate,'_quadratic_upper',fail)
    with pytest.raises(ValueError) as caught:
        certificate.kernel_dual_lower_bound(np.eye(2),np.array([-1.,1.]),np.ones(2),.1,1.)
    assert caught.value is error


def test_bad_slopes_are_still_input_errors():
    with pytest.raises(ValueError):
        certificate.kernel_dual_lower_bound(np.eye(2),np.array([-1.,1.]),np.array([np.nan,1.]),.1,1.)


def valid_unavailable_info():
    K,y=two_by_two()
    endpoint,info=certificate.kernel_dual_lower_bound(K,y,np.ones(2),.1,1.)
    assert endpoint is None
    return info


@pytest.mark.parametrize('fault',['missing_flag','true_flag','integer_flag','unknown_reason',
                                  'wrong_assumption','finite_with_false'])
def test_malformed_unavailable_pair_remains_an_error(monkeypatch,fault):
    info=deepcopy(valid_unavailable_info());endpoint=None
    if fault=='missing_flag':del info['dual_objective_is_lower_bound']
    elif fault=='true_flag':info['dual_objective_is_lower_bound']=True
    elif fault=='integer_flag':info['dual_objective_is_lower_bound']=0
    elif fault=='unknown_reason':info['fallback_reason']='unknown'
    elif fault=='wrong_assumption':info['kernel_assumption']='unverified'
    else:endpoint=.1
    monkeypatch.setattr(solver,'kernel_dual_lower_bound',lambda *args:(endpoint,info))
    with pytest.raises(ValueError):
        solver.optimality_diagnostics(np.eye(2),np.array([-1.,1.]),np.zeros(2),0.,.1,1.)


@pytest.mark.parametrize('endpoint,match',[(np.inf,'Nonfinite'),(2.,'Inconsistent primal-dual')])
def test_original_finite_and_gap_checks_remain(monkeypatch,endpoint,match):
    monkeypatch.setattr(solver,'kernel_dual_lower_bound',lambda *args:(endpoint,{}))
    with pytest.raises(FloatingPointError,match=match):
        solver.optimality_diagnostics(np.eye(2),np.array([-1.,1.]),np.zeros(2),0.,.1,1.)


@pytest.mark.parametrize('backend',['auto','cholesky','spectral','lbfgs'])
def test_material_indefiniteness_still_fails_before_certificate(monkeypatch,backend):
    def forbidden(*args):raise AssertionError('Diagnostic must not authorize the input')
    monkeypatch.setattr(solver,'kernel_dual_lower_bound',forbidden)
    with pytest.raises(ValueError,match='positive semidefinite'):
        solver.solve_kernel(-np.eye(2),np.array([-1.,1.]),.1,backend=backend,max_iter=1)


@pytest.mark.parametrize('alpha,match',[(np.ones(2),'RKHS norm squared'),
                                      (np.zeros(2),'RKHS gradient norm squared')])
def test_other_negative_quadratic_guards_remain(monkeypatch,alpha,match):
    def forbidden(*args):raise AssertionError('Earlier numerical guards must run')
    monkeypatch.setattr(solver,'kernel_dual_lower_bound',forbidden)
    with pytest.raises(FloatingPointError,match=match):
        solver.optimality_diagnostics(-np.eye(2),np.array([-1.,1.]),alpha,0.,.1,1.)


def test_existing_tiny_unit_diagonal_contradiction_remains_visible():
    K=np.ones((3,3));K[1,2]=K[2,1]=1.-2.**-52
    y=np.array([-1.,1.,1.]);raw=np.array([1.,.5,.5])
    signed=y*raw
    assert exact_quadratic(K,signed)<0
    before=K.tobytes()
    endpoint,info=certificate.kernel_dual_lower_bound(K,y,raw,1e-30,1.)
    assert endpoint is None
    unavailable(info)
    assert info['fallback_reason']=='certified_negative_dual_quadratic'
    assert K.tobytes()==before
