"""Independent literal-objective diagnostic contract controls."""
import unittest
import json
import pickle
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
import numpy as np
from dwd.literal_diagnostics import literal_objective_diagnostics as diagnostic, DiagnosticRefusal
from dwd.gen_kern_dwd import KernGDWD
observations=[]
def independent(K,y,a,b,lam,q):
 with localcontext() as c:
  c.prec=180
  to=lambda v:D.from_float(float(v))
  A=[[to(x) for x in row] for row in K];aa=list(map(to,a));bb=to(b);ll=to(lam);qq=to(q);yy=list(map(to,y));n=len(y);t=qq/(qq+1)
  margins=[yy[i]*(sum((A[i][j]*aa[j] for j in range(n)),D(0))+bb) for i in range(n)]
  losses=[1-m if m<=t else (t/m)**qq/(qq+1) for m in margins]
  z=[yy[i]*(-D(1) if m<=t else -(t/m)**(qq+1))/n for i,m in enumerate(margins)]
  gradient=[sum((A[j][i]*z[j]+ll*(A[i][j]+A[j][i])*aa[j] for j in range(n)),D(0)) for i in range(n)]
  primal=sum(losses,D(0))/n+ll*sum((aa[i]*A[i][j]*aa[j] for i in range(n) for j in range(n)),D(0))
  return primal,gradient,sum(z,D(0))

class Test(unittest.TestCase):
 def check_oracle(self,K,y,a,b,lam,q):
  result=diagnostic(K,y,a,b,lam,q);p,g,bias=independent(K,y,a,b,lam,q)
  def inside(value,pair):self.assertLessEqual(D.from_float(pair[0]),value);self.assertLessEqual(value,D.from_float(pair[1]))
  inside(p,result['objective_interval']);inside(bias,result['bias_gradient_interval'])
  for v,pair in zip(g,result['coefficient_gradient_intervals']):inside(v,pair)
  self.assertLessEqual(D.from_float(result['dual_lower_bound']),p)
  self.assertGreaterEqual(result['objective_gap_upper'],0.)
  self.assertEqual(json.loads(json.dumps(result)),result)
  return result

 def test_general_q_all_branches_and_asymmetry(self):
  rng=np.random.default_rng(902191)
  for q in (.0625,.1,.3,.5,.75,1.,1.5,2.,3.,7.25,16.,31.):
   for n in (2,4,8):
    B=rng.integers(-3,4,size=(n,n))/8.;K=B@B.T+np.eye(n)
    A=rng.integers(-2,3,size=(n,n))*2.**-45;K=K+A-A.T
    y=np.resize([-1.,1.],n);a=rng.normal(size=n)*2;b=.125
    r=self.check_oracle(K,y,a,b,.03125,q)
    observations.append({'q':q,'n':n,'branches':r['margin_branches'],'dual_kind':r['dual_kind']})

 def test_join_and_neighbor_margins(self):
  for q in (.5,1.,2.,3.,7.25):
   t=q/(q+1.);y=np.array([-1.,1.]);K=np.eye(2)
   for m in (np.nextafter(t,-np.inf),t,np.nextafter(t,np.inf)):
    self.check_oracle(K,y,y*m,0.,.1,q)

 def test_exact_linear_branch_gradient(self):
  K=np.array([[1.,.5+2**-50],[.5-2**-50,2.]]);y=np.array([-1.,1.]);a=np.array([.125,-.125]);b=.0625;lam=.125
  for q in (.1,1.,10.):
   r=self.check_oracle(K,y,a,b,lam,q)
   self.assertEqual(r['margin_branches'],['linear','linear'])
   for pair in r['coefficient_gradient_intervals']:self.assertEqual(pair[0],pair[1])

 def test_tiny_positive_pivot_retained(self):
  K=np.diag([1.,2.**-500]);y=np.array([-1.,1.]);a=np.zeros(2)
  r=diagnostic(K,y,a,0.,.25,1.)
  self.assertEqual(r['rank'],2)
  self.assertEqual(r['coefficient_gradient_intervals'],[[.5,.5],[-2.**-501,-2.**-501]])

 def test_singular_range_and_counterexample(self):
  y=np.array([-1.,1.,-1.,1.]);A=np.zeros((4,4));A[:3,:3]=[[0,1,-1],[-1,0,1],[1,-1,0]];e=2.**-50;K=np.ones((4,4))+e*A;a=-(A@y)/e
  r=self.check_oracle(K,y,a,0.,1.,1.)
  self.assertEqual(r['dual_kind'],'zero_slopes_range_failure');self.assertEqual(r['dual_lower_bound'],0.)
  self.assertEqual(r['objective_interval'],[21/64,21/64])
  r=self.check_oracle(np.ones((4,4)),y,np.zeros(4),0.,1.,.3)
  self.assertEqual(r['rank'],1)

 def test_input_validation_and_refusals(self):
  kwargs=dict(K=np.eye(2),y=[-1,1],alpha=[0.,0.],offset=0.,lambd=.1)
  updates=[dict(K=np.eye(2,dtype=complex)),dict(K=[['1','0'],['0','1']]),dict(y=[0,1]),dict(y=[1,1]),dict(alpha=[np.nan,0]),dict(offset=np.inf),dict(q=0),dict(q=-1),dict(q=np.nan),dict(lambd=0),dict(lambd=np.bool_(True)),dict(q=True),dict(q=np.array(True)),dict(q='1'),dict(max_n=True),dict(max_bits=9000),dict(precision=1),dict(raw_slopes=[1]),dict(raw_slopes=[np.inf,0]),dict(K=np.eye(3)),dict(alpha=[[0],[0]])]
  for update in updates:
   with self.subTest(update=str(update)),self.assertRaises((ValueError,DiagnosticRefusal)):diagnostic(**dict(kwargs,**update))
  for K in (np.diag([1.,-1.]),np.array([[0.,1.],[1.,0.]])):
   with self.assertRaisesRegex(DiagnosticRefusal,'indefinite'):diagnostic(**dict(kwargs,K=K))
  with self.assertRaisesRegex(DiagnosticRefusal,'dimension'):diagnostic(np.eye(17),np.resize([-1,1],17),np.zeros(17),0.,.1)
  with self.assertRaisesRegex(DiagnosticRefusal,'operation'):diagnostic(**dict(kwargs,max_operations=1))
  with self.assertRaisesRegex(DiagnosticRefusal,'bit budget'):diagnostic(**dict(kwargs,K=np.diag([1.,2.**-500]),max_bits=64))

 def test_extreme_q_is_certified_or_explicitly_refused(self):
  for q in (np.nextafter(0.,1.),1e-100,1e100,np.finfo(float).max):
   try:r=self.check_oracle(np.eye(2),np.array([-1.,1.]),np.array([-2.,2.]),0.,.1,q)
   except DiagnosticRefusal as e:observations.append({'extreme_q':q,'status':'bounded_refusal','reason':str(e)})
   else:observations.append({'extreme_q':q,'status':'certified','objective_interval':r['objective_interval']})

 def test_invalid_dimensions_refuse_before_array_conversion(self):
  class Huge:
   shape=(1000000,1000000)
   def __len__(self):return 1000000
   def __array__(self,*a,**kw):raise AssertionError('Oversized input converted')
  with self.assertRaises(ValueError):diagnostic(Huge(),[-1,1],[0,0],0.,.1)
  with self.assertRaises(DiagnosticRefusal):diagnostic(np.eye(2),Huge(),[0,0],0.,.1)
  with self.assertRaises(ValueError):diagnostic(np.eye(2),[-1,1],Huge(),0.,.1)
  with self.assertRaises(ValueError):diagnostic([[Huge(),0],[0,1]],[-1,1],[0,0],0.,.1)
  with self.assertRaises(ValueError):diagnostic(np.eye(2),[Huge(),1],[0,0],0.,.1)

 def test_fitted_model_unchanged_pickle_and_input_ownership(self):
  X=np.arange(12.).reshape(6,2)/12.;y=np.resize([-1,1],6)
  m=KernGDWD(kernel='rbf',lambd=.1,kernel_kws={'gamma':.5},stopping='fixed',max_iter=5).fit(X,y)
  K=m._compute_training_kernel(m._Xfit);state=pickle.dumps(m);before=K.copy();a=m.dual_coef_.reshape(-1)
  r=diagnostic(K,y,a,float(m.intercept_[0]),m.lambd,m.q)
  self.assertEqual(pickle.dumps(m),state);np.testing.assert_array_equal(K,before)
  np.testing.assert_array_equal(pickle.loads(state).predict(X),m.predict(X))
  self.assertFalse(r['changes_fit_or_acceptance'])


