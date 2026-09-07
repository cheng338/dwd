"""Independent contracts for opt-in acceleration, without changing MM defaults."""
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import solve as dense_solve
from sklearn.base import clone, is_classifier
from sklearn.datasets import make_circles
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from dwd._accelerated_mm import RestartedMM
from dwd._kernel_solver import solve_kernel
from dwd.gen_kern_dwd import KernGDWD, KernGDWDCV


def independent_loss(margin, q):
    return np.array([1-u if u <= q/(q+1) else q**q/((q+1)**(q+1)*u**q)
                     for u in margin])


class AccelerationMathTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(72931)
        self.X = rng.normal(size=(16, 4))
        self.K = self.X @ self.X.T
        self.y = np.r_[np.ones(9), -np.ones(7)]
        self.alpha = rng.normal(size=16) / 15
        self.b = .23

    def test_first_two_steps_match_independent_bordered_mm_general_q_and_rank(self):
        for q in (.2, 1., 3.5):
            for K in (self.K, self.K+.3*np.eye(16), np.zeros((16,16))):
                n=len(K);penalty=.13;step=n*q/(q+1)**2
                block=np.block([[K+2*penalty*step*np.eye(n), np.ones((n,1))],
                                [np.ones((1,n)),np.zeros((1,1))]])
                alpha=self.alpha.copy();b=self.b;expected=[]
                for _ in range(2):
                    scores=K@alpha+b
                    slope=np.array([-1. if u<=q/(q+1) else -(q/((q+1)*u))**(q+1)
                                    for u in self.y*scores])
                    solution=dense_solve(block,np.r_[scores-step*self.y*slope/n,0.],assume_a='sym')
                    alpha,b=solution[:-1],solution[-1]
                    expected.append((K@alpha+b,float(np.mean(independent_loss(self.y*(K@alpha+b),q))
                                                          +penalty*alpha@K@alpha)))
                for backend in ('cholesky','spectral'):
                    states=[]
                    result=solve_kernel(K,self.y,penalty,q=q,backend=backend,
                        alpha_init=self.alpha,offset_init=self.b,acceleration='restart',
                        max_iter=2,stopping='fixed',callback=lambda state:states.append(state))
                    for state,(decision,value) in zip(states[1:],expected):
                        assert_allclose(state['decision_values'],decision,rtol=2e-11,atol=2e-11)
                        self.assertAlmostEqual(state['objective'],value,delta=2e-11)
                    self.assertEqual(result['diagnostics']['acceleration_restarts'],0)

    def test_default_none_keeps_plain_mm_exactly(self):
        for backend in ('cholesky','spectral'):
            for implementation in ('optimized','reference'):
                if implementation=='reference' and backend=='cholesky':continue
                kwargs=dict(backend=backend,implementation=implementation,max_iter=20,stopping='fixed',
                            alpha_init=self.alpha,offset_init=self.b)
                left=solve_kernel(self.K,self.y,.13,**kwargs)
                right=solve_kernel(self.K,self.y,.13,acceleration=None,**kwargs)
                assert_array_equal(left['alpha'],right['alpha'])
                assert_array_equal(left['objective_history'],right['objective_history'])
                self.assertEqual(left['offset'],right['offset'])

    def test_zero_iteration_and_initial_callback_preserve_exact_state(self):
        for backend in ('cholesky','spectral'):
            states=[]
            result=solve_kernel(self.K,self.y,.13,backend=backend,acceleration='restart',
                alpha_init=self.alpha,offset_init=self.b,max_iter=0,callback=lambda s:states.append(s))
            self.assertEqual([s['iteration'] for s in states],[0])
            assert_array_equal(result['alpha'],self.alpha)
            self.assertEqual(result['offset'],self.b)
            self.assertEqual(result['diagnostics']['acceleration_proposals'],0)
            self.assertFalse(states[0]['alpha'].flags.writeable)

    def test_zero_kernel_optimizes_only_the_free_intercept(self):
        K=np.zeros((8,8));y=np.r_[np.ones(6),-np.ones(2)]
        for q in (.2,1.,3.5):
            for backend in ('cholesky','spectral'):
                result=solve_kernel(K,y,.7,q=q,backend=backend,acceleration='restart',
                                    stopping='optimality',tol=1e-9,max_iter=2000)
                self.assertAlmostEqual(result['offset'],q/(q+1)*3**(1/(q+1)),delta=2e-7)
                self.assertEqual(result['rkhs_norm_squared'],0.)
                self.assertLess(result['dual_gap'],1e-8)

    def test_natural_restart_counts_only_accepted_updates_and_is_monotone(self):
        X,y=make_circles(n_samples=120,noise=.12,factor=.5,random_state=851)
        K=rbf_kernel(X,gamma=2.);y=np.where(y,1.,-1.);seen=[]
        result=solve_kernel(K,y,.001,acceleration='restart',max_iter=120,stopping='fixed',
                            callback=lambda state:seen.append(state['iteration']))
        info=result['diagnostics']
        self.assertGreater(info['acceleration_restarts'],0)
        self.assertEqual(seen,list(range(121)))
        self.assertEqual(info['acceleration_accepted_updates'],120)
        self.assertEqual(info['acceleration_proposals'],120+info['acceleration_restarts'])
        self.assertLessEqual(np.max(np.diff(result['objective_history'])),1e-12)
        self.assertLess(result['rkhs_gradient_norm'],1e-5)

    def test_spectral_and_supplied_modes_never_call_cholesky(self):
        K=rbf_kernel(self.X,gamma=.3);values,U=np.linalg.eigh(K)
        reference=solve_kernel(K,self.y,.02,backend='cholesky',acceleration='restart',
                               max_iter=30,stopping='fixed')
        for backend,supplied in [('spectral',None),('auto',(U,values))]:
            with patch('dwd._kernel_linear_system.cho_factor',side_effect=AssertionError('No Cholesky')), \
                 patch('dwd._kernel_linear_system.cho_solve',side_effect=AssertionError('No Cholesky')):
                result=solve_kernel(K,self.y,.02,backend=backend,K_eig=supplied,acceleration='restart',
                                    max_iter=30,stopping='fixed')
            self.assertEqual(result['backend'],'spectral')
            self.assertEqual(result['diagnostics']['linear_system_diagnostics']['factor_representation'],
                             'validated_eigenbasis')
            assert_allclose(K@result['alpha']+result['offset'],K@reference['alpha']+reference['offset'],
                            rtol=1e-9,atol=2e-10)
            assert_allclose(result['objective_history'],reference['objective_history'],rtol=1e-10,atol=1e-12)

    def test_supplied_positive_small_modes_preserved_and_bad_pairs_rejected(self):
        values=np.array([1e-14,1e-10,.01,.1,.7,1.]);K=np.diag(values);y=np.tile([-1.,1.],3)
        result=solve_kernel(K,y,.02,backend='spectral',K_eig=(np.eye(6),values),
                            acceleration='restart',max_iter=3,stopping='fixed')
        self.assertEqual(result['diagnostics']['positive_eigenvalues_retained'],6)
        self.assertEqual(result['diagnostics']['positive_eigenvalues_discarded'],0)
        broken=np.eye(6);broken[0,0]=1.1
        with self.assertRaises((ValueError,FloatingPointError)):
            solve_kernel(K,y,.02,K_eig=(broken,values),acceleration='restart')

    def test_objective_default_is_same_rule_on_the_accelerated_path(self):
        full=solve_kernel(self.K,self.y,.13,acceleration='restart',stopping='fixed',max_iter=100)
        hit=np.flatnonzero(abs(np.diff(full['objective_history']))<1e-5)
        expected=int(hit[0]+1) if len(hit) else 100
        default=solve_kernel(self.K,self.y,.13,acceleration='restart')
        self.assertEqual(default['n_iter'],expected)
        assert_allclose(default['objective_history'],full['objective_history'][:expected+1],rtol=0,atol=1e-14)

    def test_callback_exceptions_and_requests_are_not_retried(self):
        for stop in (0,3):
            seen=[]
            def callback(state):
                seen.append(state['iteration'])
                return state['iteration']==stop
            result=solve_kernel(self.K,self.y,.13,acceleration='restart',stopping='fixed',callback=callback)
            self.assertEqual(seen,list(range(stop+1)))
            self.assertEqual(result['termination_reason'],'callback_stop')
            self.assertEqual(result['n_iter'],stop)
            self.assertEqual(result['diagnostics']['acceleration_accepted_updates'],stop)
        seen=[]
        def explode(state):
            seen.append(state['iteration'])
            if state['iteration']==3:raise RuntimeError('user callback')
        with self.assertRaisesRegex(RuntimeError,'user callback'):
            solve_kernel(self.K,self.y,.13,acceleration='restart',stopping='fixed',callback=explode)
        self.assertEqual(seen,[0,1,2,3])
        seen=[]
        with patch.object(RestartedMM,'_proposal',side_effect=FloatingPointError('unresolved solve')):
            with self.assertRaisesRegex(FloatingPointError,'unresolved solve'):
                solve_kernel(self.K,self.y,.13,acceleration='restart',callback=lambda s:seen.append(s['iteration']))
        self.assertEqual(seen,[0])

    def test_validation_restores_coefficients_offset_scores_and_first_tie(self):
        validation=(np.zeros((4,16)),np.array([-1.,1.,-1.,1.]));seen=[]
        result=solve_kernel(self.K,self.y,.13,acceleration='restart',validation=validation,
            stopping='validation',check_interval=2,patience=2,max_iter=20,callback=lambda s:seen.append(s))
        first=solve_kernel(self.K,self.y,.13,acceleration='restart',stopping='fixed',max_iter=1)
        self.assertEqual(result['n_iter'],4)
        self.assertEqual(result['returned_iteration'],1)
        self.assertEqual([s['iteration'] for s in result['validation_history']],[1,2,4])
        self.assertEqual([s['iteration'] for s in seen],[0,1,2,3,4])
        assert_array_equal(result['alpha'],first['alpha'])
        self.assertEqual(result['offset'],first['offset'])
        self.assertAlmostEqual(result['final_objective'],first['final_objective'],places=13)

    def test_failed_momentum_trial_gets_one_plain_retry_before_any_callback(self):
        native=RestartedMM._proposal
        calls=[];seen=[]
        def fail_first_momentum(instance,total):
            calls.append(instance.weight)
            if len(calls)==3:
                self.assertGreater(instance.weight,0.)
                instance.info['acceleration_proposals']+=1
                raise FloatingPointError('Injected checked extrapolation failure')
            return native(instance,total)
        with patch.object(RestartedMM,'_proposal',fail_first_momentum):
            result=solve_kernel(self.K,self.y,.13,acceleration='restart',stopping='fixed',max_iter=3,
                                callback=lambda state:seen.append(state['iteration']))
        ordinary=solve_kernel(self.K,self.y,.13,stopping='fixed',max_iter=3)
        self.assertEqual(seen,[0,1,2,3]);self.assertEqual(len(calls),4)
        self.assertEqual(result['diagnostics']['acceleration_numerical_restarts'],1)
        self.assertEqual(result['diagnostics']['acceleration_restarts'],1)
        self.assertEqual(result['diagnostics']['acceleration_proposals'],4)
        assert_allclose(self.K@result['alpha']+result['offset'],
                        self.K@ordinary['alpha']+ordinary['offset'],rtol=2e-12,atol=2e-12)
        calls.clear();seen.clear()
        def persistent_failure(instance,total):
            calls.append(instance.weight)
            if len(calls)>=3:raise FloatingPointError('Persistent checked solve failure')
            return native(instance,total)
        with patch.object(RestartedMM,'_proposal',persistent_failure),self.assertRaisesRegex(FloatingPointError,'Persistent'):
            solve_kernel(self.K,self.y,.13,acceleration='restart',stopping='fixed',max_iter=10,
                         callback=lambda state:seen.append(state['iteration']))
        self.assertEqual(len(calls),4)
        self.assertEqual(seen,[0,1,2])

    def test_small_convex_epigraph_oracle_general_q(self):
        try:import cvxpy as cp
        except ImportError:self.skipTest('CVXPY optional extra is unavailable.')
        for q in (.5,1.,2.):
            penalty=.035;n=len(self.y);w=cp.Variable(4);b=cp.Variable()
            r=cp.Variable(n);xi=cp.Variable(n,nonneg=True);a=q**q/(q+1)**(q+1)
            problem=cp.Problem(cp.Minimize(cp.sum(a*cp.power(r,-q)+xi)/n+penalty*cp.sum_squares(w)),
                [r==cp.multiply(self.y,self.X@w+b)+xi])
            problem.solve(solver='CLARABEL',tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10)
            result=solve_kernel(self.K,self.y,penalty,q=q,acceleration='restart',
                                stopping='optimality',tol=1e-7,max_iter=3000)
            self.assertLess(abs(result['final_objective']-problem.value),2e-7)
            self.assertLess(result['rkhs_gradient_norm'],1e-7)


class AccelerationPublicApiTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(144);self.X=rng.normal(size=(30,4));self.y=np.tile([-7,9],15)

    def model(self,**kwargs):
        parameters=dict(kernel='rbf',kernel_kws={'gamma':.3},lambd=.05,max_iter=5,stopping='fixed',
                        acceleration='restart')
        parameters.update(kwargs);return KernGDWD(**parameters)

    def test_parameter_clone_gridsearch_and_internal_cv_pass_through(self):
        model=self.model();self.assertEqual(clone(model).get_params()['acceleration'],'restart')
        self.assertTrue(is_classifier(model))
        cv=StratifiedKFold(2,shuffle=True,random_state=9)
        grid=GridSearchCV(model,{'acceleration':[None,'restart'],'lambd':[.05,.08]},cv=cv,
                          n_jobs=1,error_score='raise').fit(self.X,self.y)
        self.assertEqual(len(grid.cv_results_['params']),4)
        internal=KernGDWDCV(lambd_vals=[.05,.08],q_vals=[1],kernel='rbf',
            kernel_kws_vals=[{'gamma':.3}],cv=cv,max_iter=5,stopping='fixed',acceleration='restart')
        self.assertEqual(clone(internal).get_params()['acceleration'],'restart')
        internal.fit(self.X,self.y)
        self.assertEqual(internal.best_estimator_.acceleration,'restart')
        self.assertEqual(internal.best_estimator_.diagnostics_['acceleration'],'restart')

    def test_acceleration_change_reuses_unchanged_preparation(self):
        model=self.model(acceleration=None,backend='spectral').cv_init(self.X)
        K=model._cv_K;eigen=model._K_eig
        model.set_params(acceleration='restart')
        self.assertTrue(model._cv_cache_matches(self.X))
        with patch.object(model,'_compute_kernel',side_effect=AssertionError('Cached K must be reused')):
            model.fit(self.X,self.y)
        self.assertIs(model._cv_K,K);self.assertIs(model._K_eig,eigen)
        self.assertEqual(model.backend_,'spectral')

    def test_incompatible_and_invalid_options_rejected(self):
        for bad in [False,True,1,'fista',[],{}]:
            with self.subTest(bad=bad),self.assertRaisesRegex(ValueError,'acceleration'):
                self.model(acceleration=bad).fit(self.X,self.y)
        for options in [dict(implementation='reference'),dict(solver_mode='legacy'),dict(backend='lbfgs')]:
            with self.subTest(options=options),self.assertRaisesRegex(ValueError,'Acceleration'):
                self.model(**options).fit(self.X,self.y)
        for options in [dict(implementation='reference'),dict(backend='lbfgs')]:
            with self.assertRaisesRegex(ValueError,'Acceleration'):
                solve_kernel(np.eye(4),np.tile([-1.,1.],2),.1,acceleration='restart',**options)


if __name__=='__main__':unittest.main()
