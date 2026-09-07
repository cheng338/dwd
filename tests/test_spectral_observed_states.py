"""Reject inconsistent spectral representations before exposing checkpoints."""
import unittest
from unittest.mock import patch
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.metrics.pairwise import rbf_kernel
from dwd._kernel_solver import solve_kernel
from dwd.gen_dwd import V
import dwd._kernel_solver as solver


def traceback_functions(exception):
    names=[];trace=exception.__traceback__
    while trace is not None:
        names.append(trace.tb_frame.f_code.co_name);trace=trace.tb_next
    return names


class ObservedSpectralStateTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(20260908)
        rng.normal(size=(120,5))
        X=np.repeat(rng.normal(size=(4,3)),20,axis=0)
        self.K=rbf_kernel(X,gamma=.3)
        self.y=np.tile([-1.,1.],40)
        self.options=dict(lambd=1e-12,q=1.,alpha_init=np.zeros(80),max_iter=20,implementation='optimized')

    def check_returned_objective(self,result):
        scores=self.K@result['alpha']
        objective=float(np.mean(V(self.y*(scores+result['offset']),q=1.))+
                        self.options['lambd']*(result['alpha']@scores))
        self.assertLessEqual(abs(objective-result['final_objective']),5e-8*max(1.,abs(objective)))
        self.assertLessEqual(abs(objective-result['objective_history'][result['returned_iteration']]),
                             5e-8*max(1.,abs(objective)))

    def test_ill_conditioned_callbacks_are_consistent_or_fit_rejects(self):
        # This exact bounded case previously exposed 1.27e-4 / 8.24e-5 score
        # discrepancies for spectral MM / L-BFGS, then failed only on return.
        # Decimal60 confirmed a genuine representation mismatch in both cases.
        for backend in ('spectral','lbfgs'):
            with self.subTest(backend=backend):
                seen=[]
                def callback(state):
                    true=self.K@state['alpha']
                    seen.append((true.copy(),state['training_scores'].copy()))
                try:
                    result=solve_kernel(self.K,self.y,backend=backend,stopping='fixed',callback=callback,**self.options)
                except FloatingPointError:
                    pass
                else:self.check_returned_objective(result)
                self.assertTrue(seen)
                for true,reported in seen:
                    self.assertLessEqual(np.max(np.abs(true-reported)),5e-7*max(1.,np.max(np.abs(true))))

    def test_ill_conditioned_validation_returns_consistent_model_or_rejects(self):
        for backend in ('spectral','lbfgs'):
            with self.subTest(backend=backend):
                try:
                    result=solve_kernel(self.K,self.y,backend=backend,stopping='validation',
                        validation=(self.K[::3],self.y[::3]),patience=2,check_interval=3,**self.options)
                except FloatingPointError:
                    pass
                else:self.check_returned_objective(result)

    def test_unobserved_path_keeps_its_final_representation_guard(self):
        for backend in ('spectral','lbfgs'):
            with self.subTest(backend=backend):
                try:
                    result=solve_kernel(self.K,self.y,backend=backend,stopping='fixed',**self.options)
                except FloatingPointError:
                    pass
                else:self.check_returned_objective(result)

    def test_injected_representation_fault_must_fail_before_observation(self):
        # Inject after eigensystem validation to test the observation boundary
        # independently of any provider's treatment of tiny eigenvalues.
        K=np.eye(8);y=np.tile([-1.,1.],4)
        for backend in ('spectral','lbfgs'):
            for mode in ('callback','validation','unobserved'):
                with self.subTest(backend=backend,mode=mode):
                    seen=[]
                    options=dict(lambd=.1,q=1.,backend=backend,max_iter=2,stopping='fixed')
                    if mode=='callback':options['callback']=lambda state: seen.append(state['iteration'])
                    elif mode=='validation':options.update(stopping='validation',validation=(K,y))
                    with patch.object(solver,'_eigensystem',return_value=(np.eye(8),np.full(8,2.),{})):
                        try:solve_kernel(K,y,**options)
                        except FloatingPointError as exc:
                            if mode=='unobserved':
                                self.assertIn('coefficient conversion lost score accuracy',str(exc))
                                self.assertNotIn('record',traceback_functions(exc))
                            else:
                                self.assertIn('Observed kernel checkpoint',str(exc))
                                self.assertIn('record',traceback_functions(exc))
                        else:self.fail('A deliberately inconsistent representation must be rejected.')
                    if mode=='callback':self.assertEqual(seen,[0])

    def test_valid_observation_and_restoration_preserve_coordinates(self):
        rng=np.random.default_rng(319)
        X=rng.normal(size=(40,4));K=rbf_kernel(X,gamma=.7);y=np.tile([-1.,1.],20)
        for backend in ('spectral','lbfgs'):
            with self.subTest(backend=backend):
                seen={}
                def callback(state):
                    seen[state['iteration']]=(state['alpha'].copy(),state['offset'])
                    assert_allclose(state['training_scores'],K@state['alpha'],atol=1e-12,rtol=1e-12)
                    assert_allclose(state['decision_values'],K@state['alpha']+state['offset'],atol=1e-12,rtol=1e-12)
                result=solve_kernel(K,y,lambd=.03,q=1.7,backend=backend,implementation='optimized',max_iter=20,
                    stopping='validation',validation=(K[::3],y[::3]),patience=2,check_interval=3,callback=callback)
                alpha,offset=seen[result['returned_iteration']]
                assert_array_equal(result['alpha'],alpha)
                self.assertEqual(result['offset'],offset)
                self.assertGreater(result['diagnostics']['observed_checkpoint_checks'],0)


if __name__=='__main__':unittest.main(verbosity=2)
