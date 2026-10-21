"""Reject inconsistent spectral representations before exposing checkpoints."""
import unittest
from fractions import Fraction
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
        # Preserve the historical eigen-coordinate path independently of restart.
        self.options=dict(lambd=1e-12,q=1.,alpha_init=np.zeros(80),max_iter=20,
                          implementation='optimized',acceleration=None)

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
                    options=dict(lambd=.1,q=1.,backend=backend,max_iter=2,
                                 stopping='fixed',acceleration=None)
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

    def exact_original_state(self, K, y, alpha, offset, lambd):
        coefficients = [Fraction(float(value)) for value in alpha]
        scores = [sum((Fraction(float(k)) * a for k, a in zip(row, coefficients)),
                      Fraction(0)) for row in K]
        margins = [Fraction(float(label)) * (score + Fraction(float(offset)))
                   for label, score in zip(y, scores)]
        losses = [1-margin if margin <= Fraction(1, 2) else 1/(4*margin)
                  for margin in margins]
        norm = sum((a*score for a, score in zip(coefficients, scores)), Fraction(0))
        objective = sum(losses, Fraction(0))/len(y) + Fraction(float(lambd))*norm
        return scores, objective

    def assert_exact_original_state(self, K, y, alpha, offset, lambd, reported_objective,
                                    reported_scores=None):
        scores, objective = self.exact_original_state(K, y, alpha, offset, lambd)
        self.assertLessEqual(abs(objective-Fraction(float(reported_objective))),
                             Fraction(5e-8)*max(Fraction(1), abs(objective)))
        if reported_scores is not None:
            discrepancy = max(abs(score-Fraction(float(reported)))
                              for score, reported in zip(scores, reported_scores))
            self.assertLessEqual(discrepancy,
                                 Fraction(5e-7)*max(Fraction(1), *(abs(s) for s in scores)))

    def test_restarted_spectral_observations_match_exact_original_kernel(self):
        # Restart uses original-coordinate residual/refinement. On this case
        # naive BLAS K@alpha is less accurate than its accepted score vector.
        for acceleration in ('auto', 'restart'):
            for mode in ('callback', 'validation', 'unobserved'):
                with self.subTest(acceleration=acceleration, mode=mode):
                    seen = []
                    options = dict(self.options, acceleration=acceleration,
                                   backend='spectral', stopping='fixed')
                    if mode == 'callback':
                        def callback(state):
                            seen.append(state)
                            self.assert_exact_original_state(self.K, self.y, state['alpha'],
                                state['offset'], options['lambd'], state['objective'],
                                state['training_scores'])
                            assert_array_equal(state['decision_values'],
                                               state['training_scores'] + state['offset'])
                            self.assertFalse(state['alpha'].flags.writeable)
                            self.assertFalse(state['training_scores'].flags.writeable)
                        options['callback'] = callback
                    elif mode == 'validation':
                        options.update(stopping='validation', validation=(self.K[::3], self.y[::3]),
                                       patience=2, check_interval=3)
                    try:
                        result = solve_kernel(self.K, self.y, **options)
                    except FloatingPointError:
                        # A refused fit may expose only independently checked states.
                        pass
                    else:
                        self.assertEqual(result['diagnostics']['acceleration'], 'restart')
                        for value in (result['final_objective'],
                                      result['objective_history'][result['returned_iteration']]):
                            self.assert_exact_original_state(self.K, self.y, result['alpha'],
                                                             result['offset'], options['lambd'], value)
                    if mode == 'callback':
                        self.assertTrue(seen)

    def test_restarted_spectral_fault_is_recovered_or_refused_before_observation(self):
        K = np.eye(8)
        y = np.tile([-1., 1.], 4)
        for acceleration in ('auto', 'restart'):
            for mode in ('callback', 'validation', 'unobserved'):
                with self.subTest(acceleration=acceleration, mode=mode):
                    seen = []
                    options = dict(lambd=.1, q=1., backend='spectral', max_iter=2,
                                   stopping='fixed', acceleration=acceleration)
                    if mode == 'callback':
                        def callback(state):
                            seen.append(state['iteration'])
                            self.assert_exact_original_state(K, y, state['alpha'], state['offset'],
                                .1, state['objective'], state['training_scores'])
                        options['callback'] = callback
                    elif mode == 'validation':
                        options.update(stopping='validation', validation=(K, y))
                    with patch.object(solver, '_eigensystem',
                                      return_value=(np.eye(8), np.full(8, 2.), {})):
                        try:
                            result = solve_kernel(K, y, **options)
                        except FloatingPointError:
                            pass
                        else:
                            # The injected representation must be detected. A new
                            # checked original-system representation may recover it.
                            self.assertGreater(result['diagnostics']['linear_system_diagnostics']
                                               ['linear_recoveries'], 0)
                            for value in (result['final_objective'],
                                          result['objective_history'][result['returned_iteration']]):
                                self.assert_exact_original_state(K, y, result['alpha'],
                                                                 result['offset'], .1, value)
                    if mode == 'callback':
                        self.assertTrue(seen)
                        self.assertEqual(seen[0], 0)


if __name__=='__main__':unittest.main(verbosity=2)
