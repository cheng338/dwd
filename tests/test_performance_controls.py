"""Portable parity controls for conservative screening and objective reuse.

No environment, search path, official data, or external checkout is needed.
The frozen oracle is test-only and its hashes are verified before loading.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import unittest
from unittest.mock import patch
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import rbf_kernel
import dwd

NEW = Path(dwd.__file__).resolve().parent
OLD = Path(__file__).resolve().parent / 'fixtures/performance_oracle'
for _name, _expected in json.loads((OLD/'provenance.json').read_text())['files'].items():
    if hashlib.sha256((OLD/_name).read_bytes()).hexdigest() != _expected:
        raise RuntimeError('Frozen performance oracle changed: ' + _name)

def load(name,path):
    spec=importlib.util.spec_from_file_location('dwd.'+name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module

old_scores=load('_perf_old_scores',OLD/'_kernel_scores.py')
new_scores=load('_perf_new_scores',NEW/'_kernel_scores.py')
variants={}
for name,source,scores in (('baseline',OLD,old_scores),('objective_only',NEW,old_scores),
                           ('bound_only',OLD,new_scores),('both',NEW,new_scores)):
    solver=load('_perf_'+name,source/'_kernel_solver.py')
    solver.adaptive_kernel_matvec=scores.adaptive_kernel_matvec
    variants[name]=solver

REPORT=dict(no_MNIST_fits=True,no_official_test_reads=True,
            variants=list(variants),score_cases=[],solver_cases=[])

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class Controls(unittest.TestCase):
    def paired_scores(self,K,alpha,name):
        records=[]
        for module in (old_scores,new_scores):
            expanded=[];fast=[];original=module._expanded_row
            def row(*args):expanded.append(args[-1]);return original(*args)
            if module is new_scores:
                tier=module._dense_first_tier
                def capture(*args):
                    value=tier(*args);fast.append(bool(np.all(value)));return value
                with patch.object(module,'_dense_first_tier',side_effect=capture),patch.object(module,'_expanded_row',side_effect=row):
                    result=module.adaptive_kernel_matvec(K,alpha)
            else:
                with patch.object(module,'_expanded_row',side_effect=row):result=module.adaptive_kernel_matvec(K,alpha)
            records.append((result,expanded,fast))
        # Compare real old/new routes, not a restatement of the new formula.
        np.testing.assert_array_equal(records[0][0],records[1][0])
        self.assertEqual(records[0][1],records[1][1])
        fast_rows=set()
        for block,good in enumerate(records[1][2]):
            if good:fast_rows.update(range(128*block,min(128*(block+1),K.shape[0])))
        self.assertFalse(fast_rows.intersection(records[0][1]))
        REPORT['score_cases'].append(dict(name=name,rows=K.shape[0],columns=K.shape[1],
            expanded_rows=records[0][1],first_tier_blocks=records[1][2],bitwise_and_routing_equal=True))

    def test_dense_layouts_signed_scales_and_mutated_alpha(self):
        rng=np.random.RandomState(914)
        K=rng.normal(size=(270,128));alpha=rng.normal(size=128)
        for scale in (1.,1e5,1e100,1e-200):
            for order in ('C','F'):
                self.paired_scores(np.array(K,order=order),alpha*scale,f'random-{scale}-{order}')
        model=alpha.copy()
        self.paired_scores(K,model,'before-public-alpha-mutation')
        model*=1e8
        self.paired_scores(K,model,'after-public-alpha-mutation')
        self.paired_scores(np.repeat(K,2,axis=0)[::2],alpha,'noncontiguous-rows')

    def test_threshold_cancellation_and_near_zero_signs(self):
        n=128;eps=np.finfo(float).eps
        nominal=5e-7/(n*n*eps)
        K=np.ones((260,n));K[::3,-1]=np.nextafter(1.,2.);K[1::3,-1]=np.nextafter(1.,0.)
        for multiplier in (.1,.8,1.-1e-12,1.,1.+1e-12,1.2,2.,1e5):
            alpha=np.tile([nominal*multiplier,-nominal*multiplier],n//2)
            self.paired_scores(K,alpha,'threshold-'+str(multiplier))

    def test_sparse_duplicates_keep_original_path(self):
        # Stored duplicate terms, including exact cancellation, remain distinct.
        K=csr_matrix((np.array([1.,1.,-1.,2.,-3.,4.,1e-200,1e-200]),
                      np.array([0,0,1,1,0,1,0,0]),np.array([0,4,6,8])),shape=(3,2))
        self.assertFalse(K.has_canonical_format)
        self.paired_scores(K,np.array([1e100,-1e100]),'csr-duplicates')

    def test_loose_first_tier_falls_back_to_passing_fine_screen(self):
        K=np.ones((260,128));K[:,0]=1e100
        alpha=np.ones(128);alpha[0]=0.
        self.paired_scores(K,alpha,'cheap-too-loose-fine-passes')
        detail=REPORT['score_cases'][-1]
        self.assertEqual(detail['first_tier_blocks'],[False])
        self.assertEqual(detail['expanded_rows'],[])

    def test_one_unsuccessful_trial_resets_next_call_and_keeps_later_finite_check(self):
        K=np.ones((260,128));K[:,0]=1e100;alpha=np.ones(128);alpha[0]=0.
        original=new_scores._dense_first_tier
        with patch.object(new_scores,'_dense_first_tier',wraps=original) as calls:
            new_scores.adaptive_kernel_matvec(K,alpha)
            self.assertEqual(calls.call_count,1)
            new_scores.adaptive_kernel_matvec(np.ones_like(K),alpha)
            self.assertEqual(calls.call_count,4)
        K[129,10]=np.nan
        with self.assertRaises(FloatingPointError):new_scores.adaptive_kernel_matvec(K,alpha)

    def test_overflowing_loose_bounds_and_subnormal_products(self):
        self.paired_scores(np.array([[1.,1.],[-1.,-1.],[0.,0.]]),np.array([1e308,-1e308]),'overflow-bound-finite-score')
        eta=np.nextafter(0.,1.)
        self.paired_scores(np.array([[eta,eta],[eta,-eta]]),np.array([.5,-.5]),'subnormal-products')

    def test_nonfinite_inputs_and_underflow_policy_not_bypassed(self):
        for K,alpha in ((np.array([[np.inf,0.]]),np.array([0.,1.])),
                        (np.array([[np.nan,1.]]),np.array([1.,1.])),
                        (np.array([[-np.inf,0.]]),np.array([0.,1.])),
                        (np.ones((2,2)),np.array([np.nan,1.]))):
            for module in (old_scores,new_scores):
                with self.assertRaises((ValueError,FloatingPointError)):
                    module.adaptive_kernel_matvec(K,alpha)
        for module in (old_scores,new_scores):
            with patch.object(module,'_gradual_underflow',side_effect=FloatingPointError('unsupported arithmetic')):
                with self.assertRaises(FloatingPointError):module.adaptive_kernel_matvec(np.eye(2),np.ones(2))
        errors=[]
        for module in (old_scores,new_scores):
            with np.errstate(under='raise'):
                try:module.adaptive_kernel_matvec(np.eye(2),np.ones(2));errors.append(None)
                except FloatingPointError as exc:errors.append(type(exc).__name__)
        self.assertEqual(errors[0],errors[1])

    def test_mm_state_objective_and_observation_parity_all_variants(self):
        rng=np.random.RandomState(991);X=rng.normal(size=(64,5));K=rbf_kernel(X,gamma=.1)
        y=np.where(np.arange(64)%2==0,-1.,1.)
        scenarios=[dict(max_iter=0),dict(max_iter=10,stopping='fixed'),
                   dict(max_iter=100,stopping='objective',q=3),
                   dict(max_iter=20,stopping='fixed',callback_stop=3),
                   dict(max_iter=20,stopping='validation',validation=(K[:10],y[:10]),patience=2),
                   dict(max_iter=10,stopping='fixed',acceleration='restart'),
                   dict(max_iter=5,stopping='fixed',backend='spectral'),
                   dict(max_iter=5,stopping='fixed',implementation='reference',alpha_init=rng.normal(size=64)/64)]
        for number,scenario in enumerate(scenarios):
            results=[];record=[]
            for name,module in variants.items():
                args=dict(scenario);stop=args.pop('callback_stop',None);observed=[];count=[0]
                def callback(state):
                    observed.append((state['iteration'],state['objective'],state['alpha'].copy(),state['decision_values'].copy()))
                    return state['iteration']==stop
                if stop is not None:args['callback']=callback
                def profile(frame,event,arg):
                    if event=='call' and frame.f_code.co_name=='objective' and frame.f_code.co_filename==module.__file__:
                        count[0]+=1
                previous_profile = sys.getprofile()
                sys.setprofile(profile)
                try:result=module.solve_kernel(K,y,.001,psd_known=True,**args)
                finally:sys.setprofile(previous_profile)
                results.append((result,observed));record.append(dict(variant=name,objective_evaluations=count[0],
                    n_iter=result['n_iter'],termination=result['termination_reason']))
            baseline,observed0=results[0]
            for candidate,observed in results[1:]:
                for key in ('alpha','objective_history'):
                    np.testing.assert_array_equal(candidate[key],baseline[key])
                for key in ('offset','n_iter','termination_reason','returned_iteration','prediction_precision','final_objective'):
                    self.assertEqual(candidate[key],baseline[key])
                self.assertEqual(len(observed),len(observed0))
                for first,second in zip(observed,observed0):
                    self.assertEqual(first[:2],second[:2])
                    np.testing.assert_array_equal(first[2],second[2]);np.testing.assert_array_equal(first[3],second[3])
            if number==1:
                self.assertEqual(record[0]['objective_evaluations'],21)
                self.assertEqual(record[1]['objective_evaluations'],11)
            REPORT['solver_cases'].append(dict(scenario=number,variants=record,all_states_bitwise_equal=True))

if __name__ == '__main__':
    unittest.main()
