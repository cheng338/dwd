"""Call-local CSR norm reuse preserves the original per-batch semantics."""
import copy
import functools
import importlib
import pickle
import sys
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal
from scipy.sparse import csr_array, csr_matrix
import sklearn.metrics.pairwise as pairwise
from sklearn.utils.extmath import row_norms
from threadpoolctl import threadpool_limits

from dwd import _guarded_rbf as guarded
from dwd import _prediction_norms as norms
from dwd import _sparse_features as sparse_features
from dwd import kernel_utils
from dwd.gen_kern_dwd import KernGDWD


MODES=('sklearn','norm_sum','guarded_sklearn_rel_v1','guarded_norm_sum_rel_v1')


def snapshot(X):
    return (X.data.copy(),X.indices.copy(),X.indptr.copy(),
            getattr(X,'_has_sorted_indices',None),
            getattr(X,'_has_canonical_format',None))


def assert_untouched(test,X,before):
    after=snapshot(X)
    for left,right in zip(before[:3],after[:3]):
        assert_array_equal(left,right)
    test.assertEqual(before[3:],after[3:])


def duplicate_storage(X):
    result=csr_matrix((np.repeat(X.data/2,2),np.repeat(X.indices,2),X.indptr*2),shape=X.shape)
    result.has_sorted_indices=True
    result.has_canonical_format=True
    return result


class SparsePredictionNormReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limit=threadpool_limits(limits=1)
        rng=np.random.default_rng(826004)
        X=rng.normal(size=(32,9)); X[rng.random(X.shape)<.55]=0
        Q=rng.normal(size=(13,9)); Q[rng.random(Q.shape)<.55]=0
        cls.X=csr_matrix(X); cls.Q=csr_matrix(Q)
        cls.base=KernGDWD(kernel='rbf',kernel_kws={'gamma':.17},lambd=.05,
                         max_iter=8,stopping='fixed',prediction_batch_size=3).fit(
                             cls.X,np.tile([-3,7],16))

    @classmethod
    def tearDownClass(cls):
        cls.limit.restore_original_limits()

    def model(self,mode='guarded_sklearn_rel_v1',family=csr_matrix):
        result=copy.deepcopy(self.base)
        result._Xfit=family(result._Xfit)
        result.kernel_computation_=mode
        return result

    def compare(self,model,query,**kwargs):
        with patch.object(norms,'prepare_norm_kernel',return_value=None):
            expected=model.decision_function(query,**kwargs)
        actual=model.decision_function(query,**kwargs)
        assert_array_equal(actual,expected)
        return actual

    def compare_outcome(self,model,query):
        outcomes=[]
        for reuse in (False,True):
            with ExitStack() as stack:
                if not reuse: stack.enter_context(patch.object(norms,'prepare_norm_kernel',return_value=None))
                try: outcomes.append(('returned',model.decision_function(query)))
                except Exception as exc: outcomes.append((type(exc).__name__,str(exc)))
        self.assertEqual(outcomes[0][0],outcomes[1][0])
        if outcomes[0][0]=='returned': assert_array_equal(outcomes[0][1],outcomes[1][1])
        else: self.assertEqual(outcomes[0][1],outcomes[1][1])

    def test_canonical_csr_reuses_training_norm_and_scan_once_per_call(self):
        for family in (csr_matrix,csr_array):
            for mode in MODES:
                with self.subTest(family=family.__name__,mode=mode):
                    model=self.model(mode,family); query=family(self.Q)
                    before_model=set(vars(model)); before_basis=snapshot(model._Xfit)
                    before_query=snapshot(query)
                    counts=[]
                    for reuse in (False,True):
                        seen=[0,0]
                        def observe(frame,event,arg):
                            if event=='call' and frame.f_locals.get('X') is model._Xfit:
                                if frame.f_code is row_norms.__code__: seen[0]+=1
                                if frame.f_code is sparse_features.csr_is_canonical.__code__: seen[1]+=1
                        with ExitStack() as stack:
                            if not reuse: stack.enter_context(patch.object(norms,'prepare_norm_kernel',return_value=None))
                            sys.setprofile(observe)
                            try: actual=model.decision_function(query)
                            finally: sys.setprofile(None)
                        counts.append(seen)
                        if not reuse: expected=actual
                        else: assert_array_equal(actual,expected)
                    self.assertEqual(counts,[[5,5],[1,1]])
                    self.assertEqual(set(vars(model)),before_model)
                    assert_untouched(self,model._Xfit,before_basis)
                    assert_untouched(self,query,before_query)

    def test_batch_alias_precision_and_pickle_preserve_scores(self):
        for mode in MODES:
            for precision in ('ordinary','adaptive','compensated'):
                model=self.model(mode); model.prediction_precision_=precision
                for query in (self.Q,model._Xfit,model._Xfit.copy()):
                    for batch in (1,3,12,None,100):
                        with self.subTest(mode=mode,precision=precision,batch=batch):
                            expected=self.compare(model,query,prediction_batch_size=batch)
                            assert_array_equal(pickle.loads(pickle.dumps(model)).decision_function(
                                query,prediction_batch_size=batch),expected)

    def test_query_and_basis_mutation_are_rechecked_each_public_call(self):
        for mode in MODES:
            model=self.model(mode); query=self.Q.copy()
            self.compare(model,query)
            model._Xfit.data[0]+=.125
            query.data[-1]-=.25
            self.compare(model,query)
            model._Xfit=duplicate_storage(model._Xfit)
            before=snapshot(model._Xfit)
            self.assertIsNone(norms.prepare_norm_kernel(model,query,query))
            if mode in ('sklearn','norm_sum'):
                with self.assertRaisesRegex(ValueError,'Refit the model'):
                    model.decision_function(query)
            else:
                self.compare(model,query)
            assert_untouched(self,model._Xfit,before)

    def test_noncanonical_query_falls_back_without_mutating_input(self):
        for mode in MODES:
            model=self.model(mode)
            query=duplicate_storage(self.Q)
            before=snapshot(query)
            self.assertIsNone(norms.prepare_norm_kernel(model,query,query))
            self.compare(model,query)
            assert_untouched(self,query,before)

    def test_readonly_canonical_buffers_remain_eligible(self):
        for family in (csr_matrix,csr_array):
            model=self.model(family=family); query=family(self.Q.copy())
            for X in (model._Xfit,query):
                X.data.flags.writeable=False
                X.indices.flags.writeable=False
                X.indptr.flags.writeable=False
            before_basis=snapshot(model._Xfit); before_query=snapshot(query)
            self.assertIsNotNone(norms.prepare_norm_kernel(model,query,query))
            self.compare(model,query)
            assert_untouched(self,model._Xfit,before_basis)
            assert_untouched(self,query,before_query)
            self.assertFalse(model._Xfit.data.flags.writeable)
            self.assertFalse(query.indices.flags.writeable)

    def test_unsorted_unique_stale_flags_use_safe_original_route(self):
        for mode in MODES:
            for where in ('basis','query'):
                model=self.model(mode); query=self.Q.copy()
                X=model._Xfit if where=='basis' else query
                for i in range(X.shape[0]):
                    lo,hi=X.indptr[i:i+2]
                    X.indices[lo:hi]=X.indices[lo:hi][::-1]
                    X.data[lo:hi]=X.data[lo:hi][::-1]
                X.has_sorted_indices=X.has_canonical_format=True
                before=snapshot(X)
                self.assertIsNone(norms.prepare_norm_kernel(model,query,query))
                self.compare(model,query)
                assert_untouched(self,X,before)

    def test_mixed_layout_dtype_formats_and_subclasses_keep_original_route(self):
        class SparseSubclass(csr_matrix):
            pass
        model=self.model()
        for query in (self.Q.toarray(),self.Q.astype(np.float32),
                      self.Q.astype(np.int32),self.Q.tocsc(),self.Q.tocoo(),SparseSubclass(self.Q)):
            checked=kernel_utils.check_array(query,dtype=np.float64,accept_sparse='csr')
            self.assertIsNone(norms.prepare_norm_kernel(model,checked,query))
            self.compare(model,query)
        model._Xfit=SparseSubclass(model._Xfit)
        self.assertIsNone(norms.prepare_norm_kernel(model,self.Q,self.Q))
        self.compare(model,self.Q)

    def test_replaced_sparse_and_guarded_helpers_are_observable(self):
        targets=((sparse_features,'csr_is_canonical'),
                 (sparse_features,'canonical_sparse_features'),
                 (sparse_features,'checked_prediction_basis'),
                 (kernel_utils,'canonical_sparse_features'),
                 (kernel_utils,'checked_prediction_basis'),
                 (pairwise,'safe_sparse_dot'),
                 (guarded,'guarded_rbf'),(guarded,'row_norms'),
                 (guarded,'euclidean_distances'),(guarded,'check_pairwise_arrays'),
                 (guarded,'safe_sparse_dot'))
        model=self.model()
        for module,name in targets:
            with self.subTest(module=module.__name__,name=name):
                original=getattr(module,name)
                @functools.wraps(original)
                def wrapped(*args,**kwargs):
                    return original(*args,**kwargs)
                with patch.object(module,name,wrapped):
                    self.assertIsNone(norms.prepare_norm_kernel(model,self.Q,self.Q))
                    self.compare(model,self.Q)

    def test_mutating_pairwise_dot_helper_keeps_fresh_per_batch_norms(self):
        for mode in ('sklearn','guarded_sklearn_rel_v1'):
            outputs=[]; bases=[]; counts=[]
            for reuse in (False,True):
                model=self.model(mode); original=pairwise.safe_sparse_dot; seen=[0]
                @functools.wraps(original)
                def wrapped(left,right,**kwargs):
                    answer=original(left,right,**kwargs)
                    if left is model._Xfit:
                        seen[0]+=1
                        model._Xfit.data[0]+=.01
                    return answer
                with ExitStack() as stack:
                    stack.enter_context(patch.object(pairwise,'safe_sparse_dot',wrapped))
                    if not reuse:
                        stack.enter_context(patch.object(norms,'prepare_norm_kernel',return_value=None))
                    else:
                        self.assertIsNone(norms.prepare_norm_kernel(model,self.Q,self.Q))
                    outputs.append(model.decision_function(self.Q))
                bases.append(model._Xfit.data.copy()); counts.append(seen[0])
            assert_array_equal(outputs[0],outputs[1])
            assert_array_equal(bases[0],bases[1])
            self.assertEqual(counts,[5,5])

    def test_preimport_sparse_wrappers_decline_after_reload(self):
        for name in ('csr_is_canonical','canonical_sparse_features','checked_prediction_basis'):
            original=getattr(sparse_features,name)
            @functools.wraps(original)
            def wrapped(*args,**kwargs):
                return original(*args,**kwargs)
            try:
                with patch.object(sparse_features,name,wrapped):
                    importlib.reload(norms)
                    self.assertFalse(norms._SPARSE_SOURCE_OK)
                    model=self.model()
                    self.assertIsNone(norms.prepare_norm_kernel(model,self.Q,self.Q))
                    self.compare(model,self.Q)
            finally:
                importlib.reload(norms)
        self.assertTrue(norms._SPARSE_SOURCE_OK)

    def test_optional_sparse_norm_allocation_failure_falls_back(self):
        model=self.model()
        original=norms.extmath.csr_row_norms
        failures=[]
        def limited(*args,**kwargs):
            frame=sys._getframe(1)
            while frame:
                if frame.f_code.co_name=='prepare_norm_kernel':
                    failures.append(True)
                    raise MemoryError('optional sparse norm allocation')
                frame=frame.f_back
            return original(*args,**kwargs)
        with patch.object(norms.extmath,'csr_row_norms',limited):
            self.compare(model,self.Q)
        self.assertEqual(len(failures),1)

    def test_mutating_estimator_hooks_keep_per_batch_route(self):
        for target in ('_compute_kernel','_kernel_decision_product'):
            for class_hook in (False,True):
                outputs=[]; bases=[]; counts=[]
                for reuse in (False,True):
                    model=self.model(); original=getattr(model,target); seen=[0]
                    def wrapped(self,values,original=original,seen=seen):
                        seen[0]+=1
                        result=original(values)
                        self._Xfit.data[0]+=.01
                        return result
                    with ExitStack() as stack:
                        if class_hook:
                            stack.enter_context(patch.object(kernel_utils.KernelClfMixin,target,wrapped))
                        else:
                            setattr(model,target,types.MethodType(wrapped,model))
                        if not reuse:
                            stack.enter_context(patch.object(norms,'prepare_norm_kernel',return_value=None))
                        outputs.append(model.decision_function(self.Q))
                    bases.append(model._Xfit.data.copy()); counts.append(seen[0])
                assert_array_equal(outputs[0],outputs[1])
                assert_array_equal(bases[0],bases[1])
                self.assertEqual(counts,[5,5])

    def test_range_and_floating_error_policy_fall_back(self):
        for mode in MODES:
            for where,value in (('basis',1e200),('query',1e200)):
                model=self.model(mode); query=self.Q.copy()
                target=model._Xfit if where=='basis' else query
                target.data[0]=value
                if where=='basis':
                    self.assertIsNone(norms.prepare_norm_kernel(model,query,query))
                with np.errstate(over='ignore',invalid='ignore'):
                    self.compare_outcome(model,query)
            for policy in ('raise','warn'):
                model=self.model(mode)
                with np.errstate(under=policy):
                    self.assertIsNone(norms.prepare_norm_kernel(model,self.Q,self.Q))
        model=self.model()
        for gamma in (0.,np.nextafter(0.,1.),.17):
            model.kernel_kws_={'gamma':gamma}
            self.compare(model,self.Q)

    def test_concurrent_readonly_calls_reuse_only_private_call_state(self):
        model=self.model()
        queries=[self.Q,self.Q[::-1],self.Q*.8,self.Q*1.1]
        with patch.object(norms,'prepare_norm_kernel',return_value=None):
            expected=[model.decision_function(query) for query in queries]
        before=set(vars(model))
        with ThreadPoolExecutor(max_workers=2) as executor:
            actual=list(executor.map(model.decision_function,queries))
        for left,right in zip(actual,expected): assert_array_equal(left,right)
        self.assertEqual(set(vars(model)),before)


if __name__=='__main__':
    unittest.main()
