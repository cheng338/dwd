"""Selective RBF distance repair, fitted-policy and cache regressions."""
import hashlib
import math
import pickle
import unittest
from fractions import Fraction
from decimal import Decimal, localcontext
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix, csr_array
import sklearn.metrics.pairwise as pw
from threadpoolctl import threadpool_limits
from sklearn.utils.extmath import row_norms
from dwd.gen_kern_dwd import KernGDWD
from dwd import kernel_utils as ku
from dwd import _prediction_norms as pn
from dwd import _guarded_rbf as guarded_module
from dwd._guarded_rbf import guarded_rbf, _packed_flags
from dwd._rbf import norm_sum_rbf
from dwd._direct_rbf import accurate_rbf

def oracle(X,Y,gamma):
    X=X.toarray() if hasattr(X,'toarray') else X
    Y=Y.toarray() if hasattr(Y,'toarray') else Y
    out=np.empty((len(X),len(Y)))
    with localcontext() as c:
        c.prec=100
        for i,x in enumerate(X):
            for j,y in enumerate(Y):
                squared=sum(((Fraction(float(a))-Fraction(float(b)))**2 for a,b in zip(x,y)),Fraction())
                exponent=squared*Fraction(float(gamma))
                out[i,j]=float((-(Decimal(exponent.numerator)/Decimal(exponent.denominator))).exp())
    return out

def hashes(a):
    arrays=(a.data,a.indices,a.indptr) if hasattr(a,'indptr') else (a,)
    return [hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for v in arrays]

def fit(X,y,**kwargs):
    return KernGDWD(kernel='rbf',kernel_kws={'gamma':.3},lambd=1e-3,
                    initialization='zero',stopping='fixed',max_iter=5,random_state=7,**kwargs).fit(X,y)


class GuardedRBFTests(unittest.TestCase):
    def test_integrated_fitted_cache_hook_and_error_policy_lifecycle(self):
        records=[]
        def record(case,**data):
            records.append({'case':case,**data})
        with threadpool_limits(1):
            rng=np.random.default_rng(912)
            for n,p,m in [(32,16,17),(8,4096,5),(65,7,71),(1,7,1)]:
                X=rng.normal(size=(n,p));Y=rng.normal(size=(m,p))
                for mode in ('sklearn','norm_sum'):
                    expected=pw.rbf_kernel(X,Y,gamma=.1) if mode=='sklearn' else norm_sum_rbf(X,Y,gamma=.1)
                    actual=guarded_rbf(X,Y,gamma=.1,mode=mode)
                    assert np.array_equal(expected,actual),(n,p,m,mode,np.max(np.abs(expected-actual)))
                    norms=row_norms(X,squared=True)
                    assert np.array_equal(actual,guarded_rbf(X,Y,gamma=.1,mode=mode,x_norm=norms))
                    record('ordinary-unflagged',n=n,p=p,m=m,mode=mode,bitwise_equal=True)
            delta=np.array([[0,0,0],[.125,.25,-.5],[1,-1,2],[0,0,0],[-2,4,-8]],dtype=float)
            cases=[('small-offset',delta,.125),('large-offset',delta+2**40,.125),
                   ('one-ulp',np.array([[2**40,0],[np.nextafter(float(2**40),np.inf),0]]),2**20),
                   ('overflow',np.array([[0.],[1e155],[2e155]]),1e-308),
                   ('subnormal-gamma',np.array([[0.],[1e161]]),5e-324),
                   ('underflow-distance',np.array([[0.],[1e-160],[2e-160]]),1e308),
                   ('zero-gamma',np.array([[-1e308],[1e308]]),0.)]
            for name,X,gamma in cases:
                ref=oracle(X,X,gamma)
                for mode in ('sklearn','norm_sum'):
                    K=guarded_rbf(X,X,gamma=gamma,mode=mode,self_kernel=True)
                    cross=guarded_rbf(X,X.copy(),gamma=gamma,mode=mode)
                    assert np.allclose(K,ref,rtol=4e-15,atol=1e-15),(name,mode,K,ref)
                    assert np.array_equal(K,cross),(name,mode)
                    record(name,mode=mode,max_oracle_error=float(np.max(np.abs(K-ref))),self_cross_equal=True)
            X=delta+2**40
            for left,right in [(X,csr_matrix(X)),(csr_matrix(X),X),(csr_array(X),csr_array(X))]:
                before=(hashes(left),hashes(right))
                K=guarded_rbf(left,right,gamma=.125)
                assert np.allclose(K,oracle(X,X,.125),rtol=4e-15,atol=1e-15)
                assert before==(hashes(left),hashes(right))
                record('mixed-and-sparse',left=type(left).__name__,right=type(right).__name__)
            rng=np.random.default_rng(20260908);rng.normal(size=(120,5))
            X=np.repeat(rng.normal(size=(4,3)),20,axis=0);y=np.tile([-1,1],40)
            model=KernGDWD(kernel='rbf',kernel_kws={'gamma':.3},q=1,lambd=1e-12,max_iter=20,
                           stopping='fixed',initialization='zero',random_state=7,backend='auto').fit(X,y)
            assert model.kernel_computation_=='guarded_sklearn_rel_v1'
            K=model._compute_kernel(model._Xfit)
            scores=[model.decision_function(X.copy(),prediction_batch_size=b) for b in (1,7,80,None)]
            assert all(np.array_equal(scores[0],s) for s in scores)
            assert np.array_equal(K,model._compute_kernel(X.copy()))
            for i in range(0,80,20):assert (K[i:i+20,i:i+20]==1).all()
            restored=pickle.loads(pickle.dumps(model))
            assert np.array_equal(scores[0],restored.decision_function(X,prediction_batch_size=7))
            record('duplicate-fitted-and-serialized',mode=model.kernel_computation_,batches_equal=True,
                   min_eigenvalue=float(np.linalg.eigvalsh(K).min()))
            assert pn.prepare_norm_kernel(model,X,X) is not None
            query=pn.prepare_norm_kernel(model,X,X)
            assert np.array_equal(query(X[:7]),model._compute_kernel(X[:7]))
            record('norm-cache-active',active=True)
            legacy=pickle.loads(pickle.dumps(model));legacy.kernel_computation_='sklearn'
            assert np.array_equal(legacy._compute_kernel(X.copy()),pw.rbf_kernel(legacy._Xfit,X.copy(),gamma=.3))
            record('legacy-serialized-policy',unchanged=True)
            small=X[::4].copy();ys=y[::4].copy();ys[::2]=1;ys[1::2]=-1
            cached=KernGDWD(kernel='rbf',kernel_kws={'gamma':.3},lambd=1e-3,max_iter=5,stopping='fixed',initialization='zero').cv_init(small)
            assert cached._cv_cache_matches(small)
            cached._cv_kernel_computation='sklearn'
            assert not cached._cv_cache_matches(small)
            record('old-cache-rejected',rejected=True)
            cached._cv_kernel_computation='guarded_sklearn_rel_v1'
            old=ku.pairwise_kernels
            calls=[]
            def wrapper(*args,**kwargs):
                calls.append(True);return old(*args,**kwargs)
            ku.pairwise_kernels=wrapper
            try:
                assert cached._cv_cache_matches(small)
                hooked=fit(small,ys)
                assert hooked.kernel_computation_=='sklearn'
                before=len(calls);model._compute_kernel(X[:3]);assert len(calls)>before
                assert pn.prepare_norm_kernel(model,X,X) is None
            finally:ku.pairwise_kernels=old
            record('post-import-hook',preserved=True,guarded_owned_cache_preserved=True)
            for owner,name in ((ku.KernelClfMixin,'_compute_kernel'),
                               (KernGDWD,'_compute_training_kernel'),
                               (KernGDWD,'_known_psd_kernel')):
                original=getattr(owner,name)
                def patched(self,*a,_original=original,**k):return _original(self,*a,**k)
                setattr(owner,name,patched)
                try:
                    hooked=fit(small,ys)
                    assert hooked.kernel_computation_=='sklearn',(owner,name,hooked.kernel_computation_)
                    assert cached._cv_cache_matches(small)
                finally:setattr(owner,name,original)
                record('class-level-hook',method=name,preserved=True)
            import types
            inst=KernGDWD(kernel='rbf',kernel_kws={'gamma':.3},max_iter=2,stopping='fixed')
            original=inst._compute_kernel
            inst._compute_kernel=types.MethodType(lambda self,Q:original(Q),inst)
            inst.fit(small,ys);assert inst.kernel_computation_=='sklearn'
            record('instance-hook',preserved=True)
            class Custom(KernGDWD):
                def _compute_kernel(self,X):return super()._compute_kernel(X)
            custom=Custom(kernel='rbf',kernel_kws={'gamma':.3},max_iter=2,stopping='fixed',initialization='zero').fit(small,ys)
            assert custom.kernel_computation_=='sklearn'
            record('subclass-hook',preserved=True)
            direct=fit(small,ys,rbf_computation='direct');assert direct.kernel_computation_=='direct_v1'
            assert np.array_equal(direct._compute_kernel(small),accurate_rbf(direct._Xfit,small,gamma=.3))
            record('explicit-direct',unchanged=True)
            fn=lambda A,B:pw.rbf_kernel(A,B,gamma=.3)
            custom=KernGDWD(kernel=fn,max_iter=2,stopping='fixed',initialization='zero').fit(small,ys)
            assert custom.kernel_computation_=='sklearn'
            record('callable',unchanged=True)
            for mode in ('norm_sum','guarded_norm_sum_rel_v1'):
                model.kernel_computation_=mode
                expected=norm_sum_rbf(model._Xfit,X,gamma=.3) if mode=='norm_sum' else guarded_rbf(model._Xfit,X,gamma=.3,mode='norm_sum')
                assert np.array_equal(model._compute_kernel(X),expected)
                assert pn.prepare_norm_kernel(model,X,X) is not None
                record('explicit-fitted-mode',mode=mode)
            for order in ('F','C'):
                A=np.array(delta,order=order);A.flags.writeable=False
                guard=guarded_rbf(A,A,gamma=.125,self_kernel=True)
                assert np.array_equal(guard,guarded_rbf(A,A.copy(),gamma=.125))
                record('layout',order=order,readonly=True)
            guard=guarded_rbf(delta[:,::-1],delta[:,::-1].copy(),gamma=.125)
            assert np.allclose(guard,oracle(delta,delta,.125),atol=1e-15)
            record('negative-stride',passed=True)
            model.kernel_computation_='guarded_sklearn_rel_v1'
            for policy in ('raise','ignore','warn'):
                with np.errstate(all=policy):
                    a=model.decision_function(X,prediction_batch_size=None)
                    b=model.decision_function(X,prediction_batch_size=7)
                    assert np.array_equal(a,b)
                    if policy=='raise':assert not model._guarded_rbf_eligible()
                record('fitted-error-policy',policy=policy,batches_equal=True)
            # A fitted mode must retain its corrected construction if callers later
            # alter np.seterr. Use a synthetic fitted coefficient state here so the
            # probe isolates prediction arithmetic, without claiming solver success
            # for an overflowing feature norm or extreme regularization case.
            for name,A,gamma in cases:
                artificial=pickle.loads(pickle.dumps(model))
                artificial._Xfit=A.copy();artificial.kernel_kws_={'gamma':gamma}
                artificial.n_features_in_=A.shape[1]
                artificial.dual_coef_=np.arange(1,len(A)+1,dtype=float)[None,:]/len(A)
                artificial.intercept_=0.;artificial.prediction_precision_='ordinary'
                artificial.affine_computation_='standard'
                for policy in ('raise','ignore','warn'):
                    with np.errstate(all=policy):
                        one=artificial.decision_function(A.copy(),prediction_batch_size=1)
                        allrows=artificial.decision_function(A.copy(),prediction_batch_size=None)
                    assert np.allclose(one,allrows,rtol=4e-15,atol=1e-15),(name,policy)
                    assert np.isfinite(one).all()
                    record('pathological-fitted-error-policy',fixture=name,policy=policy,synthetic_coefficients=True)

    def test_all_clear_matches_forced_full_scan(self):
        rng = np.random.default_rng(1938)
        with threadpool_limits(1):
            for p in (1, 3, 32, 4096):
                for n, m in ((1, 1), (8, 7), (65, 71)):
                    for sparse in (False, True):
                        A, B = rng.normal(size=(n, p)), rng.normal(size=(m, p))
                        if sparse:
                            A, B = csr_matrix(A), csr_matrix(B)
                        for mode in ('sklearn', 'norm_sum'):
                            with self.subTest(p=p, shape=(n,m), sparse=sparse, mode=mode):
                                normal = guarded_rbf(A, B, gamma=1/p, mode=mode)
                                with patch.object(guarded_module, '_all_clear', return_value=False):
                                    forced = guarded_rbf(A, B, gamma=1/p, mode=mode)
                                np.testing.assert_array_equal(normal, forced)

    def test_all_clear_declines_nonfinite_distances_and_bounds(self):
        for distance in (np.array([[1., np.inf]]), np.array([[1., np.nan]])):
            self.assertFalse(guarded_module._all_clear(distance, np.array([1.]), np.array([1.,1.]), 1))
        for norm in (np.array([np.inf]), np.array([np.nan])):
            self.assertFalse(guarded_module._all_clear(np.array([[100.]]), norm, np.array([1.]), 1))

    def test_zero_gamma_still_validates_features_but_skips_distances(self):
        A = np.array([[-1e308], [1e308]])
        with patch.object(guarded_module, '_all_clear', side_effect=AssertionError('Distance scan')):
            np.testing.assert_array_equal(guarded_rbf(A, A, gamma=0), np.ones((2,2)))
        for A in (np.array([[np.nan]]), np.array([[np.inf]])):
            with self.assertRaises(ValueError):
                guarded_rbf(A, A, gamma=0)

    def test_relative_trigger_catches_positive_but_ill_conditioned_distance(self):
        # The sign-only trigger would leave this positive distance unflagged.
        X = np.array([[2.**40, 2.**40, 2.**40]])
        Y = X + np.array([[2.**20, -2.**20, 2.**19]])
        expected = oracle(X, Y, 2.**-40)
        np.testing.assert_allclose(guarded_rbf(X, Y, gamma=2.**-40), expected, rtol=4e-15, atol=1e-15)
        norms_x, norms_y = row_norms(X,squared=True), row_norms(Y,squared=True)
        represented = pw.euclidean_distances(X,Y,squared=True)
        flags = guarded_module._packed_flags(represented,norms_x,norms_y,3)
        self.assertIsNotNone(flags)
        self.assertTrue(bool(np.unpackbits(flags,axis=1,count=1)[0,0]))

    def test_actual_self_identity_does_not_scan_or_recompute_diagonal_pairs(self):
        X = np.random.default_rng(512034).normal(size=(65,32))
        for A in (X,csr_matrix(X)):
            with (patch.object(guarded_module,'_equal',side_effect=AssertionError('Identity scan')),
                  patch.object(guarded_module,'_rbf_from_validated_features',side_effect=AssertionError('Identity repair'))):
                K=guarded_rbf(A,A,gamma=.1)
            np.testing.assert_array_equal(np.diag(K),np.ones(len(X)))

    def test_diagonal_identity_does_not_follow_shape_or_self_kernel_flag(self):
        X=np.array([[0.,0.],[1.,0.],[0.,1.]])
        Y=X+np.array([.25,.125])
        for sparse in (False,True):
            A,B=(csr_matrix(X),csr_matrix(Y)) if sparse else (X,Y)
            actual=guarded_rbf(A,B,gamma=.5,mode='sklearn',self_kernel=True)
            np.testing.assert_allclose(actual,oracle(X,Y,.5),rtol=4e-15,atol=1e-15)
            self.assertTrue((np.diag(actual)<1).all())

    def test_packed_diagonal_boundaries_preserve_offdiagonal_duplicate_repairs(self):
        rng=np.random.default_rng(15203)
        for n in (1,7,8,9,63,64,65,129):
            X=rng.normal(size=(n,3))
            if n>1:
                X[-1]=X[0]
            for sparse in (False,True):
                A=csr_matrix(X) if sparse else X
                for mode in ('sklearn','norm_sum'):
                    with self.subTest(n=n,sparse=sparse,mode=mode):
                        self_kernel=guarded_rbf(A,A,gamma=.3,mode=mode,self_kernel=True)
                        cross_kernel=guarded_rbf(A,A.copy(),gamma=.3,mode=mode)
                        np.testing.assert_allclose(self_kernel,cross_kernel,rtol=4e-15,atol=1e-15)
                        np.testing.assert_array_equal(np.diag(self_kernel),np.ones(n))
                        self.assertEqual(self_kernel[0,-1],1. if n>1 else self_kernel[0,0])

    def test_packed_flags_have_independent_values_across_256_boundaries(self):
        # The expected mask is defined by below/equal/above-threshold cases,
        # then packed explicitly by bit position rather than np.packbits.
        eps, tiny = np.finfo(float).eps, np.finfo(float).tiny
        p = 127
        operation_bound = (2 * p + 3) * eps
        gamma_bound = operation_bound / (1 - operation_bound)
        factor = (2 * gamma_bound / (1 - gamma_bound)) / math.sqrt(eps)
        for n, m in ((9, 255), (9, 256), (9, 257), (255, 9), (256, 9),
                     (257, 9), (257, 513)):
            x_norm = np.arange(1, n + 1, dtype=float) / 8
            y_norm = np.arange(1, m + 1, dtype=float) / 16
            envelope = factor * (x_norm[:, None] + y_norm[None, :])
            envelope += (2 * p + 3) * tiny / math.sqrt(eps)
            category = (np.arange(n)[:, None] + np.arange(m)[None, :]) % 3
            distances = envelope.copy()
            distances[category == 0] = np.nextafter(distances[category == 0], -np.inf)
            distances[category == 1] = np.nextafter(distances[category == 1], np.inf)
            flags = category != 1
            distances[0, :3] = [np.nan, np.inf, -np.inf]
            flags[0, :3] = True
            x_norm[-1] = np.inf
            flags[-1, :] = True
            expected = np.zeros((n, (m + 7) // 8), dtype=np.uint8)
            for j in range(m):
                expected[:, j // 8] |= flags[:, j].astype(np.uint8) << (7 - j % 8)
            before = (hashes(distances), hashes(x_norm), hashes(y_norm))
            with self.subTest(shape=(n, m)):
                actual = _packed_flags(distances, x_norm, y_norm, p)
                np.testing.assert_array_equal(actual, expected)
                # Strided storage must have the same bit positions and values.
                wider = np.zeros((n, 2 * m))
                wider[:, ::2] = distances
                np.testing.assert_array_equal(_packed_flags(wider[:, ::2], x_norm, y_norm, p), expected)
                self.assertEqual(before, (hashes(distances), hashes(x_norm), hashes(y_norm)))

    def test_active_rows_preserve_ascending_repair_order_and_feature_chunks(self):
        for p in (7, 2048):
            X = np.zeros((9, p))
            Y = np.zeros((513, p))
            X[:, 0] = np.arange(9)
            Y[:, 0] = 1000 + np.arange(513)
            flags = np.zeros((9, 513), dtype=bool)
            active_rows = [1, 4, 8]
            flags[active_rows, :] = True
            flags[4, ::3] = False
            packed = np.packbits(flags, axis=1)
            before = (hashes(X), hashes(Y), hashes(packed))
            expected_calls = []
            limit = min(256, 262144 // p)
            for i in active_rows:
                columns = [j for j in range(513) if flags[i, j]]
                for start in range(0, len(columns), limit):
                    expected_calls.append((i, columns[start:start + limit], .3, 256))
            for sparse in (False, True):
                A, B = (csr_matrix(X), csr_matrix(Y)) if sparse else (X, Y)
                calls = []
                def direct(left, right, gamma, batch_size):
                    row = int(left[0, 0])
                    column = right[:, 0].toarray().ravel() if sparse else right[:, 0]
                    calls.append((row, (column - 1000).astype(int).tolist(), gamma, batch_size))
                    return np.full((1, right.shape[0]), .125)
                K = np.full(flags.shape, .5)
                with self.subTest(p=p, sparse=sparse):
                    with patch.object(guarded_module, '_rbf_from_validated_features', side_effect=direct):
                        returned = guarded_module._apply_pairs(K, packed, A, B, .3)
                    self.assertIs(returned, K)
                    self.assertEqual(calls, expected_calls)
                    np.testing.assert_array_equal(K, np.where(flags, .125, .5))
                    self.assertEqual(before, (hashes(X), hashes(Y), hashes(packed)))

    def test_active_rows_are_selected_after_identity_diagonal_is_cleared(self):
        X = np.arange(513, dtype=float)[:, None]
        flags = np.eye(513, dtype=bool)
        pairs = [(1, 0), (256, 255), (256, 512), (512, 1)]
        for i, j in pairs:
            flags[i, j] = True
        packed = np.packbits(flags, axis=1)
        calls = []
        def direct(left, right, gamma, batch_size):
            calls.append((int(left[0, 0]), right[:, 0].astype(int).tolist()))
            return np.full((1, len(right)), .125)
        K = np.full((513, 513), .5)
        with patch.object(guarded_module, '_rbf_from_validated_features', side_effect=direct):
            guarded_module._apply_pairs(K, packed, X, X, .3)
        self.assertEqual(calls, [(1, [0]), (256, [255, 512]), (512, [1])])
        expected_flags = np.zeros_like(flags)
        for i, j in pairs:
            expected_flags[i, j] = True
            self.assertEqual(K[i, j], .125)
        np.testing.assert_array_equal(packed, np.packbits(expected_flags, axis=1))
        np.testing.assert_array_equal(np.diag(K), np.ones(513))


if __name__ == '__main__':
    unittest.main()
