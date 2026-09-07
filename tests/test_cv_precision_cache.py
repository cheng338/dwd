"""Reuse corrected float64 preparation for identical lower-precision features."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy.sparse import csr_matrix
from numpy.testing import assert_allclose
from dwd.gen_kern_dwd import KernGDWD
from dwd.cv import run_cv


class CVPrecisionCacheTests(unittest.TestCase):
    def test_float32_lambda_path_prepares_once_per_fold_for_both_implementations(self):
        rng=np.random.default_rng(704)
        X=rng.normal(size=(24,4)).astype(np.float32); y=np.tile([-1,1],12)
        original=KernGDWD.cv_init
        for implementation in ('reference','optimized'):
            for features in (X,csr_matrix(X)):
                with self.subTest(implementation=implementation,sparse=hasattr(features,'toarray')):
                    calls=[]
                    def observed(model,data):
                        calls.append(data.shape)
                        return original(model,data)
                    model=KernGDWD(implementation=implementation,kernel='rbf',kernel_kws={'gamma':.3},
                        initialization='zero',stopping='fixed',max_iter=2)
                    with patch.object(KernGDWD,'cv_init',observed):
                        actual=run_cv(model,features,y,{'lambd':[.1,.2]},cv=2,refit_best=False)
                    self.assertEqual(len(calls),2)
                    expected=run_cv(model,features.astype(np.float64),y,{'lambd':[.1,.2]},cv=2,refit_best=False)
                    assert_allclose(actual[3]['mean_test_score'],expected[3]['mean_test_score'],atol=0,rtol=0)

    def test_cache_still_rejects_changed_values_and_representation(self):
        X=np.arange(24,dtype=np.float32).reshape(8,3)
        corrected=KernGDWD(kernel='rbf').cv_init(X)
        self.assertTrue(corrected._cv_cache_matches(X.copy()))
        changed=X.copy();changed[0,0]=np.nextafter(np.float32(0),np.float32(1))
        self.assertFalse(corrected._cv_cache_matches(changed))
        self.assertFalse(corrected._cv_cache_matches(X[::-1]))
        self.assertFalse(corrected._cv_cache_matches(csr_matrix(X)))
        legacy=KernGDWD(kernel='rbf',solver_mode='legacy').cv_init(X)
        self.assertTrue(legacy._cv_cache_matches(X))
        self.assertFalse(legacy._cv_cache_matches(X.astype(np.float64)))


if __name__=='__main__': unittest.main(verbosity=2)
