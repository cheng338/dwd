"""Corrected linear lambda paths reuse canonical float64 preparation."""
import unittest
from unittest.mock import patch
import numpy as np
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix
from dwd.gen_dwd import GenDWD
from dwd.cv import run_cv


class LinearPrecisionCacheTests(unittest.TestCase):
    def test_float32_dense_and_csr_prepare_once_per_fold(self):
        rng=np.random.default_rng(731)
        X=rng.normal(size=(24,4)).astype(np.float32);y=np.tile([-1,1],12)
        original=GenDWD.cv_init
        for data in (X,csr_matrix(X)):
            with self.subTest(sparse=hasattr(data,'toarray')):
                calls=[]
                def observed(model,features):
                    calls.append(features.shape)
                    return original(model,features)
                model=GenDWD(solver_mode='schur',initialization='zero',stopping='fixed',max_iter=3)
                with patch.object(GenDWD,'cv_init',observed):
                    got=run_cv(model,data,y,{'lambd':[.1,.2]},cv=2,refit_best=True)
                self.assertEqual(len(calls),2)
                expected=run_cv(model,data.astype(np.float64),y,{'lambd':[.1,.2]},cv=2,refit_best=True)
                assert_allclose(got[3]['mean_test_score'],expected[3]['mean_test_score'],atol=0,rtol=0)
                assert_allclose(got[2].coef_,expected[2].coef_,atol=0,rtol=0)
                assert_allclose(got[2].intercept_,expected[2].intercept_,atol=0,rtol=0)

    def test_changed_values_rows_representation_and_legacy_dtype_still_invalidate(self):
        X=np.arange(24,dtype=np.float32).reshape(8,3)
        model=GenDWD(solver_mode='schur').cv_init(X)
        self.assertTrue(model._cv_cache_matches(X.copy()))
        changed=X.copy();changed[0,0]=np.nextafter(np.float32(0),np.float32(1))
        self.assertFalse(model._cv_cache_matches(changed))
        self.assertFalse(model._cv_cache_matches(X[::-1]))
        self.assertFalse(model._cv_cache_matches(csr_matrix(X)))
        legacy=GenDWD(solver_mode='legacy').cv_init(X)
        self.assertTrue(legacy._cv_cache_matches(X))
        self.assertFalse(legacy._cv_cache_matches(X.astype(np.float64)))


if __name__=='__main__':unittest.main(verbosity=2)
