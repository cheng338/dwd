"""Target checks replace provider-specific probes and child recovery machinery."""
import os
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import eigh as scipy_eigh, LinAlgError
from threadpoolctl import threadpool_limits, threadpool_info

import dwd._eigen as eigen
from dwd.gen_kern_dwd import KernGDWD


class NativeEigenRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.values = np.array([-2., 0., 1e-12, 3.])
        self.A = np.diag(self.values)
        self.U = np.eye(4)

    def test_native_success_preserves_environment_threads_and_uses_no_child(self):
        environment = dict(os.environ)
        scipy_eigh(np.eye(2))  # initialize any lazily loaded runtime before inventory
        with threadpool_limits(2), patch.object(subprocess, 'Popen', side_effect=AssertionError('no child process')):
            before = threadpool_info()
            (values,U),info = eigen.validated_eigh(self.A,return_info=True)
            self.assertEqual(threadpool_info(),before)
        assert_array_equal(values,self.values)
        assert_array_equal(U,self.U)
        self.assertEqual(dict(os.environ),environment)
        self.assertEqual(info['accepted_driver'],'evd')
        self.assertEqual(info['configured_drivers'],['evd','evr','evx'])
        self.assertNotIn('provider_selfcheck',info)
        self.assertNotIn('isolated_recovery_attempted',info)

    def test_invalid_returned_basis_rejected_with_runtime_advice_and_no_child(self):
        wrong = self.U[:,::-1]
        environment = dict(os.environ)
        with patch.object(eigen,'eigh',return_value=(self.values,wrong)) as calls, \
                patch.object(subprocess,'Popen',side_effect=AssertionError('no child process')):
            with self.assertRaisesRegex(eigen.EigenDecompositionError,'Update or replace NumPy/SciPy') as caught:
                eigen.validated_eigh(self.A)
        self.assertEqual(calls.call_count,3)
        self.assertTrue(all(not item['accepted'] for item in caught.exception.attempts))
        self.assertIn('eigen-equation',str(caught.exception))
        self.assertEqual(dict(os.environ),environment)

    def test_native_exception_and_invalid_fallback_still_validate_third_driver(self):
        def provider(A,**kwargs):
            if kwargs['driver']=='evd': raise LinAlgError('injected EVD failure')
            values,U=scipy_eigh(A,driver='evx')
            if kwargs['driver']=='evr': U[:,2]*=1.1
            return values,U
        with patch.object(eigen,'eigh',side_effect=provider):
            _,info=eigen.validated_eigh(self.A,return_info=True)
        self.assertEqual([a['accepted'] for a in info['attempts']],[False,False,True])
        self.assertEqual(info['accepted_driver'],'evx')

    def test_cancellation_and_memory_failure_propagate_without_retry_or_settings_change(self):
        environment=dict(os.environ)
        for error in (KeyboardInterrupt('cancel'),MemoryError('resource failure')):
            with self.subTest(error=type(error).__name__), \
                    patch.object(eigen,'eigh',side_effect=error) as calls, \
                    patch.object(subprocess,'Popen',side_effect=AssertionError('no child process')):
                with self.assertRaises(type(error)):
                    eigen.validated_eigh(self.A)
            self.assertEqual(calls.call_count,1)
            self.assertEqual(dict(os.environ),environment)

    def test_supplied_bad_basis_never_recomputed_or_repaired(self):
        order=np.array([3,1,0,2]); signs=np.array([-1.,1.,-1.,1.])
        supplied=(self.U[:,order]*signs,self.values[order])
        with patch.object(eigen,'eigh',side_effect=AssertionError('supplied must not recompute')), \
                patch.object(subprocess,'Popen',side_effect=AssertionError('no child process')):
            values,U=eigen.validated_eigh(self.A,supplied=supplied)
            assert_array_equal(values,supplied[1]);assert_array_equal(U,supplied[0])
            wrong=supplied[0].copy();wrong[:,2]*=1.05
            with self.assertRaisesRegex(ValueError,'Supplied eigenpairs'):
                eigen.validated_eigh(self.A,supplied=(wrong,supplied[1]))

    def test_observed_internal_norm_defect_shape_is_rejected_at_all_native_attempts(self):
        # The captured faulty EVR basis had non-unit internal columns, while
        # checks of just its extreme columns passed. This fixture reproduces
        # that structural fault without distributing MNIST or giant arrays.
        n=160;values=np.linspace(0.,2.,n);U=np.eye(n);U[:,77]*=1.02
        with patch.object(eigen,'eigh',return_value=(values,U)):
            with self.assertRaises(eigen.EigenDecompositionError) as caught:
                eigen.validated_eigh(np.diag(values))
        self.assertTrue(all('eigenvector columns are not normalized' in a['issues'] for a in caught.exception.attempts))
        self.assertTrue(all(77 in a['targeted_columns'] for a in caught.exception.attempts))

    def test_values_only_checks_remain_explicitly_weaker(self):
        values,info=eigen.validated_eigh(self.A,eigvals_only=True,return_info=True)
        assert_array_equal(values,self.values)
        self.assertFalse(info['attempts'][-1]['eigenvectors_validated'])
        self.assertEqual(info['attempts'][-1]['validation_kind'],'spectral_invariants_only')

    def test_public_reference_without_provider_probe_matches_original_system(self):
        rng=np.random.default_rng(1867)
        X=rng.normal(size=(520,3))
        y=np.where(X[:,0]*X[:,1]+.2*X[:,2]>0,1,-1)
        options=dict(kernel='rbf',kernel_kws={'gamma':.9},lambd=.07,q=1.7,
                     max_iter=4,stopping='fixed',initialization='zero')
        with patch.object(subprocess,'Popen',side_effect=AssertionError('no child process')):
            reference=KernGDWD(implementation='reference',**options).fit(X,y)
            optimized=KernGDWD(implementation='optimized',**options).fit(X,y)
        assert_allclose(reference.decision_function(X),optimized.decision_function(X),atol=2e-8,rtol=2e-8)
        assert_allclose(reference.objective_history_,optimized.objective_history_,atol=2e-10,rtol=2e-10)
        info=reference.diagnostics_['eigendecomposition_validation']
        self.assertEqual(info['accepted_driver'],'evd')
        self.assertTrue(info['attempts'][-1]['eigenvectors_validated'])
        self.assertNotIn('provider_selfcheck',info)


if __name__=='__main__': unittest.main(verbosity=2)
