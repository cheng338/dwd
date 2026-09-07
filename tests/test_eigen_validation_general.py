"""Fault-injected numerical preparation tests; no large data or conic solver."""
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import eigh as scipy_eigh, LinAlgError
from scipy.sparse import csr_matrix

import dwd._eigen as implementation
from dwd._eigen import validated_eigh, EigenDecompositionError
from dwd.gen_dwd import get_P0_eig, solve_gen_dwd


class GeneralEigenValidationTests(unittest.TestCase):
    def setUp(self):
        self.values=np.array([-2.,-.5,0.,1e-12,.2,1.,2.,4.])
        self.A=np.diag(self.values)
        self.U=np.eye(len(self.values))

    def test_signed_spectrum_is_preserved_without_psd_policy(self):
        (values,U),info=validated_eigh(self.A,return_info=True)
        assert_array_equal(values,self.values)
        assert_allclose((U*values)@U.T,self.A,atol=1e-15)
        self.assertEqual(info['accepted_driver'],'evd')
        self.assertTrue(info['attempts'][0]['eigenvectors_validated'])
        self.assertEqual(len(info['attempts']),1)

    def test_valid_singular_tiny_and_repeated_eigenspaces(self):
        values=np.array([0.,0.,1e-200,1e-10,1.,1.,2.,2.])
        matrix=np.diag(values)
        U=np.eye(8)
        angle=.37
        rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        U[:,4:6]=U[:,4:6]@rotation
        order=np.array([6,4,2,0,7,5,3,1])
        signs=np.array([1.,-1.,-1.,1.,1.,-1.,1.,-1.])
        supplied=(U[:,order]*signs,values[order])
        with patch.object(implementation,'eigh',side_effect=AssertionError('supplied pairs must not recompute')):
            got_values,got_U=validated_eigh(matrix,supplied=supplied)
        assert_array_equal(got_values,supplied[1])
        assert_array_equal(got_U,supplied[0])
        self.assertEqual(np.count_nonzero(got_values>0),6)
        computed,_=validated_eigh(matrix)
        self.assertEqual(computed[2],1e-200)

    def test_scale_normalized_checks_do_not_erase_tiny_or_large_matrices(self):
        for scale in (1e-200,1e200):
            matrix=scale*np.diag([-3.,0.,1.,2.])
            values,U=validated_eigh(matrix)
            assert_allclose(values/scale,[-3.,0.,1.,2.],atol=1e-14)
            assert_allclose(U.T@U,np.eye(4),atol=1e-14)

    def test_computed_column_norm_defect_recovers_with_evr(self):
        def provider(A,**kwargs):
            values,U=scipy_eigh(A,driver='ev')
            if kwargs['driver']=='evd': U[:,3]*=1.12
            return values,U
        untouched=self.A.copy()
        with patch.object(implementation,'eigh',side_effect=provider) as calls:
            (values,U),info=validated_eigh(self.A,return_info=True)
        self.assertEqual([call.kwargs['driver'] for call in calls.call_args_list],['evd','evr'])
        self.assertEqual(info['accepted_driver'],'evr')
        self.assertIn('eigenvector columns are not normalized',info['attempts'][0]['issues'])
        assert_allclose((U*values)@U.T,self.A,atol=1e-14)
        assert_array_equal(self.A,untouched)

    def test_unit_column_norms_do_not_hide_cross_orthogonality_failure(self):
        U=self.U.copy()
        U[:,2]=(U[:,2]+.2*U[:,3])/np.sqrt(1.04)
        with patch.object(implementation,'eigh',side_effect=AssertionError('supplied must not retry')):
            with self.assertRaisesRegex(ValueError,'orthogonality') as caught:
                validated_eigh(self.A,supplied=(U,self.values))
        issues=caught.exception.info['attempts'][0]['issues']
        self.assertNotIn('eigenvector columns are not normalized',issues)

    def test_orthogonal_basis_with_wrong_matrix_correspondence_is_rejected(self):
        U=self.U.copy()
        U[:,[0,7]]=U[:,[7,0]]
        with self.assertRaisesRegex(ValueError,'eigen-equation'):
            validated_eigh(self.A,supplied=(U,self.values))

    def test_large_path_targets_norm_defective_internal_columns(self):
        n=160
        values=np.linspace(-1.,2.,n)
        U=np.eye(n)
        U[:,77]*=1.01
        with self.assertRaises(ValueError) as caught:
            validated_eigh(np.diag(values),supplied=(U,values))
        details=caught.exception.info['attempts'][0]
        self.assertEqual(details['validation_kind'],'all_column_norms_and_deterministic_actions')
        self.assertIn(77,details['targeted_columns'])
        self.assertLess(details['probe_count'],20)

    def test_large_unit_norm_cross_defect_is_caught_by_actions(self):
        n=160
        values=np.ones(n)
        U=np.eye(n)
        U[:,77]=(U[:,77]+.1*U[:,53])/np.sqrt(1.01)
        with self.assertRaisesRegex(ValueError,'orthogonality'):
            validated_eigh(np.eye(n),supplied=(U,values))

    def test_provider_exception_and_bad_evx_can_recover_with_ev(self):
        def provider(A,**kwargs):
            if kwargs['driver']=='evr': raise LinAlgError('injected EVR failure')
            values,U=scipy_eigh(A,driver='ev')
            if kwargs['driver']=='evx': U[:,2]*=1.2
            return values,U
        with patch.object(implementation,'eigh',side_effect=provider):
            _,info=validated_eigh(self.A,return_info=True,drivers=('evr','evx','ev'))
        self.assertEqual([a['driver'] for a in info['attempts']],['evr','evx','ev'])
        self.assertEqual(info['accepted_driver'],'ev')
        self.assertIn('LinAlgError',info['attempts'][0]['issues'][0])

    def test_all_driver_failure_is_deterministic_and_informative(self):
        U=self.U.copy()
        U[:,4]*=1.1
        with patch.object(implementation,'eigh',return_value=(self.values,U)):
            with self.assertRaises(EigenDecompositionError) as caught:
                validated_eigh(self.A)
        error=caught.exception
        self.assertEqual([a['driver'] for a in error.attempts],['evd','evr','evx'])
        self.assertIn('evd:',str(error))
        self.assertIn('evr:',str(error))
        self.assertIn('evx:',str(error))
        self.assertIn('BLAS/LAPACK runtime',str(error))
        self.assertIn('not normalized',str(error))

    def test_eigenvalues_only_retries_inconsistent_spectral_moments(self):
        def provider(A,**kwargs):
            self.assertTrue(kwargs['eigvals_only'])
            values=self.values.copy()
            if kwargs['driver']=='evd': values[0]=-100.
            return values
        with patch.object(implementation,'eigh',side_effect=provider):
            values,info=validated_eigh(self.A,eigvals_only=True,return_info=True)
        assert_array_equal(values,self.values)
        self.assertEqual(info['accepted_driver'],'evr')
        self.assertFalse(info['attempts'][-1]['eigenvectors_validated'])
        self.assertEqual(info['attempts'][-1]['validation_kind'],'spectral_invariants_only')

    def test_supplied_values_only_still_checks_available_vectors(self):
        U=self.U.copy()
        U[:,1]*=1.1
        with self.assertRaisesRegex(ValueError,'Supplied eigenpairs'):
            validated_eigh(self.A,supplied=(U,self.values),eigvals_only=True)

    def test_invalid_input_does_not_call_provider(self):
        cases=(np.zeros((2,3)),np.array([[1.,.1],[0.,1.]]),
               np.diag([np.nan,1.]),np.eye(2).astype(complex))
        with patch.object(implementation,'eigh',side_effect=AssertionError('invalid input must reject early')):
            for matrix in cases:
                with self.subTest(matrix=matrix),self.assertRaises(ValueError):
                    validated_eigh(matrix)

    def test_computed_result_is_sorted_without_changing_eigenpairs(self):
        order=np.arange(8)[::-1]
        with patch.object(implementation,'eigh',return_value=(self.values[order],self.U[:,order])):
            values,U=validated_eigh(self.A)
        assert_array_equal(values,self.values)
        assert_array_equal(U,self.U)

    def test_linear_helper_and_supplied_path_share_validation(self):
        X=np.array([[-2.,1.],[-1.,0.],[1.,1.],[2.,0.]])
        y=np.array([-1.,-1.,1.,1.])
        augmented=np.column_stack((np.ones(4),X))
        for features in (X,csr_matrix(X)):
            U,D=get_P0_eig(features)
            self.assertTrue(U.flags.f_contiguous)
            self.assertTrue(np.all(np.diff(D)<=0))
            assert_allclose((U*D)@U.T,augmented.T@augmented,atol=1e-13)
            supplied=(U.copy(),D.copy())
            supplied[0][:,1]*=1.1
            with self.assertRaisesRegex(ValueError,'Supplied eigenpairs'):
                solve_gen_dwd(features,y,lambd=.1,P0_eig=supplied,max_iter=2)
            reference=solve_gen_dwd(features,y,lambd=.1,max_iter=2,obj_tol=0.)
            cached=solve_gen_dwd(features,y,lambd=.1,P0_eig=(U,D),max_iter=2,obj_tol=0.)
            assert_allclose(cached[0],reference[0],atol=1e-14)
            assert_allclose(cached[1],reference[1],atol=1e-14)

    def test_private_driver_order_and_configuration_validation(self):
        with patch.object(implementation,'eigh',wraps=scipy_eigh) as call:
            _,info=validated_eigh(self.A,drivers=('evx','evr','ev'),return_info=True)
        self.assertEqual(call.call_args.kwargs['driver'],'evx')
        self.assertEqual(info['accepted_driver'],'evx')
        for drivers in ((),('bad',),('evr','evr'),['evx'],'evx'):
            with self.subTest(drivers=drivers),self.assertRaisesRegex(ValueError,'drivers'):
                validated_eigh(self.A,drivers=drivers)
        with patch.object(implementation,'eigh',side_effect=LinAlgError('native driver failed')) as call:
            with self.assertRaises(EigenDecompositionError) as caught:
                validated_eigh(self.A,drivers=('evx',))
        self.assertEqual(call.call_count,1)
        self.assertEqual(caught.exception.info['configured_drivers'],['evx'])


if __name__=='__main__': unittest.main(verbosity=2)
