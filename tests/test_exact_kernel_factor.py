"""Portable bounded exact stored-kernel certificate controls."""
from pathlib import Path
from fractions import Fraction as Q
from unittest.mock import patch
import ast
import hashlib
import os
import subprocess
import sys
import unittest

import numpy as np
import dwd._exact_kernel_factor as ekf


def independent_identity(K, factor):
    for i in range(len(K)):
        for j in range(len(K)):
            value = sum((Q(float(factor.C[i,a]))*factor.B_inverse_exact[a][b]
                         *Q(float(factor.C[j,b]))
                         for a in range(factor.rank) for b in range(factor.rank)), Q(0))
            if value != Q(float(K[i,j])):
                return False
    return True


class ExactKernelFactorTests(unittest.TestCase):
    def test_manufactured_rank_zero_one_five_and_full(self):
        F = np.vstack([np.eye(5), np.array([[1,2,3,4,5],[-1,0,2,0,1],[2,-1,0,1,2],[0,2,0,-1,1]])])/8
        vector = np.array([.125, .25, .5, -1.])
        matrices = [(np.zeros((7,7)),0),(np.outer(vector,vector),1),
                    (F@F.T,5),(np.eye(6)+np.ones((6,6))*.25,6)]
        for K, rank in matrices:
            with self.subTest(rank=rank):
                before = K.tobytes()
                result = ekf.exact_kernel_factor(K)
                self.assertTrue(result.certified, result)
                self.assertEqual(result.factor.rank, rank)
                self.assertTrue(independent_identity(K, result.factor))
                self.assertTrue(result.factor.validate(K))
                self.assertFalse(result.factor.C.flags.writeable)
                self.assertEqual(before,K.tobytes())

    def test_tiny_positive_direction_retained_without_threshold(self):
        K=np.diag([2.**-100,1.,2.])
        result=ekf.exact_kernel_factor(K)
        self.assertTrue(result.certified)
        self.assertEqual(result.factor.rank,3)
        self.assertIn(0,result.factor.pivots)
        self.assertTrue(independent_identity(K,result.factor))

    def test_native_float64_shape_finite_and_exact_symmetry_contract(self):
        for value,status in [(np.eye(2,dtype=np.float32),'requires_native_float64'),
                             ([[1.]],'invalid_matrix_shape_or_type'),
                             (np.ones(2),'invalid_matrix_shape_or_type'),
                             (np.ones((2,3)),'invalid_matrix_shape_or_type'),
                             (np.eye(2,dtype=complex),'requires_native_float64'),
                             (np.array([[np.inf]]),'nonfinite_matrix'),
                             (np.array([[np.nan]]),'nonfinite_matrix'),
                             (np.array([[1.,.5],[0.,1.]]),'not_exactly_symmetric')]:
            result=ekf.exact_kernel_factor(value)
            self.assertFalse(result.certified)
            self.assertIsNone(result.factor)
            self.assertEqual(result.status,status)

    def test_non_psd_is_no_certificate_and_no_partial_model(self):
        for K,status in [(np.diag([1.,-1.]),'exact_negative_schur_diagonal'),
                         (np.array([[0.,1.],[1.,0.]]),'zero_diagonal_nonzero_schur')]:
            result=ekf.exact_kernel_factor(K)
            self.assertEqual(result.status,status)
            self.assertIsNone(result.factor)

    def test_budget_types_checked_before_matrix_access(self):
        for name in ('max_entries','max_rank','max_bits','max_operations'):
            for bad in (-1,True,1.5,'2',None,np.inf):
                with self.subTest(name=name,value=bad),self.assertRaises(ValueError):
                    ekf.exact_kernel_factor(object(),**{name:bad})

    def test_entry_budget_precedes_finite_scan_or_conversion(self):
        K=np.empty((20,20))
        with patch.object(ekf.np,'isfinite',side_effect=AssertionError('finite scan ran')):
            result=ekf.exact_kernel_factor(K,max_entries=399)
        self.assertEqual(result.status,'entry_budget_exhausted')
        self.assertEqual(result.metadata['arithmetic']['operations'],0)
        self.assertIsNone(result.factor)

    def test_rank_bit_and_zero_work_exits_are_inapplicable(self):
        cases=[(np.eye(4),dict(max_rank=3),'rank_budget_exhausted'),
               (np.diag([2.**-100,1.]),dict(max_bits=64),'rational_bit_budget_exhausted'),
               (np.eye(4),dict(max_operations=0),'operation_budget_exhausted')]
        for K,options,status in cases:
            result=ekf.exact_kernel_factor(K,**options)
            self.assertFalse(result.certified)
            self.assertIsNone(result.factor)
            self.assertEqual(result.status,status)

    def test_identity_and_inverse_consume_the_same_work_budget(self):
        K=np.eye(4)+np.ones((4,4))*.25
        full=ekf.exact_kernel_factor(K)
        a=full.metadata['arithmetic_after_schur']['operations']
        b=full.metadata['arithmetic_after_ldlt_identity']['operations']
        c=full.metadata['arithmetic_after_inverse_identity']['operations']
        self.assertLess(a,b);self.assertLess(b,c)
        for limit in (a+1,b+1):
            outcome=ekf.exact_kernel_factor(K,max_operations=limit)
            self.assertEqual(outcome.status,'operation_budget_exhausted')
            self.assertIsNone(outcome.factor)
            self.assertEqual(outcome.metadata['arithmetic']['operations'],limit)

    def test_unreduced_intermediate_bit_growth_is_checked_before_arithmetic(self):
        value=Q(2**40+1,2**40-1)
        inverse=1/value
        for operation,other in [('mul',inverse),('div',value),('add',-value),('sub',value),('lt',inverse)]:
            arithmetic=ekf.ExactArithmetic(max_bits=64)
            with self.subTest(operation=operation),self.assertRaises(ekf.BudgetExhausted):
                getattr(arithmetic,operation)(value,other)

    def test_inexact_ldlt_injection_is_rejected_explicitly(self):
        K=np.outer(np.array([1.,2.,3.]),np.array([1.,2.,3.]))
        original=ekf._verify_ldlt
        def corrupted(matrix,columns,diagonals,pivots,arithmetic):
            columns=[column.copy() for column in columns]
            columns[0][0]+=1
            return original(matrix,columns,diagonals,pivots,arithmetic)
        with patch.object(ekf,'_verify_ldlt',side_effect=corrupted):
            result=ekf.exact_kernel_factor(K)
        self.assertFalse(result.certified)
        self.assertIsNone(result.factor)
        self.assertEqual(result.status,'exact_ldlt_or_triangular_identity_failed')

    def test_inexact_inverse_identity_injection_is_rejected(self):
        arithmetic=ekf.ExactArithmetic()
        original=arithmetic.dot
        def corrupted(a,b):
            return arithmetic.add(original(a,b),arithmetic.integer(1))
        with patch.object(arithmetic,'dot',side_effect=corrupted),self.assertRaisesRegex(ekf.CertificateFailure,'inverse_identity'):
            ekf.exact_inverse(((Q(2),Q(1)),(Q(1),Q(2))),arithmetic)

    def test_noncontiguous_input_and_factor_mutation_are_detected(self):
        K=np.array([[2.,1.],[1.,2.]],order='F')
        result=ekf.exact_kernel_factor(K)
        self.assertTrue(result.factor.validate(K))
        changed=K.copy();changed[0,0]+=1
        self.assertFalse(result.factor.validate(changed))
        result.factor.C.setflags(write=True)
        result.factor.C[0,0]+=1
        self.assertFalse(result.factor.validate(K))

    def test_python_optimized_mode_cannot_disable_certificate_checks(self):
        script="""import sys
sys.path.insert(0,sys.argv[1])
import numpy as np
import dwd._exact_kernel_factor as m
m._verify_ldlt=lambda *args:False
r=m.exact_kernel_factor(np.eye(2))
if r.certified or r.factor is not None or r.status!='exact_ldlt_or_triangular_identity_failed':
 raise SystemExit(7)
print('optimized-mode rejection passed')
"""
        completed=subprocess.run([sys.executable,'-I','-B','-O','-c',script,str(Path(ekf.__file__).resolve().parents[1])],
                                 capture_output=True,text=True,timeout=30,creationflags=0x08000000 if os.name=='nt' else 0)
        self.assertEqual(completed.returncode,0,completed.stdout+completed.stderr)
        self.assertIn('rejection passed',completed.stdout)
        source=ast.parse(Path(ekf.__file__).read_text(encoding='utf-8'))
        self.assertFalse(any(isinstance(node,ast.Assert) for node in ast.walk(source)))

    def test_previously_certified_dyadic_rank_five_matrix_is_portable(self):
        n=30
        F=np.random.RandomState(91300+n).randint(-3,4,(n,5)).astype(float)/8.
        K=F@F.T
        self.assertEqual(hashlib.sha256(K.tobytes()).hexdigest(),
            'eb13eba26fcbf855af04261aa2be6eb24481e063b6c35a1272c9972821676c14')
        result=ekf.exact_kernel_factor(K)
        self.assertTrue(result.certified)
        self.assertEqual(result.factor.rank,5)
        self.assertTrue(independent_identity(K,result.factor))
        self.assertTrue(result.factor.validate(K))


if __name__=='__main__':
    unittest.main(verbosity=2)
