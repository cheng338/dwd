"""Exact mask equivalence and safe-dispatch tests for RBF guard scanning."""
import math
import unittest
from unittest.mock import patch
import numpy as np
from dwd import _guarded_rbf as guard


def reference(distances, x_norm, y_norm, p):
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        envelope = (guard._factor(p)/math.sqrt(guard._EPS))*(x_norm[:,None]+y_norm[None,:])
        envelope += (2*p+3)*guard._TINY/math.sqrt(guard._EPS)
        flags = ~np.isfinite(distances) | ~np.isfinite(envelope) | (distances <= envelope)
        return np.packbits(flags, axis=1)


def assert_masks(distances, x_norm, y_norm, p):
    before = [a.copy() for a in (distances,x_norm,y_norm)]
    expected = reference(distances,x_norm,y_norm,p)
    with patch.object(guard,'_all_clear',return_value=False):
        actual = guard._packed_flags(distances,x_norm,y_norm,p)
    np.testing.assert_array_equal(actual,expected)
    for a,b in zip(before,(distances,x_norm,y_norm)):
        np.testing.assert_array_equal(a,b)


class GuardedScanCandidateTests(unittest.TestCase):
    def test_sparse_candidates_at_tile_byte_and_envelope_boundaries(self):
        rng = np.random.default_rng(194)
        xn=rng.uniform(.1,3,513); yn=rng.uniform(.1,3,519); p=17
        D=np.full((513,519),20.)
        for i,j in [(0,0),(0,7),(7,8),(254,255),(255,256),(256,255),(256,257),(511,512),(512,518)]:
            envelope=(guard._factor(p)/math.sqrt(guard._EPS))*(xn[i]+yn[j])
            envelope+=(2*p+3)*guard._TINY/math.sqrt(guard._EPS)
            for value in [np.nextafter(envelope,-np.inf),envelope,np.nextafter(envelope,np.inf),np.nan,np.inf,-np.inf]:
                D[i,j]=value
                assert_masks(D,xn,yn,p)
        wide=np.empty((513,1038));wide[:,::2]=D;wide[:,1::2]=-17
        assert_masks(wide[:,::2],xn,yn,p)
        assert_masks(np.asfortranarray(D),xn,yn,p)
    
    
    def test_extreme_zero_norms_and_fallback_inputs(self):
        for dtype in (np.float64,np.float32):
            for norms in ([0.,-0.,np.nextafter(0.,1.),1e-310], [1.,-1.,3.,4.],
                          [1.,np.inf,3.,4.], [1.,np.nan,3.,4.], [1e308]*4):
                xn=np.asarray(norms,dtype=dtype);yn=xn.copy()
                D=np.array([[0.,-0.,np.nextafter(0.,1.),np.inf],
                            [np.nan,-np.inf,1.,2.],[0.,1.,2.,3.],[0.,0.,0.,0.]])
                assert_masks(D,xn,yn,1000)
        xn=np.ones(4)
        assert_masks(np.eye(4),xn,xn,2**51)
    
    
    def test_finite_predicate_avoids_redundant_finiteness_arrays(self):
        xn=np.ones(257);yn=np.ones(259)
        D=np.full((257,259),2.);D[0,0]=0.;D[-1,-1]=-np.inf
        expected=reference(D,xn,yn,32)
        with patch.object(guard,'_all_clear',return_value=False), patch.object(guard.np,'isfinite',side_effect=AssertionError('Redundant finite scan')):
            actual=guard._packed_flags(D,xn,yn,32)
        np.testing.assert_array_equal(actual,expected)
    
    
    def test_self_identity_only_selects_scan_strategy(self):
        n=513;norms=np.ones(n);D=np.full((n,n),2.);np.fill_diagonal(D,0.)
        expected=reference(D,norms,norms,32)
        original_pack=np.packbits
        with patch.object(guard,'_all_clear',return_value=False), patch.object(guard.np,'packbits',wraps=original_pack) as calls:
            actual=guard._packed_flags(D,norms,norms,32)
        np.testing.assert_array_equal(actual,expected)
        assert calls.call_count==3  # Six off-diagonal tiles satisfy the same bound.
        with patch.object(guard,'_all_clear',return_value=False), patch.object(guard.np,'packbits',wraps=original_pack) as calls:
            copied=guard._packed_flags(D,norms,norms.copy(),32)
        np.testing.assert_array_equal(copied,expected)
        assert calls.call_count==9
    
    
    def test_nonfinite_distance_retains_the_full_predicate(self):
        xn=np.ones(257);yn=xn.copy();D=np.full((257,257),2.);D[0,0]=np.nan
        expected=reference(D,xn,yn,32)
        original_finite=np.isfinite
        with patch.object(guard,'_all_clear',return_value=False),patch.object(guard.np,'isfinite',wraps=original_finite) as calls:
            actual=guard._packed_flags(D,xn,yn,32)
        np.testing.assert_array_equal(actual,expected)
        assert calls.call_count > 0
    