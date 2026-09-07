"""Independent function-space MM checks, with no external files or data."""
from decimal import Decimal as D, localcontext
from fractions import Fraction as Q
import unittest
from unittest.mock import patch

import numpy as np

import dwd._certified_kernel_mm as mm
from dwd._exact_kernel_factor import exact_kernel_factor, BudgetExhausted


def dec(x):
    return D.from_float(float(x))


def dot(a, b):
    return sum((x*y for x, y in zip(a, b)), D(0))


def solve_decimal(matrix, rhs):
    """Independent pivoted elimination, not the implementation's inverse."""
    n = len(matrix)
    rows = [list(row)+[rhs[i]] for i, row in enumerate(matrix)]
    for j in range(n):
        p = max(range(j, n), key=lambda i: abs(rows[i][j]))
        rows[j], rows[p] = rows[p], rows[j]
        if not rows[j][j]:
            raise ArithmeticError('Singular independent oracle.')
        for i in range(j+1, n):
            multiplier = rows[i][j]/rows[j][j]
            for k in range(j, n+1):
                rows[i][k] -= multiplier*rows[j][k]
    result = [D(0)]*n
    for i in range(n-1, -1, -1):
        result[i] = (rows[i][-1]-dot(rows[i][i+1:n], result[i+1:]))/rows[i][i]
    return result


def ideal(F, a, delta):
    """Uncentered rank+1 primal system; F is available to the test only."""
    rows = [list(map(dec, row)) for row in F]
    columns = list(zip(*rows))
    rank = F.shape[1]
    gram = [[dot(x, y)+(dec(delta) if i == j else D(0))
             for j, y in enumerate(columns)] for i, x in enumerate(columns)]
    sums = [sum(x, D(0)) for x in columns]
    matrix = [row+[s] for row, s in zip(gram, sums)]+[sums+[D(len(a))]]
    aq = list(map(dec, a))
    solution = solve_decimal(matrix, [dot(x, aq) for x in columns]+[sum(aq, D(0))])
    return solution[:rank], solution[rank], rows, columns


def certify(K):
    result = exact_kernel_factor(K)
    if not result.certified:
        raise ArithmeticError(str(result.metadata))
    return result.factor


class CertifiedKernelMMTests(unittest.TestCase):
    def check_oracle(self, F, K, delta, a, oldb, alpha, newb, scores):
        with localcontext() as context:
            context.prec = 100
            w, increment, rows, columns = ideal(F, a, delta)
            actual_w = [dot(col, list(map(dec, alpha))) for col in columns]
            error = dot([x-y for x, y in zip(actual_w, w)],
                        [x-y for x, y in zip(actual_w, w)]).sqrt()
            self.assertLess(error, D('1e-11'))
            expected_b = dec(oldb)+increment
            represented = [dot(row, actual_w) for row in rows]
            expected = [dot(row, w)+expected_b for row in rows]
            original = [dot(list(map(dec, row)), list(map(dec, alpha))) for row in K]
            self.assertEqual(original, represented)
            self.assertLess(max(abs(g+dec(newb)-e) for g, e in zip(original, expected)), D('1e-10'))
            self.assertLess(max(abs(dec(g)-e) for g, e in zip(scores, original)), D('1e-12'))
            self.assertLess(abs(sum((g+dec(newb)-dec(oldb)-dec(x)
                                    for g, x in zip(original, a)), D(0))/D(len(a))), D('1e-12'))

    def test_previously_rejected_rank5_kernels_zero_and_native_initial_states(self):
        # The exact dyadic generator of the preserved public failure cases.
        # K=F F.T is verified independently, not inferred from eigenvalue size.
        for n, delta in ((30, 1e-14), (75, 1e-10), (120, 1e-14)):
            rng = np.random.RandomState(91300+n)
            F = rng.randint(-3, 4, size=(n, 5)).astype(float)/8.
            K = F@F.T
            fq = [[Q(x) for x in row] for row in F]
            self.assertTrue(all(Q(K[i, j]) == sum((x*y for x, y in zip(fq[i], fq[j])), Q(0))
                                for i in range(n) for j in range(n)))
            factor = certify(K)
            for initialization in ('zero', 'normalized_gaussian'):
                alpha = np.zeros(n) if initialization == 'zero' else np.random.RandomState(314159).normal(size=n)
                if initialization != 'zero':
                    alpha /= np.linalg.norm(alpha)
                preserved = alpha.copy()
                b = 0.; scores = K@alpha
                helper = mm.CertifiedKernelMM(K, factor, delta)
                y = np.where(np.arange(n)%2 == 0, -1., 1.)
                for _ in range(2):
                    margins = y*(scores+b)
                    derivative = np.full(n, -1.)
                    tail = margins > .5
                    derivative[tail] = -(.5/margins[tail])**2
                    a = scores-(n/4)*(y*derivative/n)
                    aa, bb, gg, info = helper.step(a, b)
                    self.check_oracle(F, K, delta, a, b, aa, bb, gg)
                    self.assertTrue(info['true_MM_function_checked'])
                    self.assertFalse(info['reduced_A_gauge_checked'])
                    self.assertTrue(info['absolute_offset_checked'])
                    self.assertLessEqual(np.count_nonzero(aa), 5)
                    alpha, b, scores = aa, bb, gg
                # A new sparse representative is returned; initialization is
                # never mutated by the action.
                if initialization == 'zero':
                    np.testing.assert_array_equal(preserved, np.zeros(n))

    def test_general_q_actual_delta_and_nonzero_free_offset(self):
        F = np.array([[1., 0.], [0., 1.], [1., 1.], [-1., 2.], [2., -1.], [-1., -1.]])
        K = F@F.T; factor = certify(K); n = len(K)
        y = np.array([-1., 1., -1., 1., -1., 1.])
        for q in (1, 2, 5):
            alpha = np.array([.125, -.25, .0625, .125, 0., -.125])
            scores = K@alpha; oldb = .375
            t = n*q/(q+1.)**2; lambd = .03; delta = 2*lambd*t
            margins = y*(scores+oldb)
            threshold = q/(q+1.)
            derivative = np.full(n, -1.)
            tail = margins > threshold
            derivative[tail] = -(threshold/margins[tail])**(q+1)
            a = scores-t*y*derivative/n
            aa, bb, gg, info = mm.CertifiedKernelMM(K, factor, delta).step(a, oldb)
            self.check_oracle(F, K, delta, a, oldb, aa, bb, gg)
            with localcontext() as context:
                context.prec = 100
                w = [dot(list(map(dec, col)), list(map(dec, aa))) for col in F.T]
                exact_margins = [dec(yy)*(dot(list(map(dec, row)), w)+dec(bb)) for row, yy in zip(F, y)]
                def loss(u):
                    return 1-u if u <= D(q)/D(q+1) else (D(q)/(D(q+1)*u))**q/D(q+1)
                exactJ = sum(map(loss, exact_margins), D(0))/D(n)+dec(lambd)*dot(w, w)
                numericalJ = np.mean([float(loss(dec(v))) for v in y*(gg+bb)])+lambd*info['rkhs_norm_squared']
                self.assertLess(abs(dec(numericalJ)-exactJ), D('1e-13'))

    def test_rank_zero_and_constant_features(self):
        for K in (np.zeros((5, 5)), np.full((5, 5), 4.)):
            a = np.array([1., -2., 4., 3., 0.]); oldb = 2.25
            helper = mm.CertifiedKernelMM(K, certify(K), .01)
            alpha, b, scores, info = helper.step(a, oldb)
            np.testing.assert_array_equal(alpha, np.zeros(5))
            np.testing.assert_array_equal(scores, np.zeros(5))
            self.assertEqual(b, float(Q(oldb)+sum(map(Q, a), Q(0))/5))

    def test_offset_addition_failure_not_hidden_by_large_offset_scale(self):
        K = np.zeros((4, 4)); helper = mm.CertifiedKernelMM(K, certify(K), 1.)
        with self.assertRaises(mm.MMFunctionAccuracyError) as caught:
            helper.step(np.ones(4), 1e20)
        self.assertFalse(caught.exception.info['attempts'][-1]['gates']['free_offset_stationarity'])

    def test_inaccurate_native_has_exactly_one_checked_exact_fallback(self):
        K = np.eye(4); helper = mm.CertifiedKernelMM(K, certify(K), .02)
        with patch.object(helper, '_native_beta', return_value=np.full(4, 1e8)) as call:
            _, _, _, info = helper.step(np.array([2., -1., 3., 0.]), .3)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(info['accepted_candidate'], 'rounded_exact')
        self.assertEqual(len(info['attempts']), 2)
        self.assertFalse(info['attempts'][0]['accepted'])
        self.assertTrue(all(info['attempts'][1]['gates'].values()))

    def test_both_inaccurate_candidates_raise_without_unbounded_retry(self):
        K = np.eye(4); helper = mm.CertifiedKernelMM(K, certify(K), .02)
        measure = helper._assess
        def corrupt(beta, *args, **kwargs):
            return measure(np.full(4, 1e8), *args, **kwargs)
        with patch.object(helper, '_assess', side_effect=corrupt) as call:
            with self.assertRaises(mm.MMFunctionAccuracyError):
                helper.step(np.ones(4), 0.)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(helper.steps, 0)

    def test_nonfinite_native_is_discarded(self):
        K = np.eye(3); helper = mm.CertifiedKernelMM(K, certify(K), .02)
        with patch.object(helper, '_native_beta', return_value=np.full(3, np.nan)):
            _, _, _, info = helper.step(np.array([1., 0., -1.]), 0.)
        self.assertEqual(info['accepted_candidate'], 'rounded_exact')

    def test_native_overflow_or_nonpositive_eigenvalues_skips_only_native(self):
        K = np.full((1, 1), 2.**600)
        helper = mm.CertifiedKernelMM(K, certify(K), 2.**600)
        self.assertFalse(helper.info['native_preparation']['available'])
        self.assertEqual(helper.step(np.array([2.]), 3.)[1], 5.)
        K = np.eye(2)
        with patch.object(mm, 'validated_eigh', return_value=((np.array([-1., 1.]), np.eye(2)), {})):
            helper = mm.CertifiedKernelMM(K, certify(K), .01)
        self.assertFalse(helper.info['native_preparation']['available'])
        self.assertEqual(helper.step(np.array([1., -1.]), 0.)[3]['accepted_candidate'], 'rounded_exact')

    def test_invalid_input_and_exhaustion_do_not_enter_recovery(self):
        K = np.eye(3); factor = certify(K)
        for kwargs in ({'max_entries': 8}, {'max_rank': 2}, {'max_operations': 0}, {'max_bits': 0}):
            with patch.object(mm, 'validated_eigh', side_effect=AssertionError('unexpected native preparation')):
                with self.assertRaises((ValueError, BudgetExhausted)):
                    mm.CertifiedKernelMM(K, factor, .1, **kwargs)
        helper = mm.CertifiedKernelMM(K, factor, .1)
        for a, b in (([1., 2.], 0.), ([1., np.nan, 0.], 0.), ([1j, 2., 3.], 0.), ([1., 2., 3.], np.inf)):
            with self.assertRaises(ValueError):
                helper.step(a, b)
        for delta in (0., -1., np.nan, np.inf, 1j, True):
            with self.assertRaises(ValueError):
                mm.CertifiedKernelMM(K, factor, delta)
        helper.max_operations = 1
        with self.assertRaises(BudgetExhausted):
            helper.step(np.ones(3), 0.)
        self.assertEqual(helper.steps, 0)

    def test_changed_K_or_factor_cannot_reuse_preparation(self):
        K = np.eye(3); factor = certify(K); helper = mm.CertifiedKernelMM(K, factor, .1)
        K[0, 0] = 2.
        with self.assertRaises(ValueError):
            helper.step(np.ones(3), 0.)
        K[0, 0] = 1.
        factor.C.setflags(write=True); factor.C[0, 0] += 1.
        with self.assertRaises(ValueError):
            helper.step(np.ones(3), 0.)


if __name__ == '__main__':
    unittest.main()
