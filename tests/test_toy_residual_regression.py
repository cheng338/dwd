"""Previously rejected 150-row input, with an independent Decimal80 oracle.

Only manufactured training data are used. No validation scores are evaluated.
The oracle uses pivoted bordered LU and high-precision original-equation
residual refinement, independent of both package inverse implementations.
"""
from decimal import Decimal, localcontext
import math
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import lu_factor, lu_solve
from sklearn.datasets import make_classification
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import dwd._kernel_linear_system as linear
from dwd.gen_kern_dwd import KernGDWD


class DecimalOracle:
    def __init__(self, K, shift):
        self.K = [[Decimal.from_float(float(k)) for k in row] for row in K]
        self.shift = Decimal.from_float(float(shift))
        n = len(K)
        matrix = np.block([[K + shift*np.eye(n), np.ones((n, 1))],
                           [np.ones((1, n)), np.zeros((1, 1))]])
        self.factor = lu_factor(matrix)

    def product(self, x):
        with localcontext() as context:
            context.prec = 80
            dx = [Decimal.from_float(float(v)) for v in x]
            return np.array([float(sum((k*v for k, v in zip(row, dx)), Decimal(0))) for row in self.K])

    def residual(self, rhs, x, s, target=0.):
        with localcontext() as context:
            context.prec = 80
            dx = [Decimal.from_float(float(v)) for v in x]
            ds = Decimal.from_float(float(s))
            residual = np.array([float(Decimal.from_float(float(r))-ds-self.shift*xi
                            - sum((k*v for k, v in zip(row, dx)), Decimal(0)))
                         for row, r, xi in zip(self.K, rhs, dx)])
            constraint = float(Decimal.from_float(float(target))-sum(dx, Decimal(0)))
            return residual, constraint

    def solve(self, rhs, target=0.):
        result = lu_solve(self.factor, np.r_[rhs, target])
        for _ in range(3):
            residual, constraint = self.residual(rhs, result[:-1], result[-1], target)
            result += lu_solve(self.factor, np.r_[residual, constraint])
        residual, constraint = self.residual(rhs, result[:-1], result[-1], target)
        return result[:-1], float(result[-1]), residual, constraint


def q1_path(oracle, y, lambd, initial, initial_b, count):
    alpha, b, t = initial.copy(), float(initial_b), len(y)/4.
    states = []
    for iteration in range(count+1):
        score = oracle.product(alpha)
        margin = y*(score+b)
        loss = 1.-margin
        mask = margin > .5
        loss[mask] = .25/margin[mask]
        objective = float(np.mean(loss)+lambd*math.fsum(alpha*score))
        states.append((score+b, objective))
        if iteration == count:
            break
        derivative = -np.ones(len(y))
        derivative[mask] = -.25/margin[mask]**2
        alpha, s, residual, constraint = oracle.solve(score-t*y*derivative/len(y))
        if np.max(abs(residual)) > 1e-10*max(1., np.max(abs(score-t*y*derivative/len(y)))):
            raise AssertionError('Independent oracle did not meet the original residual criterion')
        b += s
    return states


class ToyResidualRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.thread_limit = threadpool_limits(4)
        X, y = make_classification(n_samples=300, n_features=7, n_informative=4,
                                  class_sep=2., flip_y=0., random_state=712)
        y = np.where(y, 1, -1)
        tr, _ = list(StratifiedKFold(2, shuffle=True, random_state=2718).split(X, y))[0]
        cls.X, cls.y = StandardScaler().fit_transform(X[tr]), y[tr]
        cls.K = rbf_kernel(cls.X, gamma=.001)
        cls.oracle = DecimalOracle(cls.K, 2.*1e-7*(len(cls.K)/4.))

    @classmethod
    def tearDownClass(cls):
        cls.thread_limit.restore_original_limits()

    def fit(self, implementation, initialization, supplied=True, count=100, offset=0.):
        states = []
        model = KernGDWD(kernel='rbf', kernel_kws={'gamma': .001}, lambd=1e-7,
            implementation=implementation, initialization=initialization, random_state=314159,
            stopping='objective', obj_tol=1e-5, max_iter=count,
            callback=lambda state: states.append(dict(state)))
        model.fit(self.X, self.y, offset_init=offset, **({'K': self.K} if supplied else {}))
        self.assertEqual(model.n_iter_, count)
        self.assertEqual([s['iteration'] for s in states], list(range(count+1)))
        self.assertTrue(np.isfinite(model.final_objective_))
        self.assertTrue(all(np.isfinite(s['decision_values']).all() for s in states))
        return model, states

    def test_zero_initialized_full_trajectories_match_independent_high_precision_oracle(self):
        expected = q1_path(self.oracle, self.y, 1e-7, np.zeros(len(self.K)), 0., 100)
        for implementation, supplied in [('optimized', False), ('optimized', True), ('reference', True)]:
            with self.subTest(implementation=implementation, supplied=supplied):
                if implementation == 'reference':
                    with patch.object(linear, 'cho_factor', side_effect=AssertionError('Reference must remain spectral')):
                        model, states = self.fit(implementation, 'zero', supplied)
                else:
                    model, states = self.fit(implementation, 'zero', supplied)
                assert_allclose([s['decision_values'] for s in states], [s[0] for s in expected], rtol=0., atol=2e-8)
                assert_allclose([s['objective'] for s in states], [s[1] for s in expected], rtol=0., atol=3e-10)
                diagnostics = model.diagnostics_['linear_system_diagnostics']
                self.assertGreater(diagnostics['compensated_residual_checks'], 0)
                self.assertEqual(diagnostics['positive_eigenvalues_discarded'], 0)
                self.assertEqual(diagnostics['added_objective_regularization'], 0)

    def test_native_gaussian_initialization_and_free_intercept_are_preserved(self):
        initial = np.random.RandomState(314159).normal(size=len(self.K))
        initial /= np.linalg.norm(initial)
        model, states = self.fit('reference', 'auto', count=100, offset=.17)
        assert_array_equal(states[0]['alpha'], initial)
        self.assertEqual(states[0]['offset'], .17)
        # Independent first ten updates check the nonzero constrained direction
        # and intercept signs; all 100 package updates must also finish.
        expected = q1_path(self.oracle, self.y, 1e-7, initial, .17, 10)
        assert_allclose([s['decision_values'] for s in states[:11]], [s[0] for s in expected], rtol=0., atol=2e-8)
        assert_allclose([s['objective'] for s in states[:11]], [s[1] for s in expected], rtol=0., atol=3e-10)

    def test_first_accepted_model_passes_independent_original_equation_residual(self):
        rhs = self.y/4.
        for implementation in ('optimized', 'reference'):
            with self.subTest(implementation=implementation):
                _, states = self.fit(implementation, 'zero', count=1)
                state = states[-1]
                residual, constraint = self.oracle.residual(rhs, state['alpha'], state['offset'])
                self.assertLessEqual(np.max(abs(residual)), 1e-10)
                self.assertLessEqual(abs(constraint), 64*np.finfo(float).eps*max(1., math.fsum(abs(state['alpha']))))


if __name__ == '__main__':
    unittest.main(verbosity=2)
