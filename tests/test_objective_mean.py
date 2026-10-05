"""Objective reduction preserves ordinary arithmetic and recovers finite means."""
from decimal import Decimal, localcontext
import importlib
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from numpy.testing import assert_array_equal
from threadpoolctl import threadpool_limits

from dwd._accelerated_mm import RestartedMM
from dwd._objective_mean import _fallback, objective_mean
from dwd.gen_dwd import GenDWD
from dwd.gen_kern_dwd import KernGDWD


def decimal_objective(losses, penalty):
    """Independent reference for the represented binary64 inputs."""
    with localcontext() as context:
        context.prec = 2000
        total = sum((Decimal.from_float(float(value)) for value in losses.flat),
                    Decimal(0))
        return float(total / Decimal(losses.size)
                     + Decimal.from_float(float(penalty)))


class ObjectiveMeanTests(unittest.TestCase):
    def assert_same_float(self, actual, expected):
        self.assertEqual(float(actual).hex(), float(expected).hex())

    def test_representable_overflow_matches_independent_decimal_reference(self):
        largest = np.finfo(np.float64).max
        smallest = np.nextafter(0., 1.)
        cases = [
            (np.array([0., 0., 1e308, 1e308]), 0.),
            (np.array([0., 0., 1e308, 1e308]), 1e308),
            (np.full(7, largest), 0.),
            (np.array([smallest, 1e308, 1e308]), 0.),
            (np.array([1e308, 1e308, -1e308, -1e308, 3.]), .25),
            (np.array([1e308, 1e308, -1e308, -1e308, smallest]), 0.),
            (np.array([0., 1e308, 0., 1e308])[::-1], .25),
        ]
        for losses, penalty in cases:
            expected = decimal_objective(losses, penalty)
            for inside in (False, True):
                with np.errstate(over='ignore', invalid='ignore', under='ignore'):
                    previous = (np.mean(losses + penalty) if inside else
                                np.mean(losses) + penalty)
                self.assertFalse(np.isfinite(previous))
                for policy in ('warn', 'raise'):
                    with self.subTest(losses=losses, penalty=penalty,
                                      inside=inside, policy=policy):
                        with warnings.catch_warnings(record=True) as caught, \
                                np.errstate(over=policy, invalid=policy, under='ignore'), \
                                patch('dwd._objective_mean._fallback', wraps=_fallback) as fallback:
                            warnings.simplefilter('always')
                            actual = objective_mean(losses, penalty, inside=inside)
                        self.assertEqual(caught, [])
                        fallback.assert_called_once()
                        self.assert_same_float(actual, expected)

    def test_each_ordinary_evaluation_order_is_kept_exactly(self):
        rng = np.random.default_rng(27)
        cases = [(rng.random(3), float(rng.random())) for _ in range(10)]
        cases += [(np.zeros(3), 0.), (np.array([-0.]), 0.),
                  (np.array([np.nextafter(0., 1.)]), 0.),
                  (rng.random(33)[::-1], .13)]
        # The placements can round differently; neither may replace the other.
        self.assertTrue(any(np.mean(losses + penalty) != np.mean(losses) + penalty
                            for losses, penalty in cases))
        for losses, penalty in cases:
            for inside in (False, True):
                with np.errstate(under='ignore'):
                    expected = (np.mean(losses + penalty) if inside else
                                np.mean(losses) + penalty)
                for policy in ('warn', 'raise'):
                    with self.subTest(inside=inside, policy=policy), \
                            np.errstate(over=policy, invalid=policy, under='ignore'), \
                            patch('dwd._objective_mean._fallback',
                                  side_effect=AssertionError('Unexpected fallback')):
                        self.assert_same_float(
                            objective_mean(losses, penalty, inside=inside), expected)

    def test_nonfinite_inputs_and_truly_unrepresentable_objectives_raise(self):
        cases = [(np.array([1e308, 1e308]), 1e308),
                 (np.array([np.inf, 1.]), 0.),
                 (np.array([-np.inf, 1.]), 0.),
                 (np.array([np.nan, 1.]), 0.),
                 (np.array([1., 2.]), np.inf),
                 (np.array([1., 2.]), np.nan)]
        for losses, penalty in cases:
            for inside in (False, True):
                for policy in ('warn', 'raise'):
                    with self.subTest(losses=losses, penalty=penalty,
                                      inside=inside, policy=policy), \
                            np.errstate(over=policy, invalid=policy):
                        with self.assertRaises(FloatingPointError):
                            objective_mean(losses, penalty, inside=inside)

    def test_fallback_requires_nonempty_native_binary64_inputs(self):
        foreign = np.dtype('>f8' if np.little_endian else '<f8')
        cases = [np.array([], dtype=np.float64),
                 np.array([1., 2.], dtype=np.float32),
                 np.array([1., 2.], dtype=foreign),
                 np.array([1, 2], dtype=np.int64),
                 np.array([1., 2.], dtype=object),
                 np.array([1., 2.], dtype=np.complex128)]
        for losses in cases:
            with self.subTest(dtype=losses.dtype, size=losses.size):
                with self.assertRaises(FloatingPointError):
                    _fallback(losses, 0.)

    def test_explicit_underflow_policy_is_not_swallowed(self):
        losses = np.array([np.nextafter(0., 1.), 0., 0.])
        for inside in (False, True):
            with self.subTest(inside=inside), np.errstate(under='raise'), \
                    patch('dwd._objective_mean._fallback',
                          side_effect=AssertionError('Underflow must propagate')):
                with self.assertRaisesRegex(FloatingPointError, 'underflow'):
                    objective_mean(losses, 0., inside=inside)


class ObjectiveCallSiteTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([-1., -1., 1., 1.])
        self.X = np.zeros((4, 1))
        self.K = np.eye(4)
        self.linear = importlib.import_module('dwd.gen_dwd')
        self.kernel = importlib.import_module('dwd.gen_kern_dwd')
        self.solver = importlib.import_module('dwd._kernel_solver')

    def objective_routes(self):
        return [
            (self.linear, lambda: self.linear.dwd_obj(
                self.X, self.y, 1., .25, np.zeros(1), -1e308)),
            (self.kernel, lambda: self.kernel.kern_dwd_obj(
                self.K, self.y, 1., .25, np.zeros(4), -1e308)),
            (self.solver, lambda: self.solver.optimality_diagnostics(
                self.K, self.y, np.zeros(4), -1e308, .25, 1.)['final_objective']),
        ]

    def test_existing_loss_hooks_called_once_and_their_errors_propagate(self):
        for module, function in self.objective_routes():
            for policy in ('warn', 'raise'):
                with self.subTest(module=module.__name__, policy=policy), \
                        np.errstate(over=policy, invalid=policy, under='ignore'), \
                        patch.object(module, 'V', wraps=module.V) as loss:
                    self.assertEqual(function(), 5e307)
                    loss.assert_called_once()
            for exception in (FloatingPointError('loss-hook failure'),
                              ValueError('loss-hook failure')):
                with self.subTest(module=module.__name__, exception=type(exception)), \
                        patch.object(module, 'V', side_effect=exception) as loss:
                    with self.assertRaisesRegex(type(exception), 'loss-hook failure'):
                        function()
                    loss.assert_called_once()

    def test_direct_objectives_recover_per_term_penalty_overflow(self):
        beta = np.ones(1)
        alpha = self.y / 2.
        for policy in ('warn', 'raise'):
            with self.subTest(policy=policy), \
                    np.errstate(over=policy, invalid=policy, under='ignore'):
                expected = decimal_objective(
                    self.linear.V(self.y * -1e308, q=1.), 1e308)
                self.assertEqual(self.linear.dwd_obj(
                    self.X, self.y, 1., 1e308, beta, -1e308), expected)
                self.assertEqual(self.kernel.kern_dwd_obj(
                    self.K, self.y, 1., 1e308, alpha, -1e308), expected)

    def test_zero_iteration_public_fits_keep_initial_state_with_finite_objective(self):
        for policy in ('warn', 'raise'):
            with self.subTest(policy=policy), threadpool_limits(1), \
                    np.errstate(over=policy, invalid=policy, under='ignore'):
                linear = GenDWD(lambd=.25, q=1., max_iter=0, initialization='zero')
                linear.fit(self.X, self.y, beta_init=np.zeros(1), offset_init=-1e308)
                kernel = KernGDWD(kernel='precomputed', lambd=.25, q=1., max_iter=0,
                                 initialization='zero', backend='spectral')
                kernel.fit(self.K, self.y, alpha_init=np.zeros(4), offset_init=-1e308)
                for model in (linear, kernel):
                    self.assertEqual(model.n_iter_, 0)
                    self.assertEqual(model.final_objective_, 5e307)
                    assert_array_equal(model.objective_history_, [5e307])
                    self.assertEqual(float(np.asarray(model.intercept_).ravel()[0]), -1e308)
                assert_array_equal(linear.coef_, np.zeros((1, 1)))
                assert_array_equal(kernel.dual_coef_, np.zeros((1, 4)))

    def test_accelerated_proposal_uses_the_same_finite_objective(self):
        # Isolate objective arithmetic from extreme linear-system conditioning.
        class FixedSystem:
            def solve_constrained(self, rhs):
                self.last_product = np.zeros(4)
                return np.zeros(4), -1e308

        module = importlib.import_module('dwd._accelerated_mm')
        for policy in ('warn', 'raise'):
            mm = RestartedMM(FixedSystem(), self.y, .25, 1., 1.)
            with self.subTest(policy=policy), \
                    np.errstate(over=policy, invalid=policy, under='ignore'), \
                    patch.object(module, 'V', wraps=module.V) as loss:
                alpha, offset, scores, objective = mm._proposal(np.zeros(4))
                self.assertEqual(objective, 5e307)
                self.assertEqual(offset, -1e308)
                assert_array_equal(alpha, np.zeros(4))
                assert_array_equal(scores, np.zeros(4))
                self.assertEqual(mm.info['acceleration_proposals'], 1)
                loss.assert_called_once()
        with patch.object(module, 'V', side_effect=FloatingPointError('loss-hook failure')):
            with self.assertRaisesRegex(FloatingPointError, 'loss-hook failure'):
                mm._proposal(np.zeros(4))


if __name__ == '__main__':
    unittest.main()
