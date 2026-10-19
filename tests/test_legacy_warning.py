"""Visible legacy-fit warnings without changes to saved-model prediction."""

import pickle
import unittest
import warnings

import numpy as np
from numpy.testing import assert_array_equal
from sklearn.base import clone
from sklearn.metrics.pairwise import rbf_kernel

from dwd.gen_kern_dwd import KernGDWD, KernGDWDCV, solve_gen_kern_dwd


class LegacyWarningTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(13)
        self.X = rng.normal(size=(48, 6))
        self.y = np.where(self.X[:, 0] + .15 * self.X[:, 1] ** 2 > 0,
                          'positive', 'negative')
        self.query = rng.normal(size=(19, 6))
        self.K = rbf_kernel(self.X, self.X, gamma=.2)

    @staticmethod
    def legacy_warnings(caught):
        return [item for item in caught
                if issubclass(item.category, FutureWarning)
                and 'historical kernel update' in str(item.message)]

    def model(self, **changes):
        options = dict(kernel='rbf', kernel_kws={'gamma': .2},
                       solver_mode='legacy', max_iter=2, random_state=4)
        options.update(changes)
        return KernGDWD(**options)

    def test_each_fit_entrypoint_warns(self):
        calls = [
            lambda: solve_gen_kern_dwd(
                self.K, self.y, .1, solver_mode='legacy', max_iter=2,
                random_state=4),
            lambda: self.model().fit(self.X, self.y),
            lambda: KernGDWDCV(
                kernel='rbf', kernel_kws_vals=[{'gamma': .2}],
                lambd_vals=[.1], q_vals=[1.], cv=2, solver_mode='legacy',
                max_iter=2, random_state=4).fit(self.X, self.y),
        ]
        for position, call in enumerate(calls):
            with self.subTest(entrypoint=position), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                call()
            self.assertTrue(self.legacy_warnings(caught))

    def test_prediction_and_pickle_do_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            model = self.model().fit(self.X, self.y)
        expected = model.decision_function(self.query)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            restored = pickle.loads(pickle.dumps(model))
            assert_array_equal(restored.decision_function(self.query), expected)
            assert_array_equal(restored.predict(self.query), model.predict(self.query))
        self.assertFalse(self.legacy_warnings(caught))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            refitted = clone(restored).fit(self.X, self.y)
        self.assertTrue(self.legacy_warnings(caught))
        assert_array_equal(refitted.decision_function(self.query), expected)

    def test_corrected_mode_does_not_emit_legacy_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            solve_gen_kern_dwd(self.K, self.y, .1, solver_mode='schur', max_iter=2)
            self.model(solver_mode='schur').fit(self.X, self.y)
        self.assertFalse(self.legacy_warnings(caught))

    def test_invalid_low_level_inputs_fail_before_warning(self):
        base = dict(K=self.K, y=self.y, lambd=.1,
                    solver_mode='legacy', max_iter=2, random_state=4)
        for changes in ({'q': 0}, {'lambd': -1}, {'max_iter': -1},
                        {'K': self.K[:, :-1]}, {'alpha_init': np.zeros(47)},
                        {'offset_init': np.inf}):
            with self.subTest(changes=tuple(changes)), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                with self.assertRaises(ValueError):
                    solve_gen_kern_dwd(**dict(base, **changes))
            self.assertFalse(self.legacy_warnings(caught))

    def test_invalid_estimator_input_fails_before_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with self.assertRaises(ValueError):
                self.model(q=0).fit(self.X, self.y)
        self.assertFalse(self.legacy_warnings(caught))


if __name__ == '__main__':
    unittest.main()
