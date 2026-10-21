"""Dense prediction norm reuse preserves customized helper behavior."""
import copy
from contextlib import ExitStack
import functools
import importlib
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix
import sklearn.metrics.pairwise as pairwise
import sklearn.utils.extmath as extmath
from threadpoolctl import threadpool_limits

from dwd import _guarded_rbf as guarded, _prediction_norms as norms, _rbf
from dwd import _sparse_features as sparse_features
from dwd.gen_kern_dwd import KernGDWD


class DenseHelperEligibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limits = threadpool_limits(limits=1)
        rng = np.random.default_rng(826004)
        cls.X = rng.normal(size=(32, 9))
        cls.X[rng.random(cls.X.shape) < .55] = 0
        cls.Q = rng.normal(size=(13, 9))
        cls.Q[rng.random(cls.Q.shape) < .55] = 0
        cls.y = np.tile([-3, 7], 16)
        cls.base = KernGDWD(kernel='rbf', kernel_kws={'gamma': .17}, lambd=.05,
                           max_iter=8, stopping='fixed', acceleration=None,
                           prediction_batch_size=3).fit(csr_matrix(cls.X), cls.y)
        cls.cases = []
        modes = ('sklearn', 'norm_sum', 'guarded_sklearn_rel_v1', 'guarded_norm_sum_rel_v1')
        for mode in modes:
            cls.cases.append(dict(id=mode + '-ordinary', mode=mode, target=None, mutation=False))
            target = (('pairwise', 'safe_sparse_dot') if mode in ('sklearn', 'guarded_sklearn_rel_v1')
                      else ('rbf' if mode == 'norm_sum' else 'guarded', 'safe_sparse_dot'))
            for mutation in (False, True):
                cls.cases.append(dict(id=mode + '-dot-' + str(mutation), mode=mode,
                                      target=target, mutation=mutation))
        for mode in modes[2:]:
            for helper in ('guarded_rbf', 'row_norms', 'euclidean_distances',
                           'check_pairwise_arrays', 'safe_sparse_dot'):
                cls.cases.append(dict(id=mode + '-' + helper, mode=mode,
                                      target=('guarded', helper), mutation=False))
        for helper in ('row_norms', 'euclidean_distances', 'check_pairwise_arrays'):
            module = 'extmath' if helper == 'row_norms' else 'pairwise'
            cls.cases.append(dict(id='public-' + helper, mode=modes[2],
                                  target=(module, helper), mutation=False))

    @classmethod
    def tearDownClass(cls):
        cls.limits.restore_original_limits()

    def model(self, mode, dense=True):
        result = copy.deepcopy(self.base)
        result._Xfit = self.X.copy() if dense else csr_matrix(self.X)
        result.kernel_computation_ = mode
        return result

    def test_helper_replacement_and_mutation_match_original_route(self):
        modules = dict(pairwise=pairwise, guarded=guarded, rbf=_rbf, extmath=extmath)
        coordinate = tuple(np.argwhere(self.X != 0)[0])
        self.assertEqual(len(self.cases), 25)
        for case in self.cases:
            with self.subTest(case=case['id']):
                outcomes = []
                for reuse in (False, True):
                    model = self.model(case['mode'])
                    query = self.Q.copy()
                    calls, mutations = [], []
                    with ExitStack() as stack:
                        if case['target']:
                            module_name, helper = case['target']
                            module = modules[module_name]
                            original = getattr(module, helper)

                            @functools.wraps(original)
                            def wrapped(*args, **kwargs):
                                value = original(*args, **kwargs)
                                left = args[0] if args else kwargs.get('X')
                                calls.append(left is model._Xfit)
                                if case['mutation'] and left is model._Xfit:
                                    model._Xfit[coordinate] += .01
                                    mutations.append(float(model._Xfit[coordinate]))
                                return value

                            stack.enter_context(patch.object(module, helper, wrapped))
                        prepared = norms.prepare_norm_kernel(model, query, query)
                        if case['target'] is None:
                            self.assertIsNotNone(prepared)
                        else:
                            self.assertIsNone(prepared)
                        assert_array_equal(model._Xfit, self.X)
                        calls.clear()
                        mutations.clear()
                        if not reuse:
                            stack.enter_context(patch.object(norms, 'prepare_norm_kernel', return_value=None))
                        scores = model.decision_function(query)
                        outcomes.append((scores, model._Xfit.copy(), calls.copy(), mutations.copy()))
                    assert_array_equal(query, self.Q)
                for position in (0, 1):
                    assert_array_equal(outcomes[0][position], outcomes[1][position])
                self.assertEqual(outcomes[0][2:], outcomes[1][2:])
                if case['mutation']:
                    self.assertEqual(len(outcomes[0][3]), 5)

    def test_preimport_matching_dot_alias_wrappers_still_decline(self):
        original = extmath.safe_sparse_dot

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            return original(*args, **kwargs)

        try:
            with ExitStack() as stack:
                for module in (extmath, pairwise, _rbf, guarded):
                    stack.enter_context(patch.object(module, 'safe_sparse_dot', wrapped))
                importlib.reload(norms)
                self.assertFalse(norms._DOT_SOURCE_OK)
                for mode in ('sklearn', 'norm_sum', 'guarded_sklearn_rel_v1', 'guarded_norm_sum_rel_v1'):
                    with self.subTest(mode=mode):
                        self.assertIsNone(norms.prepare_norm_kernel(self.model(mode), self.Q, self.Q))
        finally:
            importlib.reload(norms)
        self.assertTrue(norms._DOT_SOURCE_OK)

    def test_preimport_guarded_entry_wrapper_declines_guarded_dense_only(self):
        original = guarded.guarded_rbf

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            return original(*args, **kwargs)

        try:
            with patch.object(guarded, 'guarded_rbf', wrapped):
                importlib.reload(norms)
                self.assertFalse(norms._GUARDED_SOURCE_OK)
                for mode in ('guarded_sklearn_rel_v1', 'guarded_norm_sum_rel_v1'):
                    with self.subTest(mode=mode):
                        self.assertIsNone(norms.prepare_norm_kernel(self.model(mode), self.Q, self.Q))
                for mode in ('sklearn', 'norm_sum'):
                    with self.subTest(mode=mode):
                        self.assertIsNotNone(norms.prepare_norm_kernel(self.model(mode), self.Q, self.Q))
                # Existing CSR source qualification remains conservative.
                sparse_query = csr_matrix(self.Q)
                self.assertIsNone(norms.prepare_norm_kernel(self.model('sklearn', dense=False), sparse_query, sparse_query))
        finally:
            importlib.reload(norms)
        self.assertTrue(norms._GUARDED_SOURCE_OK)

    def test_sparse_storage_helper_replacement_does_not_disable_dense_reuse(self):
        original = sparse_features.csr_is_canonical

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            return original(*args, **kwargs)

        with patch.object(sparse_features, 'csr_is_canonical', wrapped):
            self.assertIsNotNone(norms.prepare_norm_kernel(self.model('sklearn'), self.Q, self.Q))
            sparse_query = csr_matrix(self.Q)
            self.assertIsNone(norms.prepare_norm_kernel(self.model('sklearn', dense=False), sparse_query, sparse_query))


if __name__ == '__main__':
    unittest.main(verbosity=2)
