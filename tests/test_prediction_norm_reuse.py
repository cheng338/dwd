"""Batched RBF norm reuse must preserve prediction and extension contracts."""
import copy
import functools
import importlib
import pickle
import sys
import types
import unittest
import warnings
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix
import sklearn.metrics.pairwise as pairwise
import sklearn.utils.extmath as extmath
from threadpoolctl import threadpool_limits

from dwd import _rbf, kernel_utils
from dwd.gen_kern_dwd import KernGDWD
from dwd import _prediction_norms as norms


class PredictionNormReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limits = threadpool_limits(limits=1)
        rng = np.random.default_rng(181004)
        cls.X = rng.normal(size=(32, 7))
        cls.y = np.tile([-3, 7], 16)
        cls.Q = rng.normal(size=(13, 7))
        # Keep the fitted scientific policy explicit; separate tests exercise auto.
        opts = dict(kernel='rbf', kernel_kws={'gamma': .17}, lambd=.05,
                    max_iter=8, stopping='fixed', affine_computation='standard',
                    residual_check_order='refinement_first')
        cls.base = KernGDWD(**opts).fit(cls.X, cls.y)
        cls.direct = KernGDWD(**opts, rbf_computation='direct').fit(cls.X, cls.y)

    @classmethod
    def tearDownClass(cls):
        cls.limits.restore_original_limits()

    def model(self, mode='sklearn'):
        model = copy.deepcopy(self.direct if mode == 'direct' else self.base)
        if mode != 'direct':
            model.kernel_computation_ = mode
        model.prediction_batch_size = 3
        return model

    def compare(self, model, query, **kwargs):
        # This changes only the optimization dispatch, leaving the estimator's
        # original per-batch kernel and score methods as the reference.
        with patch.object(norms, 'prepare_norm_kernel', return_value=None):
            expected = model.decision_function(query, **kwargs)
        actual = model.decision_function(query, **kwargs)
        assert_array_equal(actual, expected)
        return actual

    def test_eligible_call_reuses_training_norm_once_without_persistent_state(self):
        self.assertTrue(norms._SOURCE_OK)
        for mode in ('sklearn', 'norm_sum'):
            with self.subTest(mode=mode):
                model = self.model(mode)
                before = set(vars(model))
                basis = model._Xfit.copy()
                query = self.Q.copy()
                code = extmath.row_norms.__code__
                counts = []
                for reuse in (False, True):
                    count = [0]
                    def observe(frame, event, arg):
                        if (event == 'call' and frame.f_code is code
                                and frame.f_locals.get('X') is model._Xfit):
                            count[0] += 1
                    with ExitStack() as stack:
                        if not reuse:
                            stack.enter_context(patch.object(norms, 'prepare_norm_kernel', return_value=None))
                        sys.setprofile(observe)
                        try:
                            scores = model.decision_function(query)
                        finally:
                            sys.setprofile(None)
                    counts.append(count[0])
                    if not reuse:
                        expected = scores
                    else:
                        assert_array_equal(scores, expected)
                self.assertEqual(counts, [5, 1])
                self.assertEqual(set(vars(model)), before)
                assert_array_equal(model._Xfit, basis)
                assert_array_equal(query, self.Q)
                assert_array_equal(pickle.loads(pickle.dumps(model)).decision_function(query), expected)

    def test_layouts_aliases_and_precision_modes_preserve_scores(self):
        for mode in ('sklearn', 'norm_sum'):
            for precision in ('ordinary', 'adaptive', 'compensated'):
                for layout in ('C', 'F', 'reverse', 'alias', 'copy'):
                    with self.subTest(mode=mode, precision=precision, layout=layout):
                        model = self.model(mode)
                        model.prediction_precision_ = precision
                        query = {'C': self.Q, 'F': np.asfortranarray(self.Q),
                                 'reverse': self.Q[::-1], 'alias': model._Xfit,
                                 'copy': model._Xfit.copy()}[layout]
                        self.compare(model, query)

    def test_sparse_float32_lists_and_subclasses_use_original_route(self):
        class ArraySubclass(np.ndarray):
            pass
        for query in (csr_matrix(self.Q), self.Q.astype(np.float32),
                      self.Q.tolist(), self.Q.view(ArraySubclass)):
            model = self.model()
            self.compare(model, query)
            checked = kernel_utils.check_array(query, dtype=np.float64, accept_sparse='csr')
            self.assertIsNone(norms.prepare_norm_kernel(model, checked, query))
        class Subclass(KernGDWD):
            pass
        model = self.model()
        model.__class__ = Subclass
        self.assertIsNone(norms.prepare_norm_kernel(model, self.Q, self.Q))
        self.compare(model, self.Q)
        model = self.model()
        model._Xfit = csr_matrix(model._Xfit)
        self.assertIsNone(norms.prepare_norm_kernel(model, self.Q, self.Q))
        self.compare(model, self.Q)

    def test_current_auto_affine_policy_and_batch_override(self):
        model = KernGDWD(kernel='rbf', kernel_kws={'gamma': .17}, lambd=.05,
                         max_iter=8, stopping='fixed', prediction_batch_size=3).fit(self.X, self.y)
        for size in (None, 1, 2, 100):
            self.compare(model, self.Q, prediction_batch_size=size)
        with patch.object(norms, 'prepare_norm_kernel', side_effect=AssertionError('unbatched call')):
            model.decision_function(self.Q, prediction_batch_size=None)
        self.assertEqual(model.prediction_batch_size, 3)

    def test_direct_basis_reuse_remains_separate(self):
        model = self.model('direct')
        with patch.object(norms, 'prepare_norm_kernel', side_effect=AssertionError('direct path')):
            scores = model.decision_function(self.Q)
        assert_array_equal(scores, self.direct.decision_function(self.Q, prediction_batch_size=3))

    def test_fitted_parameters_and_basis_mutation_do_not_leave_stale_norms(self):
        for mode in ('sklearn', 'norm_sum'):
            model = self.model(mode)
            old = model.decision_function(self.Q)
            model.kernel = 'linear'
            model.kernel_kws = {'gamma': 30.}
            assert_array_equal(model.decision_function(self.Q), old)
            for gamma in (.05, .9, None):
                model.kernel_kws_ = {'gamma': gamma}
                self.compare(model, self.Q)
            model._Xfit[0, 0] += 1.
            self.compare(model, self.Q)
            model.kernel_ = 'linear'
            model.kernel_kws_ = None
            self.assertIsNone(norms.prepare_norm_kernel(model, self.Q, self.Q))
            self.compare(model, self.Q)

    def test_custom_matrix_kernel_is_not_bypassed(self):
        model = self.model()
        calls = []
        def callback(train, query):
            calls.append(len(query))
            return train @ query.T
        model.kernel_ = callback
        model.kernel_kws_ = None
        self.assertIsNone(norms.prepare_norm_kernel(model, self.Q, self.Q))
        model.decision_function(self.Q)
        self.assertEqual(calls, [3, 3, 3, 3, 1])

    def test_mutating_estimator_hooks_are_not_bypassed(self):
        for target in ('_compute_kernel', '_kernel_decision_product'):
            for class_hook in (False, True):
                model = self.model()
                reference = copy.deepcopy(model)
                counts = [0, 0]
                outputs = []
                for index, fitted in enumerate((reference, model)):
                    original = getattr(fitted, target)
                    def wrapped(self, values, original=original, index=index):
                        counts[index] += 1
                        answer = original(values)
                        self._Xfit[0, 0] += .01
                        return answer
                    with ExitStack() as stack:
                        if class_hook:
                            stack.enter_context(patch.object(kernel_utils.KernelClfMixin, target, wrapped))
                        else:
                            setattr(fitted, target, types.MethodType(wrapped, fitted))
                        if index == 0:
                            stack.enter_context(patch.object(norms, 'prepare_norm_kernel', return_value=None))
                        outputs.append(fitted.decision_function(self.Q))
                assert_array_equal(outputs[0], outputs[1])
                self.assertEqual(counts, [5, 5])
                assert_array_equal(model._Xfit, reference._Xfit)

    def test_late_replaced_helpers_and_registry_decline_reuse(self):
        model = self.model()
        targets = [(kernel_utils, 'pairwise_kernels'), (pairwise, 'rbf_kernel'),
                   (pairwise, 'euclidean_distances'), (pairwise, 'check_pairwise_arrays'),
                   (extmath, 'row_norms'), (pairwise, 'row_norms'),
                   (_rbf, 'row_norms'), (_rbf, 'norm_sum_rbf'),
                   (_rbf, '_norm_sum_rbf_with_norms')]
        for module, name in targets:
            with self.subTest(module=module.__name__, name=name):
                original = getattr(module, name)
                @functools.wraps(original)
                def wrapped(*args, **kwargs):
                    return original(*args, **kwargs)
                with patch.object(module, name, wrapped):
                    self.assertIsNone(norms.prepare_norm_kernel(model, self.Q, self.Q))
                    self.compare(model, self.Q)
        with patch.dict(pairwise.PAIRWISE_KERNEL_FUNCTIONS, {'rbf': lambda a, b, **kw: a @ b.T}):
            self.assertIsNone(norms.prepare_norm_kernel(model, self.Q, self.Q))
            self.compare(model, self.Q)

    def test_preimport_wrappers_are_not_mistaken_for_original_helpers(self):
        for use_wraps in (False, True):
            for target in ('row_norms', 'norm_sum_rbf'):
                model = self.model('norm_sum' if target == 'norm_sum_rbf' else 'sklearn')
                original = extmath.row_norms if target == 'row_norms' else _rbf.norm_sum_rbf
                def wrapped(*args, **kwargs):
                    return original(*args, **kwargs)
                if use_wraps:
                    wrapped = functools.wraps(original)(wrapped)
                try:
                    with ExitStack() as stack:
                        if target == 'row_norms':
                            for module in (extmath, pairwise, _rbf):
                                stack.enter_context(patch.object(module, target, wrapped))
                        else:
                            stack.enter_context(patch.object(_rbf, target, wrapped))
                        importlib.reload(norms)
                        self.assertFalse(norms._SOURCE_OK)
                        self.assertIsNone(norms.prepare_norm_kernel(model, self.Q, self.Q))
                        self.compare(model, self.Q)
                finally:
                    importlib.reload(norms)
        self.assertTrue(norms._SOURCE_OK)

    def test_optional_allocation_failure_falls_back(self):
        model = self.model()
        original = np.einsum
        failures = []
        def limited(*args, **kwargs):
            frame = sys._getframe(1)
            while frame:
                if frame.f_code.co_name == 'prepare_norm_kernel':
                    failures.append(True)
                    raise MemoryError('optional norm allocation')
                frame = frame.f_back
            return original(*args, **kwargs)
        with patch.object(np, 'einsum', limited):
            self.compare(model, self.Q)
        self.assertEqual(len(failures), 1)

    def test_gamma_qualification_preserves_ordinary_validation(self):
        model = self.model()
        for gamma in (None, 0., .17, np.float64(.17), np.int64(1)):
            model.kernel_kws_ = {'gamma': gamma}
            self.assertIsNotNone(norms.prepare_norm_kernel(model, self.Q, self.Q))
            self.compare(model, self.Q)
        for gamma in (True, np.float32(.17), -1., np.inf, np.nan, 'scale'):
            model.kernel_kws_ = {'gamma': gamma}
            self.assertIsNone(norms.prepare_norm_kernel(model, self.Q, self.Q))
        model.kernel_kws_ = {'gamma': .17, 'unexpected': 2}
        self.assertIsNone(norms.prepare_norm_kernel(model, self.Q, self.Q))

    def test_concurrent_read_only_calls_have_private_norms(self):
        model = self.model()
        queries = [self.Q, self.Q[::-1], self.Q + .1, self.Q * .8]
        with patch.object(norms, 'prepare_norm_kernel', return_value=None):
            expected = [model.decision_function(query) for query in queries]
        before = set(vars(model))
        with ThreadPoolExecutor(max_workers=2) as executor:
            actual = list(executor.map(model.decision_function, queries))
        for values, reference in zip(actual, expected):
            assert_array_equal(values, reference)
        self.assertEqual(set(vars(model)), before)

    def test_warning_error_and_nonfinite_fallback_contracts(self):
        def outcome(model, query, use_reuse):
            with warnings.catch_warnings(record=True) as caught, ExitStack() as stack:
                warnings.simplefilter('always')
                if not use_reuse:
                    stack.enter_context(patch.object(norms, 'prepare_norm_kernel', return_value=None))
                try:
                    answer = model.decision_function(query)
                    status = ('returned', answer)
                except Exception as exc:
                    status = (type(exc).__name__, str(exc))
                return status, [(type(w.message).__name__, str(w.message)) for w in caught]
        for where, value in [('query', np.nan), ('query', np.inf),
                             ('basis', np.nan), ('basis', np.inf), ('basis', 1e200)]:
            model = self.model()
            query = self.Q.copy()
            (query if where == 'query' else model._Xfit)[0, 0] = value
            old, old_warnings = outcome(model, query, False)
            new, new_warnings = outcome(model, query, True)
            self.assertEqual(new[0], old[0])
            if old[0] == 'returned':
                assert_array_equal(old[1], new[1])
            else:
                self.assertEqual(new[1], old[1])
            self.assertEqual(new_warnings, old_warnings)
        for policy in ('raise', 'warn'):
            with np.errstate(under=policy):
                model = self.model()
                self.assertIsNone(norms.prepare_norm_kernel(model, self.Q, self.Q))
                old, old_warnings = outcome(model, self.Q, False)
                new, new_warnings = outcome(model, self.Q, True)
                self.assertEqual(new[0], old[0])
                if old[0] == 'returned':
                    assert_array_equal(old[1], new[1])
                else:
                    self.assertEqual(new[1], old[1])
                self.assertEqual(new_warnings, old_warnings)


if __name__ == '__main__':
    unittest.main()
