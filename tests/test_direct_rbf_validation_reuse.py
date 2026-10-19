"""Call-local direct-RBF basis validation preserves arithmetic and hook control."""
from concurrent.futures import ThreadPoolExecutor
from copy import copy
import pickle
from threading import Lock
import types
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix

from dwd import _direct_rbf as direct
from dwd.gen_kern_dwd import KernGDWD, _direct_prediction_kernel
from dwd.kernel_utils import KernelClfMixin


class _NoReuseKernGDWD(KernGDWD):
    """An inherited custom type takes the unchanged per-batch public path."""


def _fitted_fixture(basis, gamma=.25, batch=2):
    model = KernGDWD(kernel='rbf', kernel_kws={'gamma': gamma},
                    rbf_computation='direct', prediction_batch_size=batch)
    model._Xfit = basis
    model.kernel_ = 'rbf'
    model.kernel_kws_ = {'gamma': gamma}
    model.kernel_computation_ = 'direct_v1'
    model.dual_coef_ = np.linspace(-.5, .7, basis.shape[0]).reshape(1, -1)
    model.intercept_ = np.array([.125])
    model.classes_ = np.array([0, 1])
    model.prediction_precision_ = 'compensated'
    return model


def _reference(model):
    result = copy(model)
    result.__class__ = _NoReuseKernGDWD
    return result


def _payload(value):
    if isinstance(value, np.ndarray):
        return (value.shape, value.dtype.str, value.tobytes())
    return (value.shape, value.data.tobytes(), value.indices.tobytes(), value.indptr.tobytes())


class DirectRBFValidationReuseTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(18926)
        self.X = rng.normal(size=(9, 5))
        self.query = rng.normal(size=(11, 5))

    def assert_same_scores(self, model, query, batch=2):
        keys = set(model.__dict__)
        before = _payload(model._Xfit), _payload(query)
        expected = _reference(model).decision_function(query, prediction_batch_size=batch)
        actual = model.decision_function(query, prediction_batch_size=batch)
        self.assertEqual(expected.shape, actual.shape)
        self.assertEqual(expected.dtype, actual.dtype)
        self.assertEqual(expected.tobytes(), actual.tobytes())
        self.assertEqual(before, (_payload(model._Xfit), _payload(query)))
        self.assertEqual(keys, set(model.__dict__))

    def assert_same_error(self, model, query, batch=2):
        errors = []
        for estimator in (_reference(model), model):
            try:
                estimator.decision_function(query, prediction_batch_size=batch)
            except Exception as error:
                errors.append((type(error), str(error)))
            else:
                self.fail('Expected an exception')
        self.assertEqual(errors[0], errors[1])

    def test_dense_csr_and_batch_parity(self):
        for sparse_basis in (False, True):
            for sparse_query in (False, True):
                model = _fitted_fixture(csr_matrix(self.X) if sparse_basis else self.X.copy())
                query = csr_matrix(self.query) if sparse_query else self.query.copy()
                for batch in (1, 3, 20, None):
                    with self.subTest(basis=sparse_basis, query=sparse_query, batch=batch):
                        self.assert_same_scores(model, query, batch)

    def test_dense_basis_checked_once_per_call(self):
        model = _fitted_fixture(self.X.copy())
        original = direct._finite_dense
        checked = []

        def count(values):
            if values is model._Xfit:
                checked.append(values)
            return original(values)

        with patch.object(direct, '_finite_dense', side_effect=count):
            model.decision_function(self.query, prediction_batch_size=1)
            self.assertEqual(len(checked), 1)
            model.decision_function(self.query, prediction_batch_size=3)
            self.assertEqual(len(checked), 2)

    def test_read_only_fortran_and_strided(self):
        storage = np.zeros((9, 10))
        storage[:, ::2] = self.X
        for basis in (np.asfortranarray(self.X), storage[:, ::2]):
            basis.flags.writeable = False
            self.assert_same_scores(_fitted_fixture(basis), self.query)

    def test_noncanonical_csr_copy_once_and_no_mutation(self):
        basis = csr_matrix((np.array([.5, 0., .5, 2., -1., 3., 4.]),
                            np.array([1, 0, 1, 2, 0, 4, 3]),
                            np.array([0, 4, 7])), shape=(2, 5))
        model = _fitted_fixture(basis)
        self.assert_same_scores(model, self.query, 1)
        self.assert_same_scores(_fitted_fixture(self.X.copy()), basis, 1)
        original = csr_matrix.copy
        copied = []

        def count(matrix, *args, **kwargs):
            if matrix is basis:
                copied.append(matrix)
            return original(matrix, *args, **kwargs)

        with patch.object(csr_matrix, 'copy', count):
            model.decision_function(self.query, prediction_batch_size=1)
            self.assertEqual(len(copied), 1)
            model.decision_function(self.query, prediction_batch_size=2)
            self.assertEqual(len(copied), 2)

    def test_csr_duplicate_overflow_still_raises(self):
        basis = csr_matrix((np.array([np.finfo(float).max, np.finfo(float).max]),
                            np.array([0, 0]), np.array([0, 2])), shape=(1, 5))
        self.assert_same_error(_fitted_fixture(basis), self.query)

    def test_gamma_validation_and_bool_public_semantics(self):
        for gamma in (None, 0., True, .1):
            with self.subTest(gamma=gamma):
                self.assert_same_scores(_fitted_fixture(self.X.copy(), gamma=gamma), self.query)
        for gamma in (-1., np.nan, np.inf, 1j, 'bad', np.bool_(True)):
            with self.subTest(gamma=gamma):
                self.assert_same_error(_fitted_fixture(self.X.copy(), gamma=gamma), self.query)

    def test_invalid_public_batches(self):
        for batch in (0, -1, True, 1.5):
            with self.subTest(batch=batch):
                self.assert_same_error(_fitted_fixture(self.X.copy()), self.query, batch)

    def test_invalid_inputs_and_validation_order(self):
        model = _fitted_fixture(self.X.copy())
        self.assert_same_error(model, self.query[:, :-1])
        self.assert_same_error(model, self.query[:0])
        self.assert_same_error(model, np.full_like(self.query, np.inf))
        self.assert_same_error(_fitted_fixture(self.X.astype(np.float32)), self.query)
        self.assert_same_error(_fitted_fixture(csr_matrix(self.X).tocsc()), self.query)
        # A failed public query check must precede basis validation.
        model._Xfit[0, 0] = np.nan
        self.assert_same_error(model, self.query[:, :-1], batch=0)
        self.assert_same_error(model, self.query, batch=0)

    def test_basis_rechecked_after_mutation_and_failure(self):
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                model = _fitted_fixture(csr_matrix(self.X) if sparse else self.X.copy())
                before = model.decision_function(self.query)
                values = model._Xfit.data if sparse else model._Xfit
                values.flat[0] += 4
                after = model.decision_function(self.query)
                self.assertFalse(np.array_equal(before, after))
                self.assert_same_scores(model, self.query)
                values.flat[0] = np.nan
                self.assert_same_error(model, self.query)
                values.flat[0] = 1
                self.assert_same_scores(model, self.query)

    def test_exceptional_ranges_preserve_arithmetic(self):
        for value, gamma in ((1e160, 1e-320), (1e-162, 1e308),
                             (1e308, np.nextafter(0., 1.))):
            basis = np.array([[0.], [value], [-value]])
            query = np.array([[value], [0.], [-value], [value]])
            for sparse in (False, True):
                with self.subTest(value=value, gamma=gamma, sparse=sparse):
                    model = _fitted_fixture(csr_matrix(basis) if sparse else basis, gamma=gamma)
                    self.assert_same_scores(model, query, 1)

    def test_instance_kernel_and_scoring_hooks_fall_back(self):
        for name, hook in (
                ('_compute_kernel', lambda self, q: np.zeros((self._Xfit.shape[0], q.shape[0]))),
                ('_kernel_decision_product', lambda self, K: np.full((K.shape[1], 1), .25)),
                ('decision_function', lambda self, q, **kw: np.ones(q.shape[0]))):
            with self.subTest(hook=name):
                model = _fitted_fixture(self.X.copy())
                setattr(model, name, types.MethodType(hook, model))
                self.assertIsNone(_direct_prediction_kernel(model))
                self.assert_same_scores(model, self.query)

    def test_subclass_falls_back(self):
        model = _reference(_fitted_fixture(self.X.copy()))
        self.assertIsNone(_direct_prediction_kernel(model))
        self.assert_same_scores(model, self.query)

    def test_class_level_hooks_fall_back_to_current_dispatch(self):
        hooks = (
            (KernelClfMixin, '_compute_kernel', lambda self, q: np.zeros((self._Xfit.shape[0], q.shape[0]))),
            (KernelClfMixin, 'decision_function', lambda self, q, **kw: np.full(q.shape[0], 17.)),
            (KernelClfMixin, '_kernel_decision_product', lambda self, K: np.full((K.shape[1], 1), .75)),
            (KernGDWD, '_compute_training_kernel', lambda self, q: np.eye(q.shape[0])),
            (KernGDWD, '_known_psd_kernel', lambda self: False),
        )
        for owner, name, hook in hooks:
            with self.subTest(hook=name), patch.object(owner, name, hook):
                model = _fitted_fixture(self.X.copy())
                self.assertIsNone(_direct_prediction_kernel(model))
                self.assert_same_scores(model, self.query)

    def test_direct_kernel_or_validator_replacement_retains_control(self):
        for name in ('accurate_rbf', '_validated_features'):
            original = getattr(direct, name)
            with self.subTest(hook=name), patch.object(direct, name, wraps=original) as observed:
                model = _fitted_fixture(self.X.copy())
                self.assertIsNone(_direct_prediction_kernel(model))
                self.assert_same_scores(model, self.query)
                self.assertGreater(observed.call_count, 0)

    def test_concurrent_read_only_calls_have_independent_validation(self):
        model = _fitted_fixture(self.X.copy())
        jobs = [(self.query[::-1].copy() if i % 2 else self.query.copy(), (1, 3, None, 20)[i % 4])
                for i in range(16)]
        expected = [_reference(model).decision_function(q, prediction_batch_size=b) for q, b in jobs]
        original = direct._finite_dense
        lock = Lock()
        checked = []

        def count(values):
            if values is model._Xfit:
                with lock:
                    checked.append(values)
            return original(values)

        def predict(job):
            query, batch = job
            return model.decision_function(query, prediction_batch_size=batch)

        before = pickle.dumps(model, protocol=5)
        with patch.object(direct, '_finite_dense', side_effect=count):
            with ThreadPoolExecutor(max_workers=4) as pool:
                actual = list(pool.map(predict, jobs))
        self.assertEqual(len(checked), len(jobs))
        for want, got in zip(expected, actual):
            self.assertEqual(want.tobytes(), got.tobytes())
        self.assertEqual(pickle.dumps(model, protocol=5), before)
        # Sparse calls each own a canonicalized basis, with no persistent cache.
        sparse_model = _fitted_fixture(csr_matrix(self.X))
        with ThreadPoolExecutor(max_workers=4) as pool:
            actual = list(pool.map(lambda job: sparse_model.decision_function(
                job[0], prediction_batch_size=job[1]), jobs))
        for (query, batch), got in zip(jobs, actual):
            want = _reference(sparse_model).decision_function(query, prediction_batch_size=batch)
            self.assertEqual(want.tobytes(), got.tobytes())

    def test_actual_fits_serialization_and_fitted_policy(self):
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                data = np.random.default_rng(18927).normal(size=(12, 5))
                model = KernGDWD(kernel='rbf', kernel_kws={'gamma': .15}, lambd=.1,
                                rbf_computation='direct', stopping='fixed', max_iter=12,
                                prediction_batch_size=3).fit(
                                    csr_matrix(data) if sparse else data, np.tile([0, 1], 6))
                for batch in (1, 3, None):
                    self.assert_same_scores(model, self.query, batch)
                restored = pickle.loads(pickle.dumps(model))
                self.assert_same_scores(restored, self.query)
                expected = model.decision_function(self.query)
                model.set_params(rbf_computation='standard', kernel='linear')
                # Constructor edits do not change the fitted kernel policy.
                self.assertEqual(expected.tobytes(), model.decision_function(self.query).tobytes())

    def test_standard_policy_never_uses_direct_reuse(self):
        model = _fitted_fixture(self.X.copy())
        model.kernel_computation_ = 'sklearn'
        self.assertIsNone(_direct_prediction_kernel(model))
        with patch.object(direct, '_prediction_basis_kernel', side_effect=AssertionError('Unexpected direct path')):
            self.assert_same_scores(model, self.query)


if __name__ == '__main__':
    unittest.main()
