"""Public eigen inputs must reject complex values before real conversion."""

import unittest
import warnings
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal
from sklearn.exceptions import NotFittedError

import dwd._eigen as eigen
from dwd.gen_dwd import GenDWD, get_P0_eig
from dwd.gen_kern_dwd import KernGDWD, get_K_eig


def _complex_forms(values):
    """Include object arrays whose dtype alone hides complex scalar values."""
    for imaginary in (0., 1.):
        yield ('complex', imaginary), np.asarray(values, dtype=complex) + imaginary * 1j
        for scalar in (complex, np.complex128):
            array = np.asarray(values, dtype=object).copy()
            # One complex element suffices; the others remain ordinary reals.
            array.flat[-1] = scalar(complex(array.flat[-1], imaginary))
            yield (scalar.__name__, imaginary), array


class ComplexEigenInputTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[-2., 0.], [-1., 1.], [1., -1.], [2., 0.]])
        self.y = np.array([0, 0, 1, 1])
        self.K = self.X @ self.X.T + np.eye(len(self.X))

    def cases(self):
        for mode in ('schur', 'legacy'):
            yield GenDWD(max_iter=0, solver_mode=mode), self.X, 'P0_eig', get_P0_eig(self.X)
            yield KernGDWD(kernel='precomputed', max_iter=0, solver_mode=mode), self.K, 'K_eig', get_K_eig(self.K)

    def assert_rejected_without_warning(self, action, pattern):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with self.assertRaisesRegex(ValueError, pattern):
                action()
        self.assertEqual(caught, [], 'Invalid input was cast before rejection')

    def test_complex_supplied_vectors_and_values_rejected_without_recomputation(self):
        for model, data, parameter, (vectors, values) in self.cases():
            for part, original in ((0, vectors), (1, values)):
                for form, malformed in _complex_forms(original):
                    with self.subTest(model=type(model).__name__, mode=model.solver_mode,
                                      part=part, form=form):
                        supplied = [vectors, values]
                        supplied[part] = malformed
                        with patch.object(eigen, 'eigh', side_effect=AssertionError(
                                'Invalid supplied eigenpairs must not be replaced')):
                            self.assert_rejected_without_warning(
                                lambda: model.fit(data, self.y, **{parameter: supplied}),
                                'Supplied eigenpairs failed validation.*real')

    def test_complex_matrix_rejected_before_eigendecomposition(self):
        for form, malformed in _complex_forms(self.K):
            with self.subTest(form=form):
                with patch.object(eigen, 'eigh', side_effect=AssertionError(
                        'Complex matrices must not reach eigendecomposition')):
                    self.assert_rejected_without_warning(
                        lambda: get_K_eig(malformed), 'must be real symmetric')

    def test_failed_complex_cache_refit_clears_fitted_state(self):
        for model, data, parameter, (vectors, values) in self.cases():
            with self.subTest(model=type(model).__name__, mode=model.solver_mode):
                model.fit(data, self.y, **{parameter: (vectors, values)})
                malformed = np.asarray(vectors, dtype=object)
                malformed.flat[-1] = np.complex128(malformed.flat[-1] + 1j)
                self.assert_rejected_without_warning(
                    lambda: model.fit(data, self.y, **{parameter: (malformed, values)}),
                    'Supplied eigenpairs failed validation.*real')
                learned = [name for name in vars(model)
                           if name.endswith('_') and not name.startswith('_')]
                self.assertEqual(learned, [])
                self.assertFalse(hasattr(model, '_Xfit'))
                with self.assertRaises(NotFittedError):
                    model.predict(data)

    def test_valid_real_cache_forms_preserve_exact_fitted_values(self):
        for model, data, parameter, (vectors, values) in self.cases():
            # Use a deterministic start for legacy mode as well.
            model.set_params(random_state=17, max_iter=2)
            for supplied in ((vectors, values),
                             (vectors.tolist(), values.tolist()),
                             (vectors.astype(object), values)):
                with self.subTest(model=type(model).__name__, mode=model.solver_mode):
                    # Compare the same input representation with the old dtype
                    # check. List/array layouts can themselves round differently.
                    with patch.object(eigen, 'contains_complex', np.iscomplexobj):
                        model.fit(data, self.y, **{parameter: supplied})
                    expected_scores = model.decision_function(data).copy()
                    expected_history = model.objective_history_.copy()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        model.fit(data, self.y, **{parameter: supplied})
                    self.assertEqual(caught, [])
                    assert_array_equal(model.decision_function(data), expected_scores)
                    assert_array_equal(model.objective_history_, expected_history)


if __name__ == '__main__':
    unittest.main(verbosity=2)
