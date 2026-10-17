"""Complex inputs must be rejected before real-valued DWD arithmetic."""

import importlib.util
import unittest
import warnings

import numpy as np
from sklearn.exceptions import NotFittedError

from dwd.cv import run_cv
from dwd.gen_dwd import GenDWD, GenDWDCV, solve_gen_dwd
from dwd.gen_kern_dwd import KernGDWD, KernGDWDCV, solve_gen_kern_dwd
from dwd.kern_md import KernMD, kern_md
from dwd.utils import pm1
from dwd._kernel_solver import solve_kernel
from dwd._validation import contains_complex


def complex_arrays(values):
    for imaginary in (0., 3.):
        yield np.asarray(values, dtype=complex) + imaginary * 1j
        for scalar in (complex, np.complex128):
            result = np.asarray(values, dtype=object).copy()
            result.flat[-1] = scalar(complex(result.flat[-1], imaginary))
            yield result


def nested_object(value, depth):
    for _ in range(depth):
        outer = np.empty((), dtype=object)
        outer[()] = value
        value = outer
    return value


class ComplexInputTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[-1.], [1.]])
        self.K = np.eye(2)
        self.y = np.array([-1, 1])

    def routes(self):
        for mode in ('schur', 'legacy'):
            yield GenDWD(max_iter=0, solver_mode=mode), self.X, 'beta_init', np.array([.1])
            yield KernGDWD(kernel='precomputed', max_iter=0, solver_mode=mode), self.K, 'alpha_init', np.array([.1, -.1])

    def assert_rejected(self, action, pattern='real|Complex'):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with self.assertRaisesRegex(ValueError, pattern):
                action()
        self.assertEqual(caught, [], 'No complex-to-real conversion may precede rejection')

    def assert_unfitted(self, model, data):
        with self.assertRaises(NotFittedError):
            model.predict(data)
        self.assertFalse(any(name.endswith('_') and not name.startswith('_')
                             for name in vars(model)))

    def test_complex_intercept_forms_reject_on_first_fit_and_refit(self):
        for imaginary in (0., 3.):
            value = complex(.2, imaginary)
            forms = (value, np.complex128(value), np.array(value),
                     np.array(value, dtype=object), np.array(np.complex128(value), dtype=object))
            for model, data, name, initial in self.routes():
                for offset in forms:
                    for previous in (False, True):
                        with self.subTest(mode=model.solver_mode, model=type(model).__name__,
                                          value=repr(offset), previous=previous):
                            if previous:
                                model.fit(data, self.y, **{name: initial})
                            self.assert_rejected(lambda: model.fit(
                                data, self.y, offset_init=offset, **{name: initial}), 'offset_init.*real')
                            self.assert_unfitted(model, data)

    def test_object_wrapped_complex_coefficients_reject_without_casting(self):
        for model, data, name, initial in self.routes():
            for invalid in complex_arrays(initial):
                with self.subTest(mode=model.solver_mode, model=type(model).__name__, value=repr(invalid)):
                    model.fit(data, self.y, **{name: initial})
                    self.assert_rejected(lambda: model.fit(data, self.y, **{name: invalid}), name + '.*real')
                    self.assert_unfitted(model, data)

    def test_direct_solver_initial_values_reject_too(self):
        for mode in ('schur', 'legacy'):
            calls = (
                lambda **kw: solve_gen_dwd(self.X, self.y, 1., max_iter=0, solver_mode=mode, **kw),
                lambda **kw: solve_gen_kern_dwd(self.K, self.y, 1., max_iter=0, solver_mode=mode, **kw),
            )
            for call in calls:
                self.assert_rejected(lambda: call(offset_init=np.complex128(1+2j)), 'offset_init.*real')
        for name in ('lambd', 'q', 'obj_tol'):
            for value in (1+0j, np.complex128(1+2j)):
                kwargs = dict(lambd=1., max_iter=0, solver_mode='legacy')
                kwargs[name] = value
                self.assert_rejected(lambda: solve_gen_kern_dwd(self.K, self.y, **kwargs), name + '.*real')

    def test_nested_object_complex_initial_values_reject_without_casting(self):
        for depth in (1, 3):
            for imaginary in (0., 2.):
                value = nested_object(np.complex128(complex(.2, imaginary)), depth)
                for model, data, name, initial in self.routes():
                    invalid = initial.astype(object)
                    invalid[-1] = value
                    with self.subTest(depth=depth, imaginary=imaginary, model=type(model).__name__, mode=model.solver_mode):
                        self.assert_rejected(lambda: model.fit(data, self.y, **{name: invalid}), name + '.*real')
                        self.assert_rejected(lambda: model.fit(data, self.y, offset_init=value), 'offset_init.*real')

    def test_object_guard_handles_shared_and_cyclic_containers(self):
        shared = nested_object(.2, 2)
        real = np.empty(2, dtype=object)
        real[:] = [shared, shared]
        self.assertFalse(contains_complex(real))
        cycle = np.empty(2, dtype=object)
        cycle[0], cycle[1] = cycle, .2
        self.assertFalse(contains_complex(cycle))
        cycle[1] = np.complex128(.2)
        self.assertTrue(contains_complex(cycle))

    def test_complex_labels_reject_across_public_estimators_and_utilities(self):
        for invalid in complex_arrays(self.y):
            calls = (
                lambda: GenDWD(max_iter=0).fit(self.X, invalid),
                lambda: GenDWDCV().fit(self.X, invalid),
                lambda: KernGDWD(max_iter=0).fit(self.X, invalid),
                lambda: KernGDWDCV().fit(self.X, invalid),
                lambda: KernMD().fit(self.X, invalid),
                lambda: solve_gen_dwd(self.X, invalid, 1., max_iter=0),
                lambda: solve_gen_kern_dwd(self.K, invalid, 1., max_iter=0),
                lambda: kern_md(self.K, invalid),
                lambda: pm1(invalid),
                lambda: run_cv(GenDWD(max_iter=0), self.X, invalid, {'lambd': [1.]}),
            )
            for index, call in enumerate(calls):
                with self.subTest(route=index, value=repr(invalid)):
                    self.assert_rejected(call)

    @unittest.skipUnless(importlib.util.find_spec('cvxpy'), 'CVXPY is an optional dependency.')
    def test_socp_entry_points_reject_complex_labels_before_solver(self):
        from dwd.socp_dwd import DWD, solve_dwd_socp, auto_dwd_C
        from dwd.svm import SVM, solve_svm
        # These validation paths do not require an installed optimization solver.
        for invalid in complex_arrays(self.y):
            for call in (lambda: DWD().fit(self.X, invalid),
                         lambda: solve_dwd_socp(self.X, invalid, 1.),
                         lambda: auto_dwd_C(self.X, invalid),
                         lambda: SVM().fit(self.X, invalid),
                         lambda: solve_svm(self.X, invalid, 1.)):
                self.assert_rejected(call)

    def test_validation_labels_and_private_solver_arrays_reject_before_cast(self):
        for invalid in complex_arrays(self.y):
            self.assert_rejected(lambda: KernGDWD(max_iter=0, stopping='validation').fit(
                self.X, self.y, validation_data=(self.X, invalid)))
            self.assert_rejected(lambda: solve_kernel(self.K, invalid, 1., max_iter=0))
            self.assert_rejected(lambda: solve_kernel(self.K, self.y, 1., max_iter=0,
                stopping='validation', validation=(self.K, invalid)))
        for invalid in complex_arrays(self.K):
            self.assert_rejected(lambda: solve_kernel(invalid, self.y, 1., max_iter=0))
            self.assert_rejected(lambda: solve_kernel(self.K, self.y, 1., max_iter=0,
                stopping='validation', validation=(invalid, self.y)))

    def test_real_numeric_and_string_labels_remain_supported(self):
        expected = GenDWD(max_iter=2).fit(self.X, self.y).decision_function(self.X)
        for labels in (self.y, self.y.astype(object), ['left', 'right']):
            model = GenDWD(max_iter=2).fit(self.X, labels)
            np.testing.assert_array_equal(model.decision_function(self.X), expected)
            np.testing.assert_array_equal(pm1(labels), [-1., 1.])


if __name__ == '__main__':
    unittest.main()
