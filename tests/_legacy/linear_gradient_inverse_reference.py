"""Linear MM gradient reuse preserves trajectories and observable hook behavior."""
from contextlib import contextmanager
import math
import sys
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csr_matrix, issparse

import dwd.gen_dwd as gdwd


def _reference_gamma(X, y, beta, offset, lambd, q, X_beta):
    """Original step formula: independently recompute the loss gradient."""
    if X_beta is None:
        X_beta = X.dot(beta)
    z = y * gdwd.V_grad(y * (X_beta + offset), q=q) / X.shape[0]
    return np.insert(X.T.dot(z) + 2 * lambd * beta, 0, z.sum())


def _reference_implicit(X, y, beta, offset, lambd, q, U, v, g, pi,
                        UP=None, X_beta=None):
    gamma = _reference_gamma(X, y, beta, offset, lambd, q, X_beta)
    if UP is None:
        UP = np.multiply(U, 1.0 / pi)
    inverse_product = UP.dot(U.T.dot(gamma)) + g * v * v.T.dot(gamma)
    curvature = float(np.float64(q) + 2.0 + 1.0 / np.float64(q))
    step = (X.shape[0] / curvature) * inverse_product
    return step[1:], step[0]


def _reference_explicit(X, y, beta, offset, lambd, q, P_inv, X_beta=None):
    gamma = _reference_gamma(X, y, beta, offset, lambd, q, X_beta)
    curvature = float(np.float64(q) + 2.0 + 1.0 / np.float64(q))
    step = (X.shape[0] / curvature) * P_inv @ gamma
    return step[1:], step[0]


@contextmanager
def _step_reference(implicit, enabled=True):
    if not enabled:
        yield
        return
    name = 'get_step_implicit' if implicit else 'get_step_explicit'
    reference = _reference_implicit if implicit else _reference_explicit
    with patch.object(gdwd, name, reference):
        yield


def _encoded(value):
    if issparse(value):
        return ('csr', value.shape, _encoded(value.data),
                _encoded(value.indices), _encoded(value.indptr))
    if isinstance(value, np.ndarray):
        if value.dtype.kind == 'f' and value.dtype.itemsize > np.dtype(np.float64).itemsize:
            # Wider floating storage can contain unspecified padding bytes.
            return ('wide_array', value.shape, value.dtype.str,
                    tuple(_encoded(item) for item in value.ravel()))
        return ('array', value.shape, value.dtype.str, value.tobytes())
    if isinstance(value, np.floating):
        if value.dtype.itemsize > np.dtype(np.float64).itemsize:
            return ('wide_scalar', value.dtype.str, value, bool(np.signbit(value)))
        return ('scalar', value.dtype.str, value.tobytes())
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, dict):
        return {key: _encoded(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_encoded(item) for item in value)
    return value


def _model_state(model):
    return _encoded({key: value for key, value in vars(model).items()
                     if key.endswith('_')})


def _without_runtime(value):
    if isinstance(value, dict):
        return {key: _without_runtime(item) for key, item in value.items()
                if key not in ('runtime', 'init_time', 'mean_runtime', 'std_runtime')}
    if isinstance(value, (tuple, list)):
        return [_without_runtime(item) for item in value]
    return value


def _fit(X, y, *, implicit=True, reference=False, options=None, fit_options=None,
         cached=False):
    settings = dict(lambd=.07, q=1., stopping='optimality', tol=0.,
                    max_iter=6, random_state=12, implicit_P=implicit)
    settings.update(options or {})
    model = gdwd.GenDWD(**settings)
    if cached:
        model.cv_init(X)
    with _step_reference(implicit, reference):
        model.fit(X, y, **(fit_options or {}))
    return model


@contextmanager
def _observe():
    """Observe built-ins without replacing them and disabling eligibility."""
    record = {'loss_calls': 0, 'gradients': [], 'steps': [], 'reuses': 0}
    gradient_code = gdwd._BUILTIN_LINEAR_GRADIENT.__code__
    loss_code = gdwd._BUILTIN_LOSS_GRADIENT.__code__
    step_codes = (gdwd._BUILTIN_IMPLICIT_STEP.__code__,
                  gdwd._BUILTIN_EXPLICIT_STEP.__code__)

    def observer(frame, event, result):
        if event == 'call' and frame.f_code is loss_code:
            record['loss_calls'] += 1
        if event == 'return' and frame.f_code is gradient_code and result is not None:
            record['gradients'].append({
                'iterate': _encoded((frame.f_locals['beta'], frame.f_locals['offset'],
                                     frame.f_locals['X_beta'])),
                'beta': frame.f_locals['beta'].copy(),
                'offset': frame.f_locals['offset'],
                'gradient': _encoded(result[:2]),
            })
        if event == 'call' and frame.f_code in step_codes:
            supplied = frame.f_locals.get('_gradient')
            record['steps'].append(supplied is not None)
            if supplied is not None:
                latest = record['gradients'][-1]
                iterate = _encoded((frame.f_locals['beta'], frame.f_locals['offset'],
                                    frame.f_locals['X_beta']))
                if latest['iterate'] != iterate or latest['gradient'] != _encoded(supplied):
                    raise AssertionError('A step received a gradient from a different iterate.')
                record['reuses'] += 1

    previous = sys.getprofile()
    sys.setprofile(observer)
    try:
        yield record
    finally:
        sys.setprofile(previous)


def _gradient_oracle(X, y, beta, intercept, q, lambd):
    rows = X.toarray() if issparse(X) else X
    z = []
    threshold = q / (q + 1.)
    for row, label in zip(rows, y):
        margin = label * (math.fsum(float(a) * float(b) for a, b in zip(row, beta))
                          + float(intercept))
        derivative = -1. if margin <= threshold else -(threshold / margin) ** (q + 1.)
        z.append(float(label) * derivative / len(y))
    return np.array([math.fsum(z)] + [
        math.fsum(float(row[j]) * value for row, value in zip(rows, z)) + 2 * lambd * beta[j]
        for j in range(len(beta))])


class LinearGradientReuseTests(unittest.TestCase):
    def setUp(self):
        self.X = np.random.RandomState(61006).normal(size=(32, 4)) + np.arange(4) / 8.
        self.y = np.tile([-1, 1], 16)

    def assertModelsEqual(self, expected, actual, X):
        self.assertEqual(_model_state(expected), _model_state(actual))
        self.assertEqual(_encoded(expected.decision_function(X)),
                         _encoded(actual.decision_function(X)))
        assert_array_equal(expected.predict(X), actual.predict(X))

    def test_trajectory_matches_independent_original_step(self):
        for sparse in (False, True):
            X = csr_matrix(self.X) if sparse else self.X
            before = _encoded(X)
            for mode in ('schur', 'legacy'):
                for implicit in (True, False):
                    for q in (.5, 1., 2.3):
                        with self.subTest(sparse=sparse, mode=mode, implicit=implicit, q=q):
                            options = dict(solver_mode=mode, q=q)
                            expected = _fit(X, self.y, implicit=implicit, reference=True, options=options)
                            actual = _fit(X, self.y, implicit=implicit, options=options)
                            self.assertModelsEqual(expected, actual, X)
                            self.assertEqual(before, _encoded(X))

    def test_reused_gradient_belongs_to_current_dense_or_sparse_iterate(self):
        for sparse in (False, True):
            X = csr_matrix(self.X) if sparse else self.X
            for implicit in (True, False):
                with self.subTest(sparse=sparse, implicit=implicit):
                    with _observe() as actual:
                        model = _fit(X, self.y, implicit=implicit)
                    with _observe() as reference:
                        _fit(X, self.y, implicit=implicit, reference=True)
                    self.assertEqual(model.n_iter_, 6)
                    self.assertEqual(actual['reuses'], model.n_iter_)
                    self.assertEqual(actual['loss_calls'], model.n_iter_ + 1)
                    self.assertEqual(reference['loss_calls'] - actual['loss_calls'], model.n_iter_)

    def test_objective_and_fixed_stopping_keep_original_step_path(self):
        for stopping in ('objective', 'fixed'):
            for implicit in (True, False):
                with self.subTest(stopping=stopping, implicit=implicit):
                    options = dict(stopping=stopping, obj_tol=0.)
                    with _observe() as observed:
                        actual = _fit(self.X, self.y, implicit=implicit, options=options)
                    expected = _fit(self.X, self.y, implicit=implicit, options=options, reference=True)
                    self.assertModelsEqual(expected, actual, self.X)
                    self.assertEqual(observed['reuses'], 0)
                    self.assertEqual(len(observed['gradients']), 1)
                    self.assertEqual(observed['loss_calls'], actual.n_iter_ + 1)

    def test_zero_early_and_one_update_recompute_returned_model_diagnostics(self):
        for implicit in (True, False):
            for options, updates, gradients in ((dict(max_iter=0), 0, 1),
                                                (dict(tol=1e100), 0, 2),
                                                (dict(max_iter=1), 1, 2)):
                with self.subTest(implicit=implicit, options=options):
                    with _observe() as observed:
                        model = _fit(self.X, self.y, implicit=implicit, options=options)
                    self.assertEqual(model.n_iter_, updates)
                    self.assertEqual(observed['reuses'], updates)
                    self.assertEqual(len(observed['gradients']), gradients)
                    last = observed['gradients'][-1]
                    assert_array_equal(last['beta'], model.coef_.ravel())
                    self.assertEqual(float(last['offset']), float(model.intercept_[0]))
                    gradient = _gradient_oracle(self.X, self.y, model.coef_.ravel(),
                                                model.intercept_[0], 1., .07)
                    assert_allclose(model.gradient_inf_norm_, np.max(np.abs(gradient)), rtol=1e-13, atol=1e-14)
                    if updates:
                        self.assertNotEqual(_encoded(last['beta']), _encoded(observed['gradients'][0]['beta']))

    def test_cached_preparation_preserves_optimality_trajectory(self):
        for implicit in (True, False):
            with self.subTest(implicit=implicit):
                actual = _fit(self.X, self.y, implicit=implicit, cached=True)
                expected = _fit(self.X, self.y, implicit=implicit, cached=True, reference=True)
                self.assertModelsEqual(expected, actual, self.X)

    def test_cv_parameter_selection_and_predictions_match_original_steps(self):
        for implicit in (True, False):
            with self.subTest(implicit=implicit):
                options = dict(lambd_vals=[.03, .3], q_vals=[.5, 2.], cv=2,
                               implicit_P=implicit, stopping='optimality', tol=0.,
                               max_iter=4, random_state=7)
                with _step_reference(implicit):
                    expected = gdwd.GenDWDCV(**options).fit(self.X, self.y)
                actual = gdwd.GenDWDCV(**options).fit(self.X, self.y)
                self.assertEqual(expected.best_params_, actual.best_params_)
                self.assertEqual(expected.best_score_, actual.best_score_)
                self.assertModelsEqual(expected.best_estimator_, actual.best_estimator_, self.X)
                # Runtime columns are observations, not predictions or selection results.
                for name in ('all_cv_results_', 'agg_cv_results_'):
                    a, b = getattr(expected, name), getattr(actual, name)
                    self.assertEqual(_encoded(_without_runtime(a)),
                                     _encoded(_without_runtime(b)))

    def _hook_fit(self, implicit, name, *, reference=False, restore=False):
        calls = []
        with _step_reference(implicit, reference):
            original = getattr(gdwd, name)

            def replacement(*args, **kwargs):
                self.assertNotIn('_gradient', kwargs)
                calls.append(1)
                if restore:
                    setattr(gdwd, name, original)
                result = original(*args, **kwargs)
                if restore and name == 'V_grad':
                    return result * .875
                if restore and name == '_linear_gradient':
                    return result[0] + .125, result[1] * .875, result[2]
                return result

            with patch.object(gdwd, name, replacement), _observe() as observed:
                model = _fit(self.X, self.y, implicit=implicit)
        return model, len(calls), observed

    def test_replaced_hooks_receive_original_calls(self):
        for implicit in (True, False):
            for name in ('V_grad', '_linear_gradient',
                         'get_step_implicit' if implicit else 'get_step_explicit'):
                with self.subTest(implicit=implicit, name=name):
                    expected, old_calls, _ = self._hook_fit(implicit, name, reference=True)
                    actual, calls, observed = self._hook_fit(implicit, name)
                    self.assertModelsEqual(expected, actual, self.X)
                    self.assertEqual(calls, old_calls)
                    self.assertEqual(observed['reuses'], 0)

    def test_self_restoring_hooks_do_not_reuse_custom_gradient(self):
        for implicit in (True, False):
            for name in ('V_grad', '_linear_gradient',
                         'get_step_implicit' if implicit else 'get_step_explicit'):
                with self.subTest(implicit=implicit, name=name):
                    expected, old_calls, _ = self._hook_fit(implicit, name, reference=True, restore=True)
                    actual, calls, observed = self._hook_fit(implicit, name, restore=True)
                    self.assertModelsEqual(expected, actual, self.X)
                    self.assertEqual((calls, old_calls), (1, 1))
                    self.assertFalse(observed['steps'][0])

    def test_ineligible_dtype_uses_original_step_evaluations(self):
        # Exercise the eligibility branch even where longdouble is only float64.
        for implicit in (True, False):
            with self.subTest(implicit=implicit):
                expected = _fit(self.X, self.y, implicit=implicit, reference=True)
                with patch.object(gdwd, '_REUSE_GRADIENT_DTYPE', np.dtype(np.float32)), _observe() as observed:
                    actual = _fit(self.X, self.y, implicit=implicit)
                self.assertModelsEqual(expected, actual, self.X)
                self.assertEqual(observed['reuses'], 0)
                self.assertEqual(observed['loss_calls'], 2 * actual.n_iter_ + 1)

    @unittest.skipUnless(np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant,
                         'This platform has no wider-than-float64 longdouble arithmetic.')
    def test_wider_legacy_gradient_retains_original_precision(self):
        X = self.X.astype(np.longdouble)
        X[:, 0] += 3 * np.finfo(np.longdouble).eps
        # The legacy explicit inverse uses NumPy linalg.inv, which does not
        # support wider longdouble; exercise the supported implicit route.
        options = dict(solver_mode='legacy', lambd=np.longdouble('.07'))
        expected = _fit(X, self.y, options=options, reference=True)
        with _observe() as observed:
            actual = _fit(X, self.y, options=options)
        self.assertModelsEqual(expected, actual, X)
        self.assertEqual(observed['reuses'], 0)

    def _error_policy_fit(self, implicit, policy, action, reference):
        X, y = np.array([[-1.], [1.]]), np.array([-1, 1])
        counts = {'handler': 0, 'step': 0}

        def in_step():
            frame = sys._getframe(1)
            while frame:
                if frame.f_code.co_name in ('get_step_implicit', 'get_step_explicit',
                                           '_reference_implicit', '_reference_explicit'):
                    return True
                frame = frame.f_back
            return False

        class Handler:
            def act(self):
                counts['handler'] += 1
                step = in_step()
                counts['step'] += int(step)
                if action == 'disable_policy':
                    np.seterr(under='ignore')
                if step and action == 'raise':
                    raise RuntimeError('step handler requested failure')
                if step and action == 'mutate':
                    X[0, 0] -= 3.

            def __call__(self, error, flag):
                self.act()

            def write(self, message):
                self.act()

        model = gdwd.GenDWD(q=1., lambd=.1, implicit_P=implicit,
                           stopping='optimality', tol=0., max_iter=1)
        old_policy, old_handler = np.geterr(), np.geterrcall()
        try:
            np.seterrcall(Handler())
            np.seterr(under=policy)
            with _step_reference(implicit, reference), _observe() as observed, warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                try:
                    model.fit(X, y, beta_init=np.array([0.]), offset_init=1e160)
                    result = ('returned', _model_state(model))
                except RuntimeError as exc:
                    result = ('raised', str(exc), sorted(k for k in vars(model) if k.endswith('_')))
        finally:
            np.seterr(**old_policy)
            np.seterrcall(old_handler)
        return result, _encoded(X), counts, observed, len(captured)

    def _check_error_action(self, action):
        for implicit in (True, False):
            for policy in ('call', 'log'):
                with self.subTest(implicit=implicit, policy=policy, action=action):
                    expected = self._error_policy_fit(implicit, policy, action, True)
                    actual = self._error_policy_fit(implicit, policy, action, False)
                    self.assertEqual(expected[:3], actual[:3])
                    self.assertEqual(expected[3]['loss_calls'], actual[3]['loss_calls'])
                    self.assertEqual(actual[3]['reuses'], 0)
                    if action == 'raise':
                        self.assertEqual(actual[0], ('raised', 'step handler requested failure', []))
                    elif action == 'mutate':
                        self.assertEqual(actual[1], _encoded(np.array([[-4.], [1.]])))
                        self.assertEqual(actual[2]['step'], 1)
                    else:
                        self.assertEqual(actual[2]['handler'], 1)
                        self.assertEqual(actual[3]['loss_calls'], 3)

    def test_raising_error_callback_and_logger_preserve_failure_cleanup(self):
        self._check_error_action('raise')

    def test_state_mutating_error_callback_and_logger_preserve_steps(self):
        self._check_error_action('mutate')

    def test_error_handler_disabling_policy_still_bypasses_reuse(self):
        self._check_error_action('disable_policy')

    def test_ordinary_underflow_warning_reduction_preserves_numerical_state(self):
        for implicit in (True, False):
            with self.subTest(implicit=implicit):
                expected = self._error_policy_fit(implicit, 'warn', 'none', True)
                actual = self._error_policy_fit(implicit, 'warn', 'none', False)
                self.assertEqual(expected[:3], actual[:3])
                self.assertEqual(expected[4], actual[4] + 1)
                self.assertEqual(actual[3]['reuses'], 1)


if __name__ == '__main__':
    unittest.main()
