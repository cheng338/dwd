"""The unfitted kernel mixin must refuse without inspecting or changing data."""
import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from dwd.kernel_utils import KernelClfMixin


class _UnimplementedKernelClassifier(KernelClfMixin, BaseEstimator):
    pass


@pytest.mark.parametrize('seeded', [False, True])
def test_unimplemented_fit_preserves_state_and_refuses(seeded):
    model = _UnimplementedKernelClassifier()
    model.marker = object()
    if seeded:
        for name in ('classes_', 'intercept_', 'dual_coef_', '_KernelClfMixin__Xfit'):
            setattr(model, name, object())
    before = vars(model).copy()
    X = np.array([[1., 2.], [3., 4.]])
    y = np.array([-7, 9])
    with pytest.raises(NotImplementedError) as caught:
        model.fit(X, y)
    assert caught.value.args == ()
    assert vars(model).keys() == before.keys()
    assert all(vars(model)[key] is value for key, value in before.items())
    np.testing.assert_array_equal(X, [[1., 2.], [3., 4.]])
    np.testing.assert_array_equal(y, [-7, 9])
    if not seeded:
        with pytest.raises(NotFittedError):
            check_is_fitted(model)


def test_unimplemented_fit_does_not_evaluate_labels():
    class PoisonLabels:
        def __array__(self, *args, **kwargs):
            raise AssertionError('An unimplemented fit must not convert labels')

    model = _UnimplementedKernelClassifier()
    with pytest.raises(NotImplementedError):
        model.fit(object(), PoisonLabels())
    assert vars(model) == {}


def test_unimplemented_fit_does_not_invoke_attribute_writes():
    class GuardedClassifier(_UnimplementedKernelClassifier):
        def __setattr__(self, name, value):
            raise AssertionError('An unimplemented fit must not assign fitted state')

    with pytest.raises(NotImplementedError):
        GuardedClassifier().fit([[1.], [2.]], [-1, 1])
