"""Estimator capability metadata must describe the supported input contracts."""
from dataclasses import asdict

import numpy as np
import pytest
from scipy import sparse
from sklearn.base import ClassifierMixin, clone, is_classifier
from sklearn.utils import get_tags
from sklearn.utils.extmath import safe_sparse_dot

from dwd.gen_dwd import GenDWD, GenDWDCV
from dwd.gen_kern_dwd import KernGDWD, KernGDWDCV
from dwd.kern_md import KernMD
from dwd.kernel_utils import KernelScaler


SPARSE_KERNELS = ('linear', 'poly', 'polynomial', 'rbf', 'laplacian',
                  'sigmoid', 'cosine')
DENSE_KERNELS = ('precomputed', 'chi2', 'additive_chi2', 'not_a_known_kernel')
KERNEL_CLASSES = (KernGDWD, KernGDWDCV)
BINARY_CLASSES = (GenDWD, GenDWDCV, KernGDWD, KernGDWDCV, KernMD)


@pytest.fixture(params=(*BINARY_CLASSES, 'DWD', 'SVM'),
                ids=lambda value: value if isinstance(value, str) else value.__name__)
def classifier_class(request):
    """Keep the conic dependency optional without skipping the core estimators."""
    if not isinstance(request.param, str):
        return request.param
    pytest.importorskip('cvxpy', reason='Optional conic dependency is not installed',
                        exc_type=ImportError)
    if request.param == 'DWD':
        from dwd.socp_dwd import DWD
        return DWD
    from dwd.svm import SVM
    return SVM


def test_all_classifiers_advertise_binary_targets(classifier_class):
    obj = classifier_class()
    tags = get_tags(obj)
    assert is_classifier(obj)
    assert tags.estimator_type == 'classifier'
    assert tags.target_tags.required is True
    assert tags.classifier_tags.multi_class is False


def test_default_sparse_and_pairwise_contracts(classifier_class):
    tags = get_tags(classifier_class())
    assert tags.input_tags.sparse is (classifier_class is not KernMD)
    assert tags.input_tags.pairwise is False


@pytest.mark.parametrize('cls', (GenDWD, GenDWDCV))
@pytest.mark.parametrize('implicit', [True, False])
def test_linear_sparse_tag_covers_both_supported_algorithms(cls, implicit):
    tags = get_tags(cls(implicit_P=implicit))
    assert tags.input_tags.sparse is True
    assert tags.input_tags.pairwise is False


@pytest.mark.parametrize('cls', KERNEL_CLASSES)
@pytest.mark.parametrize('kernel', (*SPARSE_KERNELS, *DENSE_KERNELS))
def test_kernel_sparse_and_pairwise_tags_follow_configured_input(cls, kernel):
    tags = get_tags(cls(kernel=kernel))
    assert tags.input_tags.sparse is (kernel in SPARSE_KERNELS)
    assert tags.input_tags.pairwise is (kernel == 'precomputed')


@pytest.mark.parametrize('cls', KERNEL_CLASSES)
@pytest.mark.parametrize('backend', ['auto', 'cholesky', 'spectral', 'lbfgs'])
@pytest.mark.parametrize('computation', ['standard', 'direct'])
def test_rbf_sparse_tag_is_independent_of_solver_and_arithmetic_route(cls, backend, computation):
    tags = get_tags(cls(kernel='rbf', backend=backend, rbf_computation=computation))
    assert tags.input_tags.sparse is True
    assert tags.input_tags.pairwise is False


@pytest.mark.parametrize('kernel', (*SPARSE_KERNELS, *DENSE_KERNELS))
def test_kernmd_does_not_advertise_sparse_training_from_query_support(kernel):
    tags = get_tags(KernMD(kernel=kernel))
    assert tags.input_tags.sparse is False
    assert tags.input_tags.pairwise is (kernel == 'precomputed')


@pytest.mark.parametrize('cls', KERNEL_CLASSES)
def test_tags_refresh_after_parameter_changes_without_mutating_old_tags(cls):
    obj = cls(kernel='linear')
    before = get_tags(obj)
    obj.set_params(kernel='precomputed')
    assert get_tags(obj).input_tags.pairwise is True
    assert get_tags(obj).input_tags.sparse is False
    assert before.input_tags.pairwise is False
    assert before.input_tags.sparse is True
    obj.set_params(kernel='chi2')
    assert get_tags(obj).input_tags.sparse is False
    assert get_tags(obj).input_tags.pairwise is False
    obj.set_params(kernel='rbf')
    assert get_tags(obj).input_tags.sparse is True
    assert get_tags(obj).input_tags.pairwise is False


@pytest.mark.parametrize('cls', (*KERNEL_CLASSES, KernMD))
def test_callable_policy_does_not_execute_or_replace_the_callback(cls):
    calls = []

    def matrix_callback(X, Y, **kwargs):
        calls.append((X, Y, kwargs))
        raise AssertionError('A metadata request must never call the user kernel')

    obj = cls(kernel=matrix_callback)
    tags = get_tags(obj)
    # DWD accepts and forwards CSR features. The callable is responsible for
    # supporting them and returning a valid dense training Gram matrix.
    # KernMD rejects sparse training features before invoking its callable.
    assert tags.input_tags.sparse is (cls in KERNEL_CLASSES)
    assert tags.input_tags.pairwise is False
    copied = clone(obj)
    assert asdict(get_tags(copied)) == asdict(tags)
    assert copied.get_params(deep=False)['kernel'] is matrix_callback
    assert obj.get_params(deep=False)['kernel'] is matrix_callback
    assert calls == []


def test_each_tag_request_returns_independent_metadata(classifier_class):
    obj, other = classifier_class(), classifier_class()
    expected = asdict(get_tags(obj))
    changed = get_tags(obj)
    changed.input_tags.sparse = not changed.input_tags.sparse
    changed.input_tags.pairwise = not changed.input_tags.pairwise
    changed.classifier_tags.multi_class = not changed.classifier_tags.multi_class
    assert asdict(get_tags(obj)) == expected
    assert asdict(get_tags(other)) == expected


def test_capability_tags_preserve_unrelated_classifier_metadata(classifier_class):
    obj = classifier_class()
    actual = asdict(get_tags(obj))
    inherited = asdict(ClassifierMixin.__sklearn_tags__(obj))
    for metadata in (actual, inherited):
        metadata['classifier_tags'].pop('multi_class')
        metadata['input_tags'].pop('sparse')
        metadata['input_tags'].pop('pairwise')
    assert actual == inherited


def test_clone_and_tag_queries_do_not_mutate_parameters_or_fit(classifier_class):
    obj = classifier_class()
    before = obj.get_params(deep=False)
    expected = asdict(get_tags(obj))
    copied = clone(obj)
    assert asdict(get_tags(copied)) == expected
    after = obj.get_params(deep=False)
    assert set(after) == set(before)
    assert all(after[key] is value for key, value in before.items())
    assert not any(name.endswith('_') for name in vars(obj))
    assert not any(name.endswith('_') for name in vars(copied))


def test_kernel_scaler_remains_a_dense_transformer():
    obj = KernelScaler()
    tags = get_tags(obj)
    assert not is_classifier(obj)
    assert tags.classifier_tags is None
    assert tags.input_tags.sparse is False
    assert tags.transformer_tags is not None


@pytest.fixture
def small_binary_data():
    X = np.array([[-2., 0.], [-1., 1.], [-1., 0.],
                  [1., 0.], [1., -1.], [2., 0.]])
    return X, np.array([-1, -1, -1, 1, 1, 1])


@pytest.mark.parametrize('cls', (GenDWD, KernGDWD))
def test_sparse_training_tag_corresponds_to_accepted_fit_and_query(cls, small_binary_data):
    X, y = small_binary_data
    obj = cls(max_iter=3, stopping='fixed', random_state=23)
    assert get_tags(obj).input_tags.sparse is True
    obj.fit(sparse.csr_matrix(X), y)
    scores = obj.decision_function(sparse.csr_matrix(X))
    assert scores.shape == y.shape
    assert np.isfinite(scores).all()
    np.testing.assert_array_equal(obj.predict(sparse.csr_matrix(X)),
                                  obj.classes_[(scores > 0).astype(int)])


def test_callable_sparse_tag_corresponds_to_forwarded_csr_features(small_binary_data):
    X, y = small_binary_data
    calls = []

    def sparse_linear_kernel(A, B):
        calls.append((sparse.issparse(A), sparse.issparse(B)))
        return safe_sparse_dot(A, B.T, dense_output=True)

    obj = KernGDWD(kernel=sparse_linear_kernel, max_iter=3, stopping='fixed')
    assert get_tags(obj).input_tags.sparse is True
    assert calls == []
    obj.fit(sparse.csr_matrix(X), y)
    assert np.isfinite(obj.decision_function(sparse.csr_matrix(X))).all()
    assert calls and all(left and right for left, right in calls)


@pytest.mark.parametrize('cls', (GenDWD, KernGDWD, KernMD))
def test_binary_tag_corresponds_to_multiclass_refusal(cls, small_binary_data):
    X, _ = small_binary_data
    obj = cls()
    assert get_tags(obj).classifier_tags.multi_class is False
    with pytest.raises(ValueError, match='exactly two classes'):
        obj.fit(X, np.array([0, 1, 2, 0, 1, 2]))


@pytest.mark.parametrize('cls', (GenDWDCV, KernGDWDCV))
def test_binary_cv_tag_corresponds_to_multiclass_refusal(cls, small_binary_data):
    X, _ = small_binary_data
    # GenDWDCV delegates this rejection to its binary base estimator. Use a
    # valid small split so the test reaches that boundary rather than failing
    # because the default fold count exceeds the observations per class.
    obj = cls(cv=2, lambd_vals=(1.0,), q_vals=(1.0,),
              max_iter=3, stopping='fixed', random_state=23)
    assert get_tags(obj).classifier_tags.multi_class is False
    with pytest.raises(ValueError, match='exactly two classes'):
        obj.fit(X, np.array([0, 1, 2, 0, 1, 2]))
