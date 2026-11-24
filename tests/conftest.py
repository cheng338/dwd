"""Opt-in resource fixtures for tests that use sklearn's timed final refit."""
import pytest
from threadpoolctl import threadpool_limits

from _search_resources import single_worker_search_parent


@pytest.fixture(scope="module")
def search_parent_native_thread():
    # Transport probes use serial dummy estimators. Their search job counts
    # and backend selection (including ambient n_jobs) are under test.
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def numerical_search_parent():
    with single_worker_search_parent():
        yield
