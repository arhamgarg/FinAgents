import pytest


@pytest.fixture(autouse=True)
def _hermetic_provider_guard(hermetic_provider_guard):
    yield
