"""Shared isolation for the process-wide default profiling manager."""

import pytest

from scope_profiler import ProfileManager


@pytest.fixture(autouse=True)
def _reset_default_profile_manager():
    """Keep lifecycle state from leaking between otherwise independent tests."""
    ProfileManager._reset()
    yield
    ProfileManager._reset()
