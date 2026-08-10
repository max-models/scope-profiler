"""Tests for MPI-launcher detection (``scope_profiler.mpi_launch``)."""

import sys

import pytest

from scope_profiler.mpi_launch import (
    _OVERRIDE_ENV_VAR,
    LAUNCHER_ENV_VARS,
    get_comm,
    launched_under_mpi,
)


@pytest.fixture
def clean_env(monkeypatch):
    """An environment with no launcher variables and no override."""
    for var in LAUNCHER_ENV_VARS + (_OVERRIDE_ENV_VAR,):
        monkeypatch.delenv(var, raising=False)
    # A previous test (or the application) may have imported mpi4py; hide it
    # so detection sees a genuinely serial process.
    monkeypatch.delitem(sys.modules, "mpi4py.MPI", raising=False)
    return monkeypatch


def test_serial_run_is_not_mpi(clean_env):
    assert launched_under_mpi() is False
    assert get_comm() is None


@pytest.mark.parametrize("var", LAUNCHER_ENV_VARS)
def test_launcher_variables_are_detected(clean_env, var):
    clean_env.setenv(var, "0")
    assert launched_under_mpi() is True


def test_override_forces_mpi_on(clean_env):
    clean_env.setenv(_OVERRIDE_ENV_VAR, "1")
    assert launched_under_mpi() is True


def test_override_forces_mpi_off(clean_env):
    clean_env.setenv("OMPI_COMM_WORLD_RANK", "0")
    clean_env.setenv(_OVERRIDE_ENV_VAR, "false")
    assert launched_under_mpi() is False


def test_unrecognized_override_falls_back_to_detection(clean_env):
    clean_env.setenv(_OVERRIDE_ENV_VAR, "maybe")
    assert launched_under_mpi() is False


def test_already_initialized_mpi_is_used(clean_env):
    class FakeMPI:
        @staticmethod
        def Is_initialized():
            return True

    clean_env.setitem(sys.modules, "mpi4py.MPI", FakeMPI)
    assert launched_under_mpi() is True


def test_imported_but_uninitialized_mpi_is_not_used(clean_env):
    class FakeMPI:
        @staticmethod
        def Is_initialized():
            return False

    clean_env.setitem(sys.modules, "mpi4py.MPI", FakeMPI)
    assert launched_under_mpi() is False


def test_env_override_disables_mpi(clean_env):
    """SCOPE_PROFILER_MPI is the only way to overrule launcher detection."""
    clean_env.setenv("OMPI_COMM_WORLD_RANK", "0")
    clean_env.setenv("SCOPE_PROFILER_MPI", "0")
    assert get_comm() is None


def test_serial_setup_does_not_import_mpi4py(clean_env, tmp_path):
    """A serial ProfileManager.setup() must leave mpi4py unimported."""
    from scope_profiler import ProfileManager

    clean_env.delitem(sys.modules, "mpi4py", raising=False)

    ProfileManager.setup(file_path=str(tmp_path / "profiling_data.h5"))
    config = ProfileManager.get_config()

    assert config.comm is None
    assert config._rank == 0
    assert config._size == 1
    assert "mpi4py.MPI" not in sys.modules
