"""Tests for environment metadata collection and its HDF5 round-trip."""

import getpass
import platform
import socket
from datetime import datetime

import h5py

from scope_profiler import ProfileManager, read_h5
from scope_profiler.metadata import (
    _ENVIRONMENT_VARIABLES,
    _MAX_VALUE_CHARS,
    collect_metadata,
)

# A representative slice of a module-based HPC environment.
SAMPLE_ENVIRONMENT = {
    "LOADEDMODULES": "profile/base:gcc/12.3.0:openmpi/4.1.6--gcc--12.3.0:python/3.11.7",
    "MODULEPATH": "/prod/opt/modulefiles/profiles:/prod/opt/modulefiles/base/tools",
    "MODULESHOME": "/prod/opt/environment/module/5.2.0/none",
    "MODULES_CMD": "/prod/opt/environment/module/5.2.0/none/libexec/modulecmd.tcl",
    "MODULES_RUN_QUARANTINE": "LD_LIBRARY_PATH LD_PRELOAD",
    "LD_LIBRARY_PATH": "/spack/python-3.11.7/lib:/spack/openmpi-4.1.6/lib",
    "PYTHON_HOME": "/spack/python-3.11.7",
    "PYTHON_INC": "/spack/python-3.11.7/include",
    "PYTHON_INCLUDE": "/spack/python-3.11.7/include",
    "PYTHON_LIB": "/spack/python-3.11.7/lib",
    "VIRTUAL_ENV": "/home/user/git_repos/project/.venv",
}

SAMPLE_SLURM = {
    "SLURM_JOB_ID": "1234567",
    "SLURM_JOB_NAME": "simulation",
    "SLURM_NNODES": "4",
    "SLURM_NTASKS": "128",
    "SLURM_CPUS_PER_TASK": "8",
    "SLURMD_NODENAME": "node0123",
}


def _apply_environment(monkeypatch, environment):
    for name, value in environment.items():
        monkeypatch.setenv(name, value)


def test_basic_fields(monkeypatch):
    monkeypatch.delenv("LOADEDMODULES", raising=False)
    metadata = collect_metadata(mpi_size=2)

    assert metadata["hostname"] == socket.gethostname()
    assert metadata["user"] == getpass.getuser()
    assert metadata["platform"] == platform.platform()
    assert metadata["python_version"] == platform.python_version()
    assert metadata["scope_profiler_version"]
    timestamp = datetime.fromisoformat(metadata["timestamp"])
    assert timestamp.utcoffset().total_seconds() == 0
    assert metadata["mpi_size"] == 2
    assert metadata["total_cores"] == 2 * metadata["omp_num_threads"]


def test_uname_and_chip_information():
    metadata = collect_metadata()

    # uname carries the whole tuple, so the system and node are both in there.
    assert platform.system() in metadata["uname"]
    assert platform.node() in metadata["uname"]

    # Best-effort, but it must always yield a non-empty string.
    assert isinstance(metadata["chip_information"], str)
    assert metadata["chip_information"]


def test_environment_variables_are_captured(monkeypatch):
    _apply_environment(monkeypatch, SAMPLE_ENVIRONMENT)
    metadata = collect_metadata()

    for name, value in SAMPLE_ENVIRONMENT.items():
        assert metadata[name] == value


def test_unset_environment_variables_are_omitted(monkeypatch):
    for name in _ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    metadata = collect_metadata()

    assert not any(name in metadata for name in _ENVIRONMENT_VARIABLES)


def test_slurm_variables_are_captured(monkeypatch):
    _apply_environment(monkeypatch, SAMPLE_SLURM)
    # A site-specific variable that is not in any hard-coded list.
    monkeypatch.setenv("SLURM_SITE_SPECIFIC_THING", "value")
    metadata = collect_metadata()

    for name, value in SAMPLE_SLURM.items():
        assert metadata[name] == value
    assert metadata["SLURM_SITE_SPECIFIC_THING"] == "value"


def test_no_slurm_variables_outside_a_job(monkeypatch):
    for name in list(SAMPLE_SLURM) + ["SLURM_SITE_SPECIFIC_THING"]:
        monkeypatch.delenv(name, raising=False)
    metadata = collect_metadata()

    assert not [key for key in metadata if key.startswith("SLURM")]


def test_modules_is_a_list(monkeypatch):
    monkeypatch.setenv("LOADEDMODULES", SAMPLE_ENVIRONMENT["LOADEDMODULES"])
    metadata = collect_metadata()

    assert metadata["modules"] == [
        "profile/base",
        "gcc/12.3.0",
        "openmpi/4.1.6--gcc--12.3.0",
        "python/3.11.7",
    ]


def test_modules_empty_without_module_system(monkeypatch):
    monkeypatch.delenv("LOADEDMODULES", raising=False)

    assert collect_metadata()["modules"] == []


def test_long_values_are_truncated(monkeypatch):
    monkeypatch.setenv("PATH", "/some/very/long/path" * 20_000)
    metadata = collect_metadata()

    # HDF5 attributes cap out at 64 KB; the value must stay storable.
    assert len(metadata["PATH"]) <= _MAX_VALUE_CHARS
    assert metadata["PATH"].endswith("...[truncated]")


def test_metadata_round_trips_through_hdf5(tmp_path, monkeypatch):
    _apply_environment(monkeypatch, SAMPLE_ENVIRONMENT)
    _apply_environment(monkeypatch, SAMPLE_SLURM)

    file_path = tmp_path / "profiling_data.h5"
    ProfileManager.setup(file_path=str(file_path))
    with ProfileManager.profile_region("region"):
        pass
    ProfileManager.finalize(verbose=False)

    metadata = read_h5(file_path).metadata

    assert metadata["SLURM_JOB_ID"] == "1234567"
    assert metadata["SLURMD_NODENAME"] == "node0123"
    assert metadata["VIRTUAL_ENV"] == SAMPLE_ENVIRONMENT["VIRTUAL_ENV"]
    assert metadata["LD_LIBRARY_PATH"] == SAMPLE_ENVIRONMENT["LD_LIBRARY_PATH"]
    assert metadata["chip_information"]
    assert platform.system() in metadata["uname"]
    # Stored as a real list of strings, not a packed string or byte array.
    assert metadata["modules"] == [
        "profile/base",
        "gcc/12.3.0",
        "openmpi/4.1.6--gcc--12.3.0",
        "python/3.11.7",
    ]
    assert all(isinstance(module, str) for module in metadata["modules"])


def test_empty_modules_round_trip(tmp_path, monkeypatch):
    """An empty list has no dtype for h5py to infer — it must still store."""
    monkeypatch.delenv("LOADEDMODULES", raising=False)

    file_path = tmp_path / "no_modules.h5"
    ProfileManager.setup(file_path=str(file_path))
    with ProfileManager.profile_region("region"):
        pass
    ProfileManager.finalize(verbose=False)

    with h5py.File(file_path, "r") as handle:
        assert handle["metadata"].attrs["modules"].shape == (0,)

    assert read_h5(file_path).metadata["modules"] == []
