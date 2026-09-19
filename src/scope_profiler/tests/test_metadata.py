"""Tests for environment metadata collection and its HDF5 round-trip."""

import getpass
import os
import platform
import socket
from datetime import datetime

import h5py
import pytest

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


# Fields that identify a person, a machine, a directory or a job.
IDENTIFYING_FIELDS = {
    "user",
    "hostname",
    "uname",
    "working_directory",
    "modules",
    *_ENVIRONMENT_VARIABLES,
}

MINIMAL_FIELDS = {
    "timestamp",
    "platform",
    "chip_information",
    "python_version",
    "scope_profiler_version",
    "omp_num_threads",
    "mpi_size",
    "total_cores",
}


def test_minimal_metadata_holds_only_run_shape_and_versions(monkeypatch):
    _apply_environment(monkeypatch, SAMPLE_ENVIRONMENT)
    _apply_environment(monkeypatch, SAMPLE_SLURM)

    metadata = collect_metadata(mpi_size=4, detail="minimal")

    assert set(metadata) == MINIMAL_FIELDS
    assert metadata["mpi_size"] == 4
    assert metadata["total_cores"] == 4 * metadata["omp_num_threads"]


def test_full_metadata_is_the_default(monkeypatch):
    _apply_environment(monkeypatch, SAMPLE_ENVIRONMENT)

    assert collect_metadata().keys() == collect_metadata(detail="full").keys()
    assert MINIMAL_FIELDS | IDENTIFYING_FIELDS <= set(collect_metadata())


def test_unknown_metadata_detail_is_rejected():
    with pytest.raises(ValueError, match="metadata_detail"):
        collect_metadata(detail="everything")
    with pytest.raises(ValueError, match="metadata_detail"):
        ProfileManager.setup(metadata_detail="everything")


def test_minimal_metadata_leaves_no_trace_in_the_written_file(tmp_path, monkeypatch):
    _apply_environment(monkeypatch, SAMPLE_ENVIRONMENT)
    _apply_environment(monkeypatch, SAMPLE_SLURM)

    file_path = tmp_path / "profiling_data.h5"
    ProfileManager.setup(
        file_path=str(file_path),
        label="scaling",
        metadata_detail="minimal",
    )
    with ProfileManager.profile_region("region"):
        pass
    ProfileManager.finalize(verbose=False)

    metadata = read_h5(file_path).metadata

    assert not IDENTIFYING_FIELDS & set(metadata)
    assert not [key for key in metadata if key.startswith("SLURM")]
    assert MINIMAL_FIELDS <= set(metadata)
    # The run's own bookkeeping and the user's chosen label survive.
    assert metadata["label"] == "scaling"
    assert "start_time_ns" in metadata

    # Nothing identifying anywhere in the file's bytes, either.
    raw = file_path.read_bytes()
    # (Skip names too short to be distinguishable from arbitrary bytes.)
    for needle in (getpass.getuser(), socket.gethostname(), str(tmp_path)):
        if len(needle) >= 6:
            assert needle.encode() not in raw


def test_minimal_metadata_reduces_source_paths_to_file_names(tmp_path):
    file_path = tmp_path / "profiling_data.h5"
    ProfileManager.setup(file_path=str(file_path), metadata_detail="minimal")

    @ProfileManager.profile("decorated")
    def work():
        pass

    work()
    with ProfileManager.profile_region("block"):
        pass
    ProfileManager.finalize(verbose=False)

    results = read_h5(file_path)
    for name in ("decorated", "block"):
        assert results[name].source_file == os.path.basename(__file__)
        assert results[name].source_lineno is not None


def test_minimal_metadata_can_be_set_from_toml(tmp_path):
    config_path = tmp_path / "profiling.toml"
    config_path.write_text("[profiling]\nmetadata_detail = 'minimal'\n")

    ProfileManager.setup(config_path=config_path, deactivate_file_output=True)

    assert ProfileManager.get_config().metadata_detail == "minimal"
    assert set(ProfileManager.get_config().metadata) - MINIMAL_FIELDS == {
        "start_time_ns",
    }
