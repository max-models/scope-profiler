"""Tests for environment metadata collection and its HDF5 round-trip."""

import getpass
import os
import platform
import socket
import sys
from datetime import datetime

import h5py
import pytest

from scope_profiler import ProfileManager, read_h5
from scope_profiler.metadata import (
    _ENVIRONMENT_VARIABLES,
    _MAX_VALUE_CHARS,
    collect_metadata,
    count_nodes,
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
    metadata = collect_metadata(mpi_size=2, detail="full")

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
    metadata = collect_metadata(detail="full")

    # uname carries the whole tuple, so the system and node are both in there.
    assert platform.system() in metadata["uname"]
    assert platform.node() in metadata["uname"]

    # Best-effort, but it must always yield a non-empty string.
    assert isinstance(metadata["chip_information"], str)
    assert metadata["chip_information"]


def test_environment_variables_are_captured(monkeypatch):
    _apply_environment(monkeypatch, SAMPLE_ENVIRONMENT)
    metadata = collect_metadata(detail="full")

    for name, value in SAMPLE_ENVIRONMENT.items():
        assert metadata[name] == value


def test_unset_environment_variables_are_omitted(monkeypatch):
    for name in _ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    metadata = collect_metadata(detail="full")

    assert not any(name in metadata for name in _ENVIRONMENT_VARIABLES)


def test_slurm_variables_are_captured(monkeypatch):
    _apply_environment(monkeypatch, SAMPLE_SLURM)
    # A site-specific variable that is not in any hard-coded list.
    monkeypatch.setenv("SLURM_SITE_SPECIFIC_THING", "value")
    metadata = collect_metadata(detail="full")

    for name, value in SAMPLE_SLURM.items():
        assert metadata[name] == value
    assert metadata["SLURM_SITE_SPECIFIC_THING"] == "value"


def test_no_slurm_variables_outside_a_job(monkeypatch):
    for name in list(SAMPLE_SLURM) + ["SLURM_SITE_SPECIFIC_THING"]:
        monkeypatch.delenv(name, raising=False)
    metadata = collect_metadata(detail="full")

    assert not [key for key in metadata if key.startswith("SLURM")]


def test_modules_is_a_list(monkeypatch):
    monkeypatch.setenv("LOADEDMODULES", SAMPLE_ENVIRONMENT["LOADEDMODULES"])
    metadata = collect_metadata(detail="full")

    assert metadata["modules"] == [
        "profile/base",
        "gcc/12.3.0",
        "openmpi/4.1.6--gcc--12.3.0",
        "python/3.11.7",
    ]


def test_modules_empty_without_module_system(monkeypatch):
    monkeypatch.delenv("LOADEDMODULES", raising=False)

    assert collect_metadata(detail="full")["modules"] == []


def test_long_values_are_truncated(monkeypatch):
    monkeypatch.setenv("PATH", "/some/very/long/path" * 20_000)
    metadata = collect_metadata(detail="full")

    # HDF5 attributes cap out at 64 KB; the value must stay storable.
    assert len(metadata["PATH"]) <= _MAX_VALUE_CHARS
    assert metadata["PATH"].endswith("...[truncated]")


def test_metadata_round_trips_through_hdf5(tmp_path, monkeypatch):
    _apply_environment(monkeypatch, SAMPLE_ENVIRONMENT)
    _apply_environment(monkeypatch, SAMPLE_SLURM)

    file_path = tmp_path / "profiling_data.h5"
    ProfileManager.setup(file_path=str(file_path), metadata_detail="full")
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
    ProfileManager.setup(file_path=str(file_path), metadata_detail="full")
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
    "num_nodes",
}


def test_minimal_metadata_holds_only_run_shape_and_versions(monkeypatch):
    _apply_environment(monkeypatch, SAMPLE_ENVIRONMENT)
    _apply_environment(monkeypatch, SAMPLE_SLURM)

    metadata = collect_metadata(mpi_size=4, detail="minimal")

    # An MPI run's node count needs the other ranks; finalize() adds it.
    assert set(metadata) == MINIMAL_FIELDS - {"num_nodes"}
    assert metadata["mpi_size"] == 4
    assert metadata["total_cores"] == 4 * metadata["omp_num_threads"]


def test_minimal_metadata_is_the_default(monkeypatch):
    _apply_environment(monkeypatch, SAMPLE_ENVIRONMENT)

    assert set(collect_metadata()) == MINIMAL_FIELDS
    assert MINIMAL_FIELDS | IDENTIFYING_FIELDS <= set(collect_metadata(detail="full"))
    ProfileManager.setup(deactivate_file_output=True)
    assert ProfileManager.get_config().metadata_detail == "minimal"
    assert not IDENTIFYING_FIELDS & set(ProfileManager.get_config().metadata)


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


def test_minimal_metadata_keeps_source_paths_relative(tmp_path):
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
        # Inside the working directory: relative, so a report built from
        # there can still read the source, but no home directory.
        assert results[name].source_file == os.path.relpath(__file__)
        assert not os.path.isabs(results[name].source_file)
        assert results[name].source_lineno is not None


def test_minimal_metadata_names_a_file_outside_the_working_directory(
    tmp_path, monkeypatch
):
    file_path = tmp_path / "profiling_data.h5"
    monkeypatch.chdir(tmp_path)
    ProfileManager.setup(file_path=str(file_path), metadata_detail="minimal")
    with ProfileManager.profile_region("block"):
        pass
    ProfileManager.finalize(verbose=False)

    # The test module is imported from the src/ entry on sys.path, so it is
    # named by its module path.
    root = next(
        os.path.abspath(entry)
        for entry in sorted(sys.path, key=len, reverse=True)
        if entry and __file__.startswith(os.path.abspath(entry) + os.sep)
    )
    assert read_h5(file_path)["block"].source_file == os.path.relpath(__file__, root)


def test_recorded_path_outside_every_root_is_the_file_name(tmp_path, monkeypatch):
    from scope_profiler.profile_manager import _relative_source_path

    (tmp_path / "run").mkdir()
    monkeypatch.chdir(tmp_path / "run")
    monkeypatch.setattr(sys, "path", [str(tmp_path / "lib")])
    # The working directory first, then the import root, then the bare name.
    assert _relative_source_path(str(tmp_path / "run" / "sim.py")) == "sim.py"
    inside = tmp_path / "lib" / "pkg" / "mod.py"
    assert _relative_source_path(str(inside)) == os.path.join("pkg", "mod.py")
    assert _relative_source_path(str(tmp_path / "x" / "work.py")) == "work.py"


def test_minimal_metadata_can_be_set_from_toml(tmp_path):
    config_path = tmp_path / "profiling.toml"
    config_path.write_text("[profiling]\nmetadata_detail = 'minimal'\n")

    ProfileManager.setup(config_path=config_path, deactivate_file_output=True)

    assert ProfileManager.get_config().metadata_detail == "minimal"
    assert set(ProfileManager.get_config().metadata) - MINIMAL_FIELDS == {
        "start_time_ns",
    }


def test_line_profile_source_is_skipped_when_it_cannot_be_stored(tmp_path):
    from scope_profiler import profile_manager

    source = profile_manager.ProfileManager._line_profile_source
    assert source({"filename": "x.py", "first_lineno": 1, "line_numbers": []}) == {}
    # A file that cannot be read has nothing to store.
    assert (
        source({"filename": "/no/such.py", "first_lineno": 1, "line_numbers": [2]})
        == {}
    )
    # Nor does one too large for an HDF5 attribute.
    big = tmp_path / "big.py"
    big.write_text("x = '" + "a" * 70_000 + "'\n", encoding="utf-8")
    assert source({"filename": str(big), "first_lineno": 1, "line_numbers": [1]}) == {}

    assert profile_manager._is_package_file(profile_manager.__file__)
    assert not profile_manager._is_package_file(__file__)
    assert not profile_manager._is_package_file(tmp_path / "user.py")


def test_a_single_process_runs_on_one_node():
    assert collect_metadata()["num_nodes"] == 1
    assert collect_metadata(detail="full")["num_nodes"] == 1
    assert "num_nodes" not in collect_metadata(mpi_size=2)


class _HostComm:
    """Each rank's host name, as ``allgather`` would hand them back."""

    def __init__(self, hosts):
        self.hosts = hosts
        self.calls = 0

    def allgather(self, value):
        self.calls += 1
        return self.hosts


def test_count_nodes_counts_distinct_hosts():
    assert count_nodes(_HostComm(["n1", "n1", "n2", "n3", "n2"])) == 3
    assert count_nodes(_HostComm(["n1", "n1"])) == 1


def test_finalize_records_the_node_count_once_with_one_collective(
    tmp_path, monkeypatch
):
    from scope_profiler import h5writer
    from scope_profiler.tests.unit.test_payload_collection import FakeComm

    # The fake communicator only supports the gather-to-rank-0 writer; keep
    # MPI-enabled h5py builds (as in CI) off the parallel HDF5 path.
    monkeypatch.setattr(h5writer, "parallel_hdf5_available", lambda: False)

    class Comm(FakeComm):
        def __init__(self):
            super().__init__(rank=0, size=1)
            self.gathered = []

        def allgather(self, value):
            # This rank's host name, plus two ranks on another node.
            self.gathered.append(value)
            return [value, "other", "other"]

    file_path = tmp_path / "nodes.h5"
    ProfileManager.setup(file_path=str(file_path))
    config = ProfileManager.get_config()
    comm = Comm()
    config._comm = comm
    with ProfileManager.profile_region("region"):
        pass
    results = ProfileManager.finalize(verbose=False, return_results=True)

    assert comm.gathered == [socket.gethostname()]
    assert results.num_nodes == 2
    assert read_h5(file_path).metadata["num_nodes"] == 2
    assert read_h5(file_path).num_nodes == 2


def test_finalize_skips_the_node_count_when_nothing_is_kept():
    """No file and no results: nothing would carry the count, so no collective."""

    class Comm:
        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

        def allgather(self, value):
            raise AssertionError("no collective without output")

    ProfileManager.setup(deactivate_file_output=True)
    ProfileManager.get_config()._comm = Comm()
    with ProfileManager.profile_region("region"):
        pass
    assert ProfileManager.finalize(verbose=False) is None
