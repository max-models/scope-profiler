"""Importing the per-rank HDF5 files an ``SP_USE_HDF5`` C build writes.

A C build compiled with HDF5 writes ``<prefix>_rank<NNNNN>.h5`` -- already a
profile, one rank per file -- instead of a ``.spt`` trace, so ``import-native``
is a *merge* for that output rather than a conversion. What has to hold is
that the ranks of one run come back together, that the two formats mix in one
import, and that a merged profile lying in the same directory is not swallowed
as an input to the next import.

The files are written here with the Python writer rather than by compiling C,
so these run everywhere -- the compiled counterparts are in
``tests/test_c_hdf5_output.py``.
"""

import struct

import h5py
import numpy as np
import pytest

from scope_profiler import read_h5

from scope_profiler.h5writer import ProfilingWriter
from scope_profiler.native_trace import (
    MAGIC,
    TraceFormatError,
    find_traces,
    load_traces,
    read_native_h5,
)
from scope_profiler.profile_manager import RankPayload


def _write_rank_h5(path, rank, regions, metadata=None):
    """One rank's profile, the shape a C run's own output file has."""
    with ProfilingWriter(path, metadata or {"source": "native"}) as writer:
        writer.write_rank(
            rank,
            RankPayload(
                regions={
                    name: (
                        np.asarray(starts, dtype=np.int64),
                        np.asarray(ends, dtype=np.int64),
                    )
                    for name, (starts, ends) in regions.items()
                },
                likwid={},
                likwid_environment={},
            ),
        )
    return path


def _write_trace(path, rank, regions):
    """A version-2 ``.spt`` trace, for the mixed-format cases."""
    with open(path, "wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<i", 2))
        handle.write(struct.pack("<i", rank))
        handle.write(struct.pack("<q", len(regions)))
        for name, (starts, ends) in regions.items():
            encoded = name.encode("utf-8")
            handle.write(struct.pack("<i", len(encoded)))
            handle.write(encoded)
            handle.write(struct.pack("<i", 0))
            handle.write(struct.pack("<i", -1))
            handle.write(struct.pack("<q", len(starts)))
            handle.write(np.asarray(starts, dtype="<i8").tobytes())
            handle.write(np.asarray(ends, dtype="<i8").tobytes())
    return path


def test_read_native_h5_returns_regions_grouped_by_rank(tmp_path):
    path = _write_rank_h5(
        tmp_path / "run_rank00002.h5",
        rank=2,
        regions={"solve": ([10, 30], [20, 45])},
        metadata={"source": "native", "hostname": "node07"},
    )

    ranks, metadata = read_native_h5(path)

    assert list(ranks) == [2]
    assert list(ranks[2]) == ["solve"]
    assert list(ranks[2]["solve"].start_times_ns) == [10, 30]
    assert metadata["hostname"] == "node07"


def test_find_traces_collects_per_rank_hdf5_and_traces(tmp_path):
    _write_rank_h5(tmp_path / "run_rank00000.h5", 0, {"solve": ([1], [2])})
    _write_trace(tmp_path / "run_rank00001.spt", 1, {"solve": ([3], [4])})

    assert [path.name for path in find_traces(tmp_path)] == [
        "run_rank00000.h5",
        "run_rank00001.spt",
    ]


def test_find_traces_ignores_a_merged_profile_in_the_same_directory(tmp_path):
    """The output of one import must not become the input of the next."""
    _write_rank_h5(tmp_path / "run_rank00000.h5", 0, {"solve": ([1], [2])})
    _write_rank_h5(tmp_path / "profiling_data.h5", 0, {"solve": ([1], [2])})

    assert [path.name for path in find_traces(tmp_path)] == ["run_rank00000.h5"]

    # Naming it explicitly still reads it, for the profile whose name does not
    # follow the per-rank convention.
    named = find_traces(tmp_path / "profiling_data.h5")
    assert [path.name for path in named] == ["profiling_data.h5"]


def test_find_traces_reports_a_directory_holding_neither_kind(tmp_path):
    (tmp_path / "notes.txt").write_text("nothing to import")

    with pytest.raises(FileNotFoundError, match="no .spt traces and no per-rank"):
        find_traces(tmp_path)


def test_an_imported_profile_reconstructs_its_call_graph(tmp_path):
    """Native output records no parent links, so the nesting must be derived.

    The regression this pins: an import that wrote all-(-1) ``call_ids``
    instead of omitting the column made ``call_graph`` take the "ids are
    explicit" path, where every call collided on id -1 and the whole rank
    collapsed to a single node -- while ``call_stack``, which always derives,
    stayed correct. The two must agree.
    """
    from scope_profiler.native_trace import convert_traces

    _write_trace(
        tmp_path / "run_rank00000.spt",
        0,
        {"outer": ([0, 100], [50, 150]), "inner": ([10, 110], [20, 120])},
    )
    imported = convert_traces(tmp_path / "run_rank00000.spt", tmp_path / "out.h5")

    with h5py.File(imported, "r") as handle:
        assert "call_ids" not in handle["events"]
        assert "parent_ids" not in handle["events"]

    results = read_h5(imported)
    stack = [(entry["name"], entry["depth"]) for entry in results.call_stack(rank=0)]
    graph = [(entry["name"], entry["depth"]) for entry in results.call_graph(rank=0)]
    assert stack == [("outer", 0), ("inner", 1), ("outer", 0), ("inner", 1)]
    assert graph == stack


def test_load_traces_merges_hdf5_ranks_with_traces(tmp_path):
    _write_rank_h5(
        tmp_path / "run_rank00000.h5",
        0,
        {"solve": ([100, 300], [200, 400]), "setup": ([50], [90])},
    )
    _write_rank_h5(tmp_path / "run_rank00001.h5", 1, {"solve": ([110], [210])})
    _write_trace(tmp_path / "run_rank00002.spt", 2, {"solve": ([120], [220])})

    results = load_traces(tmp_path, label="mixed")

    assert results.num_ranks == 3
    solve = results.get_region("solve")
    assert sorted(solve.regions) == [0, 1, 2]
    assert list(solve.regions[0].start_times_ns) == [100, 300]
    assert list(solve.regions[2].start_times_ns) == [120]
    assert sorted(solve.regions) == [0, 1, 2]
    assert results.metadata["label"] == "mixed"
    assert results.metadata["source"] == "native"
    # The timeline origin is the earliest event of any rank, in either format.
    assert results.metadata["start_time_ns"] == 50


def test_load_traces_keeps_the_lowest_ranks_environment(tmp_path):
    """Rank 0's metadata describes the run, as it does for a Python profile."""
    _write_rank_h5(
        tmp_path / "run_rank00001.h5",
        1,
        {"solve": ([10], [20])},
        metadata={"source": "native", "hostname": "node01", "timestamp": "later"},
    )
    _write_rank_h5(
        tmp_path / "run_rank00000.h5",
        0,
        {"solve": ([10], [20])},
        metadata={"source": "native", "hostname": "node00", "timestamp": "earlier"},
    )

    results = load_traces(tmp_path)

    assert results.metadata["hostname"] == "node00"
    assert results.metadata["timestamp"] == "earlier"


def test_two_files_claiming_the_same_rank_are_rejected(tmp_path):
    """Silently merging them would double one rank's work into another's."""
    _write_rank_h5(tmp_path / "run_rank00000.h5", 0, {"solve": ([1], [2])})
    _write_trace(tmp_path / "other_rank00000.spt", 0, {"solve": ([3], [4])})

    with pytest.raises(TraceFormatError, match="both claim rank 0"):
        load_traces(tmp_path)


def test_a_multi_rank_profile_contributes_all_of_its_ranks(tmp_path):
    """A merged file named explicitly is read whole, not as one rank."""
    path = tmp_path / "merged.h5"
    with ProfilingWriter(path, {"source": "native"}) as writer:
        for rank in (0, 1):
            writer.write_rank(
                rank,
                RankPayload(
                    regions={
                        "solve": (
                            np.asarray([rank * 10], dtype=np.int64),
                            np.asarray([rank * 10 + 5], dtype=np.int64),
                        ),
                    },
                    likwid={},
                    likwid_environment={},
                ),
            )

    ranks, _ = read_native_h5(path)
    assert sorted(ranks) == [0, 1]

    results = load_traces(path)
    assert sorted(results.get_region("solve").regions) == [0, 1]
