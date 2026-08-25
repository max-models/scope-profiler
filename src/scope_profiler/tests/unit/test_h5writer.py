"""The write side of the HDF5 layout, and its contract with the reader."""

import h5py
import numpy as np
import pytest

from scope_profiler import read_h5
from scope_profiler.h5schema import (
    CURRENT_SCHEMA_VERSION,
    SCHEMA_ATTRIBUTE,
    HDF5SchemaError,
)
from scope_profiler.h5writer import ProfilingWriter, write_metadata, write_rank_payload
from scope_profiler.profile_manager import RankPayload

NS = 1_000_000_000


def payload(
    regions=None, likwid=None, environment=None, line_profile=None
) -> RankPayload:
    """A RankPayload with plain int64 timestamp arrays."""
    return RankPayload(
        regions={
            name: tuple(np.asarray(values, dtype=np.int64) for values in arrays)
            for name, arrays in (regions or {}).items()
        },
        likwid=likwid or {},
        likwid_environment=environment or {},
        line_profile=line_profile,
    )


def test_rank_group_holds_the_recorded_timestamps(tmp_path):
    path = tmp_path / "one_rank.h5"
    with ProfilingWriter(path, {"hostname": "node0"}) as writer:
        assert writer.write_rank(0, payload({"solve": ([0, 3 * NS], [2 * NS, 5 * NS])}))

    with h5py.File(path, "r") as handle:
        assert handle.attrs[SCHEMA_ATTRIBUTE] == CURRENT_SCHEMA_VERSION
        starts = handle["rank0/regions/solve/start_times"]
        assert starts[()].tolist() == [0, 3 * NS]
        assert handle["rank0/regions/solve/end_times"][()].tolist() == [2 * NS, 5 * NS]
        # The reader recovers metadata from the top level, never from a rank.
        assert "metadata" not in handle["rank0"]
        assert handle["metadata"].attrs["hostname"] == "node0"


def test_legacy_file_without_schema_version_still_reads(tmp_path):
    """Files produced before schema versioning remain compatible."""
    path = tmp_path / "legacy.h5"
    with h5py.File(path, "w") as handle:
        handle.create_group("metadata")
        regions = handle.create_group("rank0/regions/solve")
        regions.create_dataset("start_times", data=np.asarray([0], dtype=np.int64))
        regions.create_dataset("end_times", data=np.asarray([NS], dtype=np.int64))

    assert read_h5(path)["solve"].num_calls == 1


@pytest.mark.parametrize("version", [0, 2, "one"])
def test_reader_rejects_invalid_or_unsupported_schema_version(tmp_path, version):
    path = tmp_path / "unsupported.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs[SCHEMA_ATTRIBUTE] = version

    with pytest.raises(HDF5SchemaError, match="schema version"):
        read_h5(path)


def test_gpu_durations_round_trip_when_present(tmp_path):
    path = tmp_path / "gpu_timing.h5"
    with ProfilingWriter(path) as writer:
        writer.write_rank(0, payload({"solve": ([0, NS], [NS, 3 * NS], [7, 11])}))

    with h5py.File(path, "r") as handle:
        gpu = handle["rank0/regions/solve/gpu_durations"]
        assert gpu[()].tolist() == [7, 11]

    region = read_h5(path)["solve"]
    assert region.has_gpu_timing is True
    assert region.gpu_durations.tolist() == pytest.approx([7e-9, 11e-9])
    assert region[0].gpu_durations_ns.tolist() == [7, 11]


def test_line_profile_round_trips(tmp_path):
    path = tmp_path / "line_profile.h5"
    record = {
        "region": "solve",
        "filename": "app.py",
        "function": "solve",
        "first_lineno": 10,
        "line_numbers": np.asarray([11, 12], dtype=np.int64),
        "hits": np.asarray([1, 5], dtype=np.int64),
        "times": np.asarray([10.0, 25.0]),
        "unit": 1e-9,
    }
    with ProfilingWriter(path) as writer:
        writer.write_rank(0, payload({"solve": ([0], [NS])}, line_profile=[record]))

    with h5py.File(path, "r") as handle:
        group = handle["rank0/line_profile/0"]
        assert group.attrs["function"] == "solve"
        assert group["line_numbers"][()].tolist() == [11, 12]

    loaded = read_h5(path).line_profile[0][0]
    assert loaded["filename"] == "app.py"
    assert loaded["hits"].tolist() == [1, 5]
    assert loaded["unit"] == pytest.approx(1e-9)


def test_datasets_are_contiguous_and_exactly_sized(tmp_path):
    """A sparsely-called region must not cost a whole HDF5 chunk."""
    path = tmp_path / "sparse.h5"
    with ProfilingWriter(path) as writer:
        writer.write_rank(0, payload({"sparse": (range(5), range(1, 6))}))

    with h5py.File(path, "r") as handle:
        dataset = handle["rank0/regions/sparse/start_times"]
        assert dataset.shape == (5,)
        assert dataset.chunks is None
        assert dataset.id.get_storage_size() == 5 * 8


def test_a_rank_with_nothing_recorded_gets_no_group(tmp_path):
    """Rank groups are exactly the ranks that have something to report."""
    path = tmp_path / "one_silent.h5"
    with ProfilingWriter(path) as writer:
        assert writer.write_rank(0, payload({"solve": ([0], [NS])})) is True
        assert writer.write_rank(1, payload()) is False

    with h5py.File(path, "r") as handle:
        assert sorted(handle) == ["metadata", "rank0"]


def test_successful_writer_atomically_replaces_previous_file(tmp_path):
    path = tmp_path / "atomic.h5"
    with ProfilingWriter(path, {"generation": "old"}):
        pass

    with ProfilingWriter(path, {"generation": "new"}) as writer:
        # The old file remains readable until the context commits.
        with h5py.File(path, "r") as handle:
            assert handle["metadata"].attrs["generation"] == "old"
        writer.write_rank(0, payload({"solve": ([0], [NS])}))

    with h5py.File(path, "r") as handle:
        assert handle["metadata"].attrs["generation"] == "new"
        assert "rank0/regions/solve" in handle
    assert list(tmp_path.glob(".atomic.h5.*.tmp")) == []


def test_failed_writer_preserves_previous_file_and_discards_temporary(tmp_path):
    path = tmp_path / "atomic_failure.h5"
    with ProfilingWriter(path, {"generation": "old"}):
        pass

    with pytest.raises(RuntimeError, match="interrupted"):
        with ProfilingWriter(path, {"generation": "incomplete"}) as writer:
            writer.write_rank(0, payload({"solve": ([0], [NS])}))
            raise RuntimeError("interrupted")

    with h5py.File(path, "r") as handle:
        assert handle["metadata"].attrs["generation"] == "old"
        assert "rank0" not in handle
    assert list(tmp_path.glob(".atomic_failure.h5.*.tmp")) == []


def test_metadata_round_trips_through_the_reader(tmp_path):
    """The writer/reader contract, including list-valued attributes."""
    path = tmp_path / "meta.h5"
    metadata = {
        "hostname": "node0",
        "mpi_size": 2,
        "modules": ["gcc/12.3.0", "likwid/5.3"],
        "empty": [],
    }
    with ProfilingWriter(path, metadata) as writer:
        writer.write_rank(0, payload({"solve": ([0], [2 * NS])}))
        writer.write_rank(1, payload({"solve": ([0], [4 * NS])}))

    results = read_h5(path)
    assert results.metadata["hostname"] == "node0"
    assert results.metadata["mpi_size"] == 2
    assert results.metadata["modules"] == ["gcc/12.3.0", "likwid/5.3"]
    assert results.metadata["empty"] == []
    assert results.num_ranks == 2
    assert results["solve"].num_calls == 2
    assert results["solve"][1].total_duration == pytest.approx(4.0)


def test_write_metadata_creates_attrs_only(tmp_path):
    path = tmp_path / "bare.h5"
    with h5py.File(path, "w") as handle:
        write_metadata(handle, {"label": "run a"})
        assert list(handle["metadata"].keys()) == []
        assert handle["metadata"].attrs["label"] == "run a"


def test_ranks_are_read_back_in_rank_order(tmp_path):
    """Pooled statistics sum per-rank arrays in order, so it must be stable.

    HDF5 iterates group names alphabetically, which puts "rank10" before
    "rank2"; the reader has to sort numerically or an 11+ rank file pools its
    durations in a different order than the run computed them in memory.
    """
    path = tmp_path / "many_ranks.h5"
    with ProfilingWriter(path) as writer:
        for rank in range(12):
            writer.write_rank(rank, payload({"solve": ([0], [(rank + 1) * NS])}))

    region = read_h5(path)["solve"]
    assert list(region.regions) == list(range(12))
    assert [r.total_duration for r in region.regions.values()] == [
        float(rank + 1) for rank in range(12)
    ]


def test_write_rank_payload_refuses_a_duplicate_rank(tmp_path):
    """Writing a rank twice is a bug in the receive loop, not something to merge."""
    path = tmp_path / "dup.h5"
    with h5py.File(path, "w") as handle:
        write_rank_payload(handle, 0, payload({"solve": ([0], [NS])}))
        with pytest.raises(ValueError):
            write_rank_payload(handle, 0, payload({"solve": ([0], [NS])}))


def test_regions_not_duplicated_across_ranks(tmp_path):
    """When multiple ranks have the same regions, they should not be duplicated.

    This is a regression test for an issue where region_names was built by
    appending every region from every rank, creating duplicates that could
    affect the final regions dict.
    """
    path = tmp_path / "multi_rank.h5"
    with ProfilingWriter(path) as writer:
        # Write 3 ranks with the same regions
        for rank in range(3):
            writer.write_rank(
                rank,
                payload(
                    {
                        "main": ([0], [10 * NS]),
                        "setup": ([0], [2 * NS]),
                        "solve": ([2 * NS], [9 * NS]),
                    }
                ),
            )

    results = read_h5(path)
    # Should have exactly 3 regions, not 9 (3 ranks * 3 regions)
    assert len(results.region_names) == 3
    assert set(results.region_names) == {"main", "setup", "solve"}

    # Each region should have data from all 3 ranks
    for region_name in ["main", "setup", "solve"]:
        region = results[region_name]
        assert len(region.regions) == 3
        assert set(region.regions.keys()) == {0, 1, 2}


def test_prof_export_no_split_timeline(tmp_path):
    """Verify prof export doesn't create split timelines or duplicate calls.

    This is a regression test for issue #167 where prof exports could show
    "two timelines" due to duplicate region entries or incorrect call stack
    reconstruction.
    """
    from scope_profiler.call_stack import build_call_stack
    from scope_profiler.prof_export import build_pstats_dict, export_prof

    path = tmp_path / "multi_rank.h5"
    with ProfilingWriter(path) as writer:
        # Write multiple ranks with multiple nested regions
        for rank in range(2):
            writer.write_rank(
                rank,
                payload(
                    {
                        "main": ([0], [100 * NS]),
                        "setup": ([0], [20 * NS]),
                        "solve": ([20 * NS], [90 * NS]),
                        "assemble": ([30 * NS], [60 * NS]),
                    }
                ),
            )

    results = read_h5(path)
    regions = results.get_regions()

    # Build call stack for rank 0
    calls = build_call_stack(regions, rank=0)

    # Verify no duplicate calls - each region should appear only once per call
    region_names_in_calls = [call["name"] for call in calls]
    call_counts = {}
    for name in region_names_in_calls:
        call_counts[name] = call_counts.get(name, 0) + 1

    # Each region should have exactly 1 call (no duplicates)
    for region_name, count in call_counts.items():
        assert count == 1, f"Region {region_name} appears {count} times in call stack"

    # Build pstats dict and verify structure
    stats = build_pstats_dict(calls)

    # Verify no duplicate entries in pstats
    stat_keys = list(stats.keys())
    assert len(stat_keys) == len(
        set(key[2] for key in stat_keys)
    ), "Duplicate regions in pstats"

    # Verify each region appears exactly once
    region_entries = [key[2] for key in stat_keys if key[2] != "main"]
    assert region_entries.count("setup") <= 1
    assert region_entries.count("solve") <= 1
    assert region_entries.count("assemble") <= 1
