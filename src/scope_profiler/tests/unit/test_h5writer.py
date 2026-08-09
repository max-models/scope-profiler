"""The write side of the HDF5 layout, and its contract with the reader."""

import h5py
import numpy as np
import pytest

from scope_profiler import read_h5
from scope_profiler.h5writer import ProfilingWriter, write_metadata, write_rank_payload
from scope_profiler.profile_manager import RankPayload

NS = 1_000_000_000


def payload(regions=None, likwid=None, environment=None) -> RankPayload:
    """A RankPayload with plain int64 timestamp arrays."""
    return RankPayload(
        regions={
            name: (
                np.asarray(starts, dtype=np.int64),
                np.asarray(ends, dtype=np.int64),
            )
            for name, (starts, ends) in (regions or {}).items()
        },
        likwid=likwid or {},
        likwid_environment=environment or {},
    )


def test_rank_group_holds_the_recorded_timestamps(tmp_path):
    path = tmp_path / "one_rank.h5"
    with ProfilingWriter(path, {"hostname": "node0"}) as writer:
        assert writer.write_rank(0, payload({"solve": ([0, 3 * NS], [2 * NS, 5 * NS])}))

    with h5py.File(path, "r") as handle:
        starts = handle["rank0/regions/solve/start_times"]
        assert starts[()].tolist() == [0, 3 * NS]
        assert handle["rank0/regions/solve/end_times"][()].tolist() == [2 * NS, 5 * NS]
        # The reader recovers metadata from the top level, never from a rank.
        assert "metadata" not in handle["rank0"]
        assert handle["metadata"].attrs["hostname"] == "node0"


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
