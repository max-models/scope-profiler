"""Tests for the post-processing Python API (reader, Region, MPIRegion)."""

import h5py
import numpy as np
import pytest

import scope_profiler
from scope_profiler import MPIRegion, ProfilingH5Reader, Region

NS = 1_000_000_000


def _write_sample_h5(path, rank_regions, metadata=None):
    with h5py.File(path, "w") as h5file:
        if metadata:
            meta_grp = h5file.create_group("metadata")
            for key, value in metadata.items():
                meta_grp.attrs[key] = value
        for rank, regions in rank_regions.items():
            regions_group = h5file.create_group(f"rank{rank}").create_group("regions")
            for region_name, payload in regions.items():
                region_group = regions_group.create_group(region_name)
                if payload is None:
                    # Count-only region (time_trace=False).
                    region_group.attrs["num_calls"] = 3
                    continue
                start_times, end_times = payload
                region_group.create_dataset(
                    "start_times", data=np.asarray(start_times, dtype=np.int64)
                )
                region_group.create_dataset(
                    "end_times", data=np.asarray(end_times, dtype=np.int64)
                )


@pytest.fixture
def sample_file(tmp_path):
    """Two ranks; 'setup' runs once, 'solve' twice, with known durations."""
    path = tmp_path / "sample.h5"
    _write_sample_h5(
        path,
        {
            0: {
                "setup": ([0], [1 * NS]),
                "solve": ([2 * NS, 5 * NS], [4 * NS, 8 * NS]),
            },
            1: {
                "setup": ([0], [3 * NS]),
                "solve": ([2 * NS, 5 * NS], [6 * NS, 9 * NS]),
            },
        },
    )
    return path


def test_region_durations_are_seconds():
    """Duration stats are seconds, matching the docstrings and JSON export."""
    region = Region(
        np.array([0, 10 * NS], dtype=np.int64),
        np.array([2 * NS, 14 * NS], dtype=np.int64),
    )

    assert region.durations.tolist() == [2.0, 4.0]
    assert region.total_duration == 6.0
    assert region.average_duration == 3.0
    assert region.min_duration == 2.0
    assert region.max_duration == 4.0
    assert region.std_duration == 1.0
    assert region.first_start_time == 0.0
    assert region.last_end_time == 14.0
    assert region.get_summary()["total_duration"] == 6.0


def test_region_without_timing_is_safe():
    empty = np.empty(0, dtype=np.int64)
    region = Region(empty, empty, num_calls=7)

    assert region.num_calls == 7
    assert len(region) == 7
    assert not region.has_timing
    # Every duration stat stays defined rather than raising on empty input.
    assert region.get_summary() == {
        "num_calls": 7,
        "total_duration": 0.0,
        "average_duration": 0.0,
        "min_duration": 0.0,
        "max_duration": 0.0,
        "std_duration": 0.0,
    }


def test_mpi_region_aggregates_over_ranks(sample_file):
    solve = ProfilingH5Reader(sample_file).get_region("solve")

    assert isinstance(solve, MPIRegion)
    assert solve.ranks == [0, 1]
    assert len(solve) == 2
    assert list(solve) == [0, 1]
    assert 1 in solve and 2 not in solve
    # rank 0: 2 s + 3 s, rank 1: 4 s + 4 s
    assert solve.num_calls == 4
    assert solve.num_calls_per_rank() == {0: 2, 1: 2}
    assert solve.total_duration == pytest.approx(13.0)
    assert solve.average_duration == pytest.approx(13.0 / 4)
    assert solve.min_duration == pytest.approx(2.0)
    assert solve.max_duration == pytest.approx(4.0)
    assert sorted(solve.durations.tolist()) == pytest.approx([2.0, 3.0, 4.0, 4.0])
    assert solve.total_durations() == pytest.approx({0: 5.0, 1: 8.0})
    assert solve.first_start_time == pytest.approx(2.0)
    assert solve.last_end_time == pytest.approx(9.0)
    assert "solve" in repr(solve)


def test_mpi_region_unknown_rank_lists_available(sample_file):
    solve = ProfilingH5Reader(sample_file).get_region("solve")

    with pytest.raises(KeyError, match=r"Available ranks: \[0, 1\]"):
        solve[7]


def test_reader_mapping_interface(sample_file):
    reader = ProfilingH5Reader(sample_file)

    assert reader.region_names == ["setup", "solve"]
    assert len(reader) == 2
    assert "solve" in reader
    assert "missing" not in reader
    assert [region.name for region in reader] == ["setup", "solve"]
    assert reader["solve"] is reader.get_region("solve")
    assert reader.num_ranks == 2
    assert "sample.h5" in repr(reader)


def test_reader_unknown_region_lists_available(sample_file):
    reader = ProfilingH5Reader(sample_file)

    with pytest.raises(KeyError, match="Available regions"):
        reader.get_region("nope")


def test_reader_summary(sample_file):
    reader = ProfilingH5Reader(sample_file)

    summary = reader.summary()
    assert [row["name"] for row in summary] == ["setup", "solve"]
    assert summary[1]["num_calls"] == 4
    assert summary[1]["total_duration"] == pytest.approx(13.0)

    # Filters use the same regex semantics as get_regions().
    assert [row["name"] for row in reader.summary(include="sol")] == ["solve"]
    assert [row["name"] for row in reader.summary(exclude="sol")] == ["setup"]


def test_reader_print_summary(sample_file, capsys):
    ProfilingH5Reader(sample_file).print_summary()

    out = capsys.readouterr().out
    assert "region" in out and "total [s]" in out
    assert "setup" in out and "solve" in out


def test_reader_to_dataframe(sample_file):
    pd = pytest.importorskip("pandas")
    reader = ProfilingH5Reader(sample_file)

    frame = reader.to_dataframe()
    assert isinstance(frame, pd.DataFrame)
    assert list(frame["name"]) == ["setup", "solve"]
    assert frame.loc[frame["name"] == "solve", "num_calls"].item() == 4

    per_rank = reader.to_dataframe(per_rank=True)
    assert list(per_rank["rank"]) == [0, 1, 0, 1]
    assert per_rank.loc[
        (per_rank["name"] == "solve") & (per_rank["rank"] == 0), "total_duration"
    ].item() == pytest.approx(5.0)


def test_reader_handles_count_only_regions(tmp_path):
    path = tmp_path / "counts.h5"
    _write_sample_h5(path, {0: {"counted": None}})

    reader = ProfilingH5Reader(path)
    region = reader["counted"]

    assert region.num_calls == 3
    assert not region.has_timing
    assert region.durations.size == 0
    assert reader.summary()[0]["total_duration"] == 0.0
    assert reader.minimum_start_time == 0.0


def test_top_level_exports_and_lazy_plotting():
    """Post-processing types are importable from the package root."""
    assert scope_profiler.ProfilingH5Reader is ProfilingH5Reader
    assert scope_profiler.Region is Region
    assert scope_profiler.MPIRegion is MPIRegion

    from scope_profiler.plotting_scripts import plot_gantt

    assert scope_profiler.plot_gantt is plot_gantt
    assert "plot_flame" in dir(scope_profiler)

    with pytest.raises(AttributeError):
        scope_profiler.not_a_real_attribute
