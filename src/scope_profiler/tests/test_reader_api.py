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


def test_region_events(sample_file):
    """Region.events() gives one entry per call, optionally rebased."""
    solve = ProfilingH5Reader(sample_file)["solve"][0]

    assert solve.events() == [
        {"call_index": 0, "start": 2.0, "end": 4.0, "duration": 2.0},
        {"call_index": 1, "start": 5.0, "end": 8.0, "duration": 3.0},
    ]
    # The origin shifts timestamps but leaves durations alone.
    rebased = solve.events(origin=2.0)
    assert [event["start"] for event in rebased] == [0.0, 3.0]
    assert [event["duration"] for event in rebased] == [2.0, 3.0]


def test_region_raw_nanosecond_arrays(sample_file):
    solve = ProfilingH5Reader(sample_file)["solve"][0]

    assert solve.start_times_ns.tolist() == [2 * NS, 5 * NS]
    assert solve.end_times_ns.tolist() == [4 * NS, 8 * NS]
    assert solve.durations_ns.tolist() == [2 * NS, 3 * NS]


def test_mpi_region_events_span_ranks(sample_file):
    solve = ProfilingH5Reader(sample_file)["solve"]

    events = solve.events()
    assert len(events) == 4
    assert [event["rank"] for event in events] == [0, 0, 1, 1]
    assert {event["name"] for event in events} == {"solve"}
    assert [event["duration"] for event in events] == pytest.approx(
        [2.0, 3.0, 4.0, 4.0]
    )

    # A single rank can be selected, as an int or a list.
    assert len(solve.events(ranks=1)) == 2
    assert solve.events(ranks=[1]) == solve.events(ranks=1)
    # Ranks that never recorded the region are skipped, not an error.
    assert solve.events(ranks=[7]) == []


def test_reader_events(sample_file):
    reader = ProfilingH5Reader(sample_file)

    events = reader.events()
    assert len(events) == 6
    assert [event["name"] for event in events[:2]] == ["setup", "setup"]

    # Relative timestamps (the default) start the timeline at zero.
    assert min(event["start"] for event in events) == 0.0
    absolute = reader.events(relative=False)
    assert min(event["start"] for event in absolute) == 0.0  # sample starts at 0

    solve_events = reader.events(include="solve", ranks=0)
    assert [event["call_index"] for event in solve_events] == [0, 1]
    assert [event["start"] for event in solve_events] == pytest.approx([2.0, 5.0])


def test_reader_events_are_rebased_on_first_entry(tmp_path):
    """A run whose clock starts far from zero still yields a zeroed timeline."""
    path = tmp_path / "offset.h5"
    _write_sample_h5(path, {0: {"solve": ([100 * NS, 130 * NS], [110 * NS, 140 * NS])}})

    reader = ProfilingH5Reader(path)

    assert [event["start"] for event in reader.events()] == pytest.approx([0.0, 30.0])
    assert [event["start"] for event in reader.events(relative=False)] == pytest.approx(
        [100.0, 130.0]
    )
    assert reader.minimum_start_time == pytest.approx(100.0)
    assert reader.maximum_end_time == pytest.approx(140.0)
    assert reader.time_span == pytest.approx(40.0)


def test_reader_time_span_ignores_count_only_regions(tmp_path):
    """A count-only region must not drag the timeline origin down to zero."""
    path = tmp_path / "mixed.h5"
    _write_sample_h5(
        path,
        {0: {"counted": None, "solve": ([100 * NS], [110 * NS])}},
    )

    reader = ProfilingH5Reader(path)

    assert reader.minimum_start_time == pytest.approx(100.0)
    assert reader.time_span == pytest.approx(10.0)
    # The count-only region contributes no events.
    assert [event["name"] for event in reader.events()] == ["solve"]


def test_reader_time_span_without_timing(tmp_path):
    path = tmp_path / "counts.h5"
    _write_sample_h5(path, {0: {"counted": None}})

    reader = ProfilingH5Reader(path)

    assert reader.minimum_start_time == 0.0
    assert reader.maximum_end_time == 0.0
    assert reader.time_span == 0.0
    assert reader.events() == []


def test_reader_to_events_dataframe(sample_file):
    pd = pytest.importorskip("pandas")
    reader = ProfilingH5Reader(sample_file)

    frame = reader.to_events_dataframe()
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == [
        "name",
        "rank",
        "call_index",
        "start",
        "end",
        "duration",
    ]
    assert len(frame) == 6
    assert frame.loc[frame["name"] == "solve", "duration"].sum() == pytest.approx(13.0)


def test_reader_to_events_dataframe_columns_when_empty(tmp_path):
    """An empty selection still yields the documented columns."""
    pytest.importorskip("pandas")
    path = tmp_path / "counts.h5"
    _write_sample_h5(path, {0: {"counted": None}})

    frame = ProfilingH5Reader(path).to_events_dataframe()

    assert frame.empty
    assert list(frame.columns) == [
        "name",
        "rank",
        "call_index",
        "start",
        "end",
        "duration",
    ]


def test_reader_call_stack(tmp_path):
    """Nesting is reconstructed from containment, parents before children."""
    path = tmp_path / "nested.h5"
    _write_sample_h5(
        path,
        {
            0: {
                "outer": ([10 * NS], [110 * NS]),
                "inner": ([20 * NS, 60 * NS], [50 * NS, 100 * NS]),
                "leaf": ([25 * NS], [30 * NS]),
            }
        },
    )
    reader = ProfilingH5Reader(path)

    calls = reader.call_stack(rank=0)

    assert [(call["name"], call["depth"]) for call in calls] == [
        ("outer", 0),
        ("inner", 1),
        ("leaf", 2),
        ("inner", 1),
    ]
    # Relative timestamps by default: the outermost call starts at zero.
    assert calls[0]["start"] == 0.0
    assert calls[0]["duration"] == pytest.approx(100.0)
    assert calls[2]["parent"] == 1
    assert calls[0]["parent"] is None

    from scope_profiler.call_stack import call_stack_children, call_stack_roots

    assert call_stack_roots(calls) == [0]
    assert call_stack_children(calls) == [[1, 3], [2], [], []]

    # A filtered stack renests around what is left.
    assert [call["depth"] for call in reader.call_stack(exclude="inner")] == [0, 1]


def test_top_level_exports_and_lazy_plotting():
    """Post-processing types are importable from the package root."""
    assert scope_profiler.ProfilingH5Reader is ProfilingH5Reader
    assert scope_profiler.Region is Region
    assert scope_profiler.MPIRegion is MPIRegion

    from scope_profiler.call_stack import build_call_stack

    assert scope_profiler.build_call_stack is build_call_stack

    from scope_profiler.plotting_scripts import plot_duration_timeseries, plot_gantt

    assert scope_profiler.plot_gantt is plot_gantt
    assert scope_profiler.plot_duration_timeseries is plot_duration_timeseries
    assert "plot_flame" in dir(scope_profiler)

    from scope_profiler.speedscope_export import export_speedscope

    assert scope_profiler.export_speedscope is export_speedscope

    # Every advertised name resolves.
    for name in scope_profiler.__all__:
        assert getattr(scope_profiler, name) is not None

    with pytest.raises(AttributeError):
        scope_profiler.not_a_real_attribute
