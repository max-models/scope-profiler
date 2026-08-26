"""Tests for the post-processing Python API (ProfilingResults, Region, MPIRegion)."""

import h5py
import numpy as np
import pytest

import scope_profiler
from scope_profiler import MPIRegion, Region, read_h5

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
    assert region.first_duration == 2.0
    assert region.last_duration == 4.0
    assert region.first_start_time == 0.0
    assert region.last_end_time == 14.0
    assert region.get_summary()["total_duration"] == 6.0
    assert region.p50_duration == pytest.approx(3.0)
    assert region.p95_duration == pytest.approx(3.9)
    assert region.p99_duration == pytest.approx(3.98)


def test_region_percentile_rejects_invalid_values():
    region = Region(np.array([0], dtype=np.int64), np.array([NS], dtype=np.int64))
    with pytest.raises(ValueError):
        region.percentile_duration(-1)
    with pytest.raises(ValueError):
        region.percentile_duration(101)


def test_region_without_any_calls_is_safe():
    """A region that recorded nothing still answers every query."""
    empty = np.empty(0, dtype=np.int64)
    region = Region(empty, empty)

    assert region.num_calls == 0
    assert len(region) == 0
    assert not region.has_timing
    # Every duration stat stays defined rather than raising on empty input.
    assert region.get_summary() == {
        "num_calls": 0,
        "total_duration": 0.0,
        "inclusive_duration": 0.0,
        "exclusive_duration": 0.0,
        "average_duration": 0.0,
        "min_duration": 0.0,
        "max_duration": 0.0,
        "first_duration": 0.0,
        "last_duration": 0.0,
        "std_duration": 0.0,
    }


def test_mpi_region_aggregates_over_ranks(sample_file):
    solve = read_h5(sample_file).get_region("solve")

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
    # rank 0 starts first (tied at t=2, rank 0 wins), rank 1 ends last (t=9).
    assert solve.first_duration == pytest.approx(2.0)
    assert solve.last_duration == pytest.approx(4.0)
    assert "solve" in repr(solve)
    assert solve.p50_duration == pytest.approx(3.5)
    assert solve.p95_duration == pytest.approx(4.0)
    assert solve.rank_imbalance == pytest.approx(8.0 / 6.5)
    assert solve.rank_imbalance_pct == pytest.approx((8.0 / 6.5 - 1) * 100)


def test_mpi_region_unknown_rank_lists_available(sample_file):
    solve = read_h5(sample_file).get_region("solve")

    with pytest.raises(KeyError, match=r"Available ranks: \[0, 1\]"):
        solve[7]


def test_reader_mapping_interface(sample_file):
    results = read_h5(sample_file)

    assert results.region_names == ["setup", "solve"]
    assert len(results) == 2
    assert "solve" in results
    assert "missing" not in results
    assert [region.name for region in results] == ["setup", "solve"]
    assert results["solve"] is results.get_region("solve")
    assert results.num_ranks == 2
    assert "sample.h5" in repr(results)


def test_reader_unknown_region_lists_available(sample_file):
    results = read_h5(sample_file)

    with pytest.raises(KeyError, match="Available regions"):
        results.get_region("nope")


def test_reader_summary(sample_file):
    results = read_h5(sample_file)

    summary = results.summary()
    assert [row["name"] for row in summary] == ["setup", "solve"]
    assert summary[1]["num_calls"] == 4
    assert summary[1]["total_duration"] == pytest.approx(13.0)
    assert summary[1]["p95"] == pytest.approx(4.0)
    assert summary[1]["imbalance"] == pytest.approx((8.0 / 6.5 - 1) * 100)

    # Filters use the same regex semantics as get_regions().
    assert [row["name"] for row in results.summary(include="sol")] == ["solve"]
    assert [row["name"] for row in results.summary(exclude="sol")] == ["setup"]


def test_reader_print_summary(sample_file, capsys):
    read_h5(sample_file).print_summary()

    out = capsys.readouterr().out
    header = next(line for line in out.splitlines() if "region" in line)
    assert "region" in header and "total [s]" in header and "avg [s]" in header
    assert "min [s]" not in header and "std [s]" not in header
    assert "setup" in out and "solve" in out
    # sample_file carries no start_time_ns/finalize_time_ns, so total_time is
    # undefined and print_summary must not claim a number for it.
    assert "Total time" not in out


def test_reader_print_summary_accepts_columns(sample_file, capsys):
    read_h5(sample_file).print_summary(
        columns=["region", "ranks", "calls", "total", "avg"]
    )

    out = capsys.readouterr().out
    header = next(line for line in out.splitlines() if "region" in line)

    assert "region" in header
    assert "ranks" in header
    assert "calls" in header
    assert "total [s]" in header
    assert "avg [s]" in header
    assert "min [s]" not in header
    assert "imbalance [%]" not in header
    assert "setup" in out and "solve" in out and "TOTAL" in out


def test_reader_print_summary_shows_total_time_when_available(tmp_path, capsys):
    path = tmp_path / "with_finalize.h5"
    _write_sample_h5(
        path,
        {0: {"solve": ([100 * NS], [130 * NS])}},
        metadata={"start_time_ns": 80 * NS, "finalize_time_ns": 140 * NS},
    )

    read_h5(path).print_summary()

    out = capsys.readouterr().out
    assert "Total time (setup to finalize):" not in out


def test_reader_print_summary_sort_by_min_and_std(sample_file):
    """sort accepts every SORT_KEYS column, not just total/calls/avg/max/name."""
    from scope_profiler.summary import SORT_KEYS, region_rows

    assert {"min", "std"} <= set(SORT_KEYS)

    results = read_h5(sample_file)

    # setup: durations [1, 3] -> min 1, std 1.0
    # solve: durations [2, 3, 4, 4] -> min 2, std ~0.829
    # Descending by min: solve (2) before setup (1).
    rows = region_rows(results, sort="min")
    assert [row["name"] for row in rows] == ["solve", "setup"]

    # Descending by std: setup (1.0) before solve (~0.829).
    rows = region_rows(results, sort="std")
    assert [row["name"] for row in rows] == ["setup", "solve"]


def test_reader_to_dataframe(sample_file):
    pd = pytest.importorskip("pandas")
    results = read_h5(sample_file)

    frame = results.to_dataframe()
    assert isinstance(frame, pd.DataFrame)
    assert list(frame["name"]) == ["setup", "solve"]
    assert frame.loc[frame["name"] == "solve", "num_calls"].item() == 4

    per_rank = results.to_dataframe(per_rank=True)
    assert list(per_rank["rank"]) == [0, 1, 0, 1]
    assert per_rank.loc[
        (per_rank["name"] == "solve") & (per_rank["rank"] == 0), "total_duration"
    ].item() == pytest.approx(5.0)


def test_region_events(sample_file):
    """Region.events() gives one entry per call, optionally rebased."""
    solve = read_h5(sample_file)["solve"][0]

    assert solve.events() == [
        {"call_index": 0, "start": 2.0, "end": 4.0, "duration": 2.0},
        {"call_index": 1, "start": 5.0, "end": 8.0, "duration": 3.0},
    ]
    # The origin shifts timestamps but leaves durations alone.
    rebased = solve.events(origin=2.0)
    assert [event["start"] for event in rebased] == [0.0, 3.0]
    assert [event["duration"] for event in rebased] == [2.0, 3.0]


def test_region_raw_nanosecond_arrays(sample_file):
    solve = read_h5(sample_file)["solve"][0]

    assert solve.start_times_ns.tolist() == [2 * NS, 5 * NS]
    assert solve.end_times_ns.tolist() == [4 * NS, 8 * NS]
    assert solve.durations_ns.tolist() == [2 * NS, 3 * NS]


def test_mpi_region_events_span_ranks(sample_file):
    solve = read_h5(sample_file)["solve"]

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
    results = read_h5(sample_file)

    events = results.events()
    assert len(events) == 6
    assert [event["name"] for event in events[:2]] == ["setup", "setup"]

    # Relative timestamps (the default) start the timeline at zero.
    assert min(event["start"] for event in events) == 0.0
    absolute = results.events(relative=False)
    assert min(event["start"] for event in absolute) == 0.0  # sample starts at 0

    solve_events = results.events(include="solve", ranks=0)
    assert [event["call_index"] for event in solve_events] == [0, 1]
    assert [event["start"] for event in solve_events] == pytest.approx([2.0, 5.0])


def test_reader_events_are_rebased_on_first_entry(tmp_path):
    """A run whose clock starts far from zero still yields a zeroed timeline."""
    path = tmp_path / "offset.h5"
    _write_sample_h5(path, {0: {"solve": ([100 * NS, 130 * NS], [110 * NS, 140 * NS])}})

    results = read_h5(path)

    assert [event["start"] for event in results.events()] == pytest.approx([0.0, 30.0])
    assert [
        event["start"] for event in results.events(relative=False)
    ] == pytest.approx([100.0, 130.0])
    assert results.minimum_start_time == pytest.approx(100.0)
    assert results.maximum_end_time == pytest.approx(140.0)
    assert results.time_span == pytest.approx(40.0)


def test_reader_uses_registered_start_time_as_origin(tmp_path):
    """setup() records a start time; relative timestamps measure from it."""
    path = tmp_path / "with_start.h5"
    _write_sample_h5(
        path,
        {0: {"solve": ([100 * NS, 130 * NS], [110 * NS, 140 * NS])}},
        # The run started 20 s before the first region was entered.
        metadata={"start_time_ns": 80 * NS},
    )

    results = read_h5(path)

    assert results.run_start_time == pytest.approx(80.0)
    assert results.time_origin == pytest.approx(80.0)
    # Events now carry the 20 s of un-instrumented startup.
    assert [event["start"] for event in results.events()] == pytest.approx([20.0, 50.0])
    assert [call["start"] for call in results.call_stack()] == pytest.approx(
        [20.0, 50.0]
    )
    # The startup gap the instrumentation could not see.
    assert results.minimum_start_time - results.run_start_time == pytest.approx(20.0)
    # Durations and the profiled window are unaffected by the origin.
    assert [event["duration"] for event in results.events()] == pytest.approx(
        [10.0, 10.0]
    )
    assert results.time_span == pytest.approx(40.0)


def test_reader_origin_overrides_registered_start_time(tmp_path):
    path = tmp_path / "with_start.h5"
    _write_sample_h5(
        path,
        {0: {"solve": ([100 * NS], [110 * NS])}},
        metadata={"start_time_ns": 80 * NS},
    )
    results = read_h5(path)

    # Explicit origin wins over the registered start time...
    assert results.events(origin=results.minimum_start_time)[0]["start"] == 0.0
    assert results.call_stack(origin=results.minimum_start_time)[0]["start"] == 0.0
    assert results.to_events_dataframe(origin=100.0)["start"].tolist() == [0.0]
    # ...and over relative=False.
    assert results.events(relative=False)[0]["start"] == pytest.approx(100.0)
    assert results.events(relative=False, origin=80.0)[0]["start"] == pytest.approx(
        20.0
    )


@pytest.mark.parametrize(
    "metadata",
    [
        None,  # no metadata group at all (oldest files)
        {"hostname": "somewhere"},  # metadata, but no start time
        {"start_time_ns": "not a number"},  # unreadable value
    ],
    ids=["no_metadata", "metadata_without_start", "unreadable_start"],
)
def test_reader_without_registered_start_time_falls_back(tmp_path, metadata):
    """Post-processing works unchanged on files carrying no start time."""
    path = tmp_path / "no_start.h5"
    _write_sample_h5(
        path,
        {0: {"solve": ([100 * NS, 130 * NS], [110 * NS, 140 * NS])}},
        metadata=metadata,
    )

    results = read_h5(path)

    assert results.run_start_time is None
    # The timeline falls back to the first region entry, as before.
    assert results.time_origin == pytest.approx(results.minimum_start_time)
    assert results.startup_time == 0.0
    assert [event["start"] for event in results.events()] == pytest.approx([0.0, 30.0])
    assert [call["start"] for call in results.call_stack()] == pytest.approx(
        [0.0, 30.0]
    )
    assert results.to_events_dataframe()["start"].tolist() == pytest.approx([0.0, 30.0])
    # An explicit origin still works, and so does the raw timeline.
    assert results.events(origin=130.0)[1]["start"] == pytest.approx(0.0)
    assert results.events(relative=False)[0]["start"] == pytest.approx(100.0)
    assert results.time_span == pytest.approx(40.0)


def test_startup_time_measures_the_gap_before_the_first_region(tmp_path):
    path = tmp_path / "with_start.h5"
    _write_sample_h5(
        path,
        {0: {"solve": ([100 * NS], [110 * NS])}},
        metadata={"start_time_ns": 80 * NS},
    )

    assert read_h5(path).startup_time == pytest.approx(20.0)


def test_total_time_spans_setup_to_finalize(tmp_path):
    """total_time covers startup and teardown that time_span misses."""
    path = tmp_path / "with_finalize.h5"
    _write_sample_h5(
        path,
        # 20 s of startup before the first region, 10 s of teardown after
        # the last one, so total_time (60 s) must exceed time_span (30 s).
        {0: {"solve": ([100 * NS], [130 * NS])}},
        metadata={"start_time_ns": 80 * NS, "finalize_time_ns": 140 * NS},
    )

    results = read_h5(path)

    assert results.finalize_time == pytest.approx(140.0)
    assert results.time_span == pytest.approx(30.0)
    assert results.total_time == pytest.approx(60.0)


def test_total_time_none_without_start_or_finalize_time(tmp_path):
    """Older files (or a run that skipped setup()/finalize()) report None."""
    path = tmp_path / "no_start.h5"
    _write_sample_h5(path, {0: {"solve": ([100 * NS], [110 * NS])}})

    results = read_h5(path)

    assert results.finalize_time is None
    assert results.total_time is None

    # A registered start with no matching finalize time is still incomplete.
    path_start_only = tmp_path / "start_only.h5"
    _write_sample_h5(
        path_start_only,
        {0: {"solve": ([100 * NS], [110 * NS])}},
        metadata={"start_time_ns": 80 * NS},
    )
    assert read_h5(path_start_only).total_time is None


def test_reader_time_span_without_regions(tmp_path):
    """A run that profiled nothing still answers the timeline queries."""
    path = tmp_path / "empty.h5"
    _write_sample_h5(path, {0: {}})

    results = read_h5(path)

    assert results.minimum_start_time == 0.0
    assert results.maximum_end_time == 0.0
    assert results.time_span == 0.0
    assert results.events() == []


def test_reader_to_events_dataframe(sample_file):
    pd = pytest.importorskip("pandas")
    results = read_h5(sample_file)

    frame = results.to_events_dataframe()
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
    path = tmp_path / "empty.h5"
    _write_sample_h5(path, {0: {}})

    frame = read_h5(path).to_events_dataframe()

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
    results = read_h5(path)

    calls = results.call_stack(rank=0)

    assert [(call["name"], call["depth"]) for call in calls] == [
        ("outer", 0),
        ("inner", 1),
        ("leaf", 2),
        ("inner", 1),
    ]
    assert [call["call_id"] for call in calls] == [0, 1, 2, 3]
    assert [call["parent"] for call in calls] == [None, 0, 1, 0]
    # Relative timestamps by default: the outermost call starts at zero.
    assert calls[0]["start"] == 0.0
    assert calls[0]["duration"] == pytest.approx(100.0)
    assert calls[2]["parent"] == 1
    assert calls[0]["parent"] is None

    from scope_profiler.call_stack import call_stack_children, call_stack_roots

    assert call_stack_roots(calls) == [0]
    assert call_stack_children(calls) == [[1, 3], [2], [], []]

    # A filtered stack renests around what is left.
    assert [call["depth"] for call in results.call_stack(exclude="inner")] == [0, 1]


def test_nested_regions_expose_inclusive_and_exclusive_time(tmp_path):
    path = tmp_path / "nested_durations.h5"
    _write_sample_h5(
        path,
        {
            0: {
                "outer": ([0], [100]),
                "first": ([10], [30]),
                "second": ([50], [80]),
            }
        },
    )

    results = read_h5(path)

    assert results["outer"].inclusive_duration == pytest.approx(100e-9)
    assert results["outer"].exclusive_duration == pytest.approx(50e-9)
    assert results["first"].inclusive_duration == pytest.approx(20e-9)
    assert results["first"].exclusive_duration == pytest.approx(20e-9)
    assert results.summary(include="outer")[0]["exclusive_duration"] == pytest.approx(
        50e-9
    )

    outer_call = results.call_stack()[0]
    assert outer_call["inclusive_duration"] == pytest.approx(100e-9)
    assert outer_call["exclusive_duration"] == pytest.approx(50e-9)


def test_exclusive_time_is_reconstructed_lazily_and_once(tmp_path, monkeypatch):
    """Nesting costs more than the rest of the load; only pay for it if asked.

    Reading a file, listing regions and summarizing them (the CLI and MCP
    path, via summary.region_rows) never needs exclusive time, and must not
    reconstruct the call stack. The first caller that does need it pays for
    every region at once, and nobody pays twice.
    """
    from scope_profiler import call_stack as call_stack_module

    path = tmp_path / "lazy_exclusive.h5"
    _write_sample_h5(path, {0: {"outer": ([0], [100]), "inner": ([10], [30])}})

    builds = []
    real_build = call_stack_module.build_call_stack
    monkeypatch.setattr(
        call_stack_module,
        "build_call_stack",
        lambda *args, **kwargs: builds.append(args) or real_build(*args, **kwargs),
    )

    results = read_h5(path)
    assert sorted(results.region_names) == ["inner", "outer"]
    assert results["outer"].total_duration == pytest.approx(100e-9)
    assert results["outer"][0].durations_ns.tolist() == [100]
    assert builds == []

    assert results["inner"].exclusive_duration == pytest.approx(20e-9)
    assert len(builds) == 1

    # Both the region asked first and every other one are now filled in, and
    # asking again does not rebuild.
    assert results["outer"].exclusive_duration == pytest.approx(80e-9)
    assert results["outer"][0].exclusive_durations_ns.tolist() == [80]
    assert len(builds) == 1


def test_top_level_exports_and_lazy_plotting():
    """Post-processing types are importable from the package root."""
    assert scope_profiler.read_h5 is read_h5
    assert scope_profiler.Region is Region
    assert scope_profiler.MPIRegion is MPIRegion

    from scope_profiler.call_stack import build_call_stack

    assert scope_profiler.build_call_stack is build_call_stack

    from scope_profiler.plotting_scripts import plot_duration_timeseries, plot_gantt

    assert scope_profiler.plot_gantt is plot_gantt
    assert scope_profiler.plot_duration_timeseries is plot_duration_timeseries
    assert "plot_flame" in dir(scope_profiler)
    assert "plot_weak_scaling" in dir(scope_profiler)
    assert "plot_rank_heatmap" in dir(scope_profiler)
    assert "plot_scaling_efficiency" in dir(scope_profiler)

    from scope_profiler.speedscope_export import export_speedscope

    assert scope_profiler.export_speedscope is export_speedscope

    # Every advertised name resolves.
    for name in scope_profiler.__all__:
        assert getattr(scope_profiler, name) is not None

    with pytest.raises(AttributeError):
        scope_profiler.not_a_real_attribute


def test_merging_recomputes_exclusive_time_against_the_new_neighbours(tmp_path):
    """A merged set owns its regions' nesting, even if it was already resolved.

    merge_results() reuses the region objects, so a region whose exclusive
    time was already asked for in one set must not carry that answer into a
    set where it now has a call nested inside it.
    """
    from scope_profiler.results import merge_results

    outer_path = tmp_path / "driver.h5"
    inner_path = tmp_path / "kernels.h5"
    _write_sample_h5(outer_path, {0: {"outer": ([0], [100])}})
    _write_sample_h5(inner_path, {0: {"native:inner": ([10], [30])}})

    driver = read_h5(outer_path)
    # Resolved while "outer" is on its own: nothing is nested inside it yet.
    assert driver["outer"].exclusive_duration == pytest.approx(100e-9)

    merged = merge_results(driver, read_h5(inner_path))
    assert merged["outer"].exclusive_duration == pytest.approx(80e-9)
    assert merged["native:inner"].exclusive_duration == pytest.approx(20e-9)


def test_merging_discards_the_exclusive_totals_stored_in_each_file(tmp_path):
    """The stored total is only valid against the regions it was computed with.

    Both files here record their own exclusive time, correctly, for their own
    contents. Merged, "outer" has a call nested inside it that its own run
    never saw, so the stored value must be dropped rather than reported.
    """
    from scope_profiler.call_stack import exclusive_totals_ns
    from scope_profiler.h5writer import ProfilingWriter
    from scope_profiler.profile_manager import RankPayload
    from scope_profiler.results import merge_results

    def write(path, regions):
        arrays = {
            name: tuple(np.asarray(values, dtype=np.int64) for values in pair)
            for name, pair in regions.items()
        }
        with ProfilingWriter(path) as writer:
            writer.write_rank(
                0,
                RankPayload(
                    regions=arrays,
                    likwid={},
                    likwid_environment={},
                    exclusive_totals=exclusive_totals_ns(arrays),
                ),
            )

    driver_path = tmp_path / "driver.h5"
    kernels_path = tmp_path / "kernels.h5"
    write(driver_path, {"outer": ([0], [100])})
    write(kernels_path, {"native:inner": ([10], [30])})

    driver = read_h5(driver_path)
    assert driver["outer"].exclusive_duration == pytest.approx(100e-9)

    merged = merge_results(driver, read_h5(kernels_path))
    assert merged["outer"].exclusive_duration == pytest.approx(80e-9)
    assert merged["native:inner"].exclusive_duration == pytest.approx(20e-9)
