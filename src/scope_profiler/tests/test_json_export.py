"""The JSON profile format: what it holds, and that a round trip loses nothing."""

import gzip
import json
import math
from time import sleep

import numpy as np
import pytest

from scope_profiler import ProfileManager, read_h5, read_h5_summary
from scope_profiler.__main__ import main as cli_main
from scope_profiler.json_export import (
    JSONProfileError,
    export_json,
    is_json_path,
    read_json,
    write_json,
)
from scope_profiler.likwid_data import LikwidRegionResult
from scope_profiler.profile_io import profile_format, read_profile, read_profile_summary
from scope_profiler.results import ProfilingResults


@pytest.fixture(autouse=True)
def _reset():
    yield
    ProfileManager._reset()


def _profile(tmp_path, name="profiling_data.h5", **setup):
    """Record a small nested run and return the file it wrote."""
    out = tmp_path / name
    ProfileManager.setup(file_path=str(out), **setup)
    for _ in range(3):
        with ProfileManager.profile_region("outer"):
            sleep(0.0001)
            with ProfileManager.profile_region("inner"):
                sleep(0.0001)
    ProfileManager.finalize(verbose=False)
    return out


def _assert_same_run(expected, actual):
    """Every number, name and column the two result sets can be asked for."""
    assert actual.region_names == expected.region_names
    assert actual.summary() == expected.summary()
    assert actual.metadata == expected.metadata
    assert actual.num_ranks == expected.num_ranks
    for name in expected.region_names:
        for rank, region in expected[name].regions.items():
            other = actual[name].regions[rank]
            assert np.array_equal(other.start_times_ns, region.start_times_ns)
            assert np.array_equal(other.end_times_ns, region.end_times_ns)
            assert np.array_equal(
                other.exclusive_durations_ns,
                region.exclusive_durations_ns,
            )
            assert other.source_file == region.source_file
            assert other.source_lineno == region.source_lineno
            assert other.source_text == region.source_text
            assert other.tags == region.tags


def test_round_trip_keeps_every_event(tmp_path):
    """The point of the format: JSON in, the same run out."""
    expected = read_h5(_profile(tmp_path))

    written = write_json(expected, tmp_path / "profile.json")
    actual = read_json(written)

    _assert_same_run(expected, actual)
    assert actual.events() == expected.events()
    assert actual.call_stack() == expected.call_stack()


def test_document_is_self_describing(tmp_path):
    """A reader that is not this package can still tell what it has."""
    results = read_h5(_profile(tmp_path))

    document = json.loads(write_json(results, tmp_path / "p.json").read_text())

    assert document["format"] == "scope-profiler-profile"
    assert document["format_version"] == 1
    assert document["time_unit"] == "nanoseconds"
    assert document["exporter"].startswith("scope-profiler@")
    outer = next(row for row in document["regions"] if row["name"] == "outer")
    assert outer["rank"] == 0
    assert len(outer["start_times_ns"]) == 3
    assert all(isinstance(value, int) for value in outer["start_times_ns"])


def test_regions_keep_the_run_s_own_order(tmp_path):
    """Not sorted by name or duration: the order the file itself stores."""
    results = read_h5(_profile(tmp_path))

    document = json.loads(write_json(results, tmp_path / "p.json").read_text())

    assert [row["name"] for row in document["regions"]] == results.region_names


def test_gzip_round_trip_and_determinism(tmp_path):
    """``.json.gz`` compresses, reads back, and is byte-stable."""
    expected = read_h5(_profile(tmp_path))

    first = write_json(expected, tmp_path / "a.json.gz")
    payload = first.read_bytes()
    second = write_json(expected, tmp_path / "b.json.gz")

    assert payload[:2] == b"\x1f\x8b"
    assert payload == second.read_bytes()
    assert len(payload) < len(gzip.decompress(payload))
    _assert_same_run(expected, read_json(first))


def test_filters_select_regions_and_ranks(tmp_path):
    results = read_h5(_profile(tmp_path))

    written = write_json(results, tmp_path / "p.json", include=["inner"], ranks=[0])

    document = json.loads(written.read_text())
    assert [row["name"] for row in document["regions"]] == ["inner"]
    assert read_json(written).region_names == ["inner"]


def test_aggregation_mode_keeps_statistics_and_no_timeline(tmp_path):
    """An aggregation run has no events to write, and says so."""
    expected = read_h5(_profile(tmp_path, aggregation_mode=True))

    written = write_json(expected, tmp_path / "p.json")

    document = json.loads(written.read_text())
    row = next(row for row in document["regions"] if row["name"] == "outer")
    assert "start_times_ns" not in row
    assert row["aggregate"]["count"] == 3
    actual = read_json(written)
    assert actual.summary() == expected.summary()
    assert actual["outer"][0].num_calls == 3
    assert actual["outer"][0].start_times_ns.size == 0


def test_summary_only_results_stay_summary_only(tmp_path):
    """The float statistics of a summary read survive, and so does its state."""
    expected = read_h5_summary(_profile(tmp_path))

    actual = read_json(write_json(expected, tmp_path / "p.json"))

    assert actual.summary() == expected.summary()
    assert not actual.has_event_data
    assert not actual["outer"][0].has_event_data


def test_threads_and_tasks_round_trip(tmp_path):
    """Lane tables are what makes a threaded run's calls attributable."""

    async def _amain():
        import asyncio

        async def step():
            with ProfileManager.profile_region("task_step"):
                await asyncio.sleep(0.0001)

        await asyncio.gather(step(), step())

    import asyncio

    out = tmp_path / "threads.h5"
    ProfileManager.setup(file_path=str(out), track_threads=True, track_async=True)
    asyncio.run(_amain())
    ProfileManager.finalize(verbose=False)

    expected = read_h5(out)
    actual = read_json(write_json(expected, tmp_path / "p.json"))

    assert [thread.name for thread in actual.threads[0]] == [
        thread.name for thread in expected.threads[0]
    ]
    assert [task.coro_name for task in actual.tasks[0]] == [
        task.coro_name for task in expected.tasks[0]
    ]
    for name in expected.region_names:
        region, other = expected[name][0], actual[name][0]
        assert np.array_equal(other.thread_ids, region.thread_ids)
        assert np.array_equal(other.task_ids, region.task_ids)
        assert np.array_equal(other.await_times_ns, region.await_times_ns)


def _likwid_results(tmp_path, metrics):
    """A result set carrying one synthetic LIKWID region and a line profile."""
    return ProfilingResults(
        {},
        metadata={"mpi_size": 1},
        num_ranks=1,
        likwid={
            0: {
                "solve": LikwidRegionResult(
                    tag="solve",
                    group_id=1,
                    group_name="CLOCK",
                    cpus=[0, 1],
                    times=np.array([1.0, 2.0]),
                    call_counts=np.array([3, 4]),
                    event_names=["INSTR_RETIRED_ANY"],
                    counter_names=["FIXC0"],
                    events=np.array([[10.0, 20.0]]),
                    metric_names=["Runtime (RDTSC) [s]", "CPI"],
                    metrics=np.asarray(metrics, dtype=float),
                    source="full_api",
                ),
            },
        },
        line_profile={
            0: [
                {
                    "region": "solve",
                    "filename": "solve.py",
                    "function": "solve",
                    "first_lineno": 10,
                    "line_numbers": np.array([10, 11]),
                    "hits": np.array([1, 5]),
                    "times": np.array([1.5, 2.5]),
                    "unit": 1e-06,
                },
            ],
        },
        file_path=tmp_path / "likwid.h5",
    )


def test_likwid_and_line_profile_round_trip(tmp_path):
    expected = _likwid_results(tmp_path, [[1.0, 2.0], [3.0, 4.0]])

    actual = read_json(write_json(expected, tmp_path / "p.json"))

    region = actual.get_likwid_region("solve")
    original = expected.get_likwid_region("solve")
    assert region.group_name == original.group_name
    assert region.cpus == original.cpus
    assert region.counter_names == original.counter_names
    assert np.array_equal(region.times, original.times)
    assert np.array_equal(region.call_counts, original.call_counts)
    assert np.array_equal(region.events, original.events)
    assert np.array_equal(region.metrics, original.metrics)
    record = actual.line_profile[0][0]
    assert record["function"] == "solve"
    assert np.array_equal(record["hits"], np.array([1, 5]))
    assert record["unit"] == 1e-06


def test_non_finite_counters_survive_as_null(tmp_path):
    """LIKWID hands out NaN and inf; strict JSON has neither."""
    expected = _likwid_results(tmp_path, [[math.nan, 2.0], [math.inf, 4.0]])

    written = write_json(expected, tmp_path / "p.json")

    document = json.loads(written.read_text())
    assert document["likwid"]["0"]["solve"]["metrics"][0] == [None, 2.0]
    metrics = read_json(written).get_likwid_region("solve").metrics
    assert math.isnan(metrics[0][0]) and math.isnan(metrics[1][0])
    assert metrics[0][1] == 2.0


def test_export_json_writes_one_file_per_run(tmp_path):
    runs = [read_h5(_profile(tmp_path, name=f"run{index}.h5")) for index in range(2)]

    written = export_json(runs, tmp_path / "out" / "profile.json", verbose=False)

    assert [path.name for path in written] == [
        "profile_run0.json",
        "profile_run1.json",
    ]
    assert read_json(written[0]).region_names == runs[0].region_names


def test_export_json_of_a_non_root_rank_writes_nothing(tmp_path):
    empty = ProfilingResults({}, file_path=tmp_path / "x.h5", is_root=False)

    assert export_json(empty, tmp_path / "profile.json", verbose=False) == []


def test_reading_a_foreign_json_file_explains_itself(tmp_path):
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"traceEvents": []}), encoding="utf-8")

    with pytest.raises(JSONProfileError, match="not a scope-profiler JSON profile"):
        read_json(path)


def test_reading_a_newer_format_version_asks_for_an_upgrade(tmp_path):
    path = tmp_path / "future.json"
    path.write_text(
        json.dumps({"format": "scope-profiler-profile", "format_version": 99}),
        encoding="utf-8",
    )

    with pytest.raises(JSONProfileError, match="Upgrade scope-profiler"):
        read_json(path)


def test_reading_a_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="JSON profile not found"):
        read_json(tmp_path / "nope.json")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("run.h5", "hdf5"),
        ("run", "hdf5"),
        ("run.json", "json"),
        ("run.JSON.GZ", "json"),
        ("report.html", "html"),
        ("report.htm", "html"),
    ],
)
def test_profile_format_follows_the_file_name(name, expected):
    assert profile_format(name) == expected
    assert is_json_path(name) == (expected == "json")


def test_read_profile_dispatches_on_the_name(tmp_path):
    expected = read_h5(_profile(tmp_path))
    written = write_json(expected, tmp_path / "p.json")

    assert read_profile(written).summary() == expected.summary()
    assert read_profile(expected.file_path).summary() == expected.summary()
    assert read_profile_summary(written).region_names == expected.region_names


def _script(tmp_path):
    script = tmp_path / "script.py"
    script.write_text(
        "import time\n\ndef work():\n    time.sleep(0.001)\n\nwork()\n",
        encoding="utf-8",
    )
    return str(script)


def test_run_writes_json_and_leaves_no_hdf5_behind(tmp_path):
    """``-o name.json`` is the whole point of the extension dispatch."""
    output = tmp_path / "profile.json"

    cli_main(["run", "-q", "-o", str(output), _script(tmp_path)])

    assert output.exists()
    assert list(tmp_path.glob("*.h5")) == []
    results = read_json(output)
    assert any(name.endswith("work") for name in results.region_names)


def test_run_writes_gzipped_json(tmp_path):
    output = tmp_path / "profile.json.gz"

    cli_main(["run", "-q", "-o", str(output), _script(tmp_path)])

    assert output.read_bytes()[:2] == b"\x1f\x8b"
    assert list(tmp_path.glob("*.h5")) == []
    assert read_json(output).num_ranks == 1


def test_run_writes_an_html_report(tmp_path):
    output = tmp_path / "profile.html"

    cli_main(["run", "-q", "-o", str(output), _script(tmp_path)])

    assert "<html" in output.read_text(encoding="utf-8")
    assert list(tmp_path.glob("*.h5")) == []


def test_run_still_writes_hdf5_by_default(tmp_path):
    output = tmp_path / "profile.h5"

    cli_main(["run", "-q", "-o", str(output), _script(tmp_path)])

    assert read_h5(output).num_ranks == 1


def test_run_summary_names_the_json_it_wrote(tmp_path, capsys):
    output = tmp_path / "profile.json"

    cli_main(["run", "-o", str(output), _script(tmp_path)])

    out = capsys.readouterr().out
    assert "profile.json" in out
    assert ".scope-profiler.h5" not in out


def test_export_json_subcommand(tmp_path):
    profile = _profile(tmp_path)

    cli_main(["export", "json", str(profile), "-o", str(tmp_path / "out")])

    written = tmp_path / "out" / "profile.json"
    assert read_json(written).region_names == read_h5(profile).region_names


def test_export_json_subcommand_can_gzip_and_indent(tmp_path):
    profile = _profile(tmp_path)

    cli_main(
        [
            "export",
            "json",
            str(profile),
            "-o",
            str(tmp_path / "out"),
            "--gzip",
            "--indent",
            "2",
        ],
    )

    written = tmp_path / "out" / "profile.json.gz"
    assert b"\n" in gzip.decompress(written.read_bytes())
    assert read_json(written).num_ranks == 1


def test_the_other_subcommands_accept_a_json_profile(tmp_path, capsys):
    profile = _profile(tmp_path)
    written = write_json(read_h5(profile), tmp_path / "profile.json")

    cli_main(["inspect", str(written)])
    assert "outer" in capsys.readouterr().out

    cli_main(["diff", str(profile), str(written)])
    assert "outer" in capsys.readouterr().out

    cli_main(["report", str(written), "-o", str(tmp_path / "r.html"), "--no-charts"])
    assert (tmp_path / "r.html").exists()


def test_tags_and_source_travel_with_the_region(tmp_path):
    """What a region is, not only what it measured."""
    out = tmp_path / "tagged.h5"
    ProfileManager.setup(file_path=str(out), capture_region_source=True)
    with ProfileManager.profile_region("solve", tags=["physics", "hot"]):
        sleep(0.0001)
    ProfileManager.finalize(verbose=False)
    expected = read_h5(out)

    written = write_json(expected, tmp_path / "p.json")

    row = next(row for row in json.loads(written.read_text())["regions"])
    assert row["tags"] == ["physics", "hot"]
    assert row["source_text"]
    region = read_json(written)["solve"][0]
    assert region.tags == ("physics", "hot")
    assert region.source_text == expected["solve"][0].source_text
    assert region.source_file == expected["solve"][0].source_file


def test_awkward_metadata_values_stay_valid_json(tmp_path):
    """Metadata is whatever the environment handed the run, numpy included."""
    results = ProfilingResults(
        {},
        metadata={
            "modules": np.array(["a", "b"]),
            "hostname": b"node01",
            "cores": np.int64(8),
            "load": math.inf,
        },
        file_path=tmp_path / "meta.h5",
    )

    document = json.loads(write_json(results, tmp_path / "p.json").read_text())

    assert document["metadata"] == {
        "modules": ["a", "b"],
        "hostname": "node01",
        "cores": 8,
        "load": None,
    }


def test_rank_filter_reaches_every_per_rank_table(tmp_path):
    """Regions, counters, line profiles and lane tables all follow --ranks."""
    out = tmp_path / "threads.h5"
    ProfileManager.setup(file_path=str(out), track_threads=True)
    with ProfileManager.profile_region("solve"):
        sleep(0.0001)
    ProfileManager.finalize(verbose=False)
    per_rank = read_h5(out)
    # A second rank, assembled from the first: two-rank data without an
    # mpirun, which is all the rank filter needs to be exercised.
    for name in per_rank.region_names:
        per_rank[name].regions[1] = per_rank[name].regions[0]
    per_rank._threads[1] = per_rank._threads[0]
    per_rank._likwid = _likwid_results(tmp_path, [[1.0, 2.0]]).get_likwid_regions()
    per_rank._likwid[1] = per_rank._likwid[0]
    per_rank._line_profile[1] = per_rank._line_profile[0] = []

    document = json.loads(
        write_json(per_rank, tmp_path / "p.json", ranks=[0]).read_text(),
    )

    assert {row["rank"] for row in document["regions"]} == {0}
    assert list(document["threads"]) == ["0"]
    assert list(document["likwid"]) == ["0"]


def test_empty_counter_matrices_round_trip(tmp_path):
    """A LIKWID source that reports no metrics at all still writes.

    ``marker_file`` collection produces no derived metrics, and a result
    carrying a flat empty placeholder instead of an empty ``(0, 0)`` matrix
    has to write as an empty matrix rather than raise.
    """
    results = ProfilingResults(
        {},
        likwid={
            0: {
                "solve": LikwidRegionResult(
                    tag="solve",
                    source="marker_file",
                    events=np.zeros(0),
                    metrics=np.zeros(0),
                ),
            },
        },
        file_path=tmp_path / "likwid.h5",
    )

    actual = read_json(write_json(results, tmp_path / "p.json"))

    assert actual.get_likwid_region("solve").metrics.size == 0


def test_verbose_export_and_read_name_what_they_touch(tmp_path, capsys):
    results = read_h5(_profile(tmp_path))

    written = export_json(results, tmp_path / "p.json", verbose=True)
    read_json(written[0], verbose=True)

    out = capsys.readouterr().out
    assert "Wrote" in out
    assert "'outer'" in out


def test_write_profile_dispatches_on_the_name(tmp_path):
    """The writer half of the dispatch, including back to HDF5."""
    from scope_profiler.profile_io import write_profile

    expected = read_h5(_profile(tmp_path))

    as_json = write_profile(expected, tmp_path / "copy.json")
    as_html = write_profile(expected, tmp_path / "copy.html", include_charts=False)
    as_h5 = write_profile(expected, tmp_path / "copy.h5")

    assert read_json(as_json).region_names == expected.region_names
    assert "<html" in as_html.read_text(encoding="utf-8")
    assert read_h5(as_h5).region_names == expected.region_names
