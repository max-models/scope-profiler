"""Tests for LIKWID marker result handling that do not need LIKWID itself.

The end-to-end check against a real LIKWID run lives in
``examples_pylikwid.py``, which has to be launched by ``likwid-perfctr -m``.
Everything here exercises the storage/read-back layer with synthetic results,
so it runs anywhere.
"""

import builtins
import json
import os
import subprocess
from unittest import mock

import h5py
import numpy as np
import pytest

from scope_profiler import ProfileManager, read_h5
from scope_profiler import likwid_data as likwid_data_module
from scope_profiler import profile_config as profile_config_module
from scope_profiler import region_profiler as region_profiler_module
from scope_profiler.likwid_data import (
    LIKWID_GROUP,
    LikwidRegionResult,
    _result_from_json,
    _result_to_json,
    collect_marker_results,
    collect_marker_results_isolated,
    collect_region_snapshots,
    markers_available,
    parse_marker_file,
    snapshots_to_results,
    write_likwid_results,
)
from scope_profiler.profile_config import (
    _import_pylikwid,
    _liblikwid_search_dirs,
    _pylikwid_import_error,
)


def _make_result(tag="solve", nthreads=2):
    """Build a LikwidRegionResult with distinguishable values."""
    return LikwidRegionResult(
        tag=tag,
        group_id=0,
        group_name="CLOCK",
        cpus=list(range(nthreads)),
        times=np.linspace(0.1, 0.2, nthreads),
        call_counts=np.arange(1, nthreads + 1, dtype=np.int64),
        event_names=["INSTR_RETIRED_ANY", "CPU_CLK_UNHALTED_CORE"],
        events=np.array([[1.0, 2.0], [3.0, 4.0]])[:, :nthreads],
        metric_names=["Clock [MHz]", "CPI"],
        metrics=np.array([[2400.0, 2401.0], [0.5, 0.6]])[:, :nthreads],
        source="full_api",
    )


def _write_merged_file(path, results, rank=0):
    """Write results into a file shaped like a merged (multi-rank) output."""
    with h5py.File(path, "w") as f:
        write_likwid_results(
            f.create_group(f"rank{rank}"),
            results,
            environment={"LIKWID_EVENTS": "CLOCK"},
        )


def test_write_and_read_back_round_trip(tmp_path):
    """Results survive a write to HDF5 and a read back through read_h5."""
    path = tmp_path / "profiling_data.h5"
    original = _make_result()
    _write_merged_file(path, [original])

    results = read_h5(path)
    assert results.has_likwid
    assert results.likwid_ranks == [0]

    result = results.get_likwid_region("solve")
    assert result.tag == "solve"
    assert result.group_id == 0
    assert result.group_name == "CLOCK"
    assert result.source == "full_api"
    assert result.cpus == [0, 1]
    assert result.event_names == original.event_names
    assert result.metric_names == original.metric_names
    np.testing.assert_allclose(result.times, original.times)
    np.testing.assert_array_equal(result.call_counts, original.call_counts)
    np.testing.assert_allclose(result.events, original.events)
    np.testing.assert_allclose(result.metrics, original.metrics)


def test_environment_is_recorded(tmp_path):
    """The LIKWID event set a file was measured with is stored alongside it."""
    path = tmp_path / "profiling_data.h5"
    _write_merged_file(path, [_make_result()])

    with h5py.File(path, "r") as f:
        grp = f["rank0"][LIKWID_GROUP]
        assert grp.attrs["LIKWID_EVENTS"] == "CLOCK"
        assert grp.attrs["num_regions"] == 1


def test_reader_without_likwid_data(tmp_path):
    """A file with no LIKWID group reads back as simply having none."""
    path = tmp_path / "profiling_data.h5"
    with h5py.File(path, "w") as f:
        grp = f.create_group("rank0/regions/solve")
        grp.create_dataset("start_times", data=np.array([0], dtype=np.int64))
        grp.create_dataset("end_times", data=np.array([10], dtype=np.int64))

    results = read_h5(path)
    assert not results.has_likwid
    assert results.likwid_ranks == []
    assert results.get_likwid_regions() == {}
    with pytest.raises(KeyError):
        results.get_likwid_region("solve")


def test_multiple_ranks(tmp_path):
    """Each rank keeps its own copy of the counters."""
    path = tmp_path / "profiling_data.h5"
    with h5py.File(path, "w") as f:
        for rank in (0, 1):
            result = _make_result()
            result.times = result.times + rank
            write_likwid_results(f.create_group(f"rank{rank}"), [result])

    results = read_h5(path)
    assert results.likwid_ranks == [0, 1]
    assert results.get_likwid_region("solve", rank=1).times[0] == pytest.approx(1.1)


def test_tag_with_slash_is_escaped(tmp_path):
    """A ``/`` in a region name must not turn into nested HDF5 groups."""
    path = tmp_path / "profiling_data.h5"
    _write_merged_file(path, [_make_result(tag="solve/inner")])

    results = read_h5(path)
    # The true tag survives, even though the group name had to be escaped.
    assert results.get_likwid_region("solve/inner").tag == "solve/inner"


def test_rewriting_replaces_previous_results(tmp_path):
    """Writing twice into the same file replaces rather than raises."""
    path = tmp_path / "profiling_data.h5"
    with h5py.File(path, "w") as f:
        grp = f.create_group("rank0")
        write_likwid_results(grp, [_make_result(tag="first")])
        write_likwid_results(grp, [_make_result(tag="second")])

    results = read_h5(path)
    assert sorted(results.get_likwid_regions(0)) == ["second"]


def test_likwid_to_dataframe(tmp_path):
    """The tidy view has one row per (rank, region, thread)."""
    pd = pytest.importorskip("pandas")

    path = tmp_path / "profiling_data.h5"
    _write_merged_file(path, [_make_result(nthreads=2)])

    df = read_h5(path).likwid_to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df["region"]) == ["solve", "solve"]
    assert list(df["cpu"]) == [0, 1]
    # Events and metrics become columns named after the counter.
    assert df["INSTR_RETIRED_ANY"].tolist() == [1.0, 2.0]
    assert df["CPI"].tolist() == [0.5, 0.6]


def test_print_likwid_summary(tmp_path, capsys):
    """The summary names the region, its group and its counters."""
    path = tmp_path / "profiling_data.h5"
    _write_merged_file(path, [_make_result()])

    read_h5(path).print_likwid_summary()
    out = capsys.readouterr().out
    assert "solve" in out
    assert "CLOCK" in out
    assert "INSTR_RETIRED_ANY" in out
    assert "CPI" in out


def test_snapshots_to_results():
    """Marker-API snapshots convert into single-thread results."""
    snapshots = [
        {
            "tag": "solve",
            "nevents": 2,
            "events": np.array([10.0, 20.0]),
            "time": 1.5,
            "count": 3,
        }
    ]
    (result,) = snapshots_to_results(snapshots)
    assert result.tag == "solve"
    assert result.source == "marker_api"
    assert result.events.shape == (2, 1)
    assert result.times.tolist() == [1.5]
    assert result.call_counts.tolist() == [3]
    # No perfmon group behind a snapshot, hence placeholder names and no metrics.
    assert result.event_names == ["event_0", "event_1"]
    assert result.metric_names == []


def test_collect_region_snapshots_skips_unknown_regions():
    """Regions LIKWID does not know about are skipped, not fatal."""

    def markergetregion(tag):
        if tag == "known":
            return (1, [5.0], 0.25, 2)
        raise RuntimeError("no such region")

    pylikwid = mock.Mock(markergetregion=markergetregion)
    snapshots = collect_region_snapshots(pylikwid, ["known", "unknown"])
    assert [s["tag"] for s in snapshots] == ["known"]
    assert snapshots[0]["count"] == 2


def test_collect_region_snapshots_skips_never_entered_regions():
    """A region with a zero call count carries no measurement."""
    pylikwid = mock.Mock(markergetregion=lambda tag: (1, [0.0], 0.0, 0))
    assert collect_region_snapshots(pylikwid, ["idle"]) == []


def test_markers_available_follows_the_environment():
    """Marker data only exists when LIKWID launched the process."""
    with mock.patch.dict(os.environ, {}, clear=True):
        assert not markers_available()
    with mock.patch.dict(
        os.environ, {"LIKWID_FILEPATH": "/tmp/x", "LIKWID_EVENTS": "CLOCK"}
    ):
        assert markers_available()


#: A real two-region LIKWID marker file, as markerclose() writes it.
MARKER_FILE = """\
1 2 1
0:main-0
1:io-0
0 0 10 1 1.700627e-03 6 1.898247e+07 4.093639e+06 4.093728e+06 2.046807e+07 \
7.774658e-01 0.000000e+00
1 0 10 4 4.628672e-05 6 1.891340e+05 8.803600e+04 8.812800e+04 4.402750e+05 \
0.000000e+00 0.000000e+00
"""


def test_parse_marker_file_reads_every_field(tmp_path):
    """The marker file alone carries regions, counts, runtimes and counters."""
    path = tmp_path / "likwid.txt"
    path.write_text(MARKER_FILE)

    results = {r.tag: r for r in parse_marker_file(str(path))}
    assert sorted(results) == ["io", "main"]

    main = results["main"]
    assert main.source == "marker_file"
    assert main.group_id == 0
    assert main.cpus == [10]
    assert main.call_counts.tolist() == [1]
    assert main.times[0] == pytest.approx(1.700627e-03)
    assert main.events.shape == (6, 1)
    assert main.events[0, 0] == pytest.approx(1.898247e07)
    # No group definition is in the file, so names are positional and there
    # are no derived metrics.
    assert main.event_names == [f"event_{i}" for i in range(6)]
    assert main.metric_names == []
    assert results["io"].call_counts.tolist() == [4]


def test_parse_marker_file_handles_a_tag_containing_a_dash(tmp_path):
    """The group id is appended with '-', so tags must split from the right."""
    path = tmp_path / "likwid.txt"
    path.write_text("1 1 1\n0:my-region-0\n0 0 3 2 1.0e-03 1 5.0\n")

    (result,) = parse_marker_file(str(path))
    assert result.tag == "my-region"
    assert result.group_id == 0


def test_parse_marker_file_is_multithread_aware(tmp_path):
    """Each thread of a region becomes a column, in file order."""
    path = tmp_path / "likwid.txt"
    path.write_text(
        "2 1 1\n0:solve-0\n0 0 4 1 1.0e-03 2 1.0 2.0\n0 0 5 1 2.0e-03 2 3.0 4.0\n"
    )

    (result,) = parse_marker_file(str(path))
    assert result.cpus == [4, 5]
    assert result.events.tolist() == [[1.0, 3.0], [2.0, 4.0]]
    assert result.times.tolist() == [pytest.approx(1e-3), pytest.approx(2e-3)]


@pytest.mark.parametrize(
    "content",
    ["", "garbage\n", "1 2 1\n0:main-0\n", "not a header\n0:main-0\n"],
    ids=["empty", "garbage", "truncated", "bad-header"],
)
def test_parse_marker_file_never_raises(tmp_path, content):
    """A damaged file yields nothing rather than breaking finalize()."""
    path = tmp_path / "likwid.txt"
    path.write_text(content)
    assert parse_marker_file(str(path)) == []


def test_parse_marker_file_without_a_file(tmp_path):
    """A missing marker file is simply no data."""
    assert parse_marker_file(str(tmp_path / "nope.txt")) == []


def test_result_survives_the_process_boundary():
    """Results are serialized to cross into the isolated collector's output."""
    original = LikwidRegionResult(
        tag="solve",
        group_id=1,
        group_name="MEM_DP",
        cpus=[2, 3],
        times=np.array([0.5, 0.6]),
        call_counts=np.array([7, 8], dtype=np.int64),
        event_names=["CAS_COUNT_RD", "CAS_COUNT_RD"],
        counter_names=["MBOX0C0", "MBOX1C0"],
        events=np.array([[1.0, 2.0], [3.0, 4.0]]),
        metric_names=["CPI"],
        metrics=np.array([[0.5, 0.6]]),
        source="full_api",
    )
    restored = _result_from_json(_result_to_json(original))

    assert restored.tag == original.tag
    assert restored.group_name == original.group_name
    assert restored.cpus == original.cpus
    assert restored.counter_names == original.counter_names
    assert restored.event_labels == original.event_labels
    np.testing.assert_allclose(restored.events, original.events)
    np.testing.assert_allclose(restored.metrics, original.metrics)
    np.testing.assert_array_equal(restored.call_counts, original.call_counts)


def test_isolated_collector_survives_a_segfaulting_child(tmp_path):
    """A crash behind the process boundary must not propagate.

    This is the whole reason the perfmon read-back is run out of process: on
    hosts where LIKWID cannot really count it has aborted the interpreter, and
    at finalize() time that would destroy a completed run's output.
    """
    marker = tmp_path / "likwid.txt"
    marker.write_text(MARKER_FILE)

    # -11 is what subprocess reports for a child killed by SIGSEGV. The child
    # never got as far as writing anything, so there is nothing to salvage.
    crashed = subprocess.CompletedProcess(
        args=[], returncode=-11, stdout=b"", stderr=b""
    )
    with mock.patch.dict(
        os.environ,
        {"LIKWID_FILEPATH": str(marker), "LIKWID_EVENTS": "CLOCK"},
    ), mock.patch("subprocess.run", return_value=crashed):
        assert collect_marker_results_isolated() is None


def test_isolated_collector_salvages_results_written_before_a_crash(tmp_path):
    """LIKWID can abort during teardown, after the results are already out.

    Discarding a complete document because the child later died would throw
    away the richest data for no reason, so the JSON is what counts, not the
    exit status.
    """
    marker = tmp_path / "likwid.txt"
    marker.write_text(MARKER_FILE)

    payload = [
        _result_to_json(
            LikwidRegionResult(
                tag="solve",
                group_name="CLOCK",
                cpus=[0],
                times=np.array([1.0]),
                call_counts=np.array([2], dtype=np.int64),
                event_names=["INSTR_RETIRED_ANY"],
                counter_names=["FIXC0"],
                events=np.array([[42.0]]),
                metric_names=["CPI"],
                metrics=np.array([[0.5]]),
            )
        )
    ]

    def write_then_crash(cmd, **kwargs):
        with open(cmd[-1], "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        return subprocess.CompletedProcess(args=cmd, returncode=-11)

    with mock.patch.dict(
        os.environ,
        {"LIKWID_FILEPATH": str(marker), "LIKWID_EVENTS": "CLOCK"},
    ), mock.patch("subprocess.run", side_effect=write_then_crash):
        results = collect_marker_results_isolated()

    assert results is not None
    (result,) = results
    assert result.tag == "solve"
    assert result.metric_names == ["CPI"]


def test_isolated_collector_discards_a_truncated_document(tmp_path):
    """A child that died mid-write leaves unusable JSON, so fall back."""
    marker = tmp_path / "likwid.txt"
    marker.write_text(MARKER_FILE)

    def write_partial(cmd, **kwargs):
        with open(cmd[-1], "w", encoding="utf-8") as fh:
            fh.write('[{"tag": "solve", "times": [1.0')
        return subprocess.CompletedProcess(args=cmd, returncode=-11)

    with mock.patch.dict(
        os.environ,
        {"LIKWID_FILEPATH": str(marker), "LIKWID_EVENTS": "CLOCK"},
    ), mock.patch("subprocess.run", side_effect=write_partial):
        assert collect_marker_results_isolated() is None


def test_isolated_collector_survives_a_hanging_child(tmp_path):
    """A child that never returns is abandoned, not waited on forever."""
    marker = tmp_path / "likwid.txt"
    marker.write_text(MARKER_FILE)

    with mock.patch.dict(
        os.environ,
        {"LIKWID_FILEPATH": str(marker), "LIKWID_EVENTS": "CLOCK"},
    ), mock.patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="x", timeout=1),
    ):
        assert collect_marker_results_isolated() is None


def test_isolated_collector_skipped_outside_a_likwid_run():
    """No marker environment means there is nothing to spawn a child for."""
    with mock.patch.dict(os.environ, {}, clear=True):
        with mock.patch("subprocess.run") as run:
            assert collect_marker_results_isolated() is None
        run.assert_not_called()


def test_collection_falls_back_to_the_marker_file(monkeypatch, tmp_path):
    """When the isolated read-back fails, real values still reach the file.

    The fallback loses event names and derived metrics, but keeps the counts,
    runtimes and raw counter values -- which is the difference between a
    degraded run and a lost one.
    """
    marker = tmp_path / "likwid.txt"
    marker.write_text(MARKER_FILE)

    pylikwid = mock.Mock()
    pylikwid.markergetregion.return_value = (1, [1.0], 0.5, 1)
    monkeypatch.setattr(profile_config_module, "_import_pylikwid", lambda: pylikwid)
    monkeypatch.setattr(region_profiler_module, "_import_pylikwid", lambda: pylikwid)
    monkeypatch.setenv("LIKWID_FILEPATH", str(marker))
    monkeypatch.setenv("LIKWID_EVENTS", "CLOCK")
    monkeypatch.setenv("LIKWID_THREADS", "10")
    # Stand in for the child crashing on a host that cannot count.
    monkeypatch.setattr(
        likwid_data_module, "collect_marker_results_isolated", lambda *a, **k: None
    )

    try:
        ProfileManager.setup(use_likwid=True, file_path=str(tmp_path / "out.h5"))
        results = ProfileManager.get_config().collect_likwid_results(["main"])
    finally:
        ProfileManager._reset()

    by_tag = {r.tag: r for r in results}
    assert sorted(by_tag) == ["io", "main"]
    assert all(r.source == "marker_file" for r in results)
    assert by_tag["main"].events[0, 0] == pytest.approx(1.898247e07)
    assert by_tag["io"].call_counts.tolist() == [4]


def test_collect_marker_results_without_likwid_environment():
    """Outside a LIKWID run there is no marker file, so nothing is collected."""
    with mock.patch.dict(os.environ, {}, clear=True):
        assert collect_marker_results(mock.Mock()) == []


def test_collect_marker_results_survives_a_failing_perfmon(tmp_path):
    """A LIKWID failure degrades to "no counters", it does not propagate."""
    marker_file = tmp_path / "likwid.txt"
    marker_file.write_text("")

    pylikwid = mock.Mock()
    pylikwid.init.side_effect = RuntimeError("no access to performance counters")

    with mock.patch.dict(
        os.environ,
        {
            "LIKWID_FILEPATH": str(marker_file),
            "LIKWID_THREADS": "0,1",
            "LIKWID_EVENTS": "CLOCK",
        },
    ):
        assert collect_marker_results(pylikwid) == []


def test_collect_marker_results_rejects_error_codes(tmp_path):
    """LIKWID reports errors as negative region counts; those are not data."""
    marker_file = tmp_path / "likwid.txt"
    marker_file.write_text("")

    pylikwid = mock.Mock()
    pylikwid.init.return_value = 0
    pylikwid.addeventset.return_value = 0
    pylikwid.getnameofgroup.return_value = "CLOCK"
    pylikwid.markernumregions.return_value = -22

    with mock.patch.dict(
        os.environ,
        {
            "LIKWID_FILEPATH": str(marker_file),
            "LIKWID_THREADS": "0",
            "LIKWID_EVENTS": "CLOCK",
        },
    ):
        assert collect_marker_results(pylikwid) == []
    pylikwid.finalize.assert_called_once()


def test_event_labels_disambiguate_repeated_events():
    """A group measuring one event on many counters must stay distinguishable.

    MEM_DP programs CAS_COUNT_RD once per memory channel, so the bare event
    names collide; the counter register is what separates them.
    """
    result = LikwidRegionResult(
        tag="solve",
        event_names=["INSTR_RETIRED_ANY", "CAS_COUNT_RD", "CAS_COUNT_RD"],
        counter_names=["FIXC0", "MBOX0C0", "MBOX1C0"],
        events=np.array([[1.0], [2.0], [3.0]]),
    )
    assert result.event_labels == [
        "INSTR_RETIRED_ANY",  # unique, so left alone
        "CAS_COUNT_RD:MBOX0C0",
        "CAS_COUNT_RD:MBOX1C0",
    ]
    assert len(set(result.event_labels)) == 3


def test_event_labels_fall_back_to_position_without_counter_names():
    """Files predating counter names still get unique labels."""
    result = LikwidRegionResult(
        tag="solve",
        event_names=["CAS_COUNT_RD", "CAS_COUNT_RD"],
        counter_names=[],
        events=np.array([[2.0], [3.0]]),
    )
    assert result.event_labels == ["CAS_COUNT_RD#0", "CAS_COUNT_RD#1"]


def test_dataframe_keeps_every_channel_of_a_repeated_event(tmp_path):
    """Repeated events must not collapse into a single column."""
    pytest.importorskip("pandas")

    result = _make_result(nthreads=1)
    result.event_names = ["CAS_COUNT_RD", "CAS_COUNT_RD"]
    result.counter_names = ["MBOX0C0", "MBOX1C0"]
    result.events = np.array([[11.0], [22.0]])

    path = tmp_path / "profiling_data.h5"
    _write_merged_file(path, [result])

    df = read_h5(path).likwid_to_dataframe()
    # Both channels survive, with their own values.
    assert df["CAS_COUNT_RD:MBOX0C0"].tolist() == [11.0]
    assert df["CAS_COUNT_RD:MBOX1C0"].tolist() == [22.0]


def test_counter_names_round_trip(tmp_path):
    """Counter registers survive the write/read cycle."""
    result = _make_result(nthreads=1)
    result.counter_names = ["FIXC0", "FIXC1"]
    path = tmp_path / "profiling_data.h5"
    _write_merged_file(path, [result])

    assert read_h5(path).get_likwid_region("solve").counter_names == [
        "FIXC0",
        "FIXC1",
    ]


def test_search_dirs_come_from_the_likwid_module(tmp_path, monkeypatch):
    """The lib directory of a loaded LIKWID module is where we look first."""
    lib = tmp_path / "lib"
    lib.mkdir()
    monkeypatch.setattr(profile_config_module.shutil, "which", lambda _: None)
    with mock.patch.dict(os.environ, {"LIKWID_HOME": str(tmp_path)}, clear=True):
        assert str(lib) in _liblikwid_search_dirs()


def test_search_dirs_fall_back_to_likwid_perfctr(tmp_path, monkeypatch):
    """With no LIKWID_* variables, the binary on PATH still locates the prefix."""
    (tmp_path / "bin").mkdir()
    (tmp_path / "lib").mkdir()
    perfctr = tmp_path / "bin" / "likwid-perfctr"
    perfctr.touch()
    monkeypatch.setattr(
        profile_config_module.shutil,
        "which",
        lambda name: str(perfctr) if name == "likwid-perfctr" else None,
    )
    with mock.patch.dict(os.environ, {}, clear=True):
        assert str(tmp_path / "lib") in _liblikwid_search_dirs()


def test_search_dirs_skip_paths_that_do_not_exist(monkeypatch):
    """Stale variables pointing nowhere must not become search entries."""
    monkeypatch.setattr(profile_config_module.shutil, "which", lambda _: None)
    with mock.patch.dict(
        os.environ, {"LIKWID_HOME": "/nonexistent/likwid"}, clear=True
    ):
        assert _liblikwid_search_dirs() == []


def _patch_pylikwid_import(monkeypatch, results):
    """Make ``import pylikwid`` yield ``results`` in turn (raising exceptions).

    Every other import is left alone, so the interpreter keeps working while
    the test drives just this one module.
    """
    attempts = []
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name != "pylikwid":
            return real_import(name, *args, **kwargs)
        outcome = results[len(attempts)]
        attempts.append(name)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(builtins, "__import__", fake_import)
    return attempts


def test_import_retries_after_preloading_liblikwid(monkeypatch):
    """A missing liblikwid is recovered from by preloading it and retrying."""
    sentinel = mock.Mock(name="pylikwid")
    attempts = _patch_pylikwid_import(
        monkeypatch,
        [ImportError("liblikwid.so.5.3: cannot open shared object file"), sentinel],
    )
    monkeypatch.setattr(profile_config_module, "_preload_liblikwid", lambda: True)

    assert _import_pylikwid() is sentinel
    assert len(attempts) == 2


def test_import_fails_when_liblikwid_cannot_be_found(monkeypatch):
    """If nothing can be preloaded, the original error still surfaces."""
    attempts = _patch_pylikwid_import(
        monkeypatch, [ImportError("liblikwid.so.5.3: cannot open shared object file")]
    )
    monkeypatch.setattr(profile_config_module, "_preload_liblikwid", lambda: False)

    with pytest.raises(ImportError, match="liblikwid"):
        _import_pylikwid()
    assert len(attempts) == 1


def test_import_does_not_retry_when_pylikwid_itself_is_missing(monkeypatch):
    """Preloading cannot conjure up the bindings, so that error propagates."""
    attempts = _patch_pylikwid_import(
        monkeypatch, [ImportError("No module named 'pylikwid'")]
    )
    preloaded = []
    monkeypatch.setattr(
        profile_config_module,
        "_preload_liblikwid",
        lambda: preloaded.append(1) or True,
    )

    with pytest.raises(ImportError, match="No module named"):
        _import_pylikwid()
    assert len(attempts) == 1
    assert preloaded == []


def test_import_error_distinguishes_a_missing_shared_library():
    """pylikwid present but liblikwid unloadable must not read as "not installed"."""
    exc = ImportError(
        "liblikwid.so.5.3: cannot open shared object file: No such file or directory"
    )
    with mock.patch.dict(os.environ, {"LIKWID_HOME": "/opt/likwid"}):
        message = _pylikwid_import_error(exc)

    assert "pylikwid is installed" in message
    assert "not installed" not in message
    # The fix is spelled out with the real prefix, not left as an exercise.
    assert 'export LD_LIBRARY_PATH="/opt/likwid/lib:$LD_LIBRARY_PATH"' in message


def test_import_error_without_likwid_home():
    """Without LIKWID_HOME the hint still shows the shape of the fix."""
    exc = ImportError("liblikwid.so.5.3: cannot open shared object file")
    with mock.patch.dict(os.environ, {}, clear=True):
        message = _pylikwid_import_error(exc)

    assert "LD_LIBRARY_PATH" in message
    assert "<likwid-prefix>/lib" in message


def test_import_error_for_a_genuinely_missing_module():
    """A real absence points at the extra that installs it."""
    message = _pylikwid_import_error(ImportError("No module named 'pylikwid'"))

    assert "scope-profiler[likwid]" in message
    assert "LD_LIBRARY_PATH" not in message


def test_collection_only_happens_once(monkeypatch, tmp_path):
    """A second finalize() must not touch the torn-down marker API.

    ``markerclose()`` ends the marker API for the whole process; calling into
    it again crashes the interpreter instead of raising, so the second
    collection has to be refused outright.
    """
    pylikwid = mock.Mock()
    # A region LIKWID knows about, so the first pass has data to snapshot.
    pylikwid.markergetregion.return_value = (1, [42.0], 0.5, 1)
    # Two separate importers reach pylikwid: the config's, and the one the
    # region classes use to bind markerstart/stopregion. Both must be faked,
    # or this test quietly depends on a loadable liblikwid.
    monkeypatch.setattr(profile_config_module, "_import_pylikwid", lambda: pylikwid)
    monkeypatch.setattr(region_profiler_module, "_import_pylikwid", lambda: pylikwid)
    # markers_available() must say yes, but the marker file is absent so the
    # full API bows out and the snapshot fallback is what gets returned.
    monkeypatch.setenv("LIKWID_FILEPATH", str(tmp_path / "missing.txt"))
    monkeypatch.setenv("LIKWID_EVENTS", "CLOCK")
    monkeypatch.setenv("LIKWID_THREADS", "0")

    try:
        ProfileManager.setup(use_likwid=True, file_path=str(tmp_path / "out.h5"))
        config = ProfileManager.get_config()

        first = config.collect_likwid_results(["solve"])
        assert [r.tag for r in first] == ["solve"]
        assert first[0].source == "marker_api"

        second = config.collect_likwid_results(["solve"])
        assert second == []

        # The marker API was closed once and never queried again.
        pylikwid.markerclose.assert_called_once()
        assert pylikwid.markergetregion.call_count == 1
    finally:
        ProfileManager._reset()
