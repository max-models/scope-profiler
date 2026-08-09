"""finalize(return_results=True): the run's data without reading it back."""

from time import sleep

import pytest

from scope_profiler import ProfileManager, ProfilingResults, read_h5


def _run(label: str, num_calls: int) -> None:
    for _ in range(num_calls):
        with ProfileManager.profile_region(label):
            sleep(0.0001)


@pytest.fixture(autouse=True)
def _reset():
    yield
    ProfileManager._reset()


def test_returned_results_match_the_written_file(tmp_path):
    """The in-memory results are the same data the output file holds."""
    out = tmp_path / "profiling_data.h5"
    ProfileManager.setup(file_path=str(out))

    _run("outer", 2)
    _run("inner", 3)

    results = ProfileManager.finalize(verbose=False, return_results=True)
    from_disk = read_h5(str(out))

    assert isinstance(results, ProfilingResults)
    assert sorted(results.region_names) == sorted(from_disk.region_names)
    assert results.summary() == from_disk.summary()
    assert results.events() == from_disk.events()
    assert results.call_stack() == from_disk.call_stack()
    assert results.metadata == from_disk.metadata
    assert results.num_ranks == from_disk.num_ranks
    assert results.time_origin == from_disk.time_origin


def test_results_without_flushing_to_disk(tmp_path):
    """The point of the option: full results when nothing is written out."""
    out = tmp_path / "profiling_data.h5"
    ProfileManager.setup(file_path=str(out), flush_to_disk=False)

    _run("step", 4)
    results = ProfileManager.finalize(verbose=False, return_results=True)

    assert results.region_names == ["step"]
    assert results["step"].num_calls == 4
    assert results["step"].total_duration > 0
    assert len(results.events()) == 4


def test_count_only_regions_are_included(tmp_path):
    """With time_trace=False the counts survive, exactly as they do on disk."""
    out = tmp_path / "profiling_data.h5"
    ProfileManager.setup(file_path=str(out), time_trace=False)

    _run("counted", 3)
    results = ProfileManager.finalize(verbose=False, return_results=True)

    assert results["counted"].num_calls == 3
    assert results.summary() == read_h5(str(out)).summary()


def test_second_finalize_returns_only_its_own_events(tmp_path):
    """Each finalize() is a run boundary for the returned data too."""
    out = tmp_path / "profiling_data.h5"
    ProfileManager.setup(file_path=str(out))

    _run("step", 3)
    first = ProfileManager.finalize(verbose=False, return_results=True)
    assert first["step"].num_calls == 3

    _run("step", 5)
    second = ProfileManager.finalize(verbose=False, return_results=True)
    assert second["step"].num_calls == 5, "second run inherited the first run's events"
    # The first result set is a snapshot: finalizing again must not touch it.
    assert first["step"].num_calls == 3


def test_disabled_profiling_returns_empty_results(tmp_path):
    """Nothing recorded, but still a usable object rather than None."""
    ProfileManager.setup(profiling_activated=False)

    results = ProfileManager.finalize(verbose=False, return_results=True)

    assert isinstance(results, ProfilingResults)
    assert results.region_names == []
    assert results.summary() == []


def test_non_root_results_make_the_output_paths_no_ops(tmp_path, capsys):
    """What lets an MPI script skip the rank guards; see ProfilingResults.is_root."""
    from scope_profiler import (
        ProfilingResults,
        export_prof,
        export_speedscope,
        plot_gantt,
        write_region_statistics_json,
    )

    non_root = ProfilingResults({}, file_path=str(tmp_path / "profiling_data.h5"))
    non_root._is_root = False

    non_root.print_summary()
    assert capsys.readouterr().out == ""

    figure = tmp_path / "gantt.png"
    plot_gantt(non_root, filepath=str(figure))
    assert not figure.exists()

    assert export_prof(non_root, tmp_path / "out.prof") == []
    assert export_speedscope(non_root, tmp_path / "out.speedscope.json") == []
    assert list(tmp_path.iterdir()) == []

    stats = tmp_path / "stats.json"
    write_region_statistics_json(non_root, stats)
    assert not stats.exists()


def test_empty_input_still_raises(tmp_path):
    """Nothing to draw is a mistake; a non-root rank is not."""
    from scope_profiler import plot_gantt

    with pytest.raises(ValueError, match="No profiling data provided"):
        plot_gantt([])


def test_finalize_returns_none_by_default(tmp_path):
    """The default stays a plain None, as before."""
    ProfileManager.setup(file_path=str(tmp_path / "profiling_data.h5"))
    _run("step", 1)
    assert ProfileManager.finalize(verbose=False) is None
