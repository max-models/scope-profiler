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


def test_results_without_file_output(tmp_path):
    """The point of the option: full results when nothing is written out."""
    out = tmp_path / "profiling_data.h5"
    ProfileManager.setup(file_path=str(out), deactivate_file_output=True)

    _run("step", 4)
    results = ProfileManager.finalize(verbose=False, return_results=True)

    assert results.region_names == ["step"]
    assert results["step"].num_calls == 4
    assert results["step"].total_duration > 0
    assert len(results.events()) == 4
    assert not out.exists()


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
    ProfileManager.setup(deactivate_profiling=True)

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


def _nested_run():
    with ProfileManager.profile_region("outer"):
        with ProfileManager.profile_region("middle"):
            with ProfileManager.profile_region("leaf"):
                sleep(0.0001)


def test_call_graph_uses_the_recorded_ids_and_renests_when_filtered(tmp_path):
    """The explicit-id path, which a file written by this run always takes.

    Filtering must renest exactly as the timestamp fallback does: a call
    whose parent was excluded moves up to its nearest surviving ancestor
    rather than keeping an id that is no longer in the graph.
    """
    ProfileManager.setup(file_path=str(tmp_path / "profiling_data.h5"))
    _nested_run()
    results = ProfileManager.finalize(verbose=False, return_results=True)

    assert results["leaf"][0].call_ids is not None  # the explicit path
    assert [
        (node["name"], node["parent_id"], node["depth"])
        for node in results.call_graph()
    ] == [("outer", None, 0), ("middle", 0, 1), ("leaf", 1, 2)]

    assert [
        (node["name"], node["parent_id"], node["depth"])
        for node in results.call_graph(exclude="middle")
    ] == [("outer", None, 0), ("leaf", 0, 1)]


def test_call_graph_agrees_with_the_timestamp_reconstruction(tmp_path):
    """Stored ids and reconstructed nesting describe the same graph.

    The two number their calls differently on purpose - call_stack's
    ``call_id`` is a position in the list it returns, so it renumbers over
    whatever survived a filter, while call_graph reports the id the run
    recorded. What must match is the shape: same calls, same depths, and
    each one hanging off the same parent.
    """
    ProfileManager.setup(file_path=str(tmp_path / "profiling_data.h5"))
    _nested_run()
    results = ProfileManager.finalize(verbose=False, return_results=True)

    def shape(nodes, id_key, parent_key):
        by_id = {node[id_key]: node for node in nodes}
        return [
            (
                node["name"],
                node["depth"],
                None if node[parent_key] is None else by_id[node[parent_key]]["name"],
            )
            for node in nodes
        ]

    for kwargs in ({}, {"exclude": "middle"}, {"exclude": "outer"}):
        stored = shape(results.call_graph(**kwargs), "call_id", "parent_id")
        reconstructed = shape(results.call_stack(**kwargs), "call_id", "parent")
        assert stored == reconstructed, kwargs
