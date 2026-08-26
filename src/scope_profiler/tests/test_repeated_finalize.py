"""Calling finalize() more than once in a process, as a restarted run does."""

from time import sleep

import pytest

from scope_profiler import ProfileManager, read_h5


def _run(label: str, num_calls: int) -> None:
    for _ in range(num_calls):
        with ProfileManager.profile_region(label):
            sleep(0.0001)


def test_second_finalize_without_setup(tmp_path):
    """A second run must finalize cleanly and report only its own events."""
    out = tmp_path / "profiling_data.h5"
    ProfileManager.setup(file_path=str(out))

    _run("step", 3)
    ProfileManager.finalize(verbose=False)

    region = read_h5(str(out)).get_region("step")
    assert region.num_calls == 3

    # Second run, reusing the config and the region objects of the first.
    _run("step", 5)
    ProfileManager.finalize(verbose=False)

    region = read_h5(str(out)).get_region("step")
    assert region.num_calls == 5, "second run inherited the first run's events"
    assert len(region.durations) == 5


def test_stale_regions_do_not_leak_into_the_second_run(tmp_path):
    """A region used only in the first run must not reappear in the second."""
    out = tmp_path / "profiling_data.h5"
    ProfileManager.setup(file_path=str(out))

    _run("first_only", 2)
    ProfileManager.finalize(verbose=False)
    assert "first_only" in read_h5(str(out)).region_names

    _run("second_only", 2)
    ProfileManager.finalize(verbose=False)

    names = read_h5(str(out)).region_names
    assert "second_only" in names
    assert "first_only" not in names


def test_in_memory_counter_keeps_running_across_runs(tmp_path):
    """The file holds each run's calls; the region counter keeps the total."""
    out = tmp_path / "profiling_data.h5"
    ProfileManager.setup(file_path=str(out))

    _run("step", 3)
    ProfileManager.finalize(verbose=False)
    assert read_h5(str(out)).get_region("step").num_calls == 3

    _run("step", 5)
    ProfileManager.finalize(verbose=False)
    assert read_h5(str(out)).get_region("step").num_calls == 5

    # In memory the counter keeps running for the lifetime of the process.
    assert ProfileManager.get_region("step").num_calls == 8


def test_finalize_keeps_buffers_without_file_output(tmp_path):
    """With file output off the buffers are the only copy, so they survive."""
    ProfileManager.setup(
        file_path=str(tmp_path / "profiling_data.h5"),
        deactivate_file_output=True,
    )

    _run("step", 4)
    ProfileManager.finalize(verbose=False)

    region = ProfileManager.get_region("step")
    assert region.num_calls == 4
    assert region.ptr == 4


def test_finalize_inside_an_open_region(tmp_path):
    """A region still open at finalize() keeps its reserved slot."""
    ProfileManager.setup(file_path=str(tmp_path / "profiling_data.h5"))

    with ProfileManager.profile_region("outer"):
        ProfileManager.finalize(verbose=False)

    region = ProfileManager.get_region("outer")
    # Rewinding under the open scope would have let the exit write into a slot
    # the next call could reuse; the region is left alone instead.
    assert region.num_calls == 1
    assert region.ptr == 1
    assert region.get_durations_numpy()[0] > 0

    # The call had no end timestamp yet when finalize() ran, so it is absent
    # from the file rather than recorded as ending at zero.
    results = read_h5(str(tmp_path / "profiling_data.h5"))
    assert "outer" not in results.region_names


def test_open_region_does_not_hide_its_completed_calls(tmp_path):
    """Only the call still running is held back, not the ones already closed."""
    ProfileManager.setup(file_path=str(tmp_path / "profiling_data.h5"))

    with ProfileManager.profile_region("outer"):
        _run("inner", 2)
        ProfileManager.finalize(verbose=False)

    results = read_h5(str(tmp_path / "profiling_data.h5"))
    assert results.get_region("inner").num_calls == 2
    assert "outer" not in results.region_names


def test_session_finalizes_and_returns_results(tmp_path):
    out = tmp_path / "session.h5"
    with ProfileManager.session(
        file_path=str(out), verbose=False, return_results=True
    ) as run:
        _run("step", 2)

    assert run.results is not None
    assert run.results["step"].num_calls == 2
    assert read_h5(str(out))["step"].num_calls == 2


def test_session_finalizes_when_body_raises(tmp_path):
    out = tmp_path / "session-error.h5"
    with pytest.raises(RuntimeError):
        with ProfileManager.session(file_path=str(out), verbose=False):
            _run("step", 1)
            raise RuntimeError("expected")

    assert read_h5(str(out))["step"].num_calls == 1


def test_region_tags_round_trip_and_must_be_consistent(tmp_path):
    out = tmp_path / "tags.h5"
    ProfileManager.setup(file_path=str(out))

    with ProfileManager.profile_region("solve", tags=("compute", "hot")):
        pass
    # Omitting tags means "unspecified", which is convenient for shared
    # helpers that use a region name already configured by the caller.
    with ProfileManager.profile_region("solve"):
        pass
    with pytest.raises(ValueError, match="already has tags"):
        ProfileManager.profile_region("solve", tags=["io"])

    results = ProfileManager.finalize(verbose=False, return_results=True)
    assert results["solve"].tags == ("compute", "hot")
    assert results.summary()[0]["tags"] == ("compute", "hot")
    assert read_h5(str(out))["solve"].tags == ("compute", "hot")


def test_completed_calls_are_not_re_reported_under_an_open_one(tmp_path):
    """A call in flight must not pin the finished calls below it in the buffer.

    Before the buffer was compacted, mark_written() gave up entirely on a
    region with an open scope, so the next finalize() handed out every
    completed call a second time - with a second set of call ids.
    """
    ProfileManager.setup(file_path=str(tmp_path / "profiling_data.h5"))

    with ProfileManager.profile_region("step"):
        pass
    with ProfileManager.profile_region("step"):
        first = ProfileManager.finalize(verbose=False, return_results=True)
    second = ProfileManager.finalize(verbose=False, return_results=True)

    started_first = first["step"][0].start_times_ns.tolist()
    started_second = second["step"][0].start_times_ns.tolist()
    assert len(started_first) == 1
    assert len(started_second) == 1
    assert started_first != started_second

    # And the second run's call gets its own id rather than restarting at 0.
    assert first["step"][0].call_ids.tolist() == [0]
    assert second["step"][0].call_ids.tolist() == [1]


def test_call_ids_restart_only_when_a_new_run_starts(tmp_path):
    ProfileManager.setup(file_path=str(tmp_path / "one.h5"))
    _run("step", 2)
    first = ProfileManager.finalize(verbose=False, return_results=True)

    ProfileManager.setup(file_path=str(tmp_path / "two.h5"))
    _run("step", 2)
    second = ProfileManager.finalize(verbose=False, return_results=True)

    assert first["step"][0].call_ids.tolist() == [0, 1]
    assert second["step"][0].call_ids.tolist() == [0, 1]


def test_an_open_decorated_call_does_not_duplicate_the_finished_ones(tmp_path):
    """The one case compaction cannot handle, and must not double-report.

    A decorated call keeps its slot index in the wrapper's own frame, where
    nothing can remap it, so the buffer cannot be compacted under it. The
    slots already copied out are remembered instead.
    """
    ProfileManager.setup(file_path=str(tmp_path / "profiling_data.h5"))

    @ProfileManager.profile("work")
    def work(finalize=False):
        if finalize:
            return ProfileManager.finalize(verbose=False, return_results=True)
        return None

    work()
    first = work(finalize=True)
    second = ProfileManager.finalize(verbose=False, return_results=True)

    # The call that had returned goes out; the one still running does not,
    # and it is the only thing the next finalize() reports.
    assert first["work"][0].num_calls == 1
    assert second["work"][0].num_calls == 1
    assert (
        first["work"][0].start_times_ns.tolist()
        != second["work"][0].start_times_ns.tolist()
    )
    assert first["work"][0].call_ids.tolist() == [0]
    assert second["work"][0].call_ids.tolist() == [1]
    assert ProfileManager.get_region("work").num_calls == 2
