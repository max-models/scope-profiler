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
