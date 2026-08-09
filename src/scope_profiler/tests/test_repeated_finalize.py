"""Calling finalize() more than once in a process, as a restarted run does."""

from time import sleep

from scope_profiler import ProfileManager, read_h5


def _run(label: str, num_calls: int) -> None:
    for _ in range(num_calls):
        with ProfileManager.profile_region(label):
            sleep(0.0001)


def test_second_finalize_without_setup(tmp_path):
    """A second run must finalize cleanly and report only its own events."""
    out = tmp_path / "profiling_data.h5"
    ProfileManager.setup(file_path=str(out), flush_to_disk=True)

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
    ProfileManager.setup(file_path=str(out), flush_to_disk=True)

    _run("first_only", 2)
    ProfileManager.finalize(verbose=False)
    assert "first_only" in read_h5(str(out)).region_names

    _run("second_only", 2)
    ProfileManager.finalize(verbose=False)

    names = read_h5(str(out)).region_names
    assert "second_only" in names
    assert "first_only" not in names


def test_call_counts_are_per_run_without_timing(tmp_path):
    """Call-count-only regions also report per-run numbers, not running totals."""
    out = tmp_path / "profiling_data.h5"
    ProfileManager.setup(file_path=str(out), time_trace=False)

    _run("step", 3)
    ProfileManager.finalize(verbose=False)
    assert read_h5(str(out)).get_region("step").num_calls == 3

    _run("step", 5)
    ProfileManager.finalize(verbose=False)
    assert read_h5(str(out)).get_region("step").num_calls == 5

    # In memory the counter keeps running for the lifetime of the process.
    assert ProfileManager.get_region("step").num_calls == 8


def test_finalize_keeps_buffers_when_not_flushing(tmp_path):
    """With flush_to_disk off the buffers are the only copy, so they survive."""
    ProfileManager.setup(
        file_path=str(tmp_path / "profiling_data.h5"),
        flush_to_disk=False,
    )

    _run("step", 4)
    ProfileManager.finalize(verbose=False)

    region = ProfileManager.get_region("step")
    assert region.num_calls == 4
    assert region.ptr == 4


def test_finalize_inside_an_open_region(tmp_path):
    """A region still open at finalize() keeps its reserved slot."""
    ProfileManager.setup(
        file_path=str(tmp_path / "profiling_data.h5"),
        flush_to_disk=True,
    )

    with ProfileManager.profile_region("outer"):
        ProfileManager.finalize(verbose=False)

    region = ProfileManager.get_region("outer")
    # Rewinding under the open scope would have let the exit write into a slot
    # the next call could reuse; the region is left alone instead.
    assert region.num_calls == 1
    assert region.ptr == 1
    assert region.get_durations_numpy()[0] > 0
