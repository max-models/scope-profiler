from time import sleep

import pytest

import scope_profiler.tests.examples as examples
from scope_profiler import ProfileManager
from scope_profiler.region_profiler import (
    BaseProfileRegion,
    DisabledProfileRegion,
    FullProfileRegion,
    FullProfileRegionNoFlush,
    LikwidOnlyProfileRegion,
    NCallsOnlyProfileRegion,
    TimeOnlyProfileRegion,
    TimeOnlyProfileRegionNoFlush,
)


@pytest.mark.parametrize("time_trace", [True, False])
@pytest.mark.parametrize("use_likwid", [False])
@pytest.mark.parametrize("num_loops", [10, 50, 100])
@pytest.mark.parametrize("profiling_activated", [True, False])
def test_profile_manager(
    time_trace: bool,
    use_likwid: bool,
    num_loops: int,
    profiling_activated: bool,
):
    ProfileManager.setup(
        use_likwid=use_likwid,
        time_trace=time_trace,
        profiling_activated=profiling_activated,
        flush_to_disk=True,
    )

    examples.loop(
        label="loop1",
        num_loops=num_loops,
    )

    examples.loop(
        label="loop2",
        num_loops=num_loops * 2,
    )

    @ProfileManager.profile("test_decorator_labeled")
    def test_decorator():
        return

    @ProfileManager.profile
    def test_decorator_unlabeled():
        return

    for i in range(num_loops):
        test_decorator()
        test_decorator_unlabeled()

    with ProfileManager.profile_region("main"):
        pass

    ProfileManager.finalize()

    regions = ProfileManager.get_all_regions()

    print(
        f"{profiling_activated = } {time_trace = } {ProfileManager._config.profiling_activated = }"
    )

    if profiling_activated:
        assert regions["loop1"].num_calls == num_loops
        assert regions["loop2"].num_calls == num_loops * 2
        assert regions["test_decorator_labeled"].num_calls == num_loops
        assert regions["test_decorator_unlabeled"].num_calls == num_loops
        assert regions["main"].num_calls == 1
    else:
        assert regions["loop1"].num_calls == 0
        assert regions["loop2"].num_calls == 0
        assert regions["test_decorator_labeled"].num_calls == 0
        assert regions["test_decorator_unlabeled"].num_calls == 0
        assert regions["main"].num_calls == 0


def test_all_region_types():
    # Disabled region
    ProfileManager.setup(
        use_likwid=False,
        time_trace=False,
        profiling_activated=False,
        flush_to_disk=False,
    )

    with ProfileManager.profile_region("disabled_region"):
        pass

    region = ProfileManager.get_region("disabled_region")
    assert isinstance(region, DisabledProfileRegion)
    assert region.num_calls == 0

    # NCallsOnly region
    ProfileManager.setup(
        use_likwid=False,
        time_trace=False,
        profiling_activated=True,
        flush_to_disk=False,
    )

    with ProfileManager.profile_region("ncalls_region"):
        pass

    region = ProfileManager.get_region("ncalls_region")
    assert isinstance(region, NCallsOnlyProfileRegion)
    assert region.num_calls == 1
    assert region.get_durations_numpy().size == 0

    # Time-only region
    ProfileManager._region_cls = TimeOnlyProfileRegion
    with ProfileManager.profile_region("time_only_region"):
        sleep(0.001)

    region = ProfileManager.get_region("time_only_region")
    assert isinstance(region, TimeOnlyProfileRegion)
    assert region.num_calls == 1
    assert region.ptr == 1
    durations = region.get_durations_numpy()
    assert durations[0] > 0

    # Time-only region without flush
    ProfileManager._region_cls = TimeOnlyProfileRegionNoFlush
    with ProfileManager.profile_region("time_only_noflush"):
        sleep(0.001)

    region = ProfileManager.get_region("time_only_noflush")
    assert isinstance(region, TimeOnlyProfileRegionNoFlush)
    assert region.num_calls == 1
    assert region.ptr == 1
    durations = region.get_durations_numpy()
    assert durations[0] > 0

    # LIKWID-only region (mocked if pylikwid not installed)
    try:
        ProfileManager._region_cls = LikwidOnlyProfileRegion
        with ProfileManager.profile_region("likwid_only"):
            pass
        region = ProfileManager.get_region("likwid_only")
        assert isinstance(region, LikwidOnlyProfileRegion)
        assert region.num_calls == 1
    except ModuleNotFoundError:
        print("pylikwid not installed, skipping LIKWID-only test")

    # Full region (time + LIKWID)
    try:
        ProfileManager._region_cls = FullProfileRegion
        with ProfileManager.profile_region("full_region"):
            sleep(0.001)
        region = ProfileManager.get_region("full_region")
        assert isinstance(region, FullProfileRegion)
        assert region.num_calls == 1
        durations = region.get_durations_numpy()
        assert durations.size == 1
        assert durations[0] > 0
    except ModuleNotFoundError:
        print("pylikwid not installed, skipping FullProfileRegion test")

    # Finalize (should flush everything)
    ProfileManager.finalize(verbose=False)


if __name__ == "__main__":
    # test_readme()
    test_all_region_types()
