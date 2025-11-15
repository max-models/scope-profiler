import pytest

import scope_profiler.tests.examples as examples
from scope_profiler import ProfileManager, ProfilingConfig


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
    ProfilingConfig().reset()
    config = ProfilingConfig(
        use_likwid=use_likwid,
        time_trace=time_trace,
        profiling_activated=profiling_activated,
        flush_to_disk=True,
    )
    ProfileManager.reset()

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

    print(f"{profiling_activated = } {time_trace = }")

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


if __name__ == "__main__":
    test_profile_manager(
        time_trace=True,
        use_likwid=False,
        num_loops=100,
    )
