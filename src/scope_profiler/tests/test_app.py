import pytest

import scope_profiler.tests.examples as examples
from scope_profiler.profiling import (
    ProfileManager,
    ProfilingConfig,
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

    ProfilingConfig().reset()
    config = ProfilingConfig(
        use_likwid=use_likwid,
        time_trace=time_trace,
        profiling_activated=profiling_activated,
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

    @ProfileManager.profile
    def test_decorator():
        return

    for i in range(num_loops):
        test_decorator()

    with ProfileManager.profile_region("main"):
        pass

    if config.time_trace:
        ProfileManager.print_summary()

    ProfileManager.finalize()

    regions = ProfileManager.get_all_regions()

    if profiling_activated:
        assert regions["loop1"].num_calls == num_loops
        assert regions["loop2"].num_calls == num_loops * 2
        assert regions["test_decorator"].num_calls == num_loops
        assert regions["main"].num_calls == 1
    else:
        assert regions["loop1"].num_calls == 0
        assert regions["loop2"].num_calls == 0
        assert regions["test_decorator"].num_calls == 0
        assert regions["main"].num_calls == 0


if __name__ == "__main__":
    test_profile_manager(
        time_trace=True,
        use_likwid=False,
        num_loops=100,
    )
