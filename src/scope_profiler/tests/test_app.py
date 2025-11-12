import pytest

import scope_profiler.tests.examples as examples
from scope_profiler.profiling import (
    ProfileManager,
    ProfilingConfig,
)


@pytest.mark.parametrize("sample_duration", [1.0, 0.5, 2.0])
@pytest.mark.parametrize("sample_interval", [1.0, 0.1, 0.5])
@pytest.mark.parametrize("time_trace", [True, False])
@pytest.mark.parametrize("use_likwid", [False])
@pytest.mark.parametrize("num_loops", [10, 50, 100])
def test_profile_manager(
    sample_duration,
    sample_interval,
    time_trace,
    use_likwid,
    num_loops,
):

    config = ProfilingConfig(
        sample_duration=sample_duration,
        sample_interval=sample_interval,
        use_likwid=use_likwid,
        time_trace=time_trace,
        simulation_label="",
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

    with ProfileManager.profile_region("main"):
        pass

    if config.time_trace:
        ProfileManager.print_summary()

    regions = ProfileManager.get_all_regions()

    assert regions["loop1"].num_calls == num_loops
    assert regions["loop2"].num_calls == num_loops * 2
    assert regions["main"].num_calls == 1


if __name__ == "__main__":
    test_profile_manager()
