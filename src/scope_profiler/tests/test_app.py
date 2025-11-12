import scope_profiler.tests.examples as examples
from scope_profiler.profiling import (
    ProfileManager,
    ProfilingConfig,
)


def test_profile_manager(
    sample_duration=1.0,
    sample_interval=1.0,
    time_trace=True,
    use_likwid=False,
    num_loops=100,
):

    config = ProfilingConfig(
        sample_duration=sample_duration,
        sample_interval=sample_interval,
        use_likwid=use_likwid,
        time_trace=time_trace,
        simulation_label="",
    )

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
