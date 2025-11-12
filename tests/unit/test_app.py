def test_import_app():
    from scope_profiler.main import main

    print("app imported")
    main()


def test_profile_manager(
    sample_duration=1.0, sample_interval=1.0, time_trace=True, likwid=False
):

    from scope_profiler.profiling import (
        ProfileManager,
        ProfilingConfig,
        pylikwid_markerclose,
        pylikwid_markerinit,
    )

    print(f"{type(sample_duration) = }")
    config = ProfilingConfig(
        sample_duration=sample_duration,
        sample_interval=sample_interval,
        likwid=likwid,
        time_trace=time_trace,
        simulation_label="",
    )
    pylikwid_markerinit()
    with ProfileManager.profile_region("main"):
        x = 0
        for _ in range(10):
            with ProfileManager.profile_region("iteration"):
                x += 1.0
    pylikwid_markerclose()
    if config.time_trace:
        ProfileManager.print_summary()
        # ProfileManager.save_to_pickle("profiling_time_trace.pkl")


if __name__ == "__main__":
    test_import_app()
