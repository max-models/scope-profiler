def test_import_app():
    from scope_profiler.main import main

    print("app imported")
    main()


def test_profile_manager(
    sample_duration=1.0, sample_interval=1.0, time_trace=True, likwid=False
):

    from scope_profiler.profiling import (ProfileManager, ProfilingConfig,
                                          pylikwid_markerclose,
                                          pylikwid_markerinit)

    config = ProfilingConfig()
    config.likwid = likwid
    config.sample_duration = float(sample_duration)
    config.sample_interval = float(sample_interval)
    config.time_trace = time_trace
    config.simulation_label = ""
    pylikwid_markerinit()
    with ProfileManager.profile_region("main"):
        print("hello")
        for i in range(10):
            with ProfileManager.profile_region("iteration"):
                print("iteration")
    pylikwid_markerclose()
    if config.time_trace:
        ProfileManager.print_summary()
        # ProfileManager.save_to_pickle("profiling_time_trace.pkl")


if __name__ == "__main__":
    test_import_app()
