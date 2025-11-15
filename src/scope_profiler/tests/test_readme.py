def test_readme():
    from scope_profiler import ProfileManager, ProfilingConfig

    # Setup global profiling configuration
    config = ProfilingConfig(
        use_likwid=False,
        time_trace=True,
        flush_to_disk=True,
    )
    ProfileManager.set_config(config)

    # Profile the main() function with a decorator
    @ProfileManager.profile("main")
    def main():
        x = 0
        for i in range(10):
            # Profile each iteration with a context manager
            with ProfileManager.profile_region(region_name="iteration"):
                x += 1

    # Call main
    main()

    # Finalize profiler
    ProfileManager.finalize()


if __name__ == "__main__":
    test_readme()
