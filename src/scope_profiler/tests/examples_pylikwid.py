from scope_profiler import ProfileManager


def test_pylikwid():
    ProfileManager.setup(
        use_likwid=True,
        time_trace=True,
        flush_to_disk=True,
    )

    with ProfileManager.profile_region("main"):
        x = 0
        for i in range(10):
            # Profile each iteration with a context manager
            with ProfileManager.profile_region(region_name="iteration"):
                x += 1

    ProfileManager.finalize()


if __name__ == "__main__":
    test_pylikwid()
