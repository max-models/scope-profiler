from scope_profiler import ProfileManager


def loop(
    label,
    num_loops: int = 100,
):
    s = 0
    for i in range(num_loops):
        with ProfileManager.profile_region(region_name=label):
            s += 1


if __name__ == "__main__":
    ProfileManager.setup(
        use_likwid=False,
        time_trace=True,
        flush_to_disk=True,
    )

    num_loops = 10

    loop(
        label="loop1",
        num_loops=num_loops,
    )

    loop(
        label="loop2",
        num_loops=num_loops * 2,
    )

    ProfileManager.finalize()
