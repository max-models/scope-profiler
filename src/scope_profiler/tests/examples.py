from scope_profiler import ProfileManager


def loop(
    label,
    num_loops: int = 100,
):
    s = 0
    for i in range(num_loops):
        with ProfileManager.profile_region(region_name=label):
            s += 1


@ProfileManager.profile("fibonacci")
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def fibonacci_context_manager(n, region_name="fibonacci_ctx"):
    with ProfileManager.profile_region(region_name):
        if n < 2:
            return n
        return fibonacci_context_manager(n - 1, region_name) + fibonacci_context_manager(
            n - 2, region_name
        )


if __name__ == "__main__":
    ProfileManager.setup(
        use_likwid=False,
        time_trace=True,
        flush_to_disk=True,
    )

    num_loops = 10

    loop(
        label="loop3",
        num_loops=num_loops,
    )

    loop(
        label="loop2",
        num_loops=num_loops * 2,
    )

    ProfileManager.finalize()
