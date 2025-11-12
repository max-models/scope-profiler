from scope_profiler.profiling import ProfileManager


def loop(label, num_loops=100):
    s = 0
    for i in range(num_loops):
        with ProfileManager.profile_region(region_name=label):
            s += 1
