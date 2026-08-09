import math
import random

from scope_profiler import ProfileManager


def random_math(N=100_000):
    s = 0.0
    for _ in range(N):
        x = random.random()
        s += math.sin(x) * math.sqrt(x + 1.2345)
    return s


def test_mpi():
    ProfileManager.setup(
        use_likwid=False,
        time_trace=True,
        flush_to_disk=True,
    )

    num_computations = 10
    N = 10_000
    import time

    for _ in range(num_computations):
        with ProfileManager.profile_region("main"):
            random_math(N)
        time.sleep(0.01)

    # The gather behind return_results is collective, so every rank asks for
    # it; only rank 0 gets the merged run back, like the output file.
    results = ProfileManager.finalize(return_results=True)
    if ProfileManager.get_config()._rank == 0:
        from scope_profiler import ProfilingH5Reader

        from_disk = ProfilingH5Reader(ProfileManager.get_config().file_path)
        assert results.num_ranks == from_disk.num_ranks
        assert results.summary() == from_disk.summary()
    else:
        assert results is None


if __name__ == "__main__":
    test_mpi()
