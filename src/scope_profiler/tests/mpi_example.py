import math
import random

from scope_profiler import ProfileManager, ProfilingConfig


def random_math(N=100_000):
    s = 0.0
    for _ in range(N):
        x = random.random()
        s += math.sin(x) * math.sqrt(x + 1.2345)
    return s


def test_mpi():

    config = ProfilingConfig(
        use_likwid=False,
        time_trace=True,
        flush_to_disk=True,
    )

    ProfileManager.reset()
    num_computations = 10
    N = 100_000
    import time

    for _ in range(num_computations):
        with ProfileManager.profile_region("main"):
            random_math(N)
        time.sleep(0.01)

    ProfileManager.finalize()

    # ProfileManager.print_summary()


if __name__ == "__main__":
    test_mpi()
