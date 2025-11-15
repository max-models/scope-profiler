import math
import random
import time

import numpy as np

from scope_profiler import ProfileManager


def random_math(N=100_000):
    s = 0.0
    for _ in range(N):
        x = random.random()
        s += math.sin(x) * math.sqrt(x + 1.2345)
    return s


def test_overhead():
    ProfileManager.setup(
        use_likwid=False,
        time_trace=True,
        flush_to_disk=True,
    )

    num_computations = 1
    num_tests = 10
    N = 1_000_000

    # Without ProfileManager
    elapsed_no_manager = []
    for _ in range(num_tests):
        t0 = time.perf_counter()
        for _ in range(num_computations):
            # examples.axpy(N = 50)
            random_math(N)
        t1 = time.perf_counter()
        elapsed_no_manager.append(t1 - t0)

    # With ProfileManager
    elapsed_with_manager = []
    for _ in range(num_tests):
        t0 = time.perf_counter()
        for _ in range(num_computations):
            with ProfileManager.profile_region("main"):
                # examples.axpy(N = 50)
                random_math(N)
        t1 = time.perf_counter()
        elapsed_with_manager.append(t1 - t0)

    elapsed_no_manager = np.array(elapsed_no_manager)
    elapsed_with_manager = np.array(elapsed_with_manager)

    time_no_manager = np.min(elapsed_no_manager)
    time_manager = np.min(elapsed_with_manager)
    ratio = time_manager / time_no_manager
    print(f"{time_no_manager = }")
    print(f"{time_manager = }")
    print(f"Overhead ratio = {ratio}")

    # Very low bar, make sure the overhead is <3%
    assert ratio < 1.03

    ProfileManager.finalize()


if __name__ == "__main__":
    test_overhead()
