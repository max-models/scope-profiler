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
        from scope_profiler import read_h5

        from_disk = read_h5(ProfileManager.get_config().file_path)
        assert results.is_root
        assert results.num_ranks == from_disk.num_ranks
        assert results.summary() == from_disk.summary()
        # finalize_time_ns/start_time_ns are only rank 0's clock readings
        # (like every other run-level metadata field), so total_time is only
        # meaningful -- and only checked -- here, and must match the file.
        assert results.total_time is not None
        assert results.total_time > results.time_span
        assert results.total_time == from_disk.total_time
    else:
        # Empty and non-root, so the output calls above are no-ops here.
        assert not results.is_root
        assert results.region_names == []


if __name__ == "__main__":
    test_mpi()
