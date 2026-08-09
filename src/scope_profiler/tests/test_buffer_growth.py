"""Tests for on-demand buffer growth and the single write at finalize."""

import h5py
import numpy as np
import pytest

from scope_profiler import ProfileManager, read_h5


def _durations(region):
    return region.end_times[: region.ptr] - region.start_times[: region.ptr]


@pytest.mark.parametrize("num_calls", [1, 33, 500])
def test_buffer_grows_past_initial_capacity(num_calls):
    """The initial capacity is a starting size, not a limit."""
    ProfileManager.setup(flush_to_disk=False, buffer_limit=4)

    for _ in range(num_calls):
        with ProfileManager.profile_region("grows"):
            pass

    region = ProfileManager.get_region("grows")
    assert region.num_calls == num_calls
    assert region.ptr == num_calls
    assert region.start_times.size >= num_calls
    # Every recorded slot holds a real measurement, not uninitialized memory.
    assert np.all(_durations(region) >= 0)


def test_growth_preserves_earlier_measurements(tmp_path):
    """Reallocation must copy recorded slots to the same indices."""
    file_path = tmp_path / "growth.h5"
    ProfileManager.setup(file_path=str(file_path), buffer_limit=2)

    for _ in range(50):
        with ProfileManager.profile_region("region"):
            pass

    region = ProfileManager.get_region("region")
    starts = region.start_times[: region.ptr].copy()
    # Timestamps come from a monotonic clock, so a correct copy stays ordered;
    # a botched realloc would leave stale or garbage values out of order.
    assert np.all(np.diff(starts) > 0)

    ProfileManager.finalize(verbose=False)
    stored = read_h5(file_path)["region"][0]
    assert stored.num_calls == 50
    assert np.all(np.diff(stored.start_times) > 0)


def test_growth_during_recursion_keeps_slots_valid(tmp_path):
    """A slot reserved before growth must still receive its end timestamp."""
    file_path = tmp_path / "recursive_growth.h5"
    ProfileManager.setup(file_path=str(file_path), buffer_limit=2)

    def recurse(depth):
        with ProfileManager.profile_region("recursive"):
            if depth:
                recurse(depth - 1)

    # Deep enough to force several reallocations while outer scopes are open.
    recurse(20)

    region = ProfileManager.get_region("recursive")
    assert region.num_calls == 21
    # Every call, including the outermost ones reserved before any growth,
    # has a positive duration.
    assert np.all(_durations(region) > 0)

    ProfileManager.finalize(verbose=False)
    stored = read_h5(file_path)["recursive"][0]
    assert stored.num_calls == 21
    assert np.all(stored.durations > 0)


def test_datasets_are_exactly_sized_and_contiguous(tmp_path):
    """Writing once lets the datasets be exact-size and unchunked."""
    file_path = tmp_path / "sized.h5"
    ProfileManager.setup(file_path=str(file_path), buffer_limit=100_000)

    for _ in range(5):
        with ProfileManager.profile_region("sparse"):
            pass

    ProfileManager.finalize(verbose=False)

    with h5py.File(file_path, "r") as handle:
        dataset = handle["rank0/regions/sparse/start_times"]
        assert dataset.shape == (5,)
        # Contiguous storage: no chunk is allocated beyond the real data.
        assert dataset.chunks is None
        assert dataset.id.get_storage_size() == 5 * 8


def test_no_write_when_flush_to_disk_is_false(tmp_path):
    """flush_to_disk=False keeps everything in memory."""
    file_path = tmp_path / "in_memory.h5"
    ProfileManager.setup(file_path=str(file_path), flush_to_disk=False)

    for _ in range(3):
        with ProfileManager.profile_region("in_memory"):
            pass

    region = ProfileManager.get_region("in_memory")
    assert region.ptr == 3

    ProfileManager.finalize(verbose=False)

    with h5py.File(file_path, "r") as handle:
        # The merged file exists with metadata, but carries no region data.
        assert "rank0/regions" not in handle


def test_append_grows_the_buffer():
    """The public append() helper grows rather than overflowing."""
    ProfileManager.setup(flush_to_disk=False, buffer_limit=2)
    region = ProfileManager.profile_region("appended")

    for index in range(10):
        region.append(index, index + 1)

    assert region.ptr == 10
    assert np.array_equal(region.start_times[:10], np.arange(10))
    assert np.array_equal(region.end_times[:10], np.arange(1, 11))
