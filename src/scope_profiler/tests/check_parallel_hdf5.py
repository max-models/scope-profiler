"""End-to-end parallel-HDF5 writer check, launched directly by ``mpirun``.

This is intentionally not a normal pytest test: MPI initializes the process
before the test body, and all ranks must execute the same collective HDF5
metadata calls. CI runs it with::

    mpirun -n 3 python check_parallel_hdf5.py /tmp/parallel-profile.h5
"""

from __future__ import annotations

import os
import sys

import h5py

from scope_profiler import ProfileManager, read_h5


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_parallel_hdf5.py OUTPUT.h5")
    if not h5py.get_config().mpi:
        raise SystemExit("test requires an MPI-enabled h5py build")

    output = os.path.abspath(sys.argv[1])
    # Line-profiler records have rank-local, variable-length layouts. Including
    # them here proves the collective schema phase handles more than timing
    # arrays while still leaving all bulk data on its owning rank.
    ProfileManager.setup(
        file_path=output,
        output_mode="parallel",
        use_line_profiler=True,
        hdf5_compression="gzip",
        hdf5_compression_level=4,
        hdf5_chunk_size=2,
    )
    config = ProfileManager.get_config()
    rank = config._rank
    size = config._size
    assert size >= 2, "parallel-HDF5 test must run under mpirun -n 2 or greater"

    @ProfileManager.profile("line-work")
    def line_work(n):
        total = 0
        for value in range(n):
            total += value
        return total

    # Every rank contributes a common region and a differently named region.
    # This exercises collective creation for non-identical local layouts.
    with ProfileManager.profile_region("common", tags=("shared",)):
        sum(range(rank + 1))
    with ProfileManager.profile_region(f"rank-{rank}", tags=(f"owner-{rank}",)):
        sum(range(rank + 2))
    line_work(rank + 3)

    results = ProfileManager.finalize(verbose=False, return_results=True)
    if rank == 0:
        assert results.is_root
        assert results.num_ranks == size
        assert results["common"].num_calls == size
        assert results["common"].ranks == list(range(size))
        assert results["common"].tags == ("shared",)
        assert set(results.line_profile) == set(range(size))
        assert all(
            any(record["region"] == "line-work" for record in records)
            for records in results.line_profile.values()
        )
        for owner in range(size):
            region = results[f"rank-{owner}"]
            assert region.ranks == [owner]
            assert region.num_calls == 1
            assert region.tags == (f"owner-{owner}",)

        # Verify independently through the public on-disk reader as well.
        from_disk = read_h5(output)
        assert from_disk.summary() == results.summary()
        assert sorted(from_disk.region_names) == [
            "common",
            "line-work",
            *(f"rank-{owner}" for owner in range(size)),
        ]
        with h5py.File(output, "r") as handle:
            dataset = handle["rank0/regions/common/start_times"]
            assert dataset.compression == "gzip"
            assert dataset.compression_opts == 4
            assert dataset.chunks == (1,)
    else:
        assert not results.is_root
        assert results.region_names == []
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
