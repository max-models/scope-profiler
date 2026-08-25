"""Assert that MPI is used if and only if the run was started by a launcher.

Run twice from CI with mpi4py installed, so that the two outcomes differ only
in how the process was started::

    python3 check_mpi_launch.py serial
    mpirun -n 2 python3 check_mpi_launch.py mpi 2

``serial`` asserts that no MPI call happens at all -- in particular that
``mpi4py.MPI`` is never imported, since importing it already calls
``MPI_Init``. ``mpi`` asserts that the communicator is picked up and that all
ranks end up in the merged output.

This lives outside the pytest suite because the launcher is the thing under
test; pytest itself is never started by mpirun.
"""

import sys

import h5py

from scope_profiler import ProfileManager

OUTPUT_FILE = "mpi_launch_check.h5"


def check(mode: str, expected_size: int) -> None:
    """Profile a trivial region and verify the MPI behaviour for ``mode``."""
    ProfileManager.setup(file_path=OUTPUT_FILE)
    config = ProfileManager.get_config()

    with ProfileManager.profile_region("check"):
        sum(range(1000))

    if mode == "serial":
        assert "mpi4py.MPI" not in sys.modules, (
            "mpi4py.MPI was imported in a run that was not started by an MPI "
            "launcher; importing it calls MPI_Init."
        )
        assert config.comm is None, f"expected no communicator, got {config.comm!r}"
    else:
        assert (
            config.comm is not None
        ), "no communicator, although the run was started by an MPI launcher"

    assert (
        config._size == expected_size
    ), f"expected {expected_size} rank(s), got {config._size}"

    ProfileManager.finalize(verbose=False)

    if config._rank == 0:
        with h5py.File(OUTPUT_FILE, "r") as f:
            if "rank_region_index" in f:
                # Schema 2 stores timing data in shared columnar datasets;
                # rank-local groups are only created for auxiliary data such
                # as LIKWID and line-profiler records.
                recorded = f["rank_region_index/ranks"][()]
                ranks = sorted({f"rank{int(rank)}" for rank in recorded})
            else:
                # Keep the checker compatible with legacy schema-1 files.
                ranks = sorted(
                    key
                    for key in f
                    if key.startswith("rank") and key[4:].isdigit()
                )
        expected_ranks = [f"rank{r}" for r in range(expected_size)]
        assert ranks == expected_ranks, f"expected {expected_ranks}, got {ranks}"

    print(f"[{mode}] OK: rank {config._rank} of {config._size}, comm={config.comm}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "serial"
    if mode not in ("serial", "mpi"):
        sys.exit(f"usage: {sys.argv[0]} [serial|mpi] [expected_size]")
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    check(mode, size)
