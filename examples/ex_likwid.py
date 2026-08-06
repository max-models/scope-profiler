"""Profile regions with LIKWID hardware counters and read the counters back.

Run this under LIKWID's marker mode, otherwise LIKWID collects nothing::

    likwid-perfctr -C 0 -g CLOCK -m python examples/ex_likwid.py

    # or, across MPI ranks:
    likwid-mpirun -n 2 -g FLOPS_DP -mpi openmpi -marker python examples/ex_likwid.py

``ProfileManager.finalize()`` closes the LIKWID markers, reads every marker
region of the run back, and stores the raw events and LIKWID's derived metrics
in the HDF5 file next to the timing data.
"""

import numpy as np

from scope_profiler import ProfileManager, ProfilingH5Reader

H5_PATH = "likwid_profiling_data.h5"


@ProfileManager.profile("matmul")
def matmul(n: int) -> float:
    """Multiply two n x n matrices; a region with plenty of FLOPs to count."""
    a = np.random.rand(n, n)
    b = np.random.rand(n, n)
    return float((a @ b).sum())


def main() -> None:
    ProfileManager.setup(
        use_likwid=True,
        time_trace=True,
        flush_to_disk=True,
        file_path=H5_PATH,
    )

    with ProfileManager.profile_region("main"):
        for _ in range(3):
            matmul(256)

        with ProfileManager.profile_region("memory_bound"):
            data = np.zeros(4_000_000)
            data += 1.0

    ProfileManager.finalize()

    # Everything below reads the finished file, the same way any downstream
    # analysis would.
    reader = ProfilingH5Reader(H5_PATH)

    if not reader.has_likwid:
        print(
            "No LIKWID counters in the output -- run this under "
            "`likwid-perfctr -C 0 -g CLOCK -m python examples/ex_likwid.py`."
        )
        return

    print(f"\nLIKWID regions in {H5_PATH}:")
    reader.print_likwid_summary()

    # The structured view: one LikwidRegionResult per (rank, region).
    for rank, regions in reader.get_likwid_regions().items():
        for tag, result in regions.items():
            print(f"\nrank {rank} / {tag}  (group {result.group_name!r})")
            print(f"  hardware threads : {result.cpus}")
            print(f"  LIKWID runtime   : {result.times[0]:.6f} s")
            print(f"  calls            : {result.call_counts[0]}")
            # event_labels disambiguates events a group programs on several
            # counters (MEM_DP measures CAS_COUNT_RD once per memory channel).
            for name, values in zip(result.event_labels, result.events):
                print(f"  event  {name:<28s} {values[0]:>18.1f}")
            for name, values in zip(result.metric_names, result.metrics):
                print(f"  metric {name:<28s} {values[0]:>18.4f}")

    # ... or as a tidy table, one row per (rank, region, hardware thread).
    try:
        df = reader.likwid_to_dataframe()
    except ImportError:
        print("\n(install scope-profiler[pproc] for the DataFrame view)")
    else:
        print("\nAs a DataFrame:")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
