"""End-to-end check that LIKWID counters reach the HDF5 output.

Must be launched under LIKWID's marker mode, e.g.::

    likwid-perfctr -C 0 -g CLOCK -m python src/scope_profiler/tests/examples_pylikwid.py
    likwid-mpirun -n 2 -g FLOPS_SP -mpi openmpi -stats -marker \
        python src/scope_profiler/tests/examples_pylikwid.py

Started as a plain ``python ...`` the LIKWID marker calls become no-ops, so the
counter assertions are skipped and only the timing data is checked. That keeps
this file usable as a smoke test on machines without LIKWID.
"""

import os

from scope_profiler import ProfileManager

H5_PATH = "profiling_data_likwid.h5"


def test_pylikwid():
    """Profile a few regions with LIKWID enabled and verify the output file."""
    ProfileManager.setup(
        use_likwid=True,
        file_path=H5_PATH,
    )

    with ProfileManager.profile_region("main"):
        x = 0
        for i in range(10):
            # Profile each iteration with a context manager
            with ProfileManager.profile_region(region_name="iteration"):
                x += 1

        # Something with enough work in it to move the hardware counters.
        with ProfileManager.profile_region("busy"):
            x += sum(i * i for i in range(200_000))

    # ``finalize()`` is collective, but non-root ranks return before rank 0
    # has necessarily closed the merged HDF5 file. Requesting the in-memory
    # result avoids having every rank race to reopen that file; rank 0's
    # result is assembled from the same payload that was written to disk.
    results = ProfileManager.finalize(return_results=True)
    if not results.is_root:
        return

    # Timing data is recorded regardless of whether LIKWID is active.
    assert "main" in results.region_names
    assert "iteration" in results.region_names
    # num_calls is summed over ranks, and every rank runs the same 10 iterations.
    assert results["iteration"].num_calls == 10 * results.num_ranks

    under_likwid = bool(os.environ.get("LIKWID_FILEPATH"))
    if not under_likwid:
        print(
            "Not running under likwid-perfctr -m: LIKWID markers are no-ops, "
            "skipping the hardware counter assertions.",
        )
        assert not results.has_likwid
        return

    assert results.has_likwid, f"no LIKWID data in {H5_PATH}"
    assert results.likwid_ranks, "no rank recorded LIKWID data"

    for rank in results.likwid_ranks:
        regions = results.get_likwid_regions(rank)
        # Every profiled region should have become a LIKWID marker region.
        for name in ("main", "iteration", "busy"):
            assert (
                name in regions
            ), f"rank {rank}: {name!r} missing, got {sorted(regions)}"

        for tag, result in regions.items():
            assert result.tag == tag
            assert len(result.times) == len(result.call_counts) > 0
            assert result.events.shape == (len(result.event_names), len(result.times))
            assert result.metrics.shape == (
                len(result.metric_names),
                len(result.times),
            )
            assert result.event_names, f"{tag}: no event names"
            assert (result.call_counts > 0).all()

        # Call counts come from LIKWID's own bookkeeping rather than from the
        # hardware, so they are exact even where the counters are not.
        assert results.get_likwid_region("busy", rank=rank).call_counts[0] == 1
        assert results.get_likwid_region("iteration", rank=rank).call_counts[0] == 10

    results.print_likwid_summary()

    # Whether the counters hold real numbers is a property of the machine, not
    # of the profiler: a virtualized runner with an unreadable TSC, or one
    # where HyperThreading disables the PMCs, reports structurally valid zeros.
    # Report that rather than failing on it -- the plumbing is what this test
    # is here to check.
    total = sum(
        float(result.events.sum())
        for rank in results.likwid_ranks
        for result in results.get_likwid_regions(rank).values()
    )
    sources = {
        result.source
        for rank in results.likwid_ranks
        for result in results.get_likwid_regions(rank).values()
    }
    if total > 0:
        print(f"\nLIKWID counters are non-zero (source: {', '.join(sorted(sources))})")
    else:
        print(
            "\nWARNING: every hardware counter read zero. The marker plumbing "
            "works, but this host cannot actually count (virtualized CPU, "
            f"counters disabled by SMT, ...). Source: {', '.join(sorted(sources))}",
        )

    print(f"LIKWID data verified in {H5_PATH} for rank(s) {results.likwid_ranks}")


if __name__ == "__main__":
    test_pylikwid()
