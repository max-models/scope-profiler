#!/usr/bin/env python3
"""
Benchmark the HDF5 write, read and summary paths
================================================

The counterpart to ``benchmark_overhead.py``: that one measures what
instrumentation costs *during* a run, this one measures what a run costs
*afterwards* -- writing the merged file, reading it back, and producing a
summary from it.

Four stages are timed, on a synthetic run of ``ranks x regions x events``
nested regions:

``finalize``
    Reconstructing one rank's nesting, i.e. what
    :func:`~scope_profiler.call_stack.exclusive_totals_ns` adds to
    ``ProfileManager.finalize()``. Every rank pays this for its own regions,
    in parallel, so the per-rank number is the one that matters.
``write``
    Appending every rank's payload to the merged file. Reported per rank as
    well as in total, so ``--scaling`` can show whether the cost of adding a
    rank depends on how many ranks came before it.
``read``
    :func:`~scope_profiler.h5reader.read_h5`, which must stay proportional to
    the number of events rather than to the number of (rank, region) rows.
``summary``
    ``results.summary()``, with and without the exclusive totals the run
    stored for itself. The two differ by a full call-stack reconstruction,
    which is what storing them at write time buys back.

Run::

    python examples/benchmark_io.py                        # default size
    python examples/benchmark_io.py --ranks 256            # bigger job
    python examples/benchmark_io.py --scaling 8,32,128     # per-rank cost vs ranks
    python examples/benchmark_io.py --json                 # machine-readable

The JSON form exists so the numbers can be diffed between two revisions the
same way ``scope-profiler diff`` compares two profiles -- a regression here is
a number that moved, not a table someone has to read.
"""

import argparse
import json
import os
import tempfile
import time

import numpy as np

from scope_profiler.call_stack import exclusive_totals_ns
from scope_profiler.h5reader import read_h5
from scope_profiler.h5writer import ProfilingWriter
from scope_profiler.profile_manager import RankPayload

DEFAULT_RANKS = 64
DEFAULT_REGIONS = 20
DEFAULT_EVENTS = 2_000


def build_regions(rank: int, regions: int, events: int) -> dict:
    """One rank's timing arrays: an outer region with the rest nested inside.

    Nesting is not decoration. Exclusive time is defined by containment, so a
    flat set of regions would make the call-stack reconstruction -- the most
    expensive thing being measured here -- unrepresentatively cheap.

    Parameters
    ----------
    rank : int
        Offsets this rank's timestamps, so no two ranks are byte-identical.
    regions, events : int
        Regions per rank, and calls per region.

    Returns
    -------
    dict
        Region name -> ``(start_times, end_times)`` int64 nanosecond arrays,
        the shape :meth:`ProfileManager._snapshot_regions` produces.
    """
    outer_start = np.arange(events, dtype=np.int64) * 100_000 + rank
    built = {"outer": (outer_start, outer_start + 90_000)}
    for index in range(regions - 1):
        start = outer_start + 1_000 * (index + 1)
        built[f"inner_{index}"] = (start, start + 800)
    return built


def build_payload(rank: int, regions: int, events: int, with_totals: bool):
    """A ``RankPayload``, optionally carrying the exclusive totals of its run."""
    built = build_regions(rank, regions, events)
    return RankPayload(
        regions=built,
        likwid={},
        likwid_environment={},
        exclusive_totals=exclusive_totals_ns(built) if with_totals else None,
    )


def timed(func, *args, **kwargs):
    """Return ``(elapsed_seconds, result)`` for one call of ``func``."""
    started = time.perf_counter()
    result = func(*args, **kwargs)
    return time.perf_counter() - started, result


def write_file(path, ranks: int, regions: int, events: int, with_totals: bool) -> float:
    """Write a whole run and return the seconds spent writing it.

    Only the ``write_rank`` calls and the closing publish are counted.
    Building each rank's payload is left out because it is the caller's work,
    not the writer's, and it is O(events) -- large enough to hide the very
    thing this measures. The payloads are still built one rank at a time, so
    the whole job is never in memory at once.
    """
    elapsed = 0.0
    writer = ProfilingWriter(path, {"mpi_size": ranks, "label": "benchmark"})
    try:
        for rank in range(ranks):
            payload = build_payload(rank, regions, events, with_totals)
            seconds, _ = timed(writer.write_rank, rank, payload)
            elapsed += seconds
            del payload
    except Exception:
        writer.close(commit=False)
        raise
    close_seconds, _ = timed(writer.close)
    return elapsed + close_seconds


def measure(ranks: int, regions: int, events: int, directory: str) -> dict:
    """Time every stage once, for one run size."""
    total_events = ranks * regions * events

    # What one rank adds to its own finalize(). Timed on its own payload, not
    # on the whole job's, because ranks do this concurrently.
    one_rank = build_regions(0, regions, events)
    finalize_seconds, _ = timed(exclusive_totals_ns, one_rank)

    stored_path = os.path.join(directory, "with_totals.h5")
    plain_path = os.path.join(directory, "without_totals.h5")
    write_seconds = write_file(stored_path, ranks, regions, events, True)
    write_file(plain_path, ranks, regions, events, False)

    read_seconds, results = timed(read_h5, stored_path)
    summary_seconds, rows = timed(results.summary)
    # A second call must hit the same values without recomputing anything.
    resummary_seconds, _ = timed(results.summary)

    plain_results = read_h5(plain_path)
    rebuilt_seconds, plain_rows = timed(plain_results.summary)

    # The two paths are only worth comparing if they agree. They are computed
    # differently -- one sums what the writer stored, the other reconstructs
    # the nesting from the events -- so a mismatch is a bug, not noise.
    stored_exclusive = {row["name"]: row["exclusive_duration"] for row in rows}
    rebuilt_exclusive = {row["name"]: row["exclusive_duration"] for row in plain_rows}
    if stored_exclusive != rebuilt_exclusive:
        raise AssertionError(
            "stored and reconstructed exclusive time disagree: "
            f"{stored_exclusive} != {rebuilt_exclusive}"
        )

    return {
        "ranks": ranks,
        "regions_per_rank": regions,
        "events_per_region": events,
        "total_events": total_events,
        "rows": ranks * regions,
        "file_bytes": os.path.getsize(stored_path),
        "finalize_seconds_per_rank": finalize_seconds,
        "write_seconds": write_seconds,
        "write_seconds_per_rank": write_seconds / ranks,
        "read_seconds": read_seconds,
        "read_events_per_second": total_events / read_seconds,
        "summary_seconds": summary_seconds,
        "summary_seconds_repeat": resummary_seconds,
        "summary_seconds_reconstructed": rebuilt_seconds,
        "summary_speedup_from_stored_totals": rebuilt_seconds / summary_seconds,
    }


def print_report(result: dict) -> None:
    """Print one run size as a labelled table."""
    print(
        f"\n{result['ranks']} ranks x {result['regions_per_rank']} regions "
        f"x {result['events_per_region']} events "
        f"= {result['total_events']:,} events, "
        f"{result['rows']:,} index rows, "
        f"{result['file_bytes'] / 1e6:.1f} MB"
    )
    print("-" * 72)
    rows = [
        (
            "finalize (per rank, parallel)",
            result["finalize_seconds_per_rank"],
            "reconstructs this rank's nesting once",
        ),
        (
            "write (whole run)",
            result["write_seconds"],
            f"{result['write_seconds_per_rank'] * 1e3:.2f} ms per rank",
        ),
        (
            "read_h5",
            result["read_seconds"],
            f"{result['read_events_per_second'] / 1e6:.1f}M events/s",
        ),
        (
            "summary (stored totals)",
            result["summary_seconds"],
            f"repeat {result['summary_seconds_repeat'] * 1e3:.0f} ms",
        ),
        (
            "summary (reconstructed)",
            result["summary_seconds_reconstructed"],
            f"{result['summary_speedup_from_stored_totals']:.1f}x slower",
        ),
    ]
    for label, seconds, note in rows:
        print(f"  {label:<32} {seconds:>8.3f} s   {note}")


def print_scaling(results: list) -> None:
    """Print the per-rank costs across rank counts.

    Both columns should stay flat, and both are guards rather than
    demonstrations: writing a rank costs the same whether it is the first or
    the two-thousandth (measured up to 2048 ranks, before and after
    :class:`~scope_profiler.h5writer.ColumnarIndex` removed the per-rank index
    re-read -- that read is superlinear in principle but stays small at these
    sizes). A rising write column would mean per-rank work that grows with the
    job; a rising read column would mean the reader is touching the file per
    index row rather than per column.
    """
    print(f"\n{'ranks':>8} {'write ms/rank':>15} {'read ms/Mevent':>16} {'MB':>8}")
    print("-" * 50)
    for result in results:
        per_million = result["read_seconds"] / (result["total_events"] / 1e6) * 1e3
        print(
            f"{result['ranks']:>8} "
            f"{result['write_seconds_per_rank'] * 1e3:>15.2f} "
            f"{per_million:>16.2f} "
            f"{result['file_bytes'] / 1e6:>8.1f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark scope-profiler's HDF5 write, read and summary paths."
    )
    parser.add_argument(
        "--ranks", type=int, default=DEFAULT_RANKS, help="ranks to write"
    )
    parser.add_argument(
        "--regions", type=int, default=DEFAULT_REGIONS, help="regions per rank"
    )
    parser.add_argument(
        "--events", type=int, default=DEFAULT_EVENTS, help="calls per region"
    )
    parser.add_argument(
        "--scaling",
        type=str,
        default=None,
        help="comma-separated rank counts to sweep, e.g. 8,32,128",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="directory for the generated files (default: a temporary one)",
    )
    parser.add_argument(
        "--json", action="store_true", help="print results as JSON and nothing else"
    )
    args = parser.parse_args()

    rank_counts = (
        [int(value) for value in args.scaling.split(",")]
        if args.scaling
        else [args.ranks]
    )

    with tempfile.TemporaryDirectory() as temporary:
        directory = args.output or temporary
        os.makedirs(directory, exist_ok=True)
        results = [
            measure(ranks, args.regions, args.events, directory)
            for ranks in rank_counts
        ]

    if args.json:
        print(json.dumps(results, indent=2))
        return

    for result in results:
        print_report(result)
    if len(results) > 1:
        print_scaling(results)


if __name__ == "__main__":
    main()
