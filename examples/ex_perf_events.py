"""Collect Linux CPU performance counters for profiled regions.

Install the optional Rust-backed counter integration first::

    pip install 'scope-profiler[perf-events]'
    python examples/ex_perf_events.py

The example remains usable without that extra: scope-profiler falls back to
its built-in ``perf_event_open`` implementation. Linux may still deny access
to unprivileged users; in that case ask the system administrator to adjust
``/proc/sys/kernel/perf_event_paranoid`` for the job environment, then run
this script normally (do not run the Python program itself with ``sudo``)::

    sudo sysctl -w kernel.perf_event_paranoid=1
    python examples/ex_perf_events.py
"""

import math

from scope_profiler import PerfEventError, ProfileManager


def compute(size: int) -> float:
    """A deliberately CPU-bound region with enough work to count."""
    with ProfileManager.profile_region("compute"):
        return sum(math.sin(index) * math.cos(index) for index in range(size))


def main() -> None:
    try:
        # Counts are aggregated across every completed call per (rank, region)
        # and saved in perf_events_example.h5 along with the normal timeline.
        with ProfileManager.session(
            file_path="perf_events_example.h5",
            perf_events=["cycles", "instructions", "cache-misses"],
            return_results=True,
            verbose=False,
        ) as run:
            for _ in range(3):
                compute(500_000)
    except PerfEventError as exc:
        print(f"Cannot collect perf events: {exc}")
        return

    results = run.results
    for rank, regions in results.get_perf_events().items():
        for name, totals in regions.items():
            print(f"rank {rank} / {name}: {totals.calls} calls")
            for event, value in totals.values.items():
                print(f"  {event:<16} {value:>16,}")


if __name__ == "__main__":
    main()
