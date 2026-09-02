"""Compare compute intensity and cache pressure with perf-event plots.

Requires Linux perf permissions and the plotting extra::

    pip install 'scope-profiler[perf-events,pproc]'
    python examples/ex_perf_event_plots.py

The two regions are intentionally different: matrix multiplication does much
more arithmetic for each byte loaded, while the streaming update repeatedly
walks a large array. The IPC and cache-miss-per-instruction charts make that
difference visible without comparing raw counts from differently sized work.
"""

from pathlib import Path

import numpy as np

from scope_profiler import (
    PerfEventError,
    ProfileManager,
    ProfilingOptions,
    plot_perf_events,
)

OUTPUT_DIR = Path("figures")


def main() -> None:
    matrix_a = np.random.default_rng(1).random((384, 384))
    matrix_b = np.random.default_rng(2).random((384, 384))
    streaming_data = np.ones(12_000_000)
    options = ProfilingOptions(
        file_path=str(OUTPUT_DIR / "perf_event_plots.h5"),
        perf_events=["cycles", "instructions", "cache-misses"],
    )
    OUTPUT_DIR.mkdir(exist_ok=True)

    try:
        with ProfileManager.session(
            options=options, return_results=True, verbose=False
        ) as run:
            with ProfileManager.profile_region("compute_bound"):
                for _ in range(12):
                    matrix_a @ matrix_b
            with ProfileManager.profile_region("memory_stream"):
                for _ in range(12):
                    np.add(streaming_data, 1.0, out=streaming_data)
    except PerfEventError as exc:
        print(f"Cannot collect perf events: {exc}")
        return

    plot_perf_events(
        run.results,
        metric="ipc",
        include="compute_bound|memory_stream",
        filepath=str(OUTPUT_DIR / "perf_event_ipc.png"),
    )
    plot_perf_events(
        run.results,
        metric="cache-misses-per-ki",
        include="compute_bound|memory_stream",
        filepath=str(OUTPUT_DIR / "perf_event_cache_misses.png"),
    )
    print(
        f"Wrote plots to {OUTPUT_DIR}/perf_event_ipc.png and "
        f"{OUTPUT_DIR}/perf_event_cache_misses.png"
    )


if __name__ == "__main__":
    main()
