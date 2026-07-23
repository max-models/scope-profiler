#!/usr/bin/env python3
"""
Benchmark overhead of each scope-profiler region type
=====================================================

Measures the per-call overhead introduced by each profiling mode
relative to a bare function call, and produces a bar chart.

Run::

    python examples/benchmark_overhead.py          # save figure only
    python examples/benchmark_overhead.py --show   # also display interactively
"""

import argparse
import os
import time

import numpy as np
from maxplotlib import Canvas

from scope_profiler import ProfileManager

NUM_CALLS = 10_000
NUM_REPEATS = 7


def workload():
    """Minimal workload used as the baseline."""
    s = 0
    for i in range(50):
        s += i
    return s


def time_calls(func, num_calls, num_repeats):
    """Return the best (minimum) wall-clock time in ns over *num_repeats* trials."""
    best = float("inf")
    for _ in range(num_repeats):
        t0 = time.perf_counter_ns()
        for _ in range(num_calls):
            func()
        t1 = time.perf_counter_ns()
        best = min(best, t1 - t0)
    return best


def main():
    parser = argparse.ArgumentParser(description="Benchmark scope-profiler overhead.")
    parser.add_argument(
        "--show", action="store_true", help="Display the plot interactively"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="figures",
        help="Directory to save the figure (default: figures/)",
    )
    args = parser.parse_args()

    # Ensure the buffer can hold all calls across all repeats without
    # overflowing the no-flush variants.
    buffer_limit = NUM_CALLS * NUM_REPEATS + 1

    # ---- Baseline (bare function call, no profiling) ----
    baseline_ns = time_calls(workload, NUM_CALLS, NUM_REPEATS)
    baseline_per_call = baseline_ns / NUM_CALLS

    # ---- Profiling configurations to benchmark ----
    configs = [
        (
            "Disabled",
            dict(profiling_activated=False),
        ),
        (
            "NCallsOnly",
            dict(profiling_activated=True, time_trace=False, flush_to_disk=False),
        ),
        (
            "TimeOnly\n(no flush)",
            dict(profiling_activated=True, time_trace=True, flush_to_disk=False),
        ),
        (
            "TimeOnly\n(flush)",
            dict(profiling_activated=True, time_trace=True, flush_to_disk=True),
        ),
        (
            "LineProfiler",
            dict(profiling_activated=True, use_line_profiler=True),
        ),
    ]

    names = []
    overheads_ns = []
    totals_ns = []

    for name, kwargs in configs:
        ProfileManager.setup(buffer_limit=buffer_limit, **kwargs)

        # Define a fresh function each iteration so decoration is isolated.
        def _work():
            s = 0
            for i in range(50):
                s += i
            return s

        profiled = ProfileManager.profile("bench")(_work)

        total_ns = time_calls(profiled, NUM_CALLS, NUM_REPEATS)
        per_call = total_ns / NUM_CALLS
        overhead = per_call - baseline_per_call

        names.append(name)
        overheads_ns.append(overhead)
        totals_ns.append(per_call)

        ProfileManager.finalize(verbose=False)

    # ---- Print results table ----
    print(f"\nBaseline per call: {baseline_per_call / 1e3:.3f} µs\n")
    header = (
        f"{'Region type':<22} {'Total/call (µs)':>16} "
        f"{'Overhead/call (µs)':>20} {'Relative':>10}"
    )
    print(header)
    print("-" * len(header))
    for name, total, overhead in zip(names, totals_ns, overheads_ns):
        label = name.replace("\n", " ")
        pct = overhead / baseline_per_call * 100
        print(f"{label:<22} {total / 1e3:>16.3f} {overhead / 1e3:>20.3f} {pct:>+9.1f}%")

    # ---- Bar chart (log scale so all bars are readable) ----
    x = np.arange(len(names))
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974"]
    # Convert to µs; clamp negatives to 0.001 µs so log scale doesn't break
    overheads_us = [v / 1e3 for v in overheads_ns]
    plot_vals = [max(v, 0.001) for v in overheads_us]

    canvas = Canvas(nrows=1, ncols=1)
    subplot = canvas.add_subplot(
        title=(
            "Profiling overhead by region type<br>"
            f"(workload ≈ {baseline_per_call / 1e3:.3f} µs/call, "
            f"{NUM_CALLS:,} calls, best of {NUM_REPEATS} repeats)"
        ),
        ylabel="Overhead per call (µs, log scale)",
        grid=True,
    )
    for xi, value, color in zip(x, plot_vals, colors):
        subplot.bar([xi], [value], color=color)
    subplot.set_yscale("log")
    # Plotly renders tick labels as HTML, so multi-line names need <br>.
    subplot.set_xticks(x.tolist(), [name.replace("\n", "<br>") for name in names])

    # Annotate each bar with its value
    for xi, plot_val, val in zip(x, plot_vals, overheads_us):
        subplot.text(xi, plot_val, f"{val:.3f} µs", ha="center", va="bottom")

    fig = canvas.plot(backend="plotly")
    fig.update_layout(width=900, height=500)

    os.makedirs(args.output, exist_ok=True)
    outpath = os.path.join(args.output, "benchmark_overhead.png")
    try:
        fig.write_image(outpath)
    except Exception:
        outpath = os.path.join(args.output, "benchmark_overhead.html")
        fig.write_html(outpath)
    print(f"\nFigure saved to {outpath}")

    if args.show:
        fig.show()


if __name__ == "__main__":
    main()
