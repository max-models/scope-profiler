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

import matplotlib.pyplot as plt
import numpy as np

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
    print(
        f"\nBaseline per call: {baseline_per_call:.0f} ns "
        f"({baseline_per_call / 1000:.2f} µs)\n"
    )
    header = (
        f"{'Region type':<22} {'Total/call (ns)':>16} "
        f"{'Overhead/call (ns)':>20} {'Relative':>10}"
    )
    print(header)
    print("-" * len(header))
    for name, total, overhead in zip(names, totals_ns, overheads_ns):
        label = name.replace("\n", " ")
        pct = overhead / baseline_per_call * 100
        print(f"{label:<22} {total:>16.0f} {overhead:>20.0f} {pct:>+9.1f}%")

    # ---- Bar chart (log scale so all bars are readable) ----
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974"]
    # Clamp negatives to 1 ns so log scale doesn't break
    plot_vals = [max(v, 1) for v in overheads_ns]
    bars = ax.bar(x, plot_vals, color=colors[: len(names)])

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Overhead per call (ns, log scale)", fontsize=12)
    ax.set_title(
        f"Profiling overhead by region type\n"
        f"(workload ≈ {baseline_per_call:.0f} ns/call, "
        f"{NUM_CALLS:,} calls, best of {NUM_REPEATS} repeats)",
        fontsize=12,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate each bar with its value
    for bar, val in zip(bars, overheads_ns):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.0f} ns",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()

    os.makedirs(args.output, exist_ok=True)
    outpath = os.path.join(args.output, "benchmark_overhead.png")
    fig.savefig(outpath, dpi=150)
    print(f"\nFigure saved to {outpath}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
