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

    # Preallocate enough slots for every call across every repeat. Buffers grow
    # on demand, so this is not required for correctness -- it just keeps the
    # reallocation out of the timed loop, so what is measured is the
    # steady-state per-call cost.
    buffer_limit = NUM_CALLS * NUM_REPEATS + 1

    # ---- Baseline (bare function call, no profiling) ----
    baseline_ns = time_calls(workload, NUM_CALLS, NUM_REPEATS)
    baseline_per_call = baseline_ns / NUM_CALLS

    # ---- Profiling configurations to benchmark ----
    # `flush_to_disk` is not varied here: recording is identical either way and
    # the data is written once at finalize(), so it has no per-call cost.
    configs = [
        (
            "Disabled",
            dict(profiling_activated=False),
        ),
        (
            "NCallsOnly",
            dict(profiling_activated=True, time_trace=False),
        ),
        (
            "TimeOnly",
            dict(profiling_activated=True, time_trace=True),
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

    # ---- Bar chart (log scale: the modes span three orders of magnitude) ----
    x = np.arange(len(names))
    overheads_us = [v / 1e3 for v in overheads_ns]
    baseline_us = baseline_per_call / 1e3

    # The bars are one series -- the x labels carry the identity -- so they all
    # share a single hue rather than being coloured by position.
    BAR_COLOR = "#2a78d6"
    TEXT_COLOR = "#52514e"

    # A log axis cannot show zero or negative values. Overheads are positive in
    # practice, but "Disabled" is a bare function call measured against itself,
    # so noise can put it at or below zero; such bars are pinned to the axis
    # floor and labelled for what they are.
    # The floor sits exactly one decade below the smallest bar. Rounding it to
    # a decade boundary instead would leave a bar that happens to fall just
    # above one (say 0.107 against a floor of 0.1) drawn as an invisible sliver.
    positive = [v for v in overheads_us if v > 0]
    floor_us = min(positive) / 10 if positive else 0.01
    plot_vals = [v if v > 0 else floor_us for v in overheads_us]

    canvas = Canvas(nrows=1, ncols=1)
    subplot = canvas.add_subplot(
        title=(
            "Profiling overhead by region type<br>"
            f"(workload ≈ {baseline_us:.3f} µs/call, "
            f"{NUM_CALLS:,} calls, best of {NUM_REPEATS} repeats)"
        ),
        ylabel="Overhead per call (µs, log scale)",
        grid=True,
    )
    for xi, value in zip(x, plot_vals):
        subplot.bar([xi], [value], color=BAR_COLOR)
    subplot.set_yscale("log")
    # Plotly renders tick labels as HTML, so multi-line names need <br>.
    subplot.set_xticks(x.tolist(), [name.replace("\n", "<br>") for name in names])

    fig = canvas.plot(backend="plotly")
    fig.update_layout(width=900, height=500)

    # Everything below is placed in plotly directly. On a log axis plotly takes
    # y positions in log10 units, so the annotations and the reference line are
    # converted explicitly rather than left to guess.
    def log10(value):
        return float(np.log10(value))

    # Pin the axis: start a decade below the smallest bar, and leave ~0.4 of a
    # decade of headroom above the tallest so its label is not clipped.
    top_us = max(max(plot_vals), baseline_us)
    fig.update_yaxes(
        type="log",
        range=[log10(floor_us), log10(top_us) + 0.4],
        # One labelled tick per decade; plotly otherwise labels minor ticks as a
        # bare "5" between 10 and 100, which reads as 5.
        dtick=1,
        tickformat=".3~g",
        minor=dict(showgrid=True),
    )

    # Direct labels: only four bars, so every one is labelled.
    for xi, plot_val, val in zip(x, plot_vals, overheads_us):
        fig.add_annotation(
            x=float(xi),
            y=log10(plot_val),
            text=f"{val:.3f} µs" if val > 0 else "≈ 0 µs",
            showarrow=False,
            yshift=10,
            font=dict(color=TEXT_COLOR, size=12),
        )

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
