"""Roofline analysis from LIKWID FLOP-rate and memory-bandwidth metrics.

LIKWID reports its derived metrics per participating hardware thread.  A
region's point therefore sums those rates: a parallel region that sustains
``20 GFLOP/s`` on each of four cores belongs at ``80 GFLOP/s``, not at their
mean.  Arithmetic intensity is then ``GFLOP/s / GB/s`` and is numerically
identical to FLOP/byte.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from scope_profiler import plotting_scripts as _ps
from scope_profiler.plotting_scripts._utils import (
    DEFAULT_CMAP,
    _as_runs,
    _get_cmap_colors,
    _normalize_ranks,
    _to_hex,
    _write_csv,
    _write_json,
)
from scope_profiler.results import ProfilingResults
from scope_profiler.summary import _name_selected


def _metric_value(result, metric: str) -> float | None:
    """Return the sum of a named LIKWID metric across participating threads."""
    if metric not in result.metric_names:
        return None
    values = np.asarray(result.metrics[result.metric_names.index(metric)], dtype=float)
    value = float(np.nansum(values))
    return value if np.isfinite(value) and value > 0 else None


def _find_metric(result, requested: str | None, kind: str) -> str | None:
    if requested is not None:
        return requested if requested in result.metric_names else None
    for name in result.metric_names:
        normalized = name.lower().replace(" ", "")
        if kind == "flops" and "flop/s" in normalized:
            return name
        if kind == "bandwidth" and "bandwidth" in normalized and "/s" in normalized:
            return name
    return None


def _rate_to_giga(value: float, metric: str, kind: str) -> float | None:
    """Convert a LIKWID rate to GFLOP/s or GB/s from its displayed unit."""
    lower = metric.lower().replace(" ", "")
    if kind == "flops":
        if "gflop/s" in lower:
            factor = 1.0
        elif "mflop/s" in lower:
            factor = 1e-3
        elif "kflop/s" in lower:
            factor = 1e-6
        elif "flop/s" in lower:
            factor = 1e-9
        else:
            return None
    else:
        # LIKWID uses MBytes/s for its standard memory groups.  Accept the
        # common SI and IEC spellings so architecture-specific groups work too.
        match = re.search(r"([tgmk]?i?)bytes?/s", lower)
        if match is None:
            return None
        prefix = match.group(1)
        factors = {
            "t": 1e3,
            "g": 1.0,
            "m": 1e-3,
            "k": 1e-6,
            "ti": 1024.0,
            "gi": 2**30 / 1e9,
            "mi": 2**20 / 1e9,
            "ki": 2**10 / 1e9,
            "": 1e-9,
        }
        factor = factors.get(prefix)
        if factor is None:
            return None
    converted = value * factor
    return converted if np.isfinite(converted) and converted > 0 else None


def collect_roofline_points(
    profiling_data: ProfilingResults | Sequence[ProfilingResults],
    *,
    flops_metric: str | None = None,
    bandwidth_metric: str | None = None,
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
) -> list[dict]:
    """Derive one roofline point per selected LIKWID region and rank.

    ``flops_metric`` and ``bandwidth_metric`` select exact LIKWID derived
    metric names.  When omitted, the first names containing ``FLOP/s`` and
    ``bandwidth`` respectively are selected for each region.  Supplying exact
    names is recommended when a group offers both single- and double-precision
    FLOP rates.
    """
    points = []
    normalized_ranks = _normalize_ranks(ranks)
    for run in _as_runs(profiling_data):
        run_ranks = run.likwid_ranks if normalized_ranks is None else normalized_ranks
        for rank in run_ranks:
            if rank not in run.likwid_ranks:
                continue
            for region, result in run.get_likwid_regions(rank).items():
                if not _name_selected(region, include, exclude):
                    continue
                selected_flops = _find_metric(result, flops_metric, "flops")
                selected_bandwidth = _find_metric(result, bandwidth_metric, "bandwidth")
                if selected_flops is None or selected_bandwidth is None:
                    continue
                flops = _metric_value(result, selected_flops)
                bandwidth = _metric_value(result, selected_bandwidth)
                if flops is None or bandwidth is None:
                    continue
                performance = _rate_to_giga(flops, selected_flops, "flops")
                bandwidth_gbs = _rate_to_giga(
                    bandwidth, selected_bandwidth, "bandwidth"
                )
                if performance is None or bandwidth_gbs is None:
                    continue
                points.append(
                    {
                        "file": run.display_label,
                        "region": region,
                        "rank": int(rank),
                        "performance_gflops": performance,
                        "bandwidth_gbs": bandwidth_gbs,
                        "arithmetic_intensity_flops_per_byte": performance
                        / bandwidth_gbs,
                        "runtime_seconds": (
                            float(np.nanmax(result.times))
                            if np.asarray(result.times).size
                            else None
                        ),
                        "flops_metric": selected_flops,
                        "bandwidth_metric": selected_bandwidth,
                    },
                )
    return points


def _roofline(
    points: list[dict], peak_flops: float | None, peak_bandwidth: float | None
):
    if peak_flops is not None and peak_flops <= 0:
        raise ValueError("peak_flops must be positive GFLOP/s")
    if peak_bandwidth is not None and peak_bandwidth <= 0:
        raise ValueError("peak_bandwidth must be positive GB/s")
    empirical = peak_flops is None or peak_bandwidth is None
    ceiling_flops = peak_flops or max(point["performance_gflops"] for point in points)
    ceiling_bandwidth = peak_bandwidth or max(
        point["bandwidth_gbs"] for point in points
    )
    minimum = min(point["arithmetic_intensity_flops_per_byte"] for point in points)
    maximum = max(point["arithmetic_intensity_flops_per_byte"] for point in points)
    x = np.logspace(np.log10(minimum / 2), np.log10(maximum * 2), 160)
    return {
        "peak_flops_gflops": float(ceiling_flops),
        "peak_bandwidth_gbs": float(ceiling_bandwidth),
        "empirical_ceilings": empirical,
        "roofline": [
            {
                "arithmetic_intensity_flops_per_byte": float(value),
                "performance_gflops": float(
                    min(ceiling_flops, ceiling_bandwidth * value)
                ),
            }
            for value in x
        ],
    }


def plot_roofline(
    profiling_data: ProfilingResults | Sequence[ProfilingResults],
    *,
    flops_metric: str | None = None,
    bandwidth_metric: str | None = None,
    peak_flops: float | None = None,
    peak_bandwidth: float | None = None,
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
    filepath: str | None = None,
    show: bool = False,
    verbose: bool = True,
    cmap: str = DEFAULT_CMAP,
    data_filepath: str | Path | None = None,
    data_format: str = "csv",
    backend: str = "matplotlib",
) -> dict:
    """Plot attained performance against arithmetic intensity for LIKWID regions.

    ``peak_flops`` (GFLOP/s) and ``peak_bandwidth`` (GB/s) should be measured
    hardware ceilings for a true roofline.  If either is absent, the largest
    selected observation supplies that ceiling and the plot labels it
    *empirical*; it is useful for comparing regions but must not be read as a
    machine limit.
    """
    Canvas = _ps._get_canvas()
    points = collect_roofline_points(
        profiling_data,
        flops_metric=flops_metric,
        bandwidth_metric=bandwidth_metric,
        ranks=ranks,
        include=include,
        exclude=exclude,
    )
    if not points:
        requested = []
        if flops_metric:
            requested.append(f"FLOP metric {flops_metric!r}")
        if bandwidth_metric:
            requested.append(f"bandwidth metric {bandwidth_metric!r}")
        detail = ", ".join(requested) or "auto-detected FLOP and bandwidth metrics"
        raise ValueError(f"No LIKWID roofline points found for {detail}.")

    roof = _roofline(points, peak_flops, peak_bandwidth)
    labels = list(dict.fromkeys(point["file"] for point in points))
    colors = _get_cmap_colors(cmap, len(labels))
    color_map = {label: _to_hex(color) for label, color in zip(labels, colors)}
    canvas = Canvas(figsize=(10, 6.5))
    for label in labels:
        selected = [point for point in points if point["file"] == label]
        canvas.scatter(
            [point["arithmetic_intensity_flops_per_byte"] for point in selected],
            [point["performance_gflops"] for point in selected],
            label=label if len(labels) > 1 else None,
            color=color_map[label],
            hover=(
                [
                    "<br>".join(
                        (
                            f"<b>{point['region']}</b> (rank {point['rank']})",
                            f"intensity: {point['arithmetic_intensity_flops_per_byte']:.6g} FLOP/byte",
                            f"performance: {point['performance_gflops']:.6g} GFLOP/s",
                            f"bandwidth: {point['bandwidth_gbs']:.6g} GB/s",
                        ),
                    )
                    for point in selected
                ]
                if backend == "plotly"
                else None
            ),
        )
    canvas.add_line(
        [row["arithmetic_intensity_flops_per_byte"] for row in roof["roofline"]],
        [row["performance_gflops"] for row in roof["roofline"]],
        color="black",
        linewidth=1.6,
        linestyle="--",
        label="empirical roof" if roof["empirical_ceilings"] else "roofline",
    )
    canvas.set_xscale("log")
    canvas.set_yscale("log")
    canvas.set_xlabel("Arithmetic intensity [FLOP/byte]")
    canvas.set_ylabel("Attained performance [GFLOP/s]")
    canvas.set_title(
        "Roofline analysis"
        + (" (empirical ceilings)" if roof["empirical_ceilings"] else "")
    )
    canvas.set_grid(True)
    canvas.set_legend()

    payload = {"points": points, "colors": color_map, **roof}
    if data_filepath:
        if data_format == "json":
            _write_json(data_filepath, payload, plot="roofline")
        else:
            _write_csv(
                data_filepath,
                list(points[0]),
                [[point[key] for key in points[0]] for point in points],
            )
    if verbose:
        print(f"Plotting roofline for {len(points)} LIKWID region/rank point(s)")
    _ps._render(canvas, filepath, show, backend)
    return payload
