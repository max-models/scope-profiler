"""Bar charts for built-in Linux perf-event counter totals and ratios."""

from collections.abc import Sequence

import numpy as np

from scope_profiler import plotting_scripts as _ps
from scope_profiler.plotting_scripts._utils import (
    DEFAULT_CMAP,
    _as_runs,
    _get_cmap_colors,
    _normalize_ranks,
    _to_hex,
    _unique_labels,
)
from scope_profiler.results import ProfilingResults
from scope_profiler.summary import _name_selected

_DERIVED = {
    "ipc": ("instructions", "cycles", 1.0, "Instructions per cycle"),
    "cache-misses-per-ki": (
        "cache-misses",
        "instructions",
        1_000.0,
        "Cache misses / 1k instructions",
    ),
}


def available_perf_event_metrics(
    profiling_data: ProfilingResults | Sequence[ProfilingResults],
) -> list[str]:
    """List available raw events and valid derived perf-event ratios."""
    raw = {
        event
        for run in _as_runs(profiling_data)
        for regions in run.get_perf_events().values()
        for totals in regions.values()
        for event in totals.values
    }
    derived = [
        name
        for name, (numerator, denominator, _, _) in _DERIVED.items()
        if numerator in raw and denominator in raw
    ]
    return [*derived, *sorted(raw)]


def _metric_value(totals, metric: str) -> float:
    if metric in _DERIVED:
        numerator, denominator, scale, _ = _DERIVED[metric]
        base = totals.values.get(denominator, 0)
        return (
            float(totals.values.get(numerator, 0) * scale / base)
            if base
            else float("nan")
        )
    return float(totals.values.get(metric, float("nan")))


def plot_perf_events(
    profiling_data: ProfilingResults | Sequence[ProfilingResults],
    metric: str = "ipc",
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
    labels: Sequence[str] | None = None,
    filepath: str | None = None,
    show: bool = False,
    verbose: bool = True,
    cmap: str = DEFAULT_CMAP,
    backend: str = "matplotlib",
) -> None:
    """Plot one raw perf event or derived ratio for every region and rank.

    ``metric`` may be an event name recorded in the run, ``"ipc"``
    (instructions/cycle), or ``"cache-misses-per-ki"``. Derived metrics are
    calculated from the region's aggregated counters, not averages of call
    ratios, which preserves their proper weighting.
    """
    runs = _as_runs(profiling_data)
    if not runs:
        return
    if not any(run.has_perf_events for run in runs):
        raise ValueError("None of the selected files recorded perf-event data.")
    available = available_perf_event_metrics(runs)
    if metric not in available:
        raise ValueError(f"Metric {metric!r} is unavailable; choose from {available}")

    selected_ranks = _normalize_ranks(ranks)
    series = []
    for run in runs:
        for rank, regions in sorted(run.get_perf_events().items()):
            if selected_ranks is not None and rank not in selected_ranks:
                continue
            values = {
                name: _metric_value(totals, metric)
                for name, totals in regions.items()
                if _name_selected(name, include, exclude)
            }
            if values:
                label = (
                    run.display_label
                    if len(runs) == 1
                    else f"{run.display_label} (rank {rank})"
                )
                if len(runs) == 1:
                    label = f"rank {rank}"
                series.append((label, values))
    if not series:
        raise ValueError("No perf-event regions matched the requested filters.")

    region_names = sorted({name for _, values in series for name in values})
    series_labels = (
        _unique_labels([label for label, _ in series])
        if labels is None
        else list(labels)
    )
    if len(series_labels) != len(series):
        raise ValueError("labels must match the number of (file, rank) series.")
    ylabel = _DERIVED[metric][3] if metric in _DERIVED else metric
    if verbose:
        print(f"Plotting perf-event metric {metric!r} for: " + ", ".join(series_labels))

    Canvas = _ps._get_canvas()
    width = min(0.8 / len(series), 0.35)
    positions = np.arange(len(region_names))
    canvas = Canvas(figsize=(max(8, len(region_names) * 0.85 + 2), 4.8))
    colors = _get_cmap_colors(cmap, len(series))
    for index, (label, values) in enumerate(
        zip(series_labels, (item[1] for item in series))
    ):
        offsets = positions + (index - (len(series) - 1) / 2) * width
        canvas.bar(
            offsets,
            [values.get(name, float("nan")) for name in region_names],
            width=width,
            label=label if len(series) > 1 else None,
            color=_to_hex(colors[index]),
            edgecolor="black",
        )
    canvas.set_xticks(positions, labels=region_names)
    canvas.set_ylabel(ylabel)
    canvas.set_title(f"Perf events: {ylabel}")
    canvas.set_grid(True)
    if len(series) > 1:
        canvas.set_legend()
    _ps._render(canvas, filepath, show, backend)
