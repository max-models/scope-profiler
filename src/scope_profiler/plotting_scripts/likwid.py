"""LIKWID hardware-counter bar charts."""

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
    _unique_labels,
    _write_csv,
    _write_json,
)
from scope_profiler.results import ProfilingResults
from scope_profiler.summary import _name_selected


def available_likwid_metrics(
    profiling_data: ProfilingResults | Sequence[ProfilingResults],
) -> list[str]:
    """List the LIKWID metric/event names available for :func:`plot_likwid`.

    Union over every run, rank and region, in the order LIKWID reported them
    (derived metrics first, then raw events) -- so a caller who does not know
    what a run measured can list the valid ``metric`` values before plotting.
    """
    runs = _as_runs(profiling_data)
    names: list[str] = []
    seen: set[str] = set()
    for run in runs:
        for regions in run.get_likwid_regions().values():
            for result in regions.values():
                for name in (*result.metric_names, *result.event_labels):
                    if name not in seen:
                        seen.add(name)
                        names.append(name)
    return names


def _likwid_metric_value(result, metric: str) -> float:
    """Read one metric/event's value for a LIKWID region, averaged over threads."""
    for names, values in (
        (result.metric_names, result.metrics),
        (result.event_labels, result.events),
    ):
        if metric in names:
            index = names.index(metric)
            row = values[index]
            return float(np.mean(row)) if len(row) else float("nan")
    return float("nan")


def _likwid_bar_hover(
    run: ProfilingResults,
    rank: int,
    region_name: str,
    series_label: str,
    metric: str,
    value: float,
) -> str:
    """Hover text for one LIKWID bar.

    A LIKWID tag usually names a profiled region, in which case that rank's
    region summary is shown under the counter value. A tag with no timing
    region behind it (one marked in native code only, say) still gets its
    value.
    """
    title = f"{region_name} - {series_label}"
    extra = [(metric, f"{value:.6g}")]
    try:
        region = run.get_region(region_name)
    except (KeyError, ValueError):
        region = None
    if region is None or rank not in region:
        lines = [f"<b>{title}</b>"]
        lines.extend(f"{label}: {shown}" for label, shown in extra)
        return "<br>".join(lines)
    return _ps._hover_summary(region[rank], title=title, extra=extra)


def plot_likwid(
    profiling_data: ProfilingResults | Sequence[ProfilingResults],
    metric: str,
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
    labels: Sequence[str] | None = None,
    filepath: str | None = None,
    show: bool = False,
    verbose: bool = True,
    cmap: str = DEFAULT_CMAP,
    log_scale: bool = False,
    data_filepath: str | Path | None = None,
    data_format: str = "csv",
    backend: str = "matplotlib",
) -> None:
    """Plot one LIKWID derived metric or raw event, grouped by region and rank.

    LIKWID counters are otherwise only visible as text tables
    (:func:`~scope_profiler.summary.print_likwid_tables`); this gives the same
    numbers a bar chart, one group of bars per region and one bar per rank, so
    a metric like memory bandwidth or CPI can be compared across regions (or
    across files, for a single rank) the same way :func:`plot_durations`
    compares timings.

    Parameters
    ----------
    metric : str
        Name of a LIKWID derived metric (e.g. ``"CPI"``) or raw event/event
        label (e.g. ``"CAS_COUNT_RD"``). See :func:`available_likwid_metrics`
        for the names a run actually recorded.
    backend : str
        Backend to use for rendering: "matplotlib" (default) or "plotly".
    """
    Canvas = _ps._get_canvas()
    runs = _as_runs(profiling_data)
    if not runs:
        # Not this rank's job; rank 0 draws it.
        return

    if not any(run.has_likwid for run in runs):
        raise ValueError(
            "None of the selected files recorded LIKWID data. Runs must be "
            "started under likwid-perfctr/likwid-mpirun with use_likwid=True."
        )

    normalized_ranks = _normalize_ranks(ranks)

    # One "series" per (file, rank) so a single multi-rank file still yields a
    # readable grouped bar chart; multiple single-rank files compare across
    # files the way plot_durations does.
    series: list[tuple[str, dict[str, float]]] = []
    for run in runs:
        run_ranks = (
            [rank for rank in normalized_ranks if rank in run.likwid_ranks]
            if normalized_ranks is not None
            else run.likwid_ranks
        )
        for rank in run_ranks:
            regions = run.get_likwid_regions(rank)
            values: dict[str, float] = {
                tag: _likwid_metric_value(result, metric)
                for tag, result in regions.items()
            }
            if values:
                label = run.display_label if len(runs) > 1 else f"rank {rank}"
                if len(runs) > 1 and len(run_ranks) > 1:
                    label = f"{run.display_label} (rank {rank})"
                # The run and rank ride along so a bar can show the timing
                # summary of the region its LIKWID tag names.
                series.append((label, values, run, rank))

    if not series:
        raise ValueError(
            f"No LIKWID data found for metric {metric!r} with the requested "
            "ranks/files."
        )

    region_names = sorted(
        {
            tag
            for _, values, _, _ in series
            for tag in values
            if _name_selected(tag, include, exclude)
        }
    )
    if not region_names:
        raise ValueError("No regions matched the selected filters.")

    if labels is None:
        series_labels = _unique_labels([label for label, *_ in series])
    else:
        series_labels = list(labels)
        if len(series_labels) != len(series):
            raise ValueError("labels must match the number of (file, rank) series.")

    if verbose:
        print(f"Plotting LIKWID metric {metric!r} for: " + ", ".join(series_labels))

    num_series = len(series)
    colors = _get_cmap_colors(cmap, max(num_series, 1))
    fig_width = max(10, 0.85 * len(region_names) + 2)
    fig_height = max(4.5, 2.5 + 0.35 * num_series)
    width = min(0.8 / max(num_series, 1), 0.35)

    canvas = Canvas(figsize=(fig_width, fig_height))
    x_positions = np.arange(len(region_names))
    offset_start = -0.5 * width * (num_series - 1)

    data_rows = []
    hover_enabled = backend == "plotly"
    for idx, (label, (_, values, run, rank)) in enumerate(zip(series_labels, series)):
        bar_values = [values.get(name, float("nan")) for name in region_names]
        offsets = x_positions + offset_start + idx * width
        bar_hover = None
        if hover_enabled:
            bar_hover = [
                _likwid_bar_hover(run, rank, region_name, label, metric, value)
                for region_name, value in zip(region_names, bar_values)
            ]
        canvas.bar(
            offsets,
            bar_values,
            width=width,
            label=label if num_series > 1 else None,
            color=_to_hex(colors[idx]),
            edgecolor="black",
            alpha=0.8,
            hover=bar_hover,
        )
        if data_filepath:
            for region_name, value in zip(region_names, bar_values):
                data_rows.append([label, region_name, value])

    canvas.set_xticks(x_positions, labels=region_names)
    canvas.set_ylabel(metric)
    canvas.set_title(f"LIKWID {metric}")
    canvas.set_grid(True)
    if log_scale:
        canvas.set_yscale("log")
    if num_series > 1:
        canvas.set_legend()

    if data_filepath:
        if data_format == "json":
            bars = [
                {"series": series_label, "region": region, "value": value}
                for series_label, region, value in data_rows
            ]
            colors_map = {
                label: _to_hex(color) for label, color in zip(series_labels, colors)
            }
            _write_json(
                data_filepath, {"metric": metric, "bars": bars, "colors": colors_map}
            )
        else:
            _write_csv(data_filepath, ["series", "region", "value"], data_rows)

    _ps._render(canvas, filepath, show, backend)
