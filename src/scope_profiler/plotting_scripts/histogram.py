"""Per-region duration-distribution histograms."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from scope_profiler import plotting_scripts as _ps
from scope_profiler.plotting_scripts._utils import (
    DEFAULT_CMAP,
    _as_runs,
    _hover_region,
    _normalize_ranks,
    _panel_gridspec,
    _region_color_map,
    _to_hex,
    _unique_labels,
    _write_csv,
    _write_json,
)
from scope_profiler.plotting_scripts.statistics import _region_duration_values
from scope_profiler.results import ProfilingResults


def plot_duration_histogram(
    profiling_data: ProfilingResults | Sequence[ProfilingResults],
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
    bins: int = 30,
    filepath: str | None = None,
    show: bool = False,
    verbose: bool = True,
    cmap: str = DEFAULT_CMAP,
    log_scale: bool = False,
    data_filepath: str | Path | None = None,
    data_format: str = "csv",
    backend: str = "matplotlib",
) -> None:
    """Plot each region's call-duration distribution as a frequency line.

    One panel per file, one line per region, giving the count of calls
    falling in each duration bin -- so a region whose calls are mostly fast
    with an occasional slow outlier shows up as a peak with a long tail,
    something the mean/min/max in :func:`plot_durations` cannot distinguish
    from a uniformly slower region.

    Parameters
    ----------
    bins : int
        Number of histogram bins spanning each file's observed duration
        range (default: 30). All regions in a panel share the same bin edges,
        so the curves are directly comparable.
    backend : str
        Backend to use for rendering: "matplotlib" (default) or "plotly".
    """
    Canvas = _ps._get_canvas()
    runs = _as_runs(profiling_data)
    if not runs:
        # Not this rank's job; rank 0 draws it.
        return

    normalized_ranks = _normalize_ranks(ranks)

    reader_regions = []
    all_region_names: set[str] = set()
    for run in runs:
        regions = run.get_regions(include=include, exclude=exclude)
        if not regions:
            raise ValueError("No regions matched the selected filters.")
        all_region_names.update(region.name for region in regions)
        reader_regions.append((run, regions))

    color_map = _region_color_map(all_region_names, cmap=cmap)

    prepared = []
    for run, regions in reader_regions:
        series = []
        for region in regions:
            values = _region_duration_values(region, normalized_ranks)
            if values.size:
                series.append((region.name, values))
        if series:
            prepared.append((run, series))

    if not prepared:
        raise ValueError("No calls recorded for the requested ranks.")

    labels = _unique_labels([run.display_label for run, _ in prepared])

    if verbose:
        print("Plotting duration histograms for files: " + ", ".join(labels))

    single_panel = len(prepared) == 1
    fig_width, fig_height = 12.0, 1.0 + 4.0 * len(prepared)
    canvas = Canvas(
        nrows=len(prepared),
        ncols=1,
        figsize=(fig_width, fig_height),
        gridspec_kw=_panel_gridspec(fig_width, fig_height, 10, not single_panel),
    )

    data_rows = []
    hover_enabled = backend == "plotly"
    for idx, (run, series) in enumerate(prepared):
        row = None if single_panel else idx
        col = None if single_panel else 0

        all_values = np.concatenate([values for _, values in series])
        edges = np.histogram_bin_edges(all_values, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])

        for region_name, values in series:
            counts, _ = np.histogram(values, bins=edges)
            color = _to_hex(color_map[region_name])
            line_hover = None
            if hover_enabled:
                region, title = _hover_region(
                    run.get_region(region_name), normalized_ranks
                )
                line_hover = [
                    _ps._hover_summary(
                        region,
                        title=title,
                        extra=[
                            ("bin", f"{low:.6g} - {high:.6g} s"),
                            ("calls in bin", int(count)),
                        ],
                    )
                    for low, high, count in zip(edges[:-1], edges[1:], counts)
                ]
            canvas.add_line(
                centers,
                counts,
                row=row,
                col=col,
                linewidth=1.6,
                color=color,
                label=region_name,
                hover=line_hover,
            )
            if data_filepath:
                label = labels[idx]
                for center, low, high, count in zip(
                    centers, edges[:-1], edges[1:], counts
                ):
                    data_rows.append(
                        [
                            label,
                            region_name,
                            float(low),
                            float(high),
                            float(center),
                            int(count),
                        ]
                    )

        canvas.set_xlabel("Duration per call (seconds)", row=row, col=col)
        canvas.set_ylabel("Number of calls", row=row, col=col)
        canvas.set_title(
            "Call duration distribution" if single_panel else run.display_label,
            row=row,
            col=col,
        )
        canvas.set_grid(True, row=row, col=col)
        canvas.set_legend(row=row, col=col)
        if log_scale:
            canvas.set_yscale("log", row=row, col=col)

    if not single_panel:
        canvas.suptitle("Call duration distribution")

    if data_filepath:
        header = [
            "file",
            "region",
            "bin_low_seconds",
            "bin_high_seconds",
            "bin_center_seconds",
            "count",
        ]
        if data_format == "json":
            bins_payload = [dict(zip(header, record)) for record in data_rows]
            colors_map = {
                name: _to_hex(color) for name, color in sorted(color_map.items())
            }
            _write_json(data_filepath, {"bins": bins_payload, "colors": colors_map})
        else:
            _write_csv(data_filepath, header, data_rows)

    _ps._render(canvas, filepath, show, backend)
