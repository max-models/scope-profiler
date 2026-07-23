"""Plotting utilities for visualizing profiling data using maxplotlib.

This module provides plotting functions that can export to both matplotlib and plotly
backends using maxplotlib as the unified interface.
"""

import csv
import json
import os
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from scope_profiler.h5reader import ProfilingH5Reader


def _write_csv(
    filepath: str | Path, header: Sequence[str], rows: Sequence[Sequence]
) -> None:
    """Write rows of plotting data to a plain-text CSV file."""
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _write_json(filepath: str | Path, payload: dict) -> None:
    """Write the exact data behind a plot to a JSON file."""
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _to_hex(color) -> str:
    """Convert a matplotlib color (e.g. an RGBA tuple) to a ``#rrggbb`` string."""
    if isinstance(color, str):
        return str(color)  # Ensure it's a Python string, not numpy.str_
    try:
        from matplotlib.colors import to_hex

        return str(to_hex(color))  # Ensure result is Python string
    except (ImportError, TypeError):
        return "#1f77b4"  # Default blue


def _get_canvas():
    """Get maxplotlib Canvas for plotting."""
    try:
        from maxplotlib import Canvas
    except ImportError as exc:
        raise ImportError(
            "maxplotlib is required for plotting. Install scope-profiler[pproc] or maxplotlibx."
        ) from exc
    return Canvas


DEFAULT_CMAP = "tab20"


_FALLBACK_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


def _get_cmap_colors(cmap: str, n_colors: int) -> list[str]:
    """Sample ``n_colors`` ``#rrggbb`` strings from a matplotlib colormap.

    Hex strings (rather than RGBA tuples) are returned so the colors can be
    handed to any maxplotlib backend unchanged.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_hex

        samples = plt.get_cmap(cmap)(np.linspace(0, 1, max(n_colors, 1)))
        return [to_hex(color) for color in samples]
    except (ImportError, ValueError):
        # Fall back to a fixed palette if matplotlib is unavailable or the
        # colormap name is unknown.
        return [_FALLBACK_COLORS[i % len(_FALLBACK_COLORS)] for i in range(n_colors)]


def _add_bar(
    canvas,
    hovertexts: list[str],
    row: int | None,
    x0: float,
    x1: float,
    y_bottom: float,
    y_top: float,
    color: str,
    alpha: float,
    hovertext: str,
) -> None:
    """Draw one filled rectangle on a canvas subplot.

    maxplotlib's ``Canvas.gantt``/``Canvas.flame_chart`` cannot render this
    data: they place one lane per bar rather than per region, and the flame
    chart recomputes depth from frame *names* (so repeated or recursive
    region names land at the wrong depth) while ignoring per-region colors.
    Rectangles are therefore drawn directly, which also works on every
    backend. ``_style_bars`` restores the outline and hover text afterwards
    for Plotly, which ``fill_between`` does not forward.
    """
    col = None if row is None else 0
    canvas.fill_between(
        [x0, x1],
        y_top,
        y_bottom,
        row=row,
        col=col,
        color=color,
        alpha=alpha,
        edgecolor="black",
        linewidth=0.5,
    )
    hovertexts.append(hovertext)


def _style_bars(fig, hovertexts: Sequence[str]) -> None:
    """Attach outlines and hover text to the Plotly traces made by ``_add_bar``."""
    bar_traces = [
        trace for trace in fig.data if getattr(trace, "fill", None) == "toself"
    ]
    for trace, hovertext in zip(bar_traces, hovertexts):
        # Plotly defaults short scatter traces to "lines+markers", which would
        # dot every bar corner.
        trace.mode = "lines"
        trace.line.color = "black"
        trace.line.width = 0.5
        trace.hoveron = "fills"
        trace.hoverinfo = "text"
        trace.hovertext = hovertext


def _render(
    canvas,
    filepath: str | None,
    show: bool,
    backend: str,
    hovertexts: Sequence[str] = (),
) -> None:
    """Save and/or display a canvas, keeping Plotly hover text intact."""
    if backend == "plotly":
        fig = canvas.plot_plotly(show=False)
        _style_bars(fig, hovertexts)
        if filepath:
            if Path(filepath).suffix.lower() in {".html", ".htm"}:
                fig.write_html(filepath)
            else:
                try:
                    fig.write_image(filepath)
                except Exception as exc:
                    raise RuntimeError(
                        "Plotly image export failed. For PNG/PDF/SVG export, "
                        "install kaleido (e.g. `pip install -U kaleido`), or "
                        "export to an .html filepath instead."
                    ) from exc
        if show:
            fig.show()
        return

    if filepath:
        canvas.savefig(filepath, backend=backend)
    if show:
        canvas.show(backend=backend)
    elif backend == "matplotlib":
        _close_matplotlib_figure(canvas)


def _panel_gridspec(
    fig_width: float, fig_height: float, label_chars: int, multi_panel: bool
) -> dict:
    """Reserve figure margins for tick labels, axis labels and titles.

    maxplotlib renders without ``tight_layout``, so long y-tick labels (region
    names) and the x-axis label would otherwise be cut off at the figure edge.
    """
    label_inches = 0.5 + 0.075 * label_chars
    gridspec = {
        "left": min(0.35, label_inches / fig_width),
        "right": 0.98,
        "bottom": min(0.25, 0.7 / fig_height),
        "top": 1 - min(0.25, (1.1 if multi_panel else 0.55) / fig_height),
    }
    if multi_panel:
        gridspec["hspace"] = 0.5
    return gridspec


def _close_matplotlib_figure(canvas) -> None:
    """Release the figure maxplotlib keeps open after rendering."""
    fig = getattr(canvas, "_matplotlib_fig", None)
    if fig is None:
        return
    import matplotlib.pyplot as plt

    plt.close(fig)


def _region_color_map(region_names, cmap: str = DEFAULT_CMAP) -> dict:
    """Assign each region name a stable color from a canonical sorted order."""
    names = sorted(set(region_names))
    colors = _get_cmap_colors(cmap, len(names))
    return dict(zip(names, colors))


def _as_readers(
    profiling_data: ProfilingH5Reader | Sequence[ProfilingH5Reader],
) -> list[ProfilingH5Reader]:
    if isinstance(profiling_data, ProfilingH5Reader):
        return [profiling_data]
    return list(profiling_data)


def _unique_labels(labels: Sequence[str]) -> list[str]:
    label_counts: dict[str, int] = {}
    unique_labels: list[str] = []
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
        if label_counts[label] > 1:
            unique_labels.append(f"{label} ({label_counts[label]})")
        else:
            unique_labels.append(label)
    return unique_labels


def _normalize_ranks(
    ranks: list[int] | int | None,
) -> list[int] | None:
    if ranks is None:
        return None
    if isinstance(ranks, int):
        return [ranks]
    return list(ranks)


def _region_average_duration(
    region,
    ranks: list[int] | None = None,
) -> float:
    if ranks is None:
        selected_ranks = list(region.regions.keys())
    else:
        selected_ranks = [rank for rank in ranks if rank in region.regions]

    durations = [
        region.regions[rank].durations
        for rank in selected_ranks
        if region.regions[rank].durations.size
    ]
    if not durations:
        return float("nan")

    values = np.concatenate(durations)
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _region_duration_values(
    region,
    ranks: list[int] | None = None,
) -> np.ndarray:
    if ranks is None:
        selected_ranks = list(region.regions.keys())
    else:
        selected_ranks = [rank for rank in ranks if rank in region.regions]

    durations = [
        region.regions[rank].durations
        for rank in selected_ranks
        if region.regions[rank].durations.size
    ]
    if not durations:
        return np.array([], dtype=float)
    return np.concatenate(durations)


def _stats_from_values(values: np.ndarray) -> dict[str, float | int | None]:
    if values.size == 0:
        return {
            "count": 0,
            "average_duration_seconds": None,
            "min_duration_seconds": None,
            "max_duration_seconds": None,
            "std_duration_seconds": None,
            "total_duration_seconds": None,
        }

    return {
        "count": int(values.size),
        "average_duration_seconds": float(np.mean(values)),
        "min_duration_seconds": float(np.min(values)),
        "max_duration_seconds": float(np.max(values)),
        "std_duration_seconds": float(np.std(values)),
        "total_duration_seconds": float(np.sum(values)),
    }


def _common_region_names(
    readers: Sequence[ProfilingH5Reader],
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
) -> list[str]:
    filtered_regions = [
        reader.get_regions(include=include, exclude=exclude) for reader in readers
    ]
    if not filtered_regions or not filtered_regions[0]:
        return []

    region_name_sets = [
        {candidate.name for candidate in regions} for regions in filtered_regions[1:]
    ]
    return [
        region.name
        for region in filtered_regions[0]
        if all(region.name in names for names in region_name_sets)
    ]


_SCALING_X_FIELDS = {"num_ranks", "omp_num_threads", "total_cores"}


def _speedup_x_value(reader: ProfilingH5Reader, x_field: str):
    """Resolve the x-axis value for a single reader given ``x_field``."""
    if x_field == "num_ranks":
        return reader.num_ranks

    if x_field == "omp_num_threads":
        value = reader.metadata.get("omp_num_threads")
        if value is None:
            raise ValueError(
                f"'omp_num_threads' not found in metadata for {reader.file_path}"
            )
        return int(value)

    if x_field == "total_cores":
        value = reader.metadata.get("omp_num_threads")
        if value is None:
            raise ValueError(
                f"'omp_num_threads' not found in metadata for {reader.file_path}"
            )
        return reader.num_ranks * int(value)

    if x_field not in reader.metadata:
        raise ValueError(
            f"Metadata field {x_field!r} not found for {reader.file_path}. "
            f"Available fields: {sorted(reader.metadata)}"
        )
    return reader.metadata[x_field]


def collect_region_statistics(
    profiling_data: ProfilingH5Reader | Sequence[ProfilingH5Reader],
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
    labels: Sequence[str] | None = None,
) -> dict:
    """Collect aggregate region-duration statistics for one or more profiling files."""
    readers = _as_readers(profiling_data)
    selected_ranks = _normalize_ranks(ranks)

    if labels is None:
        labels = _unique_labels([reader.file_path.stem for reader in readers])
    else:
        labels = list(labels)

    if len(labels) != len(readers):
        raise ValueError("labels must match the number of profiling files.")

    files_payload = []
    for label, reader in zip(labels, readers):
        regions = reader.get_regions(include=include, exclude=exclude)
        region_payload = {}
        for region in regions:
            values = _region_duration_values(region, selected_ranks)
            per_rank_stats = {}
            for rank in sorted(region.regions.keys()):
                if selected_ranks is not None and rank not in selected_ranks:
                    continue
                rank_values = region.regions[rank].durations
                per_rank_stats[str(rank)] = _stats_from_values(rank_values)
            region_payload[region.name] = {
                **_stats_from_values(values),
                "per_rank": per_rank_stats,
            }

        files_payload.append(
            {
                "label": label,
                "file_path": str(Path(reader.file_path).resolve()),
                "num_ranks": reader.num_ranks,
                "region_statistics": region_payload,
            }
        )

    return {
        "units": {"durations": "seconds"},
        "filters": {
            "include": include,
            "exclude": exclude,
            "ranks": selected_ranks,
        },
        "common_regions": (
            _common_region_names(readers, include=include, exclude=exclude)
            if len(readers) > 1
            else list(files_payload[0]["region_statistics"].keys())
        ),
        "files": files_payload,
    }


def write_region_statistics_json(
    profiling_data: ProfilingH5Reader | Sequence[ProfilingH5Reader],
    filepath: str | Path,
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
    labels: Sequence[str] | None = None,
) -> dict:
    """Write aggregate region-duration statistics to a JSON file."""
    payload = collect_region_statistics(
        profiling_data=profiling_data,
        ranks=ranks,
        include=include,
        exclude=exclude,
        labels=labels,
    )
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _prepare_gantt_data(
    profiling_data: ProfilingH5Reader,
    ranks: list[int] | int | None,
    include: list[str] | str | None,
    exclude: list[str] | str | None,
) -> tuple[list, list[int], float]:
    regions = profiling_data.get_regions(include=include, exclude=exclude)
    if not regions:
        raise ValueError("No regions matched the selected filters.")

    normalized_ranks = _normalize_ranks(ranks)
    if normalized_ranks is None:
        normalized_ranks = list(range(profiling_data.num_ranks))
    else:
        invalid_ranks = [
            rank
            for rank in normalized_ranks
            if rank < 0 or rank >= profiling_data.num_ranks
        ]
        if invalid_ranks:
            raise ValueError(f"Invalid ranks requested: {invalid_ranks}")

    return regions, normalized_ranks, profiling_data.minimum_start_time


def plot_gantt(
    profiling_data: ProfilingH5Reader | Sequence[ProfilingH5Reader],
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
) -> None:
    """
    Plot a Gantt chart of all (or selected) regions with per-rank lanes using maxplotlib.

    Parameters
    ----------
    backend : str
        Backend to use for rendering: "matplotlib" (default) or "plotly".
    """
    Canvas = _get_canvas()
    readers = _as_readers(profiling_data)
    if not readers:
        raise ValueError("No profiling data provided.")

    prepared = []
    for reader in readers:
        regions, selected_ranks, first_start_time = _prepare_gantt_data(
            reader,
            ranks,
            include,
            exclude,
        )
        prepared.append((reader, regions, selected_ranks, first_start_time))

    color_map = _region_color_map(
        (region.name for _, regions, _, _ in prepared for region in regions),
        cmap=cmap,
    )
    for _, regions, _, _ in prepared:
        for region in regions:
            region.color = color_map[region.name]

    labels = _unique_labels([reader.file_path.stem for reader, _, _, _ in prepared])

    if data_filepath:
        if data_format == "json":
            intervals = []
            colors = {}
            for label, (_, regions, selected_ranks, first_start_time) in zip(
                labels, prepared
            ):
                for region in regions:
                    colors[region.name] = _to_hex(region.color)
                    for rank in selected_ranks:
                        region_data = region[rank]
                        for start, end in zip(
                            region_data.start_times, region_data.end_times
                        ):
                            intervals.append(
                                {
                                    "file": label,
                                    "rank": rank,
                                    "region": region.name,
                                    "start_seconds": start - first_start_time,
                                    "end_seconds": end - first_start_time,
                                }
                            )
            _write_json(data_filepath, {"intervals": intervals, "colors": colors})
        else:
            rows = []
            for label, (_, regions, selected_ranks, first_start_time) in zip(
                labels, prepared
            ):
                for region in regions:
                    for rank in selected_ranks:
                        region_data = region[rank]
                        for start, end in zip(
                            region_data.start_times, region_data.end_times
                        ):
                            rows.append(
                                [
                                    label,
                                    rank,
                                    region.name,
                                    start - first_start_time,
                                    end - first_start_time,
                                ]
                            )
            _write_csv(
                data_filepath,
                ["file", "rank", "region", "start_seconds", "end_seconds"],
                rows,
            )

    if verbose:
        if len(prepared) == 1:
            print(f"Plotting Gantt chart for ranks: {prepared[0][2]}")
        else:
            print(f"Plotting combined Gantt chart for files: {', '.join(labels)}")

    single_panel = len(prepared) == 1
    panel_heights = [
        max(2.5, 0.4 * len(regions) * len(selected_ranks))
        for _, regions, selected_ranks, _ in prepared
    ]
    fig_width, fig_height = 12.0, 1.0 + sum(panel_heights)
    lane_label_chars = max(
        len(f"{region.name} (rank {rank})")
        for _, regions, selected_ranks, _ in prepared
        for region in regions
        for rank in selected_ranks
    )
    canvas = Canvas(
        nrows=len(prepared),
        ncols=1,
        figsize=(fig_width, fig_height),
        gridspec_kw=_panel_gridspec(
            fig_width, fig_height, lane_label_chars, not single_panel
        ),
    )

    hovertexts: list[str] = []
    for idx, (label, (_, regions, selected_ranks, first_start_time)) in enumerate(
        zip(labels, prepared)
    ):
        row = None if single_panel else idx
        col = None if single_panel else 0

        # One lane per (region, rank); every call of that region is a bar in it.
        yticks = []
        yticklabels = []
        max_end = 0.0
        for i, region in enumerate(regions):
            for irank, rank in enumerate(selected_ranks):
                starts = region[rank].start_times - first_start_time
                ends = region[rank].end_times - first_start_time
                y = i * len(selected_ranks) + irank
                for start, end in zip(starts, ends):
                    _add_bar(
                        canvas,
                        hovertexts,
                        row,
                        x0=start,
                        x1=end,
                        y_bottom=y - 0.4,
                        y_top=y + 0.4,
                        color=_to_hex(region.color),
                        alpha=0.7,
                        hovertext=(
                            f"{region.name}<br>"
                            f"rank: {rank}<br>"
                            f"start: {start:.6f} s<br>"
                            f"end: {end:.6f} s<br>"
                            f"duration: {end - start:.6f} s"
                        ),
                    )
                    max_end = max(max_end, float(end))
                yticks.append(y)
                yticklabels.append(f"{region.name} (rank {rank})")

        canvas.set_yticks(yticks, labels=yticklabels, row=row, col=col)
        # Rectangles don't drive Plotly's autorange, so frame the panel from
        # the data.
        canvas.set_xlim(0, max_end, row=row, col=col)
        canvas.set_ylim(-0.6, len(yticks) - 0.4, row=row, col=col)
        canvas.set_xlabel("Time (seconds)", row=row, col=col)
        canvas.set_title(
            "Profiling Gantt Chart" if single_panel else label, row=row, col=col
        )
        canvas.set_grid(True, row=row, col=col)

    if not single_panel:
        canvas.suptitle("Combined Profiling Gantt Chart")

    _render(canvas, filepath, show, backend, hovertexts)


def _build_call_stack_intervals(regions: list, rank: int) -> list[dict]:
    """Reconstruct per-call nesting depth for one rank from region intervals."""
    calls = []
    for region in regions:
        if rank not in region.regions:
            continue
        region_data = region.regions[rank]
        for start, end in zip(region_data.start_times, region_data.end_times):
            calls.append(
                {
                    "name": region.name,
                    "start": float(start),
                    "end": float(end),
                    "color": region.color,
                }
            )

    calls.sort(key=lambda call: (call["start"], -call["end"]))

    open_stack: list[dict] = []
    for call in calls:
        while open_stack and open_stack[-1]["end"] <= call["start"]:
            open_stack.pop()
        call["depth"] = len(open_stack)
        open_stack.append(call)

    return calls


def plot_flame(
    profiling_data: ProfilingH5Reader | Sequence[ProfilingH5Reader],
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
) -> None:
    """
    Plot a flame graph reconstructing the call stack from region timings using maxplotlib.

    Parameters
    ----------
    backend : str
        Backend to use for rendering: "matplotlib" (default) or "plotly".
    """
    Canvas = _get_canvas()
    readers = _as_readers(profiling_data)
    if not readers:
        raise ValueError("No profiling data provided.")

    normalized_ranks = _normalize_ranks(ranks) if ranks is not None else [0]

    reader_regions = []
    all_region_names: set[str] = set()
    for reader in readers:
        regions = reader.get_regions(include=include, exclude=exclude)
        if not regions:
            raise ValueError("No regions matched the selected filters.")
        all_region_names.update(region.name for region in regions)
        reader_regions.append((reader, regions))

    color_map = _region_color_map(all_region_names, cmap=cmap)
    for _, regions in reader_regions:
        for region in regions:
            region.color = color_map[region.name]

    prepared = []
    for reader, regions in reader_regions:
        for rank in normalized_ranks:
            if rank < 0 or rank >= reader.num_ranks:
                raise ValueError(f"Invalid rank requested: {rank}")
            calls = _build_call_stack_intervals(regions, rank)
            if calls:
                prepared.append((reader, rank, calls))

    if not prepared:
        raise ValueError("No calls recorded for the requested ranks.")

    if data_filepath:
        labels = _unique_labels([reader.file_path.stem for reader, _, _ in prepared])
        if data_format == "json":
            call_records = []
            colors = {}
            for label, (_, rank, calls) in zip(labels, prepared):
                for call in calls:
                    colors[call["name"]] = _to_hex(call["color"])
                    call_records.append(
                        {
                            "file": label,
                            "rank": rank,
                            "region": call["name"],
                            "depth": call["depth"],
                            "start_seconds": call["start"],
                            "end_seconds": call["end"],
                        }
                    )
            _write_json(data_filepath, {"calls": call_records, "colors": colors})
        else:
            rows = []
            for label, (_, rank, calls) in zip(labels, prepared):
                for call in calls:
                    rows.append(
                        [
                            label,
                            rank,
                            call["name"],
                            call["depth"],
                            call["start"],
                            call["end"],
                        ]
                    )
            _write_csv(
                data_filepath,
                ["file", "rank", "region", "depth", "start_seconds", "end_seconds"],
                rows,
            )

    if verbose:
        print(
            "Plotting flame graph for: "
            + ", ".join(
                f"{reader.file_path.stem} (rank {rank})" for reader, rank, _ in prepared
            )
        )

    single_panel = len(prepared) == 1
    panel_heights = [
        max(2.0, 0.6 * (max(call["depth"] for call in calls) + 1))
        for _, _, calls in prepared
    ]
    fig_width, fig_height = 12.0, 1.0 + sum(panel_heights)
    canvas = Canvas(
        nrows=len(prepared),
        ncols=1,
        figsize=(fig_width, fig_height),
        # Depth numbers plus the "Call depth" axis label.
        gridspec_kw=_panel_gridspec(fig_width, fig_height, 8, not single_panel),
    )

    hovertexts: list[str] = []
    for idx, (reader, rank, calls) in enumerate(prepared):
        row = None if single_panel else idx
        col = None if single_panel else 0

        first_start = min(call["start"] for call in calls)
        total_span = max(call["end"] for call in calls) - first_start
        max_depth = max(call["depth"] for call in calls)

        for call in calls:
            start = call["start"] - first_start
            width = call["end"] - call["start"]
            _add_bar(
                canvas,
                hovertexts,
                row,
                x0=start,
                x1=start + width,
                y_bottom=call["depth"] - 0.45,
                y_top=call["depth"] + 0.45,
                color=_to_hex(call["color"]),
                alpha=0.85,
                hovertext=(
                    f"{call['name']}<br>"
                    f"depth: {call['depth']}<br>"
                    f"start: {start:.6f} s<br>"
                    f"end: {start + width:.6f} s<br>"
                    f"duration: {width:.6f} s"
                ),
            )
            if total_span > 0 and width / total_span > 0.02:
                canvas.text(
                    start + width / 2,
                    call["depth"],
                    call["name"],
                    row=row,
                    col=col,
                    ha="center",
                    va="center",
                    fontsize=8,
                )

        canvas.set_yticks(list(range(max_depth + 1)), row=row, col=col)
        # Rectangles don't drive Plotly's autorange, so frame the panel from
        # the data.
        canvas.set_xlim(0, total_span, row=row, col=col)
        canvas.set_ylim(-0.6, max_depth + 0.6, row=row, col=col)
        canvas.set_xlabel("Time (seconds)", row=row, col=col)
        canvas.set_ylabel("Call depth", row=row, col=col)
        canvas.set_title(f"{reader.file_path.stem} (rank {rank})", row=row, col=col)
        canvas.set_grid(True, row=row, col=col)

    if not single_panel:
        canvas.suptitle("Flame Graphs")

    _render(canvas, filepath, show, backend, hovertexts)


_DURATION_METRICS: dict[str, tuple[str, str]] = {
    "avg": ("average_duration_seconds", "Average duration per call (seconds)"),
    "min": ("min_duration_seconds", "Minimum duration per call (seconds)"),
    "max": ("max_duration_seconds", "Maximum duration per call (seconds)"),
    "total": ("total_duration_seconds", "Total duration (seconds)"),
}


def _region_metric_value(
    region,
    metric_key: str,
    ranks: list[int] | None = None,
) -> float:
    values = _region_duration_values(region, ranks=ranks)
    stats = _stats_from_values(values)
    stat_value = stats[metric_key]
    return float("nan") if stat_value is None else stat_value


def _metric_filepath(filepath: str, metric_key: str, single_metric: bool) -> str:
    if single_metric:
        return filepath
    base, ext = os.path.splitext(filepath)
    return f"{base}_{metric_key}{ext}"


def plot_durations(
    profiling_data: ProfilingH5Reader | Sequence[ProfilingH5Reader],
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
    labels: Sequence[str] | None = None,
    metrics: list[str] | str | None = None,
    filepath: str | None = None,
    show: bool = False,
    verbose: bool = True,
    cmap: str = DEFAULT_CMAP,
    data_filepath: str | Path | None = None,
    data_format: str = "csv",
    backend: str = "matplotlib",
) -> list[str]:
    """Plot duration bar charts for one or more profiling files using maxplotlib.

    Parameters
    ----------
    backend : str
        Backend to use for rendering: "matplotlib" (default) or "plotly".

    Returns
    -------
    list[str]
        List of filepaths that were written (empty if filepath is None).
    """
    Canvas = _get_canvas()
    readers = _as_readers(profiling_data)
    ranks = _normalize_ranks(ranks)

    if metrics is None:
        metric_keys = list(_DURATION_METRICS)
    elif isinstance(metrics, str):
        metric_keys = [metrics]
    else:
        metric_keys = list(metrics)

    unknown_metrics = [key for key in metric_keys if key not in _DURATION_METRICS]
    if unknown_metrics:
        raise ValueError(
            f"Unknown metric(s) {unknown_metrics}. "
            f"Valid options are: {list(_DURATION_METRICS)}"
        )

    if labels is None:
        labels = _unique_labels([reader.file_path.stem for reader in readers])
    else:
        labels = list(labels)

    if len(labels) != len(readers):
        raise ValueError("labels must match the number of profiling files.")

    region_names = _common_region_names(readers, include=include, exclude=exclude)
    if not region_names:
        raise ValueError("No regions matched the selected filters.")

    if verbose:
        print(
            f"Plotting duration comparison ({', '.join(metric_keys)}) "
            f"for files: {', '.join(labels)}"
        )

    num_readers = len(readers)
    colors = _get_cmap_colors(cmap, max(num_readers, 1))
    fig_width = max(10, 0.85 * len(region_names) + 2)
    fig_height = max(4.5, 2.5 + 0.35 * num_readers)
    width = min(0.8 / max(num_readers, 1), 0.35)

    saved_paths: list[str] = []
    data_rows = []

    for metric_key in metric_keys:
        stat_key, ylabel = _DURATION_METRICS[metric_key]

        values = [
            [
                _region_metric_value(
                    reader.get_region(region_name), stat_key, ranks=ranks
                )
                for region_name in region_names
            ]
            for reader in readers
        ]

        if data_filepath:
            for label, file_values in zip(labels, values):
                for region_name, value in zip(region_names, file_values):
                    data_rows.append([label, region_name, metric_key, value])

        canvas = Canvas(figsize=(fig_width, fig_height))

        # Create grouped bar chart
        x_positions = np.arange(len(region_names))
        offset_start = -0.5 * width * (num_readers - 1)

        for idx, (label, file_values) in enumerate(zip(labels, values)):
            offsets = x_positions + offset_start + idx * width
            canvas.bar(
                offsets,
                file_values,
                width=width,
                label=label if num_readers > 1 else None,
                color=_to_hex(colors[idx]),
                edgecolor="black",
                alpha=0.8,
            )

        canvas.set_xticks(x_positions, labels=region_names)
        canvas.set_ylabel(ylabel)
        canvas.set_title(f"Region duration comparison ({metric_key})")
        canvas.set_grid(True)
        if num_readers > 1:
            canvas.set_legend()

        metric_filepath = None
        if filepath:
            metric_filepath = _metric_filepath(
                filepath, metric_key, single_metric=len(metric_keys) == 1
            )
            saved_paths.append(metric_filepath)

        _render(canvas, metric_filepath, show, backend)

    if data_filepath:
        if data_format == "json":
            bars = [
                {
                    "file": file,
                    "region": region,
                    "metric": metric,
                    "value_seconds": value,
                }
                for file, region, metric, value in data_rows
            ]
            colors_map = {label: _to_hex(color) for label, color in zip(labels, colors)}
            _write_json(
                data_filepath,
                {"bars": bars, "colors": colors_map, "metrics": metric_keys},
            )
        else:
            _write_csv(
                data_filepath, ["file", "region", "metric", "value_seconds"], data_rows
            )

    return saved_paths


def plot_speedup(
    profiling_data: ProfilingH5Reader | Sequence[ProfilingH5Reader],
    x_field: str = "num_ranks",
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
) -> None:
    """Plot scope speedup versus a chosen parallelism/metadata field using maxplotlib.

    Parameters
    ----------
    backend : str
        Backend to use for rendering: "matplotlib" (default) or "plotly".
    """
    Canvas = _get_canvas()
    readers = _as_readers(profiling_data)
    if len(readers) < 2:
        raise ValueError("Speedup plot requires at least two profiling files.")

    region_names = _common_region_names(readers, include=include, exclude=exclude)
    if not region_names:
        raise ValueError("No regions matched the selected filters.")

    is_scaling = x_field in _SCALING_X_FIELDS
    x_per_reader = [_speedup_x_value(reader, x_field) for reader in readers]

    if is_scaling:
        x_keys = sorted({int(value) for value in x_per_reader})
    else:
        x_keys = list(dict.fromkeys(x_per_reader))

    if verbose:
        print(
            f"Plotting speedup comparison using x_field={x_field!r}, values: "
            + ", ".join(map(str, x_keys))
        )

    duration_samples: dict[str, dict] = {
        region_name: defaultdict(list) for region_name in region_names
    }
    for reader, x_value in zip(readers, x_per_reader):
        for region_name in region_names:
            duration = _region_average_duration(
                reader.get_region(region_name),
                ranks=ranks,
            )
            if np.isfinite(duration) and duration > 0:
                duration_samples[region_name][x_value].append(duration)

    baseline_key = x_keys[0]
    colors = _get_cmap_colors(cmap, len(region_names))
    fig_width = max(10, 1.2 * len(x_keys) + 3)
    fig_height = max(4.5, 2.8 + 0.35 * len(region_names))

    x_position = {key: (key if is_scaling else i) for i, key in enumerate(x_keys)}

    canvas = Canvas(figsize=(fig_width, fig_height))
    plotted = 0
    data_rows = []

    for idx, region_name in enumerate(region_names):
        region_values = duration_samples[region_name]
        baseline_samples = region_values.get(baseline_key, [])
        if not baseline_samples:
            continue

        baseline_duration = float(np.mean(baseline_samples))
        if not np.isfinite(baseline_duration) or baseline_duration <= 0:
            continue

        plot_x = []
        plot_keys = []
        speedups = []
        for key in x_keys:
            samples = region_values.get(key, [])
            if not samples:
                continue
            mean_duration = float(np.mean(samples))
            if not np.isfinite(mean_duration) or mean_duration <= 0:
                continue
            plot_x.append(x_position[key])
            plot_keys.append(key)
            speedups.append(baseline_duration / mean_duration)

        if not plot_x:
            continue

        plotted += 1
        canvas.add_line(
            plot_x,
            speedups,
            linewidth=1.8,
            color=_to_hex(colors[idx]),
            label=region_name,
        )
        if data_filepath:
            for key, speedup in zip(plot_keys, speedups):
                data_rows.append([region_name, key, speedup])

    if plotted == 0:
        raise ValueError("No valid speedup data could be computed.")

    if data_filepath:
        if data_format == "json":
            points = [
                {"region": region, x_field: key, "speedup": speedup}
                for region, key, speedup in data_rows
            ]
            colors_map = {
                name: _to_hex(color) for name, color in zip(region_names, colors)
            }
            _write_json(data_filepath, {"points": points, "colors": colors_map})
        else:
            _write_csv(data_filepath, ["region", x_field, "speedup"], data_rows)

    x_label_map = {
        "num_ranks": "MPI ranks",
        "omp_num_threads": "OpenMP threads",
        "total_cores": "MPI ranks × OpenMP threads",
    }
    x_label = x_label_map.get(x_field, x_field)

    if is_scaling:
        x_line = np.array(x_keys, dtype=float)
        canvas.add_line(
            x_line,
            x_line / baseline_key,
            linestyle="--",
            color="black",
            linewidth=1.5,
            label="Ideal scaling",
        )
        canvas.set_xticks(x_line)
    else:
        canvas.set_xticks(list(range(len(x_keys))), labels=[str(key) for key in x_keys])

    canvas.set_xlabel(x_label)
    canvas.set_ylabel("Speedup")
    canvas.set_title(f"Region speedup scaling (baseline: {x_label} = {baseline_key})")
    canvas.set_grid(True)
    canvas.set_legend()

    _render(canvas, filepath, show, backend)
