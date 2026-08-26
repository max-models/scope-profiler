"""Plotting utilities for visualizing profiling data using maxplotlib.

This module provides plotting functions that can export to both matplotlib and plotly
backends using maxplotlib as the unified interface.
"""

import csv
import json
import os
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from scope_profiler.call_stack import build_call_stack
from scope_profiler.results import ProfilingResults
from scope_profiler.summary import _name_selected


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


def _add_pyvis_controls(filepath: str | Path) -> None:
    """Add lightweight search/focus/collapse controls to a PyVis document."""
    controls = """
<div id="scope-profiler-controls" style="position:fixed;top:12px;left:12px;z-index:10;background:white;padding:8px;border:1px solid #bbb;border-radius:4px;font:14px sans-serif">
  <input id="sp-search" placeholder="Search region..." />
  <button onclick="spFocus()">Focus</button>
  <button onclick="spReset()">Reset</button>
  <button onclick="spToggle()">Expand / collapse</button>
</div>
<script>
const spHidden = new Set();
function spName() { return document.getElementById('sp-search').value.toLowerCase(); }
function spMatches() { return nodes.get().filter(n => String(n.label || '').toLowerCase().includes(spName())); }
function spFocus() {
  const hits = spMatches().map(n => n.id), keep = new Set(hits);
  let changed = true;
  while (changed) { changed = false; edges.get().forEach(e => {
    if (keep.has(e.from) && !keep.has(e.to)) { keep.add(e.to); changed = true; }
    if (keep.has(e.to) && !keep.has(e.from)) { keep.add(e.from); changed = true; }
  }); }
  nodes.get().forEach(n => nodes.update({id:n.id, hidden:!keep.has(n.id)}));
  network.fit();
}
function spReset() { spHidden.clear(); nodes.get().forEach(n => nodes.update({id:n.id, hidden:false})); network.fit(); }
function spToggle() {
  const hits = spMatches(); if (!hits.length) return;
  const roots = new Set(hits.map(n => n.id)), descendants = new Set();
  function walk(id) { edges.get().forEach(e => { if (e.from === id && !descendants.has(e.to)) { descendants.add(e.to); walk(e.to); } }); }
  roots.forEach(walk); const hide = [...descendants].some(id => !spHidden.has(id));
  descendants.forEach(id => { if (hide) spHidden.add(id); else spHidden.delete(id); nodes.update({id:id, hidden:hide}); });
}
</script>
"""
    path = Path(filepath)
    content = path.read_text(encoding="utf-8")
    path.write_text(content.replace("</body>", controls + "</body>"), encoding="utf-8")


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
            "maxplotlib is required for plotting. Install scope-profiler[pproc] "
            "or maxplotlibx (>= 0.1.7, for its gantt and flame charts)."
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


def _add_gantt_bars(
    canvas,
    row: int | None,
    lanes: Sequence[str],
    bars: Sequence[tuple[int, float, float, str]],
    alpha: float = 0.7,
) -> None:
    """Draw ``Canvas.gantt`` bars, every call of a lane on that lane's row.

    ``Canvas.gantt`` puts bar *i* of a call on row *i* (``y =
    arange(len(tasks))``, on both backends) and takes a single color for the
    whole call, since maxplotlib's Plotly backend reads ``color`` as one
    RGB(A) value. Each drawing call therefore passes one entry per lane --- so
    the entry index is the lane's row --- and only fills in the bars sharing
    one color; the rest are NaN, which draws nothing on either backend.

    Lanes hold several calls, so bars are grouped by their position within the
    lane as well: the *k*-th call of every lane goes into the same group,
    leaving at most one bar per lane per drawing call.
    """
    col = None if row is None else 0
    n_lanes = len(lanes)

    # (color, index within the lane) -> the bars drawn together.
    groups: dict[tuple[str, int], list[tuple[int, float, float]]] = defaultdict(list)
    calls_per_lane: dict[int, int] = defaultdict(int)
    for lane, start, duration, color in bars:
        slot = calls_per_lane[lane]
        calls_per_lane[lane] += 1
        groups[(color, slot)].append((lane, start, duration))

    for (color, _), group in groups.items():
        starts = np.full(n_lanes, np.nan)
        durations = np.full(n_lanes, np.nan)
        for lane, start, duration in group:
            starts[lane] = start
            durations[lane] = duration
        canvas.gantt(
            list(lanes),
            starts,
            durations,
            row=row,
            col=col,
            color=color,
            edgecolor="black",
            alpha=alpha,
        )


def _render(
    canvas,
    filepath: str | None,
    show: bool,
    backend: str,
    plotly_layout: dict | None = None,
    x_tick_rotation: float | None = None,
    return_fig: bool = False,
    matplotlib_postprocess=None,
) -> tuple[object, object] | object:
    """Save and/or display a canvas."""
    if backend == "plotly":
        fig = canvas.plot_plotly(show=False)
        layout = dict(plotly_layout or {})
        if x_tick_rotation is not None:
            layout["xaxis_tickangle"] = x_tick_rotation
        if layout:
            fig.update_layout(**layout)
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
        return fig

    if backend == "matplotlib" and (
        show
        or x_tick_rotation is not None
        or return_fig
        or matplotlib_postprocess is not None
    ):
        # maxplotlib does not currently expose tick-label rotation through
        # its backend-neutral Canvas API, so rotate the labels after the
        # Matplotlib figure has been materialized and before saving/showing.
        fig, axes = canvas.plot(backend="matplotlib", savefig=bool(filepath))
        if matplotlib_postprocess is not None:
            matplotlib_postprocess(fig, axes)
        if x_tick_rotation is not None:
            for axis in np.asarray(axes).reshape(-1):
                for label in axis.get_xticklabels():
                    label.set_rotation(x_tick_rotation)
                    # Anchor the end of each vertical label at the tick so the
                    # text grows upward into the figure instead of below it.
                    label.set_ha("right")
        if filepath:
            _save_matplotlib_figure(fig, canvas, filepath)
        if show:
            displayed_in_notebook = _show_matplotlib_figure(fig)
            if displayed_in_notebook:
                _close_matplotlib_figure(canvas, fig=fig)
        elif not return_fig:
            _close_matplotlib_figure(canvas, fig=fig)
        return fig, axes

    if filepath:
        canvas.savefig(filepath, backend=backend)
    if show:
        return canvas.show(backend=backend)
    elif backend == "matplotlib":
        _close_matplotlib_figure(canvas)
    return None


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


def _save_matplotlib_figure(fig, canvas, filepath: str | Path) -> None:
    """Save a materialized Matplotlib figure using canvas-level DPI if present."""
    savefig_kwargs = {}
    if getattr(canvas, "dpi", None) is not None:
        savefig_kwargs["dpi"] = canvas.dpi
    fig.savefig(filepath, **savefig_kwargs)


def _show_matplotlib_figure(fig) -> bool:
    """Display a Matplotlib figure in notebooks and regular Python sessions."""
    if _display_matplotlib_figure_in_notebook(fig):
        return True

    import matplotlib.pyplot as plt

    plt.show()
    return False


def _display_matplotlib_figure_in_notebook(fig) -> bool:
    """Use IPython rich display when running inside a Jupyter kernel."""
    try:
        from IPython import get_ipython
        from IPython.display import display
    except ImportError:
        return False

    shell = get_ipython()
    if shell is None:
        return False
    if "IPKernelApp" not in getattr(shell, "config", {}):
        return False

    display(fig)
    return True


def _close_matplotlib_figure(canvas, fig=None) -> None:
    """Release the figure maxplotlib keeps open after rendering."""
    fig = fig if fig is not None else getattr(canvas, "_matplotlib_fig", None)
    if fig is None:
        return
    import matplotlib.pyplot as plt

    plt.close(fig)


def _set_xticks(canvas, ticks, labels=None, **kwargs) -> bool:
    """Set x ticks, returning whether optional tick-label kwargs were accepted."""
    try:
        canvas.set_xticks(ticks, labels=labels, **kwargs)
    except TypeError:
        canvas.set_xticks(ticks, labels=labels)
        return False
    return True


def _region_color_map(region_names, cmap: str = DEFAULT_CMAP) -> dict:
    """Assign each region name a stable color from a canonical sorted order."""
    names = sorted(set(region_names))
    colors = _get_cmap_colors(cmap, len(names))
    return dict(zip(names, colors))


def _as_runs(
    profiling_data: ProfilingResults | Sequence[ProfilingResults],
) -> list[ProfilingResults]:
    """Normalize the input, dropping result sets this rank must not draw.

    Under MPI only rank 0 holds the run's data; every other rank gets an empty,
    non-root result set from ``finalize(return_results=True)``. Dropping those
    here is what lets a parallel script call the plot functions and exporters
    unguarded and still produce exactly one set of figures. An empty list back
    therefore means "not this rank's job" - callers return quietly - while an
    empty input is a mistake and raises.
    """
    if isinstance(profiling_data, ProfilingResults):
        profiling_data = [profiling_data]
    runs = list(profiling_data)
    if not runs:
        raise ValueError("No profiling data provided.")
    return [run for run in runs if run.is_root]


def _filename_slug(label: str) -> str:
    """Make a label safe to paste into a filename.

    Labels are free text and are used to name exported files, so a perfectly
    reasonable one like ``"128 ranks"`` would otherwise produce
    ``profile_128 ranks_rank0.prof``. Only the path is sanitized; the label
    inside the exported document stays as the user wrote it.
    """
    slug = re.sub(r"[^\w.+-]+", "_", label).strip("_")
    return slug or label


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


def _first_last_duration(
    region,
    ranks: list[int] | None = None,
) -> tuple[float | None, float | None]:
    """Duration of the chronologically first and last call, across ranks.

    Pooling durations across ranks loses call order, so first/last are found
    from each rank's own first/last call instead -- the earliest-starting
    rank supplies "first", the latest-ending rank supplies "last".
    """
    if ranks is None:
        selected = region.regions.values()
    else:
        selected = (region.regions[rank] for rank in ranks if rank in region.regions)
    timed = [data for data in selected if data.has_timing]
    if not timed:
        return None, None
    first = min(timed, key=lambda data: data.first_start_time).first_duration
    last = max(timed, key=lambda data: data.last_end_time).last_duration
    return first, last


def _stats_from_values(
    values: np.ndarray,
    first: float | None = None,
    last: float | None = None,
) -> dict[str, float | int | None]:
    """Compute the duration statistics shown in the region-statistics export.

    ``first``/``last`` (the chronologically first/last call's duration) can't
    be derived from ``values`` alone once several ranks have been pooled into
    one array, since that loses call order -- callers that can determine them
    (e.g. a single rank's own, order-preserving array) pass them in.
    """
    if values.size == 0:
        return {
            "count": 0,
            "average_duration_seconds": None,
            "min_duration_seconds": None,
            "max_duration_seconds": None,
            "first_duration_seconds": None,
            "last_duration_seconds": None,
            "std_duration_seconds": None,
            "total_duration_seconds": None,
        }

    return {
        "count": int(values.size),
        "average_duration_seconds": float(np.mean(values)),
        "min_duration_seconds": float(np.min(values)),
        "max_duration_seconds": float(np.max(values)),
        "first_duration_seconds": first,
        "last_duration_seconds": last,
        "std_duration_seconds": float(np.std(values)),
        "total_duration_seconds": float(np.sum(values)),
    }


def _common_region_names(
    runs: Sequence[ProfilingResults],
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
) -> list[str]:
    filtered_regions = [
        run.get_regions(include=include, exclude=exclude) for run in runs
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


def _speedup_x_value(run: ProfilingResults, x_field: str):
    """Resolve the x-axis value for a single run given ``x_field``."""
    if x_field == "num_ranks":
        return run.num_ranks

    if x_field == "omp_num_threads":
        value = run.metadata.get("omp_num_threads")
        if value is None:
            raise ValueError(
                f"'omp_num_threads' not found in metadata for {run.file_path}"
            )
        return int(value)

    if x_field == "total_cores":
        value = run.metadata.get("omp_num_threads")
        if value is None:
            raise ValueError(
                f"'omp_num_threads' not found in metadata for {run.file_path}"
            )
        return run.num_ranks * int(value)

    if x_field not in run.metadata:
        raise ValueError(
            f"Metadata field {x_field!r} not found for {run.file_path}. "
            f"Available fields: {sorted(run.metadata)}"
        )
    return run.metadata[x_field]


def collect_region_statistics(
    profiling_data: ProfilingResults | Sequence[ProfilingResults],
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
    labels: Sequence[str] | None = None,
) -> dict:
    """Collect aggregate region-duration statistics for one or more profiling files."""
    runs = _as_runs(profiling_data)
    selected_ranks = _normalize_ranks(ranks)
    if not runs:
        # Not this rank's data; rank 0 holds it all.
        return {
            "units": {"durations": "seconds"},
            "filters": {
                "include": include,
                "exclude": exclude,
                "ranks": selected_ranks,
            },
            "common_regions": [],
            "files": [],
        }

    if labels is None:
        labels = _unique_labels([run.display_label for run in runs])
    else:
        labels = list(labels)

    if len(labels) != len(runs):
        raise ValueError("labels must match the number of profiling files.")

    files_payload = []
    for label, run in zip(labels, runs):
        regions = run.get_regions(include=include, exclude=exclude)
        region_payload = {}
        for region in regions:
            values = _region_duration_values(region, selected_ranks)
            first, last = _first_last_duration(region, selected_ranks)
            per_rank_stats = {}
            for rank in sorted(region.regions.keys()):
                if selected_ranks is not None and rank not in selected_ranks:
                    continue
                rank_region = region.regions[rank]
                rank_values = rank_region.durations
                rank_first = (
                    rank_region.first_duration if rank_region.has_timing else None
                )
                rank_last = (
                    rank_region.last_duration if rank_region.has_timing else None
                )
                per_rank_stats[str(rank)] = _stats_from_values(
                    rank_values, first=rank_first, last=rank_last
                )
            region_payload[region.name] = {
                **_stats_from_values(values, first=first, last=last),
                "per_rank": per_rank_stats,
            }

        files_payload.append(
            {
                "label": label,
                "file_path": str(Path(run.file_path).resolve()),
                "num_ranks": run.num_ranks,
                "total_time_seconds": run.total_time,
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
            _common_region_names(runs, include=include, exclude=exclude)
            if len(runs) > 1
            else list(files_payload[0]["region_statistics"].keys())
        ),
        "files": files_payload,
    }


def write_region_statistics_json(
    profiling_data: ProfilingResults | Sequence[ProfilingResults],
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
    if not payload["files"]:
        # Non-root rank (see ProfilingResults.is_root): rank 0 writes the file.
        return payload
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _prepare_gantt_data(
    profiling_data: ProfilingResults,
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

    # Charts frame the recorded window: x = 0 is the first region entry, not
    # the run's registered start (run.time_origin), so that a long gap
    # between setup() and the first region does not push the bars off to one
    # side. run.events(origin=run.minimum_start_time) matches this.
    return regions, normalized_ranks, profiling_data.minimum_start_time


def plot_gantt(
    profiling_data: ProfilingResults | Sequence[ProfilingResults],
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
    return_fig: bool = False,
) -> object | None:
    """
    Plot a Gantt chart of all (or selected) regions with per-rank lanes using maxplotlib.

    Parameters
    ----------
    backend : str
        Backend to use for rendering: "matplotlib" (default) or "plotly".
    return_fig : bool
        Return the rendered figure instead of the default ``None``. Matplotlib
        returns ``(fig, axes)``; Plotly returns its figure object.
    """
    Canvas = _get_canvas()
    runs = _as_runs(profiling_data)
    if not runs:
        # Not this rank's job; rank 0 draws it.
        return

    prepared = []
    for run in runs:
        regions, selected_ranks, first_start_time = _prepare_gantt_data(
            run,
            ranks,
            include,
            exclude,
        )
        prepared.append((run, regions, selected_ranks, first_start_time))

    color_map = _region_color_map(
        (region.name for _, regions, _, _ in prepared for region in regions),
        cmap=cmap,
    )
    for _, regions, _, _ in prepared:
        for region in regions:
            region.color = color_map[region.name]

    labels = _unique_labels([run.display_label for run, _, _, _ in prepared])

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

    # One lane per (region, rank): every call of a region lands on the same
    # row, so a panel is as tall as the number of lanes it draws.
    single_panel = len(prepared) == 1
    panel_lanes: list[list[str]] = []
    panel_bars: list[list[tuple[int, float, float, str]]] = []
    for _, regions, selected_ranks, first_start_time in prepared:
        lanes: list[str] = []
        bars: list[tuple[int, float, float, str]] = []
        for region in regions:
            for rank in selected_ranks:
                # A region need not have been entered on every rank.
                if rank not in region:
                    continue
                region_data = region[rank]
                if not len(region_data.start_times):
                    continue
                lane = len(lanes)
                lanes.append(f"{region.name} (rank {rank})")
                color = _to_hex(region.color)
                for start, end in zip(region_data.start_times, region_data.end_times):
                    bars.append(
                        (
                            lane,
                            float(start - first_start_time),
                            float(end - start),
                            color,
                        )
                    )
        panel_lanes.append(lanes)
        panel_bars.append(bars)

    if not any(panel_bars):
        raise ValueError("No calls recorded for the requested ranks.")

    panel_heights = [max(2.5, 0.35 * len(lanes)) for lanes in panel_lanes]
    fig_width, fig_height = 12.0, 1.0 + sum(panel_heights)
    lane_label_chars = max(len(lane) for lanes in panel_lanes for lane in lanes)
    canvas = Canvas(
        nrows=len(prepared),
        ncols=1,
        figsize=(fig_width, fig_height),
        gridspec_kw=_panel_gridspec(
            fig_width, fig_height, lane_label_chars, not single_panel
        ),
    )

    for idx, (label, lanes, bars) in enumerate(zip(labels, panel_lanes, panel_bars)):
        row = None if single_panel else idx
        col = None if single_panel else 0

        _add_gantt_bars(canvas, row, lanes, bars)

        canvas.set_yticks(list(range(len(lanes))), labels=lanes, row=row, col=col)
        canvas.set_xlim(
            0,
            max(start + duration for _, start, duration, _ in bars),
            row=row,
            col=col,
        )
        canvas.set_ylim(-0.6, len(lanes) - 0.4, row=row, col=col)
        canvas.set_xlabel("Time (seconds)", row=row, col=col)
        canvas.set_title(
            "Profiling Gantt Chart" if single_panel else label, row=row, col=col
        )
        canvas.set_grid(True, row=row, col=col)

    if not single_panel:
        canvas.suptitle("Combined Profiling Gantt Chart")

    # One Plotly bar trace per region color; without "overlay" they would
    # share each row's height instead of each drawing on the full row.
    rendered = _render(
        canvas,
        filepath,
        show,
        backend,
        plotly_layout={"barmode": "overlay"},
        return_fig=return_fig,
    )
    return rendered if return_fig else None


def plot_flame(
    profiling_data: ProfilingResults | Sequence[ProfilingResults],
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
    return_fig: bool = False,
) -> object | None:
    """
    Plot a flame graph reconstructing the call stack from region timings using maxplotlib.

    Parameters
    ----------
    backend : str
        Backend to use for rendering: "matplotlib" (default) or "plotly".
    return_fig : bool
        Return the rendered figure instead of the default ``None``. Matplotlib
        returns ``(fig, axes)``; Plotly returns its figure object.
    """
    Canvas = _get_canvas()
    runs = _as_runs(profiling_data)
    if not runs:
        # Not this rank's job; rank 0 draws it.
        return

    normalized_ranks = _normalize_ranks(ranks) if ranks is not None else [0]

    reader_regions = []
    all_region_names: set[str] = set()
    for run in runs:
        regions = run.get_regions(include=include, exclude=exclude)
        if not regions:
            raise ValueError("No regions matched the selected filters.")
        all_region_names.update(region.name for region in regions)
        reader_regions.append((run, regions))

    color_map = _region_color_map(all_region_names, cmap=cmap)
    for _, regions in reader_regions:
        for region in regions:
            region.color = color_map[region.name]

    prepared = []
    for run, regions in reader_regions:
        for rank in normalized_ranks:
            if rank < 0 or rank >= run.num_ranks:
                raise ValueError(f"Invalid rank requested: {rank}")
            calls = build_call_stack(regions, rank)
            if calls:
                prepared.append((run, rank, calls))

    if not prepared:
        raise ValueError("No calls recorded for the requested ranks.")

    if data_filepath:
        labels = _unique_labels([run.display_label for run, _, _ in prepared])
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
                f"{run.display_label} (rank {rank})" for run, rank, _ in prepared
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

    def add_region_legend(fig, axes) -> None:
        """Make Matplotlib flame colors identify regions, not stack depth."""
        if backend != "matplotlib":
            return
        from matplotlib.patches import Patch

        axes_list = np.asarray(axes).reshape(-1)
        for axis, (_, _, calls) in zip(axes_list, prepared):
            for patch, call in zip(axis.patches, calls):
                patch.set_facecolor(_to_hex(color_map[call["name"]]))

        handles = [
            Patch(
                facecolor=_to_hex(color_map[name]),
                edgecolor="black",
                label=name,
            )
            for name in sorted(all_region_names)
        ]
        if handles:
            fig.legend(
                handles=handles,
                title="Regions",
                loc="center left",
                bbox_to_anchor=(0.82, 0.5),
            )
            fig.subplots_adjust(right=0.8)

    for idx, (run, rank, calls) in enumerate(prepared):
        row = None if single_panel else idx
        col = None if single_panel else 0

        first_start = min(call["start"] for call in calls)
        total_span = max(call["end"] for call in calls) - first_start
        max_depth = max(call["depth"] for call in calls)

        canvas.flame_chart(
            [call["name"] for call in calls],
            [call["parent"] for call in calls],
            [call["end"] - call["start"] for call in calls],
            start_times=[call["start"] - first_start for call in calls],
            row=row,
            col=col,
            edgecolor="black",
            # Canvas.flame_chart colors frames by depth from a colormap and
            # ignores per-frame colors. Only the matplotlib backend takes a
            # matplotlib colormap name; the Plotly one needs a Plotly
            # colorscale, so it keeps its own default.
            **({} if backend == "plotly" else {"colormap": cmap}),
        )

        canvas.set_yticks(list(range(max_depth + 1)), row=row, col=col)
        # The frames are drawn as rectangles, which don't drive autorange, so
        # frame the panel from the data.
        canvas.set_xlim(0, total_span, row=row, col=col)
        canvas.set_ylim(-0.6, max_depth + 1.0, row=row, col=col)
        canvas.set_xlabel("Time (seconds)", row=row, col=col)
        canvas.set_ylabel("Call depth", row=row, col=col)
        canvas.set_title(f"{run.display_label} (rank {rank})", row=row, col=col)
        canvas.set_grid(True, row=row, col=col)

    if not single_panel:
        canvas.suptitle("Flame Graphs")

    rendered = _render(
        canvas,
        filepath,
        show,
        backend,
        return_fig=return_fig,
        matplotlib_postprocess=add_region_legend if backend == "matplotlib" else None,
    )
    return rendered if return_fig else None


def plot_callgraph(
    profiling_data: ProfilingResults,
    rank: int = 0,
    include=None,
    exclude=None,
    filepath: str | None = None,
    show: bool = False,
    verbose: bool = True,
    cmap: str = DEFAULT_CMAP,
    data_filepath: str | Path | None = None,
    data_format: str = "csv",
    backend: str = "matplotlib",
    return_fig: bool = False,
    compact: bool = False,
    fluid: bool = False,
) -> object | None:
    """Plot the explicit call graph, without using timestamps or durations.

    When ``compact`` is true, all invocations of a region are represented by
    one node named after that region.  Edges then describe distinct
    caller/callee region relationships rather than individual call IDs.
    ``fluid`` applies a deterministic force-directed layout to the compact
    graph, similar to an Obsidian-style node graph.
    """
    if isinstance(profiling_data, Sequence) and not isinstance(
        profiling_data, ProfilingResults
    ):
        if len(profiling_data) != 1:
            raise ValueError("callgraph accepts one profiling file at a time")
        profiling_data = profiling_data[0]
    nodes = profiling_data.call_graph(rank=rank, include=include, exclude=exclude)
    if not nodes:
        raise ValueError("No calls recorded for the requested rank or filters.")
    edges = None
    compact = compact or fluid
    if compact:
        regions_by_name = {
            region.name: region
            for region in profiling_data.get_regions(include=include, exclude=exclude)
        }
        # The call graph is a DAG in call-id order.  Use per-invocation
        # exclusive time as the node weight and dynamic programming to find
        # the longest cumulative root-to-leaf chain.
        exclusive_by_call = {}
        cumulative_by_call = {}
        for node in nodes:
            region = regions_by_name.get(node["name"])
            rank_region = region.regions.get(rank) if region is not None else None
            if rank_region is None or node["call_index"] >= len(
                rank_region.exclusive_durations
            ):
                exclusive = 0.0
            else:
                exclusive = float(rank_region.exclusive_durations[node["call_index"]])
            exclusive_by_call[node["call_id"]] = exclusive
            parent_total = cumulative_by_call.get(node["parent_id"], 0.0)
            cumulative_by_call[node["call_id"]] = exclusive + parent_total
        endpoint = max(cumulative_by_call, key=cumulative_by_call.get, default=None)
        critical_ids = set()
        by_id = {node["call_id"]: node for node in nodes}
        while endpoint is not None and endpoint in by_id:
            critical_ids.add(endpoint)
            endpoint = by_id[endpoint]["parent_id"]
        compact_nodes = []
        seen_names = set()
        for node in nodes:
            if node["name"] not in seen_names:
                seen_names.add(node["name"])
                region = regions_by_name[node["name"]]
                rank_region = region.regions.get(rank)
                calls = rank_region.num_calls if rank_region is not None else 0
                total = rank_region.total_duration if rank_region is not None else 0.0
                exclusive_total = (
                    float(np.sum(rank_region.exclusive_durations))
                    if rank_region is not None
                    else 0.0
                )
                source = ""
                if rank_region is not None and rank_region.has_source:
                    source = f"{rank_region.source_file}:{rank_region.source_lineno}"
                compact_nodes.append(
                    {
                        "name": node["name"],
                        "depth": node["depth"],
                        "calls": calls,
                        "total_duration": total,
                        "exclusive_duration": exclusive_total,
                        "average_duration": total / calls if calls else 0.0,
                        "source": source,
                        "critical": any(
                            candidate["name"] == node["name"]
                            for candidate in nodes
                            if candidate["call_id"] in critical_ids
                        ),
                    }
                )
        edge_counts = {}
        by_id = {node["call_id"]: node for node in nodes}
        for node in nodes:
            parent = by_id.get(node["parent_id"])
            if parent is not None:
                edge = (parent["name"], node["name"])
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        edges = sorted(edge_counts)
        nodes = compact_nodes

    def fluid_positions():
        """Return a small deterministic force-directed layout for the graph."""
        names = [node["name"] for node in nodes]
        if not fluid or len(names) < 2:
            return None
        index = {name: i for i, name in enumerate(names)}
        points = np.column_stack(
            (
                np.cos(np.linspace(0, 2 * np.pi, len(names), endpoint=False)),
                np.sin(np.linspace(0, 2 * np.pi, len(names), endpoint=False)),
            )
        ).astype(float)
        graph_edges = [(index[parent], index[child]) for parent, child in edges]
        for step in range(120):
            delta = points[:, None, :] - points[None, :, :]
            distance = np.maximum(np.linalg.norm(delta, axis=2), 1e-3)
            force = (delta / distance[:, :, None] ** 2).sum(axis=1)
            for left, right in graph_edges:
                vector = points[right] - points[left]
                distance_edge = max(float(np.linalg.norm(vector)), 1e-3)
                pull = vector * (distance_edge - 0.8) / distance_edge
                force[left] += pull
                force[right] -= pull
            temperature = 0.08 * (1.0 - step / 120.0)
            points += np.clip(force, -temperature, temperature)
        points -= points.mean(axis=0)
        scale = max(float(np.abs(points).max()), 1e-6)
        points /= scale
        return {name: tuple(point) for name, point in zip(names, points)}

    fluid_layout = fluid_positions()

    if backend == "pyvis":
        try:
            from pyvis.network import Network
        except ImportError as exc:
            raise ImportError(
                "pyvis is required for the pyvis callgraph backend. "
                "Install scope-profiler[graph]."
            ) from exc
        graph = Network(height="800px", width="100%", directed=True)
        key = (lambda node: node["name"]) if compact else (lambda node: node["call_id"])
        node_keys = {key(node) for node in nodes}
        parents = {parent for parent, _ in (edges if compact else [])}
        max_total = max(
            (node.get("exclusive_duration", 0.0) for node in nodes), default=0.0
        )
        for node in nodes:
            node_key = key(node)
            label = node["name"] if compact else f"{node['name']} (#{node['call_id']})"
            if compact:
                name = node["name"]
                size = (
                    16 + 18 * (node["exclusive_duration"] / max_total) ** 0.5
                    if max_total
                    else 18
                )
                intensity = node["exclusive_duration"] / max_total if max_total else 0.0
                red = int(224 - 130 * intensity)
                green = int(242 - 150 * intensity)
                blue = int(255 - 25 * intensity)
                background = f"#{red:02x}{green:02x}{blue:02x}"
                border = "#dc2626" if node["critical"] else "#64748b"
                title = (
                    f"<b>{name}</b><br>Calls: {node['calls']}<br>"
                    f"Total: {node['total_duration']:.6g} s<br>"
                    f"Exclusive: {node['exclusive_duration']:.6g} s<br>"
                    f"Average: {node['average_duration']:.6g} s"
                    + ("<br><b>Critical path</b>" if node["critical"] else "")
                    + (f"<br>Source: {node['source']}" if node["source"] else "")
                )
                graph.add_node(
                    node_key,
                    label=label,
                    title=title,
                    level=node["depth"],
                    size=size,
                    borderWidth=4 if node["critical"] else 1,
                    color={
                        "background": background,
                        "border": border,
                        "highlight": {"background": "#fef08a", "border": "#b91c1c"},
                    },
                )
            else:
                graph.add_node(
                    node_key, label=label, title=node["name"], level=node["depth"]
                )
        graph_edges = (
            edges
            if compact
            else [
                (node["parent_id"], node["call_id"])
                for node in nodes
                if node["parent_id"] is not None
            ]
        )
        for parent, child in graph_edges:
            if parent in node_keys and child in node_keys:
                count = edge_counts.get((parent, child), 1) if compact else 1
                critical_edge = compact and all(
                    node["critical"]
                    for node in nodes
                    if node["name"] in {parent, child}
                )
                graph.add_edge(
                    parent,
                    child,
                    arrows="to",
                    label=f"×{count}" if compact and count > 1 else "",
                    title=f"{count} calls" if compact else "",
                    smooth=(
                        {"type": "curvedCW"} if parent == child else {"type": "dynamic"}
                    ),
                    color="#dc2626" if critical_edge else "#94a3b8",
                    width=3 if critical_edge else 1,
                )
        # Keep the call-depth structure legible while allowing nodes on the
        # same level to spread and settle horizontally like an Obsidian graph.
        graph.set_options(
            json.dumps(
                {
                    "layout": {
                        "hierarchical": {
                            "enabled": True,
                            "direction": "UD",
                            "sortMethod": "directed",
                            "levelSeparation": 140,
                            "nodeSpacing": 180,
                            "treeSpacing": 220,
                        }
                    },
                    "physics": {
                        "enabled": True,
                        "solver": "hierarchicalRepulsion",
                        "hierarchicalRepulsion": {
                            "nodeDistance": 180,
                            "centralGravity": 0.1,
                            "springLength": 140,
                            "springConstant": 0.01,
                            "avoidOverlap": 1,
                        },
                        "stabilization": {"iterations": 250},
                    },
                }
            )
        )
        if filepath:
            output_path = Path(filepath)
        elif show:
            import tempfile

            output_path = Path(tempfile.mkdtemp()) / "callgraph.html"
        else:
            output_path = None
        if output_path is not None:
            graph.write_html(str(output_path), open_browser=False)
            _add_pyvis_controls(output_path)
            if show:
                import webbrowser

                webbrowser.open(output_path.resolve().as_uri())
        return graph if return_fig else None
    if data_filepath:
        if compact:
            if data_format == "json":
                _write_json(
                    data_filepath,
                    {
                        "regions": nodes,
                        "edges": [
                            {"parent": parent, "child": child}
                            for parent, child in edges
                        ],
                    },
                )
            else:
                _write_csv(data_filepath, ["parent", "child"], edges)
        else:
            rows = [
                [node["call_id"], node["parent_id"], node["name"], node["depth"]]
                for node in nodes
            ]
            if data_format == "json":
                _write_json(
                    data_filepath,
                    {
                        "calls": [
                            dict(zip(("call_id", "parent_id", "name", "depth"), row))
                            for row in rows
                        ]
                    },
                )
            else:
                _write_csv(
                    data_filepath, ["call_id", "parent_id", "name", "depth"], rows
                )

    if backend == "plotly":
        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise ImportError("plotly is required for the callgraph plot") from exc
        figure = go.Figure()
        key = (lambda node: node["name"]) if compact else (lambda node: node["call_id"])
        positions = fluid_layout or {
            key(node): (index, -node["depth"]) for index, node in enumerate(nodes)
        }
        graph_edges = (
            edges
            if compact
            else [
                (node["parent_id"], node["call_id"])
                for node in nodes
                if node["parent_id"] is not None
            ]
        )
        for parent, child in graph_edges:
            if parent in positions and child in positions:
                x0, y0 = positions[parent]
                x1, y1 = positions[child]
                figure.add_trace(
                    go.Scatter(
                        x=[x0, x1],
                        y=[y0, y1],
                        mode="lines",
                        line={"color": "#999"},
                        showlegend=False,
                    )
                )
        figure.add_trace(
            go.Scatter(
                x=[positions[key(node)][0] for node in nodes],
                y=[positions[key(node)][1] for node in nodes],
                text=[
                    node["name"] if compact else f"{node['name']} ({node['call_id']})"
                    for node in nodes
                ],
                mode="markers+text",
                textposition="bottom center",
                showlegend=False,
            )
        )
        figure.update_layout(
            title=f"Call graph (rank {rank})", xaxis_visible=False, yaxis_visible=False
        )
        if filepath:
            figure.write_html(str(filepath))
        if show:
            figure.show()
        return figure if return_fig else None

    import matplotlib.pyplot as plt

    key = (lambda node: node["name"]) if compact else (lambda node: node["call_id"])
    positions = fluid_layout or {
        key(node): (index, -node["depth"]) for index, node in enumerate(nodes)
    }
    fig, axis = plt.subplots(
        figsize=(
            max(8, len(nodes) * 0.8),
            max(3, 2 + max(node["depth"] for node in nodes)),
        )
    )
    graph_edges = (
        edges
        if compact
        else [
            (node["parent_id"], node["call_id"])
            for node in nodes
            if node["parent_id"] is not None
        ]
    )
    for parent, child in graph_edges:
        if parent in positions and child in positions:
            x0, y0 = positions[parent]
            x1, y1 = positions[child]
            axis.plot([x0, x1], [y0, y1], color="#999999", linewidth=1, zorder=1)
    colors = _get_cmap_colors(cmap, max(1, len({node["name"] for node in nodes})))
    color_by_name = {
        name: colors[index % len(colors)]
        for index, name in enumerate(sorted({node["name"] for node in nodes}))
    }
    axis.scatter(
        [positions[key(node)][0] for node in nodes],
        [positions[key(node)][1] for node in nodes],
        c=[color_by_name[node["name"]] for node in nodes],
        s=140,
        zorder=2,
    )
    for node in nodes:
        x, y = positions[key(node)]
        label = node["name"] if compact else f"{node['name']}\n#{node['call_id']}"
        axis.text(x, y - 0.12, label, ha="center", va="top", fontsize=8)
    axis.set_title(f"Call graph (rank {rank})")
    axis.set_axis_off()
    fig.tight_layout()
    if filepath:
        fig.savefig(filepath, bbox_inches="tight")
    if show:
        plt.show()
    return (fig, axis) if return_fig else None


_DURATION_METRICS: dict[str, tuple[str, str]] = {
    "avg": ("average_duration_seconds", "Average duration per call (seconds)"),
    "min": ("min_duration_seconds", "Minimum duration per call (seconds)"),
    "max": ("max_duration_seconds", "Maximum duration per call (seconds)"),
    "total": ("total_duration_seconds", "Total duration (seconds)"),
}


def _pooled_metric_value(
    run: ProfilingResults,
    member_names: list[str],
    stat_key: str,
    ranks: list[int] | None = None,
) -> float:
    """Compute one duration statistic pooling several regions' calls together.

    Used both for ordinary bars (a single-element ``member_names``) and for
    combined bars (:func:`_group_regions`), where every call from every
    member region is pooled into one array before the statistic is taken --
    the same way it would be if the member regions' calls had all been
    recorded under one region name.
    """
    parts = [
        _region_duration_values(run.get_region(name), ranks=ranks)
        for name in member_names
    ]
    values = np.concatenate(parts) if parts else np.array([], dtype=float)
    stats = _stats_from_values(values)
    stat_value = stats[stat_key]
    return float("nan") if stat_value is None else stat_value


def _metric_filepath(filepath: str, metric_key: str, single_metric: bool) -> str:
    if single_metric:
        return filepath
    base, ext = os.path.splitext(filepath)
    return f"{base}_{metric_key}{ext}"


def _group_regions(
    region_names: list[str],
    combine_regions: dict[str, list[str] | str] | None,
) -> tuple[list[str], dict[str, list[str]]]:
    """Collapse regions matching a pattern into a single combined bar.

    ``combine_regions`` maps a display name (e.g. ``"setup"``) to one or more
    regex patterns (matched the same way as ``include``); every region in
    ``region_names`` matching one of a group's patterns is pooled into a
    single bar under that display name, in the position of its first match.
    A region matching patterns from more than one group is claimed by
    whichever group is listed first. Regions matching no group pass through
    unchanged, one bar each.

    Returns the ordered list of bar display names, plus a ``{display_name:
    [member region names]}`` map used to pool each bar's underlying data.
    """
    members: dict[str, list[str]] = {name: [name] for name in region_names}
    if not combine_regions:
        return list(region_names), members

    claimed: dict[str, str] = {}
    for group_name, patterns in combine_regions.items():
        matches = [
            name
            for name in region_names
            if name not in claimed and _name_selected(name, include=patterns)
        ]
        if not matches:
            raise ValueError(
                f"combine_regions group {group_name!r} matched no regions "
                f"(patterns: {patterns})."
            )
        for name in matches:
            claimed[name] = group_name
        members[group_name] = matches

    display_names = []
    seen_groups = set()
    for name in region_names:
        group_name = claimed.get(name)
        if group_name is None:
            display_names.append(name)
        elif group_name not in seen_groups:
            display_names.append(group_name)
            seen_groups.add(group_name)

    duplicates = {name for name in display_names if display_names.count(name) > 1}
    if duplicates:
        raise ValueError(
            f"combine_regions group name(s) {sorted(duplicates)} collide with "
            "an existing region name or another group; pick different names."
        )

    return display_names, members


def _sort_and_limit_region_names(
    region_names: list[str],
    runs: Sequence[ProfilingResults],
    ranks: list[int] | None,
    sort_by: str | None,
    top_n: int | None,
    members: dict[str, list[str]] | None = None,
) -> list[str]:
    """Order region names by a duration statistic and/or keep only the top N.

    ``sort_by`` picks the worst case across runs (the maximum of the chosen
    statistic over all the files being plotted together), so a multi-file
    comparison still sorts on "what's expensive anywhere", not just in the
    first file. ``sort_by="name"`` sorts alphabetically instead. Neither
    argument reorders anything when both are ``None``, which keeps the
    default the same natural (first-appearance) order as before.

    ``members`` maps each entry of ``region_names`` to the underlying region
    name(s) whose calls should be pooled for scoring (see
    :func:`_group_regions`); it defaults to each name mapping to itself.
    """
    if sort_by is None and top_n is None:
        return region_names
    if members is None:
        members = {name: [name] for name in region_names}

    if sort_by is None or sort_by == "name":
        ordered = sorted(region_names)
    else:
        if sort_by not in _DURATION_METRICS:
            raise ValueError(
                f"Unknown sort_by {sort_by!r}. Valid options are: "
                f"{['name', *_DURATION_METRICS]}"
            )
        stat_key, _ = _DURATION_METRICS[sort_by]

        def _score(name: str) -> float:
            values = [
                _pooled_metric_value(run, members[name], stat_key, ranks=ranks)
                for run in runs
            ]
            finite = [value for value in values if np.isfinite(value)]
            return max(finite) if finite else float("-inf")

        ordered = sorted(region_names, key=lambda name: (-_score(name), name))

    if top_n is not None:
        ordered = ordered[:top_n]
    return ordered


def plot_durations(
    profiling_data: ProfilingResults | Sequence[ProfilingResults],
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
    labels: Sequence[str] | None = None,
    metric: str = "total",
    sort_by: str | None = None,
    top_n: int | None = None,
    combine_regions: dict[str, list[str] | str] | None = None,
    filepath: str | None = None,
    show: bool = False,
    verbose: bool = True,
    cmap: str = DEFAULT_CMAP,
    log_scale: bool = False,
    data_filepath: str | Path | None = None,
    data_format: str = "csv",
    backend: str = "matplotlib",
    return_fig: bool = False,
) -> list[str] | list[object] | object:
    """Plot duration bar charts for one or more profiling files using maxplotlib.

    Parameters
    ----------
    combine_regions : dict[str, list[str] | str], optional
        Merge several regions into a single bar, e.g. ``{"setup": ["setup:
        .*"]}`` combines every ``setup: ...`` region into one bar named
        "setup", pooling their calls the same way ``sort_by`` and the other
        duration statistics pool a single region's calls. Each value is one
        or more regex patterns (matched like ``include``); a region matching
        several groups is claimed by whichever group is listed first.
    metric : str
        Duration metric to render (``avg``, ``min``, ``max`` or ``total``).
    backend : str
        Backend to use for rendering: "matplotlib" (default) or "plotly".
    return_fig : bool
        Return the rendered figure instead of the saved filepath list.

    Returns
    -------
    list[str]
        List of filepaths that were written (empty if filepath is None).
    """
    Canvas = _get_canvas()
    runs = _as_runs(profiling_data)
    if not runs:
        # Not this rank's job; rank 0 draws it.
        return []
    ranks = _normalize_ranks(ranks)

    metric_keys = [metric]

    unknown_metrics = [key for key in metric_keys if key not in _DURATION_METRICS]
    if unknown_metrics:
        raise ValueError(
            f"Unknown metric(s) {unknown_metrics}. "
            f"Valid options are: {list(_DURATION_METRICS)}"
        )

    if labels is None:
        labels = _unique_labels([run.display_label for run in runs])
    else:
        labels = list(labels)

    if len(labels) != len(runs):
        raise ValueError("labels must match the number of profiling files.")

    region_names = _common_region_names(runs, include=include, exclude=exclude)
    if not region_names:
        raise ValueError("No regions matched the selected filters.")
    region_names, region_members = _group_regions(region_names, combine_regions)
    region_names = _sort_and_limit_region_names(
        region_names, runs, ranks, sort_by, top_n, members=region_members
    )

    if verbose:
        print(
            f"Plotting duration comparison ({', '.join(metric_keys)}) "
            f"for files: {', '.join(labels)}"
        )

    num_readers = len(runs)
    colors = _get_cmap_colors(cmap, max(num_readers, 1))
    fig_width = max(10, 0.85 * len(region_names) + 2)
    # Angled tick labels consume space below the axes in proportion to the
    # longest region name. Grow both the figure and its bottom margin so long
    # labels remain inside the exported figure without reserving excessive
    # space for short names.
    label_space = max(0.8, 0.06 * max(map(len, region_names), default=0) + 0.25)
    fig_height = max(4.5, 2.5 + 0.35 * num_readers, 3.0 + label_space)
    bottom_margin = (label_space + 0.25) / fig_height
    width = min(0.8 / max(num_readers, 1), 0.35)

    saved_paths: list[str] = []
    rendered_figures: list[object] = []
    data_rows = []

    for metric_key in metric_keys:
        stat_key, ylabel = _DURATION_METRICS[metric_key]

        values = [
            [
                _pooled_metric_value(
                    run, region_members[region_name], stat_key, ranks=ranks
                )
                for region_name in region_names
            ]
            for run in runs
        ]

        if data_filepath:
            for label, file_values in zip(labels, values):
                for region_name, value in zip(region_names, file_values):
                    data_rows.append([label, region_name, metric_key, value])

        canvas = Canvas(
            figsize=(fig_width, fig_height),
            gridspec_kw={"bottom": bottom_margin},
        )

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

        tick_rotation_applied = _set_xticks(
            canvas,
            x_positions,
            labels=region_names,
            rotation=45,
            ha="right",
        )
        canvas.set_ylabel(ylabel)
        canvas.set_title(f"Region duration comparison ({metric_key})")
        canvas.set_grid(True)
        if log_scale:
            canvas.set_yscale("log")
        if num_readers > 1:
            canvas.set_legend()

        metric_filepath = None
        if filepath:
            metric_filepath = _metric_filepath(
                filepath, metric_key, single_metric=len(metric_keys) == 1
            )
            saved_paths.append(metric_filepath)

        rendered = _render(
            canvas,
            metric_filepath,
            show,
            backend,
            x_tick_rotation=None if tick_rotation_applied else 45,
            return_fig=return_fig,
        )
        if return_fig:
            rendered_figures.append(rendered)

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

    if return_fig:
        return rendered_figures[0]
    return saved_paths


def _duration_timeseries(
    region,
    ranks: list[int] | None,
    first_start_time: float,
) -> dict[str, np.ndarray] | None:
    """Aggregate one region's calls over ranks into a duration-versus-time series.

    Calls are matched across ranks by call index, so point ``i`` describes the
    i-th call of the region on every rank that got that far. Ranks that stopped
    calling the region earlier simply drop out of the later points, which keeps
    ragged call counts (e.g. a rank-dependent number of iterations) usable.

    Returns ``None`` if no selected rank recorded any call.
    """
    if ranks is None:
        selected_ranks = sorted(region.regions)
    else:
        selected_ranks = [rank for rank in ranks if rank in region.regions]

    per_rank = [
        (rank, region.regions[rank])
        for rank in selected_ranks
        if region.regions[rank].durations.size
    ]
    if not per_rank:
        return None

    max_calls = max(len(region_data.durations) for _, region_data in per_rank)
    times, means, minima, maxima, counts = [], [], [], [], []
    for index in range(max_calls):
        starts = [
            float(region_data.start_times[index]) - first_start_time
            for _, region_data in per_rank
            if len(region_data.durations) > index
        ]
        durations = [
            float(region_data.durations[index])
            for _, region_data in per_rank
            if len(region_data.durations) > index
        ]
        times.append(float(np.mean(starts)))
        means.append(float(np.mean(durations)))
        minima.append(float(np.min(durations)))
        maxima.append(float(np.max(durations)))
        counts.append(len(durations))

    return {
        "time": np.asarray(times, dtype=float),
        "mean": np.asarray(means, dtype=float),
        "min": np.asarray(minima, dtype=float),
        "max": np.asarray(maxima, dtype=float),
        "num_ranks": np.asarray(counts, dtype=int),
        "ranks": np.asarray([rank for rank, _ in per_rank], dtype=int),
    }


def plot_duration_timeseries(
    profiling_data: ProfilingResults | Sequence[ProfilingResults],
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
    filepath: str | None = None,
    show: bool = False,
    verbose: bool = True,
    cmap: str = DEFAULT_CMAP,
    log_scale: bool = False,
    data_filepath: str | Path | None = None,
    data_format: str = "csv",
    backend: str = "matplotlib",
    return_fig: bool = False,
) -> None:
    """Plot each region's call duration over wall-clock time, with a min-max band.

    One line per region tracks the mean duration over the ranks that recorded
    each call, shaded between the minimum and maximum duration seen across
    those ranks, so rank imbalance shows up as a widening band.

    Parameters
    ----------
    backend : str
        Backend to use for rendering: "matplotlib" (default) or "plotly".
    return_fig : bool
        Return the rendered figure instead of the default ``None``. Matplotlib
        returns ``(fig, axes)``; Plotly returns its figure object.
    """
    Canvas = _get_canvas()
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
        # As in _prepare_gantt_data: charts are framed on the first entry.
        first_start_time = run.minimum_start_time
        series = [
            (
                region.name,
                _duration_timeseries(region, normalized_ranks, first_start_time),
            )
            for region in regions
        ]
        series = [(name, values) for name, values in series if values is not None]
        if series:
            prepared.append((run, series))

    if not prepared:
        raise ValueError("No calls recorded for the requested ranks.")

    labels = _unique_labels([run.display_label for run, _ in prepared])

    if data_filepath:
        records = []
        for label, (_, series) in zip(labels, prepared):
            for region_name, values in series:
                for index in range(values["time"].size):
                    records.append(
                        [
                            label,
                            region_name,
                            index,
                            float(values["time"][index]),
                            float(values["mean"][index]),
                            float(values["min"][index]),
                            float(values["max"][index]),
                            int(values["num_ranks"][index]),
                        ]
                    )
        header = [
            "file",
            "region",
            "call_index",
            "time_seconds",
            "mean_duration_seconds",
            "min_duration_seconds",
            "max_duration_seconds",
            "num_ranks",
        ]
        if data_format == "json":
            points = [dict(zip(header, record)) for record in records]
            colors_map = {
                name: _to_hex(color) for name, color in sorted(color_map.items())
            }
            _write_json(data_filepath, {"points": points, "colors": colors_map})
        else:
            _write_csv(data_filepath, header, records)

    if verbose:
        print("Plotting duration over time for files: " + ", ".join(labels))

    single_panel = len(prepared) == 1
    fig_width, fig_height = 12.0, 1.0 + 4.0 * len(prepared)
    canvas = Canvas(
        nrows=len(prepared),
        ncols=1,
        figsize=(fig_width, fig_height),
        # Duration tick labels plus the "Duration (seconds)" axis label.
        gridspec_kw=_panel_gridspec(fig_width, fig_height, 12, not single_panel),
    )

    for idx, (run, series) in enumerate(prepared):
        row = None if single_panel else idx
        col = None if single_panel else 0

        for region_name, values in series:
            color = _to_hex(color_map[region_name])
            canvas.fill_between(
                values["time"],
                values["min"],
                values["max"],
                row=row,
                col=col,
                color=color,
                alpha=0.25,
            )
            canvas.add_line(
                values["time"],
                values["mean"],
                row=row,
                col=col,
                linewidth=1.8,
                color=color,
                label=region_name,
            )

        canvas.set_xlabel("Time (seconds)", row=row, col=col)
        canvas.set_ylabel("Duration per call (seconds)", row=row, col=col)
        canvas.set_title(
            "Region duration over time" if single_panel else run.display_label,
            row=row,
            col=col,
        )
        canvas.set_grid(True, row=row, col=col)
        canvas.set_legend(row=row, col=col)
        if log_scale:
            canvas.set_yscale("log", row=row, col=col)

    if not single_panel:
        canvas.suptitle("Region duration over time")

    rendered = _render(canvas, filepath, show, backend, return_fig=return_fig)
    return rendered if return_fig else None


def plot_speedup(
    profiling_data: ProfilingResults | Sequence[ProfilingResults],
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
    return_fig: bool = False,
) -> object | None:
    """Plot scope speedup versus a chosen parallelism/metadata field using maxplotlib.

    Parameters
    ----------
    backend : str
        Backend to use for rendering: "matplotlib" (default) or "plotly".
    """
    Canvas = _get_canvas()
    runs = _as_runs(profiling_data)
    if not runs:
        # Not this rank's job; rank 0 draws it.
        return
    if len(runs) < 2:
        raise ValueError("Speedup plot requires at least two profiling files.")

    region_names = _common_region_names(runs, include=include, exclude=exclude)
    if not region_names:
        raise ValueError("No regions matched the selected filters.")

    is_scaling = x_field in _SCALING_X_FIELDS
    x_per_reader = [_speedup_x_value(run, x_field) for run in runs]

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
    for run, x_value in zip(runs, x_per_reader):
        for region_name in region_names:
            duration = _region_average_duration(
                run.get_region(region_name),
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

    rendered = _render(canvas, filepath, show, backend, return_fig=return_fig)
    return rendered if return_fig else None


def plot_weak_scaling(
    profiling_data: ProfilingResults | Sequence[ProfilingResults],
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
    return_fig: bool = False,
) -> object | None:
    """Plot weak-scaling runtime versus a chosen parallelism/metadata field.

    Runtime is normalized to the smallest scale, so ideal weak scaling is a
    horizontal line at 1.0. Lower values are not inherently better here: the
    useful signal is how closely each region stays near that line.
    """
    Canvas = _get_canvas()
    runs = _as_runs(profiling_data)
    if not runs:
        return
    if len(runs) < 2:
        raise ValueError("Weak scaling plot requires at least two profiling files.")

    region_names = _common_region_names(runs, include=include, exclude=exclude)
    if not region_names:
        raise ValueError("No regions matched the selected filters.")

    is_scaling = x_field in _SCALING_X_FIELDS
    x_per_reader = [_speedup_x_value(run, x_field) for run in runs]
    x_keys = (
        sorted({int(value) for value in x_per_reader})
        if is_scaling
        else list(dict.fromkeys(x_per_reader))
    )

    if verbose:
        print(
            f"Plotting weak scaling comparison using x_field={x_field!r}, values: "
            + ", ".join(map(str, x_keys))
        )

    duration_samples: dict[str, dict] = {
        region_name: defaultdict(list) for region_name in region_names
    }
    for run, x_value in zip(runs, x_per_reader):
        for region_name in region_names:
            duration = _region_average_duration(
                run.get_region(region_name), ranks=ranks
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
        runtimes = []
        for key in x_keys:
            samples = region_values.get(key, [])
            if not samples:
                continue
            mean_duration = float(np.mean(samples))
            if not np.isfinite(mean_duration) or mean_duration <= 0:
                continue
            plot_x.append(x_position[key])
            plot_keys.append(key)
            runtimes.append(mean_duration / baseline_duration)

        if not plot_x:
            continue
        plotted += 1
        canvas.add_line(
            plot_x,
            runtimes,
            linewidth=1.8,
            color=_to_hex(colors[idx]),
            label=region_name,
        )
        if data_filepath:
            for key, runtime in zip(plot_keys, runtimes):
                data_rows.append([region_name, key, runtime])

    if plotted == 0:
        raise ValueError("No valid weak-scaling data could be computed.")

    if data_filepath:
        if data_format == "json":
            points = [
                {"region": region, x_field: key, "normalized_runtime": runtime}
                for region, key, runtime in data_rows
            ]
            colors_map = {
                name: _to_hex(color) for name, color in zip(region_names, colors)
            }
            _write_json(data_filepath, {"points": points, "colors": colors_map})
        else:
            _write_csv(
                data_filepath, ["region", x_field, "normalized_runtime"], data_rows
            )

    x_label_map = {
        "num_ranks": "MPI ranks",
        "omp_num_threads": "OpenMP threads",
        "total_cores": "MPI ranks × OpenMP threads",
    }
    x_label = x_label_map.get(x_field, x_field)
    if is_scaling:
        x_line = np.array(x_keys, dtype=float)
        canvas.set_xticks(x_line)
    else:
        canvas.set_xticks(list(range(len(x_keys))), labels=[str(key) for key in x_keys])
    canvas.add_line(
        [x_position[key] for key in x_keys],
        [1.0] * len(x_keys),
        linestyle="--",
        color="black",
        linewidth=1.5,
        label="Ideal weak scaling",
    )
    canvas.set_xlabel(x_label)
    canvas.set_ylabel("Normalized runtime")
    canvas.set_title(f"Weak scaling (baseline: {x_label} = {baseline_key})")
    canvas.set_grid(True)
    canvas.set_legend()

    rendered = _render(canvas, filepath, show, backend, return_fig=return_fig)
    return rendered if return_fig else None


def plot_rank_heatmap(
    profiling_data: ProfilingResults | Sequence[ProfilingResults],
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
    filepath: str | None = None,
    show: bool = False,
    verbose: bool = True,
    cmap: str = "viridis",
    data_filepath: str | Path | None = None,
    data_format: str = "csv",
    backend: str = "matplotlib",
    return_fig: bool = False,
) -> object | None:
    """Plot total region time as a rank-by-region heatmap."""
    Canvas = _get_canvas()
    runs = _as_runs(profiling_data)
    if not runs:
        return

    normalized_ranks = _normalize_ranks(ranks)
    prepared = []
    all_records = []
    for run in runs:
        regions = run.get_regions(include=include, exclude=exclude)
        if not regions:
            raise ValueError("No regions matched the selected filters.")
        selected_ranks = (
            normalized_ranks
            if normalized_ranks is not None
            else list(range(run.num_ranks))
        )
        invalid = [rank for rank in selected_ranks if rank < 0 or rank >= run.num_ranks]
        if invalid:
            raise ValueError(f"Invalid ranks requested: {invalid}")
        region_names = [region.name for region in regions]
        matrix = np.zeros((len(selected_ranks), len(region_names)), dtype=float)
        for col, region in enumerate(regions):
            for row, rank in enumerate(selected_ranks):
                if rank in region and region[rank].durations.size:
                    matrix[row, col] = float(np.sum(region[rank].durations))
                all_records.append(
                    [run.display_label, rank, region.name, matrix[row, col]]
                )
        prepared.append((run, selected_ranks, region_names, matrix))

    if verbose:
        print("Plotting rank × region duration heatmap")

    if data_filepath:
        header = ["file", "rank", "region", "total_duration_seconds"]
        if data_format == "json":
            _write_json(
                data_filepath,
                {"points": [dict(zip(header, record)) for record in all_records]},
            )
        else:
            _write_csv(data_filepath, header, all_records)

    fig_width = max(10.0, 0.8 * max(len(item[2]) for item in prepared) + 3.0)
    fig_height = max(3.5, 1.0 + 0.35 * sum(len(item[1]) for item in prepared))
    canvas = Canvas(
        nrows=len(prepared),
        ncols=1,
        figsize=(fig_width, fig_height),
        gridspec_kw={"right": 0.86},
    )
    single_panel = len(prepared) == 1
    for index, (run, selected_ranks, region_names, matrix) in enumerate(prepared):
        row = None if single_panel else index
        col = None if single_panel else 0
        canvas.imshow(matrix, cmap=cmap, aspect="auto", row=row, col=col)
        canvas.set_xticks(
            list(range(len(region_names))), labels=region_names, row=row, col=col
        )
        canvas.set_yticks(
            list(range(len(selected_ranks))),
            labels=[str(rank) for rank in selected_ranks],
            row=row,
            col=col,
        )
        canvas.set_xlabel("Region", row=row, col=col)
        canvas.set_ylabel("MPI rank", row=row, col=col)
        canvas.set_title(run.display_label, row=row, col=col)
        canvas.colorbar("Total duration (seconds)", row=row, col=col)

    if not single_panel:
        canvas.suptitle("Rank × region duration")
    rendered = _render(canvas, filepath, show, backend, return_fig=return_fig)
    return rendered if return_fig else None


def plot_scaling_efficiency(
    profiling_data: ProfilingResults | Sequence[ProfilingResults],
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
    return_fig: bool = False,
) -> object | None:
    """Plot parallel scaling efficiency (measured speedup / ideal speedup)."""
    Canvas = _get_canvas()
    runs = _as_runs(profiling_data)
    if not runs:
        return
    if len(runs) < 2:
        raise ValueError("Scaling efficiency requires at least two profiling files.")
    if x_field not in _SCALING_X_FIELDS:
        raise ValueError(
            "Scaling efficiency requires x_field to be one of: "
            + ", ".join(sorted(_SCALING_X_FIELDS))
        )

    region_names = _common_region_names(runs, include=include, exclude=exclude)
    if not region_names:
        raise ValueError("No regions matched the selected filters.")
    x_per_reader = [_speedup_x_value(run, x_field) for run in runs]
    x_keys = sorted({int(value) for value in x_per_reader})
    baseline_key = x_keys[0]
    if baseline_key <= 0:
        raise ValueError("Scaling x-axis values must be positive.")
    x_position = {key: key for key in x_keys}
    colors = _get_cmap_colors(cmap, len(region_names))
    samples = {name: defaultdict(list) for name in region_names}
    for run, x_value in zip(runs, x_per_reader):
        for name in region_names:
            duration = _region_average_duration(run.get_region(name), ranks=ranks)
            if np.isfinite(duration) and duration > 0:
                samples[name][x_value].append(duration)

    canvas = Canvas(
        figsize=(
            max(10, 1.2 * len(x_keys) + 3),
            max(4.5, 2.8 + 0.35 * len(region_names)),
        )
    )
    data_rows = []
    plotted = 0
    for index, name in enumerate(region_names):
        baseline_values = samples[name].get(baseline_key, [])
        if not baseline_values:
            continue
        baseline_duration = float(np.mean(baseline_values))
        plot_x, efficiencies = [], []
        for key in x_keys:
            values = samples[name].get(key, [])
            if not values:
                continue
            duration = float(np.mean(values))
            plot_x.append(x_position[key])
            efficiencies.append((baseline_duration / duration) / (key / baseline_key))
            data_rows.append([name, key, efficiencies[-1]])
        if plot_x:
            plotted += 1
            canvas.add_line(
                plot_x,
                efficiencies,
                linewidth=1.8,
                color=_to_hex(colors[index]),
                label=name,
            )
    if not plotted:
        raise ValueError("No valid scaling-efficiency data could be computed.")

    if data_filepath:
        header = ["region", x_field, "efficiency"]
        if data_format == "json":
            _write_json(
                data_filepath,
                {"points": [dict(zip(header, row)) for row in data_rows]},
            )
        else:
            _write_csv(data_filepath, header, data_rows)

    canvas.set_xticks(x_keys)
    canvas.add_line(
        x_keys,
        [1.0] * len(x_keys),
        linestyle="--",
        color="black",
        linewidth=1.5,
        label="Ideal efficiency",
    )
    x_label = {
        "num_ranks": "MPI ranks",
        "omp_num_threads": "OpenMP threads",
        "total_cores": "MPI ranks × OpenMP threads",
    }[x_field]
    canvas.set_xlabel(x_label)
    canvas.set_ylabel("Scaling efficiency")
    canvas.set_title(f"Scaling efficiency (baseline: {x_label} = {baseline_key})")
    canvas.set_ylim(0, 1.05)
    canvas.set_grid(True)
    canvas.set_legend()
    rendered = _render(canvas, filepath, show, backend, return_fig=return_fig)
    return rendered if return_fig else None


def plot_imbalance(
    profiling_data: ProfilingResults | Sequence[ProfilingResults],
    metric: str = "total",
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
    filepath: str | None = None,
    show: bool = False,
    verbose: bool = True,
    cmap: str = DEFAULT_CMAP,
    log_scale: bool = False,
    data_filepath: str | Path | None = None,
    data_format: str = "csv",
    backend: str = "matplotlib",
) -> None:
    """Plot each region's duration statistic per rank, to surface load imbalance.

    One point per (region, rank), connected by a line in rank order, with a
    dashed horizontal line at the mean over ranks -- so a straggler rank shows
    up as a point sitting well off its region's line. This is a per-rank view
    of the same statistics :func:`plot_durations` aggregates across ranks.

    Parameters
    ----------
    metric : str
        Which per-call duration statistic to plot per rank; one of
        ``"avg"``, ``"min"``, ``"max"``, ``"total"`` (default: ``"total"``,
        the total time a rank spent in the region).
    backend : str
        Backend to use for rendering: "matplotlib" (default) or "plotly".
    """
    Canvas = _get_canvas()
    runs = _as_runs(profiling_data)
    if not runs:
        # Not this rank's job; rank 0 draws it.
        return

    if metric not in _DURATION_METRICS:
        raise ValueError(
            f"Unknown metric {metric!r}. Valid options are: {list(_DURATION_METRICS)}"
        )
    stat_key, metric_label = _DURATION_METRICS[metric]

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
        available_ranks = (
            normalized_ranks
            if normalized_ranks is not None
            else list(range(run.num_ranks))
        )
        series = []
        for region in regions:
            rank_list: list[int] = []
            value_list: list[float] = []
            for rank in sorted(available_ranks):
                if rank not in region:
                    continue
                durations = region[rank].durations
                if not durations.size:
                    continue
                value = _stats_from_values(durations)[stat_key]
                if value is None:
                    continue
                rank_list.append(rank)
                value_list.append(value)
            if rank_list:
                series.append(
                    (
                        region.name,
                        np.asarray(rank_list, dtype=int),
                        np.asarray(value_list, dtype=float),
                    )
                )
        if series:
            prepared.append((run, series))

    if not prepared:
        raise ValueError("No calls recorded for the requested ranks.")

    labels = _unique_labels([run.display_label for run, _ in prepared])

    if data_filepath:
        records = []
        for label, (_, series) in zip(labels, prepared):
            for region_name, region_ranks, values in series:
                mean_value = float(np.mean(values))
                for rank, value in zip(region_ranks, values):
                    records.append(
                        [label, region_name, int(rank), float(value), mean_value]
                    )
        header = ["file", "region", "rank", "value_seconds", "mean_over_ranks_seconds"]
        if data_format == "json":
            points = [dict(zip(header, record)) for record in records]
            colors_map = {
                name: _to_hex(color) for name, color in sorted(color_map.items())
            }
            _write_json(
                data_filepath,
                {"metric": metric, "points": points, "colors": colors_map},
            )
        else:
            _write_csv(data_filepath, header, records)

    if verbose:
        print(f"Plotting per-rank imbalance ({metric}) for files: " + ", ".join(labels))

    single_panel = len(prepared) == 1
    fig_width, fig_height = 12.0, 1.0 + 4.0 * len(prepared)
    canvas = Canvas(
        nrows=len(prepared),
        ncols=1,
        figsize=(fig_width, fig_height),
        gridspec_kw=_panel_gridspec(fig_width, fig_height, 10, not single_panel),
    )

    for idx, (run, series) in enumerate(prepared):
        row = None if single_panel else idx
        col = None if single_panel else 0

        for region_name, region_ranks, values in series:
            color = _to_hex(color_map[region_name])
            canvas.add_line(
                region_ranks,
                values,
                row=row,
                col=col,
                linewidth=1.4,
                color=color,
                label=region_name,
            )
            canvas.scatter(
                region_ranks,
                values,
                row=row,
                col=col,
                color=color,
            )
            canvas.axhline(
                float(np.mean(values)),
                row=row,
                col=col,
                linestyle="--",
                linewidth=1.0,
                color=color,
                alpha=0.5,
            )

        canvas.set_xlabel("Rank", row=row, col=col)
        canvas.set_ylabel(metric_label, row=row, col=col)
        canvas.set_title(
            "Per-rank load imbalance" if single_panel else run.display_label,
            row=row,
            col=col,
        )
        canvas.set_grid(True, row=row, col=col)
        canvas.set_legend(row=row, col=col)
        if log_scale:
            canvas.set_yscale("log", row=row, col=col)

    if not single_panel:
        canvas.suptitle("Per-rank load imbalance")

    _render(canvas, filepath, show, backend)


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
    Canvas = _get_canvas()
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
    for idx, (run, series) in enumerate(prepared):
        row = None if single_panel else idx
        col = None if single_panel else 0

        all_values = np.concatenate([values for _, values in series])
        edges = np.histogram_bin_edges(all_values, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])

        for region_name, values in series:
            counts, _ = np.histogram(values, bins=edges)
            color = _to_hex(color_map[region_name])
            canvas.add_line(
                centers,
                counts,
                row=row,
                col=col,
                linewidth=1.6,
                color=color,
                label=region_name,
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

    _render(canvas, filepath, show, backend)


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
    Canvas = _get_canvas()
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
                series.append((label, values))

    if not series:
        raise ValueError(
            f"No LIKWID data found for metric {metric!r} with the requested "
            "ranks/files."
        )

    region_names = sorted(
        {
            tag
            for _, values in series
            for tag in values
            if _name_selected(tag, include, exclude)
        }
    )
    if not region_names:
        raise ValueError("No regions matched the selected filters.")

    if labels is None:
        series_labels = _unique_labels([label for label, _ in series])
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
    for idx, (label, values) in enumerate(zip(series_labels, (v for _, v in series))):
        bar_values = [values.get(name, float("nan")) for name in region_names]
        offsets = x_positions + offset_start + idx * width
        canvas.bar(
            offsets,
            bar_values,
            width=width,
            label=label if num_series > 1 else None,
            color=_to_hex(colors[idx]),
            edgecolor="black",
            alpha=0.8,
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

    _render(canvas, filepath, show, backend)
