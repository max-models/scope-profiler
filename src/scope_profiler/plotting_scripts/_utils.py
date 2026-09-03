"""Shared helpers for the ``scope_profiler.plotting_scripts`` package.

Canvas/render plumbing, hover-text formatting, and small normalization
helpers used across the individual chart modules.
"""

import csv
import json
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from scope_profiler.call_stack import NestingError
from scope_profiler.results import ProfilingResults

PLOT_DATA_FORMAT = "scope-profiler-plot-data"
PLOT_DATA_FORMAT_VERSION = 1


def _write_csv(
    filepath: str | Path,
    header: Sequence[str],
    rows: Sequence[Sequence],
) -> None:
    """Write rows of plotting data to a plain-text CSV file."""
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _write_json(filepath: str | Path, payload: dict, plot: str) -> None:
    """Write the exact data behind a plot to a JSON file.

    Every plot-data document carries the same envelope -- ``format``,
    ``format_version`` and the ``plot`` kind that produced it -- so a consumer
    (e.g. the ``@scope-profiler/plotly`` package) can dispatch on the file
    itself instead of guessing from its keys. Stamping it here rather than at
    each call site is what keeps the envelope on every kind.
    """
    document = {
        "format": PLOT_DATA_FORMAT,
        "format_version": PLOT_DATA_FORMAT_VERSION,
        "plot": plot,
        **{
            key: value
            for key, value in payload.items()
            if key not in ("format", "format_version", "plot")
        },
    }
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2)
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
            "or maxplotlibx (>= 0.1.9, for its gantt/flame charts and native "
            "hover support).",
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


# Statistics from ``get_summary()`` are shown under these hover labels, in
# this order; anything the method grows later still appears, under its own
# key, so the hover box stays as complete as the method it reads.
_HOVER_LABELS: dict[str, str] = {
    "num_ranks": "ranks",
    "num_calls": "calls",
    "total_duration": "total",
    "exclusive_duration": "self",
    "average_duration": "avg",
    "min_duration": "min",
    "max_duration": "max",
    "std_duration": "std",
    "first_duration": "first",
    "last_duration": "last",
    "gpu_total_duration": "GPU total",
    "gpu_average_duration": "GPU avg",
}

# ``name`` is the hover box's heading, and ``inclusive_duration`` is an alias
# of ``total_duration`` -- neither earns a line of its own.
_HOVER_SKIP = frozenset({"name", "inclusive_duration"})


def _hover_value(key: str, value) -> str:
    """Format one ``get_summary()`` entry for the hover box."""
    if value is None:
        return "-"
    if isinstance(value, (int, np.integer)) and "duration" not in key:
        return str(int(value))
    return f"{float(value):.6g} s" if "duration" in key else str(value)


def _region_summary(region) -> dict:
    """A region's ``get_summary()``, minus what a broken call graph withholds.

    ``get_summary()`` reports exclusive time, which is reconstructed from
    timestamp containment and therefore unavailable on a rank whose calls
    are not properly nested. That is a reason to leave one line out of a
    hover box, not to fail the plot.
    """
    try:
        return region.get_summary()
    except NestingError:
        summary = {
            "num_calls": region.num_calls,
            "total_duration": region.total_duration,
            "average_duration": region.average_duration,
            "min_duration": region.min_duration,
            "max_duration": region.max_duration,
            "std_duration": region.std_duration,
        }
        if hasattr(region, "ranks"):
            summary = {"num_ranks": len(region.ranks), **summary}
        return summary


def _hover_summary(
    region,
    title: str | None = None,
    extra: Sequence[tuple[str, object]] = (),
) -> str:
    """Render a region's own ``get_summary()`` as Plotly hover text.

    Both :class:`~scope_profiler.region.Region` (one rank) and
    :class:`~scope_profiler.mpi_region.MPIRegion` (pooled over ranks) expose
    ``get_summary()``, so hovering shows exactly the statistics that region
    object reports -- there is no second, parallel definition of "the usual
    information" to keep in step with it.

    ``extra`` lines (the hovered call's own start and duration, a bar's
    value, ...) are placed above the summary, since they are what identifies
    the point being hovered.
    """
    summary = _region_summary(region)
    lines = []
    heading = title if title is not None else summary.get("name")
    if heading:
        lines.append(f"<b>{heading}</b>")
    lines.extend(f"{label}: {value}" for label, value in extra)
    for key, value in summary.items():
        if key in _HOVER_SKIP:
            continue
        lines.append(
            f"{_HOVER_LABELS.get(key, key.replace('_', ' '))}: "
            f"{_hover_value(key, value)}",
        )
    return "<br>".join(lines)


def _hover_region(region, ranks: list[int] | None):
    """Pick the region object whose summary matches what is being plotted.

    A plot restricted to a single rank should not describe itself with
    statistics pooled over every rank, and ``MPIRegion[rank]`` is the same
    ``get_summary()`` one level down.
    """
    if ranks is not None and len(ranks) == 1 and ranks[0] in region:
        return region[ranks[0]], f"{region.name} (rank {ranks[0]})"
    return region, region.name


def _save_matplotlib_figure(fig, canvas, filepath: str | Path) -> None:
    """Save a materialized Matplotlib figure using canvas-level DPI if present."""
    savefig_kwargs = {}
    if getattr(canvas, "dpi", None) is not None:
        savefig_kwargs["dpi"] = canvas.dpi
    fig.savefig(filepath, **savefig_kwargs)


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


def _show_matplotlib_figure(fig) -> bool:
    """Display a Matplotlib figure in notebooks and regular Python sessions."""
    if _display_matplotlib_figure_in_notebook(fig):
        return True

    import matplotlib.pyplot as plt

    plt.show()
    return False


def _close_matplotlib_figure(canvas, fig=None) -> None:
    """Release the figure maxplotlib keeps open after rendering."""
    fig = fig if fig is not None else getattr(canvas, "_matplotlib_fig", None)
    if fig is None:
        return
    import matplotlib.pyplot as plt

    plt.close(fig)


def _render(
    canvas,
    filepath: str | None,
    show: bool,
    backend: str,
    plotly_layout: dict | None = None,
    x_tick_rotation: float | None = None,
    return_fig: bool = False,
    matplotlib_postprocess=None,
    plotly_postprocess=None,
) -> tuple[object, object] | object:
    """Save and/or display a canvas."""
    if backend == "plotly":
        fig = canvas.plot_plotly(show=False)
        if plotly_postprocess is not None:
            plotly_postprocess(fig)
        layout = dict(plotly_layout or {})
        # Multi-line hover summaries (drawn via Canvas ``hover=`` kwargs)
        # read as a block, not centred prose.
        layout.setdefault("hoverlabel", {"align": "left"})
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
                        "export to an .html filepath instead.",
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
        fig, axes = canvas.render(backend="matplotlib", savefig=bool(filepath))
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
    fig_width: float,
    fig_height: float,
    label_chars: int,
    multi_panel: bool,
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
