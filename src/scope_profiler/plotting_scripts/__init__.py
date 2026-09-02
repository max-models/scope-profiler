"""Plotting utilities for visualizing profiling data using maxplotlib.

This package provides plotting functions that can export to both matplotlib
and plotly backends using maxplotlib as the unified interface. It is split
into one module per chart type; everything is re-exported here so
``scope_profiler.plotting_scripts`` keeps behaving like a single module for
callers and tests (including attribute-level monkeypatching).
"""

from scope_profiler.plotting_scripts._utils import (
    DEFAULT_CMAP,
    _add_pyvis_controls,
    _as_runs,
    _close_matplotlib_figure,
    _display_matplotlib_figure_in_notebook,
    _filename_slug,
    _get_canvas,
    _get_cmap_colors,
    _hover_region,
    _hover_summary,
    _hover_value,
    _normalize_ranks,
    _panel_gridspec,
    _region_color_map,
    _region_summary,
    _render,
    _save_matplotlib_figure,
    _set_xticks,
    _show_matplotlib_figure,
    _to_hex,
    _unique_labels,
    _write_csv,
    _write_json,
)
from scope_profiler.plotting_scripts.callgraph import plot_callgraph
from scope_profiler.plotting_scripts.duration_timeseries import (
    _duration_timeseries,
    plot_duration_timeseries,
)
from scope_profiler.plotting_scripts.durations import (
    _group_regions,
    _stacked_segments,
    plot_durations,
)
from scope_profiler.plotting_scripts.flame import (
    FLAME_CMAP,
    plot_flame,
    plot_flame_chart,
    plot_flame_graph,
)
from scope_profiler.plotting_scripts.gantt import (
    _aggregate_gantt_intervals,
    _prepare_gantt_data,
    plot_gantt,
)
from scope_profiler.plotting_scripts.heatmap import plot_rank_heatmap
from scope_profiler.plotting_scripts.histogram import plot_duration_histogram
from scope_profiler.plotting_scripts.imbalance import plot_imbalance
from scope_profiler.plotting_scripts.likwid import (
    available_likwid_metrics,
    plot_likwid,
)
from scope_profiler.plotting_scripts.perf_events import (
    available_perf_event_metrics,
    plot_perf_events,
)
from scope_profiler.plotting_scripts.scaling import (
    plot_scaling_efficiency,
    plot_speedup,
    plot_weak_scaling,
)
from scope_profiler.plotting_scripts.statistics import (
    collect_region_statistics,
    write_region_statistics_json,
)
from scope_profiler.plotting_scripts.timeline import plot_timeline_density

__all__ = [
    "DEFAULT_CMAP",
    "FLAME_CMAP",
    # Private helpers re-exported for internal cross-module use and for
    # tests that reach into ``scope_profiler.plotting_scripts.<helper>``
    # (including monkeypatching ``_get_canvas``/``_render``/``_hover_summary``).
    "_add_pyvis_controls",
    "_aggregate_gantt_intervals",
    "_as_runs",
    "_close_matplotlib_figure",
    "_display_matplotlib_figure_in_notebook",
    "_duration_timeseries",
    "_filename_slug",
    "_get_canvas",
    "_get_cmap_colors",
    "_group_regions",
    "_hover_region",
    "_hover_summary",
    "_hover_value",
    "_normalize_ranks",
    "_panel_gridspec",
    "_prepare_gantt_data",
    "_region_color_map",
    "_region_summary",
    "_render",
    "_save_matplotlib_figure",
    "_set_xticks",
    "_show_matplotlib_figure",
    "_stacked_segments",
    "_to_hex",
    "_unique_labels",
    "_write_csv",
    "_write_json",
    "available_likwid_metrics",
    "available_perf_event_metrics",
    "collect_region_statistics",
    "plot_callgraph",
    "plot_duration_histogram",
    "plot_duration_timeseries",
    "plot_durations",
    "plot_flame",
    "plot_flame_chart",
    "plot_flame_graph",
    "plot_gantt",
    "plot_imbalance",
    "plot_likwid",
    "plot_perf_events",
    "plot_rank_heatmap",
    "plot_scaling_efficiency",
    "plot_speedup",
    "plot_timeline_density",
    "plot_weak_scaling",
    "write_region_statistics_json",
]
