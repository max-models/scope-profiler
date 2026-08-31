"""Duration bar charts: per-region totals/averages, optionally stacked by caller."""

import os
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from scope_profiler import plotting_scripts as _ps
from scope_profiler.plotting_scripts._utils import (
    DEFAULT_CMAP,
    _as_runs,
    _get_cmap_colors,
    _hover_region,
    _normalize_ranks,
    _set_xticks,
    _to_hex,
    _unique_labels,
    _write_csv,
    _write_json,
)
from scope_profiler.plotting_scripts.statistics import (
    _common_region_names,
    _region_duration_values,
    _stats_from_values,
)
from scope_profiler.region import NS_PER_SECOND
from scope_profiler.results import ProfilingResults
from scope_profiler.summary import _name_selected

_DURATION_METRICS: dict[str, tuple[str, str]] = {
    "avg": ("average_duration_seconds", "Average duration per call (seconds)"),
    "min": ("min_duration_seconds", "Minimum duration per call (seconds)"),
    "max": ("max_duration_seconds", "Maximum duration per call (seconds)"),
    "total": ("total_duration_seconds", "Total duration (seconds)"),
}

# Metrics whose bar is a sum over calls, and so can be split into self time
# plus one segment per child region (see ``plot_durations(stack_children=)``).
_STACKABLE_METRICS = frozenset({"avg", "total"})

# Label of the stacked segment holding a region's own, non-nested time.
_SELF_SEGMENT = "self"


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


def _duration_bar_hover(
    run: ProfilingResults,
    bar_name: str,
    members: list[str],
    ranks: list[int] | None,
    extra: Sequence[tuple[str, object]] = (),
    run_label: str | None = None,
) -> str:
    """Hover text for one duration bar.

    An ordinary bar is one region, so it describes itself with its own
    ``get_summary()``. A ``combine_regions`` bar has no region object behind
    it -- it is several regions pooled -- so it names its members and the
    pooled statistics instead.
    """
    heading = bar_name if run_label is None else f"{bar_name} - {run_label}"
    if len(members) == 1:
        region, title = _hover_region(run.get_region(members[0]), ranks)
        if run_label is not None:
            title = f"{title} - {run_label}"
        return _ps._hover_summary(region, title=title, extra=extra)

    lines = [f"<b>{heading}</b>"]
    lines.extend(f"{label}: {value}" for label, value in extra)
    lines.append(f"calls: {int(_pooled_metric_value(run, members, 'count', ranks))}")
    for stat_key, stat_label in (
        ("total_duration_seconds", "total"),
        ("average_duration_seconds", "avg"),
        ("min_duration_seconds", "min"),
        ("max_duration_seconds", "max"),
    ):
        value = _pooled_metric_value(run, members, stat_key, ranks)
        lines.append(f"{stat_label}: {value:.6g} s")
    lines.append("combines: " + ", ".join(members))
    return "<br>".join(lines)


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


def _stacked_segments(
    run: ProfilingResults,
    region_members: dict[str, list[str]],
    ranks: list[int] | None = None,
) -> dict[str, dict[str, float]]:
    """Split each bar's inclusive time into self time plus its direct children.

    Regions record no call graph, so the nesting is reconstructed from
    timestamp containment (:func:`~scope_profiler.call_stack.build_call_arrays`)
    -- the same call graph the flame chart draws, over *all* the run's
    regions, not just the plotted ones, since a region filtered out of the
    bars is still somebody's parent.

    Every call is credited to the bar its parent belongs to. A call whose
    parent is in the same bar (a recursive region, or two members of one
    ``combine_regions`` group nested in each other) is left in that bar's
    self time rather than becoming a segment of itself, so a bar's segments
    always sum to time spent inside it, counted once.

    Returns ``{bar name: {segment label: total nanoseconds}}``, where the
    ``"self"`` segment is exclusive time and every other key is a child
    region's name -- its bar name when that child is itself a plotted bar.
    """
    from scope_profiler.call_stack import build_call_arrays

    regions = run.get_regions()
    bar_of_region = {
        member: bar for bar, members in region_members.items() for member in members
    }
    available = sorted({rank for region in regions for rank in region.ranks})
    selected = available if ranks is None else [r for r in available if r in ranks]

    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for rank in selected:
        calls = build_call_arrays(regions, rank)
        if not len(calls):
            continue

        bar_names = list(
            dict.fromkeys(bar_of_region.get(name, name) for name in calls.names)
        )
        index_of = {name: index for index, name in enumerate(bar_names)}
        row_bar = np.array(
            [index_of[bar_of_region.get(name, name)] for name in calls.names]
        )
        call_bar = row_bar[calls.region_index]

        self_ns = np.bincount(
            call_bar, weights=calls.exclusive_ns, minlength=len(bar_names)
        )
        for index, name in enumerate(bar_names):
            totals[name]["self"] += float(self_ns[index])

        nested = calls.parent >= 0
        parent_bar = call_bar[calls.parent[nested]]
        child_bar = call_bar[nested]
        outside = parent_bar != child_bar
        if not outside.any():
            continue
        durations = (calls.end_ns - calls.start_ns)[nested][outside]
        # One key per (parent bar, child bar) pair; np.unique rather than a
        # dense len(bars)**2 histogram, which a run with many regions would
        # not fit.
        keys = (
            parent_bar[outside].astype(np.int64) * len(bar_names) + child_bar[outside]
        )
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        sums = np.bincount(inverse, weights=durations, minlength=unique_keys.size)
        for key, total in zip(unique_keys, sums):
            parent_name = bar_names[int(key) // len(bar_names)]
            child_name = bar_names[int(key) % len(bar_names)]
            totals[parent_name][child_name] += float(total)

    return {name: dict(segments) for name, segments in totals.items()}


def _stacked_bar_values(
    runs: Sequence[ProfilingResults],
    region_names: list[str],
    region_members: dict[str, list[str]],
    ranks: list[int] | None,
    metric_key: str,
) -> tuple[list[str], list[dict[str, np.ndarray]]]:
    """Build the per-run stacked bar heights, in seconds.

    Returns the ordered segment labels (``"self"`` first, then children by
    descending total across every run) and, per run, a
    ``{segment label: heights indexed like region_names}`` mapping.
    """
    per_run = [_stacked_segments(run, region_members, ranks=ranks) for run in runs]

    pooled: dict[str, float] = defaultdict(float)
    for segments in per_run:
        for region_name in region_names:
            for label, value in segments.get(region_name, {}).items():
                if label != "self":
                    pooled[label] += value
    segment_labels = ["self"] + sorted(
        pooled, key=lambda label: (-pooled[label], label)
    )

    values: list[dict[str, np.ndarray]] = []
    for run, segments in zip(runs, per_run):
        # ``avg`` divides the whole bar by the same call count the unstacked
        # bar uses, so stacking rescales a bar without changing its height.
        if metric_key == "avg":
            scales = [
                _pooled_metric_value(run, region_members[name], "count", ranks=ranks)
                for name in region_names
            ]
        else:
            scales = [1.0] * len(region_names)
        run_values: dict[str, np.ndarray] = {}
        for label in segment_labels:
            heights = np.array(
                [
                    (
                        segments.get(name, {}).get(label, 0.0) / NS_PER_SECOND / scale
                        if scale
                        else float("nan")
                    )
                    for name, scale in zip(region_names, scales)
                ]
            )
            run_values[label] = heights
        values.append(run_values)
    return segment_labels, values


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
    stack_children: bool = False,
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
    stack_children : bool
        Split each bar into the region's own (exclusive) time plus one
        segment per region called directly from it, stacked on top of each
        other, so a bar shows where its time went instead of only how much
        there was. The nesting comes from timestamp containment, the same
        reconstruction the flame chart uses, over every region in the run --
        a child that is filtered out of the bars still gets its own segment.
        Only ``total`` and ``avg`` decompose this way; ``min``/``max`` are
        rejected. Colors then identify segments rather than runs, so with
        several runs the bars are still grouped per region in run order but
        the legend names the segments.
    backend : str
        Backend to use for rendering: "matplotlib" (default) or "plotly".
    return_fig : bool
        Return the rendered figure instead of the saved filepath list.

    Returns
    -------
    list[str]
        List of filepaths that were written (empty if filepath is None).
    """
    Canvas = _ps._get_canvas()
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

    if stack_children:
        # A bar's segments have to sum to the bar: totals and per-call
        # averages do, a minimum or maximum over calls does not.
        unstackable = [key for key in metric_keys if key not in _STACKABLE_METRICS]
        if unstackable:
            raise ValueError(
                f"stack_children does not apply to metric(s) {unstackable}: "
                f"only {sorted(_STACKABLE_METRICS)} decompose into "
                "self time plus children."
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
    # Bars are drawn narrower than their spacing only when stacking: runs
    # then share their segment colors, so touching bars would read as one
    # block. Elsewhere the historical flush grouping is kept.
    bar_width = width * 0.85 if stack_children and num_readers > 1 else width

    saved_paths: list[str] = []
    rendered_figures: list[object] = []
    data_rows = []

    for metric_key in metric_keys:
        stat_key, ylabel = _DURATION_METRICS[metric_key]

        segment_labels: list[str] = []
        stacked_values: list[dict[str, np.ndarray]] = []
        if stack_children:
            segment_labels, stacked_values = _stacked_bar_values(
                runs, region_names, region_members, ranks, metric_key
            )
            segment_colors = dict(
                zip(segment_labels, _get_cmap_colors(cmap, len(segment_labels)))
            )
            values = [
                [
                    float(sum(run_values[segment][index] for segment in segment_labels))
                    for index in range(len(region_names))
                ]
                for run_values in stacked_values
            ]
        else:
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
            if stack_children:
                for label, run_values in zip(labels, stacked_values):
                    for segment in segment_labels:
                        for region_name, value in zip(
                            region_names, run_values[segment]
                        ):
                            data_rows.append(
                                [label, region_name, metric_key, segment, float(value)]
                            )
            else:
                for label, file_values in zip(labels, values):
                    for region_name, value in zip(region_names, file_values):
                        data_rows.append([label, region_name, metric_key, value])

        canvas = Canvas(
            figsize=(fig_width, fig_height),
            gridspec_kw={"bottom": bottom_margin},
        )

        # Create grouped bar chart
        hover_enabled = backend == "plotly"
        x_positions = np.arange(len(region_names))
        offset_start = -0.5 * width * (num_readers - 1)

        for idx, (label, file_values) in enumerate(zip(labels, values)):
            run = runs[idx]
            run_label = label if num_readers > 1 else None
            offsets = x_positions + offset_start + idx * width
            run_values = stacked_values[idx] if stack_children else {}
            if not stack_children:
                bar_hover = None
                if hover_enabled:
                    bar_hover = [
                        _duration_bar_hover(
                            run,
                            region_name,
                            region_members[region_name],
                            ranks,
                            # No bar value line: the bar's height is one
                            # of the statistics the summary already lists.
                            run_label=run_label,
                        )
                        for region_name in region_names
                    ]
                canvas.bar(
                    offsets,
                    file_values,
                    width=bar_width,
                    label=label if num_readers > 1 else None,
                    color=_to_hex(colors[idx]),
                    edgecolor="black",
                    alpha=0.8,
                    hover=bar_hover,
                )
                continue

            # Stack by drawing each segment's cumulative top as an opaque bar
            # from zero, tallest first, so the next one paints over it. That
            # needs nothing from the backend beyond a plain bar -- maxplotlib
            # forwards neither Matplotlib's ``bottom`` nor Plotly's ``base``.
            heights = np.vstack([run_values[segment] for segment in segment_labels])
            cumulative = np.cumsum(np.nan_to_num(heights), axis=0)
            for position in range(len(segment_labels) - 1, -1, -1):
                segment = segment_labels[position]
                # A segment describes the region whose time it is: its own
                # for a child, the bar's own for "self".
                segment_hover = None
                if hover_enabled:
                    segment_hover = [
                        _duration_bar_hover(
                            run,
                            region_name if segment == _SELF_SEGMENT else segment,
                            (
                                region_members[region_name]
                                if segment == _SELF_SEGMENT
                                else region_members.get(segment, [segment])
                            ),
                            ranks,
                            extra=[
                                (
                                    f"{region_name} / {segment}",
                                    f"{run_values[segment][index]:.6g} s",
                                )
                            ],
                            run_label=run_label,
                        )
                        for index, region_name in enumerate(region_names)
                    ]
                canvas.bar(
                    offsets,
                    cumulative[position],
                    width=bar_width,
                    label=segment if idx == 0 else None,
                    color=_to_hex(segment_colors[segment]),
                    edgecolor="black",
                    hover=segment_hover,
                )

        tick_rotation_applied = _set_xticks(
            canvas,
            x_positions,
            labels=region_names,
            rotation=45,
            ha="right",
        )
        canvas.set_ylabel(ylabel)
        title = f"Region duration comparison ({metric_key})"
        if stack_children:
            title += ", children stacked"
        canvas.set_title(title)
        canvas.set_grid(True)
        if log_scale:
            canvas.set_yscale("log")
        if stack_children or num_readers > 1:
            canvas.set_legend()

        metric_filepath = None
        if filepath:
            metric_filepath = _metric_filepath(
                filepath, metric_key, single_metric=len(metric_keys) == 1
            )
            saved_paths.append(metric_filepath)

        rendered = _ps._render(
            canvas,
            metric_filepath,
            show,
            backend,
            # Overlapping bars, not Plotly's default side-by-side grouping:
            # the stack is drawn as bars painted over each other.
            plotly_layout={"barmode": "overlay"} if stack_children else None,
            x_tick_rotation=None if tick_rotation_applied else 45,
            return_fig=return_fig,
        )
        if return_fig:
            rendered_figures.append(rendered)

    if data_filepath:
        if data_format == "json":
            if stack_children:
                bars = [
                    {
                        "file": file,
                        "region": region,
                        "metric": metric,
                        "segment": segment,
                        "value_seconds": value,
                    }
                    for file, region, metric, segment, value in data_rows
                ]
                colors_map = {
                    segment: _to_hex(color)
                    for segment, color in zip(
                        segment_labels, _get_cmap_colors(cmap, len(segment_labels))
                    )
                }
            else:
                bars = [
                    {
                        "file": file,
                        "region": region,
                        "metric": metric,
                        "value_seconds": value,
                    }
                    for file, region, metric, value in data_rows
                ]
                colors_map = {
                    label: _to_hex(color) for label, color in zip(labels, colors)
                }
            _write_json(
                data_filepath,
                {"bars": bars, "colors": colors_map, "metrics": metric_keys},
            )
        else:
            header = ["file", "region", "metric", "value_seconds"]
            if stack_children:
                header = ["file", "region", "metric", "segment", "value_seconds"]
            _write_csv(data_filepath, header, data_rows)

    if return_fig:
        return rendered_figures[0]
    return saved_paths


