"""Data behind the report charts and its scaling axes, without any HTML.

The HTML report and the Astro site generator both need the same chart payloads
(the JSON the ``plot_*`` functions export) and the same rules for what a
comparison scales over; they live here so neither depends on the other.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

#: Regions drawn in a scaling chart: the ones with the most time on the
#: smallest run, which is where scaling matters.
_SPEEDUP_REGIONS = 8


def _threads(run) -> int | None:
    """A run's OpenMP thread count, when its metadata recorded one."""
    try:
        return int(run.metadata["omp_num_threads"])
    except (KeyError, TypeError, ValueError):
        return None


def _has_call_timestamps(run) -> bool:
    """Whether a run recorded its calls, rather than only summary statistics.

    Aggregated and summary-only profiles keep per-region totals but no call
    timestamps, so the nesting a stacked durations bar needs cannot be
    reconstructed from them.
    """
    return all(
        rank_region.stored_summary is None
        for region in run.get_regions()
        for rank_region in region.regions.values()
    )


def _scaling_field(runs) -> str | None:
    """The parallelism compared runs differ in, for a speedup chart.

    MPI ranks whenever they change, even when threads change too (the cores
    axis is then a button away); threads when only they change; None when the
    runs share both, as two versions of the same code do.
    """
    if len({run.num_ranks for run in runs}) > 1:
        return "num_ranks"
    threads = [_threads(run) for run in runs]
    threads_vary = None not in threads and len(set(threads)) > 1
    return "omp_num_threads" if threads_vary else None


#: ``speedup_x`` choices for a comparison report's speedup chart, each naming
#: the run property it puts on the x-axis. ``"auto"`` picks for itself (see
#: :func:`_scaling_field`).
SPEEDUP_X_FIELDS = {
    "ranks": "num_ranks",
    "nodes": "num_nodes",
    "threads": "omp_num_threads",
    "cores": "total_cores",
}
SPEEDUP_X_CHOICES = ("auto", *SPEEDUP_X_FIELDS)


def _scaling_value(run, field: str) -> int | None:
    """A run's value of a scaling field, or None when the run did not record it."""
    if field == "num_ranks":
        return run.num_ranks
    if field == "num_nodes":
        return run.num_nodes
    threads = _threads(run)
    if threads is None:
        return None
    return threads if field == "omp_num_threads" else run.num_ranks * threads


class SpeedupAxisError(ValueError):
    """The requested speedup x-axis is unknown, or some run did not record it."""


def _check_speedup_x(runs, speedup_x: str) -> str | None:
    """The scaling field ``speedup_x`` names, checked against every run.

    ``"auto"`` resolves to :func:`_scaling_field`'s choice. A named field
    every run has to have recorded: a speedup chart silently missing some
    runs, or silently drawn over another axis, would misread the study.
    """
    if speedup_x not in SPEEDUP_X_CHOICES:
        raise SpeedupAxisError(
            f"speedup_x must be one of {', '.join(SPEEDUP_X_CHOICES)}, "
            f"got {speedup_x!r}",
        )
    if speedup_x == "auto":
        return _scaling_field(runs)
    field = SPEEDUP_X_FIELDS[speedup_x]
    missing = [run.display_label for run in runs if _scaling_value(run, field) is None]
    if missing:
        raise SpeedupAxisError(
            f"the {speedup_x} axis needs {field!r} in every run's metadata, "
            f"but {', '.join(missing)} did not record it"
            + (
                " (files written by older scope-profiler versions lack the node "
                "count of a multi-rank run)"
                if field == "num_nodes"
                else ""
            )
            + "; choose another axis, or 'auto'",
        )
    return field


def _speedup_axes(runs, field: str) -> list[str]:
    """The scaling fields a speedup chart over ``field`` can switch to.

    Every field all runs recorded and differ in, in :data:`SPEEDUP_X_FIELDS`
    order; ``field`` itself is always one. A field whose values match another
    one's run for run - cores when every run has one thread - would draw the
    same chart again and is left out.
    """
    order = list(SPEEDUP_X_FIELDS.values())
    kept: dict[str, tuple] = {}
    for candidate in [field, *(other for other in order if other != field)]:
        values = tuple(_scaling_value(run, candidate) for run in runs)
        if candidate != field and (
            None in values or len(set(values)) < 2 or values in kept.values()
        ):
            continue
        kept[candidate] = values
    return [candidate for candidate in order if candidate in kept]


def collect_chart_payloads(
    runs,
    include,
    exclude,
    ranks,
    comparison: bool = False,
    scaling_field: str | None = None,
    scaling: str = "both",
) -> tuple[list[tuple[str, dict, dict]], list[str]]:
    """Build the chart payloads of one run or of a comparison of runs.

    Returns ``(charts, failures)``: ``charts`` holds ``(title, payload,
    options)`` in display order, where ``payload`` is the JSON document a
    ``plot_*`` function exports and ``options`` its extra view settings (the
    ``variants`` of a scaling chart, a layout override); ``failures`` says why
    a chart is missing. Needs ``scope-profiler[pproc]`` and raises
    :class:`ImportError` without it.

    A comparison keeps only the charts that compare runs; each run's timeline
    and rank charts belong to its own page. ``scaling`` picks the chart(s) for
    runs of different sizes: "strong" the speedup, "weak" the weak-scaling
    efficiency, "both" the two of them.
    """
    from scope_profiler.plotting_scripts import (
        available_likwid_metrics,
        collect_region_statistics,
        plot_durations,
        plot_gantt,
        plot_imbalance,
        plot_likwid,
        plot_rank_heatmap,
        plot_speedup,
        plot_weak_scaling_efficiency,
    )

    charts: list[tuple[str, dict, dict]] = []
    failures: list[str] = []

    def payload_of(title, plotter, path: Path, *args, **kwargs) -> dict | None:
        try:
            plotter(
                *args,
                data_filepath=path,
                data_format="json",
                show=False,
                verbose=False,
                # Plot functions prepare their export payload before handing
                # the Canvas to a renderer. This private sentinel reaches the
                # no-output path and avoids materializing an unused Python
                # figure; the browser package owns rendering for reports.
                backend="data-only",
                **kwargs,
            )
        except (ImportError, ValueError) as exc:
            failures.append(f"{title}: {exc}")
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def collect(
        title,
        plotter,
        path: Path,
        *args,
        chart_options: dict | None = None,
        **kwargs,
    ) -> None:
        payload = payload_of(title, plotter, path, *args, **kwargs)
        if payload is not None:
            charts.append((title, payload, chart_options or {}))

    selected_ranks = [
        [
            rank
            for rank in (range(run.num_ranks) if ranks is None else ranks)
            if 0 <= rank < run.num_ranks
        ]
        for run in runs
    ]
    # The x-axis of a comparison's speedup chart (see _check_speedup_x).
    scaling_field = scaling_field if comparison else None

    with tempfile.TemporaryDirectory(prefix="scope-profiler-report-") as directory:
        payload_dir = Path(directory)
        for index, run in enumerate([] if comparison else runs):
            title = (
                # Naming the rank only says something with more than one.
                f"Timeline: {run.display_label}"
                + (" (rank 0)" if run.num_ranks > 1 else "")
            )
            payload = payload_of(
                title,
                plot_gantt,
                payload_dir / f"gantt-{index}.json",
                run,
                include=include,
                exclude=exclude,
                ranks=[0],
            )
            if payload is None:
                continue
            # Every row is already labelled with its region.
            charts.append((title, payload, {"layout": {"showlegend": False}}))

        scaling_values = (
            [_scaling_value(run, scaling_field) for run in runs]
            if scaling_field is not None
            else []
        )
        if scaling_field is not None and len(set(scaling_values)) < 2:
            # Only a named axis gets here; "auto" picks one the runs differ in.
            failures.append(
                f"Speedup: every run has the same {scaling_field} "
                f"({scaling_values[0]}), so there is no scaling to show",
            )
        elif scaling_field is not None:
            baseline_run = min(runs, key=lambda run: _scaling_value(run, scaling_field))
            # A line per region gets unreadable fast: the regions with the
            # most time on the smallest run, which is where scaling matters.
            # The same regions on every axis, so switching axes keeps the lines.
            leading = sorted(
                baseline_run.get_regions(include=include, exclude=exclude),
                key=lambda region: -region.total_duration,
            )[:_SPEEDUP_REGIONS]
            # One payload per axis the runs differ in, for each scaling chart;
            # the report's buttons switch between them without recomputing
            # anything in the browser.
            scaling_charts = (
                ("Speedup", plot_speedup, "speedup", ("strong", "both")),
                (
                    "Weak scaling",
                    plot_weak_scaling_efficiency,
                    "weak-scaling",
                    ("weak", "both"),
                ),
            )
            axes = _speedup_axes(runs, scaling_field)
            for title, plotter, file_stem, modes in scaling_charts:
                if scaling not in modes:
                    continue
                variants = []
                for field in axes:
                    payload = payload_of(
                        title if field == scaling_field else f"{title} over {field}",
                        plotter,
                        payload_dir / f"{file_stem}-{field}.json",
                        runs,
                        x_field=field,
                        include=[f"{re.escape(region.name)}$" for region in leading],
                        exclude=exclude,
                        ranks=ranks,
                    )
                    if payload is not None:
                        variants.append(
                            {
                                "field": field,
                                "label": payload["options"]["x_label"],
                                "payload": payload,
                            },
                        )
                default = next(
                    (item for item in variants if item["field"] == scaling_field),
                    None,
                )
                if default is not None:
                    charts.append(
                        (
                            title,
                            default["payload"],
                            {"variants": variants} if len(variants) > 1 else {},
                        ),
                    )
        if len(runs) == 2:
            # A baseline/candidate pair wants "what changed?", which reads
            # better as one signed bar per region than as two bars a viewer
            # has to subtract by eye.
            statistics = collect_region_statistics(
                runs,
                ranks=ranks,
                include=include,
                exclude=exclude,
            )
            if statistics["files"]:
                baseline, candidate = (file["label"] for file in statistics["files"])
                charts.append(
                    (
                        f"Change: {candidate} vs {baseline}",
                        {"plot": "region_statistics", **statistics},
                        {"comparison": "percent"},
                    ),
                )
        collect(
            "Region durations",
            plot_durations,
            payload_dir / "durations.json",
            runs,
            include=include,
            exclude=exclude,
            ranks=ranks,
            # Totals summed over ranks grow with the rank count; across
            # run sizes, the mean call reads like the speedup chart.
            metric="total" if len({run.num_ranks for run in runs}) == 1 else "avg",
            sort_by="total",
            # One run's bars split into the region's own time and its direct
            # children, which the ranked table cannot show at a glance. Runs
            # compare as grouped bars instead: stacking both ways on one axis
            # would merge equal segment names across runs. The split needs
            # call timestamps, which aggregated profiles do not store.
            stack_children=len(runs) == 1 and _has_call_timestamps(runs[0]),
        )
        if not comparison and charts and charts[-1][0] == "Region durations":
            # Where the time went is the first question about one run, so its
            # durations lead, above the timeline.
            charts.insert(0, charts.pop())

        # Both rank views are empty or trivial with a single rank.
        if not comparison and any(len(selected) > 1 for selected in selected_ranks):
            collect(
                "Rank imbalance",
                plot_imbalance,
                payload_dir / "rank-imbalance.json",
                runs,
                metric="total",
                include=include,
                exclude=exclude,
                ranks=ranks,
            )
            collect(
                "Rank heatmap",
                plot_rank_heatmap,
                payload_dir / "rank-heatmap.json",
                runs,
                include=include,
                exclude=exclude,
                ranks=ranks,
                exclusive=True,
            )

        likwid_metrics = available_likwid_metrics(runs)
        if likwid_metrics:
            metric = likwid_metrics[0]
            collect(
                f"LIKWID: {metric}",
                plot_likwid,
                payload_dir / "likwid.json",
                runs,
                metric=metric,
                include=include,
                exclude=exclude,
                ranks=ranks,
            )

    return charts, failures
