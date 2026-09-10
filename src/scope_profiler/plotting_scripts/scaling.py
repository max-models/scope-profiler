"""Scaling charts: speedup, weak-scaling, and parallel-efficiency curves."""

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
    _to_hex,
    _write_csv,
    _write_json,
)
from scope_profiler.plotting_scripts.statistics import (
    _common_region_names,
    _region_average_duration,
    _speedup_x_value,
)
from scope_profiler.results import ProfilingResults

_SCALING_X_FIELDS = {"num_ranks", "omp_num_threads", "total_cores"}


def _scaling_hover_texts(
    region_at_key: dict,
    region_name: str,
    x_field: str,
    keys: Sequence,
    values: Sequence[float],
    value_label: str,
    durations: Sequence[float],
    ranks: list[int] | None,
) -> list[str]:
    """Hover text for one region's curve in a scaling plot.

    Every point on the curve is a different run, so each is described by
    that run's own region summary, with the plotted value and the mean
    duration it was computed from above it.
    """
    texts = []
    for key, value, duration in zip(keys, values, durations):
        region, title = _hover_region(region_at_key[key], ranks)
        texts.append(
            _ps._hover_summary(
                region,
                title=f"{title} @ {x_field} = {key}",
                extra=[
                    (value_label, f"{value:.4g}"),
                    ("mean duration", f"{duration:.6g} s"),
                ],
            ),
        )
    return texts


_X_LABELS = {
    "num_ranks": "MPI ranks",
    "omp_num_threads": "OpenMP threads",
    "total_cores": "MPI ranks × OpenMP threads",
}


def _x_label(x_field: str) -> str:
    """Human-readable axis label for a scaling x field."""
    return _X_LABELS.get(x_field, x_field)


def _scaling_options(x_field: str, baseline_key) -> dict:
    """The ``options`` block shared by every scaling plot-data document."""
    return {
        "x_field": x_field,
        "x_label": _x_label(x_field),
        "baseline": baseline_key,
    }


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
    Canvas = _ps._get_canvas()
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
            + ", ".join(map(str, x_keys)),
        )

    duration_samples: dict[str, dict] = {
        region_name: defaultdict(list) for region_name in region_names
    }
    # The region behind each point, for its hover summary: the first run at
    # that x value, which is the one the curve is really about when several
    # runs share a scale.
    region_at_key: dict[str, dict] = {region_name: {} for region_name in region_names}
    for run, x_value in zip(runs, x_per_reader):
        for region_name in region_names:
            duration = _region_average_duration(
                run.get_region(region_name),
                ranks=ranks,
            )
            if np.isfinite(duration) and duration > 0:
                duration_samples[region_name][x_value].append(duration)
                region_at_key[region_name].setdefault(
                    x_value,
                    run.get_region(region_name),
                )

    baseline_key = x_keys[0]
    colors = _get_cmap_colors(cmap, len(region_names))
    fig_width = max(10, 1.2 * len(x_keys) + 3)
    fig_height = max(4.5, 2.8 + 0.35 * len(region_names))

    x_position = {key: (key if is_scaling else i) for i, key in enumerate(x_keys)}

    canvas = Canvas(figsize=(fig_width, fig_height))
    hover_enabled = backend == "plotly"
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
        means = []
        for key in x_keys:
            samples = region_values.get(key, [])
            if not samples:
                continue
            mean_duration = float(np.mean(samples))
            if not np.isfinite(mean_duration) or mean_duration <= 0:
                continue
            plot_x.append(x_position[key])
            plot_keys.append(key)
            means.append(mean_duration)
            speedups.append(baseline_duration / mean_duration)

        if not plot_x:
            continue

        plotted += 1
        line_hover = None
        if hover_enabled:
            line_hover = _scaling_hover_texts(
                region_at_key[region_name],
                region_name,
                x_field,
                plot_keys,
                speedups,
                "speedup",
                means,
                ranks,
            )
        canvas.add_line(
            plot_x,
            speedups,
            linewidth=1.8,
            color=_to_hex(colors[idx]),
            label=region_name,
            hover=line_hover,
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
            _write_json(
                data_filepath,
                {
                    "points": points,
                    "colors": colors_map,
                    "options": _scaling_options(x_field, baseline_key),
                },
                plot="speedup",
            )
        else:
            _write_csv(data_filepath, ["region", x_field, "speedup"], data_rows)

    x_label = _x_label(x_field)

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

    rendered = _ps._render(canvas, filepath, show, backend, return_fig=return_fig)
    return rendered if return_fig else None


def _weak_scaling_curve(
    profiling_data: ProfilingResults | Sequence[ProfilingResults],
    *,
    kind: str,
    x_field: str,
    ranks: list[int] | int | None,
    include: list[str] | str | None,
    exclude: list[str] | str | None,
    filepath: str | None,
    show: bool,
    verbose: bool,
    cmap: str,
    data_filepath: str | Path | None,
    data_format: str,
    backend: str,
    return_fig: bool,
    extra_options: dict | None = None,
) -> object | None:
    """Draw one of the two weak-scaling curves.

    Weak-scaling runtime and weak-scaling efficiency are reciprocals of each
    other -- the same durations, normalized to the baseline scale in either
    direction, against the same flat ideal line at 1.0 -- so both are drawn
    from here and differ only by :data:`_WEAK_SCALING_KINDS`.
    """
    spec = _WEAK_SCALING_KINDS[kind]
    Canvas = _ps._get_canvas()
    runs = _as_runs(profiling_data)
    if not runs:
        return
    if len(runs) < 2:
        raise ValueError(f"{spec['noun']} requires at least two profiling files.")

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
            f"Plotting {spec['gerund']} using x_field={x_field!r}, values: "
            + ", ".join(map(str, x_keys)),
        )

    duration_samples: dict[str, dict] = {
        region_name: defaultdict(list) for region_name in region_names
    }
    region_at_key: dict[str, dict] = {region_name: {} for region_name in region_names}
    for run, x_value in zip(runs, x_per_reader):
        for region_name in region_names:
            duration = _region_average_duration(
                run.get_region(region_name),
                ranks=ranks,
            )
            if np.isfinite(duration) and duration > 0:
                duration_samples[region_name][x_value].append(duration)
                region_at_key[region_name].setdefault(
                    x_value,
                    run.get_region(region_name),
                )

    baseline_key = x_keys[0]
    colors = _get_cmap_colors(cmap, len(region_names))
    fig_width = max(10, 1.2 * len(x_keys) + 3)
    fig_height = max(4.5, 2.8 + 0.35 * len(region_names))
    x_position = {key: (key if is_scaling else i) for i, key in enumerate(x_keys)}

    canvas = Canvas(figsize=(fig_width, fig_height))
    hover_enabled = backend == "plotly"
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
        y_values = []
        means = []
        for key in x_keys:
            samples = region_values.get(key, [])
            if not samples:
                continue
            mean_duration = float(np.mean(samples))
            if not np.isfinite(mean_duration) or mean_duration <= 0:
                continue
            plot_x.append(x_position[key])
            plot_keys.append(key)
            means.append(mean_duration)
            y_values.append(spec["value"](baseline_duration, mean_duration))

        if not plot_x:
            continue
        plotted += 1
        line_hover = None
        if hover_enabled:
            line_hover = _scaling_hover_texts(
                region_at_key[region_name],
                region_name,
                x_field,
                plot_keys,
                y_values,
                spec["hover_label"],
                means,
                ranks,
            )
        canvas.add_line(
            plot_x,
            y_values,
            linewidth=1.8,
            color=_to_hex(colors[idx]),
            label=region_name,
            hover=line_hover,
        )
        if data_filepath:
            for key, value in zip(plot_keys, y_values):
                data_rows.append([region_name, key, value])

    if plotted == 0:
        raise ValueError(f"No valid {spec['data_noun']} data could be computed.")

    if data_filepath:
        if data_format == "json":
            points = [
                {"region": region, x_field: key, spec["y_key"]: value}
                for region, key, value in data_rows
            ]
            colors_map = {
                name: _to_hex(color) for name, color in zip(region_names, colors)
            }
            _write_json(
                data_filepath,
                {
                    "points": points,
                    "colors": colors_map,
                    "options": {
                        **_scaling_options(x_field, baseline_key),
                        **(extra_options or {}),
                    },
                },
                plot=kind,
            )
        else:
            _write_csv(
                data_filepath,
                ["region", x_field, spec["y_key"]],
                data_rows,
            )

    x_label = _x_label(x_field)
    if is_scaling:
        canvas.set_xticks(np.array(x_keys, dtype=float))
    else:
        canvas.set_xticks(list(range(len(x_keys))), labels=[str(key) for key in x_keys])
    canvas.add_line(
        [x_position[key] for key in x_keys],
        [1.0] * len(x_keys),
        linestyle="--",
        color="black",
        linewidth=1.5,
        label=spec["ideal_label"],
    )
    canvas.set_xlabel(x_label)
    canvas.set_ylabel(spec["ylabel"])
    canvas.set_title(f"{spec['title']} (baseline: {x_label} = {baseline_key})")
    if spec["ylim"] is not None:
        canvas.set_ylim(*spec["ylim"])
    canvas.set_grid(True)
    canvas.set_legend()

    rendered = _ps._render(canvas, filepath, show, backend, return_fig=return_fig)
    return rendered if return_fig else None


_WEAK_SCALING_KINDS: dict[str, dict] = {
    "weak_scaling": {
        "noun": "Weak scaling plot",
        "data_noun": "weak-scaling",
        "gerund": "weak scaling comparison",
        # Runtime relative to the baseline scale: 2.0 means the same work per
        # rank took twice as long once the run was scaled up.
        "value": lambda baseline, duration: duration / baseline,
        "y_key": "normalized_runtime",
        "hover_label": "normalized runtime",
        "ylabel": "Normalized runtime",
        "title": "Weak scaling",
        "ideal_label": "Ideal weak scaling",
        "ylim": None,
    },
    "weak_scaling_efficiency": {
        "noun": "Weak scaling efficiency",
        "data_noun": "weak-scaling-efficiency",
        "gerund": "weak scaling efficiency",
        # The reciprocal, read as a fraction of ideal: 0.5 means half the
        # work per rank is being lost to the cost of scaling up.
        "value": lambda baseline, duration: baseline / duration,
        "y_key": "efficiency",
        "hover_label": "efficiency",
        "ylabel": "Weak-scaling efficiency",
        "title": "Weak-scaling efficiency",
        "ideal_label": "Ideal efficiency",
        "ylim": (0, 1.05),
    },
}


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

    See :func:`plot_weak_scaling_efficiency` for the same data read as a
    fraction of ideal instead.
    """
    return _weak_scaling_curve(
        profiling_data,
        kind="weak_scaling",
        x_field=x_field,
        ranks=ranks,
        include=include,
        exclude=exclude,
        filepath=filepath,
        show=show,
        verbose=verbose,
        cmap=cmap,
        data_filepath=data_filepath,
        data_format=data_format,
        backend=backend,
        return_fig=return_fig,
    )


def plot_weak_scaling_efficiency(
    profiling_data: ProfilingResults | Sequence[ProfilingResults],
    x_field: str = "num_ranks",
    ranks: list[int] | int | None = None,
    include: list[str] | str | None = None,
    exclude: list[str] | str | None = None,
    work_per_rank: Sequence[float] | None = None,
    filepath: str | None = None,
    show: bool = False,
    verbose: bool = True,
    cmap: str = DEFAULT_CMAP,
    data_filepath: str | Path | None = None,
    data_format: str = "csv",
    backend: str = "matplotlib",
    return_fig: bool = False,
) -> object | None:
    """Plot weak-scaling efficiency: baseline runtime over runtime at each scale.

    A weak-scaling study grows the problem with the machine, so every run
    does the same work per rank and ideal efficiency is a flat 1.0. A region
    at 0.6 on 64 ranks is spending 40% of its time on costs that only appear
    at that scale -- communication, load imbalance, or contention.

    This differs from :func:`plot_scaling_efficiency`, which is for a
    *strong*-scaling study: there the problem size is fixed, so the ideal is
    a speedup proportional to the rank count and efficiency divides the
    measured speedup by that. Here the ideal is constant runtime, so no such
    division applies -- using the strong-scaling plot on weak-scaling runs
    reports a near-zero efficiency that means nothing.

    Parameters
    ----------
    work_per_rank : Sequence[float], optional
        Work per rank for each profiling run, in whatever unit suits the
        problem (grid cells, particles, unknowns). Only the caller can know
        this -- it is a property of the problem, not of the profile -- so it
        is optional, but when given it is checked: the runs must agree, and
        a mismatch raises rather than plotting an efficiency that silently
        compares runs doing different amounts of work. The value is recorded
        in the exported plot data.
    """
    extra_options = None
    if work_per_rank is not None:
        work = [float(value) for value in work_per_rank]
        if len(work) != len(_as_runs(profiling_data)):
            raise ValueError(
                "work_per_rank must have one value per profiling file.",
            )
        if not work or work[0] <= 0:
            raise ValueError("work_per_rank values must be positive.")
        if any(
            not np.isclose(value, work[0], rtol=1e-9, atol=0.0) for value in work[1:]
        ):
            raise ValueError(
                "Weak-scaling efficiency requires the same work per rank in "
                f"every run, but got {work}. These runs are not a weak-scaling "
                "study; use plot_scaling_efficiency for a fixed problem size.",
            )
        extra_options = {"work_per_rank": work[0]}

    return _weak_scaling_curve(
        profiling_data,
        kind="weak_scaling_efficiency",
        x_field=x_field,
        ranks=ranks,
        include=include,
        exclude=exclude,
        filepath=filepath,
        show=show,
        verbose=verbose,
        cmap=cmap,
        data_filepath=data_filepath,
        data_format=data_format,
        backend=backend,
        return_fig=return_fig,
        extra_options=extra_options,
    )


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
    Canvas = _ps._get_canvas()
    runs = _as_runs(profiling_data)
    if not runs:
        return
    if len(runs) < 2:
        raise ValueError("Scaling efficiency requires at least two profiling files.")
    if x_field not in _SCALING_X_FIELDS:
        raise ValueError(
            "Scaling efficiency requires x_field to be one of: "
            + ", ".join(sorted(_SCALING_X_FIELDS)),
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
    region_at_key: dict[str, dict] = {name: {} for name in region_names}
    for run, x_value in zip(runs, x_per_reader):
        for name in region_names:
            duration = _region_average_duration(run.get_region(name), ranks=ranks)
            if np.isfinite(duration) and duration > 0:
                samples[name][x_value].append(duration)
                region_at_key[name].setdefault(x_value, run.get_region(name))

    canvas = Canvas(
        figsize=(
            max(10, 1.2 * len(x_keys) + 3),
            max(4.5, 2.8 + 0.35 * len(region_names)),
        ),
    )
    data_rows = []
    plotted = 0
    hover_enabled = backend == "plotly"
    for index, name in enumerate(region_names):
        baseline_values = samples[name].get(baseline_key, [])
        if not baseline_values:
            continue
        baseline_duration = float(np.mean(baseline_values))
        plot_x, efficiencies, plot_keys, means = [], [], [], []
        for key in x_keys:
            values = samples[name].get(key, [])
            if not values:
                continue
            duration = float(np.mean(values))
            plot_x.append(x_position[key])
            plot_keys.append(key)
            means.append(duration)
            efficiencies.append((baseline_duration / duration) / (key / baseline_key))
            data_rows.append([name, key, efficiencies[-1]])
        if plot_x:
            plotted += 1
            line_hover = None
            if hover_enabled:
                line_hover = _scaling_hover_texts(
                    region_at_key[name],
                    name,
                    x_field,
                    plot_keys,
                    efficiencies,
                    "efficiency",
                    means,
                    ranks,
                )
            canvas.add_line(
                plot_x,
                efficiencies,
                linewidth=1.8,
                color=_to_hex(colors[index]),
                label=name,
                hover=line_hover,
            )
    if not plotted:
        raise ValueError("No valid scaling-efficiency data could be computed.")

    if data_filepath:
        header = ["region", x_field, "efficiency"]
        if data_format == "json":
            colors_map = {
                name: _to_hex(color) for name, color in zip(region_names, colors)
            }
            _write_json(
                data_filepath,
                {
                    "points": [dict(zip(header, row)) for row in data_rows],
                    "colors": colors_map,
                    "options": _scaling_options(x_field, baseline_key),
                },
                plot="scaling_efficiency",
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
    x_label = _x_label(x_field)
    canvas.set_xlabel(x_label)
    canvas.set_ylabel("Scaling efficiency")
    canvas.set_title(f"Scaling efficiency (baseline: {x_label} = {baseline_key})")
    canvas.set_ylim(0, 1.05)
    canvas.set_grid(True)
    canvas.set_legend()
    rendered = _ps._render(canvas, filepath, show, backend, return_fig=return_fig)
    return rendered if return_fig else None
