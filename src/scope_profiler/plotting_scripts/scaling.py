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
            )
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
            + ", ".join(map(str, x_keys))
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
                    x_value, run.get_region(region_name)
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
    Canvas = _ps._get_canvas()
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
    region_at_key: dict[str, dict] = {region_name: {} for region_name in region_names}
    for run, x_value in zip(runs, x_per_reader):
        for region_name in region_names:
            duration = _region_average_duration(
                run.get_region(region_name), ranks=ranks
            )
            if np.isfinite(duration) and duration > 0:
                duration_samples[region_name][x_value].append(duration)
                region_at_key[region_name].setdefault(
                    x_value, run.get_region(region_name)
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
        runtimes = []
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
            runtimes.append(mean_duration / baseline_duration)

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
                runtimes,
                "normalized runtime",
                means,
                ranks,
            )
        canvas.add_line(
            plot_x,
            runtimes,
            linewidth=1.8,
            color=_to_hex(colors[idx]),
            label=region_name,
            hover=line_hover,
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
            _write_json(
                data_filepath,
                {
                    "points": points,
                    "colors": colors_map,
                    "options": _scaling_options(x_field, baseline_key),
                },
                plot="weak_scaling",
            )
        else:
            _write_csv(
                data_filepath, ["region", x_field, "normalized_runtime"], data_rows
            )

    x_label = _x_label(x_field)
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

    rendered = _ps._render(canvas, filepath, show, backend, return_fig=return_fig)
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
    Canvas = _ps._get_canvas()
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
        )
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
