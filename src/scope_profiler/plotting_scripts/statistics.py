"""Region-duration statistics collection (JSON export, no chart rendering)."""

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from scope_profiler.plotting_scripts._utils import (
    _as_runs,
    _normalize_ranks,
    _unique_labels,
)
from scope_profiler.results import ProfilingResults


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
