"""Parsing of merged HDF5 profiling output files.

The public entry points are
:meth:`ProfilingResults.from_h5 <scope_profiler.results.ProfilingResults.from_h5>`
and its module-level twin :func:`read_h5`; both return a plain
:class:`~scope_profiler.results.ProfilingResults`. This module holds only the
HDF5 parsing that turns a merged file into that object's constructor
arguments.
"""

import json
from copy import copy
from pathlib import Path

import h5py
import numpy as np

from scope_profiler.h5schema import HDF5SchemaError, migrate_schema, read_schema_version
from scope_profiler.likwid_data import LIKWID_GROUP, LikwidRegionResult
from scope_profiler.mpi_region import MPIRegion
from scope_profiler.region import Region
from scope_profiler.results import ProfilingResults


class SummaryDataUnavailable(ValueError):
    """Raised when a file predates fixed-size schema-2 summary columns."""


_SUMMARY_DATASET = "summary_statistics"
_SUMMARY_FIELDS = frozenset(
    {
        "total",
        "minimum",
        "maximum",
        "first",
        "last",
        "start_minimum",
        "end_maximum",
        "gpu_count",
        "gpu_total",
        "mean",
        "m2",
    }
)
_SUMMARY_ROW_DATASETS = (
    "region_ids",
    "ranks",
    "event_counts",
    "source_lines",
    "source_files",
    "source_texts",
    "tags",
    _SUMMARY_DATASET,
)


def _validate_summary_index(index, num_region_names: int) -> int:
    """Validate fixed-size row columns before constructing summary objects."""
    missing = [name for name in _SUMMARY_ROW_DATASETS if name not in index]
    if missing:
        raise HDF5SchemaError(
            "schema-2 summary index is missing dataset(s): " + ", ".join(missing)
        )
    row_count = len(index["region_ids"])
    for name in _SUMMARY_ROW_DATASETS:
        dataset = index[name]
        if dataset.ndim != 1 or len(dataset) != row_count:
            raise HDF5SchemaError(
                f"rank_region_index/{name} must be one-dimensional with "
                f"{row_count} rows, got shape {dataset.shape}"
            )

    fields = index[_SUMMARY_DATASET].dtype.names
    missing_fields = sorted(_SUMMARY_FIELDS - set(fields or ()))
    if missing_fields:
        raise HDF5SchemaError(
            "summary_statistics is missing field(s): " + ", ".join(missing_fields)
        )

    region_ids = index["region_ids"][()]
    if len(region_ids) and (
        np.any(region_ids < 0) or np.any(region_ids >= num_region_names)
    ):
        raise HDF5SchemaError("summary index contains an out-of-range region id")
    ranks = index["ranks"][()]
    pairs = list(zip(region_ids.tolist(), ranks.tolist()))
    if len(set(pairs)) != len(pairs):
        raise HDF5SchemaError("summary index contains a duplicate rank/region row")
    return row_count


def _selection_set(values, *, coerce):
    if values is None:
        return None
    if isinstance(values, (str, int)):
        values = [values]
    return {coerce(value) for value in values}


def _decode_attribute(value):
    """Convert an HDF5 attribute into a plain Python value.

    h5py hands back list-valued attributes (e.g. the loaded modules) as numpy
    arrays of bytes; callers want a list of ``str``.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [_decode_attribute(item) for item in value.tolist()]
    return value


def _read_likwid_group(group) -> dict:
    """Rebuild the LIKWID results stored under one rank's ``likwid`` group.

    Returns
    -------
    dict
        Region tag -> :class:`~scope_profiler.likwid_data.LikwidRegionResult`.
    """
    results = {}
    regions = group.get("regions")
    if regions is None:
        return results

    for region_grp in regions.values():
        attrs = region_grp.attrs
        tag = _decode_attribute(attrs.get("tag", ""))
        results[tag] = LikwidRegionResult(
            tag=tag,
            group_id=int(attrs.get("group_id", -1)),
            group_name=_decode_attribute(attrs.get("group_name", "")),
            cpus=[int(c) for c in region_grp["cpus"][()]],
            times=region_grp["times"][()],
            call_counts=region_grp["call_counts"][()],
            event_names=list(_decode_attribute(attrs.get("event_names", []))),
            # Absent in files written before counter names were recorded.
            counter_names=list(_decode_attribute(attrs.get("counter_names", []))),
            events=region_grp["events"][()],
            metric_names=list(_decode_attribute(attrs.get("metric_names", []))),
            metrics=region_grp["metrics"][()],
            source=_decode_attribute(attrs.get("source", "")),
        )
    return results


def _read_line_profile_group(group) -> list:
    """Read one rank's persisted line-profiler records."""
    records = []
    for function_grp in group.values():
        attrs = function_grp.attrs
        records.append(
            {
                "region": _decode_attribute(attrs.get("region", "")),
                "filename": _decode_attribute(attrs.get("filename", "")),
                "function": _decode_attribute(attrs.get("function", "")),
                "first_lineno": int(attrs.get("first_lineno", 0)),
                "line_numbers": function_grp["line_numbers"][()],
                "hits": function_grp["hits"][()],
                "times": function_grp["times"][()],
                "unit": float(attrs.get("unit", 1.0)),
            }
        )
    return records


def _read_columnar_regions(h5file) -> tuple[dict, list[str], dict]:
    """Read schema-2 shared event columns into the existing Region API.

    Every column is read once, whole, and then sliced in memory. Indexing the
    HDF5 datasets row by row instead costs a file round trip per row and per
    field, which dominates the read of any real run: on a 128-rank file with
    40 regions each (5120 rows, 2.6M events) that was 2.3s against 0.1s here.
    The per-region timing arrays are numpy *views* into the shared columns
    rather than copies, so holding them costs no more than the columns do.
    """
    names = [_decode_attribute(value) for value in h5file["region_table/names"][()]]
    index = h5file["rank_region_index"]
    events = h5file["events"]

    region_ids = index["region_ids"][()]
    ranks = index["ranks"][()]
    offsets = index["event_offsets"][()]
    counts = index["event_counts"][()]
    source_lines = index["source_lines"][()]
    source_files = index["source_files"][()]
    source_texts = index["source_texts"][()]
    tag_blobs = index["tags"][()]
    # Absent in files written before the run stored its own exclusive totals;
    # _NO_EXCLUSIVE_TOTAL marks a row whose writer did not compute one. Either
    # way the reader falls back to reconstructing the nesting on demand.
    totals_column = (
        index["exclusive_totals"][()] if "exclusive_totals" in index else None
    )

    start_times = events["start_times"][()]
    end_times = events["end_times"][()]
    gpu_column = events["gpu_durations"][()] if "gpu_durations" in events else None
    call_column = events["call_ids"][()] if "call_ids" in events else None
    parent_column = events["parent_ids"][()] if "parent_ids" in events else None

    per_region: dict[str, dict[int, Region]] = {name: {} for name in names}
    exclusive_totals: dict[str, dict[int, int]] = {}
    for row in range(len(region_ids)):
        name = names[int(region_ids[row])]
        rank = int(ranks[row])
        if totals_column is not None and totals_column[row] >= 0:
            exclusive_totals.setdefault(name, {})[rank] = int(totals_column[row])
        offset = int(offsets[row])
        event_slice = slice(offset, offset + int(counts[row]))
        gpu_durations = None
        if gpu_column is not None:
            candidate = gpu_column[event_slice]
            if np.any(candidate >= 0):
                gpu_durations = candidate
        call_ids = call_column[event_slice] if call_column is not None else None
        parent_ids = parent_column[event_slice] if parent_column is not None else None
        source_line = int(source_lines[row])
        per_region[name][rank] = Region(
            start_times[event_slice],
            end_times[event_slice],
            gpu_durations=gpu_durations,
            call_ids=call_ids,
            parent_ids=parent_ids,
            source_file=_decode_attribute(source_files[row]) or None,
            source_lineno=source_line if source_line >= 0 else None,
            source_text=_decode_attribute(source_texts[row]) or None,
            tags=tuple(json.loads(_decode_attribute(tag_blobs[row]) or "[]")),
        )
    return per_region, names, exclusive_totals


def _read_aggregate_regions(h5file):
    """Read the compact aggregate-only storage layout."""
    names = [_decode_attribute(value) for value in h5file["region_table/names"][()]]
    index = h5file["rank_region_index"]
    per_region = {name: {} for name in names}
    fields = [
        (key, index[key][()])
        for key in (
            "region_ids",
            "ranks",
            "aggregate_counts",
            "aggregate_totals",
            "aggregate_minimums",
            "aggregate_maximums",
            "aggregate_exclusives",
        )
    ]
    for row, values in enumerate(zip(*(array for _, array in fields))):
        region_id, rank, count, total, minimum, maximum, exclusive = values
        per_region[names[int(region_id)]][int(rank)] = Region(
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            aggregate={
                "count": int(count),
                "total": int(total),
                "minimum": int(minimum),
                "maximum": int(maximum),
                "exclusive": int(exclusive),
            },
        )
    return per_region, names, {}


def load_h5(file_path: str | Path, verbose: bool = False) -> dict:
    """
    Parse a merged profiling file into :class:`ProfilingResults` arguments.

    This is the parsing half of
    :meth:`~scope_profiler.results.ProfilingResults.from_h5`, kept separate so
    the analysis layer does not depend on h5py. Call ``from_h5`` or
    :func:`read_h5` instead unless you are building a ``ProfilingResults``
    subclass of your own.

    Parameters
    ----------
    file_path : str | Path
        Path to the merged HDF5 file containing profiling data.
    verbose : bool, optional
        Print each rank group as it is read (default: False).

    Returns
    -------
    dict
        Keyword arguments for :class:`~scope_profiler.results.ProfilingResults`.

    Raises
    ------
    FileNotFoundError
        If the specified HDF5 file does not exist.
    """
    # Importing registers optional third-party HDF5 filters (notably Zstd)
    # before any compressed dataset is opened. Built-in filters need nothing.
    try:
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass

    file_path = Path(file_path)
    num_ranks = 0
    metadata: dict = {}
    # rank -> {tag: LikwidRegionResult}; empty unless the run used LIKWID.
    likwid: dict[int, dict[str, LikwidRegionResult]] = {}
    line_profile: dict[int, list] = {}
    if not file_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {file_path}")

    # Read the file
    _region_dict = {}
    region_names = []
    exclusive_totals: dict[str, dict[int, int]] = {}
    with h5py.File(file_path, "r") as f:
        schema_version = read_schema_version(f)
        migrate_schema(f, schema_version)
        if "metadata" in f:
            metadata = {
                key: _decode_attribute(value)
                for key, value in f["metadata"].attrs.items()
            }

        if schema_version == 2:
            reader = (
                _read_aggregate_regions
                if f.attrs.get("storage_layout", "columnar") == "aggregate"
                else _read_columnar_regions
            )
            _region_dict, region_names, exclusive_totals = reader(f)
            recorded_ranks = f["rank_region_index/ranks"][()]
            inferred_ranks = (
                int(np.max(recorded_ranks)) + 1 if len(recorded_ranks) else 0
            )
            num_ranks = (
                max(int(metadata.get("mpi_size", 1)), inferred_ranks)
                if len(recorded_ranks)
                else 0
            )

        # Iterate over the rank groups in rank order. h5py yields them in
        # name order, which puts "rank10" before "rank2"; pooled statistics
        # sum per-rank arrays in this order, so a stable numeric order is what
        # makes the numbers reproducible and identical to the ones
        # ProfileManager.finalize(return_results=True) computes in memory.
        rank_group_names = sorted(
            (name for name in f if name.startswith("rank") and name[4:].isdigit()),
            key=lambda name: int(name.replace("rank", "")),
        )
        for rank_group_name in rank_group_names:
            rank_group = f[rank_group_name]
            if schema_version == 1:
                num_ranks += 1
            if verbose:
                print(f"{rank_group_name = }")
                print(rank_group_name, rank_group)
            rank = int(rank_group_name.replace("rank", ""))

            if LIKWID_GROUP in rank_group:
                likwid[rank] = _read_likwid_group(rank_group[LIKWID_GROUP])
            if "line_profile" in rank_group:
                line_profile[rank] = _read_line_profile_group(
                    rank_group["line_profile"]
                )

            if schema_version == 2 or "regions" not in rank_group:
                continue
            regions_group = rank_group["regions"]

            for region_name, region_grp in regions_group.items():
                region_names.append(region_name)
                attrs = region_grp.attrs
                region = Region(
                    region_grp["start_times"][()],
                    region_grp["end_times"][()],
                    gpu_durations=(
                        region_grp["gpu_durations"][()]
                        if "gpu_durations" in region_grp
                        else None
                    ),
                    call_ids=(
                        region_grp["call_ids"][()] if "call_ids" in region_grp else None
                    ),
                    parent_ids=(
                        region_grp["parent_ids"][()]
                        if "parent_ids" in region_grp
                        else None
                    ),
                    source_file=(
                        _decode_attribute(attrs["source_file"])
                        if "source_file" in attrs
                        else None
                    ),
                    source_lineno=(
                        int(attrs["source_lineno"])
                        if "source_lineno" in attrs
                        else None
                    ),
                    source_text=(
                        _decode_attribute(attrs["source_text"])
                        if "source_text" in attrs
                        else None
                    ),
                    tags=tuple(
                        _decode_attribute(attrs["tags"]) if "tags" in attrs else ()
                    ),
                )
                # Merge if region already exists (from another rank)
                if region_name in _region_dict:
                    _region_dict[region_name][rank] = region
                else:
                    _region_dict[region_name] = {rank: region}

    regions = {
        region_name: MPIRegion(name=region_name, regions=_region_dict[region_name])
        for region_name in dict.fromkeys(region_names)
    }

    return {
        "regions": regions,
        "metadata": metadata,
        "num_ranks": num_ranks,
        "likwid": likwid,
        "line_profile": line_profile,
        "file_path": file_path,
        "exclusive_totals": exclusive_totals,
    }


def load_h5_summary(
    file_path: str | Path,
    verbose: bool = False,
    *,
    include_likwid: bool = True,
    include_line_profile: bool = True,
    regions: str | list[str] | tuple[str, ...] | None = None,
    ranks: int | list[int] | tuple[int, ...] | None = None,
) -> dict:
    """Parse fixed-size profile statistics without reading event datasets.

    Summary columns were added compatibly within schema 2. Files that predate
    them raise :class:`SummaryDataUnavailable`; callers that require broad
    compatibility should use :func:`read_h5_summary`, which falls back to the
    normal eager reader by default.
    """
    file_path = Path(file_path)
    selected_names = _selection_set(regions, coerce=str)
    selected_ranks = _selection_set(ranks, coerce=int)
    if not file_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {file_path}")

    metadata = {}
    likwid = {}
    line_profile = {}
    exclusive_totals = {}
    with h5py.File(file_path, "r") as h5file:
        schema_version = read_schema_version(h5file)
        migrate_schema(h5file, schema_version)
        if schema_version != 2:
            raise SummaryDataUnavailable(
                "summary-only reads require a schema-2 profiling file"
            )
        if "metadata" in h5file:
            metadata = {
                key: _decode_attribute(value)
                for key, value in h5file["metadata"].attrs.items()
            }

        layout = h5file.attrs.get("storage_layout", "columnar")
        if layout == "aggregate":
            per_region, region_names, exclusive_totals = _read_aggregate_regions(h5file)
            region_names = [
                name
                for name in region_names
                if selected_names is None or name in selected_names
            ]
            per_region = {
                name: {
                    rank: region
                    for rank, region in per_region[name].items()
                    if selected_ranks is None or rank in selected_ranks
                }
                for name in region_names
            }
            region_names = [name for name in region_names if per_region[name]]
            exclusive_totals = {
                name: {
                    rank: total
                    for rank, total in rank_totals.items()
                    if selected_ranks is None or rank in selected_ranks
                }
                for name, rank_totals in exclusive_totals.items()
                if name in per_region
            }
        else:
            index = h5file["rank_region_index"]
            if _SUMMARY_DATASET not in index:
                raise SummaryDataUnavailable(
                    "profiling file has no fixed-size summary statistics"
                )

            all_region_names = [
                _decode_attribute(value) for value in h5file["region_table/names"][()]
            ]
            row_count = _validate_summary_index(index, len(all_region_names))
            all_region_ids = index["region_ids"][()]
            all_ranks = index["ranks"][()]
            keep = np.ones(row_count, dtype=bool)
            if selected_names is not None:
                keep &= np.fromiter(
                    (
                        all_region_names[int(region_id)] in selected_names
                        for region_id in all_region_ids
                    ),
                    dtype=bool,
                    count=row_count,
                )
            if selected_ranks is not None:
                keep &= np.isin(all_ranks, list(selected_ranks))
            selected_rows = np.flatnonzero(keep)

            def read_selected(name):
                dataset = index[name]
                if len(selected_rows):
                    return dataset[selected_rows]
                return np.empty(0, dtype=dataset.dtype)

            region_ids = all_region_ids[selected_rows]
            row_ranks = all_ranks[selected_rows]
            counts = read_selected("event_counts")
            source_lines = read_selected("source_lines")
            source_files = read_selected("source_files")
            source_texts = read_selected("source_texts")
            tag_blobs = read_selected("tags")
            summary_statistics = read_selected(_SUMMARY_DATASET)
            exclusive_column = (
                read_selected("exclusive_totals")
                if "exclusive_totals" in index
                else None
            )

            used_region_ids = {int(region_id) for region_id in region_ids}
            region_names = [
                name
                for region_id, name in enumerate(all_region_names)
                if region_id in used_region_ids
            ]
            per_region = {name: {} for name in region_names}
            for row, (region_id, rank, count) in enumerate(
                zip(region_ids, row_ranks, counts)
            ):
                name = all_region_names[int(region_id)]
                rank = int(rank)
                aggregate = {
                    "count": int(count),
                    **{
                        field: (
                            float(summary_statistics[field][row])
                            if field in {"mean", "m2"}
                            else int(summary_statistics[field][row])
                        )
                        for field in summary_statistics.dtype.names
                    },
                }
                exclusive = (
                    int(exclusive_column[row])
                    if exclusive_column is not None and exclusive_column[row] >= 0
                    else aggregate["total"]
                )
                aggregate["exclusive"] = exclusive
                if exclusive_column is not None and exclusive_column[row] >= 0:
                    exclusive_totals.setdefault(name, {})[rank] = exclusive
                source_line = int(source_lines[row])
                per_region[name][rank] = Region(
                    np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.int64),
                    aggregate=aggregate,
                    source_file=_decode_attribute(source_files[row]) or None,
                    source_lineno=source_line if source_line >= 0 else None,
                    source_text=_decode_attribute(source_texts[row]) or None,
                    tags=tuple(json.loads(_decode_attribute(tag_blobs[row]) or "[]")),
                    event_data_available=False,
                )

        rank_group_names = sorted(
            (name for name in h5file if name.startswith("rank") and name[4:].isdigit()),
            key=lambda name: int(name[4:]),
        )
        for rank_group_name in rank_group_names:
            rank_group = h5file[rank_group_name]
            rank = int(rank_group_name[4:])
            if selected_ranks is not None and rank not in selected_ranks:
                continue
            if verbose:
                print(f"rank_group_name = {rank_group_name!r}")
            if include_likwid and LIKWID_GROUP in rank_group:
                likwid[rank] = _read_likwid_group(rank_group[LIKWID_GROUP])
            if include_line_profile and "line_profile" in rank_group:
                line_profile[rank] = _read_line_profile_group(
                    rank_group["line_profile"]
                )

        recorded_ranks = h5file["rank_region_index/ranks"][()]
        inferred_ranks = int(np.max(recorded_ranks)) + 1 if len(recorded_ranks) else 0
        num_ranks = (
            max(int(metadata.get("mpi_size", 1)), inferred_ranks)
            if len(recorded_ranks)
            else 0
        )

    regions = {
        name: MPIRegion(name=name, regions=per_region[name]) for name in region_names
    }
    return {
        "regions": regions,
        "metadata": metadata,
        "num_ranks": num_ranks,
        "likwid": likwid,
        "line_profile": line_profile,
        "file_path": file_path,
        "exclusive_totals": exclusive_totals,
        "event_data_available": False,
    }


def read_h5(file_path: str | Path, verbose: bool = False) -> ProfilingResults:
    """
    Load a merged profiling file for post-processing.

    The discoverable spelling of
    :meth:`ProfilingResults.from_h5
    <scope_profiler.results.ProfilingResults.from_h5>`; the two are
    interchangeable::

        from scope_profiler import read_h5

        results = read_h5("profiling_data.h5")
        results.print_summary()

        solve = results["solve"]        # same as results.get_region("solve")
        solve[0].average_duration       # rank 0, in seconds

    Parameters
    ----------
    file_path : str | Path
        Path to the merged HDF5 file containing profiling data.
    verbose : bool, optional
        Print each rank group as it is read (default: False).

    Returns
    -------
    ProfilingResults
        The run's profiling data. All durations are reported in seconds.

    Raises
    ------
    FileNotFoundError
        If the specified HDF5 file does not exist.
    """
    return ProfilingResults.from_h5(file_path, verbose=verbose)


def read_h5_summary(
    file_path: str | Path,
    verbose: bool = False,
    *,
    fallback: bool = True,
    include_likwid: bool = True,
    include_line_profile: bool = True,
    regions: str | list[str] | tuple[str, ...] | None = None,
    ranks: int | list[int] | tuple[int, ...] | None = None,
) -> ProfilingResults:
    """Load fixed-size statistics, optionally falling back for older files.

    The returned object supports metadata and scalar region statistics. Event,
    timeline, call-stack and percentile APIs intentionally have no per-call
    data; use :func:`read_h5` when those are required.
    """
    try:
        return ProfilingResults(
            **load_h5_summary(
                file_path,
                verbose=verbose,
                include_likwid=include_likwid,
                include_line_profile=include_line_profile,
                regions=regions,
                ranks=ranks,
            )
        )
    except SummaryDataUnavailable:
        if not fallback:
            raise
        results = read_h5(file_path, verbose=verbose)
        selected_names = _selection_set(regions, coerce=str)
        selected_ranks = _selection_set(ranks, coerce=int)
        if selected_names is not None or selected_ranks is not None:
            results = copy(results)
            results._region_dict = {
                name: MPIRegion(
                    name=name,
                    regions={
                        rank: region
                        for rank, region in mpi_region.regions.items()
                        if selected_ranks is None or rank in selected_ranks
                    },
                )
                for name, mpi_region in results._region_dict.items()
                if selected_names is None or name in selected_names
            }
            results._region_dict = {
                name: region
                for name, region in results._region_dict.items()
                if region.regions
            }
        if not include_likwid or selected_ranks is not None:
            results._likwid = (
                {
                    rank: data
                    for rank, data in results._likwid.items()
                    if selected_ranks is None or rank in selected_ranks
                }
                if include_likwid
                else {}
            )
        if not include_line_profile or selected_ranks is not None:
            results._line_profile = (
                {
                    rank: data
                    for rank, data in results._line_profile.items()
                    if selected_ranks is None or rank in selected_ranks
                }
                if include_line_profile
                else {}
            )
        return results
