"""Parsing of merged HDF5 profiling output files.

The public entry points are
:meth:`ProfilingResults.from_h5 <scope_profiler.results.ProfilingResults.from_h5>`
and its module-level twin :func:`read_h5`; both return a plain
:class:`~scope_profiler.results.ProfilingResults`. This module holds only the
HDF5 parsing that turns a merged file into that object's constructor
arguments.
"""

from pathlib import Path

import h5py
import numpy as np

from scope_profiler.likwid_data import LIKWID_GROUP, LikwidRegionResult
from scope_profiler.h5schema import migrate_schema, read_schema_version
from scope_profiler.mpi_region import MPIRegion
from scope_profiler.region import Region
from scope_profiler.results import ProfilingResults


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
    with h5py.File(file_path, "r") as f:
        schema_version = read_schema_version(f)
        migrate_schema(f, schema_version)
        if "metadata" in f:
            metadata = {
                key: _decode_attribute(value)
                for key, value in f["metadata"].attrs.items()
            }

        # Iterate over the rank groups in rank order. h5py yields them in
        # name order, which puts "rank10" before "rank2"; pooled statistics
        # sum per-rank arrays in this order, so a stable numeric order is what
        # makes the numbers reproducible and identical to the ones
        # ProfileManager.finalize(return_results=True) computes in memory.
        rank_group_names = sorted(
            (name for name in f if name != "metadata"),
            key=lambda name: int(name.replace("rank", "")),
        )
        for rank_group_name in rank_group_names:
            rank_group = f[rank_group_name]
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

            if "regions" not in rank_group:
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
