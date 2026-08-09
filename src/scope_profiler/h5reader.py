"""Reader for merged HDF5 profiling output files."""

from pathlib import Path

import h5py
import numpy as np

from scope_profiler.likwid_data import LIKWID_GROUP, LikwidRegionResult
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


class ProfilingH5Reader(ProfilingResults):
    """
    Reads profiling data stored by ProfileRegion in an HDF5 file.

    The reader behaves like an ordered mapping of region name to
    :class:`~scope_profiler.mpi_region.MPIRegion`::

        reader = ProfilingH5Reader("profiling_data.h5")
        for region in reader:
            print(region.name, region.total_duration)

        solve = reader["solve"]        # same as reader.get_region("solve")
        solve[0].average_duration      # rank 0, in seconds

    All the analysis methods live on
    :class:`~scope_profiler.results.ProfilingResults`, which
    ``ProfileManager.finalize(return_results=True)`` also returns without
    going through a file. All durations are reported in seconds.
    """

    def __init__(
        self,
        file_path: str | Path,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the HDF5 reader by loading profiling data from the specified file.

        Parameters
        ----------
        file_path : str | Path
            Path to the HDF5 file containing profiling data.

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
        if not file_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {file_path}")

        # Read the file
        _region_dict = {}
        region_names = []
        with h5py.File(file_path, "r") as f:
            if "metadata" in f:
                metadata = {
                    key: _decode_attribute(value)
                    for key, value in f["metadata"].attrs.items()
                }

            # Iterate over all rank groups
            for rank_group_name, rank_group in f.items():
                if rank_group_name == "metadata":
                    continue
                num_ranks += 1
                if verbose:
                    print(f"{rank_group_name = }")
                    print(rank_group_name, rank_group)
                rank = int(rank_group_name.replace("rank", ""))

                if LIKWID_GROUP in rank_group:
                    likwid[rank] = _read_likwid_group(rank_group[LIKWID_GROUP])

                if "regions" not in rank_group:
                    continue
                regions_group = rank_group["regions"]

                for region_name, region_grp in regions_group.items():
                    region_names.append(region_name)
                    if "start_times" in region_grp:
                        starts = region_grp["start_times"][()]
                        ends = region_grp["end_times"][()]
                    else:
                        # Count-only region (time_trace=False): the call count
                        # is stored as an attribute, with no timestamps.
                        starts = np.empty(0, dtype=np.int64)
                        ends = np.empty(0, dtype=np.int64)
                    region = Region(
                        starts, ends, num_calls=region_grp.attrs.get("num_calls")
                    )
                    # Merge if region already exists (from another rank)
                    if region_name in _region_dict:
                        _region_dict[region_name][rank] = region
                    else:
                        _region_dict[region_name] = {rank: region}

        regions = {
            region_name: MPIRegion(name=region_name, regions=_region_dict[region_name])
            for region_name in region_names
        }

        super().__init__(
            regions,
            metadata=metadata,
            num_ranks=num_ranks,
            likwid=likwid,
            file_path=file_path,
        )
