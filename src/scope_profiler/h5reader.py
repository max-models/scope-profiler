"""Reader for merged HDF5 profiling output files."""

import re
from pathlib import Path
from typing import Iterator, List

import h5py
import numpy as np

from scope_profiler.mpi_region import MPIRegion
from scope_profiler.region import Region


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


class ProfilingH5Reader:
    """
    Reads profiling data stored by ProfileRegion in an HDF5 file.

    The reader behaves like an ordered mapping of region name to
    :class:`~scope_profiler.mpi_region.MPIRegion`::

        reader = ProfilingH5Reader("profiling_data.h5")
        for region in reader:
            print(region.name, region.total_duration)

        solve = reader["solve"]        # same as reader.get_region("solve")
        solve[0].average_duration      # rank 0, in seconds

    All durations are reported in seconds.
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
        self._file_path = Path(file_path)
        self._num_ranks = 0
        self._metadata: dict = {}
        if not self.file_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.file_path}")

        # Read the file
        _region_dict = {}
        region_names = []
        with h5py.File(self.file_path, "r") as f:
            if "metadata" in f:
                self._metadata = {
                    key: _decode_attribute(value)
                    for key, value in f["metadata"].attrs.items()
                }

            # Iterate over all rank groups
            for rank_group_name, rank_group in f.items():
                if rank_group_name == "metadata":
                    continue
                self._num_ranks += 1
                if verbose:
                    print(f"{rank_group_name = }")
                    print(rank_group_name, rank_group)
                rank = int(rank_group_name.replace("rank", ""))
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

        self._region_dict = {}

        for region_name in region_names:
            self._region_dict[region_name] = MPIRegion(
                name=region_name, regions=_region_dict[region_name]
            )

    def get_region(self, region_name: str) -> MPIRegion:
        """
        Retrieve profiling data for a specific region.

        Parameters
        ----------
        region_name : str
            Name of the region to retrieve.

        Returns
        -------
        Region
            Region object containing profiling data for all ranks.

        Raises
        ------
        KeyError
            If the specified region name does not exist.
        """
        try:
            return self._region_dict[region_name]
        except KeyError:
            raise KeyError(
                f"No region named {region_name!r} in {self.file_path}. "
                f"Available regions: {self.region_names}"
            ) from None

    @property
    def region_names(self) -> List[str]:
        """Names of all regions in the file, in order of appearance."""
        return list(self._region_dict)

    def summary(
        self,
        include: list[str] | str | None = None,
        exclude: list[str] | str | None = None,
    ) -> List[dict]:
        """
        Summarize every region, aggregated over ranks.

        Parameters
        ----------
        include, exclude : list of str or str, optional
            Regex patterns selecting which regions to summarize, matched as in
            :meth:`get_regions`.

        Returns
        -------
        List[dict]
            One dict per region (see
            :meth:`~scope_profiler.mpi_region.MPIRegion.get_summary`), ordered
            by first start time. Durations are in seconds.
        """
        return [
            region.get_summary()
            for region in self.get_regions(include=include, exclude=exclude)
        ]

    def to_dataframe(
        self,
        include: list[str] | str | None = None,
        exclude: list[str] | str | None = None,
        per_rank: bool = False,
    ):
        """
        Return the region summaries as a pandas DataFrame.

        Parameters
        ----------
        include, exclude : list of str or str, optional
            Regex patterns selecting which regions to include, matched as in
            :meth:`get_regions`.
        per_rank : bool, optional
            If True, emit one row per (region, rank) with a ``rank`` column
            instead of one aggregated row per region (default: False).

        Returns
        -------
        pandas.DataFrame
            Region statistics, with durations in seconds.

        Raises
        ------
        ImportError
            If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "to_dataframe() requires pandas. Install scope-profiler[pproc] "
                "or pandas directly."
            ) from exc

        regions = self.get_regions(include=include, exclude=exclude)
        if not per_rank:
            return pd.DataFrame(region.get_summary() for region in regions)

        rows = []
        for region in regions:
            for rank in region.ranks:
                rows.append(
                    {"name": region.name, "rank": rank, **region[rank].get_summary()}
                )
        return pd.DataFrame(rows)

    def events(
        self,
        include: list[str] | str | None = None,
        exclude: list[str] | str | None = None,
        ranks: list[int] | int | None = None,
        relative: bool = True,
    ) -> List[dict]:
        """
        Return one dict per recorded call, across all regions and ranks.

        This is the long-form ("tidy") view to build custom plots from: one
        row per call rather than one per region.

        Parameters
        ----------
        include, exclude : list of str or str, optional
            Regex patterns selecting which regions to include, matched as in
            :meth:`get_regions`.
        ranks : list of int or int, optional
            Restrict to these ranks (default: all).
        relative : bool, optional
            If True (default), timestamps are measured from the first region
            entry in the file, so the timeline starts at zero. If False, the
            raw monotonic-clock timestamps are returned; those are only
            comparable within a single run.

        Returns
        -------
        List[dict]
            Entries with keys ``name``, ``rank``, ``call_index``, ``start``,
            ``end`` and ``duration``, in seconds, ordered by region (as in
            :meth:`get_regions`) then rank then call order.

        Examples
        --------
        >>> for event in reader.events(include="solve"):  # doctest: +SKIP
        ...     print(event["rank"], event["start"], event["duration"])
        """
        origin = self.minimum_start_time if relative else 0.0
        events = []
        for region in self.get_regions(include=include, exclude=exclude):
            events.extend(region.events(ranks=ranks, origin=origin))
        return events

    def to_events_dataframe(
        self,
        include: list[str] | str | None = None,
        exclude: list[str] | str | None = None,
        ranks: list[int] | int | None = None,
        relative: bool = True,
    ):
        """
        Return every recorded call as a pandas DataFrame (one row per call).

        Parameters
        ----------
        include, exclude, ranks, relative
            As in :meth:`events`.

        Returns
        -------
        pandas.DataFrame
            Columns ``name``, ``rank``, ``call_index``, ``start``, ``end``,
            ``duration``, with times in seconds.

        Raises
        ------
        ImportError
            If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "to_events_dataframe() requires pandas. Install "
                "scope-profiler[pproc] or pandas directly."
            ) from exc

        columns = ["name", "rank", "call_index", "start", "end", "duration"]
        events = self.events(
            include=include, exclude=exclude, ranks=ranks, relative=relative
        )
        return pd.DataFrame(events, columns=columns)

    def call_stack(
        self,
        rank: int = 0,
        include: list[str] | str | None = None,
        exclude: list[str] | str | None = None,
        relative: bool = True,
    ) -> List[dict]:
        """
        Reconstruct the nested call stack for one rank.

        Regions record no call graph, so nesting is recovered from timestamp
        containment - the same reconstruction the flame chart, the ``.prof``
        export and the speedscope export use.

        Parameters
        ----------
        rank : int, optional
            Rank whose calls to reconstruct (default: 0).
        include, exclude : list of str or str, optional
            Regex patterns selecting which regions to include, matched as in
            :meth:`get_regions`.
        relative : bool, optional
            If True (default), timestamps start at zero; see :meth:`events`.

        Returns
        -------
        List[dict]
            One entry per call, parents before children, with keys ``name``,
            ``start``, ``end``, ``duration``, ``depth`` and ``parent`` (the
            index of the enclosing call in this list, or None). See
            :func:`~scope_profiler.call_stack.build_call_stack`.
        """
        from scope_profiler.call_stack import build_call_stack

        return build_call_stack(
            self.get_regions(include=include, exclude=exclude),
            rank=rank,
            origin=self.minimum_start_time if relative else 0.0,
        )

    def print_summary(
        self,
        include: list[str] | str | None = None,
        exclude: list[str] | str | None = None,
        ranks: list[int] | None = None,
        sort: str = "total",
        title: str | None = None,
        stream=None,
    ) -> None:
        """
        Print a region summary table, aggregated over ranks.

        Renders the same table as ``scope-profiler inspect`` and the summary
        printed by ``ProfileManager.finalize()``.

        Parameters
        ----------
        include, exclude : list of str or str, optional
            Regex patterns selecting which regions to print, matched as in
            :meth:`get_regions`.
        ranks : list of int, optional
            Restrict the statistics to these ranks (default: all).
        sort : str, optional
            Column to order by: ``total`` (default), ``calls``, ``avg``,
            ``max`` or ``name``.
        title : str, optional
            Heading above the table (default: the file path and rank count).
        stream : file-like, optional
            Where to write (default: stdout).
        """
        from scope_profiler.summary import print_region_table, region_rows

        rows = region_rows(
            self, include=include, exclude=exclude, ranks=ranks, sort=sort
        )
        if title is None:
            title = f"{self.file_path}  ({self.num_ranks} rank(s))"
        print_region_table(rows, title=title, stream=stream)

    def __getitem__(self, region_name: str) -> MPIRegion:
        """Get a region by name; see :meth:`get_region`."""
        return self.get_region(region_name)

    def __contains__(self, region_name: str) -> bool:
        """Whether a region with this name exists in the file."""
        return region_name in self._region_dict

    def __iter__(self) -> Iterator[MPIRegion]:
        """Iterate over all regions, in order of appearance."""
        return iter(self._region_dict.values())

    def __len__(self) -> int:
        """Number of regions in the file."""
        return len(self._region_dict)

    @property
    def file_path(self) -> Path:
        """
        Get the path to the HDF5 file.

        Returns
        -------
        Path
            The file path as a pathlib.Path object.
        """
        return self._file_path

    @property
    def metadata(self) -> dict:
        """
        Get environment metadata for the run (gathered from rank 0).

        Returns
        -------
        dict
            Metadata dict (hostname, OpenMP thread count, platform, versions,
            etc.), or an empty dict if the file predates metadata collection.
        """
        return self._metadata

    @property
    def num_ranks(self) -> int:
        """
        Get the number of ranks recorded in the profiling data.

        Returns
        -------
        int
            Number of ranks.
        """
        return self._num_ranks

    @property
    def minimum_start_time(self) -> float:
        """
        Get the minimum start time across all regions and ranks.

        This is the origin of the timeline: subtract it from any timestamp to
        get seconds since the first region entry. Regions without timing
        (profiled with ``time_trace=False``) are ignored; the result is 0.0
        if no region recorded any.

        Returns
        -------
        float
            Minimum start time in seconds.
        """
        starts = [
            region.first_start_time
            for region in self.get_regions()
            if region.has_timing
        ]
        return min(starts) if starts else 0.0

    @property
    def maximum_end_time(self) -> float:
        """
        Get the maximum end time across all regions and ranks.

        Returns
        -------
        float
            Maximum end time in seconds, or 0.0 if no region recorded timing.
        """
        ends = [
            region.last_end_time for region in self.get_regions() if region.has_timing
        ]
        return max(ends) if ends else 0.0

    @property
    def time_span(self) -> float:
        """
        Wall-clock seconds between the first region entry and the last exit.

        Returns
        -------
        float
            Duration of the profiled window in seconds, or 0.0 if no region
            recorded timing.
        """
        if not any(region.has_timing for region in self.get_regions()):
            return 0.0
        return self.maximum_end_time - self.minimum_start_time

    def get_regions(
        self,
        include: list[str] | str | None = None,
        exclude: list[str] | str | None = None,
    ) -> List[MPIRegion]:
        """Get a list of all regions in order of appearance.

        Returns
        -------
        List[Region]
            List of Region objects.
        """

        if isinstance(include, str):
            include = [include]
        if isinstance(exclude, str):
            exclude = [exclude]

        regions = []

        # Collect regions based on include/exclude filters
        for region_name, region in self._region_dict.items():
            # print(f"{region_name = } {region = }")
            # Match with regex patterns if provided
            if include is not None:
                if not any([re.match(pattern, region_name) for pattern in include]):
                    continue
            if exclude is not None:
                if any([re.match(pattern, region_name) for pattern in exclude]):
                    continue

            regions.append(region)

        # Sort regions based on first start time across all ranks
        regions.sort(
            key=lambda r: min(region.first_start_time for region in r.regions.values())
        )

        return regions

    def __repr__(self) -> str:
        """
        Return a string representation of all regions and their profiling statistics.

        Returns
        -------
        str
            Formatted string containing profiling data for all regions.
        """
        return (
            f"<ProfilingH5Reader {self.file_path.name!r}: "
            f"{len(self._region_dict)} region(s), {self._num_ranks} rank(s)>"
        )
