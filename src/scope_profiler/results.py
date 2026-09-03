"""Post-processing API over a set of profiling regions.

This is the analysis layer, deliberately independent of where the data came
from: :meth:`ProfilingResults.from_h5` builds one of these from a merged HDF5
file, while :meth:`ProfileManager.finalize(return_results=True)
<scope_profiler.profile_manager.ProfileManager.finalize>` builds the same thing
straight out of the in-memory buffers, without a round trip through disk. Both
give back the same type, and everything downstream (summaries, dataframes, the
plotting functions, the exporters) cannot tell them apart.
"""

import functools
import os
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from scope_profiler.likwid_data import LikwidRegionResult
from scope_profiler.mpi_region import MPIRegion
from scope_profiler.region import NS_PER_SECOND, EventDataUnavailableError


class ProfilingResults:
    """
    The profiling data of one run, as an ordered mapping of region name to
    :class:`~scope_profiler.mpi_region.MPIRegion`::

        results = ProfileManager.finalize(return_results=True)
        for region in results:
            print(region.name, region.total_duration)

        solve = results["solve"]        # same as results.get_region("solve")
        solve[0].average_duration       # rank 0, in seconds

    The same object comes back from a merged output file, via
    :meth:`from_h5` (or its module-level twin
    :func:`~scope_profiler.h5reader.read_h5`)::

        results = ProfilingResults.from_h5("profiling_data.h5")

    All durations are reported in seconds.
    """

    # Declared at class scope so the exclusive-duration machinery below can
    # be read by a type checker without depending on __init__ appearing first.
    _exclusive_populated: bool

    def _populate_exclusive_durations(self) -> None:
        """Derive per-call exclusive durations from all recorded intervals.

        Runs at most once per result set, on first access of any exclusive
        duration (see :meth:`Region._resolved_exclusive_durations`), and
        fills in every rank and region: the nesting of one region's calls is
        only visible against all the others recorded on the same rank, so
        there is nothing cheaper to compute for a single region alone.
        """
        from scope_profiler.call_stack import build_call_arrays

        if self._exclusive_populated:
            return
        # Set before the loop: build_call_arrays reads only start/end times,
        # but a region reached through it must not re-enter this.
        self._exclusive_populated = True

        for rank in sorted(
            {rank for region in self._region_dict.values() for rank in region.ranks},
        ):
            # Columns, not one dict per call: this runs over every event in
            # the run, which for a long simulation is tens of millions.
            arrays = build_call_arrays(self._region_dict.values(), rank)
            for row, name in enumerate(arrays.names):
                mine = arrays.region_index == row
                durations = self._region_dict[name].regions[rank]._exclusive_buffer()
                durations[arrays.call_index[mine]] = arrays.exclusive_ns[mine]

    def _metadata_time(self, key: str) -> float | None:
        """Read a ``*_time_ns`` metadata field as seconds, or None if unusable."""
        value = self._metadata.get(key)
        if value is None:
            return None
        try:
            return float(value) / NS_PER_SECOND
        except (TypeError, ValueError):
            return None

    def __init__(
        self,
        regions: dict[str, MPIRegion],
        metadata: dict | None = None,
        num_ranks: int | None = None,
        likwid: dict[int, dict[str, LikwidRegionResult]] | None = None,
        perf_events: dict[int, dict] | None = None,
        line_profile: dict[int, list] | None = None,
        file_path: str | Path = "",
        is_root: bool = True,
        exclusive_totals: dict[str, dict[int, int]] | None = None,
        event_data_available: bool = True,
        threads: dict[int, list] | None = None,
        tasks: dict[int, list] | None = None,
    ) -> None:
        """
        Assemble a result set from already-loaded regions.

        Parameters
        ----------
        regions : dict
            Region name -> :class:`~scope_profiler.mpi_region.MPIRegion`, in
            order of appearance.
        metadata : dict, optional
            Environment metadata for the run (see :attr:`metadata`).
        num_ranks : int, optional
            Number of ranks the run used (default: the number of distinct
            ranks appearing in ``regions``).
        likwid : dict, optional
            Rank -> {tag: :class:`~scope_profiler.likwid_data.LikwidRegionResult`}.
        file_path : str or Path, optional
            The run's output file. Used for labelling and error messages; it
            need not exist for in-memory results.
        is_root : bool, optional
            Whether this rank holds the run's data (default: True). See
            :attr:`is_root`.
        exclusive_totals : dict, optional
            Region name -> {rank: total exclusive nanoseconds}, as computed by
            the run itself and stored in its output file. Only pass values
            computed against *these* regions: exclusive time is defined
            against everything else recorded on the same rank, so a total
            carried over from a different result set would be wrong. Omitting
            them costs nothing but a call-stack reconstruction on first use.
        threads : dict, optional
            Rank -> list of :class:`~scope_profiler.concurrency.ThreadInfo`,
            for a run profiled with ``track_threads``. See :attr:`threads`.
        tasks : dict, optional
            Rank -> list of :class:`~scope_profiler.concurrency.TaskInfo`,
            for a run profiled with ``track_async``. See :attr:`tasks`.
        """
        self._is_root = is_root
        self._region_dict = dict(regions)
        self._metadata = dict(metadata or {})
        self._likwid = dict(likwid or {})
        self._perf_events = dict(perf_events or {})
        self._line_profile = dict(line_profile or {})
        self._file_path = Path(file_path)
        self._event_data_available = bool(event_data_available)
        self._threads = {
            int(rank): list(rows) for rank, rows in (threads or {}).items()
        }
        self._tasks = {int(rank): list(rows) for rank, rows in (tasks or {}).items()}
        if num_ranks is None:
            ranks = {rank for region in self._region_dict.values() for rank in region}
            num_ranks = len(ranks)
        self._num_ranks = num_ranks

        # Recorded by setup(); absent in files written before it existed, and
        # in any file whose run did not reach finalize(). Everything
        # downstream falls back to the first region entry, so an unreadable
        # value is ignored rather than fatal: a file is worth reading for its
        # timings even if this field is not.
        self._run_start_time = self._metadata_time("start_time_ns")
        # Recorded as the first thing finalize() does; absent in files from
        # before it existed, or from a run that set deactivate_file_output
        # and never called finalize(return_results=True) either.
        self._finalize_time = self._metadata_time("finalize_time_ns")
        # Deferred, not skipped: every region is told how to reconstruct its
        # nesting, and the first one asked for exclusive time does it for all
        # of them. Building the call stack up front costs more than the whole
        # rest of the load (4.4s of a 6.4s read on a 2.6M-event file), and
        # only some callers need it -- see _populate_exclusive_durations.
        self._exclusive_populated = False
        totals = exclusive_totals or {}
        for name, region in self._region_dict.items():
            for rank, rank_region in region.regions.items():
                # Attach first: this drops whatever a previous owner of the
                # region computed, including its stored total.
                rank_region._attach_exclusive_resolver(
                    self._populate_exclusive_durations,
                )
                stored = totals.get(name, {}).get(rank)
                if stored is not None:
                    rank_region._set_exclusive_total(stored)

    @classmethod
    def from_h5(
        cls,
        file_path: str | Path,
        verbose: bool = False,
    ) -> "ProfilingResults":
        """
        Load a merged profiling file written by
        :meth:`ProfileManager.finalize
        <scope_profiler.profile_manager.ProfileManager.finalize>`::

            results = ProfilingResults.from_h5("profiling_data.h5")
            results.print_summary()

        :func:`scope_profiler.read_h5` is the same thing under a shorter name.

        Parameters
        ----------
        file_path : str | Path
            Path to the merged HDF5 file containing profiling data.
        verbose : bool, optional
            Print each rank group as it is read (default: False).

        Returns
        -------
        ProfilingResults
            The run's profiling data, of whichever class this was called on.

        Raises
        ------
        FileNotFoundError
            If the specified HDF5 file does not exist.
        """
        # Imported here so the analysis layer does not pull in h5py, and to
        # keep the dependency one-way: h5reader imports this module.
        from scope_profiler.h5reader import load_h5

        return cls(**load_h5(file_path, verbose=verbose))

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
                f"Available regions: {self.region_names}",
            ) from None

    @property
    def region_names(self) -> list[str]:
        """Names of all regions, in order of appearance."""
        return list(self._region_dict)

    @staticmethod
    def _summary_row(region: MPIRegion) -> dict:
        """Return the public aggregate summary, including rich statistics."""
        return {
            **region.get_summary(),
            "inclusive_duration": region.inclusive_duration,
            "exclusive_duration": region.exclusive_duration,
            "tags": region.tags,
            "p50_duration": region.p50_duration,
            "p95_duration": region.p95_duration,
            "p99_duration": region.p99_duration,
            "rank_imbalance": region.rank_imbalance,
            "rank_imbalance_pct": region.rank_imbalance_pct,
            # Short aliases match the summary-table column names.
            "p50": region.p50_duration,
            "p95": region.p95_duration,
            "p99": region.p99_duration,
            "imbalance": region.rank_imbalance_pct,
        }

    def summary(
        self,
        include: list[str] | str | None = None,
        exclude: list[str] | str | None = None,
    ) -> list[dict]:
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
            by first start time. ``inclusive_duration`` includes nested
            regions; ``exclusive_duration`` excludes them. Durations are in
            seconds.
        """
        return [
            self._summary_row(region)
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
                "or pandas directly.",
            ) from exc

        regions = self.get_regions(include=include, exclude=exclude)
        if not per_rank:
            return pd.DataFrame(self._summary_row(region) for region in regions)

        rows = []
        for region in regions:
            for rank in region.ranks:
                rows.append(
                    {
                        "name": region.name,
                        "rank": rank,
                        **region[rank].get_summary(),
                        "inclusive_duration": region[rank].inclusive_duration,
                        "exclusive_duration": region[rank].exclusive_duration,
                        "p50_duration": region[rank].p50_duration,
                        "p95_duration": region[rank].p95_duration,
                        "p99_duration": region[rank].p99_duration,
                    },
                )
        return pd.DataFrame(rows)

    def _require_event_data(self, operation: str) -> None:
        """Reject event-dependent operations on fixed-size summary results."""
        if not self._event_data_available:
            raise EventDataUnavailableError(
                f"{operation}() requires per-call events, but this profile was "
                "loaded with read_h5_summary(); load it with read_h5()",
            )

    def events(
        self,
        include: list[str] | str | None = None,
        exclude: list[str] | str | None = None,
        ranks: list[int] | int | None = None,
        relative: bool = True,
        origin: float | None = None,
    ) -> list[dict]:
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
            If True (default), timestamps are measured from
            :attr:`time_origin` — the start time registered by
            ``ProfileManager.setup()``, or the first region entry for runs
            without one — so the timeline starts at zero. If False, the raw
            monotonic-clock timestamps are returned; those are only comparable
            within a single run.
        origin : float, optional
            Measure from this timestamp instead, in seconds on the recording
            clock. Overrides ``relative``; pass
            ``origin=results.minimum_start_time`` to zero the timeline on the
            first region entry regardless of what the run registered.

        Returns
        -------
        List[dict]
            Entries with keys ``name``, ``rank``, ``call_index``, ``start``,
            ``end`` and ``duration``, in seconds, ordered by region (as in
            :meth:`get_regions`) then rank then call order. Entries also carry
            ``gpu_duration`` when CUDA-event timing was enabled for that
            region.

        Examples
        --------
        >>> for event in results.events(include="solve"):  # doctest: +SKIP
        ...     print(event["rank"], event["start"], event["duration"])
        """
        self._require_event_data("events")
        if origin is None:
            origin = self.time_origin if relative else 0.0
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
        origin: float | None = None,
    ):
        """
        Return every recorded call as a pandas DataFrame (one row per call).

        Parameters
        ----------
        include, exclude, ranks, relative, origin
            As in :meth:`events`.

        Returns
        -------
        pandas.DataFrame
            Columns ``name``, ``rank``, ``call_index``, ``start``, ``end`` and
            ``duration``, with times in seconds. A ``gpu_duration`` column is
            included when any selected event has CUDA-event timing.

        Raises
        ------
        ImportError
            If pandas is not installed.
        """
        self._require_event_data("to_events_dataframe")
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "to_events_dataframe() requires pandas. Install "
                "scope-profiler[pproc] or pandas directly.",
            ) from exc

        events = self.events(
            include=include,
            exclude=exclude,
            ranks=ranks,
            relative=relative,
            origin=origin,
        )
        columns = ["name", "rank", "call_index", "start", "end", "duration"]
        if any("gpu_duration" in event for event in events):
            columns.append("gpu_duration")
        return pd.DataFrame(events, columns=columns)

    def call_stack(
        self,
        rank: int = 0,
        include: list[str] | str | None = None,
        exclude: list[str] | str | None = None,
        relative: bool = True,
        origin: float | None = None,
    ) -> list[dict]:
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
        origin : float, optional
            Measure from this timestamp instead; see :meth:`events`.

        Returns
        -------
        List[dict]
            One entry per call, parents before children, with keys ``call_id``,
            ``name``, ``start``, ``end``, ``duration``, ``depth`` and
            ``parent`` (the enclosing call's id, or None). See
            :func:`~scope_profiler.call_stack.build_call_stack`.
        """
        self._require_event_data("call_stack")
        from scope_profiler.call_stack import build_call_stack

        if origin is None:
            origin = self.time_origin if relative else 0.0
        return build_call_stack(
            self.get_regions(include=include, exclude=exclude),
            rank=rank,
            origin=origin,
        )

    def call_graph(self, rank: int = 0, include=None, exclude=None) -> list[dict]:
        """Return call relationships without timestamps or durations.

        New profiles persist ``call_id`` and ``parent_id`` for every Python
        event. Legacy profiles fall back to the timestamp-based call stack.
        The returned nodes contain only ``call_id``, ``parent_id``, ``name``,
        ``call_index`` and ``depth``.

        Filtering renests: a call whose parent was excluded is reported under
        its nearest surviving ancestor, with the depth that implies, exactly
        as :meth:`call_stack` does. Leaving the excluded id in ``parent_id``
        would hand back a graph with edges pointing at nodes that are not in
        it.

        ``call_id`` is unique within this rank, not across the file - each
        rank numbers its own calls.
        """
        retained = {
            region.name for region in self.get_regions(include=include, exclude=exclude)
        }
        on_rank = [region for region in self.get_regions() if rank in region.regions]
        explicit = bool(on_rank) and all(
            region.regions[rank].call_ids is not None for region in on_rank
        )
        if not explicit:
            return [
                {
                    "call_id": call["call_id"],
                    "parent_id": call["parent"],
                    "name": call["name"],
                    "call_index": call["call_index"],
                    "depth": call["depth"],
                }
                for call in self.call_stack(rank=rank, include=include, exclude=exclude)
            ]

        # Every call on the rank, filtered or not: an excluded region in the
        # middle of the chain still has to be walked through to find what a
        # surviving call now hangs off.
        parent_of: dict[int, int | None] = {}
        kept: dict[int, tuple[str, int]] = {}
        for region in on_rank:
            data = region.regions[rank]
            keep_region = region.name in retained
            for index, (call_id, parent_id) in enumerate(
                zip(data.call_ids.tolist(), data.parent_ids.tolist()),
            ):
                parent_of[call_id] = None if parent_id < 0 else parent_id
                if keep_region:
                    kept[call_id] = (region.name, index)

        # A parent always has a smaller id than its child, so one ascending
        # pass resolves whole ancestor chains without recursing or revisiting:
        # by the time a call is reached, its parent's answer is already known.
        nearest: dict[int, int | None] = {}
        depths: dict[int, int] = {}
        for call_id in sorted(parent_of):
            parent_id = parent_of[call_id]
            if parent_id is None:
                ancestor = None
            elif parent_id in kept:
                ancestor = parent_id
            else:
                ancestor = nearest.get(parent_id)
            nearest[call_id] = ancestor
            if call_id in kept:
                depths[call_id] = 0 if ancestor is None else depths[ancestor] + 1

        return [
            {
                "call_id": call_id,
                "parent_id": nearest[call_id],
                "name": kept[call_id][0],
                "call_index": kept[call_id][1],
                "depth": depths[call_id],
            }
            for call_id in sorted(kept)
        ]

    def print_summary(
        self,
        include: list[str] | str | None = None,
        exclude: list[str] | str | None = None,
        ranks: list[int] | None = None,
        sort: str = "start",
        title: str | None = None,
        stream=None,
        suppress_notes: bool = False,
        columns: list[str] | str | None = None,
        percentage_mode: str = "coverage",
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
            Column to order by: ``start`` (default), ``total``, ``calls``,
            ``avg``, ``min``, ``max``, ``std`` or ``name``.
        title : str, optional
            Heading above the table (default: the file path and rank count).
        stream : file-like, optional
            Where to write (default: stdout).
        columns : list of str or str, optional
            Region summary columns to print. Defaults to ``region``,
            ``calls``, ``percent``, ``total`` and ``avg``. The
            percentage is relative to ``scope_profiler.session``. Use
            ``region`` for the region-name column.
        percentage_mode : {"coverage", "exclusive"}, optional
            Quantity used for ``% session``. Defaults to wall-clock coverage;
            use ``exclusive`` to attribute time after nested regions.

        Notes
        -----
        Does nothing on a non-root rank (see :attr:`is_root`), so a parallel
        script can call it unguarded and print the table once.
        """
        from scope_profiler.summary import print_region_table, region_rows

        if not self._is_root:
            return

        rows = region_rows(
            self,
            include=include,
            exclude=exclude,
            ranks=ranks,
            sort=sort,
            percentage_mode=percentage_mode,
        )
        if title is None:
            title = self.default_title()
        print_region_table(
            rows,
            title=title,
            stream=stream,
            suppress_notes=suppress_notes,
            total_time=self.total_time,
            columns=columns,
            percentage_mode=percentage_mode,
            file_path=self.file_path,
        )

    def default_title(self) -> str:
        """
        Heading naming this run, for a summary table.

        The label leads when the run has one, since that is the name the user
        chose; the file path follows either way, because when several runs are
        being compared it is what tells them apart on disk.

        Returns
        -------
        str
            e.g. ``"results/run_a.h5 (128 ranks)"``.
        """
        rank_label = "rank" if self.num_ranks == 1 else "ranks"
        relative_path = os.path.relpath(self.file_path)
        title = f"{relative_path} ({self.num_ranks} {rank_label})"
        if self.label is not None:
            title = f"{self.label} - {title}"
        return title

    @property
    def file_path(self) -> Path:
        """
        The run's output file.

        Returns
        -------
        Path
            The file path as a pathlib.Path object. For results taken straight
            from memory this is the path ``setup()`` was configured with, which
            need not exist on disk.
        """
        return self._file_path

    @property
    def label(self) -> str | None:
        """
        The run's label, as given to ``ProfileManager.setup(label=...)``.

        Returns
        -------
        str or None
            The label, or None for a run that was not given one. Use
            :attr:`display_label` to name the run in output regardless.
        """
        label = self._metadata.get("label")
        return str(label) if label else None

    @label.setter
    def label(self, value: str | None) -> None:
        """Rename this run for the report being produced.

        What ``scope-profiler plot --label`` does: the file on disk keeps the
        label the run was given (if any), while everything downstream of here
        uses the new one. Setting it to None or "" restores the fallback to the
        file stem.
        """
        if value:
            self._metadata["label"] = str(value)
        else:
            self._metadata.pop("label", None)

    @property
    def display_label(self) -> str:
        """
        What to call this run in charts, tables and exports.

        The :attr:`label` when the run has one, and otherwise the stem of its
        output file --- which is what post-processing named runs by before
        labels existed, and remains the default.

        Returns
        -------
        str
            A non-empty name for the run.
        """
        return self.label or self._file_path.stem

    @property
    def is_root(self) -> bool:
        """
        Whether this rank holds the run's data.

        Always True for a file that was read back, and for serial runs. Under
        MPI, ``finalize(return_results=True)`` gathers everything on rank 0, so
        only rank 0's results are the root ones; the others come back empty and
        with this False.

        Everything that produces output - :meth:`print_summary`, the ``plot_*``
        functions, the exporters - does nothing for non-root results. That is
        what lets a parallel script call them unguarded and still write each
        figure exactly once, from rank 0. Read it when a script needs to make
        the same distinction for output of its own.

        Returns
        -------
        bool
            True unless this is a non-root rank's share of an MPI run.
        """
        return self._is_root

    @property
    def has_event_data(self) -> bool:
        """Whether per-call timestamps are available to event-based APIs."""
        return self._event_data_available

    @property
    def metadata(self) -> dict:
        """
        Get environment metadata for the run (gathered from rank 0).

        Returns
        -------
        dict
            Metadata dict (hostname, OpenMP thread count, platform, versions,
            etc.), or an empty dict if the run recorded none.
        """
        return self._metadata

    @property
    def threads(self) -> dict[int, list]:
        """Rank -> the threads that recorded calls, in registration order.

        Each entry is a :class:`~scope_profiler.concurrency.ThreadInfo`: name,
        OS ids, when the thread started and ended, and the CPU time it burned.
        Empty unless the run profiled with ``track_threads=True``::

            for rank, threads in results.threads.items():
                for thread in threads:
                    print(rank, thread.name, thread.cpu_time)

        The per-call ``thread_ids`` column of every region indexes into this
        rank's list, so ``results["solve"][0].for_thread(2)`` is the part of
        ``solve`` that ran on ``results.threads[0][2]``.
        """
        return self._threads

    @property
    def tasks(self) -> dict[int, list]:
        """Rank -> the asyncio tasks and greenlets the run followed.

        Each entry is a :class:`~scope_profiler.concurrency.TaskInfo`, whose
        ``running_time`` and ``awaiting_time`` split the lane's life into the
        part it held a thread and the part it was suspended. Empty unless the
        run profiled with ``track_async=True``.
        """
        return self._tasks

    def lane_label(self, lane: int, rank: int = 0) -> str:
        """A human-readable name for one lane of ``rank``.

        Lanes are the stacks a rank's calls were reconstructed into (see
        :func:`~scope_profiler.concurrency.lane_ids`). This turns one back
        into something worth putting on a chart or a speedscope profile
        selector: a task's own name where the lane is a task, the thread's
        name where it is a bare thread.
        """
        lane = int(lane)
        if lane >= 0:
            for task in self._tasks.get(int(rank), []):
                if task.index == lane:
                    thread = self.lane_label(-2 - task.thread_index, rank)
                    tail = task.coro_name.rsplit(".", 1)[-1]
                    return f"{task.name} ({tail}) on {thread}"
            return f"task {lane}"
        if lane == -1:
            return "unknown lane"
        index = -2 - lane
        for thread in self._threads.get(int(rank), []):
            if thread.index == index:
                return thread.name
        return f"thread {index}"

    def thread_summary(self, rank: int = 0) -> list[dict]:
        """Per-thread wall time, CPU time and call count for one rank.

        Wall and CPU time come from the thread table; the call count and the
        time attributed to regions come from the events, so a thread that
        burned CPU outside every profiled region shows the gap directly.

        Returns
        -------
        list of dict
            One entry per thread with keys ``index``, ``name``, ``alive``,
            ``wall_time``, ``cpu_time``, ``num_calls`` and
            ``region_time`` (the summed inclusive duration of that thread's
            top-level calls, so nested regions are not counted twice), in
            thread-table order. Empty when the run did not track threads.
        """
        from scope_profiler.call_stack import build_call_arrays

        threads = self._threads.get(int(rank), [])
        if not threads:
            return []
        arrays = build_call_arrays(self._region_dict.values(), int(rank))
        # Top level per lane, which is what "not counted twice" means once
        # several stacks share a rank.
        roots = arrays.parent < 0
        by_thread: dict[int, list[int]] = {}
        for thread in threads:
            by_thread[thread.index] = [0, 0]
        for region in self._region_dict.values():
            for region_rank, region_data in region.regions.items():
                if region_rank != int(rank) or region_data.thread_ids is None:
                    continue
                for index, count in zip(
                    *np.unique(region_data.thread_ids, return_counts=True),
                ):
                    entry = by_thread.setdefault(int(index), [0, 0])
                    entry[0] += int(count)
        durations = arrays.end_ns - arrays.start_ns
        lane_of_root = arrays.lane[roots]
        root_durations = durations[roots]
        for thread in threads:
            # A thread's lanes are its own bare lane plus every task that ran
            # on it; see concurrency.lane_ids.
            lanes = {-2 - thread.index}
            lanes.update(
                task.index
                for task in self._tasks.get(int(rank), [])
                if task.thread_index == thread.index
            )
            mask = np.isin(lane_of_root, list(lanes))
            by_thread[thread.index][1] = int(root_durations[mask].sum())
        return [
            {
                "index": thread.index,
                "name": thread.name,
                "alive": thread.alive,
                "wall_time": thread.wall_time,
                "cpu_time": thread.cpu_time,
                "num_calls": by_thread[thread.index][0],
                "region_time": by_thread[thread.index][1] / NS_PER_SECOND,
            }
            for thread in threads
        ]

    @property
    def line_profile(self) -> dict[int, list]:
        """Persisted line-profiler records keyed by rank.

        Each record contains ``region``, ``filename``, ``function``,
        ``first_lineno``, ``line_numbers``, ``hits``, ``times`` and ``unit``.
        The elapsed seconds for a line are ``times * unit``.
        """
        return self._line_profile

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
    def has_likwid(self) -> bool:
        """Whether the run recorded LIKWID hardware counter results."""
        return any(self._likwid.values())

    @property
    def likwid_ranks(self) -> list[int]:
        """Ranks that recorded LIKWID results, in ascending order."""
        return sorted(rank for rank, regions in self._likwid.items() if regions)

    def get_likwid_regions(self, rank: int | None = None) -> dict:
        """
        Get the LIKWID marker results of the run.

        Parameters
        ----------
        rank : int, optional
            Return only this rank's regions. By default every rank is
            included, keyed by rank.

        Returns
        -------
        dict
            With ``rank`` given, a mapping of region tag to
            :class:`~scope_profiler.likwid_data.LikwidRegionResult`; otherwise a
            mapping of rank to such a dict. Empty when the run did not use
            LIKWID.

        Examples
        --------
        ::

            results = read_h5("profiling_data.h5")
            for rank, regions in results.get_likwid_regions().items():
                for tag, result in regions.items():
                    for name, values in zip(result.metric_names, result.metrics):
                        print(rank, tag, name, values)
        """
        if rank is None:
            return dict(self._likwid)
        return self._likwid.get(rank, {})

    @property
    def has_perf_events(self) -> bool:
        """Whether this run recorded built-in Linux perf-event counters."""
        return any(self._perf_events.values())

    def get_perf_events(self, rank: int | None = None) -> dict:
        """Return aggregated Linux ``perf_event_open`` counts by region."""
        if rank is None:
            return dict(self._perf_events)
        return self._perf_events.get(rank, {})

    def get_likwid_region(self, tag: str, rank: int = 0) -> LikwidRegionResult:
        """
        Get one region's LIKWID results for a single rank.

        Parameters
        ----------
        tag : str
            LIKWID marker region tag (the profiled region's name).
        rank : int, optional
            Rank to read from (default: 0).

        Returns
        -------
        LikwidRegionResult
            The region's counters, event names and derived metrics.

        Raises
        ------
        KeyError
            If the rank recorded no LIKWID data or has no such region.
        """
        regions = self._likwid.get(rank, {})
        try:
            return regions[tag]
        except KeyError:
            raise KeyError(
                f"No LIKWID region {tag!r} for rank {rank} in {self.file_path}. "
                f"Available regions: {sorted(regions)}",
            ) from None

    def likwid_to_dataframe(self):
        """
        Return every LIKWID event and metric as a tidy pandas DataFrame.

        One row per (rank, region, hardware thread), with a column per event
        and per derived metric, plus the region's LIKWID runtime and call
        count. Regions measured with different event groups simply leave the
        other group's columns empty.

        Returns
        -------
        pandas.DataFrame
            Empty if there is no LIKWID data.

        Raises
        ------
        ImportError
            If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "likwid_to_dataframe() requires pandas. Install "
                "scope-profiler[pproc] or pandas directly.",
            ) from exc

        rows = []
        for rank in self.likwid_ranks:
            for tag, result in self._likwid[rank].items():
                for thread in range(len(result.times)):
                    row = {
                        "rank": rank,
                        "region": tag,
                        "thread": thread,
                        "cpu": (
                            result.cpus[thread] if thread < len(result.cpus) else np.nan
                        ),
                        "group": result.group_name,
                        "time": result.times[thread],
                        "call_count": result.call_counts[thread],
                    }
                    # event_labels, not event_names: a group like MEM_DP
                    # repeats an event across memory channels, and keying by
                    # the bare name would keep only the last channel.
                    for name, values in zip(result.event_labels, result.events):
                        row[name] = values[thread]
                    for name, values in zip(result.metric_names, result.metrics):
                        row[name] = values[thread]
                    rows.append(row)
        return pd.DataFrame(rows)

    def print_likwid_summary(self, stream=None) -> None:
        """
        Print every LIKWID region's events and derived metrics.

        Parameters
        ----------
        stream : file-like, optional
            Destination (default: stdout).
        """
        print_ = print if stream is None else functools.partial(print, file=stream)

        if not self.has_likwid:
            print_(f"No LIKWID data in {self.file_path}")
            return

        for rank in self.likwid_ranks:
            for tag, result in self._likwid[rank].items():
                header = f"rank {rank}  region {tag!r}"
                if result.group_name:
                    header += f"  group {result.group_name}"
                print_(header)
                for thread, cpu in enumerate(result.cpus or range(len(result.times))):
                    print_(
                        f"  cpu {cpu}: {result.call_counts[thread]} call(s), "
                        f"{result.times[thread]:.6f} s",
                    )
                    # Labels rather than raw names, so repeated events show
                    # which counter (memory channel, ...) they came from.
                    width = max(
                        [len(n) for n in result.event_labels + result.metric_names]
                        + [30],
                    )
                    for name, values in zip(result.event_labels, result.events):
                        print_(f"    {name:<{width}s} {values[thread]:>18.4f}")
                    for name, values in zip(result.metric_names, result.metrics):
                        print_(f"    {name:<{width}s} {values[thread]:>18.4f}")

    @property
    def minimum_start_time(self) -> float:
        """
        Get the minimum start time across all regions and ranks.

        This is the origin of the timeline: subtract it from any timestamp to
        get seconds since the first region entry. Regions with no recorded
        calls are ignored; the result is 0.0 if no region recorded any.

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
    def run_start_time(self) -> float | None:
        """
        When the run started, in seconds on the recording clock.

        This is the ``start_time_ns`` metadata field, written by
        ``ProfileManager.setup()`` — by default the moment setup() was called,
        or an earlier instant if one was passed to it.

        Returns
        -------
        float or None
            The registered start time, or None for runs without one (older
            files, or runs that never called setup()). Use
            :attr:`startup_time` for the elapsed time before the first region,
            which stays defined either way.
        """
        return self._run_start_time

    @property
    def time_origin(self) -> float:
        """
        Zero point of the relative timeline, in seconds.

        The registered :attr:`run_start_time` when there is one, and otherwise
        the first region entry (:attr:`minimum_start_time`). This is what
        :meth:`events` and :meth:`call_stack` measure from.

        Note the ``plot_*`` functions instead frame their x axis on the first
        region entry, so that a long gap between ``setup()`` and the first
        region does not fill a chart with empty space. Pass
        ``origin=results.minimum_start_time`` to :meth:`events` to reproduce
        the numbers on a chart's axis.

        Returns
        -------
        float
            Origin timestamp on the recording clock.
        """
        if self._run_start_time is not None:
            return self._run_start_time
        return self.minimum_start_time

    @property
    def startup_time(self) -> float:
        """
        Seconds between the start of the run and the first profiled region.

        Time the instrumentation never saw: imports, reading input, building
        a mesh. Zero when there is no registered start time
        (:attr:`run_start_time` is None), since the run is then only known
        from its first region onwards.

        Returns
        -------
        float
            Elapsed time before the first region entry, in seconds.
        """
        return self.minimum_start_time - self.time_origin

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

    @property
    def finalize_time(self) -> float | None:
        """
        When ``finalize()`` was called, in seconds on the recording clock.

        This is the ``finalize_time_ns`` metadata field, read as the first
        thing ``ProfileManager.finalize()`` does -- before it spends any time
        collecting or writing the run's data, so it marks the moment
        finalize() was reached rather than the moment it returned.

        Returns
        -------
        float or None
            The registered finalize time, or None for runs without one (older
            files, or a run that never reached finalize()).
        """
        return self._finalize_time

    @property
    def total_time(self) -> float | None:
        """
        Wall-clock seconds from ``setup()`` to ``finalize()``.

        Unlike :attr:`time_span` (first region entry to last exit), this
        covers the whole instrumented program: startup work before the first
        region, gaps between regions, and any teardown after the last one but
        before ``finalize()`` is called -- the number to report as "how long
        did the run take" alongside the region breakdown.

        Returns
        -------
        float or None
            :attr:`finalize_time` minus :attr:`run_start_time`, or None if
            either is missing -- an older file, or a run that set up
            profiling (or called finalize) some other way than
            ``ProfileManager.setup()``/``finalize()``.
        """
        if self._run_start_time is None or self._finalize_time is None:
            return None
        return self._finalize_time - self._run_start_time

    def get_regions(
        self,
        include: list[str] | str | None = None,
        exclude: list[str] | str | None = None,
    ) -> list[MPIRegion]:
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
            # Match with regex patterns if provided
            if include is not None and not any(
                re.match(pattern, region_name) for pattern in include
            ):
                continue
            if exclude is not None and any(
                re.match(pattern, region_name) for pattern in exclude
            ):
                continue

            regions.append(region)

        # Sort regions based on first start time across all ranks
        regions.sort(
            key=lambda r: min(region.first_start_time for region in r.regions.values()),
        )

        return regions

    def __getitem__(self, region_name: str) -> MPIRegion:
        """Get a region by name; see :meth:`get_region`."""
        return self.get_region(region_name)

    def __iter__(self) -> Iterator[MPIRegion]:
        """Iterate over all regions, in order of appearance."""
        return iter(self._region_dict.values())

    def __contains__(self, region_name: str) -> bool:
        """Whether a region with this name exists."""
        return region_name in self._region_dict

    def __len__(self) -> int:
        """Number of regions."""
        return len(self._region_dict)

    def __repr__(self) -> str:
        """
        Return a string representation of all regions and their profiling statistics.

        Returns
        -------
        str
            Formatted string containing profiling data for all regions.
        """
        return (
            f"<{type(self).__name__} {self.file_path.name!r}: "
            f"{len(self._region_dict)} region(s), {self._num_ranks} rank(s)>"
        )


def merge_results(*result_sets, label: str | None = None, file_path=None):
    """Combine several result sets into one.

    The case this exists for is a mixed-language run: a Python driver records
    its own regions while the Fortran (or other) code it calls records
    theirs, and what the user wants at the end is *one* profile covering both.
    Ranks line up by number, so rank 3's Python regions and rank 3's Fortran
    regions end up side by side.

    Parameters
    ----------
    *result_sets : ProfilingResults
        The sets to combine. Non-root sets (the empty ones every rank but 0
        gets back under MPI) are ignored, so this can be called unguarded in
        a parallel script.
    label : str, optional
        Name for the combined run. Defaults to the first set's label.
    file_path : str or Path, optional
        Output path to attribute the combined set to. Defaults to the first
        set's.

    Returns
    -------
    ProfilingResults
        One result set holding every region of every input.

    Raises
    ------
    ValueError
        If no result sets were given, or if a region name appears in more than
        one of them. Merging same-named regions would silently double-count a
        Python wrapper and the native region inside it, so the collision has to
        be resolved by the caller -- name the regions apart, for instance with
        a ``"fortran:"`` prefix.
    """
    if not result_sets:
        raise ValueError("merge_results() needs at least one result set")

    roots = [results for results in result_sets if results.is_root]
    if not roots:
        # Every input was a non-root rank's empty set: nothing to merge, and
        # the caller is a parallel script that should carry on quietly.
        return result_sets[0]

    seen: dict[str, int] = {}
    for index, results in enumerate(roots):
        for name in results.region_names:
            if name in seen and seen[name] != index:
                raise ValueError(
                    f"region {name!r} appears in more than one result set; "
                    f"merging them would double-count it. Give the regions "
                    f"distinct names (a 'fortran:' prefix, say) before merging.",
                )
            seen[name] = index

    merged: dict[str, MPIRegion] = {}
    metadata: dict = {}
    likwid: dict[int, dict] = {}
    num_ranks = 0
    for results in roots:
        # Earlier sets win, so the driver's metadata describes the run.
        metadata = {**results.metadata, **metadata}
        for rank, regions in results.get_likwid_regions().items():
            likwid.setdefault(rank, {}).update(regions)
        num_ranks = max(num_ranks, results.num_ranks)
        for region in results.get_regions():
            merged[region.name] = region

    if label is not None:
        metadata["label"] = label

    return ProfilingResults(
        merged,
        metadata=metadata,
        num_ranks=num_ranks,
        likwid=likwid,
        file_path=file_path if file_path is not None else roots[0].file_path,
    )
