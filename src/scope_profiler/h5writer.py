"""Writing a merged HDF5 profiling file.

The exact inverse of :mod:`scope_profiler.h5reader`: this module turns the
in-memory payloads ``finalize()`` collects into the on-disk layout that
:func:`~scope_profiler.h5reader.load_h5` reads back::

    metadata/                       (attrs only)
    rank<N>/regions/<name>/start_times
    rank<N>/regions/<name>/end_times
    rank<N>/likwid/regions/<tag>/   (only with use_likwid=True)

Keeping it here rather than in ``profile_manager`` means the layout has
exactly two files that must change together, and lets the writer be tested
without MPI or a full profiling run.
"""

import h5py
import numpy as np

from scope_profiler.likwid_data import write_likwid_results


def rank_group_name(rank: int) -> str:
    """Name of one rank's group. ``h5reader`` parses the rank back out of it."""
    return f"rank{rank}"


def write_metadata(h5file, metadata: dict) -> None:
    """Create the top-level ``metadata`` group from a run's metadata dict.

    Only rank 0's metadata is stored: it describes the run as a whole. The
    group holds attributes and no datasets.

    Parameters
    ----------
    h5file : h5py.File
        Destination file, opened for writing.
    metadata : dict
        Environment metadata (see :mod:`scope_profiler.metadata`).
    """
    meta_grp = h5file.create_group("metadata")
    for key, value in metadata.items():
        if isinstance(value, (list, tuple)):
            # h5py cannot infer a dtype for an empty list, and would store a
            # non-empty one as fixed-width bytes; be explicit so list-valued
            # metadata (e.g. the loaded modules) always round-trips as strings.
            meta_grp.attrs.create(key, list(value), dtype=h5py.string_dtype())
        else:
            meta_grp.attrs[key] = value


def write_regions(
    group, regions: dict, sources: dict | None = None, tags: dict | None = None
) -> None:
    """Write one rank's recorded timestamps under ``<group>/regions``.

    The datasets are created from exactly-sized arrays and without chunking,
    so a sparsely-called region costs a few hundred bytes rather than a full
    chunk.

    Parameters
    ----------
    group : h5py.Group
        The rank's group.
    regions : dict
        Region name -> ``(start_times, end_times)`` int64 arrays, in
        nanoseconds.
    sources : dict, optional
        Region name -> ``(source_file, source_lineno, source_text)``. A name
        missing here (or the argument itself) simply gets no source attrs,
        which the reader treats as "not captured".
    """
    regions_grp = group.create_group("regions")
    sources = sources or {}
    tags = tags or {}
    for name, (start_times, end_times) in regions.items():
        region_grp = regions_grp.create_group(name)
        region_grp.create_dataset(
            "start_times", data=np.asarray(start_times, dtype=np.int64)
        )
        region_grp.create_dataset(
            "end_times", data=np.asarray(end_times, dtype=np.int64)
        )
        source = sources.get(name)
        if source is not None:
            source_file, source_lineno, source_text = source
            region_grp.attrs["source_file"] = source_file
            region_grp.attrs["source_lineno"] = source_lineno
            region_grp.attrs["source_text"] = source_text
        if name in tags:
            region_grp.attrs.create("tags", list(tags[name]), dtype=h5py.string_dtype())


def write_line_profile(group, records: list | None) -> None:
    """Write copied line-profiler records for one rank."""
    if not records:
        return
    profile_grp = group.create_group("line_profile")
    for index, record in enumerate(records):
        function_grp = profile_grp.create_group(str(index))
        for key in ("region", "filename", "function"):
            function_grp.attrs[key] = record[key]
        function_grp.attrs["first_lineno"] = record["first_lineno"]
        function_grp.attrs["unit"] = record["unit"]
        function_grp.create_dataset("line_numbers", data=record["line_numbers"])
        function_grp.create_dataset("hits", data=record["hits"])
        function_grp.create_dataset("times", data=record["times"])


def write_rank_payload(h5file, rank: int, payload) -> bool:
    """Write one rank's payload into ``rank<N>``.

    A rank that recorded nothing gets no group at all, so the file's rank
    groups are exactly the ranks that have something to report.

    Parameters
    ----------
    h5file : h5py.File
        Destination file, opened for writing.
    rank : int
        The rank this payload came from.
    payload : RankPayload
        The rank's regions, LIKWID results and LIKWID environment.

    Returns
    -------
    bool
        True if a group was created, False if the payload was empty.
    """
    if not payload.regions and not payload.likwid and not payload.line_profile:
        return False

    # create_group, not require_group: each rank is written exactly once, so a
    # duplicate means a bug in the receive loop rather than something to merge.
    group = h5file.create_group(rank_group_name(rank))
    if payload.regions:
        write_regions(group, payload.regions, payload.sources, payload.tags)
    if payload.likwid:
        write_likwid_results(
            group,
            payload.likwid.values(),
            environment=payload.likwid_environment,
        )
    write_line_profile(group, payload.line_profile)
    return True


class ProfilingWriter:
    """The merged output file, open across the rank-by-rank write.

    Used as a context manager by ``finalize()``, which writes rank 0's own
    payload and then each incoming one as it arrives::

        with ProfilingWriter(path, metadata) as writer:
            writer.write_rank(0, own_payload)
            writer.write_rank(source, received_payload)
    """

    def __init__(self, file_path, metadata: dict | None = None) -> None:
        """Open ``file_path`` for writing and store the run's metadata.

        The file is opened with mode ``"w"``, so a previous run's output at
        the same path is replaced rather than merged into -- which is what
        makes a second ``finalize()`` in one process report only its own run.
        """
        self._file = h5py.File(file_path, "w")
        write_metadata(self._file, metadata or {})

    def write_rank(self, rank: int, payload) -> bool:
        """Write one rank's payload; see :func:`write_rank_payload`."""
        return write_rank_payload(self._file, rank, payload)

    def close(self) -> None:
        """Close the output file."""
        self._file.close()

    def __enter__(self) -> "ProfilingWriter":
        """Enter the context, returning the writer itself."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Close the file on the way out, including on error."""
        self.close()
