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

import errno
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np

from scope_profiler.h5schema import CURRENT_SCHEMA_VERSION, SCHEMA_ATTRIBUTE
from scope_profiler.likwid_data import write_likwid_results


def _fsync_file(path) -> None:
    """Force a closed file's contents and metadata to stable storage."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path) -> None:
    """Persist a directory entry update where the platform supports it."""
    if not hasattr(os, "O_DIRECTORY"):
        return
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    except OSError as exc:
        if exc.errno in {errno.EINVAL, errno.ENOTSUP, errno.EACCES}:
            return
        raise
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            # Some network and non-POSIX filesystems support atomic rename but
            # not directory fsync. The file itself was already fsynced above.
            if exc.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EBADF}:
                raise
    finally:
        os.close(descriptor)


def atomic_publish(temporary_path, final_path) -> None:
    """Durably replace ``final_path`` with a completed sibling file."""
    temporary_path = os.fspath(temporary_path)
    final_path = os.fspath(final_path)
    _fsync_file(temporary_path)
    os.replace(temporary_path, final_path)
    _fsync_directory(os.path.dirname(os.path.abspath(final_path)))


def parallel_hdf5_available() -> bool:
    """Whether this h5py build has the MPI-IO driver enabled."""
    return bool(getattr(h5py.get_config(), "mpi", False))


def payload_layout(payload) -> dict:
    """Return the small, array-free schema needed for collective creation."""
    sources = payload.sources or {}
    tags = payload.tags or {}
    return {
        "regions": {
            name: {
                "shapes": [tuple(np.shape(array)) for array in arrays],
                "source": sources.get(name),
                "tags": tuple(tags.get(name, ())),
            }
            for name, arrays in payload.regions.items()
        },
        "line_profile": [
            {
                "region": record["region"],
                "filename": record["filename"],
                "function": record["function"],
                "first_lineno": int(record["first_lineno"]),
                "unit": float(record["unit"]),
                "shapes": {
                    key: tuple(np.shape(record[key]))
                    for key in ("line_numbers", "hits", "times")
                },
            }
            for record in (payload.line_profile or [])
        ],
    }


def write_parallel_payload(file_path, comm, rank: int, payload, metadata: dict) -> None:
    """Collectively create one file, then let each rank write its own arrays.

    Parallel HDF5 requires metadata operations to be collective.  Only the
    array-free layouts are gathered; timestamp, GPU, and line-timing arrays
    remain local and are written independently into datasets owned by their
    rank.
    """
    layouts = comm.allgather(payload_layout(payload))
    root_metadata = comm.bcast(metadata if rank == 0 else None, root=0)

    with h5py.File(file_path, "w", driver="mpio", comm=comm) as h5file:
        h5file.attrs[SCHEMA_ATTRIBUTE] = CURRENT_SCHEMA_VERSION
        write_metadata(h5file, root_metadata)

        for owner, layout in enumerate(layouts):
            regions = layout["regions"]
            line_profile = layout["line_profile"]
            if not regions and not line_profile:
                continue
            group = h5file.create_group(rank_group_name(owner))
            if regions:
                regions_group = group.create_group("regions")
                for name, description in regions.items():
                    region_group = regions_group.create_group(name)
                    shapes = description["shapes"]
                    region_group.create_dataset(
                        "start_times", shape=shapes[0], dtype=np.int64
                    )
                    region_group.create_dataset(
                        "end_times", shape=shapes[1], dtype=np.int64
                    )
                    if len(shapes) > 2:
                        region_group.create_dataset(
                            "gpu_durations", shape=shapes[2], dtype=np.int64
                        )
                    source = description["source"]
                    if source is not None:
                        source_file, source_lineno, source_text = source
                        region_group.attrs["source_file"] = source_file
                        region_group.attrs["source_lineno"] = source_lineno
                        region_group.attrs["source_text"] = source_text
                    region_group.attrs.create(
                        "tags", list(description["tags"]), dtype=h5py.string_dtype()
                    )

            if line_profile:
                profile_group = group.create_group("line_profile")
                for index, description in enumerate(line_profile):
                    function_group = profile_group.create_group(str(index))
                    for key in (
                        "region",
                        "filename",
                        "function",
                        "first_lineno",
                        "unit",
                    ):
                        function_group.attrs[key] = description[key]
                    shapes = description["shapes"]
                    function_group.create_dataset(
                        "line_numbers", shape=shapes["line_numbers"], dtype=np.int64
                    )
                    function_group.create_dataset(
                        "hits", shape=shapes["hits"], dtype=np.int64
                    )
                    function_group.create_dataset(
                        "times", shape=shapes["times"], dtype=np.float64
                    )

        for name, arrays in payload.regions.items():
            region_group = h5file[f"{rank_group_name(rank)}/regions/{name}"]
            region_group["start_times"][:] = np.asarray(arrays[0], dtype=np.int64)
            region_group["end_times"][:] = np.asarray(arrays[1], dtype=np.int64)
            if len(arrays) > 2:
                region_group["gpu_durations"][:] = np.asarray(arrays[2], dtype=np.int64)
        for index, record in enumerate(payload.line_profile or []):
            function_group = h5file[f"{rank_group_name(rank)}/line_profile/{index}"]
            function_group["line_numbers"][:] = np.asarray(
                record["line_numbers"], dtype=np.int64
            )
            function_group["hits"][:] = np.asarray(record["hits"], dtype=np.int64)
            function_group["times"][:] = np.asarray(record["times"], dtype=np.float64)


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
        Region name -> ``(start_times, end_times)`` or
        ``(start_times, end_times, gpu_durations)`` int64 arrays, in
        nanoseconds.
    sources : dict, optional
        Region name -> ``(source_file, source_lineno, source_text)``. A name
        missing here (or the argument itself) simply gets no source attrs,
        which the reader treats as "not captured".
    """
    regions_grp = group.create_group("regions")
    sources = sources or {}
    tags = tags or {}
    for name, arrays in regions.items():
        start_times, end_times = arrays[:2]
        region_grp = regions_grp.create_group(name)
        region_grp.create_dataset(
            "start_times", data=np.asarray(start_times, dtype=np.int64)
        )
        region_grp.create_dataset(
            "end_times", data=np.asarray(end_times, dtype=np.int64)
        )
        if len(arrays) > 2 and arrays[2] is not None:
            region_grp.create_dataset(
                "gpu_durations", data=np.asarray(arrays[2], dtype=np.int64)
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

    def __init__(
        self,
        file_path,
        metadata: dict | None = None,
        mode: str = "w",
        *,
        atomic: bool | None = None,
    ) -> None:
        """Open ``file_path`` for writing and store the run's metadata.

        New files are written to a unique sibling temporary path and atomically
        replace ``file_path`` only after a successful close. A failed write
        therefore preserves any previous profile at the destination.
        """
        self._final_path = Path(file_path)
        self._atomic = mode == "w" if atomic is None else atomic
        self._temporary_path: Path | None = None
        self._closed = False
        if self._atomic and mode != "w":
            raise ValueError("atomic output is only supported with mode='w'")

        open_path = self._final_path
        if self._atomic:
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{self._final_path.name}.",
                suffix=".tmp",
                dir=self._final_path.parent,
            )
            os.close(descriptor)
            self._temporary_path = Path(temporary)
            open_path = self._temporary_path

        try:
            self._file = h5py.File(open_path, mode)
            if mode == "w":
                self._file.attrs[SCHEMA_ATTRIBUTE] = CURRENT_SCHEMA_VERSION
                write_metadata(self._file, metadata or {})
            else:
                from scope_profiler.h5schema import read_schema_version

                read_schema_version(self._file)
        except Exception:
            file_handle = getattr(self, "_file", None)
            if file_handle is not None:
                file_handle.close()
            if self._temporary_path is not None:
                self._temporary_path.unlink(missing_ok=True)
            raise

    @classmethod
    def open_existing(cls, file_path) -> "ProfilingWriter":
        """Open a completed prefix of an MPI output file for one more rank."""
        return cls(file_path, mode="r+")

    def write_rank(self, rank: int, payload) -> bool:
        """Write one rank's payload; see :func:`write_rank_payload`."""
        return write_rank_payload(self._file, rank, payload)

    def close(self, *, commit: bool = True) -> None:
        """Close the file and publish it, or discard an unsuccessful write."""
        if self._closed:
            return
        self._closed = True
        try:
            try:
                self._file.flush()
            finally:
                self._file.close()
            if self._temporary_path is not None:
                if commit:
                    atomic_publish(self._temporary_path, self._final_path)
                else:
                    self._temporary_path.unlink(missing_ok=True)
        except Exception:
            if self._temporary_path is not None:
                self._temporary_path.unlink(missing_ok=True)
            raise

    def __enter__(self) -> "ProfilingWriter":
        """Enter the context, returning the writer itself."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Publish successful writes and discard failed ones."""
        if exc_type is None:
            self.close()
            return
        try:
            self.close(commit=False)
        except Exception as close_error:
            if exc_value is not None and hasattr(exc_value, "add_note"):
                exc_value.add_note(
                    f"also failed to discard profiling output: {close_error}"
                )
