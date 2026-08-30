"""Writing a merged HDF5 profiling file.

The exact inverse of :mod:`scope_profiler.h5reader`: schema 2 stores region
names once, one compact row per rank/region pair, and all timing events in
shared typed columns. Rank groups remain only for auxiliary LIKWID and
line-profiler records.

Keeping it here rather than in ``profile_manager`` means the layout has
exactly two files that must change together, and lets the writer be tested
without MPI or a full profiling run.
"""

import errno
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np

from scope_profiler.h5schema import CURRENT_SCHEMA_VERSION, SCHEMA_ATTRIBUTE
from scope_profiler.likwid_data import write_likwid_results

_STRING_DTYPE = h5py.string_dtype(encoding="utf-8")
_NO_GPU_DURATION = -1
# Exclusive time is never negative, so this marks a row whose writer did not
# compute one -- a native-trace import, say. The reader falls back to
# reconstructing the nesting for those.
_NO_EXCLUSIVE_TOTAL = -1


def _append(dataset, values) -> int:
    """Append values to a resizable one-dimensional dataset; return offset."""
    values = np.asarray(values, dtype=dataset.dtype)
    offset = len(dataset)
    dataset.resize((offset + len(values),))
    if len(values):
        dataset[offset:] = values
    return offset


class ColumnarIndex:
    """The bookkeeping :func:`append_columnar_rank` needs, held outside the file.

    Appending a rank has to know which region names already have an id and
    which ranks have already been written. Both are recoverable from the file,
    but reading them back costs a scan of columns that grow with every rank,
    which makes writing a job quadratic in its rank count. A writer therefore
    keeps one of these for its lifetime, and the MPI relay
    (``ProfileManager._write_payload_direct``) passes it along with the write
    token instead of re-deriving it per rank.
    """

    __slots__ = ("name_to_id", "names", "ranks")

    def __init__(self, names=(), ranks=()) -> None:
        """Start from a known set of region names (in id order) and ranks."""
        self.names = list(names)
        self.name_to_id = {name: index for index, name in enumerate(self.names)}
        self.ranks = {int(rank) for rank in ranks}

    @classmethod
    def from_file(cls, h5file) -> "ColumnarIndex":
        """Recover the index by reading a partially written file."""
        if "region_table" not in h5file:
            return cls()
        return cls(
            names=[
                value.decode() if isinstance(value, bytes) else str(value)
                for value in h5file["region_table/names"][()]
            ],
            ranks=h5file["rank_region_index/ranks"][()],
        )

    def state(self) -> dict:
        """A picklable snapshot, for handing to the next rank in the relay."""
        return {"names": list(self.names), "ranks": sorted(self.ranks)}

    def register(self, names) -> list:
        """Assign ids to any unseen region names; return the new names in order."""
        new_names = [name for name in names if name not in self.name_to_id]
        for name in new_names:
            self.name_to_id[name] = len(self.names)
            self.names.append(name)
        return new_names


def dataset_storage_options(
    length: int,
    compression: str | None = None,
    compression_level: int | None = None,
    chunk_size: int | None = None,
) -> dict:
    """Build h5py keyword arguments for one one-dimensional event dataset."""
    options = {}
    if chunk_size is not None:
        options["chunks"] = (
            chunk_size if int(length) == 0 else min(int(length), chunk_size),
        )

    if compression == "gzip":
        options["compression"] = "gzip"
        if compression_level is not None:
            options["compression_opts"] = compression_level
    elif compression == "lzf":
        options["compression"] = "lzf"
    elif compression == "zstd":
        try:
            import hdf5plugin
        except ImportError as exc:
            raise ImportError(
                "Zstandard HDF5 compression requires hdf5plugin; install "
                "scope-profiler[compression]."
            ) from exc
        options.update(hdf5plugin.Zstd(clevel=compression_level or 3))
    if compression is not None:
        # Byte shuffling groups equal-significance bytes before compression;
        # monotonic int64 timestamps generally compress much better this way.
        options["shuffle"] = True
    return options


def initialize_columnar_layout(
    h5file,
    *,
    compression=None,
    compression_level=None,
    chunk_size=None,
) -> None:
    """Create the schema-2 region dictionary, pair index, and event columns."""
    regions = h5file.create_group("region_table")
    regions.create_dataset("names", shape=(0,), maxshape=(None,), dtype=_STRING_DTYPE)

    index = h5file.create_group("rank_region_index")
    for name, dtype in (
        ("region_ids", np.uint32),
        ("ranks", np.uint32),
        ("event_offsets", np.uint64),
        ("event_counts", np.uint64),
        ("source_lines", np.int64),
        # Exclusive nanoseconds per (rank, region), computed by the run that
        # recorded it; _NO_EXCLUSIVE_TOTAL where it was not. See
        # call_stack.exclusive_totals_ns.
        ("exclusive_totals", np.int64),
    ):
        index.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype)
    for name in ("source_files", "source_texts", "tags"):
        index.create_dataset(name, shape=(0,), maxshape=(None,), dtype=_STRING_DTYPE)

    events = h5file.create_group("events")
    # call_ids/parent_ids are unique within a rank, not across the file: every
    # rank numbers its own calls from its own id space. This column is the
    # concatenation of all of them, so the same id appears once per rank.
    # Slice by rank (as _read_columnar_regions does) before treating an id as
    # a key; a file-wide id -> call mapping built from this column collides.
    for name in ("start_times", "end_times", "call_ids", "parent_ids"):
        events.create_dataset(
            name,
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            **dataset_storage_options(0, compression, compression_level, chunk_size),
        )


def append_aggregate_rank(h5file, rank, payload, *, index_state=None) -> bool:
    """Append one rank of aggregate-only statistics."""
    stats = payload.aggregate_stats or {}
    if not stats:
        return False
    index_state = (
        ColumnarIndex.from_file(h5file) if index_state is None else index_state
    )
    if rank in index_state.ranks:
        raise ValueError(f"rank {rank} was written more than once")
    names = list(stats)
    new_names = index_state.register(names)
    if new_names:
        _append(h5file["region_table/names"], new_names)
    index = h5file["rank_region_index"]
    for name, dtype in (
        ("aggregate_counts", np.uint64),
        ("aggregate_totals", np.int64),
        ("aggregate_minimums", np.int64),
        ("aggregate_maximums", np.int64),
        ("aggregate_exclusives", np.int64),
    ):
        if name not in index:
            index.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype)
    _append(index["region_ids"], [index_state.name_to_id[name] for name in names])
    _append(index["ranks"], np.full(len(names), rank, dtype=np.uint32))
    _append(index["aggregate_counts"], [stats[name]["count"] for name in names])
    _append(index["aggregate_totals"], [stats[name]["total"] for name in names])
    _append(index["aggregate_minimums"], [stats[name]["minimum"] for name in names])
    _append(index["aggregate_maximums"], [stats[name]["maximum"] for name in names])
    _append(index["aggregate_exclusives"], [stats[name]["exclusive"] for name in names])
    index_state.ranks.add(int(rank))
    return True


def _concatenate(regions: dict, names: list, position: int) -> np.ndarray:
    """One int64 array of every named region's ``position``-th timing array."""
    return np.concatenate(
        [
            np.asarray(
                (
                    regions[name][position]
                    if len(regions[name]) > position
                    else np.full(len(regions[name][0]), -1, dtype=np.int64)
                ),
                dtype=np.int64,
            )
            for name in names
        ]
        or [np.empty(0, dtype=np.int64)]
    )


def append_columnar_rank(
    h5file,
    rank: int,
    payload,
    *,
    index_state: ColumnarIndex | None = None,
    compression=None,
    compression_level=None,
    chunk_size=None,
) -> bool:
    """Append one rank's region arrays to the schema-2 shared columns.

    Everything this rank contributes is appended one column at a time, not one
    region at a time: a resize plus a write per region and per column turned a
    128-rank, 40-region file into ~41,000 separate dataset operations (3.2s,
    against 0.2s batched).

    Parameters
    ----------
    index_state : ColumnarIndex, optional
        Region-name ids and already-written ranks, carried by the caller
        across ranks. Recovered from the file when omitted, which costs a read
        of two columns that grow with every rank written.
    """
    if getattr(payload, "aggregate_stats", None) is not None:
        return append_aggregate_rank(h5file, rank, payload, index_state=index_state)
    if not payload.regions:
        return False
    index_state = (
        ColumnarIndex.from_file(h5file) if index_state is None else index_state
    )
    if rank in index_state.ranks:
        raise ValueError(f"rank {rank} was written more than once")

    names_dataset = h5file["region_table/names"]
    new_names = index_state.register(payload.regions)
    if new_names:
        _append(names_dataset, new_names)
    index_state.ranks.add(int(rank))

    events = h5file["events"]
    total_before = len(events["start_times"])
    needs_gpu = any(
        len(arrays) > 2 and arrays[2] is not None for arrays in payload.regions.values()
    )
    if needs_gpu and "gpu_durations" not in events:
        events.create_dataset(
            "gpu_durations",
            shape=(total_before,),
            maxshape=(None,),
            dtype=np.int64,
            fillvalue=_NO_GPU_DURATION,
            **dataset_storage_options(
                total_before, compression, compression_level, chunk_size
            ),
        )

    sources = payload.sources or {}
    tags = payload.tags or {}
    names = list(payload.regions)
    counts = [len(payload.regions[name][0]) for name in names]
    # Regions are written back to back, so this rank's rows point at
    # consecutive slices of the shared event columns.
    offsets = total_before + np.cumsum([0, *counts[:-1]], dtype=np.uint64)

    _append(events["start_times"], _concatenate(payload.regions, names, 0))
    _append(events["end_times"], _concatenate(payload.regions, names, 1))
    _append(events["call_ids"], _concatenate(payload.regions, names, 3))
    _append(events["parent_ids"], _concatenate(payload.regions, names, 4))
    if "gpu_durations" in events:
        _append(
            events["gpu_durations"],
            np.concatenate(
                [
                    (
                        np.asarray(payload.regions[name][2], dtype=np.int64)
                        if len(payload.regions[name]) > 2
                        and payload.regions[name][2] is not None
                        else np.full(count, _NO_GPU_DURATION, dtype=np.int64)
                    )
                    for name, count in zip(names, counts)
                ]
                or [np.empty(0, dtype=np.int64)]
            ),
        )

    resolved_sources = [sources.get(name, ("", -1, "")) for name in names]
    exclusive_totals = payload.exclusive_totals or {}
    index = h5file["rank_region_index"]
    if "exclusive_totals" not in index:
        # A file this process did not create, from a version that predates the
        # column. Backfill the rows already in it as "not computed".
        index.create_dataset(
            "exclusive_totals",
            shape=(len(index["ranks"]),),
            maxshape=(None,),
            dtype=np.int64,
            fillvalue=_NO_EXCLUSIVE_TOTAL,
        )
    _append(index["region_ids"], [index_state.name_to_id[name] for name in names])
    _append(index["ranks"], np.full(len(names), rank, dtype=np.uint32))
    _append(index["event_offsets"], offsets)
    _append(index["event_counts"], counts)
    _append(
        index["source_lines"],
        [-1 if source[1] is None else source[1] for source in resolved_sources],
    )
    _append(index["source_files"], [source[0] or "" for source in resolved_sources])
    _append(index["source_texts"], [source[2] or "" for source in resolved_sources])
    _append(index["tags"], [json.dumps(list(tags.get(name, ()))) for name in names])
    _append(
        index["exclusive_totals"],
        [exclusive_totals.get(name, _NO_EXCLUSIVE_TOTAL) for name in names],
    )
    return True


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


def compression_filter_available(compression: str | None) -> bool:
    """Whether the active HDF5 library can encode the requested filter."""
    if compression is None:
        return True
    filter_ids = {
        "gzip": h5py.h5z.FILTER_DEFLATE,
        "lzf": h5py.h5z.FILTER_LZF,
        "zstd": 32015,
    }
    if compression == "zstd":
        try:
            import hdf5plugin  # noqa: F401
        except ImportError:
            return False
    return bool(h5py.h5z.filter_avail(filter_ids[compression]))


def payload_layout(payload) -> dict:
    """Return the small, array-free schema needed for collective creation."""
    sources = payload.sources or {}
    tags = payload.tags or {}
    exclusive_totals = payload.exclusive_totals or {}
    return {
        "regions": {
            name: {
                "shapes": [tuple(np.shape(array)) for array in arrays],
                "has_gpu": len(arrays) > 2 and arrays[2] is not None,
                "source": sources.get(name),
                "tags": tuple(tags.get(name, ())),
                "exclusive_total": int(exclusive_totals.get(name, _NO_EXCLUSIVE_TOTAL)),
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


def write_parallel_payload(
    file_path,
    comm,
    rank: int,
    payload,
    metadata: dict,
    *,
    compression: str | None = None,
    compression_level: int | None = None,
    chunk_size: int | None = None,
) -> None:
    """Collectively create schema 2, then write each rank's assigned slices."""
    layouts = comm.allgather(payload_layout(payload))
    root_metadata = comm.bcast(metadata if rank == 0 else None, root=0)

    region_names = []
    for layout in layouts:
        for name in layout["regions"]:
            if name not in region_names:
                region_names.append(name)
    name_to_id = {name: index for index, name in enumerate(region_names)}

    pairs = []
    event_offset = 0
    any_gpu = False
    for owner, layout in enumerate(layouts):
        for name, description in layout["regions"].items():
            count = description["shapes"][0][0]
            pairs.append((owner, name, event_offset, count, description))
            event_offset += count
            any_gpu = any_gpu or description["has_gpu"]

    def fixed_string_data(values):
        encoded = [str(value).encode("utf-8") for value in values]
        width = max(1, max((len(value) for value in encoded), default=1))
        return encoded, h5py.string_dtype("utf-8", width)

    source_files = [
        description["source"][0] if description["source"] else ""
        for *_, description in pairs
    ]
    source_texts = [
        description["source"][2] if description["source"] else ""
        for *_, description in pairs
    ]
    tags = [json.dumps(list(description["tags"])) for *_, description in pairs]
    encoded_names, names_dtype = fixed_string_data(region_names)
    encoded_source_files, source_files_dtype = fixed_string_data(source_files)
    encoded_source_texts, source_texts_dtype = fixed_string_data(source_texts)
    encoded_tags, tags_dtype = fixed_string_data(tags)

    with h5py.File(file_path, "w", driver="mpio", comm=comm) as h5file:
        h5file.attrs[SCHEMA_ATTRIBUTE] = CURRENT_SCHEMA_VERSION
        h5file.attrs["storage_layout"] = "columnar"
        write_metadata(h5file, root_metadata)

        region_table = h5file.create_group("region_table")
        region_table.create_dataset(
            "names", shape=(len(region_names),), dtype=names_dtype
        )
        pair_index = h5file.create_group("rank_region_index")
        for name, dtype in (
            ("region_ids", np.uint32),
            ("ranks", np.uint32),
            ("event_offsets", np.uint64),
            ("event_counts", np.uint64),
            ("source_lines", np.int64),
            ("exclusive_totals", np.int64),
        ):
            pair_index.create_dataset(name, shape=(len(pairs),), dtype=dtype)
        for name, dtype in (
            ("source_files", source_files_dtype),
            ("source_texts", source_texts_dtype),
            ("tags", tags_dtype),
        ):
            pair_index.create_dataset(name, shape=(len(pairs),), dtype=dtype)

        events = h5file.create_group("events")
        for name in ("start_times", "end_times", "call_ids", "parent_ids"):
            events.create_dataset(
                name,
                shape=(event_offset,),
                dtype=np.int64,
                **dataset_storage_options(
                    event_offset, compression, compression_level, chunk_size
                ),
            )
        if any_gpu:
            events.create_dataset(
                "gpu_durations",
                shape=(event_offset,),
                dtype=np.int64,
                fillvalue=_NO_GPU_DURATION,
                **dataset_storage_options(
                    event_offset, compression, compression_level, chunk_size
                ),
            )

        if rank == 0:
            region_table["names"][:] = encoded_names
            pair_index["region_ids"][:] = [
                name_to_id[name] for _, name, _, _, _ in pairs
            ]
            pair_index["ranks"][:] = [owner for owner, _, _, _, _ in pairs]
            pair_index["event_offsets"][:] = [offset for _, _, offset, _, _ in pairs]
            pair_index["event_counts"][:] = [count for _, _, _, count, _ in pairs]
            pair_index["source_lines"][:] = [
                description["source"][1] if description["source"] is not None else -1
                for *_, description in pairs
            ]
            pair_index["source_files"][:] = encoded_source_files
            pair_index["source_texts"][:] = encoded_source_texts
            pair_index["tags"][:] = encoded_tags
            pair_index["exclusive_totals"][:] = [
                description.get("exclusive_total", _NO_EXCLUSIVE_TOTAL)
                for *_, description in pairs
            ]
        comm.Barrier()

        own_pairs = [pair for pair in pairs if pair[0] == rank]
        own_offset = own_pairs[0][2] if own_pairs else 0
        own_count = sum(pair[3] for pair in own_pairs)
        own_slice = slice(own_offset, own_offset + own_count)
        starts = (
            np.concatenate(
                [
                    np.asarray(arrays[0], dtype=np.int64)
                    for arrays in payload.regions.values()
                ]
            )
            if payload.regions
            else np.empty(0, dtype=np.int64)
        )
        ends = (
            np.concatenate(
                [
                    np.asarray(arrays[1], dtype=np.int64)
                    for arrays in payload.regions.values()
                ]
            )
            if payload.regions
            else np.empty(0, dtype=np.int64)
        )
        with events["start_times"].collective:
            events["start_times"][own_slice] = starts
        with events["end_times"].collective:
            events["end_times"][own_slice] = ends
        for field, column in (("call_ids", 3), ("parent_ids", 4)):
            values = (
                np.concatenate(
                    [
                        np.asarray(arrays[column], dtype=np.int64)
                        for arrays in payload.regions.values()
                    ]
                )
                if payload.regions
                else np.empty(0, dtype=np.int64)
            )
            with events[field].collective:
                events[field][own_slice] = values
        if any_gpu:
            gpu_values = (
                np.concatenate(
                    [
                        (
                            np.asarray(arrays[2], dtype=np.int64)
                            if len(arrays) > 2 and arrays[2] is not None
                            else np.full(
                                len(arrays[0]), _NO_GPU_DURATION, dtype=np.int64
                            )
                        )
                        for arrays in payload.regions.values()
                    ]
                )
                if payload.regions
                else np.empty(0, dtype=np.int64)
            )
            with events["gpu_durations"].collective:
                events["gpu_durations"][own_slice] = gpu_values

        for owner, layout in enumerate(layouts):
            line_profile = layout["line_profile"]
            if not line_profile:
                continue
            group = h5file.create_group(rank_group_name(owner))
            profile_group = group.create_group("line_profile")
            for index, description in enumerate(line_profile):
                function_group = profile_group.create_group(str(index))
                for key in ("region", "filename", "function", "first_lineno", "unit"):
                    function_group.attrs[key] = description[key]
                shapes = description["shapes"]
                for key, dtype in (
                    ("line_numbers", np.int64),
                    ("hits", np.int64),
                    ("times", np.float64),
                ):
                    function_group.create_dataset(
                        key,
                        shape=shapes[key],
                        dtype=dtype,
                        **dataset_storage_options(
                            shapes[key][0],
                            compression,
                            compression_level,
                            chunk_size,
                        ),
                    )
        for index, record in enumerate(payload.line_profile or []):
            function_group = h5file[f"{rank_group_name(rank)}/line_profile/{index}"]
            function_group["line_numbers"][:] = np.asarray(
                record["line_numbers"], dtype=np.int64
            )
            function_group["hits"][:] = np.asarray(record["hits"], dtype=np.int64)
            function_group["times"][:] = np.asarray(record["times"], dtype=np.float64)


def write_regions(
    group,
    regions: dict,
    sources: dict | None = None,
    tags: dict | None = None,
    *,
    compression: str | None = None,
    compression_level: int | None = None,
    chunk_size: int | None = None,
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
            "start_times",
            data=np.asarray(start_times, dtype=np.int64),
            **dataset_storage_options(
                len(start_times), compression, compression_level, chunk_size
            ),
        )
        region_grp.create_dataset(
            "end_times",
            data=np.asarray(end_times, dtype=np.int64),
            **dataset_storage_options(
                len(end_times), compression, compression_level, chunk_size
            ),
        )
        if len(arrays) > 2 and arrays[2] is not None:
            region_grp.create_dataset(
                "gpu_durations",
                data=np.asarray(arrays[2], dtype=np.int64),
                **dataset_storage_options(
                    len(arrays[2]), compression, compression_level, chunk_size
                ),
            )
        if len(arrays) > 4:
            # Stored like the timestamps, not raw: these are the most
            # compressible columns in the file (parent_ids is near-constant,
            # call_ids near-monotone) and they double the bytes per event.
            for field, values in (("call_ids", arrays[3]), ("parent_ids", arrays[4])):
                region_grp.create_dataset(
                    field,
                    data=np.asarray(values, dtype=np.int64),
                    **dataset_storage_options(
                        len(values), compression, compression_level, chunk_size
                    ),
                )
        source = sources.get(name)
        if source is not None:
            source_file, source_lineno, source_text = source
            region_grp.attrs["source_file"] = source_file
            region_grp.attrs["source_lineno"] = source_lineno
            region_grp.attrs["source_text"] = source_text
        if name in tags:
            region_grp.attrs.create("tags", list(tags[name]), dtype=h5py.string_dtype())


def write_line_profile(
    group,
    records: list | None,
    *,
    compression: str | None = None,
    compression_level: int | None = None,
    chunk_size: int | None = None,
) -> None:
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
        for key in ("line_numbers", "hits", "times"):
            function_grp.create_dataset(
                key,
                data=record[key],
                **dataset_storage_options(
                    len(record[key]), compression, compression_level, chunk_size
                ),
            )


def write_rank_payload(
    h5file,
    rank: int,
    payload,
    *,
    index_state: ColumnarIndex | None = None,
    compression: str | None = None,
    compression_level: int | None = None,
    chunk_size: int | None = None,
) -> bool:
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
    index_state : ColumnarIndex, optional
        Region-name ids and written ranks carried across calls; see
        :func:`append_columnar_rank`.

    Returns
    -------
    bool
        True if a group was created, False if the payload was empty.
    """
    aggregate = getattr(payload, "aggregate_stats", None) is not None
    if "region_table" not in h5file:
        h5file.attrs[SCHEMA_ATTRIBUTE] = CURRENT_SCHEMA_VERSION
        h5file.attrs["storage_layout"] = "columnar"
        initialize_columnar_layout(
            h5file,
            compression=compression,
            compression_level=compression_level,
            chunk_size=chunk_size,
        )
    if aggregate:
        h5file.attrs["storage_layout"] = "aggregate"
    if (
        not payload.regions
        and not payload.likwid
        and not payload.line_profile
        and not getattr(payload, "aggregate_stats", None)
    ):
        return False
    wrote_regions = append_columnar_rank(
        h5file,
        rank,
        payload,
        index_state=index_state,
        compression=compression,
        compression_level=compression_level,
        chunk_size=chunk_size,
    )
    if not payload.likwid and not payload.line_profile:
        return wrote_regions

    # Auxiliary records retain rank-local groups because their matrices and
    # function records have different schemas from timing events.
    group = h5file.create_group(rank_group_name(rank))
    if payload.likwid:
        write_likwid_results(
            group,
            payload.likwid.values(),
            environment=payload.likwid_environment,
        )
    write_line_profile(
        group,
        payload.line_profile,
        compression=compression,
        compression_level=compression_level,
        chunk_size=chunk_size,
    )
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
        index_state: ColumnarIndex | None = None,
        compression: str | None = None,
        compression_level: int | None = None,
        chunk_size: int | None = None,
    ) -> None:
        """Open ``file_path`` for writing and store the run's metadata.

        New files are written to a unique sibling temporary path and atomically
        replace ``file_path`` only after a successful close. A failed write
        therefore preserves any previous profile at the destination.

        ``index_state`` carries the region-name ids and written ranks of a file
        this process did not create itself (see :class:`ColumnarIndex`); when
        omitted for an existing file they are read back from it.
        """
        self._final_path = Path(file_path)
        self._atomic = mode == "w" if atomic is None else atomic
        self._temporary_path: Path | None = None
        self._closed = False
        self._compression = compression
        self._compression_level = compression_level
        self._chunk_size = chunk_size
        self._index_state = index_state
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
                self._file.attrs["storage_layout"] = "columnar"
                write_metadata(self._file, metadata or {})
                initialize_columnar_layout(
                    self._file,
                    compression=compression,
                    compression_level=compression_level,
                    chunk_size=chunk_size,
                )
            else:
                from scope_profiler.h5schema import read_schema_version

                read_schema_version(self._file)
            if self._index_state is None:
                # One read of the two index columns, here, rather than one per
                # write_rank() call: see ColumnarIndex.
                self._index_state = (
                    ColumnarIndex()
                    if mode == "w"
                    else ColumnarIndex.from_file(self._file)
                )
        except Exception:
            file_handle = getattr(self, "_file", None)
            if file_handle is not None:
                file_handle.close()
            if self._temporary_path is not None:
                self._temporary_path.unlink(missing_ok=True)
            raise

    @classmethod
    def open_existing(
        cls,
        file_path,
        *,
        index_state: ColumnarIndex | None = None,
        compression: str | None = None,
        compression_level: int | None = None,
        chunk_size: int | None = None,
    ) -> "ProfilingWriter":
        """Open a completed prefix of an MPI output file for one more rank."""
        return cls(
            file_path,
            mode="r+",
            index_state=index_state,
            compression=compression,
            compression_level=compression_level,
            chunk_size=chunk_size,
        )

    @property
    def index_state(self) -> ColumnarIndex:
        """This file's region-name ids and written ranks, for the next writer."""
        return self._index_state

    def write_rank(self, rank: int, payload) -> bool:
        """Write one rank's payload; see :func:`write_rank_payload`."""
        return write_rank_payload(
            self._file,
            rank,
            payload,
            index_state=self._index_state,
            compression=self._compression,
            compression_level=self._compression_level,
            chunk_size=self._chunk_size,
        )

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
