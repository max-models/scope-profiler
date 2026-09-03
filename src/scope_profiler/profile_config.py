"""Configuration for profiling managers."""

import os
import shutil
from dataclasses import dataclass, fields
from pathlib import Path
from time import perf_counter_ns
from typing import TYPE_CHECKING, Any

try:  # Python 3.11+
    import tomllib as _tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as _tomllib  # type: ignore[no-redef]

tomllib = _tomllib

from scope_profiler.concurrency import ConcurrencyTracker
from scope_profiler.metadata import collect_metadata
from scope_profiler.mpi_launch import get_comm

if TYPE_CHECKING:
    from mpi4py.MPI import Intercomm


_CONFIG_FIELDS = {
    "deactivate_profiling",
    "deactivate_file_output",
    "use_likwid",
    "perf_events",
    "use_line_profiler",
    "use_memray",
    "memory_profile_path",
    "memray_native_traces",
    "memray_trace_python_allocators",
    "memray_follow_fork",
    "use_nvtx",
    "use_gpu_timing",
    "gpu_timing_backend",
    "recursive_profile",
    "buffer_limit",
    "file_path",
    "output_mode",
    "hdf5_compression",
    "hdf5_compression_level",
    "hdf5_chunk_size",
    "label",
    "capture_region_source",
    "aggregation_mode",
    "track_threads",
    "track_async",
}


def load_profiling_config(path: str | os.PathLike[str]) -> dict:
    """Load the ``[profiling]`` table from a TOML settings file.

    A top-level table is also accepted for small files.  Paths in the file
    retain TOML's normal meaning and are interpreted relative to the current
    working directory, just like paths passed to :meth:`ProfileManager.setup`.
    """
    config_path = Path(path)
    try:
        with config_path.open("rb") as stream:
            raw = tomllib.load(stream)
    except OSError as exc:
        raise ValueError(f"Could not read profiling config {path!r}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid profiling TOML {path!r}: {exc}") from exc

    settings = raw.get("profiling", raw)
    if not isinstance(settings, dict):
        raise ValueError("Profiling config [profiling] must be a TOML table")
    unknown = sorted(set(settings) - _CONFIG_FIELDS)
    if unknown:
        names = ", ".join(unknown)
        raise ValueError(f"Unknown profiling setting(s): {names}")
    return dict(settings)


@dataclass
class ProfilingOptions:
    """A bag of :meth:`ProfileManager.setup` settings, to reuse or pass around.

    Every field mirrors a ``setup()``/``session()`` keyword argument and
    defaults to ``None``, meaning "unset -- let ``setup()``'s own default, or
    its ``config_path`` file, decide". Pass one to ``setup()`` or
    ``session()`` instead of repeating the same handful of keyword arguments
    at every call site::

        options = ProfilingOptions(use_likwid=True, file_path="run.h5")
        ProfileManager.setup(options=options)
        # ... or, equivalently:
        with ProfileManager.session(options=options):
            ...

    An explicit keyword argument passed alongside ``options`` still wins over
    the same field on ``options``, so a shared ``ProfilingOptions`` can be
    reused across runs with one-off overrides::

        ProfileManager.setup(options=options, file_path="run_b.h5")

    Attributes
    ----------
    file_path : str or None
        Path to the output profiling data file (default:
        ``"profiling_data.h5"``).
    label : str or None
        Short name for this run (default: None, i.e. the output file's
        stem). Post-processing uses it wherever a run has to be named --
        chart legends, the summary heading, ``scope-profiler inspect``, the
        JSON statistics -- and it is stored in the output file as the
        ``label`` metadata field, so it survives into every later
        post-processing step.
    use_likwid : bool or None
        Enable LIKWID hardware counter collection (default: False).
    perf_events : sequence of str or None
        Linux ``perf_event_open`` events to collect per region (default: None).
    use_line_profiler : bool or None
        Enable line-by-line profiling via line_profiler (default: False).
    use_memray : bool or None
        Record process-wide memory allocations with Memray (default: False).
    deactivate_profiling : bool or None
        Turn profiling off entirely (``setup()`` default: False). Every
        region becomes a no-op, so the instrumentation can stay in the code
        at near-zero cost instead of being removed.
    use_nvtx : bool or None
        Add NVTX ranges to profiled regions for NVIDIA Nsight tools
        (default: False). Requires ``scope-profiler[nvtx]``.
    use_gpu_timing : bool or None
        Record CUDA-event elapsed device time for each profiled region
        (default: False). CPU timestamps are still recorded, so the normal
        timeline remains enqueue-side timing.
    gpu_timing_backend : str, object, or None
        CUDA-event backend for ``use_gpu_timing``: ``"auto"``, ``"torch"``,
        ``"cupy"``, or a custom object implementing ``record_event()`` and
        ``elapsed_time_ns(start_event, end_event)`` (default: ``"auto"``).
    deactivate_file_output : bool or None
        Write no HDF5 file at all (default: False), not even the run
        metadata. Pair it with ``finalize(return_results=True)`` to analyse a
        run entirely in memory.
    recursive_profile : bool or None
        Enable recursive profiling for all decorated functions by default
        (default: False). Overridable per decorator with
        ``@ProfileManager.profile(..., recursive=...)``.
    aggregation_mode : bool or None
        Record only count, inclusive total, minimum, maximum, and exclusive
        total per region (default: False). Timeline events are unavailable
        in this mode; it cannot be combined with line, GPU, NVTX, or LIKWID
        profiling.
    track_threads : bool or None
        Record which thread each call ran on, and describe every thread the
        run touched (default: False). Required for correct results from
        concurrently used regions; see
        :class:`~scope_profiler.region_profiler.ThreadedProfileRegion`.
    track_async : bool or None
        Also record which asyncio task or greenlet each call ran in, and how
        much of the call was spent awaiting (default: False). Implies
        ``track_threads``.
    capture_region_source : bool or None
        Record where each region is defined -- the ``with`` block or the
        decorated function -- once per distinct source file, the first time
        any of its regions is created (default: False). See
        :attr:`~scope_profiler.region.Region.source_text`. Off by default:
        the cost tracks that file's total size (one ``ast.parse`` + tree
        walk), so it stays under a millisecond for a typical file but can
        reach tenths of a second per rank for a single file with thousands
        of lines, paid independently by every rank.
    buffer_limit : int or None
        Initial number of profiling events preallocated per region (default:
        1024). Buffers grow on demand, so this is a starting size rather
        than a limit; raise it for very hot regions to avoid repeated
        reallocation.
    output_mode : str or None
        MPI file writer: ``"auto"``, ``"direct"``, or ``"parallel"``
        (default: ``"auto"``). ``auto`` prefers MPI-enabled h5py when
        compatible with the active instrumentation and otherwise lets ranks
        append directly to one serial-HDF5 file in token order.
    hdf5_compression : str or None
        Compression filter for timestamp and GPU-duration datasets:
        ``None``, ``"gzip"``, ``"lzf"``, or ``"zstd"`` (default: None).
    hdf5_compression_level : int or None
        GZIP level 0--9 or Zstandard level 1--22 (default: None).
    hdf5_chunk_size : int or None
        Maximum events per dataset chunk (default: None). Enables chunked
        partial reads even without compression.
    """

    file_path: str | None = None
    label: str | None = None
    use_likwid: bool | None = None
    perf_events: list[str] | tuple[str, ...] | str | None = None
    use_line_profiler: bool | None = None
    use_memray: bool | None = None
    memory_profile_path: str | None = None
    memray_native_traces: bool | None = None
    memray_trace_python_allocators: bool | None = None
    memray_follow_fork: bool | None = None
    deactivate_profiling: bool | None = None
    use_nvtx: bool | None = None
    use_gpu_timing: bool | None = None
    gpu_timing_backend: Any = None
    deactivate_file_output: bool | None = None
    recursive_profile: bool | None = None
    aggregation_mode: bool | None = None
    track_threads: bool | None = None
    track_async: bool | None = None
    capture_region_source: bool | None = None
    buffer_limit: int | None = None
    output_mode: str | None = None
    hdf5_compression: str | None = None
    hdf5_compression_level: int | None = None
    hdf5_chunk_size: int | None = None

    def to_kwargs(self) -> dict:
        """This options' explicitly-set fields, as ``setup()`` keyword arguments."""
        return {
            field.name: value
            for field in fields(self)
            if (value := getattr(self, field.name)) is not None
        }


# try:
# import pylikwid
#     _PYLIKWID_AVAILABLE = True
# except ImportError:
#     pylikwid = None
#     _PYLIKWID_AVAILABLE = False


def _liblikwid_search_dirs() -> list:
    """Directories that plausibly hold LIKWID's shared library.

    Derived from the variables a ``likwid`` environment module exports, then
    from the location of ``likwid-perfctr`` itself, which is on ``PATH``
    whenever the module is loaded.
    """
    dirs = []

    for var in ("LIKWID_LIB", "LIKWID_HOME", "LIKWID_ROOT", "LIKWID_PREFIX"):
        root = os.environ.get(var)
        if not root:
            continue
        # LIKWID_LIB may already point at the library directory itself.
        dirs.extend([root, os.path.join(root, "lib"), os.path.join(root, "lib64")])

    perfctr = shutil.which("likwid-perfctr")
    if perfctr:
        prefix = os.path.dirname(os.path.dirname(os.path.realpath(perfctr)))
        dirs.extend([os.path.join(prefix, "lib"), os.path.join(prefix, "lib64")])

    seen = set()
    unique_dirs = []
    for directory in dirs:
        if os.path.isdir(directory) and directory not in seen:
            seen.add(directory)
            unique_dirs.append(directory)
    return unique_dirs


def _preload_liblikwid() -> bool:
    """Load LIKWID's shared library into this process, if it can be found.

    ``pylikwid`` is linked against ``liblikwid.so``, so importing it fails
    unless the dynamic loader can find that library. Clusters routinely ship a
    ``likwid`` module that puts ``likwid-perfctr`` on ``PATH`` and exports
    ``LIKWID_HOME`` without ever touching ``LD_LIBRARY_PATH``, which makes the
    import fail even though everything needed is installed.

    Setting ``LD_LIBRARY_PATH`` from inside Python cannot help --- the loader
    reads it once, at process start. Opening the library explicitly with
    ``RTLD_GLOBAL`` does: its symbols then satisfy the pylikwid extension when
    it is imported moments later.

    Returns
    -------
    bool
        True if a library was loaded, False if none could be found.
    """
    import ctypes
    import glob

    for directory in _liblikwid_search_dirs():
        # Prefer the fully versioned name (liblikwid.so.5.3) over the bare
        # symlink, so a directory holding several versions resolves the same
        # way the loader would.
        candidates = sorted(glob.glob(os.path.join(directory, "liblikwid.so*")))
        for path in reversed(candidates):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                return True
            except OSError:
                continue
    return False


def _import_pylikwid():
    """Dynamically import the pylikwid module.

    If the import fails only because LIKWID's shared library is not on the
    loader path, the library is located and preloaded and the import retried;
    see :func:`_preload_liblikwid`.

    Returns
    -------
    module
        The imported `pylikwid` module.

    Raises
    ------
    ImportError
        If the module cannot be imported.
    """
    try:
        import pylikwid
    except ImportError as exc:
        # A missing liblikwid is recoverable; a missing pylikwid is not.
        if "liblikwid" not in str(exc) or not _preload_liblikwid():
            raise
        import pylikwid

    return pylikwid


def _pylikwid_import_error(exc: ImportError) -> str:
    """Explain why ``import pylikwid`` failed, and how to fix it.

    The two failures look alike from the outside but need opposite fixes:
    the bindings can be genuinely absent, or they can be installed and merely
    unable to find LIKWID's shared library at load time (typical on clusters,
    where a ``likwid`` module puts ``likwid-perfctr`` on ``PATH`` without
    touching ``LD_LIBRARY_PATH``). Reporting the second as "not installed"
    sends people off to reinstall a package they already have.
    """
    message = str(exc)

    if "liblikwid" in message:
        likwid_home = os.environ.get("LIKWID_HOME") or os.environ.get("LIKWID_ROOT")
        hint = (
            f'export LD_LIBRARY_PATH="{likwid_home}/lib:$LD_LIBRARY_PATH"'
            if likwid_home
            else 'export LD_LIBRARY_PATH="<likwid-prefix>/lib:$LD_LIBRARY_PATH"'
        )
        return (
            "LIKWID profiling requested and pylikwid is installed, but the "
            f"LIKWID runtime library could not be loaded ({message}). Point "
            "the dynamic loader at your LIKWID installation, e.g.\n"
            f"    {hint}\n"
            "On a cluster, load the LIKWID module first (module load likwid)."
        )

    return (
        "LIKWID profiling requested but the pylikwid module could not be "
        f"imported ({message}). Install scope-profiler[likwid], or pylikwid "
        "directly. It builds against an existing LIKWID installation, so "
        "install or load LIKWID first."
    )


class ProfilingConfig:
    """Configuration for one profiling manager.

    This class centralizes configuration for LIKWID performance counters,
    buffer limits, and file paths. Each manager owns a separate instance, so
    independently configured profiling sessions can coexist. Constructing it
    is purely local: it reads the communicator for rank and size but issues no
    MPI call of its own, so ``setup()`` does not have to be collective.
    """

    def _memray_path(self, configured_path: str | None) -> Path:
        """Return this rank's unique Memray capture path."""
        path = Path(configured_path) if configured_path else Path(self._file_path)
        if configured_path is None:
            path = path.with_suffix(".memray.bin")
        if self._size > 1:
            path = path.with_name(f"{path.stem}.rank{self._rank}{path.suffix}")
        return path

    def __init__(
        self,
        file_path: str = "profiling_data.h5",
        label: str | None = None,
        use_likwid: bool = False,
        perf_events: list[str] | tuple[str, ...] | str | None = None,
        use_line_profiler: bool = False,
        use_memray: bool = False,
        memory_profile_path: str | None = None,
        memray_native_traces: bool = False,
        memray_trace_python_allocators: bool = False,
        memray_follow_fork: bool = False,
        deactivate_profiling: bool = False,
        use_nvtx: bool = False,
        use_gpu_timing: bool = False,
        gpu_timing_backend="auto",
        deactivate_file_output: bool = False,
        recursive_profile: bool = False,
        aggregation_mode: bool = False,
        track_threads: bool = False,
        track_async: bool = False,
        capture_region_source: bool = False,
        buffer_limit: int = 1024,
        output_mode: str = "auto",
        hdf5_compression: str | None = None,
        hdf5_compression_level: int | None = None,
        hdf5_chunk_size: int | None = None,
    ):
        """Initialize the profiling configuration.

        Parameters
        ----------
        file_path : str
            Global output file path for combined profiling data.
        label : str or None
            Short name for this run, used by post-processing wherever a run
            has to be named: chart legends, the summary heading, the JSON
            statistics. Defaults to None, in which case the output file's stem
            is used. Persisted as the ``label`` metadata field.
        use_likwid : bool
            Enable LIKWID marker API if available.
        use_line_profiler : bool
            Enable line-by-line profiling via line_profiler.
        use_memray : bool
            Record process-wide allocations with Memray in a separate native
            ``.bin`` capture.
        deactivate_profiling : bool
            Turn profiling off entirely. Every region becomes a no-op, so
            instrumentation can stay in the code at near-zero cost.
        use_nvtx : bool
            Add NVTX ranges to profiled regions for NVIDIA Nsight tools.
        use_gpu_timing : bool
            Record CUDA-event elapsed device time for each profiled region.
        gpu_timing_backend : str or object
            CUDA-event backend: ``"auto"``, ``"torch"``, ``"cupy"``, or an
            object implementing ``record_event()`` and ``elapsed_time_ns()``.
        deactivate_file_output : bool
            Write no HDF5 file at all, not even the run metadata. The recorded
            data then lives only in memory, where
            ``finalize(return_results=True)`` can still return it.
        recursive_profile : bool
            Enable recursive profiling by default for decorated functions.
        aggregation_mode : bool
            Record only count, inclusive total, minimum, maximum, and
            exclusive total per region. Timeline events are unavailable in
            this mode; it cannot be combined with line, GPU, NVTX, or LIKWID
            profiling.
        track_threads : bool
            Give every thread its own buffers and stamp each call with the
            thread it ran on, so regions entered concurrently record correct,
            separable timelines. Cannot be combined with line, GPU, NVTX,
            LIKWID or aggregation profiling.
        track_async : bool
            Additionally follow asyncio tasks and greenlets: each call carries
            the lane it ran in and the time that lane spent suspended inside
            the call, and the run reports per-task running and awaiting
            totals. Implies ``track_threads``.
        capture_region_source : bool
            Record where each region is defined (see
            :attr:`~scope_profiler.region.Region.source_text`), once per
            distinct source file, the first time any of its regions is
            created. Off by default: measured cost is driven almost entirely
            by that file's total size (an ``ast.parse`` + one tree walk), not
            by the number or size of the regions in it, and every rank pays
            it independently and concurrently -- under a millisecond for a
            typical file of a few hundred lines regardless of MPI rank count,
            but tenths of a second *per rank* for a single ~10,000-line file
            with many regions, compounding into whole seconds under
            contention on a job with more ranks than idle cores (~0.3s at 8
            ranks, ~2.9s at 64, measured on a shared, oversubscribed login
            node with such a file). Set to True to enable it; for a typical,
            modestly sized codebase the cost is negligible.
        buffer_limit : int
            Initial number of in-memory records to preallocate per region.
            The buffers grow on demand, so this is a starting size, not a cap.
        output_mode : str
            MPI HDF5 writer: ``"auto"`` uses parallel HDF5 when available and
            safe, otherwise serializes direct per-rank writes; ``"direct"``
            always uses the latter; ``"parallel"`` requires an MPI-enabled
            h5py build. Serial runs are unaffected.
        hdf5_compression : str or None
            Dataset compression: ``None``, ``"gzip"``, ``"lzf"``, or
            ``"zstd"``. Zstandard requires the ``compression`` extra.
        hdf5_compression_level : int or None
            GZIP level 0--9 or Zstandard level 1--22. Ignored when compression
            is disabled; LZF has no configurable level.
        hdf5_chunk_size : int or None
            Maximum number of events per HDF5 chunk. ``None`` leaves datasets
            contiguous unless compression requires h5py to choose chunks.

        Notes
        -----
        MPI is not configurable: collectives are used exactly when the process
        was started by an MPI launcher, so a plain ``python script.py`` never
        touches MPI. See :mod:`scope_profiler.mpi_launch` for the detection and
        its ``SCOPE_PROFILER_MPI`` override.
        """

        # The run's origin, on the perf_counter_ns clock. Persisted as
        # metadata, and the point post-processing measures its relative
        # timeline from.
        self._start_time_ns = perf_counter_ns()

        # Serial runs must not import mpi4py (which would call MPI_Init) nor
        # issue any collective, so the communicator stays None unless this
        # process really is part of an MPI job.
        self._comm = get_comm()
        self._deactivate_profiling = deactivate_profiling
        self._paused = False
        self._deactivate_file_output = deactivate_file_output
        self._use_likwid = use_likwid
        from scope_profiler.perf_events import validate_events

        self._perf_events = (
            validate_events(perf_events) if perf_events is not None else ()
        )
        self._use_line_profiler = use_line_profiler
        self._use_memray = use_memray
        self._memray_tracker = None
        self._use_nvtx = use_nvtx
        self._use_gpu_timing = use_gpu_timing
        self._gpu_timing_backend = gpu_timing_backend
        self._recursive_profile = recursive_profile
        self._buffer_limit = buffer_limit
        self._file_path = file_path
        if output_mode not in {"auto", "direct", "parallel"}:
            raise ValueError(
                "output_mode must be 'auto', 'direct', or 'parallel', "
                f"got {output_mode!r}"
            )
        self._output_mode = output_mode
        if isinstance(hdf5_compression, str):
            hdf5_compression = hdf5_compression.strip().lower()
            if hdf5_compression in {"", "none"}:
                hdf5_compression = None
        if hdf5_compression not in {None, "gzip", "lzf", "zstd"}:
            raise ValueError(
                "hdf5_compression must be None, 'gzip', 'lzf', or 'zstd', "
                f"got {hdf5_compression!r}"
            )
        if hdf5_chunk_size is not None and (
            isinstance(hdf5_chunk_size, bool)
            or not isinstance(hdf5_chunk_size, int)
            or hdf5_chunk_size <= 0
        ):
            raise ValueError("hdf5_chunk_size must be a positive integer or None")
        if hdf5_compression_level is not None and (
            isinstance(hdf5_compression_level, bool)
            or not isinstance(hdf5_compression_level, int)
        ):
            raise ValueError("hdf5_compression_level must be an integer or None")
        if (
            hdf5_compression == "gzip"
            and hdf5_compression_level is not None
            and not 0 <= hdf5_compression_level <= 9
        ):
            raise ValueError("GZIP compression level must be between 0 and 9")
        if (
            hdf5_compression == "zstd"
            and hdf5_compression_level is not None
            and not 1 <= hdf5_compression_level <= 22
        ):
            raise ValueError("Zstandard compression level must be between 1 and 22")
        if hdf5_compression == "lzf" and hdf5_compression_level is not None:
            raise ValueError("LZF compression does not accept a compression level")
        self._hdf5_compression = hdf5_compression
        self._hdf5_compression_level = hdf5_compression_level
        self._hdf5_chunk_size = hdf5_chunk_size
        self._capture_region_source = capture_region_source
        if aggregation_mode and (
            use_line_profiler
            or use_gpu_timing
            or use_likwid
            or use_nvtx
            or self._perf_events
        ):
            raise ValueError(
                "aggregation_mode cannot be combined with line, GPU, NVTX, LIKWID, or perf events"
            )
        self._aggregation_mode = aggregation_mode

        # track_async is a strict refinement of track_threads: a task lane is
        # identified relative to the thread it runs on, and its buffers are
        # the thread's.
        track_threads = bool(track_threads or track_async)
        if track_threads and (
            use_line_profiler
            or use_gpu_timing
            or use_likwid
            or self._perf_events
            or use_nvtx
            or aggregation_mode
        ):
            raise ValueError(
                "track_threads/track_async cannot be combined with line, GPU, "
                "NVTX, LIKWID, perf events, or aggregation profiling"
            )
        self._track_threads = track_threads
        self._track_async = bool(track_async)
        self._tracker = (
            ConcurrencyTracker(track_async=self._track_async) if track_threads else None
        )

        # Local queries, not collectives: nothing here has to be reached by
        # every rank in lockstep. Rank 0 writes the whole output file at
        # finalize() from data the other ranks send it, so there is no
        # per-rank staging file and no shared directory to agree on.
        self._rank = 0 if self._comm is None else self._comm.Get_rank()
        self._size = 1 if self._comm is None else self._comm.Get_size()
        self._memory_profile_path = (
            self._memray_path(memory_profile_path)
            if use_memray and not deactivate_profiling
            else None
        )
        if self._memory_profile_path is not None:
            from scope_profiler.memray import MemrayAllocationTracker

            self._memray_tracker = MemrayAllocationTracker(
                self._memory_profile_path,
                native_traces=memray_native_traces,
                trace_python_allocators=memray_trace_python_allocators,
                follow_fork=memray_follow_fork,
            )
            self._memray_tracker.start()

        # Environment metadata (hostname, OpenMP threads, versions, ...).
        # Collected on every rank, but only rank 0's copy ends up persisted
        # (see ProfileManager.finalize), so it is treated as global for the run.
        self._metadata = collect_metadata(mpi_size=self._size)
        # Persisted so post-processing can express timestamps relative to the
        # start of the run rather than to the first region entry.
        self._metadata["start_time_ns"] = self._start_time_ns
        # Only stored when set, so that "has a label" stays distinguishable
        # from "was labelled with the empty string" downstream.
        self._label = label or None
        if self._label is not None:
            self._metadata["label"] = self._label
        if self._memory_profile_path is not None:
            self._metadata["memory_profile_path"] = str(self._memory_profile_path)

        self._pylikwid: Any = None
        # Aggregate nesting belongs to this configuration. Keeping it here
        # prevents regions owned by two simultaneously active managers from
        # being mistaken for parent and child calls.
        self._aggregate_stack: list = []
        # markerclose() must run exactly once: it writes the marker file and
        # tears the perfmon module down, so a second call (e.g. a second
        # finalize()) would have nothing left to close.
        self._likwid_closed = False
        if self.use_likwid:
            # pylikwid.markerinit()
            try:
                self._pylikwid = _import_pylikwid()
                self.pylikwid_markerinit()
            except ImportError as e:
                raise ImportError(_pylikwid_import_error(e)) from e

    @classmethod
    def reset(cls):
        """Compatibility no-op retained for callers of older releases.

        Configurations are ordinary per-manager objects now, so there is no
        process-wide instance to reset.
        """

    def pylikwid_markerinit(self):
        """Initialize LIKWID markers if LIKWID is enabled."""
        if self._pylikwid is not None:
            self._pylikwid.markerinit()

    def stop_memory_profiling(self) -> None:
        """Finish Memray's process-wide allocation capture, if enabled."""
        if self._memray_tracker is not None:
            self._memray_tracker.stop()

    def pylikwid_markerclose(self):
        """Close LIKWID markers to finalize measurement regions.

        Idempotent: repeated calls (a second ``finalize()``, say) do nothing.
        """
        if self._likwid_closed or self._pylikwid is None:
            return
        self._likwid_closed = True
        self._pylikwid.markerclose()

    def collect_likwid_results(self, region_names) -> list:
        """Close the LIKWID markers and return the run's counter results.

        Tries three sources, richest first, so a host where LIKWID cannot do
        the fancy parts still ends up with real numbers in the HDF5 file:

        1. the perfmon read-back, run in a subprocess because it can crash
           the interpreter outright on hosts that cannot really count;
        2. LIKWID's marker file, parsed directly --- real values, but no
           event names or derived metrics;
        3. a marker-API snapshot taken before the markers were closed, for
           when there is no marker file at all.

        See :mod:`scope_profiler.likwid_data` for why the first one is fenced
        off.

        Parameters
        ----------
        region_names : iterable of str
            Region names to snapshot via the marker API.

        Returns
        -------
        list of scope_profiler.likwid_data.LikwidRegionResult
            Empty when LIKWID is disabled, the process was not started under
            ``likwid-perfctr -m`` / ``likwid-mpirun -marker``, or the markers
            were already collected.

        Notes
        -----
        A process can only do this once. Closing the markers tears the marker
        API down, and querying it afterwards crashes the interpreter rather
        than raising, so a second ``finalize()`` returns nothing instead of
        reading again. The counters of the whole process therefore end up in
        the file written by the *first* ``finalize()``.
        """
        if not self.use_likwid or self._pylikwid is None or self._likwid_closed:
            return []

        from scope_profiler.likwid_data import (
            collect_marker_results_isolated,
            collect_region_snapshots,
            markers_available,
            parse_marker_file,
            snapshots_to_results,
        )

        if not markers_available():
            # markerinit() degrades to a no-op outside likwid-perfctr, so there
            # are no counters to read -- not an error, just nothing to report.
            self.pylikwid_markerclose()
            return []

        # Taken while the markers are still open; only used if both paths
        # below come up empty.
        snapshots = collect_region_snapshots(self._pylikwid, region_names)
        self.pylikwid_markerclose()

        results = collect_marker_results_isolated()
        if results:
            return results
        return parse_marker_file() or snapshots_to_results(snapshots)

    def likwid_environment(self) -> dict:
        """Return the ``LIKWID_*`` environment variables of this process."""
        from scope_profiler.likwid_data import likwid_environment

        return likwid_environment()

    @property
    def comm(self) -> "Intercomm | None":
        """MPI communicator or None if MPI is unavailable."""
        return self._comm

    @property
    def deactivate_profiling(self) -> bool:
        """Return whether profiling is globally turned off."""
        return self._deactivate_profiling

    @property
    def buffer_limit(self) -> int:
        """Initial per-region buffer capacity; buffers grow beyond it as needed."""
        return self._buffer_limit

    @property
    def file_path(self) -> str:
        """Global output file path for combined profiling data."""
        return self._file_path

    @property
    def output_mode(self) -> str:
        """MPI HDF5 output strategy: auto, direct, or parallel."""
        return self._output_mode

    @property
    def hdf5_compression(self) -> str | None:
        """Compression filter used for timestamp datasets."""
        return self._hdf5_compression

    @property
    def hdf5_compression_level(self) -> int | None:
        """Configured GZIP or Zstandard compression level."""
        return self._hdf5_compression_level

    @property
    def hdf5_chunk_size(self) -> int | None:
        """Maximum number of events per timestamp chunk."""
        return self._hdf5_chunk_size

    @property
    def use_likwid(self) -> bool:
        """Return whether LIKWID profiling is enabled."""
        return self._use_likwid

    @property
    def perf_events(self) -> tuple[str, ...]:
        """Requested Linux perf event names, or an empty tuple when disabled."""
        return self._perf_events

    @property
    def use_line_profiler(self) -> bool:
        """Return whether line_profiler profiling is enabled."""
        return self._use_line_profiler

    @property
    def use_memray(self) -> bool:
        """Return whether Memray allocation profiling is enabled."""
        return self._use_memray

    @property
    def memory_profile_path(self) -> Path | None:
        """Memray capture path, or None when allocation profiling is off."""
        return self._memory_profile_path

    @property
    def use_nvtx(self) -> bool:
        """Return whether NVTX annotations are enabled."""
        return self._use_nvtx

    @property
    def use_gpu_timing(self) -> bool:
        """Return whether CUDA-event GPU timing is enabled."""
        return self._use_gpu_timing

    @property
    def gpu_timing_backend(self):
        """CUDA-event timing backend selector or backend object."""
        return self._gpu_timing_backend

    @property
    def deactivate_file_output(self) -> bool:
        """Return whether the run writes no HDF5 file at all."""
        return self._deactivate_file_output

    @property
    def recursive_profile(self) -> bool:
        """Return whether recursive decorator profiling is enabled by default."""
        return self._recursive_profile

    @property
    def capture_region_source(self) -> bool:
        """Return whether a region's defining source is captured at creation."""
        return self._capture_region_source

    @property
    def aggregation_mode(self) -> bool:
        """Whether regions retain aggregates instead of individual events."""
        return self._aggregation_mode

    @property
    def track_threads(self) -> bool:
        """Whether every call records the thread it ran on."""
        return self._track_threads

    @property
    def track_async(self) -> bool:
        """Whether every call records its asyncio task or greenlet."""
        return self._track_async

    @property
    def tracker(self):
        """This run's :class:`~scope_profiler.concurrency.ConcurrencyTracker`.

        None unless ``track_threads`` is set, which is what keeps the default
        configuration free of any thread, asyncio or greenlet hook.
        """
        return self._tracker

    @property
    def paused(self) -> bool:
        """Whether runtime timing collection is temporarily suspended."""
        return self._paused

    @property
    def start_time_ns(self) -> int:
        """The run's start time (ns, ``perf_counter_ns`` clock).

        The moment the configuration was created. Persisted as metadata and
        used as the timeline origin when reading the results back.
        """
        return self._start_time_ns

    @property
    def label(self) -> str | None:
        """Short name for this run, or None if none was given to ``setup()``."""
        return self._label

    @property
    def metadata(self) -> dict:
        """Environment metadata collected on this rank (hostname, OpenMP threads, ...)."""
        return self._metadata
