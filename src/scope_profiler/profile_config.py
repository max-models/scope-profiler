"""Singleton configuration for the profiling system."""

import os
import shutil
from time import perf_counter_ns
from typing import TYPE_CHECKING

from scope_profiler.metadata import collect_metadata
from scope_profiler.mpi_launch import get_comm

if TYPE_CHECKING:
    from mpi4py.MPI import Intercomm

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
    return [d for d in dirs if os.path.isdir(d) and not (d in seen or seen.add(d))]


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
    """Singleton class for managing global profiling settings.

    This class centralizes configuration for LIKWID performance counters,
    buffer limits, and file paths. Constructing it is purely local: it reads
    the communicator for rank and size but issues no MPI call of its own, so
    ``setup()`` does not have to be collective.
    """

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        """Ensure only one instance of ProfilingConfig exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        deactivate_profiling: bool = False,
        deactivate_file_output: bool = False,
        use_likwid: bool = False,
        use_line_profiler: bool = False,
        recursive_profile: bool = False,
        buffer_limit: int = 1024,
        file_path: str = "profiling_data.h5",
        label: str | None = None,
        capture_region_source: bool = True,
    ):
        """Initialize the profiling configuration.

        Parameters
        ----------
        deactivate_profiling : bool
            Turn profiling off entirely. Every region becomes a no-op, so
            instrumentation can stay in the code at near-zero cost.
        deactivate_file_output : bool
            Write no HDF5 file at all, not even the run metadata. The recorded
            data then lives only in memory, where
            ``finalize(return_results=True)`` can still return it.
        use_likwid : bool
            Enable LIKWID marker API if available.
        use_line_profiler : bool
            Enable line-by-line profiling via line_profiler.
        recursive_profile : bool
            Enable recursive profiling by default for decorated functions.
        buffer_limit : int
            Initial number of in-memory records to preallocate per region.
            The buffers grow on demand, so this is a starting size, not a cap.
        file_path : str
            Global output file path for combined profiling data.
        label : str or None
            Short name for this run, used by post-processing wherever a run
            has to be named: chart legends, the summary heading, the JSON
            statistics. Defaults to None, in which case the output file's stem
            is used. Persisted as the ``label`` metadata field.
        capture_region_source : bool
            Record where each region is defined (see
            :attr:`~scope_profiler.region.Region.source_text`), once per
            distinct source file, the first time any of its regions is
            created. Measured cost is driven almost entirely by that file's
            total size (an ``ast.parse`` + one tree walk), not by the number
            or size of the regions in it: under a millisecond for a typical
            file of a few hundred lines, regardless of MPI rank count, but
            tenths of a second *per rank* for a single ~10,000-line file with
            many regions -- and that cost is paid independently and
            concurrently by every rank, so on a job with more ranks than idle
            cores it can compound into whole seconds under contention (~0.3s
            at 8 ranks, ~2.9s at 64, measured on a shared, oversubscribed
            login node with such a file). Set to False to skip it entirely if
            that matters for your job, or source is not reliably readable
            (a frozen/packaged deployment).


        Notes
        -----
        MPI is not configurable: collectives are used exactly when the process
        was started by an MPI launcher, so a plain ``python script.py`` never
        touches MPI. See :mod:`scope_profiler.mpi_launch` for the detection and
        its ``SCOPE_PROFILER_MPI`` override.
        """

        if self._initialized:
            return

        # The run's origin, on the perf_counter_ns clock. Persisted as
        # metadata, and the point post-processing measures its relative
        # timeline from.
        self._start_time_ns = perf_counter_ns()

        # Serial runs must not import mpi4py (which would call MPI_Init) nor
        # issue any collective, so the communicator stays None unless this
        # process really is part of an MPI job.
        self._comm = get_comm()
        self._deactivate_profiling = deactivate_profiling
        self._deactivate_file_output = deactivate_file_output
        self._use_likwid = use_likwid
        self._use_line_profiler = use_line_profiler
        self._recursive_profile = recursive_profile
        self._buffer_limit = buffer_limit
        self._file_path = file_path
        self._capture_region_source = capture_region_source

        # Local queries, not collectives: nothing here has to be reached by
        # every rank in lockstep. Rank 0 writes the whole output file at
        # finalize() from data the other ranks send it, so there is no
        # per-rank staging file and no shared directory to agree on.
        self._rank = 0 if self._comm is None else self._comm.Get_rank()
        self._size = 1 if self._comm is None else self._comm.Get_size()

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

        self._pylikwid = None
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
        self._initialized = True

    @classmethod
    def reset(cls):
        """Reset the singleton so it can be reinitialized."""
        cls._instance = None
        cls._initialized = False

    def pylikwid_markerinit(self):
        """Initialize LIKWID markers if LIKWID is enabled."""
        self._pylikwid.markerinit()

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
    def use_likwid(self) -> bool:
        """Return whether LIKWID profiling is enabled."""
        return self._use_likwid

    @property
    def use_line_profiler(self) -> bool:
        """Return whether line_profiler profiling is enabled."""
        return self._use_line_profiler

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
