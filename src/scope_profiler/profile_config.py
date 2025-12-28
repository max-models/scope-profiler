import os
import tempfile
from time import perf_counter_ns
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mpi4py.MPI import Intercomm

try:
    from mpi4py import MPI

    _MPI_AVAILABLE = True
except ImportError:
    MPI = None
    _MPI_AVAILABLE = False

# try:
# import pylikwid
#     _PYLIKWID_AVAILABLE = True
# except ImportError:
#     pylikwid = None
#     _PYLIKWID_AVAILABLE = False


def _import_pylikwid():
    """Dynamically import the pylikwid module.

    Returns
    -------
    module
        The imported `pylikwid` module.

    Raises
    ------
    ImportError
        If the module cannot be imported.
    """
    import pylikwid

    return pylikwid


class ProfilingConfig:
    """Singleton class for managing global profiling settings.

    This class centralizes configuration for time tracing,
    LIKWID performance counters, buffer limits, and file paths.
    It ensures consistent profiling state across MPI ranks and
    creates per-rank temporary storage for profiling output.
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
        profiling_activated: bool = True,
        use_likwid: bool = False,
        time_trace: bool = True,
        flush_to_disk: bool = True,
        buffer_limit: int = 100_000,
        file_path: str = "profiling_data.h5",
    ):
        """Initialize the profiling configuration.

        Parameters
        ----------
        profiling_activated : bool
            Whether profiling features are enabled.
        use_likwid : bool
            Enable LIKWID marker API if available.
        time_trace : bool
            Enable time trace profiling.
        flush_to_disk : bool
            If True, flush profiling buffers to disk periodically.
        buffer_limit : int
            Maximum number of in-memory records before flushing.
        file_path : str
            Global output file path for combined profiling data.
        """

        if self._initialized:
            return

        self._config_creation_time = perf_counter_ns()

        if _MPI_AVAILABLE:
            self._comm = MPI.COMM_WORLD
        else:
            self._comm = None
        self._profiling_activated = profiling_activated
        self._use_likwid = use_likwid
        self._time_trace = time_trace
        self._flush_to_disk = flush_to_disk
        self._buffer_limit = buffer_limit
        self._file_path = file_path

        self._rank = 0 if self._comm is None else self._comm.Get_rank()
        self._size = 1 if self._comm is None else self._comm.Get_size()

        # Only rank 0 creates the TemporaryDirectory
        if self._rank == 0:
            self._temp_dir_obj = tempfile.TemporaryDirectory(prefix="profile_h5_")
            temp_dir = self._temp_dir_obj.name
        else:
            temp_dir = None
            self._temp_dir_obj = None  # still define to keep the attribute

        # Broadcast the directory path to all ranks
        if self._comm is not None:
            temp_dir = self._comm.bcast(temp_dir, root=0)

        self.temp_dir = temp_dir
        self._global_file_path = self.file_path

        # Temporary file with rank-specific timings
        self._local_file_path = self.get_local_filepath(self._rank)

        self._pylikwid = None
        if self.use_likwid:
            # pylikwid.markerinit()
            try:
                self._pylikwid = _import_pylikwid()
                self.pylikwid_markerinit()
            except ImportError as e:
                raise ImportError(
                    "LIKWID profiling requested but pylikwid module not installed"
                ) from e
        self._initialized = True

    def get_local_filepath(self, rank):
        """Return the per-rank local profiling file path.

        Parameters
        ----------
        rank : int
            MPI rank identifier.

        Returns
        -------
        str
            The path to the per-rank HDF5 file.
        """
        return os.path.join(self.temp_dir, f"rank_{rank}.h5")

    @classmethod
    def reset(cls):
        """Reset the singleton so it can be reinitialized."""
        cls._instance = None
        cls._initialized = False

    def pylikwid_markerinit(self):
        """Initialize LIKWID markers if LIKWID is enabled."""
        self._pylikwid.markerinit()

    def pylikwid_markerclose(self):
        """Close LIKWID markers to finalize measurement regions."""
        self._pylikwid.markerclose()

    @property
    def comm(self) -> "Intercomm | None":
        """MPI communicator or None if MPI is unavailable."""
        return self._comm

    @property
    def profiling_activated(self) -> bool:
        """Return whether profiling is globally enabled."""
        return self._profiling_activated

    @property
    def buffer_limit(self) -> int:
        """Maximum number of buffered profiling records."""
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
    def flush_to_disk(self) -> bool:
        """Return whether profiling buffers should flush to disk."""
        return self._flush_to_disk

    @property
    def time_trace(self) -> bool:
        """Return whether time trace profiling is enabled."""
        return self._time_trace

    @property
    def config_creation_time(self) -> int:
        """Timestamp (ns) when the configuration was created."""
        return self._config_creation_time
