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


def _import_pylikwid():
    import pylikwid

    return pylikwid


class ProfilingConfig:
    """Singleton class for managing global profiling configuration."""

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
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

        comm = self.comm  # TODO, just use MPI.COMM_WORLD
        self._rank = 0 if comm is None else comm.Get_rank()
        self._size = 1 if comm is None else comm.Get_size()

        self.temp_dir_obj = tempfile.TemporaryDirectory(prefix="profile_h5_")
        self.temp_dir = self.temp_dir_obj.name

        self._global_file_path = self.file_path

        # Temporary file with rank-specific timings
        self._local_file_path = self.get_local_filepath(self._rank)

        self._pylikwid = None
        if self.use_likwid:
            try:
                self._pylikwid = _import_pylikwid()
                self.pylikwid_markerinit()
            except ImportError as e:
                raise ImportError(
                    "LIKWID profiling requested but pylikwid module not installed"
                ) from e
        self._initialized = True

    def get_local_filepath(self, rank):
        return os.path.join(self.temp_dir, f"rank_{rank}.h5")

    @classmethod
    def reset(cls):
        """Reset the singleton so it can be reinitialized."""
        cls._instance = None
        cls._initialized = False

    def pylikwid_markerinit(self):
        """Initialize LIKWID profiling markers."""
        if self.use_likwid and self._pylikwid:
            self._pylikwid.markerinit()

    def pylikwid_markerclose(self):
        """Close LIKWID profiling markers."""
        if self.use_likwid and self._pylikwid:
            self._pylikwid.markerclose()

    @property
    def comm(self) -> "Intercomm | None":
        return self._comm

    @property
    def profiling_activated(self) -> bool:
        return self._profiling_activated

    @property
    def buffer_limit(self) -> int:
        return self._buffer_limit

    @property
    def file_path(self) -> str:
        return self._file_path

    @property
    def use_likwid(self) -> bool:
        return self._use_likwid

    @property
    def flush_to_disk(self) -> bool:
        return self._flush_to_disk

    @property
    def time_trace(self) -> bool:
        return self._time_trace

    @property
    def config_creation_time(self) -> int:
        return self._config_creation_time
