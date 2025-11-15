import os
import tempfile
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mpi4py.MPI import Intercomm


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
            # Default values
            cls._instance.comm = None
            cls._instance.profiling_activated = True
            cls._instance.use_likwid = False
            cls._instance.time_trace = False
            cls._instance.flush_to_disk = True
            cls._instance.buffer_limit = 10_000
            cls._instance.file_path = "profiling_data.h5"
        return cls._instance

    def __init__(
        self,
        comm: "Intercomm | None" = None,
        profiling_activated: bool = True,
        use_likwid: bool = False,
        time_trace: bool = True,
        flush_to_disk: bool = True,
        buffer_limit: int = 10_000,
        file_path: str = "profiling_data.h5",
    ):
        if self._initialized:
            return

        self.config_creation_time = time.perf_counter_ns()

        # Only update if value provided
        self.comm = comm
        self.profiling_activated = profiling_activated
        self.use_likwid = use_likwid
        self.time_trace = time_trace
        self.flush_to_disk = flush_to_disk
        self.buffer_limit = buffer_limit
        self.file_path = file_path

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

    @comm.setter
    def comm(self, value: "Intercomm | None") -> None:
        assert value is None or isinstance(value, Intercomm)
        self._comm = value

    @property
    def profiling_activated(self) -> bool:
        return self._profiling_activated

    @profiling_activated.setter
    def profiling_activated(self, value: bool) -> None:
        assert isinstance(value, bool)
        self._profiling_activated = value

    @property
    def buffer_limit(self) -> int:
        return self._buffer_limit

    @buffer_limit.setter
    def buffer_limit(self, value: int) -> None:
        assert isinstance(value, int)
        self._buffer_limit = value

    @property
    def file_path(self) -> str:
        return self._file_path

    @file_path.setter
    def file_path(self, value: str) -> None:
        assert isinstance(value, str)
        self._file_path = value

    @property
    def use_likwid(self) -> bool:
        return self._likwid

    @use_likwid.setter
    def use_likwid(self, value: bool) -> None:
        assert isinstance(value, bool)
        self._likwid = value

    @property
    def flush_to_disk(self) -> bool:
        return self._flush_to_disk

    @flush_to_disk.setter
    def flush_to_disk(self, value) -> None:
        if not isinstance(value, bool):
            raise TypeError("flush_to_disk must be a bool")
        self._flush_to_disk = value

    @property
    def time_trace(self) -> bool:
        return self._time_trace

    @time_trace.setter
    def time_trace(self, value: bool) -> None:
        assert isinstance(value, bool)
        self._time_trace = value

    @property
    def config_creation_time(self) -> int:
        return self._config_creation_time

    @config_creation_time.setter
    def config_creation_time(self, value: int) -> None:
        assert isinstance(value, int)
        self._config_creation_time = value
