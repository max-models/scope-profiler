from time import perf_counter_ns
from typing import TYPE_CHECKING

import h5py
import numpy as np

from scope_profiler.profile_config import ProfilingConfig

if TYPE_CHECKING:
    from mpi4py.MPI import Intercomm


def _import_pylikwid():
    import pylikwid

    return pylikwid


class MockProfileRegion:
    """A dummy ProfileRegion that does nothing, used when profiling is disabled."""

    def __init__(self, region_name, config=None):
        self._region_name = region_name
        self._ncalls = 0
        self._started = False

    def append(self, start, end):
        pass

    def flush(self):
        pass

    @property
    def region_name(self):
        return self._region_name

    @property
    def num_calls(self):
        return self._ncalls

    @property
    def started(self):
        return False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class ProfileRegion:
    """Context manager for profiling specific code regions using LIKWID markers."""

    def __init__(
        self,
        region_name: str,
        config: ProfilingConfig,
    ):
        self._region_name = region_name
        self._config = config

        self._comm = self.config.comm
        self._time_trace = self.config.time_trace
        self._buffer_limit = self.config.buffer_limit

        self._flush_to_disk = self.config.flush_to_disk
        self._profiling_activated = self.config.profiling_activated

        # Timer data
        self._ncalls = 0
        self._start_times = []
        self._end_times = []
        self._started = False

        self._group_path = f"regions/{self._region_name}"

        # Construct per-rank filename
        with h5py.File(self.config._local_file_path, "a") as f:
            grp = f.require_group(self._group_path)
            for name in ("start_times", "end_times"):
                if name not in grp:
                    grp.create_dataset(
                        name,
                        shape=(0,),
                        maxshape=(None,),
                        dtype="i8",
                        chunks=True,
                        # compression="gzip",
                    )

        # Cache likwid markers
        if self.config.use_likwid:
            self._likwid_marker_start = _import_pylikwid().markerstartregion
            self._likwid_marker_stop = _import_pylikwid().markerstopregion
        else:
            self._likwid_marker_start = None
            self._likwid_marker_stop = None

    def append(self, start: float, end: float) -> None:
        """Append a timing directly (used by decorator for speed)."""
        self._start_times.append(start)
        self._end_times.append(end)
        if self._flush_to_disk and len(self._start_times) >= self._buffer_limit:
            self.flush()

    def flush(self) -> None:
        """Append buffered profiling data to the HDF5 file and clear memory."""
        if not self._start_times:
            return

        starts = self.start_times
        ends = self.end_times

        with h5py.File(self.config._local_file_path, "a") as f:
            grp = f[self._group_path]
            for name, data in [
                ("start_times", starts),
                ("end_times", ends),
            ]:
                ds = grp[name]
                old_size = ds.shape[0]
                new_size = old_size + len(data)
                ds.resize((new_size,))
                ds[old_size:new_size] = data

        self._start_times.clear()
        self._end_times.clear()

    @property
    def comm(self) -> "Intercomm | None":
        return self._comm

    @property
    def profiling_activated(self) -> bool:
        return self._profiling_activated

    @property
    def config(self) -> ProfilingConfig:
        return self._config

    @property
    def durations(self) -> np.ndarray:
        return self.end_times - self.start_times

    @property
    def end_times(self) -> np.ndarray:
        return np.array(self._end_times, dtype=int) - self.config.config_creation_time

    @property
    def flush_to_disk(self) -> bool:
        return self._flush_to_disk

    @property
    def num_calls(self) -> int:
        return self._ncalls

    @property
    def region_name(self) -> str:
        return self._region_name

    @property
    def start_times(self) -> np.ndarray:
        return np.array(self._start_times, dtype=int) - self.config.config_creation_time

    @property
    def started(self) -> bool:
        return self._started

    def __enter__(self):
        if not self._profiling_activated:
            return self

        # Pylikwid markerstartregion
        if self._likwid_marker_start:
            self._likwid_marker_start(self._region_name)

        if self._time_trace:
            self._start_time = perf_counter_ns()
            self._start_times.append(self._start_time)
            self._started = True

        self._ncalls += 1

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not self._profiling_activated:
            return

        # Pylikwid markerstartregion
        if self._likwid_marker_stop:
            self._likwid_marker_stop(self._region_name)

        if self._time_trace and self.started:
            end_time = perf_counter_ns()
            self._end_times.append(end_time)
            self._started = False

            if self._flush_to_disk and len(self._start_times) >= self._buffer_limit:
                self.flush()
