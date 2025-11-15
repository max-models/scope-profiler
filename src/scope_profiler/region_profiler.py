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
        self.region_name = region_name
        self.num_calls = 0

    def append(self, start, end):
        pass

    def flush(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def get_durations_numpy(self) -> None:
        return None

    def get_end_times_numpy(self) -> None:
        return None

    def get_start_times_numpy(self) -> None:
        return None


class ProfileRegion:
    """Context manager for profiling specific code regions using LIKWID markers."""

    __slots__ = (
        "region_name",
        "config",
        "comm",
        "time_trace",
        "buffer_limit",
        "flush_to_disk",
        "profiling_activated",
        "num_calls",
        "start_times",
        "end_times",
        "group_path",
        "likwid_marker_start",
        "likwid_marker_stop",
    )

    def __init__(
        self,
        region_name: str,
        config: ProfilingConfig,
    ):
        self.region_name = region_name
        self.config = config

        self.comm = self.config.comm
        self.time_trace = self.config.time_trace
        self.buffer_limit = self.config.buffer_limit

        self.flush_to_disk = self.config.flush_to_disk
        self.profiling_activated = self.config.profiling_activated

        # Timer data
        self.num_calls = 0
        self.start_times = []
        self.end_times = []

        self.group_path = f"regions/{self.region_name}"

        # Construct per-rank filename
        with h5py.File(self.config._local_file_path, "a") as f:
            grp = f.require_group(self.group_path)
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
            self.likwid_marker_start = _import_pylikwid().markerstartregion
            self.likwid_marker_stop = _import_pylikwid().markerstopregion
        else:
            self.likwid_marker_start = None
            self.likwid_marker_stop = None

    def append(self, start: float, end: float) -> None:
        """Append a timing directly (used by decorator for speed)."""
        self.start_times.append(start)
        self.end_times.append(end)
        if self.flush_to_disk and len(self.start_times) >= self.buffer_limit:
            self.flush()

    def flush(self) -> None:
        """Append buffered profiling data to the HDF5 file and clear memory."""
        if not self.start_times:
            return

        starts = self.get_start_times_numpy()
        ends = self.get_end_times_numpy()

        with h5py.File(self.config._local_file_path, "a") as f:
            grp = f[self.group_path]
            for name, data in [
                ("start_times", starts),
                ("end_times", ends),
            ]:
                ds = grp[name]
                old_size = ds.shape[0]
                new_size = old_size + len(data)
                ds.resize((new_size,))
                ds[old_size:new_size] = data

        self.start_times.clear()
        self.end_times.clear()

    def __enter__(self):
        if not self.profiling_activated:
            return self

        # Pylikwid markerstartregion
        if self.likwid_marker_start:
            self.likwid_marker_start(self.region_name)

        if self.time_trace:
            self.start_times.append(perf_counter_ns())

        self.num_calls += 1

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not self.profiling_activated:
            return

        # Pylikwid markerstartregion
        if self.likwid_marker_stop:
            self.likwid_marker_stop(self.region_name)

        if self.time_trace:
            self.end_times.append(perf_counter_ns())

            if self.flush_to_disk and len(self.start_times) >= self.buffer_limit:
                self.flush()

    def get_durations_numpy(self) -> np.ndarray:
        return self.get_end_times_numpy() - self.get_start_times_numpy()

    def get_end_times_numpy(self) -> np.ndarray:
        return np.array(self.end_times, dtype=int) - self.config.config_creation_time

    def get_start_times_numpy(self) -> np.ndarray:
        return np.array(self.start_times, dtype=int) - self.config.config_creation_time
