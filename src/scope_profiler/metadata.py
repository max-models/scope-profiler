"""Collects descriptive metadata about the environment a profiling run executed in."""

import ctypes
import datetime
import getpass
import os
import platform
import socket
from typing import Dict, Union

MetadataValue = Union[str, int]

# Common OpenMP runtime library names across platforms/compilers.
_OMP_LIBRARY_NAMES = (
    "libomp.so",
    "libgomp.so.1",
    "libiomp5.so",
    "libomp.dylib",
    "libiomp5.dylib",
)


def _detect_omp_num_threads() -> int:
    """Best-effort detection of the number of OpenMP threads available.

    Tries to query the OpenMP runtime directly via ``omp_get_max_threads``,
    so the recorded value is correct even when ``OMP_NUM_THREADS`` is unset
    (OpenMP then defaults to the number of available cores rather than 1).
    Falls back to the ``OMP_NUM_THREADS`` environment variable, then to 1.
    """
    for libname in _OMP_LIBRARY_NAMES:
        try:
            lib = ctypes.CDLL(libname)
            return int(lib.omp_get_max_threads())
        except (OSError, AttributeError):
            continue

    env_value = os.environ.get("OMP_NUM_THREADS")
    if env_value:
        try:
            # OMP_NUM_THREADS may be a comma-separated list for nested
            # parallelism; the first value is the outermost level.
            return int(env_value.split(",")[0].strip())
        except ValueError:
            pass

    return 1


def collect_metadata() -> Dict[str, MetadataValue]:
    """Gather metadata describing the current run's environment.

    Returns
    -------
    dict
        Mapping of metadata field name to a str or int value, suitable for
        storing as HDF5 attributes.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        scope_profiler_version = version("scope-profiler")
    except PackageNotFoundError:
        scope_profiler_version = "unknown"

    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "scope_profiler_version": scope_profiler_version,
        "working_directory": os.getcwd(),
        "omp_num_threads": _detect_omp_num_threads(),
        "user": getpass.getuser(),
    }
