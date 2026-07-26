"""Collects descriptive metadata about the environment a profiling run executed in.

Metadata is gathered on every rank but only rank 0's copy is persisted (see
``ProfileManager.finalize``), so it describes the run as a whole. Per-task
values such as ``SLURM_PROCID`` therefore reflect rank 0.
"""

import ctypes
import datetime
import getpass
import os
import platform
import socket
import subprocess
from typing import Dict, List, Union

MetadataValue = Union[str, int, List[str]]

# Common OpenMP runtime library names across platforms/compilers.
_OMP_LIBRARY_NAMES = (
    "libomp.so",
    "libgomp.so.1",
    "libiomp5.so",
    "libomp.dylib",
    "libiomp5.dylib",
)

# Environment variables recorded verbatim, under their own (upper-case) names,
# when set. These are what usually explains "why did this run behave
# differently" on a cluster: the toolchain, the module stack and the
# interpreter it resolved to.
_ENVIRONMENT_VARIABLES = (
    "LD_LIBRARY_PATH",
    "LOADEDMODULES",
    "MODULEPATH",
    "MODULESHOME",
    "MODULES_CMD",
    "MODULES_RUN_QUARANTINE",
    "PATH",
    "PYTHON_HOME",
    "PYTHON_INC",
    "PYTHON_INCLUDE",
    "PYTHON_LIB",
    "VIRTUAL_ENV",
)

# Every SLURM_*/SLURMD_* variable present is captured, so a run can be traced
# back to its batch job without knowing which fields the site exports.
_SLURM_PREFIXES = ("SLURM_", "SLURMD_")

# HDF5 attributes cannot exceed 64 KB; a pathological PATH or LD_LIBRARY_PATH
# would otherwise make finalize() fail at the very end of a run.
_MAX_VALUE_CHARS = 60_000
_TRUNCATION_MARKER = "...[truncated]"


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


def _detect_chip_information() -> str:
    """Best-effort description of the CPU this run executed on.

    ``platform.processor()`` is close to useless on both Linux (empty) and
    macOS (just ``arm``), so the model name is read from the platform's own
    source first and only then falls back to what ``platform`` reports.
    """
    system = platform.system()

    if system == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    # x86 uses "model name", arm64 "Model name" via lscpu-style
                    # kernels; both appear as "<key> : <value>".
                    key, _, value = line.partition(":")
                    if key.strip().lower() in ("model name", "cpu model"):
                        return value.strip()
        except OSError:
            pass
    elif system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass

    return platform.processor() or platform.machine() or "unknown"


def _collect_environment_variables() -> Dict[str, str]:
    """Return the recorded environment variables that are actually set."""
    collected = {
        name: os.environ[name]
        for name in _ENVIRONMENT_VARIABLES
        if os.environ.get(name)
    }
    collected.update(
        {
            name: value
            for name, value in os.environ.items()
            if name.startswith(_SLURM_PREFIXES) and value
        }
    )
    return collected


def _loaded_modules() -> List[str]:
    """Parse ``LOADEDMODULES`` into a list of module names.

    Environment Modules and Lmod both export a colon-separated list; the
    variable is absent (and the list empty) off a module-based system.
    """
    loaded = os.environ.get("LOADEDMODULES", "")
    return [module for module in loaded.split(":") if module]


def _truncate(value: MetadataValue) -> MetadataValue:
    """Clip over-long strings so they remain storable as HDF5 attributes."""
    if isinstance(value, str) and len(value) > _MAX_VALUE_CHARS:
        return value[: _MAX_VALUE_CHARS - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER
    return value


def collect_metadata(mpi_size: int = 1) -> Dict[str, MetadataValue]:
    """Gather metadata describing the current run's environment.

    Parameters
    ----------
    mpi_size : int, optional
        Number of MPI ranks the run was launched with (default: 1). Used to
        derive ``total_cores`` (``mpi_size * omp_num_threads``), a single
        combined parallelism value useful as a scaling-plot x-axis.

    Returns
    -------
    dict
        Mapping of metadata field name to a str, int or list-of-str value,
        suitable for storing as HDF5 attributes. Derived fields use
        lower-case names; captured environment variables keep their own
        upper-case names (``PATH``, ``SLURM_JOB_ID``, ...) and are only
        present when set.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        scope_profiler_version = version("scope-profiler")
    except PackageNotFoundError:
        scope_profiler_version = "unknown"

    omp_num_threads = _detect_omp_num_threads()

    metadata: Dict[str, MetadataValue] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "uname": " ".join(platform.uname()),
        "chip_information": _detect_chip_information(),
        "python_version": platform.python_version(),
        "scope_profiler_version": scope_profiler_version,
        "working_directory": os.getcwd(),
        "omp_num_threads": omp_num_threads,
        "mpi_size": mpi_size,
        "total_cores": mpi_size * omp_num_threads,
        "user": getpass.getuser(),
        "modules": _loaded_modules(),
    }
    metadata.update(_collect_environment_variables())

    return {key: _truncate(value) for key, value in metadata.items()}
