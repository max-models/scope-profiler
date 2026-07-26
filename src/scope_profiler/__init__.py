"""scope-profiler: lightweight region-based profiling for Python and HPC applications."""

from importlib.metadata import PackageNotFoundError, version

from scope_profiler.h5reader import ProfilingH5Reader
from scope_profiler.mpi_region import MPIRegion
from scope_profiler.profile_manager import ProfileManager
from scope_profiler.region import Region

try:
    __version__ = version("scope-profiler")
except PackageNotFoundError:
    __version__ = "unknown"

# Plotting pulls in the optional maxplotlib stack, so these are resolved on
# first access (PEP 562) rather than at import time, keeping
# `import scope_profiler` cheap inside the applications being profiled.
_LAZY_PLOTTING = frozenset(
    {
        "collect_region_statistics",
        "plot_durations",
        "plot_flame",
        "plot_gantt",
        "plot_speedup",
        "write_region_statistics_json",
    }
)

__all__ = [
    "MPIRegion",
    "ProfileManager",
    "ProfilingH5Reader",
    "Region",
    *sorted(_LAZY_PLOTTING),
]


def __getattr__(name: str):
    """Resolve the plotting helpers lazily; see ``_LAZY_PLOTTING``."""
    if name in _LAZY_PLOTTING:
        from scope_profiler import plotting_scripts

        return getattr(plotting_scripts, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    """Include the lazily-resolved plotting helpers in ``dir()``."""
    return sorted(set(globals()) | _LAZY_PLOTTING)
