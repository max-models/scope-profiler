"""scope-profiler: lightweight region-based profiling for Python and HPC applications."""

from importlib.metadata import PackageNotFoundError, version

from scope_profiler.call_stack import (
    build_call_stack,
    call_stack_children,
    call_stack_roots,
)
from scope_profiler.h5reader import read_h5
from scope_profiler.likwid_data import LikwidRegionResult
from scope_profiler.mpi_region import MPIRegion
from scope_profiler.profile_manager import ProfileManager
from scope_profiler.region import Region
from scope_profiler.results import ProfilingResults

try:
    __version__ = version("scope-profiler")
except PackageNotFoundError:
    __version__ = "unknown"

# Plotting pulls in the optional maxplotlib stack, and the exporters pull in
# the plotting module, so these are resolved on first access (PEP 562) rather
# than at import time, keeping `import scope_profiler` cheap inside the
# applications being profiled.
_LAZY_ATTRS = {
    "collect_region_statistics": "plotting_scripts",
    "plot_duration_timeseries": "plotting_scripts",
    "plot_durations": "plotting_scripts",
    "plot_flame": "plotting_scripts",
    "plot_gantt": "plotting_scripts",
    "plot_speedup": "plotting_scripts",
    "write_region_statistics_json": "plotting_scripts",
    "export_prof": "prof_export",
    "export_speedscope": "speedscope_export",
    "collect_file_metadata": "inspection",
    "inspect_file": "inspection",
}

__all__ = [
    "LikwidRegionResult",
    "MPIRegion",
    "ProfileManager",
    "ProfilingResults",
    "Region",
    "build_call_stack",
    "call_stack_children",
    "call_stack_roots",
    "read_h5",
    *sorted(_LAZY_ATTRS),
]


def __getattr__(name: str):
    """Resolve the plotting and export helpers lazily; see ``_LAZY_ATTRS``."""
    module_name = _LAZY_ATTRS.get(name)
    if module_name is not None:
        import importlib

        module = importlib.import_module(f"scope_profiler.{module_name}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    """Include the lazily-resolved helpers in ``dir()``."""
    return sorted(set(globals()) | set(_LAZY_ATTRS))
