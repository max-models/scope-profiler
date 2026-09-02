"""scope-profiler: lightweight region-based profiling for Python and HPC applications."""

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from scope_profiler.call_stack import (
    CallArrays,
    NestingError,
    build_call_arrays,
    build_call_stack,
    call_stack_children,
    call_stack_roots,
)
from scope_profiler.h5reader import CorruptProfileError, read_h5, read_h5_summary
from scope_profiler.json_export import (
    JSONProfileError,
    export_json,
    read_json,
    write_json,
)
from scope_profiler.likwid_data import LikwidRegionResult
from scope_profiler.mpi_region import MPIRegion
from scope_profiler.perf_events import PerfEventError, PerfEventTotals
from scope_profiler.profile_config import ProfilingOptions
from scope_profiler.profile_io import read_profile, write_profile
from scope_profiler.profile_manager import ProfileManager
from scope_profiler.region import EventDataUnavailableError, Region
from scope_profiler.results import ProfilingResults, merge_results

try:
    __version__ = version("scope-profiler")
except PackageNotFoundError:
    __version__ = "unknown"

# Keep these imports visible to language servers without importing the optional
# plotting/reporting dependencies at runtime.  Dynamic module attributes are
# otherwise invisible to hover, completion, and static type checkers.
if TYPE_CHECKING:
    from scope_profiler.chrome_trace_export import export_chrome_trace
    from scope_profiler.html_report import create_html_report
    from scope_profiler.inspection import collect_file_metadata, inspect_file
    from scope_profiler.plotting_scripts import (
        collect_region_statistics,
        plot_duration_timeseries,
        plot_durations,
        plot_flame,
        plot_flame_chart,
        plot_flame_graph,
        plot_gantt,
        plot_perf_events,
        plot_rank_heatmap,
        plot_scaling_efficiency,
        plot_speedup,
        plot_weak_scaling,
        write_region_statistics_json,
    )
    from scope_profiler.prof_export import export_prof
    from scope_profiler.speedscope_export import export_speedscope

__all__ = [
    "CallArrays",
    "CorruptProfileError",
    "EventDataUnavailableError",
    "JSONProfileError",
    "LikwidRegionResult",
    "MPIRegion",
    "NestingError",
    "PerfEventError",
    "PerfEventTotals",
    "ProfileManager",
    "ProfilingOptions",
    "ProfilingResults",
    "Region",
    "build_call_arrays",
    "build_call_stack",
    "call_stack_children",
    "call_stack_roots",
    "collect_file_metadata",
    "collect_region_statistics",
    "create_html_report",
    "export_json",
    "export_chrome_trace",
    "export_prof",
    "export_speedscope",
    "inspect_file",
    "merge_results",
    "plot_duration_timeseries",
    "plot_durations",
    "plot_flame",
    "plot_flame_chart",
    "plot_flame_graph",
    "plot_gantt",
    "plot_perf_events",
    "plot_rank_heatmap",
    "plot_scaling_efficiency",
    "plot_speedup",
    "plot_weak_scaling",
    "read_h5",
    "read_h5_summary",
    "read_json",
    "read_profile",
    "write_json",
    "write_profile",
    "write_region_statistics_json",
]


def __getattr__(name: str) -> Any:
    """Load optional top-level helpers only when they are first used."""
    if name in {
        "collect_region_statistics",
        "plot_duration_timeseries",
        "plot_durations",
        "plot_flame",
        "plot_flame_chart",
        "plot_flame_graph",
        "plot_gantt",
        "plot_rank_heatmap",
        "plot_perf_events",
        "plot_scaling_efficiency",
        "plot_speedup",
        "plot_weak_scaling",
        "write_region_statistics_json",
    }:
        from scope_profiler import plotting_scripts

        return getattr(plotting_scripts, name)
    if name == "export_prof":
        from scope_profiler.prof_export import export_prof

        return export_prof
    if name == "export_chrome_trace":
        from scope_profiler.chrome_trace_export import export_chrome_trace

        return export_chrome_trace
    if name == "export_speedscope":
        from scope_profiler.speedscope_export import export_speedscope

        return export_speedscope
    if name in {"collect_file_metadata", "inspect_file"}:
        from scope_profiler import inspection

        return getattr(inspection, name)
    if name == "create_html_report":
        from scope_profiler.html_report import create_html_report

        return create_html_report
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    """Include the lazily-resolved helpers in ``dir()``."""
    return sorted(set(globals()) | set(__all__))
