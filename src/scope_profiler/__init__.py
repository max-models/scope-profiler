"""scope-profiler: lightweight region-based profiling for Python and HPC applications."""

from importlib.metadata import PackageNotFoundError, version

from scope_profiler.profile_manager import ProfileManager

try:
    __version__ = version("scope-profiler")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "ProfileManager",
]
