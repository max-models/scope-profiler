"""Launching profiling jobs for any program, locally or with SLURM, and packaging their results.

- :class:`ProfilingCase` -- a command to profile at several rank counts; each
  :meth:`~ProfilingCase.launch` runs it locally or submits it to SLURM, and
  :meth:`~ProfilingCase.finalize` packages every run (profiles, results,
  metadata) as it finishes, optionally pushing it to a git results repository.
- :mod:`~scope_profiler.jobs.clusters` -- per-machine presets (``#SBATCH``
  pragmas, module setup, node hardware) from Python or TOML.
- :mod:`~scope_profiler.jobs.config` -- cases described in TOML
  (``scope-profiler jobs run case.toml``).
- :func:`run_sweep` -- many cases from one process.
- :class:`GitResultsStore` -- the results repository.

Nothing here imports MPI or the profiler itself: the driver only writes and
submits job scripts, so it runs on a login node with any Python.
"""

from scope_profiler.jobs.case import (
    ProfilingCase,
    begin_sweep,
    end_sweep,
    finalize_all,
)
from scope_profiler.jobs.clusters import (
    ClusterPreset,
    detect_cluster,
    get_cluster,
    list_clusters,
    load_clusters,
    register_cluster,
)
from scope_profiler.jobs.config import load_case_config, run_case_config
from scope_profiler.jobs.schedulers import (
    DryRunScheduler,
    LocalScheduler,
    SlurmScheduler,
    detect_launcher,
    detect_scheduler,
    render_sbatch_script,
)
from scope_profiler.jobs.store import GitResultsStore
from scope_profiler.jobs.sweep import run_sweep, variant_entries

__all__ = [
    "ClusterPreset",
    "DryRunScheduler",
    "GitResultsStore",
    "LocalScheduler",
    "ProfilingCase",
    "SlurmScheduler",
    "begin_sweep",
    "detect_cluster",
    "detect_launcher",
    "detect_scheduler",
    "end_sweep",
    "finalize_all",
    "get_cluster",
    "list_clusters",
    "load_case_config",
    "load_clusters",
    "register_cluster",
    "render_sbatch_script",
    "run_case_config",
    "run_sweep",
    "variant_entries",
]
