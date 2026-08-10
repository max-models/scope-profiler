"""The isolated LIKWID collector must never take its parent's MPI with it.

``collect_marker_results_isolated`` forks a child from a live MPI rank. Open
MPI does not support that at all, and it turns fatal if the child initializes
MPI itself: it attaches to the rank's shared-memory transport and tears the
segment down when it exits, after which the parent's next MPI call segfaults
inside the progress engine. That is invisible in a serial test run and only
shows up under ``likwid-mpirun``, so it is pinned down here.
"""

import subprocess
import sys

from scope_profiler.likwid_data import _child_environment
from scope_profiler.mpi_launch import LAUNCHER_ENV_VARS


def test_child_environment_disables_mpi(monkeypatch):
    """Every launcher variable is stripped and the override is set."""
    for var in LAUNCHER_ENV_VARS:
        monkeypatch.setenv(var, "0")

    env = _child_environment()

    assert env["SCOPE_PROFILER_MPI"] == "0"
    for var in LAUNCHER_ENV_VARS:
        assert var not in env, f"{var} would make the child join the MPI job"


def test_child_environment_keeps_the_import_path(monkeypatch):
    """The child still has to find this scope_profiler and pylikwid."""
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")

    env = _child_environment()

    assert "/somewhere/else" in env["PYTHONPATH"]
    for path in sys.path:
        if path:
            assert path in env["PYTHONPATH"]


def test_child_does_not_import_mpi4py_even_under_a_launcher(monkeypatch):
    """The end-to-end property: importing scope_profiler must not call MPI_Init.

    ``ProfileManager._config = ProfilingConfig()`` runs at import time and
    resolves the communicator, so a child that still looks like a rank would
    initialize MPI just by importing the package.
    """
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, scope_profiler; print('mpi4py.MPI' in sys.modules)",
        ],
        env=_child_environment(),
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )

    assert result.stdout.strip() == "False", (
        "the isolated collector's environment let the child import mpi4py, "
        "which calls MPI_Init inside a process forked from an MPI rank"
    )
