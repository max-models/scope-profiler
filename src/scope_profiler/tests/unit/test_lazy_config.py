"""Importing the library must never join an MPI job.

Resolving the communicator imports mpi4py, which calls ``MPI_Init``. If that
happened at import time, ``import scope_profiler`` inside any process the
launcher marked as a rank -- including one forked from a rank, as the LIKWID
counter read-back does -- would silently enter MPI and corrupt the parent's
shared-memory transport. So the configuration is built on first use, not at
import.
"""

import subprocess
import sys

from scope_profiler import ProfileManager
from scope_profiler.mpi_launch import LAUNCHER_ENV_VARS

# What a rank's environment looks like to a child process.
RANK_ENV = {"OMPI_COMM_WORLD_RANK": "0"}


def run_probe(code: str, extra_env: dict) -> str:
    """Run ``code`` in a fresh interpreter with ``extra_env`` added."""
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [path for path in sys.path if path] + [env.get("PYTHONPATH", "")],
    ).strip(os.pathsep)
    env.update(extra_env)
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return result.stdout.strip()


def test_importing_under_a_launcher_does_not_initialize_mpi():
    """The property this whole module exists for."""
    probe = "import sys, scope_profiler; print('mpi4py.MPI' in sys.modules)"
    assert run_probe(probe, RANK_ENV) == "False"


def test_every_launcher_variable_is_covered():
    """Not just Open MPI's: any launcher's per-rank variable must be safe."""
    for var in LAUNCHER_ENV_VARS:
        probe = "import sys, scope_profiler; print('mpi4py.MPI' in sys.modules)"
        assert run_probe(probe, {var: "0"}) == "False", f"{var} pulled MPI in"


def test_the_config_is_resolved_on_first_use():
    """Deferred, not skipped: asking for the config still sets MPI up."""
    probe = (
        "import sys\n"
        "from scope_profiler import ProfileManager\n"
        "before = 'mpi4py.MPI' in sys.modules\n"
        "ProfileManager.get_config()\n"
        "print(before, 'mpi4py.MPI' in sys.modules)"
    )
    assert run_probe(probe, RANK_ENV) == "False True"


def test_get_config_creates_one_and_keeps_it():
    ProfileManager._reset()
    assert ProfileManager._config is None

    config = ProfileManager.get_config()
    assert config is not None
    assert ProfileManager.get_config() is config

    ProfileManager._reset()
    assert ProfileManager._config is None


def test_reset_restores_the_state_a_fresh_import_leaves():
    """A reset is indistinguishable from never having configured anything."""
    from scope_profiler.region_profiler import DisabledProfileRegion

    ProfileManager.setup(deactivate_file_output=True)
    with ProfileManager.profile_region("something"):
        pass
    assert ProfileManager.get_all_regions()

    ProfileManager._reset()

    assert ProfileManager._config is None
    assert ProfileManager._region_cls is DisabledProfileRegion
    assert ProfileManager.get_all_regions() == {}
