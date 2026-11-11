"""Cluster presets: the batch settings and hardware of each machine profiling jobs run on.

A preset bundles what a job on one machine (or one partition of it) needs:
the ``#SBATCH`` pragmas (partition, account, time limit, ...), the shell
commands that prepare the environment (``module load ...``), the hardware of
a node (to work out how many nodes a rank count needs) and how to recognise
the machine from its environment.

Nothing about a particular cluster is built in. Presets come from

- :func:`register_cluster` in Python, or
- TOML files (:func:`load_clusters`), including every file listed in the
  ``SCOPE_PROFILER_CLUSTERS`` environment variable (separated by
  ``os.pathsep``), which are read on first use.

A TOML file has one table per preset::

    [clusters.mycluster_cpu]
    cpus_per_node = 128
    gpus_per_node = 0
    setup = ["module purge", "module load gcc openmpi"]
    detect = { HOSTNAME = "mycluster" }   # environment variable -> substring

    [clusters.mycluster_cpu.sbatch]
    partition = "general"
    account = "my_project"
    time = "00:15:00"
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

CLUSTERS_ENV_VAR = "SCOPE_PROFILER_CLUSTERS"


@dataclass
class ClusterPreset:
    """The batch settings and hardware of one machine or partition.

    Parameters
    ----------
    name : str
        Key of the preset, e.g. ``"mycluster_gpu"``.
    sbatch : dict
        ``#SBATCH`` pragmas, keyed by option name (``"time"``,
        ``"partition"``, ...; underscores and dashes are interchangeable).
        ``nodes`` and ``ntasks_per_node`` are filled in per launch.
    setup : list[str]
        Shell commands run at the top of every job script, before the case's
        own setup commands (e.g. ``module load``).
    cpus_per_node, gpus_per_node : int or None
        Hardware of one node. ``cpus_per_node`` is needed to derive the node
        count of a launch that does not give one.
    detect : dict[str, str]
        Environment variable -> substring. The preset matches the current
        machine if any of the variables contains its substring. ``HOSTNAME``
        falls back to :func:`socket.gethostname`.
    description : str
        Free text, shown by ``scope-profiler jobs clusters``.
    """

    name: str
    sbatch: dict[str, Any] = field(default_factory=dict)
    setup: list[str] = field(default_factory=list)
    cpus_per_node: int | None = None
    gpus_per_node: int | None = None
    detect: dict[str, str] = field(default_factory=dict)
    description: str = ""

    def matches(self, environ: dict[str, str] | None = None) -> bool:
        """Whether this preset's ``detect`` rules match the environment."""
        environ = os.environ if environ is None else environ
        for variable, needle in self.detect.items():
            value = environ.get(variable, "")
            if not value and variable == "HOSTNAME":
                value = socket.gethostname()
            if needle and needle in value:
                return True
        return False

    def nodes_for(self, ntasks: int) -> int:
        """Number of nodes needed to place ``ntasks`` ranks, one per core."""
        if not self.cpus_per_node:
            return 1
        return max(1, -(-ntasks // self.cpus_per_node))

    def to_dict(self) -> dict[str, Any]:
        """The preset as a JSON-serialisable dict."""
        return {
            "name": self.name,
            "sbatch": dict(self.sbatch),
            "setup": list(self.setup),
            "cpus_per_node": self.cpus_per_node,
            "gpus_per_node": self.gpus_per_node,
            "detect": dict(self.detect),
            "description": self.description,
        }


_REGISTRY: dict[str, ClusterPreset] = {}
_ENV_LOADED = False


def register_cluster(
    preset: ClusterPreset | None = None, /, **kwargs: Any
) -> ClusterPreset:
    """Register a preset (replacing one of the same name) and return it.

    Accepts a :class:`ClusterPreset` or its fields as keyword arguments::

        register_cluster(name="laptop", cpus_per_node=8)
    """
    if preset is None:
        preset = ClusterPreset(**kwargs)
    elif kwargs:
        raise TypeError("pass either a ClusterPreset or keyword arguments, not both")
    _REGISTRY[preset.name] = preset
    return preset


def _preset_from_table(name: str, table: dict[str, Any]) -> ClusterPreset:
    known = {
        "sbatch",
        "setup",
        "cpus_per_node",
        "gpus_per_node",
        "detect",
        "description",
    }
    unknown = set(table) - known
    if unknown:
        raise ValueError(
            f"cluster {name!r}: unknown key(s) {sorted(unknown)}; expected some of {sorted(known)}"
        )
    return ClusterPreset(
        name=name,
        sbatch=dict(table.get("sbatch", {})),
        setup=list(table.get("setup", [])),
        cpus_per_node=table.get("cpus_per_node"),
        gpus_per_node=table.get("gpus_per_node"),
        detect={str(k): str(v) for k, v in table.get("detect", {}).items()},
        description=str(table.get("description", "")),
    )


def load_clusters(path: str | os.PathLike[str]) -> list[ClusterPreset]:
    """Register every preset of a TOML file (see the module docstring) and return them."""
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    tables = raw.get("clusters", raw)
    return [
        register_cluster(_preset_from_table(name, table))
        for name, table in tables.items()
    ]


def _load_env_clusters() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    for entry in os.environ.get(CLUSTERS_ENV_VAR, "").split(os.pathsep):
        if entry.strip():
            load_clusters(entry.strip())


def list_clusters() -> dict[str, ClusterPreset]:
    """All registered presets, by name (including those of ``SCOPE_PROFILER_CLUSTERS``)."""
    _load_env_clusters()
    return dict(_REGISTRY)


def get_cluster(name: str) -> ClusterPreset:
    """The registered preset called ``name``."""
    presets = list_clusters()
    try:
        return presets[name]
    except KeyError:
        known = ", ".join(sorted(presets)) or "none"
        raise KeyError(
            f"no cluster preset {name!r} (registered: {known}); "
            f"register one with register_cluster() or a TOML file in ${CLUSTERS_ENV_VAR}",
        ) from None


def detect_cluster(environ: dict[str, str] | None = None) -> ClusterPreset | None:
    """The first registered preset whose ``detect`` rules match, or None (e.g. on a laptop)."""
    for preset in list_clusters().values():
        if preset.matches(environ):
            return preset
    return None


def _reset_registry() -> None:
    """Forget every preset (for tests)."""
    global _ENV_LOADED
    _REGISTRY.clear()
    _ENV_LOADED = False
