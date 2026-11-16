"""Profiling cases described in TOML, for ``scope-profiler jobs run case.toml``.

A case file needs no Python. Relative paths (``inputs``, ``work_dir``,
``repo`` and cluster files) are relative to the TOML file; the command runs in
its run directory, so it refers to its inputs through ``{case_dir}``, where
they are copied::

    [case]
    label = "heat_strong_scaling"
    name = "Heat equation, strong scaling"
    command = ["python", "{case_dir}/heat.py", "--profile-out", "{profile}"]
    inputs = ["heat.py"]
    postprocess = ["python", "{case_dir}/plot.py", "{run_dir}"]
    env = { OMP_NUM_THREADS = "1" }
    metadata = { code = "heat", grid = "256^3" }

    [[launch]]                 # one table per group of launches
    ntasks = [1, 2, 4, 8]      # an int or a list
    sbatch = { time = "00:10:00" }

    [variants.jacobi]          # optional: one case per variant
    args = ["--solver", "jacobi"]
    [variants.cg]
    args = ["--solver", "cg"]
    description = "conjugate gradients"

    [store]                    # optional, used with --upload
    url = "git@github.com:me/profiling-results.git"

    [clusters.mycluster]       # optional presets, see scope_profiler.jobs.clusters
    cpus_per_node = 128
    sbatch = { partition = "general", account = "me" }

Every key of ``[case]`` is a :class:`~scope_profiler.jobs.case.ProfilingCase`
argument; every key of ``[[launch]]`` one of
:meth:`~scope_profiler.jobs.case.ProfilingCase.launch`. A variant's ``args``
come before the launch's own, its ``env`` and ``metadata`` are merged into
the case's, and its key is appended to the label.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from scope_profiler.jobs.case import ProfilingCase, finalize_all
from scope_profiler.jobs.clusters import _preset_from_table, register_cluster
from scope_profiler.jobs.store import GitResultsStore

_CASE_KEYS = {f.name for f in dataclasses.fields(ProfilingCase) if f.init} - {
    "store",
    "upload",
}
_LAUNCH_KEYS = {"ntasks", "nodes", "args", "env", "metadata", "cluster", "sbatch"}
_VARIANT_KEYS = {"args", "env", "metadata", "description", "name"}
_TOP_KEYS = {"case", "launch", "variants", "store", "clusters"}


class CaseConfigError(ValueError):
    """An invalid case TOML file."""


def _check_keys(where: str, table: dict[str, Any], allowed: set[str]) -> None:
    unknown = set(table) - allowed
    if unknown:
        raise CaseConfigError(
            f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(allowed)}"
        )


def load_case_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read and validate a case TOML file; registers its ``[clusters]`` presets."""
    path = Path(path).resolve()
    try:
        with path.open("rb") as stream:
            raw = tomllib.load(stream)
    except tomllib.TOMLDecodeError as exc:
        raise CaseConfigError(f"invalid TOML in {path}: {exc}") from exc
    _check_keys(str(path), raw, _TOP_KEYS)

    case = dict(raw.get("case", {}))
    _check_keys(f"{path} [case]", case, _CASE_KEYS)
    for key in ("label", "command"):
        if key not in case:
            raise CaseConfigError(f"{path} [case]: {key!r} is required")
    base = path.parent
    case["inputs"] = [str((base / item).resolve()) for item in case.get("inputs", [])]
    for key in ("work_dir", "repo"):
        if key in case:
            case[key] = str((base / case[key]).resolve())
    if "collect" in case:
        case["collect"] = tuple(case["collect"])
    if "results_suffixes" in case:
        case["results_suffixes"] = tuple(case["results_suffixes"])

    launches = raw.get("launch", [{"ntasks": 1}])
    if isinstance(launches, dict):
        launches = [launches]
    for index, launch in enumerate(launches):
        _check_keys(f"{path} [[launch]] #{index + 1}", launch, _LAUNCH_KEYS)

    variants = raw.get("variants", {})
    for key, variant in variants.items():
        _check_keys(f"{path} [variants.{key}]", variant, _VARIANT_KEYS)

    store = raw.get("store")
    if store is not None:
        _check_keys(
            f"{path} [store]", store, {"url", "branch", "message", "push_attempts"}
        )

    for name, table in raw.get("clusters", {}).items():
        register_cluster(_preset_from_table(name, table))

    return {
        "path": path,
        "case": case,
        "launch": launches,
        "variants": variants,
        "store": store,
    }


def build_cases(
    config: dict[str, Any],
    variants: list[str] | None = None,
    upload: bool = False,
    scheduler: str | None = None,
) -> list[tuple[ProfilingCase, list[dict[str, Any]]]]:
    """The cases of a loaded config, one per selected variant, each with its launch arguments.

    Nothing is launched yet; :func:`run_case_config` does that.
    """
    available = config["variants"]
    selected = variants or list(available) or [None]
    unknown = [key for key in selected if key is not None and key not in available]
    if unknown:
        raise CaseConfigError(
            f"unknown variant(s) {unknown}; available: {sorted(available)}"
        )

    store = GitResultsStore(**config["store"]) if config["store"] and upload else None
    out = []
    for key in selected:
        variant = available.get(key, {}) if key is not None else {}
        kwargs = dict(config["case"])
        if key is not None:
            kwargs["label"] = f"{kwargs['label']}_{key}"
            kwargs["env"] = {**kwargs.get("env", {}), **variant.get("env", {})}
            kwargs["metadata"] = {
                **kwargs.get("metadata", {}),
                "variant": key,
                **variant.get("metadata", {}),
            }
            if "name" in variant:
                kwargs["name"] = variant["name"]
            if "description" in variant:
                kwargs["description"] = " ".join(
                    filter(None, [kwargs.get("description"), variant["description"]])
                )
        if scheduler is not None:
            kwargs["scheduler"] = scheduler
        case = ProfilingCase(**kwargs, upload=upload, store=store)

        launches = []
        for launch in config["launch"]:
            ntasks = launch.get("ntasks", 1)
            for n in ntasks if isinstance(ntasks, list) else [ntasks]:
                launches.append(
                    {
                        **{k: v for k, v in launch.items() if k != "ntasks"},
                        "ntasks": int(n),
                        "args": [*variant.get("args", []), *launch.get("args", [])],
                    },
                )
        out.append((case, launches))
    return out


def run_case_config(
    path: str | os.PathLike[str],
    variants: list[str] | None = None,
    upload: bool = False,
    scheduler: str | None = None,
    poll_interval: float = 30.0,
    finalize: bool = True,
) -> list[ProfilingCase]:
    """Launch every run of a case TOML file (all variants unless some are selected), then package them.

    With ``finalize=True`` this waits for the runs; inside a sweep the cases
    are only registered, like :meth:`ProfilingCase.finalize` does.
    """
    from scope_profiler.jobs import case as case_module

    cases = []
    for case, launches in build_cases(
        load_case_config(path), variants, upload, scheduler
    ):
        for launch in launches:
            case.launch(**launch)
        cases.append(case)
    if case_module._SWEEP_CASES is not None:
        case_module._SWEEP_CASES.extend(cases)
    elif finalize:
        finalize_all(cases, poll_interval=poll_interval)
    return cases
