"""Running many profiling cases as one sweep, from one process.

Started as separate processes, several case scripts would each prepare the
shared packaging folder (or results clone) and push to it concurrently. A
sweep runs them all in this process instead: every script submits its jobs
right away and only registers its case (:func:`~scope_profiler.jobs.case.begin_sweep`),
and :func:`~scope_profiler.jobs.case.finalize_all` then packages (and with
``upload`` pushes) every run as its job finishes::

    from scope_profiler.jobs import run_sweep, variant_entries

    entries = variant_entries("cases/submit_poisson.py") + [("cases/heat.toml", [])]
    n_failed = run_sweep(entries)

An entry is a Python script (run as ``__main__`` with the given arguments,
which builds :class:`~scope_profiler.jobs.case.ProfilingCase` objects and
calls ``finalize``) or a case TOML file (:mod:`scope_profiler.jobs.config`,
arguments ``--variant <key>`` select variants).
"""

from __future__ import annotations

import importlib.util
import os
import runpy
import shlex
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

from scope_profiler.jobs import case as case_module
from scope_profiler.jobs.config import load_case_config, run_case_config


def _load_module(script: Path):
    """Import a script as a module without running its ``__main__`` block."""
    name = f"_scope_profiler_sweep_{script.stem}_{abs(hash(str(script)))}"
    spec = importlib.util.spec_from_file_location(name, script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def variant_entries(
    script: str | os.PathLike[str], args: Sequence[str] = ()
) -> list[tuple[Path, list[str]]]:
    """One sweep entry per variant of ``script``, with ``--variant <key>``.

    The variants are the keys of a Python script's module-level ``VARIANTS``
    dict, or the ``[variants]`` of a case TOML file. A script without
    variants gives one entry. ``args`` are added to every entry.
    """
    script = Path(script).resolve()
    if script.suffix == ".toml":
        keys = list(load_case_config(script)["variants"])
    else:
        keys = list(getattr(_load_module(script), "VARIANTS", None) or {})
    if not keys:
        return [(script, list(args))]
    return [(script, ["--variant", key, *args]) for key in keys]


def _command(script: Path, args: Sequence[str]) -> str:
    return shlex.join([str(script), *args])


def _parse_toml_args(args: Sequence[str]) -> dict:
    variants, upload, scheduler = [], False, None
    it = iter(args)
    for arg in it:
        if arg == "--variant":
            variants.append(next(it))
        elif arg == "--upload":
            upload = True
        elif arg == "--scheduler":
            scheduler = next(it)
        else:
            raise ValueError(f"unsupported argument {arg!r} for a case TOML file")
    return {"variants": variants or None, "upload": upload, "scheduler": scheduler}


def run_sweep(
    entries: Sequence[tuple[str | os.PathLike[str], Sequence[str]]],
    upload: bool = False,
    dry_run: bool = False,
    poll_interval: float = 30.0,
) -> int:
    """Run every entry in this process, then package (and push) all runs as they finish.

    Parameters
    ----------
    entries : sequence of (path, arguments)
        Python scripts or case TOML files, each with its arguments, e.g.
        from :func:`variant_entries`.
    upload : bool
        Add ``--upload`` to every entry (the scripts must accept it and pass
        it on as ``ProfilingCase(upload=...)``).
    dry_run : bool
        Only print the entries.
    poll_interval : float
        Seconds between polls of the SLURM queue.

    Returns
    -------
    int
        Number of entries that failed.
    """
    entries = [
        (Path(path).resolve(), [*args, *(["--upload"] if upload else [])])
        for path, args in entries
    ]
    print(f"Sweep of {len(entries)} entr{'y' if len(entries) == 1 else 'ies'}:")
    for path, args in entries:
        print(f"  {_command(path, args)}")
    if dry_run:
        return 0

    failed = []
    case_module.begin_sweep()
    try:
        for path, args in entries:
            print(f"\n=== {_command(path, args)}", flush=True)
            argv = sys.argv
            sys.argv = [str(path), *args]
            try:
                if path.suffix == ".toml":
                    run_case_config(
                        path, poll_interval=poll_interval, **_parse_toml_args(args)
                    )
                else:
                    runpy.run_path(str(path), run_name="__main__")
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    failed.append((path, args))
                    print(f"FAILED (exit code {exc.code}): {_command(path, args)}")
            except Exception:
                failed.append((path, args))
                traceback.print_exc()
                print(f"FAILED: {_command(path, args)}")
            finally:
                sys.argv = argv
    finally:
        cases = case_module.end_sweep()

    case_module.finalize_all(cases, poll_interval=poll_interval)

    if failed:
        print(f"\n{len(failed)} of {len(entries)} entries failed:")
        for path, args in failed:
            print(f"  {_command(path, args)}")
    return len(failed)
