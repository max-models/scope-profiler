"""Running job scripts: on this machine, or through SLURM.

Both schedulers take the same job script body (a list of shell lines) and
return a job id that :meth:`iter_finished` yields once the job is over:

- :class:`LocalScheduler` writes a bash script and runs it to completion
  right away, so its jobs are finished by the time they are waited on.
- :class:`SlurmScheduler` renders ``#SBATCH`` pragmas in front of it,
  submits it with ``sbatch --parsable`` and polls ``squeue``/``sacct``.

No third-party SLURM library is needed; the commands are called directly.
"""

from __future__ import annotations

import getpass
import os
import shutil
import subprocess
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

LOCAL_ENV_VAR = "SCOPE_PROFILER_JOBS_LOCAL"

# SLURM job states in which a job still holds or waits for resources.
_ACTIVE_STATES = {
    "BOOT_FAIL_PENDING",
    "CONFIGURING",
    "COMPLETING",
    "PENDING",
    "REQUEUED",
    "RESIZING",
    "RUNNING",
    "SIGNALING",
    "STAGE_OUT",
    "SUSPENDED",
}


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def has_module_system() -> bool:
    """True on a machine with environment modules (Lmod / Tcl modules)."""
    return bool(
        os.environ.get("MODULESHOME")
        or os.environ.get("LMOD_CMD")
        or shutil.which("modulecmd")
    )


def detect_launcher(use_slurm: bool = True) -> str | None:
    """The MPI launcher: ``srun`` under SLURM, otherwise ``mpirun``/``mpiexec``; None if there is none.

    With ``use_slurm=False`` ``srun`` is never chosen, even where it exists:
    on a login node it would request a new allocation instead of running on
    this machine.
    """
    if use_slurm and shutil.which("srun"):
        return "srun"
    for launcher in ("mpirun", "mpiexec"):
        if shutil.which(launcher):
            return launcher
    return None


def _sbatch_key(key: str) -> str:
    return key.replace("_", "-")


def render_sbatch_script(pragmas: dict[str, Any], body: Iterable[str]) -> str:
    """A batch script: ``#SBATCH --key=value`` lines followed by ``body``.

    A value of ``True`` gives a bare flag (``--exclusive``); ``None`` and
    ``False`` leave the option out.
    """
    lines = ["#!/bin/bash"]
    for key, value in pragmas.items():
        if value is None or value is False:
            continue
        option = f"--{_sbatch_key(key)}"
        lines.append(
            f"#SBATCH {option}" if value is True else f"#SBATCH {option}={value}"
        )
    lines.append("")
    lines.extend(body)
    lines.append("")
    return "\n".join(lines)


class LocalScheduler:
    """Runs each job script to completion on this machine, one after the other."""

    name = "local"

    def submit(
        self, script_path: Path, body: list[str], pragmas: dict[str, Any] | None = None
    ) -> tuple[None, str]:
        """Write ``body`` as a bash script, run it, and return ``(None, script_text)``.

        ``pragmas`` are ignored. A non-zero exit is reported, not raised: the
        run is still packaged with whatever output it left behind.
        """
        text = "\n".join(["#!/bin/bash", *body, ""])
        script_path.write_text(text, encoding="utf-8")
        script_path.chmod(0o755)
        result = subprocess.run(["bash", str(script_path)], check=False)
        if result.returncode != 0:
            print(f"WARNING: {script_path} exited with code {result.returncode}.")
        self.last_returncode = result.returncode
        return None, text

    def iter_finished(
        self, job_ids: Iterable[Any], poll_interval: float = 0.0
    ) -> Iterator[tuple[Any, str | None]]:
        """Local jobs are finished once submitted: yield each with an unknown state."""
        for job_id in job_ids:
            yield job_id, None


class SlurmScheduler:
    """Submits job scripts with ``sbatch`` and polls the queue for finished ones."""

    name = "slurm"

    def __init__(self, user: str | None = None):
        self.user = user or getpass.getuser()

    def submit(
        self, script_path: Path, body: list[str], pragmas: dict[str, Any] | None = None
    ) -> tuple[str, str]:
        """Render and write the script, submit it, and return ``(job_id, script_text)``."""
        text = render_sbatch_script(pragmas or {}, body)
        script_path.write_text(text, encoding="utf-8")
        script_path.chmod(0o755)
        result = subprocess.run(
            ["sbatch", "--parsable", str(script_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"sbatch {script_path} failed ({result.returncode}): {result.stderr.strip()}"
            )
        # `--parsable` prints "<id>" or "<id>;<cluster>".
        return result.stdout.strip().split(";")[0], text

    def active_jobs(self) -> set[str]:
        """Ids of this user's jobs that are still in the queue (pending or running)."""
        result = subprocess.run(
            ["squeue", "-h", "-u", self.user, "-o", "%i %T"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"squeue failed ({result.returncode}): {result.stderr.strip()}"
            )
        active = set()
        for line in result.stdout.splitlines():
            parts = line.split()
            if parts and (len(parts) == 1 or parts[1] in _ACTIVE_STATES):
                active.add(parts[0])
        return active

    @staticmethod
    def job_states(job_ids: Iterable[str]) -> dict[str, str | None]:
        """Final accounting state of each job (``COMPLETED``, ``FAILED``, ...), None where unknown."""
        job_ids = list(job_ids)
        states: dict[str, str | None] = dict.fromkeys(job_ids)
        if not job_ids or shutil.which("sacct") is None:
            return states
        result = subprocess.run(
            ["sacct", "-n", "-X", "-P", "-o", "JobID,State", "-j", ",".join(job_ids)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return states
        for line in result.stdout.splitlines():
            job_id, _, state = line.partition("|")
            if job_id in states and state:
                # "CANCELLED by 1234" -> "CANCELLED"
                states[job_id] = state.split()[0]
        return states

    def iter_finished(
        self, job_ids: Iterable[str], poll_interval: float = 10.0
    ) -> Iterator[tuple[str, str | None]]:
        """Yield ``(job_id, state)`` for each job as soon as it leaves the queue, in completion order.

        Leaving the queue only means the job is over, not that it succeeded:
        ``state`` is its accounting state, or None if ``sacct`` cannot tell.
        """
        pending = list(dict.fromkeys(job_ids))
        while pending:
            active = self.active_jobs()
            finished = [job_id for job_id in pending if job_id not in active]
            states = self.job_states(finished)
            for job_id in finished:
                pending.remove(job_id)
                yield job_id, states.get(job_id)
            if pending:
                time.sleep(poll_interval)


class DryRunScheduler:
    """Writes the job scripts (with their ``#SBATCH`` pragmas) without running anything."""

    name = "dry-run"

    def submit(
        self, script_path: Path, body: list[str], pragmas: dict[str, Any] | None = None
    ) -> tuple[None, str]:
        """Write the script and return ``(None, script_text)``."""
        text = render_sbatch_script(pragmas or {}, body)
        script_path.write_text(text, encoding="utf-8")
        script_path.chmod(0o755)
        return None, text

    def iter_finished(
        self, job_ids: Iterable[Any], poll_interval: float = 0.0
    ) -> Iterator[tuple[Any, str | None]]:
        """Nothing ran, so nothing finishes."""
        return iter(())


def detect_scheduler(
    scheduler: str | None = None,
) -> LocalScheduler | SlurmScheduler | DryRunScheduler:
    """The scheduler to use: ``"local"``, ``"slurm"``, ``"dry-run"``, or (None) SLURM where ``sbatch`` exists.

    ``SCOPE_PROFILER_JOBS_LOCAL=1`` forces local runs even where SLURM is
    available (e.g. a quick check on a login node).
    """
    if scheduler is None:
        if _truthy(os.environ.get(LOCAL_ENV_VAR)) or shutil.which("sbatch") is None:
            scheduler = "local"
        else:
            scheduler = "slurm"
    if scheduler == "local":
        return LocalScheduler()
    if scheduler == "slurm":
        return SlurmScheduler()
    if scheduler == "dry-run":
        return DryRunScheduler()
    raise ValueError(
        f"scheduler must be 'local', 'slurm', 'dry-run' or None, got {scheduler!r}"
    )
