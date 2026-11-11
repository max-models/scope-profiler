"""Profiling cases: launching a program at several rank counts, then packaging what it produced.

A :class:`ProfilingCase` describes one thing to profile -- a command, the
files it needs, the environment it runs in -- and :meth:`ProfilingCase.launch`
runs it once per call, typically in a loop over rank counts::

    from scope_profiler.jobs import ProfilingCase

    case = ProfilingCase(
        label="heat_strong_scaling",
        name="Heat equation, strong scaling",
        command=["python", "heat.py", "--n", "256", "--profile-out", "{profile}"],
        inputs=["heat.py"],
        env={"OMP_NUM_THREADS": "1"},
    )
    for ntasks in (1, 2, 4, 8):
        case.launch(ntasks)
    case.finalize()

Each launch writes a job script and either submits it to SLURM (where
``sbatch`` exists) or runs it on this machine. :meth:`ProfilingCase.finalize`
then waits for the jobs and packages each run as soon as it finishes: its
profiles (``*.h5`` by default), whatever a post-processing command wrote into
``<run_dir>/results``, and a metadata file describing the run. With
``upload=True`` the package is committed and pushed to a git results
repository (:class:`~scope_profiler.jobs.store.GitResultsStore`) after every
finished run.

Placeholders in ``command``, ``postprocess`` and ``cwd`` are replaced per
launch: ``{run_dir}`` (the run's own directory), ``{profile}``
(``{run_dir}/profiling_data.h5``), ``{case_dir}``, ``{ntasks}``, ``{nodes}``,
``{launch_id}`` and ``{label}``; a list element that is exactly ``{args}``
expands to the launch's ``args``. The job script also exports them as
``SCOPE_PROFILER_RUN_DIR``, ``SCOPE_PROFILER_NTASKS``, ... for programs that
prefer to read the environment.

Directory layout under ``work_dir`` (``_profiling_jobs`` by default)::

    runs/<timestamp>-<commit>/<label>/      one case: job scripts, inputs
        run_01/                             one launch: output, logs
    packaged/                               (results/ clone with upload=True)
        <timestamp>-<commit>-<label>/
            case_metadata.json
            inputs/...
            run01/run_metadata.json, *.h5, results files
"""

from __future__ import annotations

import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scope_profiler.jobs.clusters import ClusterPreset, detect_cluster, get_cluster
from scope_profiler.jobs.schedulers import (
    DryRunScheduler,
    SlurmScheduler,
    detect_launcher,
    detect_scheduler,
)
from scope_profiler.jobs.store import GitResultsStore

DEFAULT_WORK_DIR = "_profiling_jobs"
PROFILE_FILE_NAME = "profiling_data.h5"
RUN_METADATA_FILE = "run_metadata.json"
DEFAULT_RESULTS_SUFFIXES = (
    ".png",
    ".svg",
    ".pdf",
    ".npy",
    ".html",
    ".json",
    ".csv",
    ".txt",
)

_PLACEHOLDER = re.compile(
    r"\{(run_dir|profile|case_dir|ntasks|nodes|launch_id|label)\}"
)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "case"


def _unique_dir(path: Path) -> Path:
    candidate, suffix = path, 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}-{suffix}")
        suffix += 1
    return candidate


def _git_output(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _git_info(repo: Path | None) -> dict[str, Any]:
    """Commit, branch and dirty flag of the git checkout at ``repo`` (all None outside one)."""
    if repo is None:
        return {
            "git_repository": None,
            "git_commit": None,
            "git_branch": None,
            "git_dirty": None,
        }
    status = _git_output(repo, "status", "--porcelain", "--untracked-files=no")
    return {
        "git_repository": _git_output(repo, "rev-parse", "--show-toplevel"),
        "git_commit": _git_output(repo, "rev-parse", "HEAD"),
        "git_branch": _git_output(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": None if status is None else bool(status),
    }


def _pip_freeze() -> str | None:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _substitute(text: str, values: dict[str, str]) -> str:
    """Replace the known ``{placeholders}``; any other braces are left alone."""
    return _PLACEHOLDER.sub(lambda match: values[match.group(1)], text)


def _render_command(
    command: Sequence[str] | str,
    values: dict[str, str],
    args: Sequence[str],
    append_args: bool = True,
) -> str:
    """A shell command line from ``command`` with placeholders and ``args`` filled in.

    A string is used as written (only placeholders are replaced), so it may
    contain pipes or redirections; ``{args}`` in it is replaced by the
    arguments. List elements are quoted one by one, and an element that is
    exactly ``{args}`` is replaced by the arguments. Without ``{args}`` they
    are appended at the end if ``append_args``, and dropped otherwise.
    """
    quoted_args = [shlex.quote(str(arg)) for arg in args]
    if isinstance(command, str):
        line = _substitute(command, values)
        if "{args}" in line:
            return line.replace("{args}", " ".join(quoted_args))
        return " ".join([line, *quoted_args]) if append_args else line
    parts: list[str] = []
    spliced = False
    for element in command:
        if element == "{args}":
            parts.extend(quoted_args)
            spliced = True
        else:
            parts.append(shlex.quote(_substitute(str(element), values)))
    if not spliced and append_args:
        parts.extend(quoted_args)
    return " ".join(parts)


@dataclass
class ProfilingCase:
    """A program to profile at one or more rank counts, and how to package its output.

    Parameters
    ----------
    label : str
        Short identifier, used in directory, job and package names.
    command : list[str] or str
        The program to run, e.g. ``["python", "app.py", "--out", "{run_dir}"]``.
        It runs in ``cwd`` with stdout/stderr in ``<run_dir>/run.out``, and is
        prefixed by the MPI launcher (``srun -n N`` / ``mpirun -n N``) when
        ``mpi`` is set. See the module docstring for the placeholders.
    name, description : str
        Human-readable title and description, recorded in the case metadata.
    metadata : dict
        Anything else worth recording about the case (code version, problem
        size, ...), stored as-is in ``case_metadata.json``. Must be JSON
        serialisable.
    inputs : list of paths
        Files copied into the case directory before the first launch and
        packaged under ``inputs/`` (parameter files, scripts).
    postprocess : list[str] or str or None
        Command run (serially, without the launcher) after each run, e.g.
        ``["python", "pproc.py", "{run_dir}", "{args}"]``; it gets the
        launch's ``args`` only where it says ``{args}``. Files it writes
        into ``<run_dir>/results`` are packaged with the run.
    setup : list[str]
        Shell commands run at the top of every job script, after the cluster
        preset's own ones (e.g. ``module load``, ``source env.sh``).
    env : dict[str, str]
        Environment variables exported in every job script.
    activate_venv : bool
        Activate the virtual environment of this process (``$VIRTUAL_ENV``)
        in the job scripts, so jobs use the same interpreter and packages.
    mpi : bool
        Run ``command`` through the MPI launcher.
    launcher : str or None
        The launcher prefix, e.g. ``"mpirun --oversubscribe"``; detected
        (``srun`` under SLURM, else ``mpirun``/``mpiexec``) if None. The rank
        count is passed as ``-n <ntasks>``.
    cwd : str
        Working directory of the run (placeholders allowed).
    collect : tuple[str]
        Glob patterns, relative to the run directory, of the profiles to
        package; their relative paths are kept.
    results_dir : str
        Folder of the run directory whose files (filtered by
        ``results_suffixes``) are packaged next to the profiles.
    results_suffixes : tuple[str]
        File types of ``results_dir`` that are packaged.
    work_dir : path
        Where runs and packages are written.
    repo : path or None
        Git checkout whose commit, branch and dirty state are recorded (and
        used in folder names); defaults to the current directory if that is
        inside a git repository.
    scheduler : {"local", "slurm", "dry-run"} or None
        How jobs run; SLURM where ``sbatch`` exists, otherwise locally, if
        None (``SCOPE_PROFILER_JOBS_LOCAL=1`` forces local runs).
        ``"dry-run"`` only writes the job scripts.
    cluster : str or ClusterPreset or None
        Preset for SLURM jobs (``#SBATCH`` pragmas, setup, node hardware);
        the detected one if None. See :mod:`scope_profiler.jobs.clusters`.
    upload : bool
        Push the package to ``store`` after every finished run. The results
        repository is cloned up front, so an unreachable one fails before
        any job is submitted.
    store : GitResultsStore or None
        Results repository; from ``SCOPE_PROFILER_RESULTS_REPO`` if None.
    record_pip_freeze : bool
        Record ``pip freeze`` of this interpreter in the case metadata.
    """

    label: str
    command: list[str] | str
    name: str = ""
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    inputs: list[str | os.PathLike[str]] = field(default_factory=list)
    postprocess: list[str] | str | None = None
    setup: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    activate_venv: bool = True
    mpi: bool = True
    launcher: str | None = None
    cwd: str = "{run_dir}"
    collect: tuple[str, ...] = ("**/*.h5",)
    results_dir: str = "results"
    results_suffixes: tuple[str, ...] = DEFAULT_RESULTS_SUFFIXES
    work_dir: str | os.PathLike[str] = DEFAULT_WORK_DIR
    repo: str | os.PathLike[str] | None = None
    scheduler: str | None = None
    cluster: str | ClusterPreset | None = None
    upload: bool = False
    store: GitResultsStore | None = None
    record_pip_freeze: bool = True

    # Set up by __post_init__ / launch / finalize; not constructor arguments.
    timestamp: datetime = field(init=False)
    case_dir: Path = field(init=False)
    packaged_root: Path = field(init=False)
    destination_dir: Path | None = field(init=False, default=None)
    job_infos: list[dict[str, Any]] = field(init=False, default_factory=list)
    packaged_runs: list[dict[str, Any]] = field(init=False, default_factory=list)
    launch_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.work_dir = Path(self.work_dir).resolve()
        self.inputs = [Path(path).resolve() for path in self.inputs]
        for path in self.inputs:
            if not path.is_file():
                raise FileNotFoundError(
                    f"input file {path} of case {self.label!r} does not exist"
                )

        if self.repo is None and _git_output(
            Path.cwd(), "rev-parse", "--show-toplevel"
        ):
            self.repo = Path.cwd()
        self.repo = Path(self.repo).resolve() if self.repo is not None else None
        self.git = _git_info(self.repo)

        self._scheduler = detect_scheduler(self.scheduler)
        self.scheduler = self._scheduler.name
        if isinstance(self.cluster, str):
            self.cluster = get_cluster(self.cluster)
        elif self.cluster is None and isinstance(
            self._scheduler, SlurmScheduler | DryRunScheduler
        ):
            self.cluster = detect_cluster()

        if self.mpi and self.launcher is None:
            self.launcher = detect_launcher(use_slurm=self.scheduler != "local")
            if self.launcher is None and self.scheduler != "local":
                # Batch scripts run on the compute nodes, where srun exists.
                self.launcher = "srun"

        self.timestamp = datetime.now(timezone.utc)
        commit = (self.git["git_commit"] or "")[:8]
        token = "-".join(
            filter(None, [self.timestamp.strftime("%Y%m%dT%H%M%SZ"), commit])
        )
        self.case_dir = _unique_dir(self.work_dir / "runs" / token / _slug(self.label))
        self.case_dir.mkdir(parents=True)
        self._package_name = f"{token}-{_slug(self.label)}"
        for path in self.inputs:
            shutil.copy2(path, self.case_dir / path.name)

        if self.upload:
            self.store = self.store or GitResultsStore.from_env()
            self.packaged_root = self.work_dir / "results"
            if self.packaged_root not in _PREPARED_ROOTS:
                self.store.clone(self.packaged_root)
        else:
            self.packaged_root = self.work_dir / "packaged"
            self.packaged_root.mkdir(parents=True, exist_ok=True)
        _PREPARED_ROOTS.add(self.packaged_root)
        print(f"Profiling case {self.label!r}: {self.case_dir} ({self.scheduler})")

    # ------------------------------------------------------------------
    # Launching
    # ------------------------------------------------------------------
    def _placeholders(
        self, run_dir: Path, ntasks: int, nodes: int, launch_id: int
    ) -> dict[str, str]:
        return {
            "run_dir": str(run_dir),
            "profile": str(run_dir / PROFILE_FILE_NAME),
            "case_dir": str(self.case_dir),
            "ntasks": str(ntasks),
            "nodes": str(nodes),
            "launch_id": str(launch_id),
            "label": self.label,
        }

    def build_commands(
        self,
        ntasks: int,
        run_dir: Path,
        launch_id: int,
        nodes: int = 1,
        args: Sequence[str] = (),
        env: dict[str, str] | None = None,
    ) -> list[str]:
        """The body of the job script of one launch (without ``#SBATCH`` pragmas)."""
        values = self._placeholders(run_dir, ntasks, nodes, launch_id)
        main = _render_command(self.command, values, args)
        if self.mpi:
            if self.launcher:
                main = f"{self.launcher} -n {ntasks} {main}"
            elif ntasks > 1:
                raise RuntimeError(
                    f"no MPI launcher found to run {self.label!r} on {ntasks} ranks; "
                    "install MPI or pass launcher=...",
                )

        exports = {
            **self.env,
            **(env or {}),
            "SCOPE_PROFILER_RUN_DIR": values["run_dir"],
            "SCOPE_PROFILER_PROFILE": values["profile"],
            "SCOPE_PROFILER_CASE_DIR": values["case_dir"],
            "SCOPE_PROFILER_NTASKS": values["ntasks"],
            "SCOPE_PROFILER_LAUNCH_ID": values["launch_id"],
            "SCOPE_PROFILER_LABEL": self.label,
        }
        cluster_setup = (
            self.cluster.setup if isinstance(self.cluster, ClusterPreset) else []
        )
        venv = os.environ.get("VIRTUAL_ENV")
        activate = (
            [f"source {shlex.quote(str(Path(venv) / 'bin' / 'activate'))}"]
            if self.activate_venv
            and venv
            and (Path(venv) / "bin" / "activate").exists()
            else []
        )
        run_out = shlex.quote(str(run_dir / "run.out"))
        lines = [
            # Module systems are not always clean under `set -e`, so the
            # cluster's own setup runs first.
            *cluster_setup,
            "set -e",
            *activate,
            *self.setup,
            *(
                f"export {key}={shlex.quote(str(value))}"
                for key, value in exports.items()
            ),
            'echo "----------------------------------------"',
            f"echo {shlex.quote(f'Profiling case {self.label}, run {launch_id:02d} ({ntasks} ranks)')}",
            f"mkdir -p {shlex.quote(str(run_dir))}",
            f"cd {shlex.quote(_substitute(self.cwd, values))}",
            f"{main} > {run_out} 2>&1 || {{ status=$?; tail -n 40 {run_out}; exit $status; }}",
        ]
        if self.postprocess is not None:
            pproc_out = shlex.quote(str(run_dir / "postprocess.out"))
            pproc = _render_command(self.postprocess, values, args, append_args=False)
            lines += [
                f"mkdir -p {shlex.quote(str(run_dir / self.results_dir))}",
                f"{pproc} > {pproc_out} 2>&1 || {{ status=$?; tail -n 40 {pproc_out}; exit $status; }}",
            ]
        lines += [f"echo {shlex.quote(f'Finished {self.label}, run {launch_id:02d}')}"]
        return lines

    def launch(
        self,
        ntasks: int = 1,
        nodes: int | None = None,
        args: Sequence[str] = (),
        env: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
        cluster: str | ClusterPreset | None = None,
        sbatch: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Run (locally) or submit (SLURM) the case once, with ``ntasks`` ranks.

        Parameters
        ----------
        ntasks : int
            Number of MPI ranks (tasks).
        nodes : int or None
            SLURM nodes to spread the ranks over; derived from the preset's
            ``cpus_per_node`` if None. Ignored for local runs.
        args : sequence of str
            Extra arguments of this launch, appended to ``command`` (or put
            where it has ``{args}``), e.g. a variant's flags.
        env : dict or None
            Environment variables of this launch only, e.g.
            ``{"CUNUMPY_BACKEND": "cupy"}``.
        metadata : dict or None
            Recorded in this run's metadata, e.g. ``{"backend": "gpu"}``.
        cluster : str or ClusterPreset or None
            Preset of this launch (e.g. a GPU partition), overriding the case's.
        sbatch : dict or None
            ``#SBATCH`` pragmas overriding those of the preset, e.g.
            ``{"time": "00:30:00"}``.

        Returns
        -------
        dict or None
            The run's record (also appended to ``job_infos``), or None if a
            local run was skipped because ``ntasks`` exceeds the local cores.
        """
        if self.scheduler == "local" and self.mpi and ntasks > (os.cpu_count() or 1):
            print(
                f"Skipping {self.label!r} with {ntasks} ranks: exceeds the {os.cpu_count()} local cores."
            )
            return None
        if self.scheduler == "local" and self.mpi and not self.launcher and ntasks > 1:
            print(
                f"Skipping {self.label!r} with {ntasks} ranks: no MPI launcher (mpirun/mpiexec) found."
            )
            return None

        preset = (
            get_cluster(cluster)
            if isinstance(cluster, str)
            else cluster or self.cluster
        )
        if self.scheduler == "local":
            nodes = 1
        elif nodes is None:
            nodes = preset.nodes_for(ntasks) if isinstance(preset, ClusterPreset) else 1
        if ntasks % nodes != 0:
            raise ValueError(f"ntasks ({ntasks}) is not divisible by nodes ({nodes})")

        self.launch_count += 1
        launch_id = self.launch_count
        run_dir = self.case_dir / f"run_{launch_id:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        script_path = self.case_dir / f"job_{launch_id:02d}.sh"
        body = self.build_commands(
            ntasks, run_dir, launch_id, nodes=nodes, args=args, env=env
        )

        pragmas: dict[str, Any] = {}
        if self.scheduler != "local":
            pragmas = {
                "output": str(run_dir / "slurm-%j.out"),
                "error": str(run_dir / "slurm-%j.err"),
                **(preset.sbatch if isinstance(preset, ClusterPreset) else {}),
                **(sbatch or {}),
                "job_name": f"{_slug(self.label)}_{launch_id:02d}",
                "nodes": nodes,
                "ntasks_per_node": ntasks // nodes,
            }

        if self.scheduler == "local":
            print(
                f"Running {self.label!r} with {ntasks} ranks locally ({script_path}) ..."
            )
        job_id, script_text = self._scheduler.submit(script_path, body, pragmas)
        if self.scheduler == "slurm":
            print(f"Submitted {self.label!r} with {ntasks} ranks as job {job_id}.")
        elif self.scheduler == "dry-run":
            print(f"Wrote {script_path} (dry run, not submitted).")

        info = {
            "launch_id": launch_id,
            "ntasks": ntasks,
            "nodes": nodes,
            "args": [str(arg) for arg in args],
            "env": dict(env or {}),
            "metadata": dict(metadata or {}),
            "cluster": preset.name if isinstance(preset, ClusterPreset) else None,
            "scheduler": self.scheduler,
            "job_id": job_id,
            "sbatch": pragmas or None,
            "job_script_path": str(script_path),
            "job_script": script_text,
            "run_directory": str(run_dir),
            "state": None,
        }
        if self.scheduler == "local":
            returncode = getattr(self._scheduler, "last_returncode", None)
            info["returncode"] = returncode
            info["state"] = "COMPLETED" if returncode == 0 else "FAILED"
        self.job_infos.append(info)
        return info

    # ------------------------------------------------------------------
    # Packaging
    # ------------------------------------------------------------------
    def case_metadata(self) -> dict[str, Any]:
        """The contents of ``case_metadata.json``: the case, machine, software and packaged runs."""
        if not hasattr(self, "_software"):
            # Collected once: `pip freeze` is slow and does not change during a run.
            self._software = {
                **self.git,
                "python_version": sys.version,
                "python_executable": sys.executable,
                "scope_profiler_version": _scope_profiler_version(),
                "pip_freeze": _pip_freeze() if self.record_pip_freeze else None,
            }
        cluster = self.cluster if isinstance(self.cluster, ClusterPreset) else None
        return {
            "general_information": {
                "label": self.label,
                "name": self.name or self.label,
                "description": self.description,
                "time_date_utc": self.timestamp.isoformat(),
                "datetime_token": self.timestamp.strftime("%Y%m%dT%H%M%SZ"),
                "case_directory": str(self.case_dir),
            },
            "metadata": self.metadata,
            "command": self.command,
            "postprocess": self.postprocess,
            "setup": self.setup,
            "env": self.env,
            "inputs": [f"inputs/{path.name}" for path in self.inputs],
            "scheduler": self.scheduler,
            "hardware_information": {
                "cluster": cluster.to_dict() if cluster else None,
                "submit_host": socket.getfqdn(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "cpu_count": os.cpu_count(),
                "slurm_cluster_name": os.environ.get("SLURM_CLUSTER_NAME"),
            },
            "software_information": self._software,
            "runs": self.packaged_runs,
        }

    def package_case_metadata(self) -> Path:
        """Create the packaged folder and (re)write its inputs and ``case_metadata.json``."""
        if self.destination_dir is None:
            self.destination_dir = self.packaged_root / self._package_name
        self.destination_dir.mkdir(parents=True, exist_ok=True)
        if self.inputs:
            (self.destination_dir / "inputs").mkdir(exist_ok=True)
            for path in self.inputs:
                shutil.copy2(
                    self.case_dir / path.name,
                    self.destination_dir / "inputs" / path.name,
                )
        (self.destination_dir / "case_metadata.json").write_text(
            json.dumps(self.case_metadata(), indent=2, default=str),
            encoding="utf-8",
        )
        return self.destination_dir

    def package_run(self, info: dict[str, Any]) -> bool:
        """Copy one finished run's profiles and results into the packaged folder.

        Writes ``run<id>/run_metadata.json`` (the launch record, plus the
        ``run_metadata.json`` the program itself left in its run directory,
        if any, under ``application_metadata``) and refreshes
        ``case_metadata.json``. A run that produced neither profiles nor
        results (it never started, or failed early) is not packaged.

        Returns
        -------
        bool
            Whether anything was packaged.
        """
        if self.destination_dir is None:
            self.package_case_metadata()
        run_dir = Path(info["run_directory"])
        destination = self.destination_dir / f"run{info['launch_id']:02d}"

        profiles = sorted(
            {
                path
                for pattern in self.collect
                for path in run_dir.glob(pattern)
                if path.is_file()
            },
        )
        results_root = run_dir / self.results_dir
        results = (
            sorted(
                path
                for path in results_root.rglob("*")
                if path.is_file() and path.suffix.lower() in self.results_suffixes
            )
            if results_root.is_dir()
            else []
        )
        results = [path for path in results if path not in profiles]
        if not profiles and not results:
            return False

        packaged_profiles = []
        for source in profiles:
            target = destination / source.relative_to(run_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            packaged_profiles.append(
                target.relative_to(self.destination_dir).as_posix()
            )
        packaged_results = []
        for source in results:
            target = destination / source.relative_to(results_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            packaged_results.append(target.relative_to(self.destination_dir).as_posix())

        application = None
        app_metadata = run_dir / RUN_METADATA_FILE
        if app_metadata.is_file():
            try:
                application = json.loads(app_metadata.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                application = None
        run_metadata = {
            **info,
            "label": self.label,
            "packaged_files": {
                "profiles": packaged_profiles,
                "results": packaged_results,
            },
            "application_metadata": application,
        }
        destination.mkdir(parents=True, exist_ok=True)
        (destination / RUN_METADATA_FILE).write_text(
            json.dumps(run_metadata, indent=2, default=str),
            encoding="utf-8",
        )
        self.packaged_runs.append(
            {
                "launch_id": info["launch_id"],
                "ntasks": info["ntasks"],
                "folder": destination.name,
                "run_metadata": f"{destination.name}/{RUN_METADATA_FILE}",
                "state": info.get("state"),
            },
        )
        self.package_case_metadata()
        return True

    def push(self) -> None:
        """Commit and push the packaged folder to the results repository (``upload=True`` only)."""
        if self.upload and self.store is not None:
            self.store.push(
                self.packaged_root,
                label=self.label,
                commit=self.git["git_commit"] or "",
            )

    def finalize(self, poll_interval: float = 10.0) -> None:
        """Wait for the runs and package (and with ``upload`` push) each one as it finishes.

        The case metadata is packaged (and pushed) first, so the folder exists
        in the results repository while the jobs are still queued. Inside a
        sweep (:func:`begin_sweep`) this only registers the case; the sweep's
        :func:`finalize_all` then waits for the runs of all cases together.
        """
        if _SWEEP_CASES is not None:
            _SWEEP_CASES.append(self)
            print(
                f"Registered {len(self.job_infos)} run(s) of {self.label!r}; packaging is left to the sweep."
            )
            return
        finalize_all([self], poll_interval=poll_interval)

    def report(self) -> None:
        """Print what was packaged, and how to push or inspect it."""
        if not self.packaged_runs:
            print(
                f"No profiling output found for {self.label!r}; nothing was packaged."
            )
            return
        ids = [run["launch_id"] for run in self.packaged_runs]
        print(
            f"Packaged {len(ids)} run(s) of {self.label!r} (ids {sorted(ids)}) into {self.destination_dir}"
        )
        if not self.upload:
            print("Push it to a results repository later, as packaged, with:")
            print(
                f"    scope-profiler jobs upload {shlex.quote(str(self.destination_dir))} --repo <url>"
            )
        first = self.destination_dir / self.packaged_runs[0]["folder"]
        print("Inspect a run with, e.g.:")
        print(f"    scope-profiler inspect {shlex.quote(str(first))}/*.h5 --sort total")


def _scope_profiler_version() -> str | None:
    try:
        from scope_profiler import __version__
    except ImportError:  # pragma: no cover - only in a broken install
        return None
    return __version__


# ---------------------------------------------------------------------------
# Several cases at once
# ---------------------------------------------------------------------------
# Cases whose `finalize` was deferred by a sweep, or None outside one.
_SWEEP_CASES: list[ProfilingCase] | None = None
# Packaging folders (and results clones) prepared in this process, so that the
# cases of one sweep share a clone instead of each re-cloning (and so dropping
# what the others packaged).
_PREPARED_ROOTS: set[Path] = set()


def begin_sweep() -> None:
    """Start collecting cases: from now on :meth:`ProfilingCase.finalize` only registers its case.

    All cases of a sweep submit their jobs right away (SLURM runs them as the
    partitions allow), and :func:`finalize_all` packages their runs one at a
    time as they finish, from one process and with one results clone.
    """
    global _SWEEP_CASES
    _SWEEP_CASES = []


def end_sweep() -> list[ProfilingCase]:
    """Stop collecting cases and return the ones registered since :func:`begin_sweep`."""
    global _SWEEP_CASES
    cases, _SWEEP_CASES = _SWEEP_CASES or [], None
    return cases


def _iter_finished(
    cases: Sequence[ProfilingCase],
    poll_interval: float,
) -> Iterator[tuple[ProfilingCase, dict[str, Any]]]:
    """Yield ``(case, run)`` for every run of ``cases`` as soon as it is over, in completion order.

    Runs that already finished (local runs) come first; one poll of the SLURM
    queue then serves the jobs of all cases.
    """
    pending: dict[Any, tuple[ProfilingCase, dict[str, Any]]] = {}
    slurm: SlurmScheduler | None = None
    for case in cases:
        for info in case.job_infos:
            if info.get("job_id") is None:
                if info["scheduler"] != "dry-run":
                    yield case, info
            else:
                pending[info["job_id"]] = (case, info)
                if slurm is None and isinstance(case._scheduler, SlurmScheduler):
                    slurm = case._scheduler
    if not pending:
        return
    for job_id, state in (slurm or SlurmScheduler()).iter_finished(
        list(pending), poll_interval=poll_interval
    ):
        case, info = pending.pop(job_id)
        info["state"] = state
        if state is not None and state != "COMPLETED":
            print(
                f"WARNING: job {job_id} of {case.label!r} ({info['ntasks']} ranks) ended in state {state}."
            )
        else:
            print(f"Job {job_id} of {case.label!r} ({info['ntasks']} ranks) finished.")
        if pending:
            print(f"{len(pending)} run(s) still pending ...", flush=True)
        yield case, info


def finalize_all(cases: Sequence[ProfilingCase], poll_interval: float = 30.0) -> None:
    """:meth:`ProfilingCase.finalize` for several cases: package (and push) every run as it finishes.

    The case metadata of every case is packaged (and pushed) first, then each
    finished run, one at a time, in the order the jobs finish.
    """
    if not cases:
        print("No cases to finalize.")
        return
    for case in cases:
        case.package_case_metadata()
    _push_all(cases)

    n_runs = sum(len(case.job_infos) for case in cases)
    print(f"Waiting for {n_runs} run(s) of {len(cases)} case(s) ...")
    for case, info in _iter_finished(cases, poll_interval):
        if case.package_run(info):
            print(
                f"Packaged {case.label!r} run {info['launch_id']:02d} ({info['ntasks']} ranks)."
            )
            _push_all([case])
        else:
            print(
                f"No profiling output for {case.label!r} run {info['launch_id']:02d}; nothing to package."
            )

    for case in cases:
        case.report()


def _push_all(cases: Sequence[ProfilingCase]) -> None:
    """Push each results clone once, whichever of ``cases`` packaged into it."""
    seen: set[Path] = set()
    for case in cases:
        if case.upload and case.packaged_root not in seen:
            seen.add(case.packaged_root)
            case.push()
