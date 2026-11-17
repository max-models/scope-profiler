"""Keeping a results repository in shape: check what it holds, and thin out old executions.

A results repository (see :mod:`scope_profiler.jobs.store`) collects case folders
packaged by :class:`~scope_profiler.jobs.case.ProfilingCase`, one per execution of a
case, usually below a directory per code and case (``mycode/<label>/<package>``).

- :func:`check_results` (``scope-profiler jobs check DIR``) finds what would break a
  catalogue or the repository: metadata that does not parse, runs it lists but that are
  missing, profiles that do not open, files too large for a git host.
- :func:`prune_results` (``scope-profiler jobs prune DIR --keep N``) removes all but the
  newest executions of every case, so a nightly job does not grow the repository forever.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

CASE_METADATA_FILE = "case_metadata.json"
#: GitHub refuses files above 100 MB and warns above 50 MB.
DEFAULT_MAX_BYTES = 50_000_000


@dataclass
class Package:
    """A case folder packaged by ``scope-profiler jobs``: one execution of a case."""

    path: Path
    label: str
    date: datetime | None

    @property
    def case_dir(self) -> Path:
        """The directory that holds all executions of the same case."""
        return self.path.parent


def _read_metadata(folder: Path) -> dict:
    return json.loads((folder / CASE_METADATA_FILE).read_text(encoding="utf-8"))


def _parse_date(value) -> datetime | None:
    if not value:
        return None
    try:
        date = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return date if date.tzinfo else date.replace(tzinfo=timezone.utc)


def find_packages(root: str | Path) -> list[Package]:
    """Every packaged case folder below ``root`` (folders with a ``case_metadata.json``)."""
    packages = []
    for metadata_path in sorted(Path(root).rglob(CASE_METADATA_FILE)):
        if ".git" in metadata_path.parts:
            continue
        folder = metadata_path.parent
        try:
            general = _read_metadata(folder).get("general_information") or {}
        except (OSError, ValueError):
            general = {}
        packages.append(
            Package(
                path=folder,
                label=str(general.get("label") or folder.name),
                date=_parse_date(general.get("time_date_utc")),
            )
        )
    return packages


def check_results(
    root: str | Path, max_bytes: int = DEFAULT_MAX_BYTES, open_profiles: bool = True
) -> list[str]:
    """What is wrong in the results repository at ``root``; an empty list if nothing.

    Checks every packaged case folder: its ``case_metadata.json`` parses, every run it
    lists that produced output has its folder and ``run_metadata.json``, and every
    profile it lists exists. Below ``root`` as a whole, every ``.h5`` file opens (with
    ``open_profiles``) and no file is larger than ``max_bytes``.
    """
    root = Path(root)
    problems: list[str] = []
    for package in find_packages(root):
        where = package.path.relative_to(root)
        try:
            metadata = _read_metadata(package.path)
        except (OSError, ValueError) as exc:
            problems.append(f"{where}/{CASE_METADATA_FILE}: not valid JSON ({exc})")
            continue
        if not isinstance(metadata, dict):
            problems.append(f"{where}/{CASE_METADATA_FILE}: not a JSON object")
            continue
        for run in metadata.get("runs") or []:
            if not isinstance(run, dict) or run.get("failed"):
                continue
            folder = run.get("folder")
            if not folder or not (package.path / folder).is_dir():
                problems.append(
                    f"{where}: run {run.get('launch_id')} is listed but its folder {folder!r} is missing"
                )
                continue
            run_metadata = run.get("run_metadata")
            if run_metadata and not (package.path / run_metadata).is_file():
                problems.append(f"{where}: {run_metadata} is missing")
            for profile in run.get("profiles") or []:
                if not (package.path / profile).is_file():
                    problems.append(f"{where}: profile {profile} is missing")
    for path in sorted(root.rglob("*")):
        if ".git" in path.relative_to(root).parts or not path.is_file():
            continue
        where = path.relative_to(root)
        size = path.stat().st_size
        if size > max_bytes:
            problems.append(
                f"{where}: {size / 1e6:.1f} MB, larger than {max_bytes / 1e6:.0f} MB"
            )
        if open_profiles and path.suffix == ".h5":
            import h5py

            try:
                with h5py.File(path, "r"):
                    pass
            except OSError as exc:
                problems.append(f"{where}: cannot be opened ({exc})")
    return problems


def prune_results(
    root: str | Path,
    keep: int,
    older_than_days: float | None = None,
    apply: bool = False,
    now: datetime | None = None,
) -> list[Path]:
    """Remove all but the ``keep`` newest executions of every case below ``root``.

    The executions of a case are the packaged folders in one directory with the same
    label, ordered by their date. With ``older_than_days`` only executions older than
    that are removed, however many there are. Nothing is deleted unless ``apply`` is
    set; the folders that are (or would be) removed are returned either way. Commit the
    removal in the results repository afterwards.
    """
    if keep < 1:
        raise ValueError("keep must be at least 1: a case keeps its newest execution")
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=older_than_days) if older_than_days else None
    groups: dict[tuple[Path, str], list[Package]] = {}
    for package in find_packages(root):
        groups.setdefault((package.case_dir, package.label), []).append(package)
    removed = []
    for packages in groups.values():
        # Undated folders sort first by name (their timestamp prefix), dated ones by date.
        packages.sort(
            key=lambda p: (
                p.date or datetime.min.replace(tzinfo=timezone.utc),
                p.path.name,
            )
        )
        for package in packages[:-keep]:
            if (
                cutoff is not None
                and package.date is not None
                and package.date > cutoff
            ):
                continue
            removed.append(package.path)
            if apply:
                shutil.rmtree(package.path)
    return removed
