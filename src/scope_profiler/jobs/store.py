"""Pushing packaged profiling results to a git repository.

A results repository collects the packaged case folders of many runs (one
folder per case and run, see :mod:`scope_profiler.jobs.case`), e.g. to feed a
dashboard of performance over time. With a store, a case packages straight
into a clone of that repository, so publishing a finished run is a commit
and a push; a case packaged without one can be pushed later, unchanged, with
:meth:`GitResultsStore.upload` (``scope-profiler jobs upload <folder>``).

The repository URL is given explicitly or read from
``SCOPE_PROFILER_RESULTS_REPO``.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

RESULTS_REPO_ENV_VAR = "SCOPE_PROFILER_RESULTS_REPO"


def _git(*args: str, check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], check=check, **kwargs)


@dataclass
class GitResultsStore:
    """A git repository that packaged case folders are committed and pushed to.

    Parameters
    ----------
    url : str
        Clone URL of the results repository.
    branch : str or None
        Branch to clone and push; the remote's default branch if None.
    message : str
        Commit message template; ``{label}`` and ``{commit}`` (the short
        commit of the profiled code) are filled in.
    push_attempts : int
        Pushes tried (rebasing on the remote in between) before giving up,
        since several runs may push to the same repository concurrently.
    """

    url: str
    branch: str | None = None
    message: str = "Add profiling data for {label} at {commit}"
    push_attempts: int = 5

    @classmethod
    def from_env(cls, **kwargs) -> GitResultsStore:
        """A store for the repository in ``SCOPE_PROFILER_RESULTS_REPO``."""
        url = os.environ.get(RESULTS_REPO_ENV_VAR)
        if not url:
            raise RuntimeError(
                f"no results repository: pass a GitResultsStore(url=...) or set ${RESULTS_REPO_ENV_VAR}",
            )
        return cls(url=url, **kwargs)

    def clone(self, clone_dir: Path) -> Path:
        """Clone the repository (shallow) into ``clone_dir``, replacing whatever is there."""
        clone_dir = Path(clone_dir)
        if clone_dir.exists():
            shutil.rmtree(clone_dir)
        clone_dir.parent.mkdir(parents=True, exist_ok=True)
        branch = ["--branch", self.branch] if self.branch else []
        _git("clone", "--depth", "1", *branch, self.url, str(clone_dir))
        return clone_dir

    def push(self, clone_dir: Path, label: str = "", commit: str = "") -> bool:
        """Commit everything new in ``clone_dir`` and push it; False if there was nothing to push."""
        clone_dir = str(clone_dir)
        _git("-C", clone_dir, "add", ".")
        if (
            _git("-C", clone_dir, "diff", "--cached", "--quiet", check=False).returncode
            == 0
        ):
            print("No changes to push to the results repository.")
            return False
        message = self.message.format(
            label=label or "profiling run", commit=(commit or "unknown")[:8]
        )
        _git("-C", clone_dir, "commit", "-q", "-m", message)

        for attempt in range(self.push_attempts):
            if (
                _git(
                    "-C", clone_dir, "push", "-q", "origin", "HEAD", check=False
                ).returncode
                == 0
            ):
                print(f"Pushed profiling data to {self.url}.")
                return True
            print(
                f"Push attempt {attempt + 1} failed; rebasing on the remote before retrying."
            )
            _git("-C", clone_dir, "fetch", "-q", "origin")
            upstream = f"origin/{self.branch}" if self.branch else "origin/HEAD"
            if (
                _git(
                    "-C",
                    clone_dir,
                    "rev-parse",
                    "-q",
                    "--verify",
                    upstream,
                    check=False,
                    capture_output=True,
                ).returncode
                != 0
            ):
                upstream = "FETCH_HEAD"
            _git("-C", clone_dir, "rebase", "-q", upstream)
            time.sleep(random.uniform(1, 5))
        raise RuntimeError(
            f"failed to push profiling data to {self.url} after {self.push_attempts} attempts"
        )

    def upload(self, case_dir: str | os.PathLike[str]) -> bool:
        """Push one already-packaged case folder, as it stands, without re-running anything.

        The folder is copied into a fresh clone (replacing an earlier upload
        of the same name) and pushed.
        """
        case_dir = Path(case_dir).resolve()
        metadata_path = case_dir / "case_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"{case_dir} is not a packaged profiling case: no case_metadata.json in it"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        general = metadata.get("general_information", {})
        commit = metadata.get("software_information", {}).get("git_commit") or ""

        with tempfile.TemporaryDirectory(prefix="scope-profiler-results-") as tmp:
            clone_dir = self.clone(Path(tmp) / "results")
            destination = clone_dir / case_dir.name
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(case_dir, destination)
            return self.push(
                clone_dir, label=general.get("label", case_dir.name), commit=commit
            )
