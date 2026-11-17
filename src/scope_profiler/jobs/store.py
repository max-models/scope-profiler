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
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

RESULTS_REPO_ENV_VAR = "SCOPE_PROFILER_RESULTS_REPO"


def slug(value: str) -> str:
    """``value`` as a file name: runs of other characters become ``_``."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "case"


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
    path : str
        Directory inside the repository that packaged case folders go into;
        ``{label}`` is replaced by the case's label. ``""`` (the default) puts
        them at the top. A repository shared by several codes uses, say,
        ``mycode/{label}``, which is the layout a catalogue
        (``scope-profiler site``) finds a case's executions in.
    public : bool
        The repository is public: cases uploaded to it leave host names and
        private paths out of their metadata (see ``ProfilingCase.redact``).
    after_push : list[str] or str or None
        Shell commands run after every successful push, e.g. to rebuild the
        docs that show the results: ``gh workflow run docs.yml -R me/mycode``.
        ``{label}``, ``{commit}`` and ``{url}`` are filled in. A failing command
        is reported, not raised: the results are pushed either way.
    """

    url: str
    branch: str | None = None
    message: str = "Add profiling data for {label} at {commit}"
    push_attempts: int = 5
    path: str = ""
    public: bool = False
    after_push: list[str] | str | None = None

    def subdirectory(self, label: str) -> Path:
        """Where the packaged folders of the case ``label`` go, relative to the clone."""
        relative = Path(self.path.format(label=slug(label))) if self.path else Path()
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                f"the store's path must stay inside the repository, not {self.path!r}"
            )
        return relative

    @classmethod
    def from_env(cls, **kwargs) -> GitResultsStore:
        """A store for the repository in ``SCOPE_PROFILER_RESULTS_REPO``."""
        url = os.environ.get(RESULTS_REPO_ENV_VAR)
        if not url:
            raise RuntimeError(
                f"no results repository: pass a GitResultsStore(url=...) or set ${RESULTS_REPO_ENV_VAR}",
            )
        return cls(url=url, **kwargs)

    def run_after_push(self, label: str = "", commit: str = "") -> list[int]:
        """Run the ``after_push`` commands; return their exit codes (a failure only warns)."""
        commands = self.after_push or []
        if isinstance(commands, str):
            commands = [commands]
        values = {"label": label, "commit": (commit or "")[:8], "url": self.url}
        codes = []
        for template in commands:
            command = re.sub(
                r"\{(label|commit|url)\}", lambda m: values[m.group(1)], template
            )
            done = subprocess.run(["bash", "-c", command], check=False)
            if done.returncode != 0:
                print(
                    f"WARNING: after_push command failed (exit code {done.returncode}): {command}"
                )
            codes.append(done.returncode)
        return codes

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
                self.run_after_push(label=label, commit=commit)
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
            label = general.get("label", case_dir.name)
            destination = clone_dir / self.subdirectory(label) / case_dir.name
            if destination.exists():
                shutil.rmtree(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(case_dir, destination)
            return self.push(
                clone_dir, label=general.get("label", case_dir.name), commit=commit
            )
