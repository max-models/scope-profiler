#!/usr/bin/env python3
"""Set the release version in every file that carries it.

Two files spell the same release out in their own syntax, and a release where
they disagree ships a CMake package claiming the wrong version::

    python3 tools/set_version.py 0.6.1
    python3 tools/set_version.py --check      # verify they agree; CI-friendly

``--changelog`` additionally retitles the ``## Unreleased`` heading to the new
version and today's date, which is the step that is easiest to forget.

Deliberately *not* touched:

* ``packages/plotly/package.json`` --- the npm package is versioned and
  published independently (``.github/workflows/publish-plotly.yml``).
* ``SP_FORMAT_VERSION``, ``CURRENT_SCHEMA_VERSION``, ``FORMAT_VERSION`` ---
  these version *data formats*, not the library, and move on their own
  schedule. A release that leaves them alone is the normal case.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent.parent

#: Semantic version, without a pre-release or build suffix -- the only shape
#: both setuptools and CMake's ``project(VERSION)`` accept unambiguously.
VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


class VersionSite:
    """One file, and the single capture group holding its version."""

    def __init__(self, path: str, pattern: str, description: str) -> None:
        # The name is the site's own identity, not something derived from
        # ``path``: tests point ``path`` at a copy outside the repository.
        self.name = path
        self.path = REPOSITORY / path
        self.pattern = re.compile(pattern, re.MULTILINE)
        self.description = description

    def read(self) -> str:
        """The version currently written in this file."""
        match = self.pattern.search(self.path.read_text())
        if match is None:
            raise SystemExit(
                f"{self.name}: no version found. The file's layout changed; "
                f"update VERSION_SITES in {Path(__file__).name}.",
            )
        return match.group(1)

    def write(self, version: str) -> bool:
        """Set this file's version; True if the file changed."""
        text = self.path.read_text()
        match = self.pattern.search(text)
        if match is None:
            raise SystemExit(
                f"{self.name}: no version found. The file's layout changed; "
                f"update VERSION_SITES in {Path(__file__).name}.",
            )
        if match.group(1) == version:
            return False
        start, end = match.span(1)
        self.path.write_text(text[:start] + version + text[end:])
        return True


VERSION_SITES = (
    VersionSite(
        "pyproject.toml",
        r'^version = "([^"]+)"',
        "the Python package",
    ),
    VersionSite(
        "CMakeLists.txt",
        r"^project\(scope-profiler VERSION ([^ )]+)",
        "the installed CMake package",
    ),
)

CHANGELOG = REPOSITORY / "CHANGELOG.md"
UNRELEASED = re.compile(r"^## Unreleased\s*$", re.MULTILINE)


def check() -> int:
    """Report whether every site agrees; 0 when they do."""
    found = {site.name: site.read() for site in VERSION_SITES}
    versions = set(found.values())
    for name, version in found.items():
        print(f"  {version:<12} {name}")
    if len(versions) != 1:
        print("\nversions disagree", file=sys.stderr)
        return 1
    print(f"\nall sites agree on {versions.pop()}")
    return 0


def retitle_changelog(version: str) -> bool:
    """Turn the ``## Unreleased`` heading into a dated release heading."""
    text = CHANGELOG.read_text()
    if UNRELEASED.search(text) is None:
        print(f"  {CHANGELOG.name}: no '## Unreleased' heading, left alone")
        return False
    # Local date, but reached through an aware UTC "now": the releaser expects
    # the date on their own calendar, and a naive today() is what DTZ011 warns
    # about. The 3.10 floor rules out datetime.UTC, hence timezone.utc.
    today = dt.datetime.now(dt.timezone.utc).astimezone().date()
    heading = f"## {version} - {today.isoformat()}"
    CHANGELOG.write_text(UNRELEASED.sub(heading, text, count=1))
    print(f"  {CHANGELOG.name}: '## Unreleased' -> '{heading}'")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "version",
        nargs="?",
        help="the new version, as MAJOR.MINOR.PATCH",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report the current versions and exit non-zero if they disagree",
    )
    parser.add_argument(
        "--changelog",
        action="store_true",
        help="also retitle the CHANGELOG's '## Unreleased' heading",
    )
    args = parser.parse_args(argv)

    if args.check:
        if args.version is not None:
            parser.error("--check takes no version argument")
        return check()
    if args.version is None:
        parser.error("a version is required (or use --check)")
    if not VERSION_PATTERN.match(args.version):
        parser.error(f"{args.version!r} is not a MAJOR.MINOR.PATCH version")

    changed = False
    for site in VERSION_SITES:
        previous = site.read()
        if site.write(args.version):
            print(f"  {site.name}: {previous} -> {args.version}  ({site.description})")
            changed = True
        else:
            print(f"  {site.name}: already {args.version}")
    if args.changelog:
        changed |= retitle_changelog(args.version)

    print(f"\n{'updated to' if changed else 'already at'} {args.version}")
    if not args.changelog:
        print("CHANGELOG.md was not touched; pass --changelog to date its heading.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
