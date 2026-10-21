"""Render the repository's Quarto READMEs to GitHub-flavoured Markdown.

The generated ``.md`` files are consumed by GitHub and PyPI.  Keep the source
in ``.qmd`` files so that the same content can also be rendered by Quarto into
other formats later.  The documentation site (``docs/``) does not use them.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def qmd_sources() -> list[Path]:
    """Return tracked Quarto sources, in stable order."""

    excluded = {".git", ".venv", "node_modules", "sim_1"}
    return sorted(
        path
        for path in ROOT.rglob("*.qmd")
        if not any(part in excluded for part in path.relative_to(ROOT).parts)
    )


def render(source: Path) -> None:
    # Running in the source directory makes --output unambiguous and avoids
    # Quarto's path handling turning a relative output into an absolute one.
    subprocess.run(
        [
            "quarto",
            "render",
            source.name,
            "--to",
            "gfm",
            "--no-execute",
            "--output",
            source.with_suffix(".md").name,
        ],
        cwd=source.parent,
        check=True,
    )


def main() -> None:
    sources = qmd_sources()
    if not sources:
        raise SystemExit("No .qmd sources found")
    for source in sources:
        print(f"Rendering {source.relative_to(ROOT)}")
        render(source)


if __name__ == "__main__":
    main()
