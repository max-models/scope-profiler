"""Render the repository's Quarto sources to GitHub-flavoured Markdown.

The generated ``.md`` files are consumed by GitHub, PyPI, and Sphinx.  Keep
the source in ``.qmd`` files so that the same content can also be rendered by
Quarto into other formats later.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def qmd_sources() -> list[Path]:
    """Return tracked/documentation Quarto sources, in stable order."""

    excluded = {".git", ".venv", "docs/build", "sim_1"}
    return sorted(
        path
        for path in ROOT.rglob("*.qmd")
        if not any(part in excluded for part in path.relative_to(ROOT).parts)
    )


def render(source: Path) -> None:
    output = source.with_suffix(".md")
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
            output.name,
        ],
        cwd=source.parent,
        check=True,
    )
    # Pandoc inserts a space before fenced-directive attributes.  MyST's
    # ``eval-rst`` fence is sensitive to that spelling, so normalize it after
    # Quarto renders the document.
    content = output.read_text()
    content = content.replace("``` {eval-rst}", "```{eval-rst}")
    # Preserve MyST's explicit-label syntax when it precedes a heading.
    content = re.sub(r"(?m)^(\([^\n]+\)=) ?\\(#+ )", r"\1\n\2", content)
    output.write_text(content)


def main() -> None:
    sources = qmd_sources()
    if not sources:
        raise SystemExit("No .qmd sources found")
    for source in sources:
        print(f"Rendering {source.relative_to(ROOT)}")
        render(source)


if __name__ == "__main__":
    main()
