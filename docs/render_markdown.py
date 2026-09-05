"""Render the repository's Quarto sources to GitHub-flavoured Markdown.

The generated ``.md`` files are consumed by GitHub, PyPI, and Sphinx.  Keep
the source in ``.qmd`` files so that the same content can also be rendered by
Quarto into other formats later.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def qmd_sources() -> list[Path]:
    """Return tracked/documentation Quarto sources, in stable order."""

    excluded = {".git", ".venv", "sim_1"}
    return sorted(
        path
        for path in ROOT.rglob("*.qmd")
        if not any(part in excluded for part in path.relative_to(ROOT).parts)
        and path.relative_to(ROOT).parts[:2] != ("docs", "build")
        and path.relative_to(ROOT).parts[:3] != ("docs", "source", "tutorials")
    )


def render(source: Path) -> None:
    output = source.with_suffix(".md")
    # Running in the source directory makes --output unambiguous and avoids
    # Quarto's path handling turning a relative output into an absolute one.
    command = [
        "quarto",
        "render",
        source.name,
        "--to",
        "gfm",
        "--output",
        output.name,
    ]
    is_tutorial = source.parent == ROOT / "tutorials"
    command.insert(-2, "--execute" if is_tutorial else "--no-execute")
    environment = os.environ.copy()
    kernel_directory = None
    if is_tutorial:
        # Quarto's global ``python3`` kernelspec may belong to an unrelated
        # environment. Give it a temporary higher-priority kernelspec that
        # uses the interpreter running this documentation build.
        kernel_directory = tempfile.TemporaryDirectory(
            prefix="scope-profiler-docs-kernel-"
        )
        kernel_root = Path(kernel_directory.name)
        kernel_path = kernel_root / "kernels" / "python3"
        kernel_path.mkdir(parents=True)
        (kernel_path / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [
                        sys.executable,
                        "-m",
                        "ipykernel_launcher",
                        "-f",
                        "{connection_file}",
                    ],
                    "display_name": "scope-profiler docs",
                    "language": "python",
                }
            )
        )
        jupyter_path = environment.get("JUPYTER_PATH")
        environment["JUPYTER_PATH"] = (
            str(kernel_root)
            if not jupyter_path
            else os.pathsep.join((str(kernel_root), jupyter_path))
        )
        # Always exercise the checkout being documented, not an older wheel.
        source_path = str(ROOT / "src")
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            source_path if not existing else os.pathsep.join((source_path, existing))
        )
    try:
        subprocess.run(
            command,
            cwd=source.parent,
            env=environment,
            check=True,
        )
    finally:
        if kernel_directory is not None:
            kernel_directory.cleanup()
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
