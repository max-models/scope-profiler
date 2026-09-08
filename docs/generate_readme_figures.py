"""Generate the figures embedded in the repository README."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"


def main() -> None:
    generators = (
        ROOT / "examples" / "generate_readme_figures.py",
        ROOT / "examples" / "benchmark_overhead.py",
    )
    for generator in generators:
        subprocess.run(
            [sys.executable, str(generator), "-o", str(FIGURES)],
            cwd=ROOT,
            check=True,
        )

    # The storage figure needs several run sizes to have a curve to draw, so
    # it takes a rank sweep rather than the "-o DIR" the others do. Kept small
    # enough to stay a figure build rather than a benchmark session.
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / "benchmark_io.py"),
            "--scaling",
            "2,8,32,128",
            "--regions",
            "8",
            "--events",
            "500",
            "--figure",
            str(FIGURES),
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
