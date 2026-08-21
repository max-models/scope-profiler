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


if __name__ == "__main__":
    main()
