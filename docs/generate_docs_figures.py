"""Generate figures used by the hosted documentation pages."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures" / "cli"


def main() -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / "generate_cli_docs_figures.py"),
            "-o",
            str(FIGURES),
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
