"""Generate the command output embedded in the documentation.

The examples deliberately run the public API and CLI instead of duplicating
their output in a Markdown fence. Every ``make -C docs html`` refreshes these
build artefacts before Sphinx reads the sources.
"""

from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "docs" / "source" / "_generated"


def write_output(name: str, content: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / name).write_text(content.rstrip() + "\n")


def write_summary(profile: Path, output_name: str) -> None:
    """Render one summary through the same public API shown in the docs."""
    from scope_profiler import read_h5

    summary = io.StringIO()
    with contextlib.redirect_stdout(summary):
        read_h5(profile).print_summary(title="profiling_data.h5  (1 rank(s))")
    write_output(output_name, summary.getvalue())


def make_overview_profile(path: Path) -> None:
    """Create the profile represented by the first quickstart summary."""
    from scope_profiler import ProfileManager

    with ProfileManager.session(file_path=path, verbose=False):
        for _ in range(100):
            with ProfileManager.profile_region("matrix_multiply"):
                sum(range(100))
        for _ in range(1_000):
            with ProfileManager.profile_region("time_step"):
                sum(range(100))


def make_complete_profile(path: Path) -> None:
    """Create the profile represented by the complete quickstart example."""
    from scope_profiler import ProfileManager

    with ProfileManager.session(file_path=path, verbose=False):
        with ProfileManager.profile_region("main"):
            for _ in range(10):
                with ProfileManager.profile_region("iteration"):
                    sum(range(100))


def make_plot_cli_profile(path: Path) -> None:
    """Create the nested workload used by the plotting guide's text output."""
    from scope_profiler import ProfileManager

    with ProfileManager.session(file_path=path, verbose=False):
        with ProfileManager.profile_region("setup"):
            sum(range(100))
        for step in range(3):
            with ProfileManager.profile_region("timestep"):
                with ProfileManager.profile_region("assemble"):
                    sum(range(100))
                with ProfileManager.profile_region("solve"):
                    sum(range(1_000))
                if step == 1:
                    with ProfileManager.profile_region("io"):
                        sum(range(100))


def write_inspect_output(profile: Path, output_name: str, display_name: str) -> None:
    """Run the public inspect command and hide its temporary input path."""
    inspected = subprocess.run(
        [
            sys.executable,
            "-m",
            "scope_profiler",
            "inspect",
            str(profile),
            "--regions-only",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    write_output(output_name, inspected.stdout.replace(str(profile), display_name))


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="scope-profiler-docs-output-") as tmp:
        profile = Path(tmp) / "profiling_data.h5"
        make_overview_profile(profile)
        write_summary(profile, "quickstart-overview.txt")

        write_inspect_output(profile, "inspect-summary.txt", "profiling_data.h5")

        make_complete_profile(profile)
        write_summary(profile, "quickstart-complete.txt")

        run_two = Path(tmp) / "run_2.h5"
        make_plot_cli_profile(run_two)
        write_inspect_output(run_two, "plot-cli-inspect.txt", "run_2.h5")


if __name__ == "__main__":
    main()
