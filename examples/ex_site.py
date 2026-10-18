"""
Astro pages for a catalogue of profiling runs
=============================================

``scope-profiler site`` turns a YAML manifest that lists profiling files into
the pages and chart data of a catalogue, inside an Astro/Starlight project. The
Astro site's own build then makes the website; the ``scope-profiler-catalog``
template repository is such a project.

This script profiles a small workload at two problem sizes, writes a manifest
for them, builds the catalogue into a temporary Astro project and checks what
was written.

The pages show everything ``scope-profiler report`` does: headline numbers, hotspots, the
region table, load balance, line profiles, hardware counters, comparisons and charts.

Run::

    python examples/ex_site.py

Then, for your own runs, write a manifest like the one printed below and run::

    scope-profiler site catalogue.yaml -o docs
"""

import json
import math
import tempfile
from pathlib import Path

from scope_profiler import ProfileManager
from scope_profiler.astro_site import build_site, load_manifest


@ProfileManager.profile("assemble")
def assemble(size):
    return [math.sin(i) * math.cos(i) for i in range(size)]


@ProfileManager.profile("solve")
def solve(values):
    return sum(math.sqrt(abs(value)) + math.log1p(abs(value)) for value in values)


def profile_run(size, path):
    with ProfileManager.session(
        use_likwid=False,
        file_path=str(path),
        verbose=False,
    ):
        with ProfileManager.region("simulation"):
            solve(assemble(size))


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name, size in (("small", 20_000), ("large", 80_000)):
            profile_run(size, root / f"{name}.h5")

        manifest_path = root / "catalogue.yaml"
        manifest_path.write_text(
            """\
title: Example catalogue
description: One workload at two problem sizes.
base: /example
cases:
  - id: toy-solver
    title: Toy solver
    description: The same workload at two problem sizes.
    tags: [example]
    scaling: none
    info: {note: sizes are 20 000 and 80 000}
    runs:
      - {path: small.h5, label: small}
      - {path: large.h5, label: large}
""",
            encoding="utf-8",
        )
        print(manifest_path.read_text(encoding="utf-8"))

        docs = root / "docs"
        result = build_site(load_manifest(manifest_path), docs)
        print(f"Wrote {len(result.files)} files for {result.runs} runs")

        content = docs / "src/content/docs/catalogue"
        case = content / "toy-solver"
        for page in (
            content / "index.mdx",
            content / "toy-solver.mdx",
            case / "small.mdx",
        ):
            assert page.is_file(), page
        # A page is components plus a JSON data file with what it shows.
        small = json.loads((case / "small.data.json").read_text())
        assert small["report"]["summary"]["kpis"], "headline numbers"
        assert small["report"]["regions"]["groups"], "the region table"
        # The case compares the runs, linking each run's page under the base path.
        compared = json.loads((content / "toy-solver.data.json").read_text())
        assert compared["links"]["runs"][0] == "/example/catalogue/toy-solver/small/"
        assert len(compared["report"]["candidates"]) == 1
        # Every chart the pages draw has its data, as plot-data JSON.
        for data in (small, compared):
            for panel in data["charts"]["panels"]:
                chart = json.loads(
                    (docs / "public" / panel["src"].lstrip("/")).read_text()
                )
                assert chart["plot"], panel["src"]
        # Nothing that identifies the machine is published.
        everything = "".join(p.read_text() for p in content.rglob("*") if p.is_file())
        assert "hostname" not in everything


if __name__ == "__main__":
    main()
