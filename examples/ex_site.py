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

Run::

    python examples/ex_site.py

Then, for your own runs, write a manifest like the one printed below and run::

    scope-profiler site catalogue.yaml -o docs
"""

import json
import math
import re
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
        for page in (content / "index.mdx", case / "index.mdx", case / "small.mdx"):
            assert page.is_file(), page
        # The pages link to each other under the base path, and to charts under /catalogue.
        assert (
            "[small](/example/catalogue/toy-solver/small/)"
            in (case / "index.mdx").read_text()
        )
        for url in re.findall(r'src="([^"]+)"', (case / "small.mdx").read_text()):
            chart = json.loads((docs / "public" / url.lstrip("/")).read_text())
            assert chart["format"] == "scope-profiler-plot-data", url
        # Nothing that identifies the machine is published.
        assert "hostname" not in "".join(p.read_text() for p in content.rglob("*.mdx"))


if __name__ == "__main__":
    main()
