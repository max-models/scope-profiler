---
title: Building the docs
description: How this documentation site is built from Markdown pages, notebooks, generated output and docstrings.
---

The site in `docs/` is built with [Astro](https://astro.build/) and
[Starlight](https://starlight.astro.build/). It has four kinds of pages:

- **Hand-written pages** in `docs/src/content/docs/`, as Markdown or MDX.
- **Tutorials**, converted from the Jupyter notebooks in `tutorials/` by `starlight-notebooks`:
  each notebook is executed and its outputs (text, images, tables) are written into an MDX page.
- **API reference**, generated from the docstrings by
  [starlight-pydocs](https://ewels.github.io/starlight-pydocs/), one page per module. The
  [Python API](/scope-profiler/reference/api/) page places single objects among prose with
  `<Autodoc name="..." />`.
- **Command output**: `docs/generate_docs_outputs.py` runs the CLI and the API and writes their
  output to `docs/src/generated/`, which the MDX pages import (`?raw`) instead of copying it into
  a code fence.

The figures in `figures/` and the `@scope-profiler/plotly` sources in `packages/plotly/src/` are
served as they are, at `figures/` and `plotly/` of the site (`docs/public/` links to them), so
the interactive pages always use the checkout's package. Refresh the figures with
`make figures`; a docs build only uses them.

The configuration (`defineDocs` in `docs/astro.config.mjs`), the look, the header, the notebook
tool and the build workflow come from the shared
[starlight-theme](https://github.com/max-models/starlight-theme) of the max-models projects;
change them there, and update the version in `docs/package.json` and
`.github/workflows/docs.yml` to pick up a new release.

## Requirements

- Node 22 or newer.
- The package installed with the `docs` extra: `pip install -e ".[docs]"`.
- An MPI implementation (`mpiexec`) for the MPI examples of the generated output.

## Commands

```bash
make docs-install     # npm packages and the Python docs extra
make docs-notebooks   # execute tutorials/*.ipynb and convert them to pages
make docs-outputs     # run the CLI and API examples into docs/src/generated/
make docs-dev         # live preview at http://localhost:4321/scope-profiler/
make docs-build       # the static site in docs/dist, as in CI
make docs-preview     # serve docs/dist
make docs-clean       # remove generated pages and build output
```

`make docs-notebooks` only reruns a notebook whose cells changed since the last run; pass
`FORCE=1` to rerun all of them.
