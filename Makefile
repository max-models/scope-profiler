# Repository entry points:
#   make readme   Refresh README.md and its checked-in figure assets.
#   make docs-build   Build the documentation site (docs/, Astro + Starlight) as in CI.
#   make figures  Explicitly regenerate every checked-in figure asset.
#   make version VERSION=x.y.z   Set the release version everywhere.
#   make check-version           Verify every file agrees on it.
#
# The documentation site uses @max-models/starlight-theme (docs/node_modules): the notebook
# tool runs notebooks/**/*.ipynb (the example notebooks and the guides), docs/generate_docs_outputs.py the command output the pages import.
PYTHON ?= .venv/bin/python
DOCS_ENV = DOCS_PYTHON=$(abspath $(PYTHON))
NOTEBOOKS = cd docs && $(DOCS_ENV) npx starlight-notebooks
FORCE ?=

.PHONY: help readme readme-figures markdown docs-figures figures version check-version \
	docs-install docs-notebooks docs-outputs docs-dev docs-build docs-preview docs-clean

help:  ## list the targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

readme-figures:  ## regenerate the README figures in figures/
	@python3 docs/generate_readme_figures.py

readme: readme-figures  ## refresh README.md and its figures
	@quarto render README.qmd --to gfm --no-execute --output README.md

markdown:  ## render every Quarto README (.qmd) to GitHub-flavoured Markdown
	@python3 docs/render_markdown.py

docs-figures:  ## regenerate the docs figures in figures/
	@python3 docs/generate_docs_figures.py

# Figure assets are versioned under figures/. Refresh them explicitly with
# `make figures`; a docs build only uses those canonical assets.
figures: readme-figures docs-figures  ## regenerate every checked-in figure

version:  ## set the release version everywhere (VERSION=x.y.z)
	@test -n "$(VERSION)" || { echo "usage: make version VERSION=x.y.z" >&2; exit 2; }
	@python3 tools/set_version.py "$(VERSION)"

check-version:  ## verify every file agrees on the version
	@python3 tools/set_version.py --check

docs-install:  ## Python docs extra and the npm packages
	$(PYTHON) -m pip install -e ".[docs]"
	cd docs && npm ci

docs-notebooks:  ## execute the changed notebooks and convert all of them (FORCE=1 reruns all)
	$(NOTEBOOKS) execute $(if $(FORCE),--force,)
	$(NOTEBOOKS) convert

docs-outputs:  ## run the CLI and API examples into docs/src/generated/
	$(PYTHON) docs/generate_docs_outputs.py

docs-dev: docs-outputs  ## live preview at http://localhost:4321/scope-profiler/
	$(NOTEBOOKS) convert
	cd docs && $(DOCS_ENV) DOCS_VALIDATE_LINKS=false npx astro dev

docs-build: docs-notebooks docs-outputs  ## the static site in docs/dist, as in CI
	cd docs && npx starlight-check-mdx && $(DOCS_ENV) npx astro build

docs-preview:  ## serve docs/dist
	cd docs && npx astro preview

docs-clean:  ## remove generated pages, assets and build output
	$(NOTEBOOKS) clean
	rm -rf docs/dist docs/.astro docs/src/generated
