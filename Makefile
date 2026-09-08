# Repository entry points:
#   make readme   Refresh README.md and its checked-in figure assets.
#   make docs     Build the documentation site using the checked-in figures.
#   make figures  Explicitly regenerate every checked-in figure asset.
#   make version VERSION=x.y.z   Set the release version everywhere.
#   make check-version           Verify every file agrees on it.
#
# The implementation lives in docs/Makefile; these targets keep common tasks
# available from the repository root.
.PHONY: readme docs figures version check-version

readme:
	@$(MAKE) -C docs readme

docs:
	@$(MAKE) -C docs html

figures:
	@$(MAKE) -C docs figures

version:
	@test -n "$(VERSION)" || { echo "usage: make version VERSION=x.y.z" >&2; exit 2; }
	@python3 tools/set_version.py "$(VERSION)"

check-version:
	@python3 tools/set_version.py --check
