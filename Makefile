# Repository entry points:
#   make readme   Refresh README.md and its checked-in figure assets.
#   make docs     Build the documentation site using the checked-in figures.
#   make figures  Explicitly regenerate every checked-in figure asset.
#
# The implementation lives in docs/Makefile; these targets keep common tasks
# available from the repository root.
.PHONY: readme docs figures

readme:
	@$(MAKE) -C docs readme

docs:
	@$(MAKE) -C docs html

figures:
	@$(MAKE) -C docs figures
