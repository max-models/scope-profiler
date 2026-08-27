.PHONY: readme docs figures

readme:
	@$(MAKE) -C docs readme

docs:
	@$(MAKE) -C docs html

figures:
	@$(MAKE) -C docs figures
