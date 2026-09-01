# Changelog

## Unreleased

### Added

- Added Jupyter/IPython magics via `%load_ext scope_profiler.ipython_magics`,
  and a tutorial notebook covering them. Installed with the new `notebook`
  extra. Recording: `%%scope`, `%scope_timeit`, `%%scope_line`,
  `%%scope_recursive` (every call in a cell, nothing instrumented) and
  `%%scope_agg` (aggregation mode). Working with recorded runs:
  `%scope_load` (an HDF5 run from an MPI job or `scope-profiler run`),
  `%scope_df` (pandas), `%scope_last`, `%scope_compare`, `%scope_export`
  and `%scope_reset`.

## 0.4.2 - 2026-08-31

### Fixed

- Included the refactored `scope_profiler.plotting_scripts` package in the
  release artifact so top-level plotting imports work after installation.

## 0.4.0 - 2026-08-31

### Fixed

- Restored the `TOTAL` row in summaries printed during finalization while
  keeping `inspect` output focused on individual regions.
- Fixed HTML reports that requested the default percentage column.
- Updated the C and Fortran standalone examples to invoke the profiler with
  the configured Python interpreter, so source-tree environments work without
  an installed console-script entry point.
- Kept Plotly hover data backend-neutral by relying on `maxplotlibx` to remove
  it before Matplotlib rendering.

- Added MIT licensing metadata and release-package validation.
- Made run timestamps explicit UTC ISO-8601 values.
- Added incremental mypy checking for the core public result and configuration
  APIs.

## 0.3.6

- Added support for HTML profiling reports and expanded post-processing plots.
