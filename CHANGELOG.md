# Changelog

## Unreleased

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
