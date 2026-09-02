# Changelog

## Unreleased

### Added

- Every `export plot-data --format json` document now carries the same
  envelope -- `format`, `format_version` and the `plot` kind that produced it
  -- instead of only four of the kinds carrying it. It is stamped centrally
  when the file is written, so a new plot kind cannot ship without it, and
  `region_statistics.json` carries it too.
- `@scope-profiler/plotly` gained `buildFigure(payload)`, which dispatches on
  that `plot` field, so a page can render whatever the profiler wrote without
  naming a builder. It rejects a foreign document or a `format_version` newer
  than the package supports, and falls back to `inferPlotKind(payload)` for
  JSON written before the envelope covered every kind.
- `@scope-profiler/plotly` gained builders for the payloads that had none:
  `buildDensityFigure` (timeline occupancy), `buildCallgraphFigure` (a Sankey
  of either callgraph shape), `buildRegionSummaryFigure` (the slowest regions
  in `region_statistics.json`), `buildLikwidFigure`, and
  `buildWeakScalingFigure` / `buildScalingEfficiencyFigure`.
- The weak-scaling and scaling-efficiency exports now write the `colors` and
  `options` blocks their speedup sibling already wrote, so all three plot with
  the same axis labels and baseline.
- Added a JSON output format, chosen by the output file's extension the way
  viztracer's is: `scope-profiler run -o profile.json` (or `.json.gz`) writes
  the run as JSON, `-o report.html` writes a rendered HTML report, and
  anything else stays HDF5, which remains the default. The run itself always
  writes HDF5 and the requested format is rendered from it afterwards, so
  `-o` never changes what a run measures; under MPI rank 0 does the
  conversion.
- Added `scope-profiler export json`, `--gzip` and `--indent` included, which
  converts an existing profile the same way.
- The JSON document is a lossless copy of the run rather than a view for one
  viewer: per-call timestamps in integer nanoseconds, thread and task tables,
  LIKWID counters and line-profiler records. `read_json()` rebuilds a
  `ProfilingResults` indistinguishable from `read_h5()`'s, and `write_json()`
  / `export_json()` write one from any result set.
- `inspect`, `plot`, `report`, `diff`, `check`, `tui`, `line-profile`,
  `benchmark` and the exporters read a JSON profile wherever they read an
  HDF5 one, through the new `read_profile()` / `write_profile()` dispatch on
  the file name.
- Added `track_threads=True`, which profiles every thread rather than
  assuming one. Each thread gets its own timestamp buffers and its own scope
  stack, so regions entered concurrently no longer reserve and close each
  other's slots, and every recorded call carries the thread it ran on.
  Nesting, exclusive time and the call graph are reconstructed per thread.
  The run also describes each thread -- name, OS ids, exact start and end
  times, and CPU time -- through `ProfilingResults.threads` and
  `ProfilingResults.thread_summary()`, and `Region.for_thread()` slices a
  region down to one thread.
- Added `track_async=True` (which implies `track_threads`), following every
  asyncio task and, when `greenlet` is installed, every greenlet. Interleaved
  tasks become separate lanes instead of bogus nesting, each call records the
  time its task spent suspended inside it (`Region.await_times`), and
  `ProfilingResults.tasks` reports per-task step counts with running and
  awaiting totals. Loops are found through `BaseEventLoop.run_forever` and
  `create_task`, an application's own task factory is chained rather than
  replaced, and `tracker.instrument_loop(loop)` covers loop implementations
  that reach neither.
- Added the per-call `thread_ids`, `task_ids` and `await_ns` event columns and
  the `thread_table` / `task_table` groups to the HDF5 layout, within schema
  2. They are written only by a run that tracked threads, and a file without
  them reads back exactly as before.

### Changed

- A process forked out of an active `track_threads`/`track_async` session now
  stands down: the thread, asyncio and greenlet hooks are removed in the child
  and the inherited lane tables dropped, since nothing in the child will ever
  finalize that run. Without it a forked worker running an event loop
  accumulated a task record per task for its whole life. The child tracks
  again as soon as it opens a session of its own, which is how a
  multiprocessing worker is meant to be profiled -- one run, one file, per
  process.
- `export_speedscope` writes one profile per lane -- named after its thread or
  task -- rather than one per rank, and `call_stack.split_by_lane()` exposes
  the same split. An evented speedscope profile's timestamps must not go
  backwards, which two interleaved lanes walked as one call tree produce.
- `ProfileManager.session()` now removes the thread, asyncio and greenlet
  hooks when the session ends. The lower-level `setup()`/`finalize()` pair
  keeps them until the next `setup()`, since `finalize()` there can be a
  checkpoint in the middle of a run.

### Fixed

- `scope-profiler export plot-data --plots callgraph` no longer fails with
  `AttributeError: 'Namespace' object has no attribute 'compact_callgraph'`.
  `callgraph` was a valid `--plots` choice for the export, but its two flags
  were registered only on `scope-profiler plot`; both commands now share them.
- `@scope-profiler/plotly`'s flame builder honours the documented
  `filterRegion` option, which it silently ignored. Calls whose parent the
  filter removed are re-parented onto their nearest surviving ancestor rather
  than dropped, so the icicle stays one tree.
- `%%scope_recursive` reports a failing cell with its source again, on every
  IPython version. The magic sliced its own frame off the traceback and then
  left the renderer's `tb_offset` at its default, which on some versions
  drops an outermost frame too -- between them they removed the cell's only
  frame, leaving the exception with no code shown at all, while on versions
  that drop nothing the magic's own frame appeared beside the cell. The
  slice is now the only one that happens: `tb_offset=0` is passed
  explicitly, so no version's default takes part.
- `call_stack.build_call_arrays` leaves `CallArrays.lane` empty for a run that
  tracked no threads, instead of materializing a full-length column of `-1`.
  Filling it would have added an allocation, a concatenate and a gather over
  every event to the reconstruction of every single-threaded run (~8 ms and
  48 MB per two million events) to say only "one stack".
- `call_stack.build_call_arrays` reconstructs each lane separately instead of
  treating a whole rank as one stack, so overlapping calls from different
  threads or tasks no longer raise `NestingError` on a run that recorded
  which lane they belong to.

## 0.5.0 - 2026-09-01

### Added

- Added independent `ProfileManager()` instances, allowing multiple profiling
  sessions with separate configurations, regions, decorators, call IDs, and
  output files to coexist. The class-level `ProfileManager` API remains the
  backward-compatible process-wide default manager.
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
