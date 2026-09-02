# Changelog

## Unreleased

### Added

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

- `%%scope_recursive` reports a failing cell with its source again. The magic
  sliced its own frame off the traceback before handing it to
  `showtraceback()`, which drops the outermost frame itself (`InteractiveTB`
  has `tb_offset = 1`); between them they removed the cell's only frame,
  leaving the exception with no code shown at all.
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
