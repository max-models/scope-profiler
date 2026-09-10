# Changelog

## Unreleased

### Fixed

- `--metrics` with more than one statistic kept only the last one. Every
  metric was plotted by its own `plot_durations` call, and each was handed the
  same `durations_data.json` path and the same image path, so each overwrote
  what the one before it had written. The metrics are now rendered in one
  call, which exports all of them and gives each figure its own
  `durations_plot_<metric>.png`.
- The HDF5 guide now identifies schema 3 as the current Python output schema,
  and a test keeps the documented HDF5, JSON, and native format versions tied
  to their implementation constants.
- Platform-independent perf-event tests no longer depend on the host being
  Linux, and the native-import test invokes the active Python interpreter
  instead of assuming a `python` command exists.
- Reading a profile written by a newer schema silently reported zero regions.
  Both the full and the summary-only reader tested the schema version for
  equality with 2 rather than a lower bound, so any later schema fell through
  to the pre-columnar layout, which finds nothing.
- `hdf5_chunk_size` now applies to every dataset in the file. It reached only
  the four event columns; the eleven `rank_region_index` and `region_table`
  datasets kept h5py's default guess of 1024 elements whatever was configured.
- `call_graph()` returned a single collapsed node for any profile written by
  the C or Fortran API, or imported from one. Native output records no parent
  links, but the writer stored a full `call_ids`/`parent_ids` column of `-1`
  rather than omitting it; the reader took that at face value, every call
  collided on id `-1`, and the graph degenerated. Those columns are now
  written only by a run that actually numbered its calls -- as
  `gpu_durations` and the thread/task lanes already were -- so the reader
  falls back to reconstructing the nesting from the timestamps, which
  `call_stack()` was doing correctly all along. Native profiles also halve in
  size, having stored two int64 columns per event that carried nothing.

### Changed

- **HDF5 schema 3.** A region's calls are stored as the gap since its previous
  call and the duration of each call, rather than two absolute nanosecond
  timestamps: `events/start_deltas` and `events/durations` replace
  `events/start_times` and `events/end_times`. The information is identical
  and the round trip is exact -- the first value of each run is absolute --
  but the magnitudes drop from ~60 bits to ~15, which is most of what makes
  compression effective: 290 KiB to 74 KiB on a 100,000-event profile at the
  same gzip level. Each run is encoded independently, so a rank still writes
  the events it owns without needing any other rank's data and the parallel
  HDF5 writer stays a plain per-rank slice write. Schema 1 and 2 files
  continue to read.
- A finished profile is repacked when it is published, storing small datasets
  contiguously. HDF5 allocates a full chunk and a chunk-index B-tree for a
  growable dataset as soon as its first element is written, and never returns
  freed space to the file, so a one-region run spent 193 KiB to hold 0.4 KiB
  of data; it is now 17 KiB. Files large enough for the overhead not to matter
  are left alone.

### Added

- **Weak-scaling efficiency.** `plot_weak_scaling_efficiency()` (and
  `scope-profiler plot weak_scaling_efficiency`) plots baseline runtime over
  runtime at each scale, against a flat ideal of 1.0. This is the reading a
  weak-scaling study needs, and the existing `plot_scaling_efficiency` is not
  it: that one divides by an ideal speedup proportional to the rank count,
  which is right for a fixed problem size and reports a near-zero efficiency
  that means nothing when the problem grows with the machine. Pass
  `work_per_rank` -- one value per file, in any unit -- to have the runs
  checked for the constant work per rank the plot assumes, rather than
  silently comparing runs that did different amounts of it. The npm package
  builds it as `buildWeakScalingEfficiencyFigure`.
- `plot_durations()` takes `metrics=[...]` for several statistics in one call.
  Each still gets its own figure, but they share one data export with the
  metric named per row, so a single JSON can back a chart whose metric the
  viewer switches.
- `first` and `last` join `avg`/`min`/`max`/`total` as duration metrics, in
  `plot_durations()` and `--metrics`: the chronologically first and last
  call's duration, which separate one-off warmup -- JIT, allocation, the first
  device transfer -- from steady state. `write_region_statistics_json` already
  reported both; now they can be plotted.
- `export_flamegraph_svg()` and `scope-profiler export flamegraph` render the
  reconstructed call tree to a standalone SVG through
  [flameprof](https://pypi.org/project/flameprof/) (added to the `pproc`
  extra): the aggregated flame graph as a self-contained document with no
  JavaScript, for a static site or a report. It carries a `viewBox` rather
  than flameprof's fixed pixel width, so a page can inline it -- which is also
  the only way the per-frame tooltips survive -- and scale it; `--fixed-width`
  keeps the standalone form.
- The npm package takes a `theme`: `"light"`, `"dark"`, or your own tokens,
  per build call or once through `setTheme()`. It colours text, gridlines, the
  hover surface and the dashed ideal lines. The default `auto` is what every
  figure did before -- no text colour, a grey grid that reads on any
  background -- so nothing changes for a page that does not ask.
- `buildComparisonFigure` puts two runs of a `region_statistics` document side
  by side over the regions both recorded, vertically and without a top-N cap:
  the "what changed between these two runs?" reading of
  `buildRegionSummaryFigure`, which grew `files`, `orientation`,
  `commonRegionsOnly` and short metric names (`total`, `avg`, …) to serve it.
- CI now runs the ordinary test suite on every supported Python version
  (3.10--3.14) and adds a native macOS job for portable functionality. The
  macOS environment includes Open MPI and mpi4py so it also exercises the
  lazy MPI initialization contract.
- Immutable compatibility fixtures cover HDF5 schemas 1--3, JSON profile v1,
  and native trace v1/v2, so reader compatibility is checked against committed
  fixed representative bytes rather than only files emitted by the current
  writer.
- `examples/benchmark_io.py` reports storage alongside its timings: the merged
  file in bytes per recorded event, written plain and again through
  `hdf5_compression="auto"`, with what the filter costs to write and to read.
  `--figure` draws bytes-per-event against run size, showing the fixed cost
  amortising away and the point at which automatic compression starts
  applying. It also writes through the real publication path now, so the sizes
  it reports are the ones a user gets.
- `examples/benchmark_overhead.py` covers the `track_threads` and
  `track_async` modes, so its figure and the budgets in `test_overhead.py`
  describe the same set of region types.
- Overhead and file-size analysis is now measured by the test suite, not only
  by the benchmark scripts. `test_overhead_io.py` budgets what `finalize()`
  spends writing a profile and what post-processing spends reading one back
  --- including the publication pass, automatic compression, summary-only
  reads and the timestamp codec --- and joins `test_overhead.py` under the
  `overhead` marker that CI already runs in an isolated job.
  `test_storage_size.py` covers bytes per event, the fixed floor, and how both
  scale with events, ranks and regions; file size is deterministic, so its
  budgets are tight rather than an order of magnitude clear. Alongside the
  absolute budgets each module asserts _scaling_ --- a ratio between the same
  measurement at two sizes --- which is what catches a change in the shape of
  a cost rather than only one large enough to blow a budget.
- `hdf5_compression="auto"` compresses a run's event columns only once it is
  large enough for the saving to repay the write CPU, and leaves small
  profiles uncompressed. With the schema-3 encoding above, a 100,000-event
  profile drops from 3313 KiB to 142 KiB (23x) for about 12 ms of extra write
  time.
- `session()`, `region()`, `profile`, `setup()` and `finalize()` are
  importable from the package root, so everyday instrumentation needs no class
  name: `with sp.region("solve"):`. They are the `ProfileManager` class
  methods themselves, acting on the same process-wide manager, not wrappers.
- `ProfileManager.region()` is the new name for `profile_region()`, which it
  describes better --- it gets or creates a region rather than profiling one.
  `profile_region` remains available as an alias for existing instrumentation;
  the two are the same object.
- `load()` reads a profile back whatever format it is in, choosing by the
  file's contents rather than its name, so a JSON profile under a `.h5` name
  still opens. `sniff_profile_format()` exposes that detection on its own, and
  `read_h5()`/`read_json()`/`read_profile()` remain for a known format. An
  HTML report is refused with an explanation rather than a parse error.
- Settings that share a prefix can be given as groups --- `MemrayOptions`,
  `GPUOptions` and `HDF5Options` --- on `ProfilingOptions` or as
  `[profiling.memray]`, `[profiling.gpu]` and `[profiling.hdf5]` sub-tables in
  a TOML config. They spell the same settings without the prefix; setting one
  thing both ways raises rather than picking a winner.

### Changed

- `setup()` and `session()` take their settings as `**overrides` instead of
  restating 28 keyword parameters. `ProfilingOptions` is now the single place
  each setting is declared: the accepted names are derived from it, and their
  defaults live only in `ProfilingConfig.__init__`, so the four places a
  setting used to be repeated can no longer drift. An unrecognised keyword
  raises `TypeError` naming the closest real setting, as an unrecognised TOML
  key now does too.

- The C API can write its profile as HDF5 directly, in the same schema-2
  layout a Python run produces, so `scope-profiler inspect`/`plot` and
  `read_h5()` open a C run's output with no import step. It is opt-in at
  compile time (`-DSP_USE_HDF5` plus libhdf5, or CMake's
  `-DSCOPE_PROFILER_ENABLE_HDF5=ON`), because it is the only thing that gives
  the library a dependency, and is the default format wherever it is compiled
  in; `sp_set_output_format()` / `sp_profiler_set_output_format()` pin either
  format explicitly, and asking for HDF5 in a build without it returns
  `SP_ERR_UNSUPPORTED` and keeps writing the `.spt` trace.
- `scope-profiler import-native` and `finalize(native_traces=...)` both read
  the per-rank `.h5` files an HDF5 C build writes, mixed freely with `.spt`
  traces from other ranks or from Fortran; `import-native` picks both up from
  a directory. A merged profile in that
  directory is not treated as input, so re-running an import does not fold a
  previous result into the next one.
- An import now carries each region's source location through into the HDF5
  file it writes, instead of dropping it.

- Optional C++11 MPI wrappers and a matching mpi4py communicator proxy profile
  point-to-point, collective, nonblocking-initiation, and wait operations with
  a shared label schema for message bytes, peer/root, tag, communicator, and
  request origin. The Python layer imports no MPI module itself, while the
  CMake `scope-profiler::mpi` add-on supplies native MPI headers and linkage.
- `scope-profiler run` automatically profiles mpi4py programs that use
  `MPI.COMM_WORLD`, `MPI.COMM_SELF`, or communicators derived from them without
  modifying their source code. The same behavior is available to normal
  sessions through `ProfilingOptions(profile_mpi_calls=True)` or TOML;
  `--no-mpi-calls` disables the CLI interception.
- Added a runnable mpi4py example using
  `ProfilingOptions(profile_mpi_calls=True)` with predefined and derived
  communicators.
- C++ call sites can use `SP_PROFILE_SCOPE`, `SP_PROFILE_FUNCTION`, and their
  explicit-context variants. Defining `SP_DISABLE_PROFILING` removes the
  instrumentation completely, including argument evaluation, region strings,
  and references to the native profiler ABI.
- C++17 projects can define `SP_HEADER_ONLY` (or link the corresponding CMake
  target) to get inline recorder definitions shared across translation units,
  without compiling or distributing `scope_profiler.c`.
- Native C++ scopes can drive LIKWID marker counters directly with
  `SP_USE_LIKWID`, `sp::LikwidSession`, and `sp::likwid_thread_init()`, while
  still recording the normal scope-profiler timeline.
- A first-class CMake package exports `scope-profiler::native`,
  `scope-profiler::cpp`, `scope-profiler::header-only`, and, when configured,
  `scope-profiler::likwid`; the same targets work from an installed package or
  through FetchContent.
- The C API gained an explicit-context form for library code that must not
  interfere with a caller's own profiling (or with another instance of
  itself): `sp_create()`/`sp_destroy()` plus `sp_profiler_region()`,
  `sp_profiler_begin()`/`sp_profiler_end()`, `sp_profiler_finalize()` and the
  rest, each taking an `sp_profiler *`. The existing `sp_init()`/`sp_region()`/
  `sp_begin()`/`sp_end()`/`sp_finalize()` globals are unchanged and now
  delegate to a hidden default context, so no existing call site needs to
  change.
- `sp_profiler_scope_begin()`/`sp_scope_end()` return a checked token for one
  exact call rather than identifying it by region alone, so an incorrectly
  ordered end is refused (rather than silently mistiming the wrong call, to be
  caught later on import) and a moved-from token is naturally inert.
  `sp_profiler_end_last()` ends whichever call, in any region, was opened most
  recently.
- `sp_profiler_reset()` discards recorded calls and per-region statistics
  while keeping every region handle valid, for periodic reporting followed by
  a fresh measurement window; `sp_profiler_get_region_stats()` reads a
  region's call count and total/min/max duration; `sp_profiler_flush()` writes
  what has been recorded so far without stopping profiling.
- `sp_region_at()` / `sp_profiler_region_at()` record where a region is
  defined (source file and line) alongside its name, using the same shape the
  Python API's own source navigation and exports already understand. The
  native trace format gained a version 2 to carry it; version 1 (written by
  the Fortran API) keeps reading exactly as before, so a mixed C/Fortran run
  still merges into one profile. A name first registered without a location
  (via `sp_region()`) can pick one up on a later `sp_region_at()` call for the
  same name -- the location backfills onto the existing handle rather than
  being dropped -- and `SP_REGION_AT()`/`SP_PROFILER_REGION_AT()` macros fill
  in `__FILE__`/`__LINE__` automatically.
- `scope_profiler.hpp`, a header-only C++11 addition next to the C API, wraps
  the checked scope token in a move-only RAII class (`sp::Scope`) so a region
  is left however its scope exits -- return, break, or an exception -- via
  `sp_profiler_scope_begin()`/`sp_scope_end()` under the hood, against either
  an explicit `sp_profiler *` or (via `sp_default_profiler()`, also new) the
  default context.
- Structured status codes (`sp_status`, `sp_error_string()`) and introspection
  (`sp_profiler_last_error()`, `sp_profiler_output_path()`,
  `sp_profiler_is_active()`, `sp_profiler_num_regions()`,
  `sp_profiler_region_name()`) replace the previous single `SP_INVALID_REGION`
  sentinel for the explicit-context API.
- Standalone HTML reports now embed the version-matched
  `@scope-profiler/plotly` builders and render the report's plot-data JSON with
  them. The Plotly runtime remains embedded too, so reports share the web
  package's figure definitions without requiring a CDN or network access.
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
- The HTML report's overview names the region with the most _exclusive_ time
  as the hot spot, rather than the largest inclusive total. An enclosing
  region's total is mostly its children's, so the old line reliably named a
  wrapper: a run whose overview read "`setup: total` dominates the recorded
  time: 0.126519 s" now reads "`setup: derham` dominates the recorded time:
  0.0524961 s in the region itself, excluding nested regions", with a second
  line noting that `setup: total` has the largest total but spends 0.126152 s
  of it inside nested regions. The percentage is now a share of the time
  attributed to regions, which sums to a whole, instead of a share of a total
  that counted nested time once per level.
- `scope-profiler report --charts-cdn` loads Plotly from `https://cdn.plot.ly`
  instead of embedding it. The embedded runtime is ~4.7 MB of every report
  whatever the profile size, so this takes a typical report from ~4.8 MB to
  ~80 kB, for one that needs a network connection to draw its charts. The
  version is pinned to the one that would have been embedded, and a report
  that cannot reach the CDN replaces its charts with a message saying so.
  Embedding remains the default: a report copied off a compute node still
  opens with no network.
- The HTML report has a region search box, filtering the statistics tables and
  every embedded chart from one control. Terms are comma-separated and matched
  case-insensitively anywhere in a region name, `^` anchors a term to the start
  of it, and an empty box shows everything -- the syntax the profiling-data
  site already uses. The charts are redrawn through the builders' own
  `filterRegion` option and `Plotly.react`, so filtering never introduces a
  second notion of what a region match is.
- HTML reports now add duration-over-time, rank-imbalance, compact call-graph,
  and (when recorded) LIKWID metric charts, plus complete conditional LIKWID
  and Linux perf-event tables. A contents bar, links from overview findings to
  their region rows, and expand/collapse-all chart controls make larger reports
  easier to navigate.
- Region selections are now linked across HTML-report tables and charts.
  Clicking a table row or Plotly point highlights the same region everywhere;
  chart clicks also open and scroll to its detailed statistics, with a clear
  action beside the region filter.
- `@scope-profiler/plotly` is tested against JSON the exporter really wrote:
  `packages/plotly/test/fixtures/` holds a two-run export of every plot kind,
  regenerated by `generate_fixtures.py`, and a Python test asserts the package
  has a builder for every kind the export can write.
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
  the `thread_table` / `task_table` groups to the HDF5 layout, within schema 2. They are written only by a run that tracked threads, and a file without
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
- `@scope-profiler/plotly` no longer merges two runs into one series. The rank
  heatmap keyed its cells by rank alone, so a second file's values silently
  overwrote the first's -- a 30-row, two-run export drew 15 cells and showed
  one run's numbers under both. Imbalance, the duration time series and the
  histogram likewise pooled runs into a single trace, which for imbalance drew
  a line that revisited every rank. All four now key their series by file as
  well, label it, and tell runs apart by marker symbol or bar pattern while
  the region keeps its colour.
- `@scope-profiler/plotly` builds large figures in one grouping pass instead of
  filtering the whole row array once per series. Building a 200k-interval,
  400-region gantt went from 770 ms to 58 ms.
- The HTML report's "Timeline" chart is a Gantt chart again. The browser
  builder gave each rank one lane and drew every region of that rank onto it,
  so a nested profile collapsed into a single striped row in which the
  outermost region covered everything inside it -- for a single-rank run, one
  bar. It now draws a lane per region and rank, bottom-up so the enclosing
  region is the bottom lane -- the same layout `scope-profiler plot gantt` has
  always drawn -- with `{ laneBy: "rank" }` for the old one-row-per-rank view
  of a flat profile.
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
