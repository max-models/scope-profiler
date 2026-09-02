

<!-- Generated README.md is rendered from this file by docs/render_markdown.py. -->

# scope-profiler

Profile Python code regions—and optionally C, Fortran, MPI, NVTX, and
[LIKWID](https://github.com/RRZE-HPC/likwid)—with one consistent API and
HDF5 output format.

``` bash
pip install scope-profiler
```

## Quick start

``` python
from scope_profiler import ProfileManager

with ProfileManager.session():
    @ProfileManager.profile("main")
    def main():
        with ProfileManager.profile_region("work"):
            sum(range(100))  # replace with the code you want to measure

    main()
# writes profiling_data.h5 and prints a summary
```

Create independent managers when two profiling sessions need to coexist.
Each manager records only calls made through that manager and writes its
own output:

``` python
compute_profiler = ProfileManager()
io_profiler = ProfileManager()

with compute_profiler.session(file_path="compute.h5", verbose=False):
    with io_profiler.session(file_path="io.h5", verbose=False):
        with compute_profiler.profile_region("solve"):
            solve()
        with io_profiler.profile_region("checkpoint"):
            write_checkpoint()
```

These are independent nested sessions. LIKWID’s marker state is
process-global, so only one overlapping session may use
`use_likwid=True`.

Concurrency is a separate switch. `track_threads=True` gives every
thread its own buffers and its own lane in the reconstructed call graph,
and reports each thread’s name, lifetime and CPU time;
`track_async=True` does the same for asyncio tasks and greenlets, and
splits every call into the time its task held the thread and the time it
spent awaiting:

``` python
with ProfileManager.session(track_async=True, return_results=True) as run:
    asyncio.run(main())

fetch = run.results["fetch"][0]
fetch.durations - fetch.await_times     # time actually spent running
for task in run.results.tasks[0]:
    print(task.name, task.running_time, task.awaiting_time)
```

You can also profile a script without changing its source:

``` bash
scope-profiler run my_script.py
scope-profiler inspect profiling_data.h5
scope-profiler plot default profiling_data.h5 -o figures
scope-profiler report profiling_data.h5 -o report.html
```

## Profile a pytest suite

The installed package provides an opt-in pytest plugin. It records one
region for every selected test, named from its pytest node id, so the
normal post-processing commands can show which tests consume the suite’s
time:

``` bash
pytest --scope-profile --scope-profile-out pytest-profile.h5
scope-profiler inspect pytest-profile.h5 --regions-only --sort total
scope-profiler plot durations pytest-profile.h5 -o pytest-plots
```

By default it measures only each test’s `call` phase. Include fixture
setup and teardown when those are relevant to the investigation:

``` bash
pytest --scope-profile --scope-profile-phases=all
```

The plugin uses its own profiling manager, so tests that call
`ProfileManager.setup()` or `ProfileManager.finalize()` remain isolated.
The result measures complete test phases; it does not recursively trace
every function pytest calls. Existing application-level Scope Profiler
regions are therefore best collected in a dedicated application run when
detailed function-level attribution is needed. With `pytest-xdist`, the
plugin cannot yet be used: its workers must not write the same HDF5
file.

The extension of `-o` picks the output format. HDF5 is the default, and
stays the better choice for a long run — it is read back column by
column rather than whole. JSON holds exactly the same data — every call,
in integer nanoseconds — for anything that would rather not open an HDF5
file:

``` bash
scope-profiler run -o profile.json my_script.py     # or .json.gz
scope-profiler run -o report.html my_script.py      # rendered report
scope-profiler export json profiling_data.h5 -o exports  # convert an existing run
scope-profiler inspect profile.json                 # read one back anywhere
```

Profiling can be suspended around setup, I/O, or other phases that
should not appear in the trace. Pause at scope boundaries and resume
when measurement is needed again:

``` python
ProfileManager.pause()
simulation.prepare_output()
ProfileManager.resume()
```

`pause()` and `resume()` are safe to call repeatedly. Pausing while a
profiled scope is open raises an error, so a recorded interval can never
silently span the paused period.

For time-stepping simulations, `sample_every()` provides the same
control with an explicit timestep number:

``` python
with ProfileManager.sample_every(10) as profile_step:
    for timestep in range(num_steps):
        with profile_step(timestep):
            simulation.step()
```

The equivalent fully manual form is useful when the simulation has
additional conditions around profiling:

``` python
for timestep in range(num_steps):
    if timestep % 10 == 0:
        ProfileManager.resume()
    else:
        ProfileManager.pause()

    with ProfileManager.profile_region("simulation.step"):
        simulation.step()
```

Here only timesteps `0`, `10`, `20`, and so on are recorded. Call
`ProfileManager.setup()` before the loop; the initial state is enabled,
so the first `resume()` is optional but makes the intent explicit.

Reports embed interactive timeline and duration charts when the optional
post-processing dependencies are installed
(`pip install "scope-profiler[pproc]"`).

## Example output

The plotting tools include duration summaries and timelines for finding
expensive regions:

For dense traces, the Gantt view supports time windows, duration
filtering, call coalescing, and call-depth collapsing. A binned
occupancy heatmap avoids drawing every short event:

``` bash
scope-profiler plot gantt profiling_data.h5 -o figures \
  --min-duration 0.001 --aggregate-calls 25 --collapse-depth 2
scope-profiler plot density profiling_data.h5 -o figures \
  --bins 200 --min-duration 0.0001 --start-time 0 --end-time 10
```

Use `--aggregation-mode` with `scope-profiler run` when only aggregate
timing statistics are needed and the per-call timeline should not be
recorded.

![Duration
summary](https://raw.githubusercontent.com/max-models/scope-profiler/refs/heads/devel/figures/durations_plot.png)

![Gantt
chart](https://raw.githubusercontent.com/max-models/scope-profiler/refs/heads/devel/figures/gantt_plot.png)

The overhead benchmark measures the cost of each instrumentation mode:

``` bash
python examples/benchmark_overhead.py
```

![Profiling overhead by region
type](https://raw.githubusercontent.com/max-models/scope-profiler/refs/heads/devel/figures/benchmark_overhead.png)

## In a notebook

`%load_ext scope_profiler.ipython_magics` adds magics for the
measure/compare loop, so a notebook needs no `session()` boilerplate:

``` python
%%scope_recursive
result = solve(problem)     # every call recorded, nothing instrumented
```

``` python
%scope_compare baseline candidate
```

`%%scope` times a cell as one region, `%%scope_line` breaks a function
down by line, `%%scope_agg` handles regions entered millions of times,
and `%scope_load` pulls in an HDF5 run from an MPI job to compare
against. See the [notebook magics
guide](https://max-models.github.io/scope-profiler/guide/notebook_magics.html).

``` bash
pip install "scope-profiler[notebook]"
```

## Documentation

- [Installation](https://max-models.github.io/scope-profiler/installation.html)
- [Quick
  start](https://max-models.github.io/scope-profiler/quickstart.html)
- [Python API and
  post-processing](https://max-models.github.io/scope-profiler/guide/hdf5_and_python_api.html)
- [CLI reference](https://max-models.github.io/scope-profiler/cli.html)
- [Plotly figures for the
  web](https://max-models.github.io/scope-profiler/guide/plotly_package.html)
- [Configuration and profiling
  regions](https://max-models.github.io/scope-profiler/guide/configuration.html)
- [MPI](https://max-models.github.io/scope-profiler/guide/mpi.html),
  [C](https://max-models.github.io/scope-profiler/guide/c.html), and
  [Fortran](https://max-models.github.io/scope-profiler/guide/fortran.html)
- [LIKWID](https://max-models.github.io/scope-profiler/guide/likwid.html),
  [line
  profiling](https://max-models.github.io/scope-profiler/guide/line_profiler.html),
  and [MCP](https://max-models.github.io/scope-profiler/guide/mcp.html)
- [Jupyter/IPython
  magics](https://max-models.github.io/scope-profiler/guide/notebook_magics.html)
- [Tutorial
  notebooks](https://max-models.github.io/scope-profiler/tutorials.html)
- [Examples](https://github.com/max-models/scope-profiler/tree/devel/examples)

### Build the documentation locally

The hosted documentation is built with Sphinx from the `.qmd` sources.
Install [Quarto](https://quarto.org/docs/get-started/) and Pandoc first,
then create a development environment and run the docs target from the
repository root:

``` bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[docs]"
make -C docs html
```

This refreshes the generated command output and Markdown sources before
building the HTML site. Open `docs/build/html/index.html` locally, or
serve that directory with any static-file server. The GitHub Pages
workflow uses the same `make -C docs html` command.

## Development

``` bash
pip install -e '.[dev]'
pytest
```

See
[AGENTS.md](https://github.com/max-models/scope-profiler/blob/devel/AGENTS.md)
for the measured benchmark workflow used when optimizing this project.
