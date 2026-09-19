<!-- Generated README.md is rendered from this file by docs/render_markdown.py. -->

# scope-profiler

Profile Python code regions—and optionally C, Fortran, MPI, NVTX, and
[LIKWID](https://github.com/RRZE-HPC/likwid)—with one consistent API and
HDF5 output format.

```bash
pip install scope-profiler
```

## Quick start

```python
import scope_profiler as sp

with sp.session():
    @sp.profile("main")
    def main():
        with sp.region("work"):
            sum(range(100))  # replace with the code you want to measure

    main()
# writes profiling_data.h5 and prints a summary
```

`session`, `region`, `profile`, `setup` and `finalize` are the everyday
calls, importable from the package root. They are `ProfileManager` class
methods acting on the process-wide default manager, so the two spellings
are interchangeable:

```python
from scope_profiler import ProfileManager

with ProfileManager.session():
    with ProfileManager.region("work"):
        ...
```

For a script that already has regions, the CLI owns the profiling
lifecycle:

```python
import scope_profiler as sp

@sp.profile("main")
def main():
    with sp.region("iteration"):
        work()

main()
```

Run it through Scope Profiler to collect those regions and finalize the
output:

```bash
scope-profiler run app.py
```

Running the same file with `python app.py` leaves profiling disabled and
creates no output file. Existing `session()`, `setup()`, and
`finalize()` workflows remain available when application code needs
direct lifecycle control.

See the
[quickstart](https://max-models.github.io/scope-profiler/quickstart.html)
and [profiling modes
guide](https://max-models.github.io/scope-profiler/guide/modes.html) for
recursive tracing, tags, concurrency, output formats, and advanced APIs.

## Profile a pytest suite

The installed package provides an opt-in pytest plugin. It records one
region for every selected test, named from its pytest node id, so the
normal post-processing commands can show which tests consume the suite’s
time:

```bash
pytest --scope-profile --scope-profile-out pytest-profile.h5
scope-profiler inspect pytest-profile.h5 --regions-only --sort total
scope-profiler plot durations pytest-profile.h5 -o pytest-plots
```

By default it measures only each test’s `call` phase. Include fixture
setup and teardown when those are relevant to the investigation:

```bash
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

See the [profiling modes
guide](https://max-models.github.io/scope-profiler/guide/modes.html) for
output formats, filtering, concurrency, aggregation, pause/resume, and
sampling.

## In a notebook

Install the notebook extra and see the [notebook magics
guide](https://max-models.github.io/scope-profiler/guide/notebook_magics.html)
for cell profiling, recursive tracing, comparisons, and exports:

```bash
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

```bash
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

```bash
pip install -e '.[dev]'
pytest
```

See
[AGENTS.md](https://github.com/max-models/scope-profiler/blob/devel/AGENTS.md)
for the measured benchmark workflow used when optimizing this project.
