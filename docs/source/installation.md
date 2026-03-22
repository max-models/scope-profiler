# Installation

## Basic install

Install the latest release from [PyPI](https://pypi.org/project/scope-profiler/):

```bash
pip install scope-profiler
```

This pulls in the only hard dependencies: **numpy** and **h5py**.

## Optional extras

scope-profiler ships several optional dependency groups that you can install
with the bracket syntax:

| Extra           | Install command                               | What it adds                                                                         |
| --------------- | --------------------------------------------- | ------------------------------------------------------------------------------------ |
| `line-profiler` | `pip install "scope-profiler[line-profiler]"` | Line-by-line profiling via [line_profiler](https://github.com/pyutils/line_profiler) |
| `mpi`           | `pip install "scope-profiler[mpi]"`           | MPI support via [mpi4py](https://mpi4py.readthedocs.io/)                             |
| `dev`           | `pip install "scope-profiler[dev]"`           | All of the above plus linting, formatting, and docs tools                            |

## Development install

Clone the repository and install in editable mode with all development
dependencies:

```bash
git clone https://github.com/max-models/scope-profiler.git
cd scope-profiler
pip install -e ".[dev]"
```

### LIKWID (optional)

LIKWID hardware counter support requires the
[LIKWID](https://github.com/RRZE-HPC/likwid) library and the
[pylikwid](https://github.com/RRZE-HPC/pylikwid) Python bindings to be
installed on the system. See the
[LIKWID documentation](https://github.com/RRZE-HPC/likwid/wiki) for
build instructions.

## Verify installation

```python
>>> from scope_profiler import ProfileManager
>>> ProfileManager.setup()
>>> with ProfileManager.profile_region("test"):
...     pass
>>> ProfileManager.finalize()
Region: test
  Total Calls : 1
  ...
```
