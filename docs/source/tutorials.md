# Tutorials & examples

Step-by-step tutorials and self-contained example scripts.

## Example scripts

The `examples/` directory in the repository contains ready-to-run
scripts:

- **`ex_line_profiling.py`** --- demonstrates line-by-line profiling
  with `use_line_profiler=True`.
- **`ex_recursive_profiling.py`** --- profiles nested function calls from
  one decorated entrypoint using `recursive_profile=True`.
- **`ex_cli_profiling.py`** --- an uninstrumented script, profiled with the
  `scope-profiler` CLI command (no decorators or setup calls needed).
- **`benchmark_overhead.py`** --- measures per-call overhead of every
  profiling mode and produces a bar chart.

Run any example with:

```bash
python examples/<script>.py
```

## Notebooks

```{toctree}
:maxdepth: 1
:glob:

tutorials/*
```
