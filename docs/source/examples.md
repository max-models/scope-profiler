

# Examples and workflows

The repository includes small scripts for specific profiling tasks and a
repeatable AI optimization workflow.

## Profiling examples

- [`ex_cli_profiling.py`](https://github.com/max-models/scope-profiler/blob/devel/examples/ex_cli_profiling.py)
  — profile an unmodified script with the CLI.
- [`ex_line_profiling.py`](https://github.com/max-models/scope-profiler/blob/devel/examples/ex_line_profiling.py)
  — collect line-by-line timings.
- [`ex_recursive_profiling.py`](https://github.com/max-models/scope-profiler/blob/devel/examples/ex_recursive_profiling.py)
  — profile nested calls.
- [`ex_region_source.py`](https://github.com/max-models/scope-profiler/blob/devel/examples/ex_region_source.py)
  — retain and inspect region source code.
- [`benchmark_overhead.py`](https://github.com/max-models/scope-profiler/blob/devel/examples/benchmark_overhead.py)
  — measure instrumentation overhead.

Run an example with:

``` bash
python examples/<script>.py
```

## AI optimization workflow

The [agent workflow
example](https://github.com/max-models/scope-profiler/tree/devel/examples/agent_workflow)
demonstrates measure → edit → re-measure → correctness-check using the
CLI or MCP tools. The runnable repository workload is configured in
[`benchmarks/sensor.toml`](https://github.com/max-models/scope-profiler/blob/devel/benchmarks/sensor.toml).

See
[`AGENTS.md`](https://github.com/max-models/scope-profiler/blob/devel/AGENTS.md)
for the instructions used by Codex and other coding agents.
