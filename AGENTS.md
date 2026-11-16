# Working on scope-profiler as a coding agent

Instructions for Codex, Claude Code and other coding agents working in this repository, and for
agents that use scope-profiler to optimize other code.

## Development

```bash
pip install -e '.[dev]'     # add ,mcp for the MCP server
pre-commit install          # ruff format, ruff check, ssort, nbstripout, ...
pytest                      # the suite; CI enforces the coverage floor in pyproject.toml
```

- Changes go into `devel` through pull requests; add an entry to `CHANGELOG.md` under
  "Unreleased".
- The CLI reference is `docs/src/content/docs/reference/cli.mdx`; the guides are Jupyter
  notebooks in `notebooks/guides/`, executed on every docs build. `README.md` is rendered from
  `README.qmd` (`make markdown`): edit the `.qmd`.
- The MCP tools in `src/scope_profiler/mcp_server/tools.py` are thin wrappers over the Python
  API (`summary`, `diff`, `inspection`, `benchmark`); put new profiling, comparison or HDF5 logic
  in those modules, not in the MCP layer.

## The measured optimization loop

Keep a change only when it is both faster and still correct, and measure both instead of
estimating them:

1. **Measure a baseline.** With a benchmark config (TOML: the script, the number of runs and
   warmups, the correctness command, the threshold), run

   ```bash
   scope-profiler benchmark run benchmarks/sensor.toml --label baseline --json
   ```

   It runs the warmups, the correctness check and the measured runs, and writes a manifest with
   every profile and the timing statistics to `<output_dir>/<name>/<label>/benchmark.json`,
   `output_dir` being relative to the config file (here
   `benchmarks/.scope-profiler/sensor/baseline/benchmark.json`; the JSON gives it as
   `manifest_path`).
   Find where the time goes with `scope-profiler inspect <profile>.h5 --regions-only --sort total`
   before changing code.
2. **Change the code**, one idea at a time.
3. **Measure the candidate**: the same command with `--label candidate`.
4. **Judge**:

   ```bash
   scope-profiler benchmark compare <baseline>/benchmark.json <candidate>/benchmark.json --json
   ```

   Keep the change only when `decision` is `"keep"`: faster than the baseline by at least
   `threshold_pct` (on the median of the runs) and the correctness command passed. Otherwise
   revert it. Use `scope-profiler diff` on two profiles to see which regions moved.
5. **Repeat** from step 2, picking the next target from the candidate's profile, not the
   baseline's: the ranking of what is expensive changes after each edit.

Two configs are checked in: `benchmarks/sensor.toml` (the repository's sensor workload, with
`benchmarks/test_sensor_correctness.py` as its gate) and `examples/agent_workflow/benchmark.toml`,
a small worked example of the loop described in `examples/agent_workflow/README.qmd`.

## Through the MCP server

With `pip install "scope-profiler[mcp]"` and the server registered
(`claude mcp add scope-profiler -- scope-profiler-mcp`, see the
[MCP guide](https://max-models.github.io/scope-profiler/guide/mcp/)), the same loop runs as
tool calls: `run_benchmark` and `compare_benchmarks` for steps 1, 3 and 4, `run_profile`,
`inspect_profile` and `compare_profiles` for single profiles. They return structured JSON:
read `overall.faster`, `overall.speedup` and `decision` from them instead of subtracting numbers
from two tables.

## Rules

- Never weaken or skip the correctness gate to make a change pass.
- A single run per side is not enough to trust a change of a few percent: use the benchmark
  runner's repeated runs, and report the spread with the result.
- Do not commit profiles, manifests or `.scope-profiler/` output.
