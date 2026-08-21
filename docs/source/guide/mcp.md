

# MCP server for AI coding agents

[MCP](https://modelcontextprotocol.io) (Model Context Protocol) is a
standard way for an AI coding agent – Claude Code, or any other
MCP-capable client – to call tools exposed by a local server.
`scope-profiler-mcp` is such a server: it exposes profiling,
benchmarking and comparison as structured tools, so an agent can inspect
a run’s performance and check whether a code change helped, as part of
its own workflow, without you copy-pasting tables into the chat.

It is a thin adapter, not a second implementation: every tool calls
straight into the same Python API {doc}`plot_cli`, {doc}`/cli` and the
{doc}`hdf5_and_python_api` page describe — `read_h5`, the `region_rows`/
`diff_rows` used by `inspect`/`diff`, `ProfileManager.run_script`, and
the `plotting_scripts` functions behind `plot`. Nothing about profiling,
HDF5 parsing, or comparing runs is reimplemented for MCP.

``` text
Claude Code
    |  MCP (stdio)
    v
scope-profiler-mcp
    |  the same Python API the CLI uses
    v
scope-profiler
    |
    v
inspect / benchmark / compare -> structured JSON
```

This page covers using the MCP server as it stands today: an agent can
read profiling data and judge whether a change helped. It does **not**
modify your code, run an optimization search, or make commits – an agent
using these tools still has to decide what to try next and make the edit
itself, the same as it would from a `git diff` and your instructions.

## Installing

``` bash
pip install "scope-profiler[mcp]"
```

This is a separate extra from the base install:
`pip install scope-profiler` alone does not pull in the `mcp` package,
so normal profiling and the `scope-profiler` CLI are unaffected either
way. `plot_profile` (see below) additionally needs the `pproc` extra,
exactly as `scope-profiler plot` does:

``` bash
pip install "scope-profiler[mcp,pproc]"
```

## Starting the server

``` bash
scope-profiler-mcp
```

or equivalently:

``` bash
python -m scope_profiler.mcp_server
```

This speaks MCP over stdio and is meant to be launched by an MCP client
(Claude Code, another agent, or `mcp dev`/an MCP inspector for manual
testing) – not run directly in a terminal you intend to type into.

## Configuring Claude Code

Add it as an MCP server, either via the CLI:

``` bash
claude mcp add scope-profiler -- scope-profiler-mcp
```

or by adding it directly to your project’s `.mcp.json` (or Claude Code’s
global MCP config):

``` json
{
  "mcpServers": {
    "scope-profiler": {
      "command": "scope-profiler-mcp"
    }
  }
}
```

If `scope-profiler-mcp` is not on `PATH` (e.g. installed into a
virtualenv Claude Code does not activate), point `command` at the
interpreter instead:

``` json
{
  "mcpServers": {
    "scope-profiler": {
      "command": "/path/to/venv/bin/python",
      "args": ["-m", "scope_profiler.mcp_server"]
    }
  }
}
```

Once configured, restart Claude Code (or run `/mcp` to check connection
status) and the four tools below become available for it to call.

## Available tools

### `inspect_profile`

Structured version of `scope-profiler inspect`: runtime, rank count,
region statistics (sorted, capped at `top_n` – raise it or pass `0` for
no limit), metadata grouped the same way (run info, hardware,
MPI/OpenMP, Slurm), and a LIKWID hardware-counter summary when the run
recorded one. Large, rarely-useful metadata (raw environment variables,
unrecognized fields) is collapsed to a count by default;
`full_metadata=True` returns everything.

### `compare_profiles`

Structured version of `scope-profiler diff`, plus the whole-run
comparison `diff` does not compute: given a baseline and a candidate
file, it returns `overall.faster` (bool), `overall.speedup`,
`overall.relative_change_pct` and `overall.absolute_diff_seconds`,
alongside the per-region deltas and a `regressions`/`improvements` split
by `threshold_pct`. The arithmetic is done here, in Python – an agent
should never need to subtract two numbers from two separate
`inspect_profile` calls to answer “did this get faster?”.

### `run_profile`

Runs a script under the profiler (equivalent to `scope-profiler run`)
and returns an `inspect_profile`-style summary of the result. The script
runs in its own subprocess, with a timeout, so it cannot hang or crash
the MCP server; a non-zero exit, a timeout, or a missing script all come
back as a clear tool error rather than a partial result. `script_args`
are passed through as a plain argument list (no shell involved),
matching how `ProfileManager.run_script` already works.

### `plot_profile` (optional)

Renders one figure (`gantt`, `flame`, `durations`, `timeseries`,
`speedup`) via the same functions `scope-profiler plot` uses, and
returns the path it wrote. Most agent workflows should prefer the
numbers from `inspect_profile`/`compare_profiles`; reach for this only
when a human is going to look at the image afterwards (e.g. attaching it
to a PR). Requires the `pproc` extra.

## Example: an agent judging a code change

A typical loop, in words, for “is this function faster after my
change?”:

1.  The agent calls `run_profile` on a benchmark script before making
    any change, and keeps the returned `file_path` (or passes
    `output_path` itself to control where it lands) as the baseline.
2.  It edits the code.
3.  It calls `run_profile` again on the same script for the candidate
    run.
4.  It calls `compare_profiles` with the two file paths and reads
    `overall.faster` and `overall.speedup` directly, plus `regressions`/
    `improvements` to see which regions moved.
5.  It reports the result (and, optionally, calls `plot_profile` to
    attach a Gantt or duration chart) – and decides, itself, whether to
    keep the change, try something else, or ask you.

Step 5 is where this stops: the agent reasons about the structured
numbers you get from steps 1-4, the same as it would reason about a
`git diff`, but scope-profiler does not close the loop by editing code
or committing on its own.

## Testing without a live client

The server’s tools are plain Python functions under
`scope_profiler.mcp_server.tools`, importable and testable without the
`mcp` package at all:

``` python
from scope_profiler.mcp_server.tools import inspect_profile

print(inspect_profile("profiling_data.h5", top_n=5))
```

`scope_profiler.mcp_server.server.create_server()` builds the full MCP
server object (tool registration, JSON schemas) without starting a
transport, useful for scripting against it directly:

``` python
import asyncio
from scope_profiler.mcp_server.server import create_server

server = create_server()
print(asyncio.run(server.list_tools()))
```
