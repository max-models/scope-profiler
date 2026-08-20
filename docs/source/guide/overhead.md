

# Profiling overhead

scope-profiler is designed for production HPC workloads where
instrumentation must not distort the measurements. This page documents
the per-call overhead of each profiling mode.

## Benchmark

The benchmark script (`examples/benchmark_overhead.py`) times a small
workload function through each profiling mode and subtracts the bare
function-call baseline to isolate the overhead.

``` bash
python examples/benchmark_overhead.py          # save figure
python examples/benchmark_overhead.py --show   # display interactively
```

`qkyibmo /_static/figures/benchmark_overhead.png :alt: Profiling overhead by region type :width: 100%`

## Results summary

| Region type      | Overhead / call |
|------------------|----------------:|
| **Disabled**     |         ~0.1 µs |
| **TimeOnly**     |        ~0.33 µs |
| **LineProfiler** |          ~50 µs |

*(Numbers measured on an Apple M-series CPU; absolute values will vary,
but the relative ordering is stable.)*

## What this means for HPC

The **TimeOnly** mode — the default and most commonly used — adds
roughly **0.33 µs** per instrumented call. In practice:

- A 64×64 matrix multiply takes ~36 µs, so the overhead is **\< 2 %**.
- A 256×256 matrix multiply takes ~780 µs, giving **\< 0.1 %** overhead.
- Typical simulation time steps run for milliseconds or longer, making
  the overhead unmeasurable.

The profiler can also be **fully deactivated** at startup
(`deactivate_profiling=True`) without removing any instrumentation from
the source code. In this mode the overhead drops to ~0.1 µs — barely
above the cost of a bare function call.

## LineProfiler

The `line_profiler` mode is intentionally heavier (~50 µs per call)
because it instruments every source line in the profiled function. It is
meant for **targeted debugging of individual functions**, not for
always-on use in hot loops.

## Where the time goes

A recorded call is two `perf_counter_ns()` reads (~33 ns each from
Python) plus a slot reservation; the rest is the cost of the `with`
statement or the decorator wrapper itself. Those two together are about
half the total and are out of the library’s hands: an empty `with` on a
Python object already costs ~109 ns, and on a C-implemented one ~76 ns.
Nothing touches the filesystem: the timestamps accumulate in a numpy
buffer that doubles when it fills, and the whole buffer is written once,
at `finalize()`. Writing is therefore not part of the per-call cost at
all — `deactivate_file_output` changes what happens at the end of the
run, not what happens in the loop.
