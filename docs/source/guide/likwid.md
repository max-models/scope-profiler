# LIKWID hardware counters

scope-profiler can wrap every profiled region in a
[LIKWID](https://github.com/RRZE-HPC/likwid) marker region, using the
[pylikwid](https://github.com/RRZE-HPC/pylikwid) bindings. At `finalize()` the
markers are closed, every marker region of the run is read back, and the raw
hardware events together with LIKWID's derived metrics are stored in the HDF5
file next to the timings.

## Installation

```bash
pip install "scope-profiler[likwid]"
```

`pylikwid` builds against an existing LIKWID installation, so install (or
`module load`) LIKWID first. If the LIKWID module does not put `liblikwid.so`
on `LD_LIBRARY_PATH` --- a common cluster setup --- scope-profiler finds and
loads it from `LIKWID_HOME` (or the prefix of `likwid-perfctr` on `PATH`)
rather than failing the import. See {doc}`../installation`.

## Enabling LIKWID

Pass `use_likwid=True` to `setup()`, and start the process under LIKWID's
marker mode:

```python
from scope_profiler import ProfileManager

ProfileManager.setup(use_likwid=True, file_path="profiling_data.h5")

with ProfileManager.profile_region("solve"):
    ...

ProfileManager.finalize()
```

```bash
likwid-perfctr -C 0 -g FLOPS_DP -m python script.py

# across MPI ranks
likwid-mpirun -n 2 -g FLOPS_DP -mpi openmpi -marker python script.py
```

The `-m` / `-marker` flag is what matters: without it LIKWID sets no
environment, the marker calls become no-ops, and the run records timings but
no counters. `-g` selects the event group (`likwid-perfctr -a` lists the ones
your CPU supports).

## What gets collected

Both of LIKWID's APIs are used, and they complement each other:

- The **marker API** (`markergetregion`) is sampled while the markers are
  still open. It always works, but reports only raw numbers for the calling
  thread.
- The **full API** re-registers the event sets from `LIKWID_EVENTS` after
  `markerclose()` has written the marker file, then reads that file back with
  `markerreadfile()`. This yields *every* region of the run, for every
  hardware thread, with event **names**, per-thread call counts and LIKWID's
  derived metrics (`Clock [MHz]`, `CPI`, `DP [MFLOP/s]`, `Energy [J]`, ...).

The full API is preferred; the marker-API snapshot is the fallback used when
the performance counters cannot be re-opened. A region records which one it
came from in its `source` attribute (`"full_api"` or `"marker_api"`).

Counter collection never fails a run: if LIKWID cannot be read, `finalize()`
still writes the timing data.

## HDF5 layout

Each rank's counters sit beside its regions:

```text
profiling_data.h5
├── metadata/
└── rank0/
    ├── regions/                     (timestamps, as usual)
    └── likwid/                      (attrs: the LIKWID_* environment, num_regions)
        └── regions/
            └── solve/
                ├── (attrs)  tag, group_id, group_name, source,
                │            event_names, metric_names
                ├── cpus         (int64, nthreads)   hardware threads involved
                ├── times        (float64, nthreads) LIKWID runtime in seconds
                ├── call_counts  (int64, nthreads)   times the region was entered
                ├── events       (float64, nevents x nthreads)  raw counters
                └── metrics      (float64, nmetrics x nthreads) derived metrics
```

`events[e, t]` and `metrics[m, t]` share the thread axis with `times[t]` and
`cpus[t]`. Because `/` separates HDF5 groups, a `/` in a region name is stored
as `|` in the group name --- the `tag` attribute always holds the real name.

## Reading the counters back

```python
from scope_profiler import ProfilingH5Reader

reader = ProfilingH5Reader("profiling_data.h5")
reader.has_likwid          # False for a run without counters
reader.likwid_ranks        # ranks that recorded counters

# Everything, keyed by rank then region tag
for rank, regions in reader.get_likwid_regions().items():
    for tag, result in regions.items():
        print(rank, tag, result.group_name, result.call_counts[0])
        for name, values in zip(result.event_names, result.events):
            print("  ", name, values[0])
        for name, values in zip(result.metric_names, result.metrics):
            print("  ", name, values[0])

# One region on one rank
solve = reader.get_likwid_region("solve", rank=0)
solve.metric_names         # ['Runtime (RDTSC) [s]', 'Clock [MHz]', 'CPI', ...]
solve.metrics[2, 0]        # CPI on the first hardware thread
```

`print_likwid_summary()` dumps all of it as text, and `likwid_to_dataframe()`
returns a tidy pandas table with one row per (rank, region, hardware thread)
and one column per event and metric:

```python
df = reader.likwid_to_dataframe()
df.groupby("region")["CPI"].mean()
```

## Profiling modes

`use_likwid` combines with `time_trace`; see {doc}`configuration` for the full
dispatch table. `time_trace=True, use_likwid=True` (the default once LIKWID is
on) records timestamps *and* counters; `time_trace=False, use_likwid=True`
records counters only.

## Full example

A runnable end-to-end example lives in
[`examples/ex_likwid.py`](https://github.com/max-models/scope-profiler/blob/devel/examples/ex_likwid.py):

```bash
likwid-perfctr -C 0 -g FLOPS_DP -m python examples/ex_likwid.py
```
