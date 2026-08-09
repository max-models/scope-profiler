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

Counter collection runs at `finalize()`, on top of a job that has already
finished. Losing a long run's output because a counter read went wrong would
be absurd, so there are three sources, tried richest first. Each region
records which one produced it in its `source` attribute.

| `source`        | How                                                       | What you get                                        |
| --------------- | --------------------------------------------------------- | --------------------------------------------------- |
| `full_api`      | perfmon re-init + `markerreadfile()`, **in a subprocess**  | Everything: event names, counter registers, metrics |
| `marker_file`   | LIKWID's marker file parsed directly, no LIKWID calls      | Real values; positional event names, no metrics     |
| `marker_api`    | `markergetregion()` before the markers were closed         | Calling thread only; no names, no metrics           |

The full API is the only path that can name events and compute derived
metrics (`Clock [MHz]`, `CPI`, `DP [MFLOP/s]`, `Energy [J]`, ...), because
those live in LIKWID's group definitions rather than in the marker file.

It is also the only step that can bring the interpreter down instead of
raising: re-initializing perfmon has been observed to **segfault** on hosts
that cannot really count --- a virtualized runner with an unreadable TSC, or
one where HyperThreading disables the PMCs. It therefore runs in a child
process. If that child dies, the parent notices, falls back to parsing the
marker file, and the run still ends up with real call counts, runtimes and
counter values in the HDF5 file. (If the child crashes *after* writing its
results, they are kept --- a complete JSON document is proof the work
finished.)

So counter collection never fails a run, and rarely degrades one.

```{note}
Whether the counters hold meaningful numbers is a property of the machine, not
of the profiler. A virtualized CPU may report structurally valid zeros for
every event; LIKWID prints `WARN: Counter PMC0 is only available with
deactivated HyperThreading` and similar in that case. The call counts are
LIKWID's own bookkeeping and stay exact regardless.
```

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

## From the command line

`scope-profiler pproc file.h5 --summary` prints the region statistics and, if
the run recorded counters, one LIKWID table per rank and event group:

```text
LIKWID counters (rank 0, group MEM_DP)
  counter                                        main     matmul  memory_bound
  ----------------------------------------------------------------------------
  call count                                        1          3             1
  runtime [s]                                0.161376   0.110618     0.0301046
  ----------------------------------------------------------------------------
  Events
  INSTR_RETIRED_ANY                          70478340   65807930       4541968
  CPU_CLK_UNHALTED_CORE                      78814130   63363840      14723860
  ...
  CAS_COUNT_RD:MBOX0C0                        1220547    1002233        198776
  CAS_COUNT_RD:MBOX1C0                        1208866     998410        197254
  ...
  ----------------------------------------------------------------------------
  Metrics
  Runtime (RDTSC) [s]                        0.161376   0.110618     0.0301046
  Clock [MHz]                                 2376.17    2376.37       2374.59
  CPI                                        0.796200   0.717700       1.78210
  Memory bandwidth [MBytes/s]                 9799.37    12387.5       11098.8
  Operational intensity [FLOP/Byte]            0.1335     0.2300        0.0155
  ----------------------------------------------------------------------------
```

Regions are the columns and counters the rows, because a run usually has a
few regions and a few dozen counters. Columns are ordered costliest-first, to
match the region table above it. One table is emitted per event group, since
a group is what fixes which events and metrics exist. `--include`, `--exclude`
and `--ranks` filter this table too.

Note the `CAS_COUNT_RD:MBOX0C0` style names: see
[repeated events](#repeated-events) below.

(repeated-events)=
## Repeated events

An event name is not a unique key. A group such as `MEM_DP` programs the same
event on one counter per memory channel, so `CAS_COUNT_RD` legitimately
appears eight times on an eight-channel socket (channels with no DIMM read
zero). LIKWID's derived `Memory bandwidth` is the sum over all of them.

`LikwidRegionResult.event_labels` is therefore what you should key anything by:
names that occur once are returned unchanged, repeated ones get the hardware
counter appended (`CAS_COUNT_RD:MBOX0C0`, ...). `event_names` and
`counter_names` hold the raw pair if you need them.

```python
result = results.get_likwid_region("solve")
dict(zip(result.event_labels, result.events[:, 0]))   # safe
dict(zip(result.event_names, result.events[:, 0]))    # loses all but one channel
```

## Reading the counters back

```python
from scope_profiler import read_h5

results = read_h5("profiling_data.h5")
results.has_likwid          # False for a run without counters
results.likwid_ranks        # ranks that recorded counters

# Everything, keyed by rank then region tag
for rank, regions in results.get_likwid_regions().items():
    for tag, result in regions.items():
        print(rank, tag, result.group_name, result.call_counts[0])
        for name, values in zip(result.event_names, result.events):
            print("  ", name, values[0])
        for name, values in zip(result.metric_names, result.metrics):
            print("  ", name, values[0])

# One region on one rank
solve = results.get_likwid_region("solve", rank=0)
solve.metric_names         # ['Runtime (RDTSC) [s]', 'Clock [MHz]', 'CPI', ...]
solve.metrics[2, 0]        # CPI on the first hardware thread
```

`print_likwid_summary()` dumps all of it as text, and `likwid_to_dataframe()`
returns a tidy pandas table with one row per (rank, region, hardware thread)
and one column per event and metric:

```python
df = results.likwid_to_dataframe()
df.groupby("region")["CPI"].mean()
```

## Profiling modes

`use_likwid=True` selects `FullProfileRegion`, which records timestamps *and*
counters; see {doc}`configuration` for the full dispatch table. There is no
counters-only mode --- timestamps are cheap next to the marker calls, and
having them makes the counters far easier to interpret.

## Full example

A runnable end-to-end example lives in
[`examples/ex_likwid.py`](https://github.com/max-models/scope-profiler/blob/devel/examples/ex_likwid.py):

```bash
likwid-perfctr -C 0 -g FLOPS_DP -m python examples/ex_likwid.py
```
