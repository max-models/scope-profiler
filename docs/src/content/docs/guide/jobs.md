---
title: Profiling jobs
---

> **Install for this page:** `pip install scope-profiler`. Nothing extra: the
> job driver only writes and submits shell scripts. The profiled program
> brings its own MPI.

Profiling a code at several rank counts, on a laptop and on a cluster, and
keeping the results comparable over time takes more than a profiler:
job scripts per rank count, batch settings per machine, waiting for the
jobs, collecting the profiles and recording what was run where.
`scope_profiler.jobs` does that for any program:

- a **case** is a command plus the files and environment it needs;
- every **launch** runs it once, with a rank count (and optionally extra
  arguments or environment), as a SLURM job where `sbatch` exists and as a
  local bash script otherwise;
- **finalize** waits for the jobs and **packages** each run as soon as it
  finishes: its profiles, the files a post-processing step wrote, and a
  metadata file;
- with **upload**, every package is committed and pushed to a git
  **results repository**.

## A case in TOML

```toml
[case]
label = "heat_strong_scaling"
name = "Heat equation, strong scaling"
command = ["python", "{case_dir}/heat.py", "--profile-out", "{profile}"]
inputs = ["heat.py"]
env = { OMP_NUM_THREADS = "1" }
metadata = { code = "heat.py" }

[[launch]]
ntasks = [1, 2, 4]
sbatch = { time = "00:05:00" }

[variants.small]
args = ["--cells", "100000"]
[variants.large]
args = ["--cells", "10000000"]
```

```bash
scope-profiler jobs run case.toml                    # every variant
scope-profiler jobs run case.toml --variant small    # one of them
scope-profiler jobs run case.toml --scheduler dry-run  # only write the job scripts
```

The full example is in
[`examples/jobs`](https://github.com/max-models/scope-profiler/tree/devel/examples/jobs).
The keys of `[case]` are the arguments of `ProfilingCase` below, those of
`[[launch]]` the arguments of `ProfilingCase.launch`. Relative paths are
relative to the TOML file. A variant's `args` come before the launch's own,
its `env` and `metadata` are merged into the case's, and its key is appended
to the label.

## A case in Python

When the launches need logic (rank counts that depend on the problem size,
GPU runs on another partition), write the case in Python:

```python
from scope_profiler.jobs import ProfilingCase

case = ProfilingCase(
    label="heat_strong_scaling",
    command=["python", "{case_dir}/heat.py", "--profile-out", "{profile}"],
    inputs=["heat.py"],
    postprocess=["python", "{case_dir}/plot.py", "{run_dir}"],
    env={"OMP_NUM_THREADS": "1"},
)
for ntasks in (1, 2, 4, 8):
    case.launch(ntasks, args=["--cells", "100000"])
case.launch(
    4,
    env={"BACKEND": "gpu"},
    metadata={"backend": "gpu"},
    cluster="mycluster_gpu",
    sbatch={"time": "00:30:00"},
)
case.finalize()
```

### Placeholders

`command`, `postprocess`, `cwd`, the values of `env` (of the case and of a
launch), the `setup` lines (of the case and of its cluster preset) and the
`#SBATCH` values may contain:

| Placeholder | Value |
| --- | --- |
| `{run_dir}` | the run's own directory (the default working directory) |
| `{profile}` | `{run_dir}/profiling_data.h5` |
| `{case_dir}` | the case directory, where `inputs` are copied |
| `{ntasks}`, `{nodes}`, `{launch_id}`, `{label}` | of this launch |
| `{args}` | (a whole list element) the launch's `args` |

`args` are appended to `command` unless it contains `{args}`; `postprocess`
gets them only where it says `{args}`. The job script also exports
`SCOPE_PROFILER_RUN_DIR`, `SCOPE_PROFILER_PROFILE`, `SCOPE_PROFILER_CASE_DIR`,
`SCOPE_PROFILER_NTASKS`, `SCOPE_PROFILER_LAUNCH_ID` and
`SCOPE_PROFILER_LABEL`, for programs that would rather read the environment.
So `env = { APP_LOG = "{run_dir}/app.log" }` gives every run its own log, and
`sbatch = { time = "00:{ntasks}0:00" }` a time limit that grows with the rank
count.

### Preparing a case

`prepare` is a command run once, before the first launch, on the machine the
driver runs on (a login node), e.g. to compile the code:

```toml
[case]
prepare = ["make", "-j8"]     # or a shell line: "pip install -e . && make"
```

It runs in the case directory with the case's `env` (`{case_dir}` and `{label}`
work in it), writes its output to `<case_dir>/prepare.out` and is recorded in
`case_metadata.json` with its duration and exit code. If it fails, the case
stops before any job is submitted, with the end of its output in the error.

With `mpi=True` (the default) the command is prefixed by the launcher:
`srun -n N` under SLURM, `mpirun -n N` (or `mpiexec`) locally, or whatever
`launcher=` says. A local launch that asks for more ranks than the machine
has cores is skipped, so a case written for a 128-core node still runs its
small rank counts on a laptop. The job script activates the virtual
environment the driver runs in (`activate_venv=False` to turn it off) and
then runs the case's `setup` commands.

## Clusters

A cluster preset holds what the jobs of one machine need: `#SBATCH` pragmas,
setup commands (modules), the cores and GPUs of a node (to derive the node
count of a launch) and how to recognise the machine. None are built in;
put them in a TOML file and point `SCOPE_PROFILER_CLUSTERS` at it (or pass
`--cluster-config`):

```toml
[clusters.mycluster_cpu]
cpus_per_node = 128
setup = ["module purge", "module load gcc openmpi python"]
detect = { HOSTNAME = "mycluster" }   # environment variable -> substring
sbatch = { partition = "general", account = "my_project", time = "00:15:00" }
```

```bash
export SCOPE_PROFILER_CLUSTERS=$HOME/clusters.toml
scope-profiler jobs clusters        # lists them, marks the detected one
```

In Python, `register_cluster(name=..., sbatch={...}, ...)` does the same.
The preset of a case is the detected one unless `cluster=` names another;
a launch can pick its own (e.g. a GPU partition), and `sbatch={...}`
overrides single pragmas. Set `SCOPE_PROFILER_JOBS_LOCAL=1` to run locally
even where SLURM exists (a quick check on a login node).

## What is packaged

```text
_profiling_jobs/
    runs/<timestamp>-<commit>/<label>/       job scripts and copied inputs
        run_01/                              one launch: run.out, profiles, results/
    packaged/                                (results/, a clone, with upload)
        <timestamp>-<commit>-<label>/
            case_metadata.json               case, machine, software, list of runs
            inputs/
            run01/
                run_metadata.json            launch, job script, job state, files
                profiling_data.h5
                figure.png                   from <run_dir>/results/
```

`case_metadata.json` records the case (label, description, your
`metadata`), the git commit, branch and dirty state of the profiled
checkout, the Python environment (`pip freeze`), the scheduler and cluster
preset. `run_metadata.json` records the launch (ranks, nodes, args, env),
the job script exactly as submitted, the SLURM job id and final state, the
packaged files and, under `application_metadata`, the `run_metadata.json`
the program itself may have written into its run directory. A run that left
no profile and no results (it crashed, hit its time limit, never started) has
no folder, but `case_metadata.json` still lists it, marked `failed`, with its
job state and the last lines of its output; a catalogue shows it as a failed
run instead of leaving the point out.

The metadata names the machine and the person who ran the case: the submit
host, the checkout and work directories, the virtual environment. With
`redact=True` (or a store marked `public`, below) those are replaced by
`<host>`, `<work_dir>`, `<repo>`, `<venv>` and `~` in everything packaged.

Inspect a packaged run like any other profile:

```bash
scope-profiler inspect _profiling_jobs/packaged/*/run01/profiling_data.h5 --sort total
scope-profiler plot speedup '_profiling_jobs/packaged/<case>/run*/profiling_data.h5'
```

## Results repository

```bash
export SCOPE_PROFILER_RESULTS_REPO=git@github.com:me/profiling-results.git
scope-profiler jobs run case.toml --upload
```

With upload, the repository is cloned before any job is submitted (so an
unreachable one fails early), the case metadata is pushed straight away and
every run as soon as it is packaged; concurrent pushes are rebased and
retried. A case packaged without upload can be pushed later, unchanged:

```bash
scope-profiler jobs upload _profiling_jobs/packaged/<case> --repo <url>
```

A repository shared by several codes keeps each code's cases in a directory of their own:
`path` in the `[store]` table (or `GitResultsStore(path=...)`, or `--path` of `jobs upload`)
names it, with `{label}` for the case's label.

```toml
[store]
url = "git@github.com:me/profiling-results.git"
path = "mycode/{label}"   # mycode/<label>/<timestamp>-<commit>-<label>/run01/...
```

That is the layout a catalogue reads: with
`executions: {path: "mycode/<label>/{execution}/run*/profiling_data.h5"}` in its manifest,
[`scope-profiler site`](/scope-profiler/reference/cli/#scope-profiler-site) makes every
uploaded case folder an execution of the case, labelled by its date and commit, with the
commit, branch and cluster from `case_metadata.json` as its metadata and each run named by its
ranks (`np4`, and `np4 · cupy` when the launch's `metadata` says so). New uploads need no
change to the manifest.

Two more settings of the store:

```toml
[store]
public = true      # a public repository: package with redact=True
after_push = ["gh workflow run docs.yml -R me/mycode"]   # e.g. rebuild the catalogue
```

`after_push` commands run after every successful push, with `{label}`, `{commit}`
and `{url}` filled in; one that fails is reported, and the results stay pushed.
Rebuilding the docs that show the results this way updates the catalogue as soon
as a run is in, without waiting for the next commit to the code.

### Looking after the repository

```bash
scope-profiler jobs check results/                 # exit code 1 on any problem
scope-profiler jobs prune results/ --keep 30       # list what would go
scope-profiler jobs prune results/ --keep 30 --older-than 90 --yes
```

`check` reads every packaged case folder: its metadata parses, the runs it lists
have their folders, metadata and profiles, every `.h5` file opens and no file is
larger than `--max-mb` (50 MB; GitHub refuses 100 MB). Run it in the
repository's CI. `prune` removes all but the newest `--keep` executions of every
case (by date), with `--older-than` only those older than that many days; it
lists them until `--yes` deletes them, and the removal is then committed like any
other change.

## Sweeps

Several cases started as separate processes would each prepare the shared
packaging folder and push to it at the same time. A sweep runs them in one
process: each submits its jobs at once (SLURM runs them as the partitions
allow) and the runs of all cases are packaged one at a time as they finish.

```bash
scope-profiler jobs sweep case.toml other_case.toml submit_gpu.py --upload
scope-profiler jobs sweep submit_heat.py --variants   # one entry per VARIANTS key
```

A Python entry is a script whose `__main__` builds cases and calls
`finalize()`; inside a sweep that call only registers the case. With
`--variants`, a script gets one entry per key of its module-level
`VARIANTS` dict (with `--variant <key>`), and a TOML file one per
`[variants]` table. From Python: `run_sweep(variant_entries("a.py") +
[("b.toml", [])])`. Keep the sweep process running until it is done (e.g. in
`tmux`); it polls the queue every 30 s (`--poll-interval`).
