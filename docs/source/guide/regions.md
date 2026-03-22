# Profiling regions

A **region** is the fundamental unit of measurement in scope-profiler.
Every time a region is entered and exited, the profiler records a
start/end timestamp pair and increments the call counter.

## Creating regions

### Decorator --- `@ProfileManager.profile`

Ideal for wrapping entire functions:

```python
from scope_profiler import ProfileManager

@ProfileManager.profile("solver")
def solve(A, b):
    return np.linalg.solve(A, b)
```

Without an explicit name the function name is used:

```python
@ProfileManager.profile
def solve(A, b):
    return np.linalg.solve(A, b)
# region name: "solve"
```

### Context manager --- `ProfileManager.profile_region()`

Ideal for profiling a section of code inside a function:

```python
def time_step(state, dt):
    with ProfileManager.profile_region("rhs_evaluation"):
        rhs = compute_rhs(state)

    with ProfileManager.profile_region("state_update"):
        state += dt * rhs
```

### Mixing both

Decorators and context managers can be mixed freely. They share the
same region registry, so using the same name in both places accumulates
into a single region:

```python
@ProfileManager.profile("compute")
def compute_batch(data):
    for chunk in data:
        with ProfileManager.profile_region("compute"):
            process(chunk)
# Both the decorator and the context manager contribute
# to the same "compute" region.
```

## Region identity

Regions are identified by **name** (a string). The first call to
`profile_region("foo")` creates the region; subsequent calls with the
same name return the existing instance. This means:

- The same region can be entered from multiple call sites.
- Call counts and timestamps accumulate across all sites.
- Regions are scoped to the current `ProfileManager.setup()` session.
  Calling `setup()` again clears all regions.

## Nesting

Regions can be nested arbitrarily:

```python
with ProfileManager.profile_region("outer"):
    with ProfileManager.profile_region("inner"):
        work()
```

Each region independently records its own timestamps. The library does
**not** compute parent-child relationships --- that can be done in
post-processing by comparing the recorded intervals.

## Accessing region data at runtime

Before calling `finalize()`, you can inspect in-memory data:

```python
region = ProfileManager.get_region("solver")

# Number of completed calls
region.num_calls

# Durations of buffered (not yet flushed) calls, in nanoseconds
durations = region.get_durations_numpy()
```

All registered regions:

```python
for name, region in ProfileManager.get_all_regions().items():
    print(f"{name}: {region.num_calls} calls")
```
