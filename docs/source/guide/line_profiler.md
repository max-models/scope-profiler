# Line-by-line profiling

scope-profiler can integrate [line_profiler](https://github.com/pyutils/line_profiler)
to give you per-source-line timing breakdowns of decorated functions,
alongside the usual region-level statistics.

## Installation

```bash
pip install "scope-profiler[line-profiler]"
```

## Enabling line profiling

Pass `use_line_profiler=True` to `setup()`:

```python
from scope_profiler import ProfileManager

ProfileManager.setup(use_line_profiler=True)
```

This selects `LineProfilerRegion` for all regions. Each region records
nanosecond timestamps **and** enables `line_profiler` tracing for every
function registered via the `@ProfileManager.profile` decorator.

## Example

```python
import math
import random

from scope_profiler import ProfileManager

ProfileManager.setup(use_line_profiler=True)


@ProfileManager.profile("compute")
def compute(N=50_000):
    s = 0.0
    for _ in range(N):
        x = random.random()
        s += math.sin(x) * math.sqrt(x + 1.0)
    return s


@ProfileManager.profile("allocate")
def allocate(N=100_000):
    a = [i * i for i in range(N)]
    b = []
    for i in range(N):
        b.append(i * i)
    return a, b


compute()
allocate()
ProfileManager.finalize()
```

Output:

```text
Region: compute
  Total Calls : 1
  Total Time  : 0.033359 s
  ...
----------------------------------------
Region: allocate
  Total Calls : 1
  Total Time  : 0.042457 s
  ...
----------------------------------------
Timer unit: 1e-09 s

Total time: 0.029169 s
File: examples/ex_line_profiling.py
Function: compute at line 10

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    10                                           @ProfileManager.profile("compute")
    11                                           def compute(N=50_000):
    12         1       1000.0   1000.0      0.0      s = 0.0
    13     50001    8581000.0    171.6     29.4      for _ in range(N):
    14     50000    9370000.0    187.4     32.1          x = random.random()
    15     50000   11215000.0    224.3     38.4          s += math.sin(x) * math.sqrt(x + 1.0)
    16         1       2000.0   2000.0      0.0      return s
```

The line-by-line table shows, for each source line:

| Column      | Meaning                                        |
| ----------- | ---------------------------------------------- |
| **Hits**    | Number of times the line was executed          |
| **Time**    | Total time spent on that line (in timer units) |
| **Per Hit** | Average time per execution                     |
| **% Time**  | Fraction of the function's total time          |

## Decorator vs. context manager

- **Decorator** (`@ProfileManager.profile`) --- automatically registers
  the function with `line_profiler`. This is the primary use case.
- **Context manager** (`with ProfileManager.profile_region()`) ---
  enables/disables the profiler around the block. Any functions
  previously registered via the decorator path will be profiled while
  the context is active. The code inside the `with` block itself is
  **not** line-profiled (line_profiler needs a function reference).

## Accessing stats programmatically

```python
region = ProfileManager.get_region("compute")

# Get the line_profiler stats object
stats = region.get_stats()

# Print formatted output
region.print_stats()
```

## Overhead considerations

Line profiling adds ~40 µs per call because `line_profiler` instruments
every source line in the function. It is designed for **targeted
debugging**, not for always-on use in hot loops. See {doc}`overhead`
for benchmark data.
