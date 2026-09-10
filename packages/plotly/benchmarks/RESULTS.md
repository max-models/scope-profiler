# Heatmap identity lookup measurement

The baseline is the new identity-aware implementation before replacing its
per-cell linear search with the existing cell index. It is **not** the previous
published package and does not measure the overall cost of all new features.

Fixed input: 8,192 cells, 128 regions, 64 ranks; four builds per Node process.
The shared scope-profiler runner performed one warmup and five measured runs,
including Node process startup, and ran the package tests for correctness.

| Measurement | Baseline | Indexed lookup |
| --- | ---: | ---: |
| Median (seconds) | 1.587993 | 0.113882 |
| Mean (seconds) | 1.600610 | 0.117942 |
| Standard deviation (seconds) | 0.094520 | 0.012649 |

Comparison: `decision: keep`, correctness passed, 92.83% lower median.
Local manifests: `.scope-profiler/plotly-builders/{baseline,candidate}/benchmark.json`
under the repository root. Hardware/runtime differences will change timings;
rerun the declarative workflow before evaluating another optimization.
