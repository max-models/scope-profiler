# @scope-profiler/plotly

## Validation, updates, and linked charts

Builders validate consumed records, including finite numbers and interval order.
Errors identify the array, row index, and field. `validatePlotData()` also checks
the envelope; unknown additional fields remain compatible. Present format
versions must be positive integers. Duplicate heatmap/bar cells and call IDs
are errors; aggregate repeated measurements before building a figure.

`createColorRegistry(names, overrides)` returns an immutable name-to-color map.
Default colors are also stable across filtering and row ordering. Use explicit
colors when more categories need distinct hues than the eight-color palette.

```js
import {
  createFigureBuilder, createColorRegistry, getPointIdentity,
  updateFigure, disposeFigure,
} from "@scope-profiler/plotly";

const build = createFigureBuilder({
  theme: "dark",
  colors: createColorRegistry(["solve", "assemble"]),
  layout: { uirevision: "profile-123" },
});
await updateFigure(Plotly, element, build(payload));
element.on("plotly_click", ({ points }) => {
  const identity = getPointIdentity(points[0]);
  // identity: { region, file, rank, call_id }; unavailable fields are null.
  if (identity) console.log(identity.region);
});
// On unmount:
disposeFigure(Plotly, element);
```

`createFigureBuilder()` snapshots defaults without modifying global theme state.
Nested layout objects merge, while arrays replace. Keep `layout.uirevision`
constant to retain zoom during `updateFigure()`; change it to reset the view.
The fallback to `newPlot()` for bundles without `react()` cannot preserve zoom.
`disposeFigure()` requires a bundle with `purge()`.

Data-bearing traces expose `customdata.identity`; existing numeric hover
fields are available as numeric keys on the same object. Reference/ideal lines
have no selection identity. Imbalance means and time-series variability traces
share their measured series' legend group.

## Variability and comparisons

`buildDurationTimeseriesFigure(payload, { variability: "band" })` displays a
min/max band; `"error"` displays asymmetric error bars. Missing bounds produce
gaps, and the default `"none"` retains the mean curve alone.

`buildComparisonFigure(payload, { comparison: "absolute" | "percent" })`
subtracts the first selected run from the second. `files` selects exactly two
runs, `metric` selects a statistic, and `sortBy: "regression"` (default) sorts
largest increases first; `"name"` sorts alphabetically. Percentage changes use
the first run as denominator. Missing measurements and zero percentage
baselines produce null values and `figure.diagnostics`, never invented zeros.
The default `"side-by-side"` mode retains grouped bars. Higher values are not
necessarily worse for every metric; interpret the sign for your chosen metric.

## Graph and density semantics

Flame figures reject ancestry cycles. Sankey figures remove edges that would
introduce a cycle after region aggregation and report them in
`figure.diagnostics`. Compact graph edges prefer `edge[valueKey]`, then
`edge.value`. A child total is used only when the original graph has exactly
one incoming non-self edge, with an explicit inference diagnostic. Unattributable
weights are omitted with a diagnostic. Zero measured weights stay zero.

Density figures with unequal grids use separate lane traces and exact bin
edges; gaps remain empty and overlapping bins are rejected. Raw-seconds traces
share one color range across lanes.

## Development checks

```sh
npm ci
npm run lint
npm run check:types
npm run test:coverage # Node 24 (CI runtime)
npx playwright install chromium
npm run test:browser
npm run sync:asset
```

The browser suite renders exporter fixtures in light/dark themes and checks
hover text, zoom retention, teardown, and a deterministic large timeline. It
writes screenshots and build/render/heap measurements to `test-results`.
The broad timing limits detect catastrophic regressions, not small speedups.

For measured optimization, run the repository benchmark workflow using
`packages/plotly/benchmarks/benchmark.toml`. It performs warmups and five runs,
records medians and variation, and runs the package tests as correctness checks.
`npm run benchmark` also prints per-build timings and process memory for the
fixed 8,192-cell workload. Keep optimizations only after comparison says `keep`.

Pure, framework-neutral Plotly figure builders for JSON written by
`scope-profiler export plot-data --format json`. The package does not import
Plotly; applications choose their own Plotly bundle.

Install the builder and a Plotly bundle:

```sh
npm install @scope-profiler/plotly plotly.js-dist-min
```

```js
import Plotly from "plotly.js-dist-min";
import { buildGanttFigure, renderFigure } from "@scope-profiler/plotly";

const payload = await fetch("/figures/gantt_data.json").then((response) =>
  response.json(),
);
const figure = buildGanttFigure(payload);
await renderFigure(Plotly, document.querySelector("#gantt"), figure);
```

## One call for any payload

Every document written by `export plot-data` carries the `plot` kind that
produced it, so `buildFigure` can pick the builder for you:

```js
import { buildFigure, renderFigure } from "@scope-profiler/plotly";

const payload = await fetch("/figures/rank_heatmap_data.json").then(
  (response) => response.json(),
);
await renderFigure(
  Plotly,
  document.querySelector("#chart"),
  buildFigure(payload),
);
```

`validatePlotData(payload, options)` performs the same checks on its own and
returns the resolved kind, for a page that wants to report a bad file rather
than catch a build.

`buildFigure` rejects a document that is not `scope-profiler-plot-data` and one
whose `format_version` is newer than this package supports. JSON written before
scope-profiler stamped that envelope on every kind still works: the kind is then
inferred from the payload shape, and `{ plot: "gantt" }` settles it by hand.

## Builders

`buildGanttFigure`, `buildFlameFigure`, `buildCallgraphFigure`,
`buildDensityFigure`, `buildDurationsFigure`, `buildDurationTimeseriesFigure`,
`buildHistogramFigure`, `buildRankHeatmapFigure`, `buildImbalanceFigure`,
`buildRegionSummaryFigure`, `buildComparisonFigure`, `buildLikwidFigure`,
`buildSpeedupFigure`, `buildWeakScalingFigure`,
`buildScalingEfficiencyFigure` and `buildWeakScalingEfficiencyFigure` each
take `(payload, options)` and return a plain `{ data, layout }` figure.

Common options: `colors` (region or series name to color), `filterRegion(name,
row)` to drop rows, `layout` to merge into the generated layout, `metric` where
a payload carries several, and `theme` (see below).

`filterRegion` always receives the name being filtered first, but its second
argument is the record the builder is walking, and that differs by figure: a
row of the payload's own array for most of them, a node or a call for
`buildCallgraphFigure`, and the region's statistics object for
`buildRegionSummaryFigure` and `buildComparisonFigure`.

`buildGanttFigure` also takes `laneBy`: the default `"region"` gives each
region and rank its own lane, which is the only way a nested profile stays
legible; `"rank"` gives the compact one-row-per-rank view, which suits a flat
profile compared across many ranks.

The region_statistics builders take `metric` as either a short name or the
stored field: `SUMMARY_METRICS` maps `avg`, `min`, `max`, `total`, `first`,
`last`, `std` and `count` to the fields a document stores them under, and a
name in neither spelling is rejected rather than drawn as an empty chart.

The two efficiency curves store their y column under the same name, so which
reading a payload gets comes from the document's own `plot` field rather than
its rows. `buildScalingEfficiencyFigure` is for a _strong_-scaling study, where
the problem size is fixed and the ideal is a speedup proportional to the rank
count; `buildWeakScalingEfficiencyFigure` is for one that grows the problem
with the machine, where the ideal is constant runtime.

## Rendering and redrawing

`renderFigure(plotly, element, figure, config)` draws a figure with any
Plotly-compatible bundle. `updateFigure(...)` takes the same arguments but
redraws into an element that already holds a plot, using Plotly's `react`, so
the viewer's zoom and pan survive. Use it for anything that rebuilds a figure
the viewer is already looking at -- a theme toggle, a changed filter, a new
metric -- and keep `renderFigure` for the first draw.

```js
setTheme(darkMode ? "dark" : "light");
await updateFigure(Plotly, element, buildFigure(payload));
```

## Themes

Chrome -- text, gridlines, hover surface, the dashed ideal lines -- comes from
a theme. The default `auto` sets no text colour and uses a half-transparent
grey grid that reads on any background, so a figure inherits the host page.
Pass `theme: "light" | "dark"`, or your own token object, to a builder, or call
`setTheme(...)` once to change the default for every later build. Plotly bakes
colours into the layout, so a theme change means rebuilding the figures. The
categorical series palette is not themed: those hues read on both backgrounds.

## More than one run in a payload

`export plot-data` accepts several profiles at once, and the payload then
carries a `file` column. Builders keep those runs apart: the gantt, density and
rank heatmap give each run its own lanes, and the histogram, imbalance and
duration time series give each run its own trace, labelled `run / region`. The
region keeps its colour across runs, so a run is told apart by marker symbol
(lines) or bar pattern (bars). A single-run payload is unchanged -- series are
named by region alone.
