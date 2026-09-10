# @scope-profiler/plotly

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

The two efficiency curves store their y column under the same name, so which
reading a payload gets comes from the document's own `plot` field rather than
its rows. `buildScalingEfficiencyFigure` is for a *strong*-scaling study, where
the problem size is fixed and the ideal is a speedup proportional to the rank
count; `buildWeakScalingEfficiencyFigure` is for one that grows the problem
with the machine, where the ideal is constant runtime.

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
