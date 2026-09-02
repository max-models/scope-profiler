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

const payload = await fetch("/figures/gantt_data.json").then((response) => response.json());
const figure = buildGanttFigure(payload);
await renderFigure(Plotly, document.querySelector("#gantt"), figure);
```

## One call for any payload

Every document written by `export plot-data` carries the `plot` kind that
produced it, so `buildFigure` can pick the builder for you:

```js
import { buildFigure, renderFigure } from "@scope-profiler/plotly";

const payload = await fetch("/figures/rank_heatmap_data.json").then((response) => response.json());
await renderFigure(Plotly, document.querySelector("#chart"), buildFigure(payload));
```

`buildFigure` rejects a document that is not `scope-profiler-plot-data` and one
whose `format_version` is newer than this package supports. JSON written before
scope-profiler stamped that envelope on every kind still works: the kind is then
inferred from the payload shape, and `{ plot: "gantt" }` settles it by hand.

## Builders

`buildGanttFigure`, `buildFlameFigure`, `buildCallgraphFigure`,
`buildDensityFigure`, `buildDurationsFigure`, `buildDurationTimeseriesFigure`,
`buildHistogramFigure`, `buildRankHeatmapFigure`, `buildImbalanceFigure`,
`buildRegionSummaryFigure`, `buildLikwidFigure`, `buildSpeedupFigure`,
`buildWeakScalingFigure` and `buildScalingEfficiencyFigure` each take
`(payload, options)` and return a plain `{ data, layout }` figure.

Common options: `colors` (region or series name to color), `filterRegion(name,
row)` to drop rows, `layout` to merge into the generated layout, and `metric`
where a payload carries several.
