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

Builders accept both the current scope-profiler payloads and versioned payloads
with `format: "scope-profiler-plot-data"` and `format_version: 1`.
