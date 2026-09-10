// A compile-only exercise of src/index.d.ts. The declarations were shipped
// entirely unchecked, so a builder could gain an option -- `laneBy` did -- and
// stay invisible to every TypeScript consumer. `npm run check:types` fails if
// any line here stops type-checking.
import {
  buildFigure,
  buildGanttFigure,
  buildRegionSummaryFigure,
  buildRooflineFigure,
  inferPlotKind,
  renderFigure,
  updateFigure,
  resolveTheme,
  setTheme,
  PLOT_BUILDERS,
  SUMMARY_METRICS,
  type Figure,
  type PlotKind,
  type PlotlyLike,
} from "../../src/index.js";

declare const payload: object;
declare const plotly: PlotlyLike;
declare const element: Element;

const figure: Figure = buildFigure(payload, { theme: "dark" });
const traces: object[] = figure.data;

// Every builder option named in the README has to exist on the type.
buildGanttFigure(payload, { laneBy: "rank" });
buildGanttFigure(payload, { laneBy: "region", theme: { grid: "#333" } });
buildRegionSummaryFigure(payload, {
  metric: "total",
  topN: 5,
  orientation: "v",
  commonRegionsOnly: true,
  files: ["a", 1],
});
buildRooflineFigure(payload, {
  filterRegion: (region, row) => !!region && !!row,
});

setTheme("light");
setTheme({ grid: "#eee", text: "#111" });
const tokens: string = resolveTheme("dark").grid;

const kind: PlotKind | undefined = inferPlotKind(payload);
if (kind) PLOT_BUILDERS[kind](payload);

const metricField: string | undefined = SUMMARY_METRICS.total;

renderFigure(plotly, element, figure);
updateFigure(plotly, element, figure, { staticPlot: true });

export { traces, tokens, metricField };
