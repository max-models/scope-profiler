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

const payload = { intervals: [] };
declare const plotly: PlotlyLike;
declare const element: Element;

const figure: Figure = buildFigure(payload, { theme: "dark" });
const traces: object[] = figure.data;

// Every builder option named in the README has to exist on the type.
buildGanttFigure(payload, { laneBy: "rank" });
buildGanttFigure(payload, { laneBy: "region", theme: { grid: "#333" } });
buildRegionSummaryFigure(
  { files: [] },
  {
    metric: "total",
    topN: 5,
    orientation: "v",
    commonRegionsOnly: true,
    files: ["a", 1],
  },
);
buildRooflineFigure(
  { points: [] },
  {
    filterRegion: (region, row) => !!region && !!row,
  },
);

setTheme("light");
setTheme({ grid: "#eee", text: "#111" });
const tokens: string = resolveTheme("dark").grid;

const kind: PlotKind | undefined = inferPlotKind(payload);
PLOT_BUILDERS.gantt(payload);
buildFigure({ plot: "gantt", intervals: [] }, { laneBy: "rank" });
buildFigure({ plot: "timeseries", points: [] }, { variability: "band" });
// @ts-expect-error a gantt document must contain intervals
buildFigure({ plot: "gantt", points: [] });
// @ts-expect-error intervals need both endpoints
buildGanttFigure({ intervals: [{ region: "solve", start_seconds: 0 }] });
// @ts-expect-error unknown option spelling
buildFigure(payload, { laneBy: "thread" });

const metricField: string | undefined = SUMMARY_METRICS.total;

renderFigure(plotly, element, figure);
updateFigure(plotly, element, figure, { staticPlot: true });

export { traces, tokens, metricField, kind };
