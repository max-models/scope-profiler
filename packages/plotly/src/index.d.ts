export interface Figure {
  data: object[];
  layout: object;
}
export interface PlotlyLike {
  newPlot(
    element: Element | string,
    data: object[],
    layout: object,
    config?: object,
  ): unknown;
}
export type PlotKind =
  | "gantt"
  | "density"
  | "flame"
  | "flame_chart"
  | "flame_graph"
  | "callgraph"
  | "durations"
  | "timeseries"
  | "speedup"
  | "weak_scaling"
  | "scaling_efficiency"
  | "rank_heatmap"
  | "histogram"
  | "imbalance"
  | "likwid"
  | "region_statistics";
export interface BuildOptions {
  colors?: Record<string, string>;
  filterRegion?: (region: string, row: object) => boolean;
  layout?: object;
  metric?: string;
  xField?: string;
  ideal?: boolean;
  rootLabel?: string;
  plot?: PlotKind;
}
export function buildGanttFigure(
  payload: object,
  options?: BuildOptions,
): Figure;
export function buildFlameFigure(
  payload: object,
  options?: BuildOptions,
): Figure;
export function buildDurationsFigure(
  payload: object,
  options?: BuildOptions,
): Figure;
export function buildSpeedupFigure(
  payload: object,
  options?: BuildOptions & { yField?: string },
): Figure;
export function buildWeakScalingFigure(
  payload: object,
  options?: BuildOptions,
): Figure;
export function buildScalingEfficiencyFigure(
  payload: object,
  options?: BuildOptions,
): Figure;
export function buildDurationTimeseriesFigure(
  payload: object,
  options?: BuildOptions,
): Figure;
export function buildHistogramFigure(
  payload: object,
  options?: BuildOptions,
): Figure;
export function buildRankHeatmapFigure(
  payload: object,
  options?: BuildOptions & { valueKey?: string; colorscale?: string },
): Figure;
export function buildDensityFigure(
  payload: object,
  options?: BuildOptions & {
    valueKey?: "occupancy" | "occupied_seconds";
    colorscale?: string;
  },
): Figure;
export function buildImbalanceFigure(
  payload: object,
  options?: BuildOptions,
): Figure;
export function buildRegionSummaryFigure(
  payload: object,
  options?: BuildOptions & { topN?: number },
): Figure;
export function buildCallgraphFigure(
  payload: object,
  options?: BuildOptions & { valueKey?: string },
): Figure;
export function buildLikwidFigure(
  payload: object,
  options?: BuildOptions & { logScale?: boolean },
): Figure;
export const PLOT_DATA_FORMAT: string;
export const SUPPORTED_FORMAT_VERSION: number;
export const PLOT_BUILDERS: Record<
  PlotKind,
  (payload: object, options?: BuildOptions) => Figure
>;
export function inferPlotKind(payload: object): PlotKind | undefined;
export function buildFigure(payload: object, options?: BuildOptions): Figure;
export function renderFigure(
  plotly: PlotlyLike,
  element: Element | string,
  figure: Figure,
  config?: object,
): unknown;
