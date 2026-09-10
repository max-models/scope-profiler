export interface PointIdentity {
  region: string | null;
  file: string | null;
  rank: number | null;
  call_id: string | number | null;
  source?: string;
}
export interface Diagnostic {
  code: string;
  message: string;
  region?: string;
  source?: string;
  target?: string;
}
export interface Figure {
  data: object[];
  layout: object;
  diagnostics?: Diagnostic[];
}
export interface PlotlyLike {
  newPlot(
    element: Element | string,
    data: object[],
    layout: object,
    config?: object,
  ): unknown;
  react?(
    element: Element | string,
    data: object[],
    layout: object,
    config?: object,
  ): unknown;
  purge?(element: Element | string): unknown;
}
export type ThemeName = "auto" | "light" | "dark";
export interface ThemeTokens {
  text?: string;
  muted?: string;
  grid?: string;
  hoverBg?: string;
  neutral?: string;
}
export interface ResolvedTheme extends ThemeTokens {
  grid: string;
  neutral: string;
}
export interface RegionRow {
  region: string;
  file?: string;
  rank?: number;
}
export interface Interval extends RegionRow {
  start_seconds: number;
  end_seconds: number;
  call_id?: number | string;
}
export type FlameCall = RegionRow & {
  call_id: number | string;
  parent_call_id?: number | string | null;
  start_seconds: number;
} & (
    | { end_seconds: number; inclusive_duration_seconds?: number }
    | { end_seconds?: number; inclusive_duration_seconds: number }
  );
export interface DurationBar extends RegionRow {
  metric: string;
  value_seconds: number;
  segment?: string;
}
export interface TimeseriesPoint extends RegionRow {
  time_seconds: number;
  mean_duration_seconds: number;
  min_duration_seconds?: number;
  max_duration_seconds?: number;
  call_index?: number;
}
export interface HistogramBin extends RegionRow {
  bin_low_seconds: number;
  bin_high_seconds: number;
  bin_center_seconds: number;
  count: number;
}
export interface DensityPoint extends RegionRow {
  bin_start_seconds: number;
  bin_end_seconds: number;
  occupied_seconds: number;
}
export interface ImbalancePoint extends RegionRow {
  rank: number;
  value_seconds: number;
  mean_over_ranks_seconds: number;
}
export interface HeatmapPoint extends RegionRow {
  [metric: string]: string | number | undefined;
}
export interface ScalingPoint extends RegionRow {
  num_ranks?: number;
  speedup?: number;
  normalized_runtime?: number;
  efficiency?: number;
  [field: string]: string | number | undefined;
}
export interface LikwidBar extends RegionRow {
  series: string;
  value: number;
}
export interface RoofPoint extends RegionRow {
  arithmetic_intensity_flops_per_byte: number;
  performance_gflops: number;
  bandwidth_gbs?: number;
}
export interface RoofCeiling {
  arithmetic_intensity_flops_per_byte: number;
  performance_gflops: number;
}
export interface CallgraphNode {
  name: string;
  depth: number;
  total_duration?: number;
  [metric: string]: string | number | undefined;
}
export interface CallgraphCall {
  name: string;
  depth: number;
  call_id: number | string;
  parent_id: number | string | null;
  file?: string;
  rank?: number;
}
export interface CallgraphEdge {
  parent: string;
  child: string;
  value?: number;
  total_duration?: number;
  [metric: string]: string | number | undefined;
}
export interface RegionStatistics {
  count?: number;
  total_duration_seconds?: number;
  average_duration_seconds?: number;
  min_duration_seconds?: number;
  max_duration_seconds?: number;
  std_duration_seconds?: number;
  first_duration_seconds?: number;
  last_duration_seconds?: number;
  per_rank?: Record<string, RegionStatistics>;
}
export interface SummaryFile {
  label?: string;
  region_statistics: Record<string, RegionStatistics>;
}
export interface PayloadBase {
  format?: "scope-profiler-plot-data";
  format_version?: number;
  colors?: Record<string, string>;
}
export interface GanttPayload extends PayloadBase {
  plot?: "gantt";
  intervals: Interval[];
}
export interface FlamePayload extends PayloadBase {
  plot?: "flame" | "flame_chart" | "flame_graph";
  calls: FlameCall[];
}
export interface DurationsPayload extends PayloadBase {
  plot?: "durations";
  bars: DurationBar[];
  metrics?: string[];
  options?: { metric?: string };
}
export interface TimeseriesPayload extends PayloadBase {
  plot?: "timeseries";
  points: TimeseriesPoint[];
}
export interface HistogramPayload extends PayloadBase {
  plot?: "histogram";
  bins: HistogramBin[];
}
export interface DensityPayload extends PayloadBase {
  plot?: "density";
  points: DensityPoint[];
}
export interface ImbalancePayload extends PayloadBase {
  plot?: "imbalance";
  points: ImbalancePoint[];
  metric?: string;
}
export interface HeatmapPayload extends PayloadBase {
  plot?: "rank_heatmap";
  points: HeatmapPoint[];
}
export type ScalingKind =
  "speedup" | "weak_scaling" | "scaling_efficiency" | "weak_scaling_efficiency";
export interface ScalingPayload extends PayloadBase {
  plot?: ScalingKind;
  points: ScalingPoint[];
  options?: { x_field?: string; x_label?: string; baseline?: number };
}
export interface LikwidPayload extends PayloadBase {
  plot?: "likwid";
  bars: LikwidBar[];
  metric?: string;
}
export interface RooflinePayload extends PayloadBase {
  plot?: "roofline";
  points: RoofPoint[];
  roofline?: RoofCeiling[];
  empirical_ceilings?: boolean;
}
export interface SummaryPayload extends PayloadBase {
  plot?: "region_statistics";
  files: SummaryFile[];
}
export type CallgraphPayload = PayloadBase & { plot?: "callgraph" } & (
    | { regions: CallgraphNode[]; edges: CallgraphEdge[] }
    | { calls: CallgraphCall[] }
  );
export interface PayloadByKind {
  gantt: GanttPayload;
  flame: FlamePayload;
  flame_chart: FlamePayload;
  flame_graph: FlamePayload;
  durations: DurationsPayload;
  timeseries: TimeseriesPayload;
  histogram: HistogramPayload;
  density: DensityPayload;
  imbalance: ImbalancePayload;
  rank_heatmap: HeatmapPayload;
  speedup: ScalingPayload;
  weak_scaling: ScalingPayload;
  scaling_efficiency: ScalingPayload;
  weak_scaling_efficiency: ScalingPayload;
  likwid: LikwidPayload;
  roofline: RooflinePayload;
  region_statistics: SummaryPayload;
  callgraph: CallgraphPayload;
}
export type PlotKind = keyof PayloadByKind;
/** Explicit documents narrow on their plot discriminator. */
export type PlotData = {
  [K in PlotKind]: PayloadByKind[K] & { plot: K };
}[PlotKind];
export type LegacyPlotData = PayloadByKind[PlotKind];
export interface SummaryOptions {
  topN?: number;
  files?: (string | number)[];
  orientation?: "h" | "v";
  commonRegionsOnly?: boolean;
}
export interface BuildOptions extends SummaryOptions {
  colors?: Readonly<Record<string, string>>;
  filterRegion?: (region: string, row: object) => boolean;
  /** Nested objects merge; arrays replace. Set uirevision to preserve zoom. */
  layout?: object;
  metric?: string;
  xField?: string;
  yField?: string;
  ideal?: boolean;
  rootLabel?: string;
  plot?: PlotKind;
  theme?: ThemeName | ThemeTokens;
  laneBy?: "rank" | "region";
  valueKey?: string;
  colorscale?: string;
  logScale?: boolean;
  variability?: "none" | "band" | "error";
  comparison?: "side-by-side" | "absolute" | "percent";
  sortBy?: "regression" | "name";
}
export function setTheme(theme?: ThemeName | ThemeTokens): void;
export function resolveTheme(theme?: ThemeName | ThemeTokens): ResolvedTheme;
export function createColorRegistry(
  names?: Iterable<string>,
  supplied?: Record<string, string>,
): Readonly<Record<string, string>>;
export function createFigureBuilder(
  defaults?: BuildOptions,
): (payload: LegacyPlotData, options?: BuildOptions) => Figure;
export function getPointIdentity(
  point: { customdata?: { identity?: PointIdentity } } | null | undefined,
): PointIdentity | null;
export function buildGanttFigure(
  payload: GanttPayload,
  options?: BuildOptions,
): Figure;
export function buildFlameFigure(
  payload: FlamePayload,
  options?: BuildOptions,
): Figure;
export function buildDurationsFigure(
  payload: DurationsPayload,
  options?: BuildOptions,
): Figure;
export function buildSpeedupFigure(
  payload: ScalingPayload,
  options?: BuildOptions,
): Figure;
export function buildWeakScalingFigure(
  payload: ScalingPayload,
  options?: BuildOptions,
): Figure;
export function buildScalingEfficiencyFigure(
  payload: ScalingPayload,
  options?: BuildOptions,
): Figure;
export function buildWeakScalingEfficiencyFigure(
  payload: ScalingPayload,
  options?: BuildOptions,
): Figure;
export function buildDurationTimeseriesFigure(
  payload: TimeseriesPayload,
  options?: BuildOptions,
): Figure;
export function buildHistogramFigure(
  payload: HistogramPayload,
  options?: BuildOptions,
): Figure;
export function buildRankHeatmapFigure(
  payload: HeatmapPayload,
  options?: BuildOptions,
): Figure;
export function buildDensityFigure(
  payload: DensityPayload,
  options?: BuildOptions,
): Figure;
export function buildImbalanceFigure(
  payload: ImbalancePayload,
  options?: BuildOptions,
): Figure;
export const SUMMARY_METRICS: Readonly<Record<string, string>>;
export function buildRegionSummaryFigure(
  payload: SummaryPayload,
  options?: BuildOptions,
): Figure;
export function buildComparisonFigure(
  payload: SummaryPayload,
  options?: BuildOptions,
): Figure;
export function buildCallgraphFigure(
  payload: CallgraphPayload,
  options?: BuildOptions,
): Figure;
export function buildLikwidFigure(
  payload: LikwidPayload,
  options?: BuildOptions,
): Figure;
export function buildRooflineFigure(
  payload: RooflinePayload,
  options?: BuildOptions,
): Figure;
export const PLOT_DATA_FORMAT: "scope-profiler-plot-data";
export const SUPPORTED_FORMAT_VERSION: number;
export const PLOT_BUILDERS: {
  [K in PlotKind]: (
    payload: PayloadByKind[K],
    options?: BuildOptions,
  ) => Figure;
};
export function inferPlotKind(payload: unknown): PlotKind | undefined;
export function validatePlotData(
  payload: unknown,
  options?: BuildOptions,
): PlotKind;
export function buildFigure(
  payload: LegacyPlotData,
  options?: BuildOptions,
): Figure;
export function renderFigure(
  plotly: PlotlyLike,
  element: Element | string,
  figure: Figure,
  config?: object,
): unknown;
export function updateFigure(
  plotly: PlotlyLike,
  element: Element | string,
  figure: Figure,
  config?: object,
): unknown;
export function disposeFigure(
  plotly: Pick<PlotlyLike, "purge">,
  element: Element | string,
): unknown;
