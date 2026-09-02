export interface Figure { data: object[]; layout: object }
export interface PlotlyLike { newPlot(element: Element | string, data: object[], layout: object, config?: object): unknown }
export interface BuildOptions { colors?: Record<string, string>; filterRegion?: (region: string, row: object) => boolean; layout?: object; metric?: string; xField?: string; ideal?: boolean; rootLabel?: string }
export function buildGanttFigure(payload: object, options?: BuildOptions): Figure;
export function buildFlameFigure(payload: object, options?: BuildOptions): Figure;
export function buildDurationsFigure(payload: object, options?: BuildOptions): Figure;
export function buildSpeedupFigure(payload: object, options?: BuildOptions): Figure;
export function buildDurationTimeseriesFigure(payload: object, options?: BuildOptions): Figure;
export function buildHistogramFigure(payload: object, options?: BuildOptions): Figure;
export function buildRankHeatmapFigure(payload: object, options?: BuildOptions & { valueKey?: string; colorscale?: string }): Figure;
export function buildImbalanceFigure(payload: object, options?: BuildOptions): Figure;
export function renderFigure(plotly: PlotlyLike, element: Element | string, figure: Figure, config?: object): unknown;
