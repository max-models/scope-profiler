/** Framework-neutral Plotly specifications for scope-profiler plot-data. */

const DEFAULT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"];
const NEUTRAL = "#898781";

function colorMap(names, supplied = {}) {
  const map = new Map();
  let index = 0;
  for (const name of names) {
    if (map.has(name)) continue;
    map.set(name, supplied[name] ?? DEFAULT_COLORS[index++ % DEFAULT_COLORS.length]);
  }
  return map;
}

function values(payload, key) {
  if (!payload || !Array.isArray(payload[key])) throw new TypeError(`Expected a scope-profiler plot-data payload with a ${key} array.`);
  return payload[key];
}

function baseLayout(overrides = {}) {
  return {
    paper_bgcolor: "transparent", plot_bgcolor: "transparent",
    font: { family: "Inter, ui-sans-serif, system-ui, sans-serif", size: 12 },
    hovermode: "closest", hoverlabel: { namelength: -1 },
    margin: { l: 100, r: 24, t: 32, b: 64 },
    legend: { orientation: "h", y: -0.2, x: 0 }, ...overrides,
  };
}

function axis(overrides = {}) {
  return { automargin: true, gridcolor: "rgba(128, 128, 128, 0.2)", zeroline: false, ...overrides };
}

function withEmptyState(layout, hasData) {
  if (hasData) return layout;
  return {
    ...layout,
    annotations: [{ text: "No data to display.", showarrow: false, xref: "paper", yref: "paper", x: 0.5, y: 0.5 }],
  };
}

function filtered(rows, options) {
  return typeof options?.filterRegion === "function" ? rows.filter((row) => options.filterRegion(row.region, row)) : rows;
}

/** Build a multi-run, multi-rank timeline. Regions are colored; file/rank are lanes. */
export function buildGanttFigure(payload, options = {}) {
  const intervals = filtered(values(payload, "intervals"), options);
  const regions = [...new Set(intervals.map((row) => row.region))];
  const colors = colorMap(regions, options.colors ?? payload.colors);
  const lanes = [...new Set(intervals.map((row) => `${row.file ?? "run"} / rank ${row.rank ?? 0}`))];
  const data = regions.map((region) => {
    const rows = intervals.filter((row) => row.region === region);
    return { type: "bar", orientation: "h", name: region,
      y: rows.map((row) => `${row.file ?? "run"} / rank ${row.rank ?? 0}`),
      x: rows.map((row) => row.end_seconds - row.start_seconds), base: rows.map((row) => row.start_seconds),
      marker: { color: colors.get(region), line: { color: "rgba(0, 0, 0, 0.28)", width: 0.5 } },
      customdata: rows.map((row) => [row.file, row.rank]),
      hovertemplate: `<b>${region}</b><br>%{customdata[0]} / rank %{customdata[1]}<br>start: %{base:.6g} s<br>duration: %{x:.6g} s<extra></extra>`,
    };
  });
  const layout = baseLayout({ barmode: "overlay", height: Math.max(280, 48 * lanes.length + 150), showlegend: regions.length > 1, xaxis: axis({ title: "Time (s)" }), yaxis: axis({ categoryorder: "array", categoryarray: lanes, autorange: "reversed", showgrid: false }), ...options.layout });
  return { data, layout: withEmptyState(layout, intervals.length > 0) };
}

/** Build an icicle flame chart using scope-profiler's explicit call IDs. */
export function buildFlameFigure(payload, options = {}) {
  const allCalls = values(payload, "calls");
  const calls = filtered(allCalls, options);
  const regions = [...new Set(calls.map((call) => call.region))];
  const colors = colorMap(regions, options.colors ?? payload.colors);
  const root = "scope-profiler-root";
  const callKey = (call) => `${call.file ?? "run"}:${call.rank ?? 0}:${call.call_id}`;
  const parentKey = (call) => (call.parent_call_id == null ? null : `${call.file ?? "run"}:${call.rank ?? 0}:${call.parent_call_id}`);
  const duration = (call) => call.inclusive_duration_seconds ?? call.end_seconds - call.start_seconds;
  // A filter can remove a call whose children survive; re-parent each survivor
  // onto its nearest surviving ancestor so the icicle stays a single tree
  // instead of silently dropping the orphans.
  const byKey = new Map(allCalls.map((call) => [callKey(call), call]));
  const kept = new Set(calls.map(callKey));
  const anchor = (call) => {
    let key = parentKey(call);
    while (key != null && !kept.has(key)) key = byKey.has(key) ? parentKey(byKey.get(key)) : null;
    return key ?? root;
  };
  const anchors = calls.map(anchor);
  const rootDuration = calls.reduce((sum, call, index) => (anchors[index] === root ? sum + duration(call) : sum), 0);
  const ids = [root], labels = [options.rootLabel ?? "All calls"], parents = [""], markerColors = [NEUTRAL], hovertext = ["All calls"];
  calls.forEach((call, index) => {
    ids.push(callKey(call)); labels.push(call.region); parents.push(anchors[index]);
    markerColors.push(colors.get(call.region));
    hovertext.push(`<b>${call.region}</b><br>${call.file ?? "run"} / rank ${call.rank ?? 0}<br>start: ${call.start_seconds.toPrecision(6)} s<br>inclusive: ${duration(call).toPrecision(6)} s`);
  });
  const layout = baseLayout({ height: 500, margin: { l: 24, r: 24, t: 24, b: 24 }, ...options.layout });
  return { data: [{ type: "icicle", ids, labels, parents, values: [rootDuration, ...calls.map(duration)], branchvalues: "total", tiling: { orientation: "h" }, marker: { colors: markerColors, line: { color: "rgba(255, 255, 255, 0.55)", width: 1 } }, hovertext, hoverinfo: "text" }], layout: withEmptyState(layout, calls.length > 0) };
}

export function buildDurationsFigure(payload, options = {}) {
  const metric = options.metric ?? payload.options?.metric ?? payload.metrics?.[0] ?? "total";
  const bars = filtered(values(payload, "bars"), options).filter((bar) => bar.metric === metric);
  // A stacked-children export is already decomposed into segments. Preserve
  // that decomposition instead of letting duplicate region rows overwrite.
  const stacked = bars.some((bar) => bar.segment != null);
  const groups = [...new Set(bars.map((bar) => stacked ? bar.segment : bar.rank == null ? bar.file : `rank ${bar.rank}`))];
  const regions = [...new Set(bars.map((bar) => bar.region))];
  const colors = colorMap(groups, options.colors ?? payload.colors);
  const data = groups.map((group) => {
    const rows = bars.filter((bar) => (stacked ? bar.segment : bar.rank == null ? bar.file : `rank ${bar.rank}`) === group);
    const byRegion = new Map(rows.map((bar) => [bar.region, bar.value_seconds]));
    return { type: "bar", name: group, x: regions, y: regions.map((region) => byRegion.get(region) ?? null), marker: { color: colors.get(group), line: { color: "rgba(0, 0, 0, 0.22)", width: 0.5 } }, hovertemplate: `<b>%{x}</b><br>${group}: %{y:.6g} s<extra></extra>` };
  });
  const layout = baseLayout({ barmode: stacked ? "stack" : "group", height: Math.max(360, 34 * regions.length + 180), showlegend: groups.length > 1, xaxis: axis({ tickangle: -35 }), yaxis: axis({ title: `${metric} duration (s)` }), ...options.layout });
  return { data, layout: withEmptyState(layout, bars.length > 0) };
}

// The three scaling exports differ only in the y column they carry and the
// shape of their ideal line, so one builder serves all of them.
const SCALING_KINDS = {
  speedup: { yKey: "speedup", title: "Speedup", suffix: "×", idealName: "Ideal speedup", ideal: (value, baseline) => value / baseline },
  weak_scaling: { yKey: "normalized_runtime", title: "Normalized runtime", suffix: "×", idealName: "Ideal weak scaling", ideal: () => 1 },
  scaling_efficiency: { yKey: "efficiency", title: "Scaling efficiency", suffix: "", idealName: "Ideal efficiency", ideal: () => 1 },
};

function scalingKind(payload, options) {
  const named = options.plot ?? payload?.plot;
  if (named && SCALING_KINDS[named]) return SCALING_KINDS[named];
  if (options.yField) {
    const match = Object.values(SCALING_KINDS).find((kind) => kind.yKey === options.yField);
    return match ?? { ...SCALING_KINDS.speedup, yKey: options.yField, title: options.yField };
  }
  const row = payload?.points?.[0];
  return (row && Object.values(SCALING_KINDS).find((kind) => row[kind.yKey] != null)) ?? SCALING_KINDS.speedup;
}

/** Build a scaling curve: speedup, weak scaling, or parallel efficiency. */
export function buildSpeedupFigure(payload, options = {}) {
  const kind = scalingKind(payload, options);
  const xField = options.xField ?? payload.options?.x_field ?? "num_ranks";
  const points = filtered(values(payload, "points"), options);
  const regions = [...new Set(points.map((point) => point.region))];
  const colors = colorMap(regions, options.colors ?? payload.colors);
  const xValues = [...new Set(points.map((point) => point[xField]))].sort((a, b) => typeof a === "number" && typeof b === "number" ? a - b : String(a).localeCompare(String(b)));
  const numeric = xValues.every((value) => typeof value === "number");
  const data = regions.map((region) => { const rows = points.filter((point) => point.region === region).sort((a, b) => xValues.indexOf(a[xField]) - xValues.indexOf(b[xField])); return { type: "scatter", mode: "lines+markers", name: region, x: rows.map((row) => row[xField]), y: rows.map((row) => row[kind.yKey]), line: { color: colors.get(region), width: 2.4 }, marker: { color: colors.get(region), size: 7 }, hovertemplate: `<b>%{x}</b><br>${region}: %{y:.3g}${kind.suffix}<extra></extra>` }; });
  const baseline = payload.options?.baseline ?? xValues[0];
  if (numeric && options.ideal !== false) data.push({ type: "scatter", mode: "lines", name: kind.idealName, x: xValues, y: xValues.map((value) => kind.ideal(value, baseline)), line: { color: "#777", dash: "dash" }, hoverinfo: "skip" });
  const layout = baseLayout({ height: 420, showlegend: data.length > 1, xaxis: axis({ title: payload.options?.x_label ?? xField, tickvals: xValues }), yaxis: axis({ title: kind.title, rangemode: "tozero" }), ...options.layout });
  return { data, layout: withEmptyState(layout, points.length > 0) };
}

/** Build a weak-scaling curve (runtime normalized to the baseline scale). */
export function buildWeakScalingFigure(payload, options = {}) {
  return buildSpeedupFigure(payload, { ...options, plot: "weak_scaling" });
}

/** Build a parallel-efficiency curve (measured speedup over ideal speedup). */
export function buildScalingEfficiencyFigure(payload, options = {}) {
  return buildSpeedupFigure(payload, { ...options, plot: "scaling_efficiency" });
}

/** Build mean call duration over time, one trace per region. */
export function buildDurationTimeseriesFigure(payload, options = {}) {
  const points = filtered(values(payload, "points"), options);
  const regions = [...new Set(points.map((point) => point.region))];
  const colors = colorMap(regions, options.colors ?? payload.colors);
  const data = regions.map((region) => {
    const rows = points.filter((point) => point.region === region).sort((a, b) => a.time_seconds - b.time_seconds);
    return { type: "scatter", mode: "lines+markers", name: region, x: rows.map((row) => row.time_seconds), y: rows.map((row) => row.mean_duration_seconds), line: { color: colors.get(region), width: 2.2 }, marker: { color: colors.get(region), size: 5 }, customdata: rows.map((row) => [row.min_duration_seconds, row.max_duration_seconds, row.call_index]), hovertemplate: `<b>${region}</b><br>time: %{x:.6g} s<br>mean: %{y:.6g} s<br>min–max: %{customdata[0]:.4g}–%{customdata[1]:.4g} s<extra></extra>` };
  });
  const layout = baseLayout({ height: 420, showlegend: regions.length > 1, xaxis: axis({ title: "Time (s)" }), yaxis: axis({ title: "Mean call duration (s)" }), ...options.layout });
  return { data, layout: withEmptyState(layout, points.length > 0) };
}

/** Build duration distributions from histogram bin records. */
export function buildHistogramFigure(payload, options = {}) {
  const bins = filtered(values(payload, "bins"), options);
  const regions = [...new Set(bins.map((bin) => bin.region))];
  const colors = colorMap(regions, options.colors ?? payload.colors);
  const data = regions.map((region) => {
    const rows = bins.filter((bin) => bin.region === region).sort((a, b) => a.bin_center_seconds - b.bin_center_seconds);
    return { type: "bar", name: region, x: rows.map((bin) => bin.bin_center_seconds), y: rows.map((bin) => bin.count), width: rows.map((bin) => bin.bin_high_seconds - bin.bin_low_seconds), marker: { color: colors.get(region), line: { color: "rgba(0, 0, 0, 0.2)", width: 0.5 } }, hovertemplate: `<b>${region}</b><br>%{x:.6g} s: %{y} calls<extra></extra>` };
  });
  const layout = baseLayout({ barmode: "overlay", height: 400, showlegend: regions.length > 1, xaxis: axis({ title: "Call duration (s)" }), yaxis: axis({ title: "Calls" }), ...options.layout });
  return { data, layout: withEmptyState(layout, bins.length > 0) };
}

/** Build a rank × region heatmap from duration records. */
export function buildRankHeatmapFigure(payload, options = {}) {
  const points = filtered(values(payload, "points"), options);
  const regions = [...new Set(points.map((point) => point.region))];
  const ranks = [...new Set(points.map((point) => point.rank))].sort((a, b) => a - b);
  const inferredValueKey = points[0] ? Object.keys(points[0]).find((key) => key.endsWith("_duration_seconds")) : undefined;
  const valueKey = options.valueKey ?? inferredValueKey ?? "total_duration_seconds";
  const byCell = new Map(points.map((point) => [`${point.rank}:${point.region}`, point[valueKey]]));
  const data = [{ type: "heatmap", x: regions, y: ranks.map(String), z: ranks.map((rank) => regions.map((region) => byCell.get(`${rank}:${region}`) ?? null)), colorscale: options.colorscale ?? "Viridis", colorbar: { title: "Seconds" }, hovertemplate: "rank %{y}<br>%{x}: %{z:.6g} s<extra></extra>" }];
  const layout = baseLayout({ height: Math.max(320, 44 * ranks.length + 150), xaxis: axis({ title: "Region" }), yaxis: axis({ title: "Rank", autorange: "reversed", showgrid: false }), ...options.layout });
  return { data, layout: withEmptyState(layout, points.length > 0) };
}

/** Build per-rank duration lines, with a dashed rank mean for each region. */
export function buildImbalanceFigure(payload, options = {}) {
  const points = filtered(values(payload, "points"), options);
  const regions = [...new Set(points.map((point) => point.region))];
  const colors = colorMap(regions, options.colors ?? payload.colors);
  const data = regions.flatMap((region) => {
    const rows = points.filter((point) => point.region === region).sort((a, b) => a.rank - b.rank);
    const color = colors.get(region);
    return [
      { type: "scatter", mode: "lines+markers", name: region, x: rows.map((row) => row.rank), y: rows.map((row) => row.value_seconds), line: { color, width: 2.2 }, marker: { color, size: 7 }, hovertemplate: `<b>${region}</b><br>rank %{x}: %{y:.6g} s<extra></extra>` },
      { type: "scatter", mode: "lines", name: `${region} mean`, x: rows.map((row) => row.rank), y: rows.map((row) => row.mean_over_ranks_seconds), line: { color, dash: "dot", width: 1.3 }, hoverinfo: "skip", showlegend: false },
    ];
  });
  const layout = baseLayout({ height: 420, showlegend: regions.length > 1, xaxis: axis({ title: "Rank", dtick: 1 }), yaxis: axis({ title: `${payload.metric ?? "Duration"} (s)` }), ...options.layout });
  return { data, layout: withEmptyState(layout, points.length > 0) };
}

/** Build a timeline-occupancy heatmap from binned density records. */
export function buildDensityFigure(payload, options = {}) {
  const points = filtered(values(payload, "points"), options);
  const lane = (point) => `${point.file ?? "run"} / ${point.region}`;
  const lanes = [...new Set(points.map(lane))];
  const starts = [...new Set(points.map((point) => point.bin_start_seconds))].sort((a, b) => a - b);
  const width = points.length ? points[0].bin_end_seconds - points[0].bin_start_seconds : 0;
  // Occupancy is the share of the bin the region was inside, which compares
  // across runs of different length; raw seconds stay available via valueKey.
  const asFraction = (options.valueKey ?? "occupancy") === "occupancy";
  const byCell = new Map(points.map((point) => [`${lane(point)}:${point.bin_start_seconds}`, point]));
  const cell = (laneName, start, pick) => { const point = byCell.get(`${laneName}:${start}`); return point ? pick(point) : null; };
  const span = (point) => point.bin_end_seconds - point.bin_start_seconds;
  const data = [{
    type: "heatmap", x: starts.map((start) => start + width / 2), y: lanes,
    z: lanes.map((laneName) => starts.map((start) => cell(laneName, start, (point) => (asFraction ? (span(point) > 0 ? point.occupied_seconds / span(point) : null) : point.occupied_seconds)))),
    customdata: lanes.map((laneName) => starts.map((start) => cell(laneName, start, (point) => point.occupied_seconds))),
    colorscale: options.colorscale ?? "Viridis", ...(asFraction ? { zmin: 0, zmax: 1 } : {}),
    colorbar: { title: asFraction ? "Occupancy" : "Seconds" },
    hovertemplate: `%{y}<br>t = %{x:.6g} s<br>${asFraction ? "occupancy: %{z:.3f}<br>" : ""}occupied: %{customdata:.4g} s<extra></extra>`,
  }];
  const layout = baseLayout({ height: Math.max(320, 34 * lanes.length + 150), xaxis: axis({ title: "Time (s)" }), yaxis: axis({ categoryorder: "array", categoryarray: lanes, autorange: "reversed", showgrid: false }), ...options.layout });
  return { data, layout: withEmptyState(layout, points.length > 0) };
}

const SUMMARY_LABELS = { count: "Calls", average_duration_seconds: "Average duration (s)", min_duration_seconds: "Minimum duration (s)", max_duration_seconds: "Maximum duration (s)", first_duration_seconds: "First call duration (s)", last_duration_seconds: "Last call duration (s)", std_duration_seconds: "Duration std. dev. (s)", total_duration_seconds: "Total duration (s)" };

/** Build a ranked region bar chart from a region_statistics document. */
export function buildRegionSummaryFigure(payload, options = {}) {
  const files = values(payload, "files");
  const metric = options.metric ?? "total_duration_seconds";
  const limit = options.topN ?? 20;
  const keep = typeof options.filterRegion === "function" ? options.filterRegion : () => true;
  const totals = new Map();
  for (const file of files) {
    for (const [region, stats] of Object.entries(file.region_statistics ?? {})) {
      if (!keep(region, stats)) continue;
      totals.set(region, (totals.get(region) ?? 0) + (stats[metric] ?? 0));
    }
  }
  // Rank by the pooled metric so the slowest regions lead, then keep the head
  // of the list: a long run has more regions than a bar chart can carry.
  const regions = [...totals.entries()].sort((a, b) => b[1] - a[1]).slice(0, limit).map(([region]) => region);
  const labels = files.map((file) => file.label ?? "run");
  const colors = colorMap(labels, options.colors ?? payload.colors);
  const unit = metric === "count" ? "" : " s";
  const data = files.map((file, index) => {
    const stats = file.region_statistics ?? {};
    return { type: "bar", orientation: "h", name: labels[index], y: regions, x: regions.map((region) => stats[region]?.[metric] ?? null),
      marker: { color: colors.get(labels[index]), line: { color: "rgba(0, 0, 0, 0.22)", width: 0.5 } },
      customdata: regions.map((region) => stats[region]?.count ?? null),
      hovertemplate: `<b>%{y}</b><br>${labels[index]}: %{x:.6g}${unit}<br>calls: %{customdata}<extra></extra>` };
  });
  const layout = baseLayout({ barmode: "group", height: Math.max(320, 26 * regions.length + 160), showlegend: files.length > 1, xaxis: axis({ title: SUMMARY_LABELS[metric] ?? metric }), yaxis: axis({ categoryorder: "array", categoryarray: [...regions].reverse(), showgrid: false }), ...options.layout });
  return { data, layout: withEmptyState(layout, regions.length > 0) };
}

/** Build a Sankey call graph from either callgraph export shape.
 *
 * The compact export collapses repeated invocations into one node per region,
 * which can turn recursion into a cycle; a Sankey cannot draw one, so links
 * that do not increase call depth are dropped. Use the flame chart to see
 * recursion in full.
 */
export function buildCallgraphFigure(payload, options = {}) {
  const compact = Array.isArray(payload?.regions);
  if (!compact && !Array.isArray(payload?.calls)) throw new TypeError("Expected a scope-profiler plot-data payload with a regions or calls array.");
  const keep = typeof options.filterRegion === "function" ? options.filterRegion : () => true;
  const weightKey = options.valueKey ?? "total_duration";
  let nodes, links, unit;
  if (compact) {
    const regions = payload.regions.filter((region) => keep(region.name, region));
    const depths = new Map(regions.map((region) => [region.name, region.depth]));
    const weights = new Map(regions.map((region) => [region.name, region[weightKey] ?? 0]));
    nodes = regions.map((region) => region.name);
    links = (payload.edges ?? []).filter(({ parent, child }) => depths.has(parent) && depths.has(child) && depths.get(child) > depths.get(parent))
      .map(({ parent, child }) => ({ source: parent, target: child, value: weights.get(child) || 1 }));
    unit = weightKey.endsWith("duration") ? " s" : "";
  } else {
    const calls = payload.calls.filter((call) => keep(call.name, call));
    const byId = new Map(calls.map((call) => [call.call_id, call]));
    const counts = new Map();
    for (const call of calls) {
      const parent = byId.get(call.parent_id);
      if (!parent || call.depth <= parent.depth) continue;
      const key = `${parent.name}\u0000${call.name}`;
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    nodes = [...new Set(calls.map((call) => call.name))];
    links = [...counts.entries()].map(([key, value]) => { const [source, target] = key.split("\u0000"); return { source, target, value }; });
    unit = " calls";
  }
  const index = new Map(nodes.map((name, position) => [name, position]));
  const colors = colorMap(nodes, options.colors ?? payload.colors);
  const data = [{
    type: "sankey", orientation: "h",
    node: { label: nodes, color: nodes.map((name) => colors.get(name)), pad: 14, thickness: 16, line: { color: "rgba(0, 0, 0, 0.25)", width: 0.5 } },
    link: { source: links.map((link) => index.get(link.source)), target: links.map((link) => index.get(link.target)), value: links.map((link) => link.value),
      hovertemplate: `%{source.label} \u2192 %{target.label}<br>%{value:.6g}${unit}<extra></extra>` },
  }];
  const layout = baseLayout({ height: Math.max(320, 26 * nodes.length + 160), margin: { l: 24, r: 24, t: 24, b: 24 }, ...options.layout });
  return { data, layout: withEmptyState(layout, links.length > 0) };
}

/** Build a grouped bar chart of one LIKWID hardware-counter metric. */
export function buildLikwidFigure(payload, options = {}) {
  const bars = filtered(values(payload, "bars"), options);
  const series = [...new Set(bars.map((bar) => bar.series))];
  const regions = [...new Set(bars.map((bar) => bar.region))];
  const colors = colorMap(series, options.colors ?? payload.colors);
  const metric = options.metric ?? payload.metric ?? "value";
  const data = series.map((name) => {
    const byRegion = new Map(bars.filter((bar) => bar.series === name).map((bar) => [bar.region, bar.value]));
    return { type: "bar", name, x: regions, y: regions.map((region) => byRegion.get(region) ?? null), marker: { color: colors.get(name), line: { color: "rgba(0, 0, 0, 0.22)", width: 0.5 } }, hovertemplate: `<b>%{x}</b><br>${name}: %{y:.6g}<extra></extra>` };
  });
  const layout = baseLayout({ barmode: "group", height: Math.max(360, 34 * regions.length + 180), showlegend: series.length > 1, xaxis: axis({ tickangle: -35 }), yaxis: axis({ title: metric, ...(options.logScale ? { type: "log" } : {}) }), ...options.layout });
  return { data, layout: withEmptyState(layout, bars.length > 0) };
}

export const PLOT_DATA_FORMAT = "scope-profiler-plot-data";
export const SUPPORTED_FORMAT_VERSION = 1;

/** Builder for each `plot` kind written by `export plot-data --format json`. */
export const PLOT_BUILDERS = {
  gantt: buildGanttFigure,
  density: buildDensityFigure,
  flame: buildFlameFigure,
  flame_chart: buildFlameFigure,
  flame_graph: buildFlameFigure,
  callgraph: buildCallgraphFigure,
  durations: buildDurationsFigure,
  timeseries: buildDurationTimeseriesFigure,
  speedup: buildSpeedupFigure,
  weak_scaling: buildSpeedupFigure,
  scaling_efficiency: buildSpeedupFigure,
  rank_heatmap: buildRankHeatmapFigure,
  histogram: buildHistogramFigure,
  imbalance: buildImbalanceFigure,
  likwid: buildLikwidFigure,
  region_statistics: buildRegionSummaryFigure,
};

/** Guess the plot kind of a payload written before the envelope existed. */
export function inferPlotKind(payload) {
  if (!payload || typeof payload !== "object") return undefined;
  if (Array.isArray(payload.intervals)) return "gantt";
  if (Array.isArray(payload.bins)) return "histogram";
  if (Array.isArray(payload.files) && payload.files[0]?.region_statistics) return "region_statistics";
  if (Array.isArray(payload.regions) && Array.isArray(payload.edges)) return "callgraph";
  if (Array.isArray(payload.bars)) return payload.bars[0]?.series != null ? "likwid" : "durations";
  if (Array.isArray(payload.calls)) return payload.calls[0]?.parent_id !== undefined ? "callgraph" : "flame";
  const point = Array.isArray(payload.points) ? payload.points[0] : undefined;
  if (!point) return undefined;
  if (point.bin_start_seconds != null) return "density";
  if (point.mean_duration_seconds != null) return "timeseries";
  if (point.mean_over_ranks_seconds != null) return "imbalance";
  if (point.speedup != null) return "speedup";
  if (point.normalized_runtime != null) return "weak_scaling";
  if (point.efficiency != null) return "scaling_efficiency";
  if (point.rank != null) return "rank_heatmap";
  return undefined;
}

/** Build the right figure for any plot-data document, without naming a builder.
 *
 * Dispatches on the document's own `plot` field, falling back to the payload
 * shape for files written before scope-profiler stamped the envelope on every
 * kind.
 */
export function buildFigure(payload, options = {}) {
  if (payload?.format != null && payload.format !== PLOT_DATA_FORMAT) throw new TypeError(`Expected a ${PLOT_DATA_FORMAT} document, got ${JSON.stringify(payload.format)}.`);
  const version = payload?.format_version;
  if (typeof version === "number" && version > SUPPORTED_FORMAT_VERSION) throw new TypeError(`Plot-data format version ${version} is newer than this package supports (${SUPPORTED_FORMAT_VERSION}); upgrade @scope-profiler/plotly.`);
  const kind = options.plot ?? payload?.plot ?? inferPlotKind(payload);
  const builder = kind && PLOT_BUILDERS[kind];
  if (!builder) throw new TypeError(kind ? `No figure builder for plot kind ${JSON.stringify(kind)}.` : "Could not determine the plot kind; pass options.plot.");
  return builder(payload, { plot: kind, ...options });
}

/** Render a figure with any Plotly-compatible bundle. */
export function renderFigure(plotly, element, figure, config = {}) {
  if (!plotly || typeof plotly.newPlot !== "function") throw new TypeError("renderFigure requires a Plotly-compatible object with newPlot().");
  return plotly.newPlot(element, figure.data, figure.layout, { responsive: true, displaylogo: false, ...config });
}
