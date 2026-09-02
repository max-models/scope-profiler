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
  const calls = values(payload, "calls");
  const regions = [...new Set(calls.map((call) => call.region))];
  const colors = colorMap(regions, options.colors ?? payload.colors);
  const root = "scope-profiler-root";
  const ids = [root], labels = [options.rootLabel ?? "All calls"], parents = [""], markerColors = [NEUTRAL], hovertext = ["All calls"];
  const roots = calls.filter((call) => call.parent_call_id == null);
  const rootDuration = roots.reduce((sum, call) => sum + (call.inclusive_duration_seconds ?? call.end_seconds - call.start_seconds), 0);
  const callKey = (call) => `${call.file ?? "run"}:${call.rank ?? 0}:${call.call_id}`;
  for (const call of calls) {
    ids.push(callKey(call)); labels.push(call.region);
    parents.push(call.parent_call_id == null ? root : `${call.file ?? "run"}:${call.rank ?? 0}:${call.parent_call_id}`);
    markerColors.push(colors.get(call.region));
    hovertext.push(`<b>${call.region}</b><br>${call.file ?? "run"} / rank ${call.rank ?? 0}<br>start: ${call.start_seconds.toPrecision(6)} s<br>inclusive: ${(call.inclusive_duration_seconds ?? call.end_seconds - call.start_seconds).toPrecision(6)} s`);
  }
  const layout = baseLayout({ height: 500, margin: { l: 24, r: 24, t: 24, b: 24 }, ...options.layout });
  return { data: [{ type: "icicle", ids, labels, parents, values: [rootDuration, ...calls.map((call) => call.inclusive_duration_seconds ?? call.end_seconds - call.start_seconds)], branchvalues: "total", tiling: { orientation: "h" }, marker: { colors: markerColors, line: { color: "rgba(255, 255, 255, 0.55)", width: 1 } }, hovertext, hoverinfo: "text" }], layout: withEmptyState(layout, calls.length > 0) };
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

export function buildSpeedupFigure(payload, options = {}) {
  const xField = options.xField ?? payload.options?.x_field ?? "num_ranks";
  const points = filtered(values(payload, "points"), options);
  const regions = [...new Set(points.map((point) => point.region))];
  const colors = colorMap(regions, options.colors ?? payload.colors);
  const xValues = [...new Set(points.map((point) => point[xField]))].sort((a, b) => typeof a === "number" && typeof b === "number" ? a - b : String(a).localeCompare(String(b)));
  const numeric = xValues.every((value) => typeof value === "number");
  const data = regions.map((region) => { const rows = points.filter((point) => point.region === region).sort((a, b) => xValues.indexOf(a[xField]) - xValues.indexOf(b[xField])); return { type: "scatter", mode: "lines+markers", name: region, x: rows.map((row) => row[xField]), y: rows.map((row) => row.speedup), line: { color: colors.get(region), width: 2.4 }, marker: { color: colors.get(region), size: 7 }, hovertemplate: `<b>%{x}</b><br>${region}: %{y:.3g}×<extra></extra>` }; });
  const baseline = payload.options?.baseline ?? xValues[0];
  if (numeric && options.ideal !== false) data.push({ type: "scatter", mode: "lines", name: "Ideal speedup", x: xValues, y: xValues.map((value) => value / baseline), line: { color: "#777", dash: "dash" }, hoverinfo: "skip" });
  const layout = baseLayout({ height: 420, showlegend: data.length > 1, xaxis: axis({ title: payload.options?.x_label ?? xField, tickvals: xValues }), yaxis: axis({ title: "Speedup", rangemode: "tozero" }), ...options.layout });
  return { data, layout: withEmptyState(layout, points.length > 0) };
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

/** Render a figure with any Plotly-compatible bundle. */
export function renderFigure(plotly, element, figure, config = {}) {
  if (!plotly || typeof plotly.newPlot !== "function") throw new TypeError("renderFigure requires a Plotly-compatible object with newPlot().");
  return plotly.newPlot(element, figure.data, figure.layout, { responsive: true, displaylogo: false, ...config });
}
