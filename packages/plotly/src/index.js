/** Framework-neutral Plotly specifications for scope-profiler plot-data. */
/** Runtime checks shared by direct builders and the envelope validator. */
function validateRecords(kind, payload, options = {}) {
  const fields = {
    gantt: ["intervals", "start_seconds", "end_seconds"],
    flame: ["calls", "start_seconds"],
    durations: ["bars", "value_seconds"],
    histogram: [
      "bins",
      "bin_low_seconds",
      "bin_high_seconds",
      "bin_center_seconds",
      "count",
    ],
    timeseries: ["points", "time_seconds", "mean_duration_seconds"],
    imbalance: ["points", "rank", "value_seconds", "mean_over_ranks_seconds"],
    density: [
      "points",
      "bin_start_seconds",
      "bin_end_seconds",
      "occupied_seconds",
    ],
    rank_heatmap: ["points"],
    scaling: ["points"],
    likwid: ["bars", "value"],
    roofline: [
      "points",
      "arithmetic_intensity_flops_per_byte",
      "performance_gflops",
    ],
    callgraph: [Array.isArray(payload.regions) ? "regions" : "calls", "depth"],
    region_statistics: ["files"],
  };
  const [key, ...required] = fields[kind] ?? [];
  if (!key) return;
  const rows = payload[key];
  if (!Array.isArray(rows)) throw new TypeError(`Expected a ${key} array.`);
  const fail = (index, field, reason) => {
    throw new TypeError(`${key}[${index}].${field}: ${reason}`);
  };
  rows.forEach((row, index) => {
    if (!row || typeof row !== "object" || Array.isArray(row))
      fail(index, "record", "must be an object");
    if (kind !== "region_statistics") {
      const name = kind === "callgraph" ? "name" : "region";
      if (typeof row[name] !== "string") fail(index, name, "must be a string");
    }
    if ((kind === "flame" || kind === "callgraph") && key === "calls") {
      for (const field of [
        "call_id",
        kind === "flame" ? "parent_call_id" : "parent_id",
      ]) {
        if (field !== "call_id" && row[field] == null) continue;
        if (typeof row[field] !== "string" && !Number.isFinite(row[field]))
          fail(index, field, "must be a string or finite number");
      }
    }
    for (const field of required) {
      if (!Number.isFinite(row[field]))
        fail(index, field, "must be a finite number");
    }
    for (const [field, value] of Object.entries(row)) {
      if (typeof value === "number" && !Number.isFinite(value))
        fail(index, field, "must be finite");
      if (
        value != null &&
        (field.endsWith("_seconds") || field === "rank") &&
        !Number.isFinite(value)
      )
        fail(index, field, "must be a finite number");
    }
    for (const [start, end] of [
      ["start_seconds", "end_seconds"],
      ["bin_low_seconds", "bin_high_seconds"],
      ["bin_start_seconds", "bin_end_seconds"],
    ]) {
      if (row[start] != null && row[end] != null && row[end] < row[start])
        fail(index, end, `must be >= ${start}`);
    }
    if (kind === "flame") {
      if (
        !Number.isFinite(row.inclusive_duration_seconds) &&
        !Number.isFinite(row.end_seconds)
      )
        fail(index, "end_seconds", "or inclusive_duration_seconds is required");
      if (row.inclusive_duration_seconds < 0)
        fail(index, "inclusive_duration_seconds", "must be nonnegative");
    }
    if (kind === "timeseries") {
      if (
        row.min_duration_seconds > row.mean_duration_seconds ||
        row.max_duration_seconds < row.mean_duration_seconds
      )
        fail(index, "bounds", "must bracket mean_duration_seconds");
    }
    if (kind === "rank_heatmap") {
      const field =
        options.valueKey ??
        Object.keys(row).find((name) => name.endsWith("_duration_seconds")) ??
        "total_duration_seconds";
      if (!Number.isFinite(row[field]))
        fail(index, field, "must be a finite number");
    }
    if (kind === "scaling") {
      const x = options.xField ?? payload.options?.x_field ?? "num_ranks";
      const y = options.yField;
      if (typeof row[x] !== "string" && !Number.isFinite(row[x]))
        fail(index, x, "must be a string or finite number");
      if (y && !Number.isFinite(row[y]))
        fail(index, y, "must be a finite number");
    }
    if (
      kind === "roofline" &&
      (row.arithmetic_intensity_flops_per_byte <= 0 ||
        row.performance_gflops <= 0)
    )
      fail(index, "rates", "must be positive on logarithmic axes");
    if (kind === "region_statistics") {
      if (
        !row.region_statistics ||
        typeof row.region_statistics !== "object" ||
        Array.isArray(row.region_statistics)
      )
        fail(index, "region_statistics", "must be an object");
      for (const [region, stats] of Object.entries(row.region_statistics)) {
        if (!stats || typeof stats !== "object")
          fail(index, region, "statistics must be an object");
        for (const [field, value] of Object.entries(stats)) {
          if (
            (field === "count" || field.endsWith("_seconds")) &&
            value != null &&
            !Number.isFinite(value)
          )
            fail(
              index,
              `${region}.${field}`,
              "must be a finite number or null",
            );
        }
      }
    }
  });
  if (kind === "callgraph" && payload.edges != null) {
    if (!Array.isArray(payload.edges))
      throw new TypeError("edges must be an array.");
    payload.edges.forEach((edge, index) => {
      if (
        !edge ||
        typeof edge.parent !== "string" ||
        typeof edge.child !== "string"
      )
        throw new TypeError(
          `edges[${index}]: parent and child must be strings.`,
        );
    });
  }
}

/** Refuse last-write-wins data loss. Keys are serialized tuples, not labels. */
function uniqueMap(entries, context) {
  const result = new Map();
  for (const [key, value] of entries) {
    if (result.has(key))
      throw new TypeError(`${context}: duplicate cell ${JSON.stringify(key)}`);
    result.set(key, value);
  }
  return result;
}

/** Iterative traversal also handles deeply nested profiles without stack overflow. */
function validateParents(calls, keyOf, parentOf) {
  const byKey = uniqueMap(
    calls.map((call) => [keyOf(call), call]),
    "calls (duplicate call ID)",
  );
  const done = new Set();
  for (const start of byKey.keys()) {
    const path = new Set();
    let key = start;
    while (key != null && byKey.has(key) && !done.has(key)) {
      if (path.has(key)) throw new TypeError(`calls: ancestor cycle at ${key}`);
      path.add(key);
      key = parentOf(byKey.get(key));
    }
    for (const visited of path) done.add(visited);
  }
  return byKey;
}

const DEFAULT_COLORS = [
  "#2a78d6",
  "#eb6834",
  "#1baf7a",
  "#eda100",
  "#e87ba4",
  "#008300",
  "#4a3aa7",
  "#e34948",
];

function colorMap(names, supplied = {}) {
  const map = new Map();
  for (const name of names) {
    if (map.has(name)) continue;
    map.set(
      name,
      (Object.hasOwn(supplied, name) ? supplied[name] : undefined) ??
        stableColor(name),
    );
  }
  return map;
}

function stableColor(name) {
  let hash = 2166136261;
  for (const char of String(name))
    hash = Math.imul(hash ^ char.codePointAt(0), 16777619);
  return DEFAULT_COLORS[(hash >>> 0) % DEFAULT_COLORS.length];
}

function interactionData(rows, metrics = () => []) {
  return rows.map((row) => ({
    ...metrics(row),
    identity: {
      region: row.region ?? row.name ?? null,
      file: row.file ?? "run",
      rank: row.rank ?? null,
      call_id: row.call_id ?? null,
    },
  }));
}

/** Read an identity from a Plotly click/hover/selection point. */
export function getPointIdentity(point) {
  return point?.customdata?.identity ?? null;
}

/** Share an explicit palette across independently built figures. */
export function createColorRegistry(names = [], supplied = {}) {
  return Object.freeze(Object.fromEntries(colorMap(names, supplied)));
}

function mergeLayout(base, overrides = {}) {
  const result = { ...base };
  for (const [key, value] of Object.entries(overrides)) {
    if (["__proto__", "constructor", "prototype"].includes(key)) continue;
    result[key] =
      value && Object.getPrototypeOf(value) === Object.prototype
        ? mergeLayout(
            result[key] &&
              typeof result[key] === "object" &&
              !Array.isArray(result[key])
              ? result[key]
              : {},
            value,
          )
        : cloneLayoutValue(value);
  }
  return result;
}

function cloneLayoutValue(value) {
  if (Array.isArray(value)) return value.map(cloneLayoutValue);
  if (value && Object.getPrototypeOf(value) === Object.prototype)
    return mergeLayout({}, value);
  return value;
}

// Region, file and series names come from the profiled application, so a
// hovertemplate that interpolates one is interpolating untrusted text. Plotly
// substitutes %{...} inside a template and renders what is left as HTML, so a
// region named "%{y}" rewrote every hover label that mentioned it, and one
// carrying a tag injected markup into them. Escape both spellings; the result
// is only ever used as literal text.
function label(name) {
  return String(name)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/%\{/g, "%&#123;");
}

function values(payload, key, kind, options) {
  if (!payload || !Array.isArray(payload[key]))
    throw new TypeError(
      `Expected a scope-profiler plot-data payload with a ${key} array.`,
    );
  if (kind) validateRecords(kind, payload, options);
  return payload[key];
}

// Chrome colours (text, gridlines, hover surface, the dashed ideal lines)
// for each theme a host page can be in. "auto" commits to nothing: it leaves
// the text colour unset and paints gridlines in a half-transparent grey that
// reads on either background, which is what every figure did before themes
// existed and so stays the default. A host that knows which theme it is in
// passes "light" or "dark" -- or its own token object -- and gets chrome that
// matches, since a grey that works on both is never the best on either.
const THEMES = {
  auto: {
    text: undefined,
    muted: undefined,
    grid: "rgba(128, 128, 128, 0.2)",
    hoverBg: undefined,
    neutral: "#777",
  },
  light: {
    text: "#111827",
    muted: "#6b7280",
    grid: "#e1e0d9",
    hoverBg: "#ffffff",
    neutral: "#777",
  },
  dark: {
    text: "#e5e7eb",
    muted: "#9ca3af",
    grid: "#2a2f3a",
    hoverBg: "#171a21",
    neutral: "#8b8b8b",
  },
};

let defaultTheme = "auto";

/** Set the theme every builder uses when its options do not name one.
 *
 * A page with a dark-mode toggle sets this once per toggle and rebuilds its
 * figures, instead of threading the theme through every build call.
 * Accepts a theme name or an object of token overrides.
 */
export function setTheme(theme) {
  defaultTheme =
    theme && typeof theme === "object" ? { ...theme } : (theme ?? "auto");
}

/** The theme tokens currently in effect, or those a build option resolves to. */
export function resolveTheme(theme = defaultTheme) {
  if (theme && typeof theme === "object") return { ...THEMES.auto, ...theme };
  return { ...(Object.hasOwn(THEMES, theme) ? THEMES[theme] : THEMES.auto) };
}

// Builders take their layout helpers from here rather than calling the
// module-level ones, so a theme reaches every layout and axis in a figure
// without being passed down to each call.
function palette(options) {
  const theme = resolveTheme(options?.theme);
  return {
    theme,
    baseLayout: (overrides = {}) =>
      mergeLayout(baseLayout(overrides, theme), options.layout),
    axis: (overrides = {}) => axis(overrides, theme),
  };
}

function baseLayout(overrides = {}, theme = resolveTheme()) {
  return {
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: {
      family: "Inter, ui-sans-serif, system-ui, sans-serif",
      size: 12,
      ...(theme.text ? { color: theme.text } : {}),
    },
    hovermode: "closest",
    hoverlabel: {
      namelength: -1,
      ...(theme.hoverBg
        ? {
            bgcolor: theme.hoverBg,
            bordercolor: theme.grid,
            font: { color: theme.text },
          }
        : {}),
    },
    margin: { l: 100, r: 24, t: 32, b: 64 },
    legend: {
      orientation: "h",
      y: -0.2,
      x: 0,
      ...(theme.muted ? { font: { color: theme.muted } } : {}),
    },
    ...overrides,
  };
}

function axis(overrides = {}, theme = resolveTheme()) {
  const styled = {
    automargin: true,
    gridcolor: theme.grid,
    zeroline: false,
    ...(theme.text ? { zerolinecolor: theme.grid, linecolor: theme.grid } : {}),
    ...(theme.muted ? { tickfont: { color: theme.muted } } : {}),
    ...overrides,
  };
  // Builders pass an axis title as a bare string. Keep that spelling but give
  // it the theme's muted colour, which a plain string cannot carry.
  if (theme.muted && styled.title != null)
    styled.title =
      typeof styled.title === "string"
        ? { text: styled.title, font: { color: theme.muted } }
        : {
            ...styled.title,
            font: { color: theme.muted, ...styled.title.font },
          };
  return styled;
}

function withEmptyState(layout, hasData) {
  if (hasData) return layout;
  return {
    ...layout,
    // Append rather than assign: an empty payload used to discard whatever
    // annotations the caller's own `layout` override had put here.
    annotations: [
      ...(Array.isArray(layout.annotations) ? layout.annotations : []),
      {
        text: "No data to display.",
        showarrow: false,
        xref: "paper",
        yref: "paper",
        x: 0.5,
        y: 0.5,
      },
    ],
    // A full grid with axis titles and no marks on it reads as a broken chart
    // rather than an empty one, so the message stands on its own. Figures
    // without cartesian axes (icicle, sankey) ignore these.
    xaxis: { ...layout.xaxis, visible: false },
    yaxis: { ...layout.yaxis, visible: false },
  };
}

// Pooled magnitude per region, largest first -- the order both the durations
// bars and the heatmap columns are laid out in, and the one
// `buildRegionSummaryFigure` has always ranked by.
function totalsByRegion(rows, value) {
  const totals = new Map();
  for (const row of rows)
    totals.set(row.region, (totals.get(row.region) ?? 0) + (value(row) ?? 0));
  return new Map([...totals].sort((a, b) => b[1] - a[1]));
}

function filtered(rows, options) {
  return typeof options?.filterRegion === "function"
    ? rows.filter((row) => options.filterRegion(row.region, row))
    : rows;
}

// One pass instead of a filter per series. A large trace has both many rows and
// many regions, so filtering the whole array once per series is quadratic: on
// 200k intervals over 400 regions that is the difference between ~690 ms and
// ~8 ms. Map preserves first-appearance order, which is the order the series
// are drawn and listed in.
function groupBy(rows, key) {
  const groups = new Map();
  for (const row of rows) {
    const name = key(row);
    const bucket = groups.get(name);
    if (bucket) bucket.push(row);
    else groups.set(name, [row]);
  }
  return groups;
}

// Several payloads carry rows from more than one run. Dropping that column
// merges runs into one series -- silently, and wrongly -- so every builder that
// can see two runs keys its series by file as well, and says so in the label.
function runAware(rows) {
  const files = new Set(rows.map((row) => row.file ?? "run"));
  const multi = files.size > 1;
  return {
    multi,
    files: [...files],
    label: (row) =>
      multi ? `${row.file ?? "run"} / ${row.region}` : row.region,
    key: (row) =>
      multi ? `${row.file ?? "run"}\u0000${row.region}` : row.region,
  };
}

// Region colour stays with the region across runs (and honours the payload's
// own colours), so a run is told apart by marker shape or bar pattern instead.
const FILE_SYMBOLS = ["circle", "square", "diamond", "triangle-up", "cross"];
const FILE_PATTERNS = ["", "/", "\\", "x", "-"];

/** Build a multi-run, multi-rank timeline: a lane per region and rank.
 *
 * A lane per rank alone cannot show a nested profile: every region of a rank
 * lands on one row, and the outermost region -- the session, typically -- is
 * drawn over everything inside it. One lane per region and rank is also what
 * `scope-profiler plot gantt` draws, so the two agree. Pass
 * `{ laneBy: "rank" }` for the compact one-row-per-rank view, which suits a
 * flat profile compared across many ranks.
 */
export function buildGanttFigure(payload, options = {}) {
  const { baseLayout, axis } = palette(options);
  const intervals = filtered(values(payload, "intervals", "gantt"), options);
  const byRegion = groupBy(intervals, (row) => row.region);
  const colors = colorMap(byRegion.keys(), options.colors ?? payload.colors);
  const multi = new Set(intervals.map((row) => row.file ?? "run")).size > 1;
  const rankLane = (row) => `${row.file ?? "run"} / rank ${row.rank ?? 0}`;
  // Matching `plot gantt`'s own lane label, extended by the run only when the
  // payload holds more than one.
  const regionLane = (row) =>
    `${multi ? `${row.file ?? "run"} / ` : ""}${row.region} (rank ${row.rank ?? 0})`;
  const laneOf = options.laneBy === "rank" ? rankLane : regionLane;
  const lanes = [...new Set(intervals.map(laneOf))];
  // A lane index rather than the lane string on every bar. The label is
  // already in `lanes`; repeating it per interval costs one string per row and
  // makes Plotly resolve a category for each of them, which a 200k-interval
  // trace feels. The axis carries the names back via ticktext.
  const laneIndex = new Map(lanes.map((lane, position) => [lane, position]));
  const data = [...byRegion].map(([region, rows]) => {
    return {
      type: "bar",
      orientation: "h",
      name: region,
      y: rows.map((row) => laneIndex.get(laneOf(row))),
      x: rows.map((row) => row.end_seconds - row.start_seconds),
      base: rows.map((row) => row.start_seconds),
      // A categorical axis gave every bar its slot; a linear one sizes bars
      // from the data, so the thickness has to be said out loud.
      width: 0.8,
      marker: {
        color: colors.get(region),
        line: { color: "rgba(0, 0, 0, 0.28)", width: 0.5 },
      },
      customdata: interactionData(rows, (row) => [
        label(row.file ?? "run"),
        row.rank ?? 0,
      ]),
      hovertemplate: `<b>${label(region)}</b><br>%{customdata[0]} / rank %{customdata[1]}<br>start: %{base:.6g} s<br>duration: %{x:.6g} s<extra></extra>`,
    };
  });
  const byRank = options.laneBy === "rank";
  const perLane = byRank ? 48 : 26;
  // Region lanes read bottom-up, so the first region -- the enclosing one --
  // sits at the bottom, as `scope-profiler plot gantt` draws it. Rank lanes
  // keep rank 0 on top, like the rank heatmap.
  const layout = baseLayout({
    barmode: "overlay",
    height: Math.max(280, perLane * lanes.length + 150),
    showlegend: byRegion.size > 1,
    xaxis: axis({ title: "Time (s)" }),
    yaxis: axis({
      tickmode: "array",
      tickvals: lanes.map((_, position) => position),
      ticktext: lanes,
      range: byRank ? [lanes.length - 0.5, -0.5] : [-0.5, lanes.length - 0.5],
      showgrid: false,
    }),
  });
  return { data, layout: withEmptyState(layout, intervals.length > 0) };
}

/** Build an icicle flame chart using scope-profiler's explicit call IDs. */
export function buildFlameFigure(payload, options = {}) {
  const { theme, baseLayout } = palette(options);
  const allCalls = values(payload, "calls", "flame");
  const calls = filtered(allCalls, options);
  const regions = [...new Set(calls.map((call) => call.region))];
  const colors = colorMap(regions, options.colors ?? payload.colors);
  const root = "scope-profiler-root";
  const callKey = (call) =>
    `${call.file ?? "run"}:${call.rank ?? 0}:${call.call_id}`;
  const parentKey = (call) =>
    call.parent_call_id == null
      ? null
      : `${call.file ?? "run"}:${call.rank ?? 0}:${call.parent_call_id}`;
  const duration = (call) =>
    call.inclusive_duration_seconds ?? call.end_seconds - call.start_seconds;
  // A filter can remove a call whose children survive; re-parent each survivor
  // onto its nearest surviving ancestor so the icicle stays a single tree
  // instead of silently dropping the orphans.
  const byKey = validateParents(allCalls, callKey, parentKey);
  const kept = new Set(calls.map(callKey));
  const anchor = (call) => {
    let key = parentKey(call);
    while (key != null && !kept.has(key))
      key = byKey.has(key) ? parentKey(byKey.get(key)) : null;
    return key ?? root;
  };
  const anchors = calls.map(anchor);
  const rootDuration = calls.reduce(
    (sum, call, index) =>
      anchors[index] === root ? sum + duration(call) : sum,
    0,
  );
  const ids = [root],
    labels = [options.rootLabel ?? "All calls"],
    parents = [""],
    markerColors = [theme.neutral],
    hovertext = ["All calls"];
  calls.forEach((call, index) => {
    ids.push(callKey(call));
    labels.push(call.region);
    parents.push(anchors[index]);
    markerColors.push(colors.get(call.region));
    hovertext.push(
      `<b>${label(call.region)}</b><br>${label(call.file ?? "run")} / rank ${call.rank ?? 0}<br>start: ${call.start_seconds.toPrecision(6)} s<br>inclusive: ${duration(call).toPrecision(6)} s`,
    );
  });
  const layout = baseLayout({
    height: 500,
    margin: { l: 24, r: 24, t: 24, b: 24 },
  });
  return {
    data: [
      {
        type: "icicle",
        ids,
        labels,
        parents,
        values: [rootDuration, ...calls.map(duration)],
        branchvalues: "total",
        tiling: { orientation: "h" },
        marker: {
          colors: markerColors,
          line: { color: "rgba(255, 255, 255, 0.55)", width: 1 },
        },
        hovertext,
        customdata: interactionData([{ region: null }, ...calls]),
        hoverinfo: "text",
      },
    ],
    layout: withEmptyState(layout, calls.length > 0),
  };
}

export function buildDurationsFigure(payload, options = {}) {
  const { baseLayout, axis } = palette(options);
  const metric =
    options.metric ??
    payload.options?.metric ??
    payload.metrics?.[0] ??
    "total";
  const bars = filtered(values(payload, "bars", "durations"), options).filter(
    (bar) => bar.metric === metric,
  );
  // A stacked-children export is already decomposed into segments. Preserve
  // that decomposition instead of letting duplicate region rows overwrite.
  const stacked = bars.some((bar) => bar.segment != null);
  const groups = groupBy(bars, (bar) =>
    stacked ? bar.segment : bar.rank == null ? bar.file : `rank ${bar.rank}`,
  );
  // Ranked by the pooled metric, so the costly regions lead. Appearance order
  // is whatever the exporter happened to walk, which with two runs of
  // different region sets interleaves them unpredictably.
  const regions = [...totalsByRegion(bars, (bar) => bar.value_seconds).keys()];
  // A group is a rank, a run, or a stacked child region depending on the
  // export, but a caller -- and the payload's own `colors` -- keys colours by
  // the name it knows. "rank 3" is a label this builder invents, so accept a
  // colour supplied under the bare rank rather than silently dropping to the
  // default cycle.
  const supplied = options.colors ?? payload.colors ?? {};
  const byGroup = {};
  for (const [group, rows] of groups) {
    const color = supplied[group] ?? supplied[rows[0].rank];
    if (color != null) byGroup[group] = color;
  }
  const colors = colorMap(groups.keys(), byGroup);
  const data = [...groups].map(([group, rows]) => {
    const byRegion = uniqueMap(
      rows.map((bar) => [bar.region, bar.value_seconds]),
      "durations",
    );
    return {
      type: "bar",
      name: group,
      x: regions,
      y: regions.map((region) => byRegion.get(region) ?? null),
      customdata: interactionData(
        regions.map((region) => ({
          ...rows.find((row) => row.region === region),
          region,
        })),
      ),
      marker: {
        color: colors.get(group),
        line: { color: "rgba(0, 0, 0, 0.22)", width: 0.5 },
      },
      hovertemplate: `<b>%{x}</b><br>${label(group)}: %{y:.6g} s<extra></extra>`,
    };
  });
  const layout = baseLayout({
    barmode: stacked ? "stack" : "group",
    height: Math.max(360, 34 * regions.length + 180),
    showlegend: groups.size > 1,
    xaxis: axis({ tickangle: -35 }),
    yaxis: axis({ title: `${metric} duration (s)` }),
  });
  return { data, layout: withEmptyState(layout, bars.length > 0) };
}

// The three scaling exports differ only in the y column they carry and the
// shape of their ideal line, so one builder serves all of them.
const SCALING_KINDS = {
  speedup: {
    yKey: "speedup",
    title: "Speedup",
    suffix: "×",
    idealName: "Ideal speedup",
    ideal: (value, baseline) => value / baseline,
  },
  weak_scaling: {
    yKey: "normalized_runtime",
    title: "Normalized runtime",
    suffix: "×",
    idealName: "Ideal weak scaling",
    ideal: () => 1,
  },
  scaling_efficiency: {
    yKey: "efficiency",
    title: "Scaling efficiency",
    suffix: "",
    idealName: "Ideal efficiency",
    ideal: () => 1,
  },
  // Shares its y column with scaling_efficiency, so it has to be named --
  // by the document's own `plot` field, or options.plot -- rather than
  // recognised from the rows. Listed after it so a payload with neither
  // still infers the strong-scaling reading it always did.
  weak_scaling_efficiency: {
    yKey: "efficiency",
    title: "Weak-scaling efficiency",
    suffix: "",
    idealName: "Ideal efficiency",
    ideal: () => 1,
  },
};

function scalingKind(payload, options) {
  const named = options.plot ?? payload?.plot;
  if (named && SCALING_KINDS[named]) return SCALING_KINDS[named];
  if (options.yField) {
    const match = Object.values(SCALING_KINDS).find(
      (kind) => kind.yKey === options.yField,
    );
    return (
      match ?? {
        ...SCALING_KINDS.speedup,
        yKey: options.yField,
        title: options.yField,
      }
    );
  }
  const row = payload?.points?.[0];
  return (
    (row &&
      Object.values(SCALING_KINDS).find((kind) => row[kind.yKey] != null)) ??
    SCALING_KINDS.speedup
  );
}

/** Build a scaling curve: speedup, weak scaling, or parallel efficiency. */
export function buildSpeedupFigure(payload, options = {}) {
  const { theme, baseLayout, axis } = palette(options);
  const kind = scalingKind(payload, options);
  const xField = options.xField ?? payload.options?.x_field ?? "num_ranks";
  const points = filtered(
    values(payload, "points", "scaling", { ...options, yField: kind.yKey }),
    options,
  );
  const byRegion = groupBy(points, (point) => point.region);
  const colors = colorMap(byRegion.keys(), options.colors ?? payload.colors);
  const xValues = [...new Set(points.map((point) => point[xField]))].sort(
    (a, b) =>
      typeof a === "number" && typeof b === "number"
        ? a - b
        : String(a).localeCompare(String(b)),
  );
  const order = new Map(xValues.map((value, position) => [value, position]));
  const numeric = xValues.every((value) => typeof value === "number");
  const data = [...byRegion].map(([region, unsorted]) => {
    const rows = [...unsorted].sort(
      (a, b) => order.get(a[xField]) - order.get(b[xField]),
    );
    return {
      type: "scatter",
      mode: "lines+markers",
      name: region,
      x: rows.map((row) => row[xField]),
      y: rows.map((row) => row[kind.yKey]),
      customdata: interactionData(rows),
      line: { color: colors.get(region), width: 2.4 },
      marker: { color: colors.get(region), size: 7 },
      hovertemplate: `<b>%{x}</b><br>${label(region)}: %{y:.3g}${kind.suffix}<extra></extra>`,
    };
  });
  const baseline = payload.options?.baseline ?? xValues[0];
  if (
    numeric &&
    points.length &&
    options.ideal !== false &&
    (!Number.isFinite(baseline) || baseline <= 0)
  )
    throw new TypeError("Scaling baseline must be a positive finite number.");
  if (numeric && options.ideal !== false)
    data.push({
      type: "scatter",
      mode: "lines",
      name: kind.idealName,
      x: xValues,
      y: xValues.map((value) => kind.ideal(value, baseline)),
      line: { color: theme.neutral, dash: "dash" },
      hoverinfo: "skip",
    });
  const layout = baseLayout({
    height: 420,
    showlegend: data.length > 1,
    xaxis: axis({
      title: payload.options?.x_label ?? xField,
      tickvals: xValues,
    }),
    yaxis: axis({ title: kind.title, rangemode: "tozero" }),
  });
  return { data, layout: withEmptyState(layout, points.length > 0) };
}

/** Build a weak-scaling curve (runtime normalized to the baseline scale). */
export function buildWeakScalingFigure(payload, options = {}) {
  return buildSpeedupFigure(payload, { ...options, plot: "weak_scaling" });
}

/** Build a parallel-efficiency curve (measured speedup over ideal speedup). */
export function buildScalingEfficiencyFigure(payload, options = {}) {
  return buildSpeedupFigure(payload, {
    ...options,
    plot: "scaling_efficiency",
  });
}

/** Build a weak-scaling efficiency curve (baseline runtime over runtime).
 *
 * For a study that grows the problem with the machine, where the ideal is
 * constant runtime -- not the rank-proportional speedup
 * `buildScalingEfficiencyFigure` measures against.
 */
export function buildWeakScalingEfficiencyFigure(payload, options = {}) {
  return buildSpeedupFigure(payload, {
    ...options,
    plot: "weak_scaling_efficiency",
  });
}

/** Build mean call duration over time, one trace per region. */
export function buildDurationTimeseriesFigure(payload, options = {}) {
  const { baseLayout, axis } = palette(options);
  const points = filtered(values(payload, "points", "timeseries"), options);
  const runs = runAware(points);
  const colors = colorMap(
    points.map((point) => point.region),
    options.colors ?? payload.colors,
  );
  const series = groupBy(points, runs.key);
  const data = [...series].map(([, unsorted]) => {
    const rows = [...unsorted].sort((a, b) => a.time_seconds - b.time_seconds);
    const region = rows[0].region,
      name = runs.label(rows[0]);
    return {
      type: "scatter",
      mode: "lines+markers",
      name,
      legendgroup: JSON.stringify([rows[0].file ?? "run", region]),
      x: rows.map((row) => row.time_seconds),
      y: rows.map((row) => row.mean_duration_seconds),
      ...(options.variability === "error"
        ? {
            error_y: {
              type: "data",
              symmetric: false,
              array: rows.map((row) =>
                row.max_duration_seconds == null
                  ? null
                  : row.max_duration_seconds - row.mean_duration_seconds,
              ),
              arrayminus: rows.map((row) =>
                row.min_duration_seconds == null
                  ? null
                  : row.mean_duration_seconds - row.min_duration_seconds,
              ),
            },
          }
        : {}),
      line: { color: colors.get(region), width: 2.2 },
      marker: {
        color: colors.get(region),
        size: 5,
        symbol:
          FILE_SYMBOLS[
            runs.files.indexOf(rows[0].file ?? "run") % FILE_SYMBOLS.length
          ],
      },
      customdata: interactionData(rows, (row) => [
        row.min_duration_seconds,
        row.max_duration_seconds,
        row.call_index,
      ]),
      hovertemplate: `<b>${label(name)}</b><br>time: %{x:.6g} s<br>mean: %{y:.6g} s<br>min–max: %{customdata[0]:.4g}–%{customdata[1]:.4g} s<extra></extra>`,
    };
  });
  if (options.variability === "band") {
    const bands = [...series].flatMap(([, unsorted]) => {
      const rows = [...unsorted].sort(
        (a, b) => a.time_seconds - b.time_seconds,
      );
      const common = {
        type: "scatter",
        mode: "lines",
        x: rows.map((row) => row.time_seconds),
        legendgroup: JSON.stringify([rows[0].file ?? "run", rows[0].region]),
        showlegend: false,
        hoverinfo: "skip",
        line: { width: 0, color: colors.get(rows[0].region) },
      };
      return [
        { ...common, y: rows.map((row) => row.min_duration_seconds ?? null) },
        {
          ...common,
          y: rows.map((row) => row.max_duration_seconds ?? null),
          fill: "tonexty",
          opacity: 0.2,
        },
      ];
    });
    data.unshift(...bands);
  }
  const layout = baseLayout({
    height: 420,
    showlegend: series.size > 1,
    xaxis: axis({ title: "Time (s)" }),
    yaxis: axis({ title: "Mean call duration (s)" }),
  });
  return { data, layout: withEmptyState(layout, points.length > 0) };
}

/** Build duration distributions from histogram bin records. */
export function buildHistogramFigure(payload, options = {}) {
  const { baseLayout, axis } = palette(options);
  const bins = filtered(values(payload, "bins", "histogram"), options);
  const runs = runAware(bins);
  const colors = colorMap(
    bins.map((bin) => bin.region),
    options.colors ?? payload.colors,
  );
  const series = groupBy(bins, runs.key);
  const data = [...series].map(([, unsorted]) => {
    const rows = [...unsorted].sort(
      (a, b) => a.bin_center_seconds - b.bin_center_seconds,
    );
    const region = rows[0].region,
      name = runs.label(rows[0]);
    const pattern =
      FILE_PATTERNS[
        runs.files.indexOf(rows[0].file ?? "run") % FILE_PATTERNS.length
      ];
    return {
      type: "bar",
      name,
      x: rows.map((bin) => bin.bin_center_seconds),
      y: rows.map((bin) => bin.count),
      customdata: interactionData(rows),
      width: rows.map((bin) => bin.bin_high_seconds - bin.bin_low_seconds),
      marker: {
        color: colors.get(region),
        line: { color: "rgba(0, 0, 0, 0.2)", width: 0.5 },
        ...(runs.multi ? { pattern: { shape: pattern, solidity: 0.35 } } : {}),
      },
      hovertemplate: `<b>${label(name)}</b><br>%{x:.6g} s: %{y} calls<extra></extra>`,
    };
  });
  const layout = baseLayout({
    barmode: "overlay",
    height: 400,
    showlegend: series.size > 1,
    xaxis: axis({ title: "Call duration (s)" }),
    yaxis: axis({ title: "Calls" }),
  });
  return { data, layout: withEmptyState(layout, bins.length > 0) };
}

/** Build a rank × region heatmap from duration records. */
export function buildRankHeatmapFigure(payload, options = {}) {
  const { baseLayout, axis } = palette(options);
  const points = filtered(
    values(payload, "points", "rank_heatmap", options),
    options,
  );
  const multi = new Set(points.map((point) => point.file ?? "run")).size > 1;
  // A lane per run and rank. Keying cells by rank alone silently let a second
  // run overwrite the first, showing one run's numbers under both labels.
  const laneOf = (point) =>
    multi
      ? `${point.file ?? "run"} / rank ${point.rank ?? 0}`
      : String(point.rank ?? 0);
  const lanes = [...new Set(points.map(laneOf))].sort((a, b) =>
    a.localeCompare(b, undefined, { numeric: true }),
  );
  const inferredValueKey = points[0]
    ? Object.keys(points[0]).find((key) => key.endsWith("_duration_seconds"))
    : undefined;
  const valueKey =
    options.valueKey ?? inferredValueKey ?? "total_duration_seconds";
  // Columns ranked by pooled duration rather than by the order the exporter
  // happened to walk, which with two runs of different region sets interleaves
  // them differently every time.
  const regions = [
    ...totalsByRegion(points, (point) => point[valueKey]).keys(),
  ];
  const byCell = uniqueMap(
    points.map((point) => [`${laneOf(point)}\u0000${point.region}`, point]),
    "rank_heatmap",
  );
  const data = [
    {
      type: "heatmap",
      x: regions,
      y: lanes,
      z: lanes.map((lane) =>
        regions.map(
          (region) => byCell.get(`${lane}\u0000${region}`)?.[valueKey] ?? null,
        ),
      ),
      customdata: lanes.map((lane) =>
        regions.map(
          (region) =>
            interactionData([
              { ...byCell.get(`${lane}\u0000${region}`), region },
            ])[0],
        ),
      ),
      colorscale: options.colorscale ?? "Viridis",
      colorbar: { title: "Seconds" },
      hovertemplate: `${multi ? "%{y}" : "rank %{y}"}<br>%{x}: %{z:.6g} s<extra></extra>`,
    },
  ];
  const layout = baseLayout({
    height: Math.max(320, 44 * lanes.length + 150),
    xaxis: axis({ title: "Region" }),
    yaxis: axis({
      title: multi ? "Run / rank" : "Rank",
      autorange: "reversed",
      showgrid: false,
    }),
  });
  return { data, layout: withEmptyState(layout, points.length > 0) };
}

/** Build per-rank duration lines, with a dashed rank mean for each region. */
export function buildImbalanceFigure(payload, options = {}) {
  const { baseLayout, axis } = palette(options);
  const points = filtered(values(payload, "points", "imbalance"), options);
  const runs = runAware(points);
  const colors = colorMap(
    points.map((point) => point.region),
    options.colors ?? payload.colors,
  );
  // The mean is computed per run, so a run gets its own line and its own mean;
  // pooling them drew one zig-zagging series that revisited every rank.
  const series = groupBy(points, runs.key);
  const data = [...series].flatMap(([, unsorted]) => {
    const rows = [...unsorted].sort((a, b) => a.rank - b.rank);
    const region = rows[0].region,
      name = runs.label(rows[0]);
    const color = colors.get(region);
    const symbol =
      FILE_SYMBOLS[
        runs.files.indexOf(rows[0].file ?? "run") % FILE_SYMBOLS.length
      ];
    return [
      {
        type: "scatter",
        mode: "lines+markers",
        name,
        legendgroup: JSON.stringify([rows[0].file ?? "run", region]),
        x: rows.map((row) => row.rank),
        y: rows.map((row) => row.value_seconds),
        customdata: interactionData(rows),
        line: { color, width: 2.2 },
        marker: { color, size: 7, symbol },
        hovertemplate: `<b>${label(name)}</b><br>rank %{x}: %{y:.6g} s<extra></extra>`,
      },
      {
        type: "scatter",
        mode: "lines",
        name: `${name} mean`,
        legendgroup: JSON.stringify([rows[0].file ?? "run", region]),
        x: rows.map((row) => row.rank),
        y: rows.map((row) => row.mean_over_ranks_seconds),
        customdata: interactionData(rows),
        line: { color, dash: "dot", width: 1.3 },
        hoverinfo: "skip",
        showlegend: false,
      },
    ];
  });
  const layout = baseLayout({
    height: 420,
    showlegend: series.size > 1,
    xaxis: axis({ title: "Rank", dtick: 1 }),
    yaxis: axis({ title: `${payload.metric ?? "Duration"} (s)` }),
  });
  return { data, layout: withEmptyState(layout, points.length > 0) };
}

/** Build a timeline-occupancy heatmap from binned density records. */
export function buildDensityFigure(payload, options = {}) {
  const { baseLayout, axis } = palette(options);
  const points = filtered(values(payload, "points", "density"), options);
  const lane = (point) => `${point.file ?? "run"} / ${point.region}`;
  const lanes = [...new Set(points.map(lane))];
  // Each cell sits at the centre of its own bin. Deriving one bin width from
  // the first point and applying it to every lane put the second run's cells
  // at the wrong times whenever two runs of different length were binned into
  // the same number of bins -- which is exactly what the exporter does.
  const centre = (point) =>
    (point.bin_start_seconds + point.bin_end_seconds) / 2;
  const centres = [...new Set(points.map(centre))].sort((a, b) => a - b);
  // Occupancy is the share of the bin the region was inside, which compares
  // across runs of different length; raw seconds stay available via valueKey.
  const asFraction = (options.valueKey ?? "occupancy") === "occupancy";
  // NUL-joined: a lane name is a file and a region, either of which may hold
  // the ":" that used to separate the two halves of this key.
  const byCell = uniqueMap(
    points.map((point) => [`${lane(point)}\u0000${centre(point)}`, point]),
    "density",
  );
  const cell = (laneName, at, pick) => {
    const point = byCell.get(`${laneName}\u0000${at}`);
    return point ? pick(point) : null;
  };
  const span = (point) => point.bin_end_seconds - point.bin_start_seconds;
  const data = [
    {
      type: "heatmap",
      x: centres,
      y: lanes,
      z: lanes.map((laneName) =>
        centres.map((at) =>
          cell(laneName, at, (point) =>
            asFraction
              ? span(point) > 0
                ? point.occupied_seconds / span(point)
                : null
              : point.occupied_seconds,
          ),
        ),
      ),
      customdata: lanes.map((laneName) =>
        centres.map((at) =>
          cell(
            laneName,
            at,
            (point) =>
              interactionData([point], (row) => [row.occupied_seconds])[0],
          ),
        ),
      ),
      colorscale: options.colorscale ?? "Viridis",
      ...(asFraction ? { zmin: 0, zmax: 1 } : {}),
      colorbar: { title: asFraction ? "Occupancy" : "Seconds" },
      hovertemplate: `%{y}<br>t = %{x:.6g} s<br>${asFraction ? "occupancy: %{z:.3f}<br>" : ""}occupied: %{customdata[0]:.4g} s<extra></extra>`,
    },
  ];
  // Plotly infers edges from centers. That is only accurate on one uniform
  // grid. For unequal grids use one trace per lane with explicit bin edges;
  // this preserves measured boundaries and never invents interpolated data.
  const grids = [...groupBy(points, lane)].map(([name, rows]) => {
    rows = [...rows].sort((a, b) => a.bin_start_seconds - b.bin_start_seconds);
    for (let i = 1; i < rows.length; i++) {
      if (rows[i].bin_start_seconds < rows[i - 1].bin_end_seconds)
        throw new TypeError(`density: overlapping bins in ${name}`);
    }
    return [name, rows];
  });
  const widths = new Set(points.map(span));
  const sameGrid =
    new Set(
      grids.map(([, rows]) =>
        JSON.stringify(
          rows.map((row) => [row.bin_start_seconds, row.bin_end_seconds]),
        ),
      ),
    ).size <= 1;
  const explicitEdges =
    !sameGrid ||
    widths.size > 1 ||
    grids.some(([, rows]) =>
      rows.some(
        (row, index) =>
          index > 0 &&
          row.bin_start_seconds !== rows[index - 1].bin_end_seconds,
      ),
    );
  if (explicitEdges) {
    const template = data[0];
    const maximumSeconds = points.reduce(
      (maximum, row) => Math.max(maximum, row.occupied_seconds),
      0,
    );
    data.splice(
      0,
      data.length,
      ...grids.map(([name, rows], index) => {
        const edges = [
          ...new Set(
            rows.flatMap((row) => [row.bin_start_seconds, row.bin_end_seconds]),
          ),
        ].sort((a, b) => a - b);
        const byStart = new Map(
          rows.map((row) => [row.bin_start_seconds, row]),
        );
        const cells = edges.slice(0, -1).map((start) => byStart.get(start));
        return {
          ...template,
          x: edges,
          y: [index - 0.5, index + 0.5],
          showscale: index === 0,
          z: [
            cells.map((row) =>
              row
                ? asFraction
                  ? span(row) > 0
                    ? row.occupied_seconds / span(row)
                    : null
                  : row.occupied_seconds
                : null,
            ),
          ],
          customdata: [
            cells.map((row) =>
              row
                ? interactionData([row], (point) => [point.occupied_seconds])[0]
                : null,
            ),
          ],
          hovertemplate: `<b>${label(name)}</b><br>t = %{x:.6g} s<br>${asFraction ? "occupancy: %{z:.3f}<br>" : ""}occupied: %{customdata[0]:.4g} s<extra></extra>`,
          ...(!asFraction
            ? {
                zmin: 0,
                zmax: maximumSeconds,
              }
            : {}),
        };
      }),
    );
  }
  const layout = baseLayout({
    height: Math.max(320, 34 * lanes.length + 150),
    xaxis: axis({ title: "Time (s)" }),
    yaxis: axis({
      categoryorder: "array",
      categoryarray: lanes,
      autorange: "reversed",
      showgrid: false,
      ...(explicitEdges
        ? {
            type: "linear",
            tickmode: "array",
            tickvals: lanes.map((_, index) => index),
            ticktext: lanes,
            range: [lanes.length - 0.5, -0.5],
            autorange: false,
          }
        : {}),
    }),
  });
  return { data, layout: withEmptyState(layout, points.length > 0) };
}

/** Axis label for each field a region_statistics document stores. */
const SUMMARY_LABELS = {
  count: "Calls",
  average_duration_seconds: "Average duration (s)",
  min_duration_seconds: "Minimum duration (s)",
  max_duration_seconds: "Maximum duration (s)",
  first_duration_seconds: "First call duration (s)",
  last_duration_seconds: "Last call duration (s)",
  std_duration_seconds: "Duration std. dev. (s)",
  total_duration_seconds: "Total duration (s)",
};

/** Short metric names accepted by the region_statistics builders.
 *
 * The names the CLI and the durations export use, mapped to the field they are
 * stored under in a region_statistics document, so a caller can say "total"
 * wherever it says "total" everywhere else. The stored field names are
 * accepted too; exported so a page can offer the list rather than guess it.
 */
export const SUMMARY_METRICS = Object.freeze({
  avg: "average_duration_seconds",
  min: "min_duration_seconds",
  max: "max_duration_seconds",
  total: "total_duration_seconds",
  first: "first_duration_seconds",
  last: "last_duration_seconds",
  std: "std_duration_seconds",
  count: "count",
});

// Pick the runs to draw, in the order asked for. `files` names them by label
// (or by index), which is how a page lets a viewer compare two runs out of a
// document that holds many.
function selectFiles(files, selection) {
  if (!Array.isArray(selection)) return files;
  return selection
    .map((wanted) =>
      typeof wanted === "number"
        ? files[wanted]
        : files.find((file) => (file.label ?? "run") === wanted),
    )
    .filter(Boolean);
}

/** Build a ranked region bar chart from a region_statistics document. */
export function buildRegionSummaryFigure(payload, options = {}) {
  const { baseLayout, axis } = palette(options);
  const files = selectFiles(
    values(payload, "files", "region_statistics"),
    options.files,
  );
  const metric =
    SUMMARY_METRICS[options.metric] ??
    options.metric ??
    "total_duration_seconds";
  // An unrecognised metric used to draw a full chart of nulls, which reads as
  // "this run recorded nothing" rather than "that is not a metric".
  if (!(metric in SUMMARY_LABELS))
    throw new TypeError(
      `Unknown region-statistics metric ${JSON.stringify(options.metric)}; expected one of ${[
        ...Object.keys(SUMMARY_METRICS),
        ...Object.keys(SUMMARY_LABELS),
      ]
        .map((name) => JSON.stringify(name))
        .join(", ")}.`,
    );
  const limit = options.topN ?? 20;
  const horizontal = (options.orientation ?? "h") === "h";
  const keep =
    typeof options.filterRegion === "function"
      ? options.filterRegion
      : () => true;
  // Comparing runs is only meaningful over the regions they share: a region
  // one run never entered would otherwise draw as a bar of zero against a
  // real one, which reads as "got faster" rather than "not measured here".
  const common =
    options.commonRegionsOnly && files.length > 1
      ? (region) =>
          files.every((file) => (file.region_statistics ?? {})[region] != null)
      : () => true;
  const totals = new Map();
  for (const file of files) {
    for (const [region, stats] of Object.entries(
      file.region_statistics ?? {},
    )) {
      if (!keep(region, stats) || !common(region)) continue;
      totals.set(region, (totals.get(region) ?? 0) + (stats[metric] ?? 0));
    }
  }
  // Rank by the pooled metric so the slowest regions lead, then keep the head
  // of the list: a long run has more regions than a bar chart can carry.
  const regions = [...totals.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([region]) => region);
  const labels = files.map((file) => file.label ?? "run");
  const colors = colorMap(labels, options.colors ?? payload.colors);
  const unit = metric === "count" ? "" : " s";
  const values_ = (stats) =>
    regions.map((region) => stats[region]?.[metric] ?? null);
  const data = files.map((file, index) => {
    const stats = file.region_statistics ?? {};
    const magnitudes = values_(stats);
    return {
      type: "bar",
      ...(horizontal
        ? { orientation: "h", y: regions, x: magnitudes }
        : { x: regions, y: magnitudes }),
      name: labels[index],
      marker: {
        color: colors.get(labels[index]),
        line: { color: "rgba(0, 0, 0, 0.22)", width: 0.5 },
      },
      customdata: interactionData(
        regions.map((region) => ({ region, file: file.label ?? "run" })),
        (row) => [stats[row.region]?.count ?? null],
      ),
      hovertemplate: horizontal
        ? `<b>%{y}</b><br>${label(labels[index])}: %{x:.6g}${unit}<br>calls: %{customdata[0]}<extra></extra>`
        : `<b>%{x}</b><br>${label(labels[index])}: %{y:.6g}${unit}<br>calls: %{customdata[0]}<extra></extra>`,
    };
  });
  const magnitudeAxis = axis({ title: SUMMARY_LABELS[metric] ?? metric });
  const categoryAxis = axis({
    categoryorder: "array",
    // A horizontal bar chart fills from the bottom up, so the ranking has to
    // be reversed to read top-down; a vertical one already reads left to right.
    categoryarray: horizontal ? [...regions].reverse() : regions,
    showgrid: false,
    ...(horizontal ? {} : { tickangle: -35 }),
  });
  const layout = baseLayout({
    barmode: "group",
    ...(horizontal
      ? {
          height: Math.max(320, 26 * regions.length + 160),
          xaxis: magnitudeAxis,
          yaxis: categoryAxis,
        }
      : {
          height: Math.max(360, 26 * regions.length + 200),
          margin: { l: 100, r: 24, t: 32, b: 140 },
          xaxis: categoryAxis,
          yaxis: magnitudeAxis,
        }),
    showlegend: files.length > 1,
  });
  return { data, layout: withEmptyState(layout, regions.length > 0) };
}

/** Build a side-by-side comparison of two runs in a region_statistics document.
 *
 * The same bars as `buildRegionSummaryFigure`, narrowed to the runs named in
 * `options.files` and to the regions both of them recorded, and drawn
 * vertically with every shared region kept rather than a ranked top slice --
 * the reading for "what changed between these two runs?" rather than "where
 * did this run spend its time?".
 */
export function buildComparisonFigure(payload, options = {}) {
  if (options.comparison && options.comparison !== "side-by-side") {
    const files = selectFiles(
      values(payload, "files", "region_statistics"),
      options.files ?? [0, 1],
    );
    if (files.length !== 2)
      throw new TypeError("Delta comparison requires exactly two runs.");
    const metric =
      SUMMARY_METRICS[options.metric] ??
      options.metric ??
      "total_duration_seconds";
    if (!Object.hasOwn(SUMMARY_LABELS, metric))
      throw new TypeError(`Unknown region-statistics metric ${metric}`);
    if (!["absolute", "percent"].includes(options.comparison))
      throw new TypeError(
        "comparison must be side-by-side, absolute, or percent.",
      );
    const [before, after] = files.map((file) => file.region_statistics);
    const regions = [
      ...new Set([...Object.keys(before), ...Object.keys(after)]),
    ].filter(
      (region) =>
        !options.filterRegion ||
        options.filterRegion(region, after[region] ?? before[region]),
    );
    const rows = regions.map((region) => {
      const baseline = before[region]?.[metric],
        candidate = after[region]?.[metric];
      const missing = baseline == null || candidate == null;
      const delta = missing ? null : candidate - baseline;
      const value =
        missing || (options.comparison === "percent" && baseline === 0)
          ? null
          : options.comparison === "percent"
            ? (delta / baseline) * 100
            : delta;
      return {
        region,
        file: files[1].label ?? "run",
        baseline,
        candidate,
        value,
        reason: missing
          ? "Not measured in both runs"
          : value == null
            ? "Zero baseline: percentage undefined"
            : "",
      };
    });
    if (options.sortBy !== "name")
      rows.sort((a, b) => (b.value ?? -Infinity) - (a.value ?? -Infinity));
    else rows.sort((a, b) => a.region.localeCompare(b.region));
    const selected = rows.slice(0, options.topN ?? Infinity);
    const { baseLayout, axis } = palette(options);
    return {
      data: [
        {
          type: "bar",
          name: "Change",
          x: selected.map((row) => row.region),
          y: selected.map((row) => row.value),
          customdata: interactionData(selected, (row) => [
            row.baseline ?? null,
            row.candidate ?? null,
            row.reason,
          ]),
          hovertemplate:
            "%{x}<br>change: %{y:.6g}<br>baseline: %{customdata[0]}<br>candidate: %{customdata[1]}<br>%{customdata[2]}<extra></extra>",
        },
      ],
      diagnostics: selected
        .filter((row) => row.value == null)
        .map((row) => ({
          code: "undefined-comparison",
          region: row.region,
          message: row.reason,
        })),
      layout: withEmptyState(
        baseLayout({
          xaxis: axis({ title: "Region" }),
          yaxis: axis({
            title:
              options.comparison === "percent"
                ? "Change (%)"
                : `Change in ${SUMMARY_LABELS[metric]}`,
          }),
        }),
        selected.some((row) => row.value != null),
      ),
    };
  }
  return buildRegionSummaryFigure(payload, {
    orientation: "v",
    topN: Infinity,
    commonRegionsOnly: true,
    ...options,
    files: options.files ?? [0, 1],
  });
}

/** Build a Sankey call graph from either callgraph export shape.
 *
 * The compact export collapses repeated invocations into one node per region,
 * which can turn recursion into a cycle; a Sankey cannot draw one, so links
 * that do not increase call depth are dropped. Use the flame chart to see
 * recursion in full.
 */
export function buildCallgraphFigure(payload, options = {}) {
  const { baseLayout } = palette(options);
  const compact = Array.isArray(payload?.regions);
  if (!compact && !Array.isArray(payload?.calls))
    throw new TypeError(
      "Expected a scope-profiler plot-data payload with a regions or calls array.",
    );
  validateRecords("callgraph", payload, options);
  const diagnostics = [];
  const keep =
    typeof options.filterRegion === "function"
      ? options.filterRegion
      : () => true;
  const weightKey = options.valueKey ?? "total_duration";
  let nodes, links, unit;
  if (compact) {
    const regions = payload.regions.filter((region) =>
      keep(region.name, region),
    );
    const depths = uniqueMap(
      regions.map((region) => [region.name, region.depth]),
      "callgraph nodes",
    );
    const weights = new Map(
      regions.map((region) => [region.name, region[weightKey]]),
    );
    nodes = regions.map((region) => region.name);
    links = (payload.edges ?? [])
      .filter(({ parent, child }) => depths.has(parent) && depths.has(child))
      .map((edge) => {
        const { parent, child } = edge;
        const incoming = (payload.edges ?? []).filter(
          (other) => other.child === child && other.parent !== child,
        ).length;
        let value = edge[weightKey] ?? edge.value;
        if (value == null && incoming === 1) {
          value = weights.get(child);
          if (value != null)
            diagnostics.push({
              code: "inferred-edge-weight",
              source: parent,
              target: child,
              message:
                "Weight inferred from child total with one incoming edge.",
            });
        }
        if (value == null) {
          diagnostics.push({
            code: "missing-edge-weight",
            source: parent,
            target: child,
            message:
              "Edge omitted: no measured weight and child total cannot be attributed.",
          });
          return null;
        }
        if (!Number.isFinite(value) || value < 0)
          throw new TypeError(
            `callgraph edge ${parent} -> ${child}: weight must be finite and nonnegative`,
          );
        return { source: parent, target: child, value };
      })
      .filter(Boolean);
    unit = weightKey.endsWith("duration") ? " s" : "";
  } else {
    const calls = payload.calls.filter((call) => keep(call.name, call));
    const callKey = (call) =>
      JSON.stringify([call.file ?? "run", call.rank ?? 0, call.call_id]);
    const parentKey = (call) =>
      call.parent_id == null
        ? null
        : JSON.stringify([call.file ?? "run", call.rank ?? 0, call.parent_id]);
    validateParents(payload.calls, callKey, parentKey);
    const byId = new Map(calls.map((call) => [callKey(call), call]));
    const counts = new Map();
    for (const call of calls) {
      const parent = byId.get(parentKey(call));
      if (!parent || call.depth <= parent.depth) continue;
      const key = JSON.stringify([parent.name, call.name]);
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    nodes = [...new Set(calls.map((call) => call.name))];
    links = [...counts.entries()].map(([key, value]) => {
      const [source, target] = JSON.parse(key);
      return { source, target, value };
    });
    unit = " calls";
  }
  const adjacency = new Map(nodes.map((name) => [name, []]));
  links = links.filter((link) => {
    const pending = [link.target],
      visited = new Set();
    while (pending.length) {
      const node = pending.pop();
      if (node === link.source) {
        diagnostics.push({
          code: "cycle-edge-omitted",
          source: link.source,
          target: link.target,
          message:
            "Recursive relationship omitted from the Sankey; use the flame chart for full ancestry.",
        });
        return false;
      }
      if (visited.has(node)) continue;
      visited.add(node);
      pending.push(...adjacency.get(node));
    }
    adjacency.get(link.source).push(link.target);
    return true;
  });
  const index = new Map(nodes.map((name, position) => [name, position]));
  const colors = colorMap(nodes, options.colors ?? payload.colors);
  const data = [
    {
      type: "sankey",
      visible: links.some((link) => link.value > 0),
      orientation: "h",
      node: {
        label: nodes,
        customdata: interactionData(nodes.map((name) => ({ name }))),
        color: nodes.map((name) => colors.get(name)),
        pad: 14,
        thickness: 16,
        line: { color: "rgba(0, 0, 0, 0.25)", width: 0.5 },
      },
      link: {
        customdata: links.map((link) => ({
          identity: {
            region: link.target,
            source: link.source,
            file: null,
            rank: null,
            call_id: null,
          },
        })),
        source: links.map((link) => index.get(link.source)),
        target: links.map((link) => index.get(link.target)),
        value: links.map((link) => link.value),
        hovertemplate: `%{source.label} \u2192 %{target.label}<br>%{value:.6g}${unit}<extra></extra>`,
      },
    },
  ];
  const layout = baseLayout({
    height: Math.max(320, 26 * nodes.length + 160),
    margin: { l: 24, r: 24, t: 24, b: 24 },
  });
  return {
    data,
    layout: withEmptyState(
      layout,
      links.some((link) => link.value > 0),
    ),
    diagnostics,
  };
}

/** Build a grouped bar chart of one LIKWID hardware-counter metric. */
export function buildLikwidFigure(payload, options = {}) {
  const { baseLayout, axis } = palette(options);
  const bars = filtered(values(payload, "bars", "likwid"), options);
  const series = groupBy(bars, (bar) => bar.series);
  const regions = [...new Set(bars.map((bar) => bar.region))];
  const colors = colorMap(series.keys(), options.colors ?? payload.colors);
  const metric = options.metric ?? payload.metric ?? "value";
  const data = [...series].map(([name, rows]) => {
    const byRegion = uniqueMap(
      rows.map((bar) => [bar.region, bar.value]),
      "likwid",
    );
    return {
      type: "bar",
      name,
      x: regions,
      y: regions.map((region) => byRegion.get(region) ?? null),
      customdata: interactionData(
        regions.map((region) => ({
          ...rows.find((row) => row.region === region),
          region,
        })),
      ),
      marker: {
        color: colors.get(name),
        line: { color: "rgba(0, 0, 0, 0.22)", width: 0.5 },
      },
      hovertemplate: `<b>%{x}</b><br>${label(name)}: %{y:.6g}<extra></extra>`,
    };
  });
  const layout = baseLayout({
    barmode: "group",
    height: Math.max(360, 34 * regions.length + 180),
    showlegend: series.size > 1,
    xaxis: axis({ tickangle: -35 }),
    yaxis: axis({
      title: metric,
      ...(options.logScale ? { type: "log" } : {}),
    }),
  });
  return { data, layout: withEmptyState(layout, bars.length > 0) };
}

/** Build a log-log roofline plot from per-region LIKWID-derived rates. */
export function buildRooflineFigure(payload, options = {}) {
  const { theme, baseLayout, axis } = palette(options);
  const points = filtered(values(payload, "points", "roofline"), options);
  const series = groupBy(points, (point) => point.file ?? "run");
  const colors = colorMap(series.keys(), options.colors ?? payload.colors);
  const data = [...series].map(([name, rows]) => ({
    type: "scatter",
    mode: "markers",
    name,
    x: rows.map((point) => point.arithmetic_intensity_flops_per_byte),
    y: rows.map((point) => point.performance_gflops),
    marker: { color: colors.get(name), size: 9 },
    customdata: interactionData(rows, (point) => [
      label(point.region),
      point.rank,
      point.bandwidth_gbs,
    ]),
    hovertemplate:
      "<b>%{customdata[0]}</b> (rank %{customdata[1]})" +
      "<br>intensity: %{x:.6g} FLOP/byte" +
      "<br>performance: %{y:.6g} GFLOP/s" +
      "<br>bandwidth: %{customdata[2]:.6g} GB/s<extra></extra>",
  }));
  const roof = payload.roofline ?? [];
  if (!Array.isArray(roof)) throw new TypeError("roofline must be an array.");
  validateRecords("roofline", {
    points: roof.map((point) => ({ ...point, region: "ceiling" })),
  });
  if (roof.length) {
    data.push({
      type: "scatter",
      mode: "lines",
      name: payload.empirical_ceilings ? "empirical roof" : "roofline",
      x: roof.map((point) => point.arithmetic_intensity_flops_per_byte),
      y: roof.map((point) => point.performance_gflops),
      // The one line in the package that used to be a fixed near-black, which
      // is invisible on the dark theme this builder otherwise honours.
      line: { color: theme.text ?? theme.neutral, width: 2, dash: "dash" },
      hovertemplate:
        "intensity: %{x:.6g} FLOP/byte<br>ceiling: %{y:.6g} GFLOP/s<extra></extra>",
    });
  }
  const layout = baseLayout({
    title: {
      text: payload.empirical_ceilings
        ? "Roofline analysis (empirical ceilings)"
        : "Roofline analysis",
      ...(theme.text ? { font: { color: theme.text } } : {}),
    },
    // This is the one figure that carries a title, and baseLayout's 32px top
    // margin is sized for the seventeen that do not.
    margin: { l: 100, r: 24, t: 56, b: 64 },
    xaxis: axis({ title: "Arithmetic intensity [FLOP/byte]", type: "log" }),
    yaxis: axis({ title: "Attained performance [GFLOP/s]", type: "log" }),
    showlegend: series.size > 1 || roof.length > 0,
  });
  return { data, layout: withEmptyState(layout, points.length > 0) };
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
  weak_scaling_efficiency: buildSpeedupFigure,
  rank_heatmap: buildRankHeatmapFigure,
  histogram: buildHistogramFigure,
  imbalance: buildImbalanceFigure,
  likwid: buildLikwidFigure,
  roofline: buildRooflineFigure,
  region_statistics: buildRegionSummaryFigure,
};

const PLOT_ARRAYS = {
  gantt: "intervals",
  density: "points",
  flame: "calls",
  flame_chart: "calls",
  flame_graph: "calls",
  durations: "bars",
  timeseries: "points",
  speedup: "points",
  weak_scaling: "points",
  scaling_efficiency: "points",
  weak_scaling_efficiency: "points",
  rank_heatmap: "points",
  histogram: "bins",
  imbalance: "points",
  likwid: "bars",
  roofline: "points",
  region_statistics: "files",
};

/** Guess the plot kind of a payload written before the envelope existed. */
export function inferPlotKind(payload) {
  if (!payload || typeof payload !== "object") return undefined;
  if (Array.isArray(payload.intervals)) return "gantt";
  if (Array.isArray(payload.bins)) return "histogram";
  if (Array.isArray(payload.files) && payload.files[0]?.region_statistics)
    return "region_statistics";
  if (Array.isArray(payload.regions) && Array.isArray(payload.edges))
    return "callgraph";
  if (Array.isArray(payload.bars))
    return payload.bars[0]?.series != null ? "likwid" : "durations";
  if (Array.isArray(payload.calls))
    return payload.calls[0]?.parent_id !== undefined ? "callgraph" : "flame";
  const point = Array.isArray(payload.points) ? payload.points[0] : undefined;
  if (!point) return undefined;
  if (point.bin_start_seconds != null) return "density";
  if (point.mean_duration_seconds != null) return "timeseries";
  if (point.mean_over_ranks_seconds != null) return "imbalance";
  if (point.speedup != null) return "speedup";
  if (point.normalized_runtime != null) return "weak_scaling";
  if (point.efficiency != null) return "scaling_efficiency";
  if (point.arithmetic_intensity_flops_per_byte != null) return "roofline";
  if (point.rank != null) return "rank_heatmap";
  return undefined;
}

/** Validate a plot-data envelope and return its resolved plot kind.
 *
 * The validator deliberately requires only the array each builder consumes:
 * exporter versions may add fields without breaking existing dashboards.
 */
export function validatePlotData(payload, options = {}) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload))
    throw new TypeError("plot-data must be an object.");
  if (payload.format != null && payload.format !== PLOT_DATA_FORMAT)
    throw new TypeError(
      `Expected a ${PLOT_DATA_FORMAT} document, got ${JSON.stringify(payload.format)}.`,
    );
  if (
    Object.hasOwn(payload, "format_version") &&
    (!Number.isInteger(payload.format_version) || payload.format_version < 1)
  )
    throw new TypeError("format_version must be a positive integer.");
  if (
    typeof payload.format_version === "number" &&
    payload.format_version > SUPPORTED_FORMAT_VERSION
  )
    throw new TypeError(
      `Plot-data format version ${payload.format_version} is newer than this package supports (${SUPPORTED_FORMAT_VERSION}); upgrade @scope-profiler/plotly.`,
    );
  const kind = options.plot ?? payload.plot ?? inferPlotKind(payload);
  if (!kind || !Object.hasOwn(PLOT_BUILDERS, kind))
    throw new TypeError(
      kind
        ? `No figure builder for plot kind ${JSON.stringify(kind)}.`
        : "Could not determine the plot kind; pass options.plot.",
    );
  if (kind === "callgraph") {
    if (!Array.isArray(payload.calls) && !Array.isArray(payload.regions))
      throw new TypeError(
        'Plot kind "callgraph" requires calls or regions data.',
      );
  } else if (!Array.isArray(payload[PLOT_ARRAYS[kind]])) {
    throw new TypeError(
      `Plot kind ${JSON.stringify(kind)} requires a ${PLOT_ARRAYS[kind]} array.`,
    );
  }
  const recordKind = kind.startsWith("flame")
    ? "flame"
    : Object.hasOwn(SCALING_KINDS, kind)
      ? "scaling"
      : kind;
  validateRecords(recordKind, payload, {
    ...options,
    ...(recordKind === "scaling"
      ? { yField: scalingKind(payload, { ...options, plot: kind }).yKey }
      : {}),
  });
  return kind;
}

/** Build the right figure for any plot-data document, without naming a builder.
 *
 * Dispatches on the document's own `plot` field, falling back to the payload
 * shape for files written before scope-profiler stamped the envelope on every
 * kind.
 */
export function buildFigure(payload, options = {}) {
  const kind = validatePlotData(payload, options);
  const builder = PLOT_BUILDERS[kind];
  return builder(payload, { plot: kind, ...options });
}

const RENDER_CONFIG = { responsive: true, displaylogo: false };

/** Render a figure with any Plotly-compatible bundle. */
export function renderFigure(plotly, element, figure, config = {}) {
  if (!plotly || typeof plotly.newPlot !== "function")
    throw new TypeError(
      "renderFigure requires a Plotly-compatible object with newPlot().",
    );
  return plotly.newPlot(element, figure.data, figure.layout, {
    ...RENDER_CONFIG,
    ...config,
  });
}

/** Redraw a figure into an element that already holds one.
 *
 * `renderFigure` builds the plot from scratch, which throws away the viewer's
 * zoom and pan. A theme toggle, a changed filter or a new metric rebuilds the
 * figure but should not move the view, so route those through here: it uses
 * Plotly's `react`, and falls back to `newPlot` for a bundle without one.
 */
export function updateFigure(plotly, element, figure, config = {}) {
  const draw =
    plotly && typeof plotly.react === "function"
      ? plotly.react
      : plotly?.newPlot;
  if (typeof draw !== "function")
    throw new TypeError(
      "updateFigure requires a Plotly-compatible object with react() or newPlot().",
    );
  return draw.call(plotly, element, figure.data, figure.layout, {
    ...RENDER_CONFIG,
    ...config,
  });
}

/** Release Plotly event handlers and rendering resources on unmount. */
export function disposeFigure(plotly, element) {
  if (typeof plotly?.purge !== "function")
    throw new TypeError("disposeFigure requires purge().");
  return plotly.purge(element);
}

/** Isolated defaults for a dashboard, without changing the global theme. */
export function createFigureBuilder(defaults = {}) {
  const snapshot = mergeLayout({}, { theme: resolveTheme(), ...defaults });
  return (payload, options = {}) =>
    buildFigure(payload, mergeLayout(snapshot, options));
}
