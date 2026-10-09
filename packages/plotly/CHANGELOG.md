# Changelog

## Unreleased

- `selectTimelineRegions(payload, options)` returns the regions
  `buildGanttFigure` draws for `filterRegion`, `maxDepth` and `topN`, so
  another chart can show the same ones.
- `buildDurationsFigure` gives a stacked segment only the bars it sits under,
  ordering the axis through `layout.xaxis.categoryarray`; a segment per child
  region used to carry a column for every region.
- `buildGanttFigure` takes `maxDepth`, which hides calls nested deeper than
  that below a top-level call (intervals carry an optional `depth`), and
  `topN`, which keeps the regions with the most summed call time among the
  calls still shown.
- Gantt lanes and flame and roofline hovers leave the rank out for a run with
  one rank, as told by the payload's optional `file_ranks` (run label -> rank
  count). Payloads without it keep naming every rank.
- Validate row fields, format versions, call ancestry, and duplicate cells with
  actionable errors. Preserve nested layout defaults and isolate theme presets.
- Use stable colors; add `createColorRegistry()` and scoped `createFigureBuilder()`.
- Add time-series variability bands/error bars and absolute/percent comparison
  deltas, including diagnostics for missing or zero baselines.
- Add `customdata.identity`, `getPointIdentity()`, linked legend groups, and
  `disposeFigure()`. Custom data now uses objects with numeric metric keys;
  consumers accessing raw custom data should migrate to `getPointIdentity()`.
- Preserve unequal density-bin boundaries, reject overlaps, and retain gaps.
- Detect cycles after Sankey aggregation, prefer measured edge weights, expose
  omitted/inferred edges, and hide traces with no usable links.
- Ship typed payloads and a discriminated `PlotData` union with consumer checks.
- Add browser rendering/interaction tests, JavaScript coverage enforcement,
  deterministic benchmark configuration, and bundled-asset synchronization.

## 0.4.0

### Fixed

- Region, file and series names are escaped before they reach a hover label. A
  name containing `%{...}` rewrote every hover template that mentioned it, and
  one containing markup injected it into the label.
- An empty payload no longer discards the `annotations` a caller passed through
  `options.layout`, and hides the axes behind the "No data to display." message
  instead of leaving an empty grid that reads as a broken chart.
- `buildDensityFigure` places each cell at its own bin centre. A single bin
  width taken from the first point misplaced the second run's cells whenever
  two runs of different length were binned into the same number of bins.
- `buildRooflineFigure` draws its ceiling in the theme's text colour rather than
  a fixed near-black, which was invisible on the dark theme, and leaves room at
  the top of the figure for the title it is alone in carrying.
- `buildDurationsFigure` accepts a colour keyed by the bare rank behind a
  `rank N` series, which it previously ignored in favour of the default cycle.
- `buildRegionSummaryFigure` and `buildComparisonFigure` reject an unknown
  `metric` instead of drawing a full chart of nulls.

### Changed

- `buildDurationsFigure` and `buildRankHeatmapFigure` order their categories by
  pooled cost rather than by the order the exporter happened to walk, which
  interleaved two runs with different region sets unpredictably.
- `buildGanttFigure` keeps its lanes on a categorical y axis, in
  `layout.yaxis.categoryarray`, and each bar reuses its lane's string. An
  array-mode tick axis (`tickvals`/`ticktext`) is laid out in time quadratic in
  its lanes: 1000 lanes took 3.6 s to draw instead of 0.6 s.

### Added

- `updateFigure(plotly, element, figure, config)` redraws through Plotly's
  `react`, keeping the viewer's zoom and pan across a rebuild.
- `SUMMARY_METRICS`, the short metric names the region_statistics builders
  accept, so a page can offer the list rather than guess it.
- `laneBy`, `updateFigure`, `SUMMARY_METRICS` and `PlotlyLike.react`/`purge` are
  declared in `index.d.ts`, which is now type-checked in CI, as is a
  compile-only usage fixture. `package.json` gained a top-level `types` entry
  so classic `node` module resolution finds the declarations at all.
- ESLint and Prettier configuration, and a `lint` script, gated in CI.
