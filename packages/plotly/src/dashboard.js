/** Browser glue for a page that renders scope-profiler figures.
 *
 * `index.js` is deliberately framework-neutral and pure: it turns a plot-data
 * document into `{ data, layout }` and touches nothing else. A page built on
 * it still needs two things every time, and both were being written from
 * scratch by each deployment:
 *
 * * **Somewhere to keep the figures.** Plotly bakes colours into the layout,
 *   so a theme toggle means rebuilding every figure on the page -- which means
 *   remembering which containers hold one and what each was built from.
 * * **A filter syntax.** `filterRegion` takes a predicate, and should: a
 *   library has no business deciding how a user spells a filter. But every
 *   dashboard grows the same comma-separated substring box, so here is one.
 *
 * This module is a separate entry point because it reaches for `document` and
 * `window`; importing `@scope-profiler/plotly` never does.
 */

import { buildFigure, renderFigure, setTheme, updateFigure } from "./index.js";

/** Split a filter box's contents into terms.
 *
 * A filter is a comma-separated list, each term matched case-insensitively as
 * a substring of the region name. Empty terms -- a trailing comma while
 * someone is still typing -- are dropped.
 */
export function parseRegionFilter(text) {
  return String(text ?? "")
    .split(",")
    .map((term) => term.trim().toLowerCase())
    .filter(Boolean);
}

/** Whether a region name matches any of `terms`.
 *
 * A leading "^" anchors a term to the start of the name, which is how a
 * dashboard tells a group of regions apart from the regions that merely
 * mention it: "prop:" also matches "setup prop: X", "^prop:" does not.
 */
export function matchesRegionFilter(region, terms) {
  const name = String(region ?? "").toLowerCase();
  return terms.some((term) =>
    term.startsWith("^") ? name.startsWith(term.slice(1)) : name.includes(term),
  );
}

/** A `filterRegion` predicate for a filter box, or undefined when it is empty.
 *
 * Undefined rather than a predicate that accepts everything: an empty box
 * means every region, and passing no filter at all is both faster and what
 * the builders document.
 */
export function regionFilter(text) {
  const terms = parseRegionFilter(text);
  return terms.length
    ? (region) => matchesRegionFilter(region, terms)
    : undefined;
}

/** The theme the host page is in, as `setTheme` spells it.
 *
 * Reads `data-theme` from the document element, the attribute a theme toggle
 * conventionally stamps there. With neither value set the page is following
 * the system preference, which "auto" already handles by committing to no
 * text colour.
 */
export function documentTheme() {
  if (typeof document === "undefined") return "auto";
  const theme = document.documentElement?.dataset?.theme;
  return theme === "dark" || theme === "light" ? theme : "auto";
}

/** Track the figures on a page so they can all be rebuilt at once.
 *
 * Each container remembers the payload and options it was last drawn from, so
 * `refresh()` can rebuild it under whatever the theme now is. A container that
 * has already been drawn is redrawn through `updateFigure`, which keeps the
 * viewer's zoom and pan across a theme toggle or a changed filter.
 *
 * ```js
 * const figures = createFigureRegistry(Plotly);
 * figures.watch();                       // rebuild on a "themechanged" event
 * await figures.render(el, payload, { regionFilter: box.value });
 * ```
 *
 * @param plotly A Plotly-compatible bundle.
 * @param options.theme A theme, or a function returning one, read at each
 *   render. Defaults to `documentTheme`.
 * @param options.config Plotly config, merged into every render.
 * @param options.build Figure builder, defaulting to `buildFigure`. Override
 *   for a figure no plot kind names, such as `buildComparisonFigure`.
 */
export function createFigureRegistry(plotly, options = {}) {
  const {
    theme = documentTheme,
    config = {},
    build: defaultBuild = buildFigure,
  } = options;
  // The spec lives here rather than on the container, so a page cannot be
  // left holding a detached node through a property nobody remembers setting.
  const specs = new Map();
  const currentTheme = () => (typeof theme === "function" ? theme() : theme);

  // A container that has left the document is dead weight: the page swapped
  // it out, and redrawing into it paints nothing. Anything without the DOM
  // property -- a test double, a server-side stub -- counts as live.
  const isLive = (container) => container?.isConnected !== false;

  async function draw(container, spec) {
    const { payload, options: buildOptions = {}, drawn } = spec;
    const build = buildOptions.build ?? defaultBuild;
    const { build: _build, regionFilter: filterText, ...rest } = buildOptions;
    const figure = build(payload, {
      theme: currentTheme(),
      ...(filterText != null ? { filterRegion: regionFilter(filterText) } : {}),
      ...rest,
    });
    // First draw builds the plot; later ones react into it, so a theme toggle
    // does not reset the view the reader had scrolled to.
    const paint = drawn ? updateFigure : renderFigure;
    spec.drawn = true;
    return paint(plotly, container, figure, config);
  }

  return {
    /** Draw `payload` into `container` and remember how, for `refresh()`. */
    render(container, payload, buildOptions = {}) {
      if (!container) return undefined;
      const spec = specs.get(container) ?? {};
      Object.assign(spec, { payload, options: buildOptions });
      specs.set(container, spec);
      return draw(container, spec);
    },
    /** Rebuild every live figure under the current theme. */
    refresh() {
      setTheme(currentTheme());
      const drawing = [];
      for (const [container, spec] of specs) {
        if (!isLive(container)) specs.delete(container);
        else drawing.push(draw(container, spec));
      }
      return Promise.all(drawing);
    },
    /** Stop tracking one container, and tear its plot down if it has one.
     *
     * Dropping the spec alone leaves Plotly's own event handlers attached to
     * a node the page is finished with, so purge as well where the bundle
     * offers it.
     */
    forget(container) {
      const tracked = specs.get(container);
      if (tracked?.drawn && typeof plotly?.purge === "function")
        plotly.purge(container);
      return specs.delete(container);
    },
    /** Stop tracking every container, tearing down each plot. */
    clear() {
      for (const container of [...specs.keys()]) this.forget(container);
    },
    /** Figures currently tracked, live or not. */
    get size() {
      return specs.size;
    },
    /**
     * Rebuild on an event, and return the function that stops listening.
     * Defaults to `window` and the "themechanged" event a toggle can dispatch.
     */
    watch(target, event = "themechanged") {
      const source =
        target ?? (typeof window === "undefined" ? undefined : window);
      if (!source?.addEventListener) return () => {};
      const listener = () => this.refresh();
      source.addEventListener(event, listener);
      return () => source.removeEventListener(event, listener);
    },
  };
}
