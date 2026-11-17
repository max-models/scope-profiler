// Draws the report's chart panels with @scope-profiler/plotly: a port of the standalone report's
// chart module. The panels follow the page's region filter (scopeProfilerOnRegionFilter), the
// Regions and Depth sliders, and the highlighted region (scopeProfilerOnRegionSelect), all
// provided by client.js; a click on a chart highlights the region it shows.
import { buildComparisonFigure, buildFigure } from '@scope-profiler/plotly';
import { durationText } from './compare.js';

const root = document.getElementById('sp-charts');

if (root) {
	const regionIndex = JSON.parse(root.dataset.regionIndex || '{}');
	const runMeta = JSON.parse(root.dataset.runMeta || '{}');
	const charts = [...root.querySelectorAll('.chart')].map((element) => ({
		id: element.id,
		element,
		title: element.dataset.title,
		options: JSON.parse(element.dataset.options || '{}'),
		variants: element.dataset.variants ? JSON.parse(element.dataset.variants) : null,
		src: element.dataset.src,
		payload: null,
		figure: null,
		stale: false,
		scale: { x: 'linear', y: 'linear' },
		stack: element.dataset.stack ? JSON.parse(element.dataset.stack) : null,
		barLabels: new Set(),
	}));
	let plotly = null;

	const theme = () => (document.documentElement.dataset.theme === 'light' ? 'light' : 'dark');

	const payloadRegions = (payload) => {
		const rows = [
			...(payload.intervals ?? []),
			...(payload.bars ?? []),
			...(payload.points ?? []),
			...(payload.calls ?? []),
			...(payload.regions ?? []),
		];
		return new Set(rows.map((row) => row.region ?? row.name).filter(Boolean));
	};

	const traceMatchesRegion = (trace, region) => {
		const name = String(trace.name ?? '');
		return (
			name === region ||
			name.endsWith(` / ${region}`) ||
			name === `${region} mean` ||
			name.endsWith(` / ${region} mean`)
		);
	};

	const highlightFigure = (chart, figure, region) => {
		if (!region || !payloadRegions(chart.payload).has(region)) return figure;
		for (const trace of figure.data) {
			if (trace.type === 'icicle') {
				const labels = trace.labels ?? [];
				const original = Array.isArray(trace.marker?.colors) ? trace.marker.colors : [];
				trace.marker.colors = labels.map((label, index) =>
					label === region ? (original[index] ?? '#d97706') : 'rgba(156,163,175,0.22)',
				);
			} else if (trace.type === 'heatmap') {
				figure.layout.shapes = [
					...(figure.layout.shapes ?? []),
					{
						type: 'rect',
						xref: 'x',
						yref: 'paper',
						x0: region,
						x1: region,
						x0shift: -0.5,
						x1shift: 0.5,
						y0: 0,
						y1: 1,
						fillcolor: 'rgba(245,158,11,0.16)',
						line: { color: '#d97706', width: 3 },
					},
				];
			} else if (trace.type === 'bar' && trace.orientation === 'h') {
				trace.opacity = traceMatchesRegion(trace, region) ? 1 : 0.16;
			} else if (trace.type === 'bar') {
				trace.marker.opacity = (trace.x ?? []).map((value) => (value === region ? 1 : 0.16));
				trace.marker.line = {
					...(trace.marker.line ?? {}),
					color: (trace.x ?? []).map((value) => (value === region ? '#92400e' : 'rgba(0,0,0,0.12)')),
					width: (trace.x ?? []).map((value) => (value === region ? 2 : 0.5)),
				};
			} else if (trace.type === 'scatter') {
				trace.opacity = traceMatchesRegion(trace, region) ? 1 : 0.14;
			}
		}
		return figure;
	};

	const regionFromPoint = (chart, point) => {
		const regions = payloadRegions(chart.payload);
		const identity = point.customdata?.identity;
		if (identity && regions.has(identity.region)) return identity.region;
		for (const candidate of [point.label, point.x, point.y, point.data?.name]) {
			if (regions.has(candidate)) return candidate;
		}
		const traceName = String(point.data?.name ?? '');
		for (const region of regions) {
			if (
				traceName === `${region} mean` ||
				traceName.endsWith(` / ${region}`) ||
				traceName.endsWith(` / ${region} mean`)
			) {
				return region;
			}
		}
		return null;
	};

	const runFromPoint = (chart, point, region) => {
		if (point.customdata?.identity?.file != null) return point.customdata.identity.file;
		if (Array.isArray(point.customdata) && typeof point.customdata[0] === 'string') {
			return point.customdata[0];
		}
		const name = String(point.data?.name ?? '');
		return region && name.endsWith(` / ${region}`) ? name.slice(0, -region.length - 3) : null;
	};

	let activeTerms = [];
	let selectedRegion = null;
	const termFilter = (region) =>
		activeTerms.some((term) =>
			term.startsWith('^')
				? String(region).toLowerCase().startsWith(term.slice(1))
				: String(region).toLowerCase().includes(term),
		);

	// The filter bar's Regions and Depth sliders limit every chart to the same regions: those no
	// deeper than maxDepth, then the topN of them by time. The timeline also hides the deeper calls
	// of the regions it keeps.
	const regionLimits = {};
	let limitedRegions = null;
	const limitedSet = () => {
		const { topN, maxDepth } = regionLimits;
		let names = Object.keys(regionIndex).filter((name) => {
			const depth = regionIndex[name][1];
			return (
				(!activeTerms.length || termFilter(name)) && (maxDepth == null || depth == null || depth <= maxDepth)
			);
		});
		if (topN != null) {
			names = names.sort((a, b) => regionIndex[b][0] - regionIndex[a][0]).slice(0, topN);
		}
		return new Set(names);
	};
	const updateLimitedRegions = () => {
		limitedRegions = Object.keys(regionLimits).length ? limitedSet() : null;
		window.scopeProfilerSetRegionLimit?.(limitedRegions);
	};
	const chartFilter = (region) =>
		(!activeTerms.length || termFilter(region)) && (!limitedRegions || limitedRegions.has(region));

	// Axis scales: linear, log2 or log10, for every numeric axis. A categorical axis (region names, the
	// timeline's lanes) has nothing to scale, and its buttons stay hidden.
	const layoutKey = (axis) => `${axis}axis`;
	const axisScalable = (figure, axis) =>
		!figure.layout[layoutKey(axis)]?.ticktext &&
		figure.data.some(
			(trace) =>
				!['heatmap', 'icicle', 'sankey', 'pie'].includes(trace.type) &&
				Array.isArray(trace[axis]) &&
				typeof trace[axis].find((value) => value !== null && value !== undefined) === 'number',
		);

	// What identifies each run, written above its bars: the run's label, its MPI ranks, OpenMP threads,
	// nodes, cores (ranks times threads) and wall time, and the bar's own value.
	const BAR_LABELS = {
		label: (meta) => meta.label,
		ranks: (meta) => `${meta.ranks} rank${meta.ranks === 1 ? '' : 's'}`,
		threads: (meta) => (meta.threads ? `${meta.threads} thread${meta.threads === 1 ? '' : 's'}` : null),
		nodes: (meta) => (meta.nodes ? `${meta.nodes} node${meta.nodes === 1 ? '' : 's'}` : null),
		cores: (meta) => (meta.cores ? `${meta.cores} core${meta.cores === 1 ? '' : 's'}` : null),
		wall: (meta) => (meta.wall ? durationText(meta.wall) : null),
	};

	/** The timeline starts in one colour, so no region stands out until one is picked (the page's highlight then marks it). */
	function neutralColors(figure) {
		for (const trace of figure.data) {
			if (trace.type === 'bar') trace.marker = { ...trace.marker, color: '#1f77b4' };
		}
	}

	function labelBars(chart, figure) {
		if (!chart.barLabels.size) return;
		for (const trace of figure.data) {
			const meta = runMeta[trace.name];
			if (trace.type !== 'bar' || !meta || !Array.isArray(trace.y)) continue;
			const fixed = Object.keys(BAR_LABELS)
				.filter((key) => chart.barLabels.has(key))
				.map((key) => BAR_LABELS[key](meta))
				.filter(Boolean);
			trace.text = trace.y.map((value) =>
				[...fixed, ...(chart.barLabels.has('value') && typeof value === 'number' ? [durationText(value)] : [])].join(' · '),
			);
			// Written upright above the bar, at a fixed size, not shrunk to the bar's width.
			trace.textposition = 'outside';
			trace.textangle = -90;
			trace.constraintext = 'none';
			trace.cliponaxis = false;
			trace.textfont = { ...(trace.textfont ?? {}), size: 11 };
		}
		// Room above the plot for them.
		const margin = figure.layout.margin ?? {};
		figure.layout.margin = { ...margin, t: Math.max(margin.t ?? 100, 150) };
	}

	/** Put `figure`'s axes on the chart's chosen scales; log2 is a log axis with ticks at powers of two. */
	function applyScales(chart, figure) {
		const scalable = { x: axisScalable(figure, 'x'), y: axisScalable(figure, 'y') };
		// Room for the controls that sit at the axes: above the y axis, right of the x axis.
		const margin = figure.layout.margin ?? {};
		figure.layout.margin = {
			...margin,
			...(scalable.y ? { t: Math.max(margin.t ?? 100, 46) } : {}),
			...(scalable.x ? { r: Math.max(margin.r ?? 80, 86) } : {}),
		};
		for (const axis of ['x', 'y']) {
			if (!scalable[axis] || chart.scale[axis] === 'linear') continue;
			const key = layoutKey(axis);
			const { range, tickvals, ticktext, tickmode, dtick, ...rest } = figure.layout[key] ?? {};
			figure.layout[key] = { ...rest, type: 'log', autorange: true };
		}
		return scalable;
	}

	/** Ticks at powers of two across a log axis's range (`range` is in log10 units). */
	function powersOfTwo(range) {
		const low = Math.floor(range[0] / Math.log10(2) - 1e-9);
		const high = Math.ceil(range[1] / Math.log10(2) + 1e-9);
		const step = Math.max(1, Math.ceil((high - low) / 24));
		const values = [];
		for (let power = low; power <= high; power += step) values.push(2 ** power);
		return values;
	}

	function showScaleControls(chart, scalable) {
		const panel = chart.element.closest('details');
		for (const group of panel.querySelectorAll('.chart-scale')) {
			group.hidden = !scalable[group.dataset.axisName];
		}
		for (const button of panel.querySelectorAll('.chart-scale-button')) {
			button.setAttribute('aria-pressed', String(chart.scale[button.dataset.scaleAxis] === button.dataset.scale));
		}
	}

	/** The wall time of each run over the executions of a case: one line per run, in the site's colours. */
	function trendFigure(payload) {
		const css = getComputedStyle(document.documentElement);
		const color = css.getPropertyValue('--sl-color-text').trim() || undefined;
		const grid = css.getPropertyValue('--sl-color-gray-5').trim() || undefined;
		const labels = payload.executions.map((execution) => execution.label);
		return {
			data: payload.series.map((series) => ({
				type: 'scatter',
				mode: 'lines+markers',
				name: series.label,
				x: labels,
				y: series.values,
				connectgaps: false,
				hovertemplate: '%{x}<br>%{y:.4g} s<extra>%{fullData.name}</extra>',
			})),
			layout: {
				title: { text: payload.title, font: { color } },
				font: { color },
				height: 460,
				margin: { t: 90, r: 170, b: 90, l: 80 },
				xaxis: { title: { text: 'execution' }, type: 'category', automargin: true, tickangle: -20, gridcolor: grid },
				yaxis: { title: { text: 'wall time (s)' }, rangemode: 'tozero', gridcolor: grid },
				legend: { orientation: 'v', x: 1.02, y: 1 },
			},
		};
	}

	/** Put each axis's scale control at its axis: the y control beside the top of the y axis (stacked, so it fits the margin and keeps clear of the bar labels above the plot), the x control past the right end of the x axis. */
	function placeScaleControls(chart) {
		const size = chart.element._fullLayout?._size;
		if (!size) return;
		const frame = chart.element.parentElement;
		const y = frame.querySelector('.axis-scale-y');
		const x = frame.querySelector('.axis-scale-x');
		if (y && !y.hidden) {
			y.style.left = '0px';
			y.style.top = `${size.t}px`;
		}
		if (x && !x.hidden) {
			x.style.left = `${size.l + size.w + 10}px`;
			x.style.top = `${Math.max(0, size.t + size.h - x.offsetHeight)}px`;
		}
	}

	async function ensurePayload(chart) {
		if (!chart.payload) {
			const response = await fetch(chart.src);
			if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
			chart.payload = await response.json();
		}
	}

	async function draw(chart) {
		const target = chart.element;
		// A chart in a collapsed panel waits until the panel opens: a timeline of hundreds of lanes is
		// seconds of drawing that nobody may look at.
		const panel = target.closest('details');
		if (panel && !panel.open) {
			chart.stale = true;
			return;
		}
		chart.stale = false;
		try {
			await ensurePayload(chart);
			plotly ??= (await import('plotly.js-dist-min')).default;
			// The grouped bars of a comparison of many runs would be unreadable: draw the compared ones.
			const compared = window.scopeProfilerComparedRuns;
			const rowsKey = { durations: 'bars', imbalance: 'points', rank_heatmap: 'points' }[chart.payload.plot];
			const shown =
				compared && rowsKey && Array.isArray(chart.payload[rowsKey])
					? { ...chart.payload, [rowsKey]: chart.payload[rowsKey].filter((row) => compared.includes(row.file)) }
					: chart.payload;
			let options =
				activeTerms.length || limitedRegions ? { ...chart.options, filterRegion: chartFilter } : chart.options;
			if (chart.payload.plot === 'gantt' && regionLimits.maxDepth != null) {
				options = { ...options, maxDepth: regionLimits.maxDepth };
			}
			// buildFigure dispatches region_statistics to the ranked summary; only buildComparisonFigure
			// draws the signed per-region change.
			const build = chart.payload.plot === 'trend' ? trendFigure : options.comparison ? buildComparisonFigure : buildFigure;
			const { layout: extra, ...rest } = options;
			const built = build(shown, { ...rest, layout: extra, theme: theme() });
			if (chart.payload.plot === 'gantt') neutralColors(built);
			const figure = highlightFigure(chart, built, selectedRegion);
			figure.layout = { ...figure.layout, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)' };
			labelBars(chart, figure);
			const scalable = applyScales(chart, figure);
			chart.figure = figure;
			target.classList.remove('chart-error');
			const rendered = plotly.react(target, figure.data, figure.layout, { responsive: true, displaylogo: false });
			showScaleControls(chart, scalable);
			if (!target.dataset.scalePlacement) {
				target.dataset.scalePlacement = 'true';
				Promise.resolve(rendered).then(() => target.on('plotly_afterplot', () => placeScaleControls(chart)));
			}
			// A log2 axis needs the range Plotly chose, so its ticks follow the drawing.
			Promise.resolve(rendered).then(() => {
				const update = {};
				for (const axis of ['x', 'y']) {
					if (!scalable[axis] || chart.scale[axis] !== 'log2') continue;
					const key = layoutKey(axis);
					const values = powersOfTwo(target._fullLayout[key].range);
					const texts = values.map((value) => (value >= 1 ? String(value) : `1/${1 / value}`));
					update[`${key}.tickmode`] = 'array';
					update[`${key}.tickvals`] = values;
					update[`${key}.ticktext`] = texts;
					figure.layout[key] = { ...figure.layout[key], tickmode: 'array', tickvals: values, ticktext: texts };
				}
				if (Object.keys(update).length) plotly.relayout(target, update);
				placeScaleControls(chart);
			});
			if (!target.dataset.regionClickBound) {
				target.dataset.regionClickBound = 'true';
				Promise.resolve(rendered).then(() =>
					target.on('plotly_click', (event) => {
						const point = event.points?.[0];
						if (!point) return;
						const region = regionFromPoint(chart, point);
						if (region && typeof window.scopeProfilerSelectRegion === 'function') {
							// "toast": highlight in place, and offer the jump to the table rather than scrolling
							// away from the chart being explored.
							window.scopeProfilerSelectRegion(region, runFromPoint(chart, point, region), 'toast');
						}
					}),
				);
			}
		} catch (error) {
			target.classList.add('chart-error');
			target.textContent = `Could not render chart: ${error.message}`;
		}
	}

	const redraw = () => {
		updateLimitedRegions();
		for (const chart of charts) draw(chart);
	};
	// Both hooks call their listener as soon as it registers, which would draw every chart twice on
	// load; changes arriving together are drawn once.
	let redrawQueued = false;
	const scheduleRedraw = () => {
		if (redrawQueued) return;
		redrawQueued = true;
		queueMicrotask(() => {
			redrawQueued = false;
			redraw();
		});
	};

	for (const chart of charts) {
		const panel = chart.element.closest('details');
		panel?.addEventListener('toggle', () => {
			if (!panel.open) return;
			if (chart.stale || !chart.figure) draw(chart);
			else plotly?.Plots.resize(chart.element);
		});
	}

	// Redraw when the compared runs change.
	window.addEventListener('scopeprofiler:compared', scheduleRedraw);

	// Redraw on a switch between light and dark.
	new MutationObserver(scheduleRedraw).observe(document.documentElement, {
		attributes: true,
		attributeFilter: ['data-theme'],
	});

	for (const button of document.querySelectorAll('[data-chart-action]')) {
		button.addEventListener('click', () => {
			const open = button.dataset.chartAction === 'expand';
			for (const panel of root.querySelectorAll('details.chart-panel')) panel.open = open;
		});
	}

	// "Open in new tab": the chart as drawn, filter and highlight included, on a page of its own,
	// sized to the window, with the Plotly runtime from its CDN.
	const escapeHtml = (value) =>
		String(value).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
	const openInNewTab = (chart) => {
		if (!chart.figure) return;
		const version = root.dataset.plotlyVersion;
		// The page is drawn in the site's colours, as the chart on this page is, whichever theme it is in.
		const style = getComputedStyle(document.documentElement);
		const background = style.getPropertyValue('--sl-color-bg').trim() || '#fff';
		const foreground = style.getPropertyValue('--sl-color-text').trim() || '#000';
		const figure = JSON.stringify({ data: chart.figure.data, layout: chart.figure.layout }).replace(/</g, '\\u003c');
		const page = `<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escapeHtml(chart.title)}</title><style>html, body { margin: 0; background: ${escapeHtml(background)}; }
#chart { width: 100vw; }</style></head><body><div id="chart"></div>
<script crossorigin="anonymous" src="https://cdn.plot.ly/plotly-${escapeHtml(version)}.min.js"><\/script><script>
const figure = ${figure};
const target = document.getElementById("chart");
const layout = { ...figure.layout, autosize: true, paper_bgcolor: ${JSON.stringify(background)}, plot_bgcolor: ${JSON.stringify(background)},
  title: figure.layout.title ?? { text: ${JSON.stringify(escapeHtml(chart.title))}, font: { color: ${JSON.stringify(foreground)} } } };
delete layout.width;
if (!(layout.height > window.innerHeight)) { delete layout.height; target.style.height = "100vh"; }
Plotly.newPlot(target, figure.data, layout, { responsive: true, displaylogo: false });
<\/script></body></html>`;
		const url = URL.createObjectURL(new Blob([page], { type: 'text/html' }));
		window.open(url, '_blank');
		setTimeout(() => URL.revokeObjectURL(url), 60000);
	};
	for (const button of document.querySelectorAll('.chart-open')) {
		const chart = charts.find((item) => item.id === button.dataset.chart);
		if (chart) button.addEventListener('click', () => openInNewTab(chart));
	}

	// At its maximum a slider means all. The readout follows the thumb, while the charts, which can
	// take most of a second to draw, wait for it to rest.
	const topLabel = (value, max) => (value >= max ? `all ${max}` : `top ${value} of ${max}`);
	const depthLabel = (value, max) =>
		value >= max
			? 'all levels'
			: value === 0
				? 'top level only'
				: `${value} level${value > 1 ? 's' : ''} of children`;
	for (const slider of document.querySelectorAll('.region-top, .region-depth')) {
		const isTop = slider.classList.contains('region-top');
		const key = isTop ? 'topN' : 'maxDepth';
		const readout = slider.nextElementSibling;
		const read = () => {
			const value = Number(slider.value);
			if (value >= Number(slider.max)) delete regionLimits[key];
			else regionLimits[key] = value;
		};
		// The page opens on the slider's own value: past 500 regions, the top 500.
		read();
		let timer = null;
		const apply = () => {
			window.clearTimeout(timer);
			const before = regionLimits[key];
			read();
			if (regionLimits[key] !== before) scheduleRedraw();
		};
		slider.addEventListener('input', () => {
			const value = Number(slider.value);
			const max = Number(slider.max);
			readout.textContent = isTop ? topLabel(value, max) : depthLabel(value, max);
			window.clearTimeout(timer);
			timer = window.setTimeout(apply, 150);
		});
		slider.addEventListener('change', apply);
	}

	for (const button of document.querySelectorAll('.chart-scale-button')) {
		const chart = charts.find((item) => item.id === button.dataset.chart);
		if (!chart) continue;
		button.addEventListener('click', () => {
			chart.scale[button.dataset.scaleAxis] = button.dataset.scale;
			draw(chart);
		});
	}

	// The menu closes when the reader clicks elsewhere.
	document.addEventListener('click', (event) => {
		for (const menu of document.querySelectorAll('.bar-labels[open]')) {
			if (!menu.contains(event.target)) menu.open = false;
		}
	});

	for (const box of document.querySelectorAll('.bar-label-box')) {
		const chart = charts.find((item) => item.id === box.dataset.chart);
		if (!chart) continue;
		box.addEventListener('change', () => {
			if (box.checked) chart.barLabels.add(box.dataset.key);
			else chart.barLabels.delete(box.dataset.key);
			draw(chart);
		});
	}

	for (const box of document.querySelectorAll('.chart-stack-box')) {
		const chart = charts.find((item) => item.id === box.dataset.chart);
		if (!chart?.stack) continue;
		box.addEventListener('change', () => {
			chart.src = box.checked ? chart.stack.stacked_src : chart.stack.flat_src;
			chart.payload = null;
			draw(chart);
		});
	}

	// The scaling charts' x-axis buttons: each swaps in the data computed for that axis, and the
	// description's axis name with it.
	for (const button of document.querySelectorAll('.chart-axis')) {
		const chart = charts.find((item) => item.id === button.dataset.chart);
		const variant = chart?.variants?.[Number(button.dataset.axis)];
		if (!variant) continue;
		button.addEventListener('click', async () => {
			chart.src = variant.src;
			chart.payload = null;
			const panel = button.closest('.chart-panel');
			for (const other of panel.querySelectorAll('.chart-axis')) {
				other.setAttribute('aria-pressed', String(other === button));
			}
			for (const label of panel.querySelectorAll('[data-axis-label]')) {
				label.textContent = variant.label;
			}
			draw(chart);
		});
	}

	if (typeof window.scopeProfilerOnRegionFilter === 'function') {
		window.scopeProfilerOnRegionFilter((terms) => {
			activeTerms = terms;
			scheduleRedraw();
		});
	}
	if (typeof window.scopeProfilerOnRegionSelect === 'function') {
		window.scopeProfilerOnRegionSelect((region) => {
			selectedRegion = region;
			scheduleRedraw();
		});
	}
	if (typeof window.scopeProfilerOnRegionFilter !== 'function' && typeof window.scopeProfilerOnRegionSelect !== 'function') {
		scheduleRedraw();
	}
}
