// Comparing runs in the browser. A study can have 100 runs, which is 9,900 ordered pairs, so the
// page does not carry a comparison for each: it carries the compact dataset of
// `scope_profiler.report_model.compare_dataset` and computes the pair the reader picks, here.
//
// The functions below are a port of the Python that builds the HTML report's comparison
// (`report_data._comparison_entries`/`_change_items`/`_metric_text` and `report_model.change_model`,
// `wall_model`, `candidate_model`, `compare_table_model`) and give the same numbers and texts.
// scripts/ui_smoke.py checks that against the Python's own output, so the two cannot drift apart.

const SESSION = 'scope_profiler.session';
const SAME_PCT = 2.0; // changes smaller than this read as noise
const CHANGE_FLOOR_PCT = 0.5; // own-time changes below this share of the baseline are not listed
const CHANGE_COUNT = 5;
const METRICS = ['total', 'own', 'avg', 'calls']; // the order of a run's values in the dataset
/** Show every run against the baseline, rather than one pair, up to this many runs. */
const ALL_RUNS_LIMIT = 12;

// ---- formatting, matching Python's ----------------------------------------------------------

/** Python's `f"{x:.{p}g}"`. */
export function pyG(x, p) {
	if (x === 0) return '0';
	const rounded = Number(x.toPrecision(p));
	const e = rounded === 0 ? 0 : Math.floor(Math.log10(Math.abs(rounded)));
	if (e < -4 || e >= p) {
		let [mantissa, exponent] = rounded.toExponential(p - 1).split('e');
		if (mantissa.includes('.')) mantissa = mantissa.replace(/0+$/, '').replace(/\.$/, '');
		const sign = exponent[0] === '-' ? '-' : '+';
		return `${mantissa}e${sign}${exponent.replace(/^[+-]/, '').padStart(2, '0')}`;
	}
	let text = rounded.toFixed(Math.max(0, p - 1 - e));
	if (text.includes('.')) text = text.replace(/0+$/, '').replace(/\.$/, '');
	return text;
}

const grouped = (value) => Math.abs(value).toLocaleString('en-US');

/** A duration in the unit that keeps it short: 51.2 ms, not 0.0512 s. */
export function durationText(seconds) {
	if (seconds === null || seconds === undefined) return '-';
	const magnitude = Math.abs(seconds);
	if (magnitude === 0) return '0 s';
	if (magnitude < 1e-6) return `${pyG(seconds * 1e9, 3)} ns`;
	if (magnitude < 1e-3) return `${pyG(seconds * 1e6, 3)} µs`;
	if (magnitude < 1) return `${pyG(seconds * 1e3, 3)} ms`;
	if (magnitude < 1000) return `${pyG(seconds, 4)} s`;
	return `${(seconds < 0 ? '-' : '') + grouped(Math.round(magnitude))} s`;
}

/** A percent change with a real minus sign, so columns of them line up. */
export function signedPct(value) {
	const text = `${value < 0 ? '-' : '+'}${Math.abs(value).toFixed(1)}%`;
	return text.replace('-', '−');
}

const metricValue = (values, metric) => (values ? values[METRICS.indexOf(metric)] : null);

const metricText = (value, metric) =>
	value === null || value === undefined ? '-' : metric === 'calls' ? grouped(value) : durationText(value);

// ---- the comparison --------------------------------------------------------------------------

/** Align the runs' region rows by call path, in call-tree order; the first run's tree sets the order. */
export function comparisonEntries(dataset, runIndexes) {
	const entries = [];
	let index = new Map();
	const insertAt = (key) => {
		if (!key.startsWith('path:') || !key.slice(5).includes(' > ')) return entries.length;
		const path = key.slice(5);
		const parent = 'path:' + path.slice(0, path.lastIndexOf(' > '));
		if (!index.has(parent)) return entries.length;
		let position = index.get(parent) + 1;
		const prefix = parent + ' > ';
		while (position < entries.length && entries[position].key.startsWith(prefix)) position += 1;
		return position;
	};
	runIndexes.forEach((runIndex, slot) => {
		const run = dataset.runs[runIndex];
		for (const key of run.order) {
			if (!index.has(key)) {
				const info = dataset.regions[key];
				entries.splice(insertAt(key), 0, {
					key,
					name: info.name,
					depth: info.depth,
					context: info.context,
					values: runIndexes.map(() => null),
				});
				index = new Map(entries.map((entry, position) => [entry.key, position]));
			}
			entries[index.get(key)].values[slot] = run.values[key];
		}
	});
	return entries;
}

/** How `value` compares with `base`, for a comparison table cell. */
export function changeModel(base, value, metric) {
	const missing = (x) => x === null || x === undefined;
	if (missing(base) && missing(value)) return { kind: 'none', size: null };
	if (missing(base)) return { kind: 'new', size: Math.abs(value) };
	if (missing(value)) return { kind: 'gone', size: Math.abs(base) };
	const delta = value - base;
	if (!base) return { kind: 'new', size: Math.abs(delta) };
	const pct = (100 * delta) / base;
	let tone;
	let title;
	if (metric === 'calls') {
		tone = delta === 0 ? 'same' : delta < 0 ? 'neutral down' : 'neutral';
		title = `${delta >= 0 ? '+' : '-'}${grouped(delta)} calls`;
	} else {
		tone = Math.abs(pct) < SAME_PCT ? 'same' : delta < 0 ? 'faster' : 'slower';
		const speed = !value ? '' : delta < 0 ? ` · ${(base / value).toFixed(2)}× faster` : ` · ${(value / base).toFixed(2)}× slower`;
		title = `${delta >= 0 ? '+' : '−'}${durationText(Math.abs(delta))}${speed}`;
	}
	return { kind: 'delta', tone, pct, label: signedPct(pct), title, size: Math.abs(delta) };
}

/** The largest own-time changes of one run against the baseline, in one direction. */
function changeItems(entries, slot, direction) {
	const owns = entries.map((entry) => [metricValue(entry.values[0], 'own') ?? 0, metricValue(entry.values[slot], 'own') ?? 0]);
	const floor = (CHANGE_FLOOR_PCT / 100) * owns.reduce((sum, [base]) => sum + base, 0);
	const changes = [];
	entries.forEach((entry, position) => {
		const [base, value] = owns[position];
		const delta = value - base;
		if (delta * direction > 0 && Math.abs(delta) >= floor && (!base || (100 * Math.abs(delta)) / base >= SAME_PCT)) {
			changes.push({ ...entry, base, value, delta });
		}
	});
	changes.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
	return changes.slice(0, CHANGE_COUNT);
}

function wallModel(before, after) {
	if (!before || !after) return { after_text: durationText(after) };
	const pct = (100 * (after - before)) / before;
	return {
		before_text: durationText(before),
		after_text: durationText(after),
		pct,
		pct_text: signedPct(pct),
		tone: Math.abs(pct) < SAME_PCT ? '' : pct < 0 ? 'good' : 'bad',
		speed: after < before ? `${(before / after).toFixed(2)}× faster` : `${(after / before).toFixed(2)}× slower`,
	};
}

function changeItemModel(change, scale) {
	return {
		name: change.name,
		context: change.context,
		outside: change.name === SESSION,
		delta: change.delta,
		delta_text: durationText(Math.abs(change.delta)),
		sign: change.delta > 0 ? '+' : '−',
		pct_text: change.base ? ` (${signedPct((100 * change.delta) / change.base)})` : ' (new)',
		width: scale ? (100 * Math.abs(change.delta)) / scale : 0.0,
	};
}

/** One run against the baseline: headline numbers and what moved (`entries` has them at slots 0 and `slot`). */
export function candidateModel(dataset, entries, baseIndex, runIndex, slot = 1) {
	const improvements = changeItems(entries, slot, -1);
	const regressions = changeItems(entries, slot, 1);
	const scale = Math.max(0, ...[...improvements, ...regressions].map((change) => Math.abs(change.delta)));
	const extreme = (changes) =>
		changes.length
			? { name: changes[0].name, sign: changes[0].delta > 0 ? '+' : '−', delta_text: durationText(Math.abs(changes[0].delta)) }
			: null;
	return {
		label: dataset.runs[runIndex].label,
		base_label: dataset.runs[baseIndex].label,
		wall: wallModel(dataset.runs[baseIndex].wall, dataset.runs[runIndex].wall),
		extremes: { 'Largest improvement': extreme(improvements), 'Largest regression': extreme(regressions) },
		improvements: improvements.map((change) => changeItemModel(change, scale)),
		regressions: regressions.map((change) => changeItemModel(change, scale)),
		limit: CHANGE_COUNT,
	};
}

/** Every region's duration in every run (the first being the baseline), with changes against it. */
export function compareTableModel(dataset, runIndexes) {
	const entries = comparisonEntries(dataset, runIndexes);
	const rows = entries.map((entry, order) => {
		const cells = [];
		const sizes = {};
		entry.values.forEach((values, slot) => {
			const text = {};
			const changes = {};
			for (const metric of METRICS) {
				const value = metricValue(values, metric);
				text[metric] = metricText(value, metric);
				if (slot) {
					changes[metric] = changeModel(metricValue(entry.values[0], metric), value, metric);
					// Sorting by change follows the last run against the baseline.
					sizes[metric] = changes[metric].size;
				}
			}
			cells.push({ values: text, changes: slot ? changes : null });
		});
		return {
			name: entry.name,
			depth: entry.depth,
			context: entry.context,
			title: [...entry.context, entry.name].join(' › '),
			order,
			sizes,
			cells,
		};
	});
	return {
		labels: runIndexes.map((index) => dataset.runs[index].label),
		metrics: METRICS.map((key) => dataset.metrics.find((metric) => metric.key === key)),
		rows,
		per_rank: dataset.per_rank,
	};
}

/** The candidates of `runIndexes` (after the first) against the first. */
export function candidateModels(dataset, runIndexes) {
	const entries = comparisonEntries(dataset, runIndexes);
	return runIndexes.slice(1).map((runIndex, position) => candidateModel(dataset, entries, runIndexes[0], runIndex, position + 1));
}

// ---- drawing ---------------------------------------------------------------------------------

function el(tag, attributes = {}, ...children) {
	const node = document.createElement(tag);
	for (const [name, value] of Object.entries(attributes)) {
		if (value === null || value === undefined || value === false) continue;
		if (name === 'class') node.className = value;
		else node.setAttribute(name, value === true ? '' : String(value));
	}
	for (const child of children.flat(Infinity)) {
		if (child === null || child === undefined || child === false) continue;
		node.append(child instanceof Node ? child : document.createTextNode(String(child)));
	}
	return node;
}

const deltaCell = (change) => {
	if (change.kind === 'none') return document.createTextNode('-');
	if (change.kind === 'new' || change.kind === 'gone') return el('span', { class: `badge ${change.kind}` }, change.kind);
	return el('span', { class: `delta ${change.tone}`, title: change.title }, el('span', { class: 'delta-text' }, change.label));
};

function candidateBlock(candidate) {
	const wall = candidate.wall;
	const hasBefore = wall.before_text !== undefined;
	const kpis = [
		el(
			'div',
			{ class: `kpi ${wall.tone ?? ''}`.trim() },
			el('span', { class: 'kpi-label' }, 'Wall time'),
			el('span', { class: 'kpi-value' }, hasBefore ? `${wall.before_text} → ${wall.after_text}` : wall.after_text),
			hasBefore && el('span', { class: 'kpi-detail' }, `${wall.pct_text} · ${wall.speed}`),
		),
		...Object.entries(candidate.extremes).map(([title, extreme]) =>
			el(
				'div',
				{ class: 'kpi' },
				el('span', { class: 'kpi-label' }, title),
				el('span', { class: 'kpi-value' }, extreme ? el('code', {}, extreme.name) : 'none'),
				extreme && el('span', { class: 'kpi-detail' }, `${extreme.sign}${extreme.delta_text} own time`),
			),
		),
	];
	const list = (heading, items, tone) =>
		el(
			'div',
			{},
			el('h4', {}, heading),
			items.length === 0
				? el('p', { class: 'muted' }, 'None.')
				: el(
						'ol',
						{ class: 'change-list' },
						items.map((item) =>
							el(
								'li',
								{ class: 'change-item', 'data-filter-region': item.name },
								el(
									'span',
									{ class: 'hs-name' },
									item.outside ? el('em', {}, 'outside any region') : el('strong', {}, item.name),
									item.context.length > 0 && el('span', { class: 'hs-path' }, `in ${item.context.join(' › ')}`),
								),
								el('span', { class: 'change-track' }, el('span', { class: `change-fill ${tone}`, style: `width:${pyG(item.width, 3)}%` })),
								el('span', { class: `change-value delta-text ${tone}` }, `${item.sign}${item.delta_text}`, el('span', { class: 'muted' }, item.pct_text)),
							),
						),
					),
		);
	return el(
		'div',
		{ class: 'compare-candidate' },
		el('h3', {}, `${candidate.label} vs ${candidate.base_label}`),
		el('div', { class: 'kpis' }, kpis),
		el('div', { class: 'changes' }, list('Faster', candidate.improvements, 'faster'), list('Slower', candidate.regressions, 'slower')),
	);
}

function compareTable(model, metric, order) {
	const labels = model.labels;
	const head = el(
		'tr',
		{},
		el('th', {}, 'region'),
		el('th', {}, labels[0]),
		labels.slice(1).map((label) => [el('th', {}, label), el('th', {}, `vs ${labels[0]}`)]),
	);
	const body = model.rows.map((row) =>
		el(
			'tr',
			{
				class: 'select-row',
				'data-region': row.name,
				'data-filter-region': row.name,
				'data-order': row.order,
				...Object.fromEntries(Object.entries(row.sizes).map(([key, size]) => [`data-change-${key}`, size === null ? '' : pyG(size, 9)])),
			},
			el('td', { title: row.title }, el('span', { class: 'cmp-cell' }, el('span', { class: 'cmp-name', style: `padding-left:${pyG(1.1 * row.depth, 3)}em` }, row.name))),
			row.cells.map((cell, slot) => [
				el('td', {}, model.metrics.map((m) => el('span', { 'data-m': m.key }, cell.values[m.key]))),
				slot > 0 && el('td', {}, model.metrics.map((m) => el('span', { 'data-m': m.key }, deltaCell(cell.changes[m.key])))),
			]),
		),
	);
	const table = el('table', { class: `compare-table metric-${metric}${order === 'change' ? ' flat' : ''}`, id: 'sp-compare-table' }, el('thead', {}, head), el('tbody', {}, body));
	if (order === 'change') {
		const rows = [...table.tBodies[0].rows];
		rows.sort((a, b) => (parseFloat(b.dataset[`change${metric[0].toUpperCase()}${metric.slice(1)}`]) || -1) - (parseFloat(a.dataset[`change${metric[0].toUpperCase()}${metric.slice(1)}`]) || -1));
		rows.forEach((row) => table.tBodies[0].append(row));
	}
	return table;
}

// ---- the page --------------------------------------------------------------------------------

function init() {
	const root = document.getElementById('sp-compare');
	if (!root) return;
	const dataset = JSON.parse(document.getElementById('sp-compare-data').textContent);
	const count = dataset.runs.length;
	const labels = dataset.runs.map((run) => run.label);
	const changesBox = document.getElementById('sp-changes');
	const tableBox = document.getElementById('sp-compare-box');
	const baseSelect = document.getElementById('sp-compare-base');
	const candSelect = document.getElementById('sp-compare-cand');
	const allBox = document.getElementById('sp-compare-all');
	const canShowAll = count <= ALL_RUNS_LIMIT;

	const state = { base: 0, cand: count - 1, all: false, metric: 'total', order: 'tree' };
	const query = new URLSearchParams(window.location.search);
	const fromQuery = (name) => (query.has(name) && labels.includes(query.get(name)) ? labels.indexOf(query.get(name)) : null);
	// A page can say which pair to open on (the same run in two executions, say).
	const fromDefault = (name) => (dataset.default && labels.includes(dataset.default[name]) ? labels.indexOf(dataset.default[name]) : null);
	state.base = fromQuery('base') ?? fromDefault('base') ?? state.base;
	state.cand = fromQuery('cand') ?? fromDefault('cand') ?? state.cand;
	if (state.base === state.cand) state.cand = state.base === 0 ? count - 1 : 0;
	if (query.get('runs') === 'all' && canShowAll) state.all = true;
	if (query.get('runs') === 'pair') state.all = false;

	const runsToShow = () => (state.all ? [state.base, ...labels.map((_, i) => i).filter((i) => i !== state.base)] : [state.base, state.cand]);

	function render() {
		const runs = runsToShow();
		changesBox.replaceChildren(...candidateModels(dataset, runs).map(candidateBlock));
		changesBox.append(el('p', { class: 'muted table-note' }, 'Changes in own time - time outside nested regions - so each change is counted once, where it happened.'));
		const model = compareTableModel(dataset, runs);
		tableBox.replaceChildren(
			compareTable(model, state.metric, state.order),
			el('p', { class: 'muted table-note' }, model.per_rank ? "The runs have different rank counts, so total and own times are per rank: each region's time divided by the ranks that ran it. " : '', 'Regions are matched by call path. ', el('em', {}, 'own'), ' is the time outside nested regions, so own-time changes add up to the change in the session. Changes under 2% are greyed out; hover a change for its size in seconds. Click a row to highlight the region in the charts.'),
		);
		baseSelect.value = String(state.base);
		candSelect.value = String(state.cand);
		candSelect.disabled = state.all;
		if (allBox) allBox.checked = state.all;
		for (const button of document.querySelectorAll('[data-compare-metric]')) button.setAttribute('aria-pressed', String(button.dataset.compareMetric === state.metric));
		for (const button of document.querySelectorAll('[data-compare-sort]')) button.setAttribute('aria-pressed', String(button.dataset.compareSort === state.order));
		// The URL names the comparison, so it can be shared.
		const next = new URLSearchParams(window.location.search);
		next.set('base', labels[state.base]);
		next.set('cand', labels[state.cand]);
		if (canShowAll) next.set('runs', state.all ? 'all' : 'pair');
		window.history.replaceState(null, '', `${window.location.pathname}?${next}${window.location.hash}`);
		window.scopeProfilerRefilter?.();
		window.scopeProfilerMarkSelection?.();
		// The charts of several runs follow the choice: only the compared runs are drawn.
		window.scopeProfilerComparedRuns = runs.map((index) => labels[index]);
		window.dispatchEvent(new CustomEvent('scopeprofiler:compared'));
	}

	baseSelect.addEventListener('change', () => {
		state.base = Number(baseSelect.value);
		if (state.cand === state.base) state.cand = state.base === 0 ? count - 1 : 0;
		render();
	});
	candSelect.addEventListener('change', () => {
		state.cand = Number(candSelect.value);
		if (state.base === state.cand) state.base = state.cand === 0 ? count - 1 : 0;
		render();
	});
	document.getElementById('sp-compare-swap')?.addEventListener('click', () => {
		[state.base, state.cand] = [state.cand, state.base];
		render();
	});
	allBox?.addEventListener('change', () => {
		state.all = allBox.checked;
		render();
	});
	for (const button of document.querySelectorAll('[data-compare-metric]')) {
		button.addEventListener('click', () => {
			state.metric = button.dataset.compareMetric;
			render();
		});
	}
	for (const button of document.querySelectorAll('[data-compare-sort]')) {
		button.addEventListener('click', () => {
			state.order = button.dataset.compareSort;
			render();
		});
	}

	// For scripts/ui_smoke.py, which checks these against the Python's own models.
	window.scopeProfilerCompare = { dataset, candidateModels, compareTableModel };
	render();
}

init();
