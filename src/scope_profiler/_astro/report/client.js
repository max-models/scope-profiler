// The interactions of the report components, a port of the standalone HTML report's script: region
// selection shared by every view, the collapsible call tree, the region filter, sortable tables,
// the comparison table's metric and order, and the "show all" buttons. The components render the
// same DOM hooks (data-region, tr.select-row, tbody[data-path], ...) the report does.

function selection() {
	const listeners = [];
	let selectedRegion = null;
	const status = document.getElementById('region-selection');
	const clear = document.getElementById('clear-region-selection');
	const toast = document.getElementById('region-toast');
	// Fixed to the window, so it lives outside the page content's stacking context, which would
	// put it below Starlight's side columns.
	if (toast) document.body.appendChild(toast);
	let toastTimer = null;
	const SELECTABLE = 'tbody[data-region] > .region-row, tr.select-row[data-region]';

	const owner = (row) => (row.classList.contains('region-row') ? row.parentNode : row);

	// A function longer than its capped box opens scrolled to its hottest line.
	function scrollToHotLines(detail) {
		detail.querySelectorAll('.lp-scroll').forEach((box) => {
			const hot = box.querySelector('tr.lp-hot');
			if (!hot || box.dataset.scrolled) return;
			box.dataset.scrolled = '1';
			const head = box.querySelector('thead');
			const offset = hot.offsetTop - (head ? head.offsetHeight : 0);
			if (offset + hot.offsetHeight > box.clientHeight) {
				box.scrollTop = offset - (box.clientHeight - hot.offsetHeight) / 3;
			}
		});
	}

	function reveal(target) {
		const body = owner(target);
		if (target.classList.contains('region-row')) {
			if (window.scopeProfilerRevealRegion) window.scopeProfilerRevealRegion(body);
			const detail = body.querySelector('.region-detail');
			if (detail) {
				detail.hidden = false;
				target.setAttribute('aria-expanded', 'true');
				scrollToHotLines(detail);
			}
		}
		target.scrollIntoView({ behavior: 'smooth', block: 'center' });
	}

	function hideToast() {
		if (toast) toast.hidden = true;
		window.clearTimeout(toastTimer);
	}

	function showToast(region, target) {
		if (!toast) return;
		toast.querySelector('.toast-text').textContent = 'Highlighted ' + region;
		const jump = toast.querySelector('.toast-jump');
		jump.hidden = !target;
		jump.onclick = () => {
			hideToast();
			if (target) reveal(target);
		};
		toast.hidden = false;
		window.clearTimeout(toastTimer);
		toastTimer = window.setTimeout(hideToast, 6000);
	}

	// How a selection moves the page depends on where it came from: "scroll" jumps to the region's
	// table row (the hotspot list); "toast" stays put and offers the jump (chart clicks); "none"
	// does nothing (a click on the table row itself).
	function select(region, run, mode) {
		mode = mode || 'scroll';
		selectedRegion = region || null;
		let target = null;
		document.querySelectorAll(SELECTABLE).forEach((row) => {
			const data = owner(row).dataset;
			const match = selectedRegion !== null && data.region === selectedRegion;
			row.classList.toggle('region-selected', match);
			if (match && !target && (!run || data.run === run)) target = row;
		});
		if (status) status.textContent = selectedRegion ? 'Highlighted: ' + selectedRegion : '';
		if (clear) clear.hidden = !selectedRegion;
		listeners.forEach((listener) => {
			try {
				listener(selectedRegion);
			} catch (error) {
				/* keep other views responsive */
			}
		});
		if (!selectedRegion) {
			hideToast();
			return;
		}
		if (mode === 'scroll' && target) reveal(target);
		else if (mode === 'toast') showToast(selectedRegion, target);
	}

	window.scopeProfilerSelectRegion = select;
	window.scopeProfilerOnRegionSelect = (listener) => {
		listeners.push(listener);
		listener(selectedRegion);
	};
	if (clear) clear.addEventListener('click', () => select(null));
	if (toast) toast.querySelector('.toast-close').addEventListener('click', hideToast);

	document.querySelectorAll('.region-row').forEach((row) => {
		function toggleDetail() {
			const detail = row.parentNode.querySelector('.region-detail');
			if (!detail) return;
			detail.hidden = !detail.hidden;
			row.setAttribute('aria-expanded', String(!detail.hidden));
			if (!detail.hidden) scrollToHotLines(detail);
			const body = row.closest('tbody[data-region]');
			if (body) select(body.dataset.region, body.dataset.run, 'none');
		}
		row.addEventListener('click', toggleDetail);
		row.addEventListener('keydown', (event) => {
			if (event.target !== row || (event.key !== 'Enter' && event.key !== ' ')) return;
			event.preventDefault();
			toggleDetail();
		});
	});

	// Delegated, because the comparison tables are drawn after the page loads.
	document.addEventListener('click', (event) => {
		const row = event.target.closest?.('tr.select-row');
		if (!row) return;
		const region = row.dataset.region;
		select(region === selectedRegion ? null : region, row.dataset.run, 'none');
	});

	// Mark the selected region in rows drawn since the selection was made.
	window.scopeProfilerMarkSelection = () => {
		document.querySelectorAll(SELECTABLE).forEach((row) => {
			row.classList.toggle('region-selected', selectedRegion !== null && owner(row).dataset.region === selectedRegion);
		});
	};

	document.querySelectorAll('.hotspot').forEach((button) => {
		button.addEventListener('click', () => select(button.dataset.region, button.dataset.run, 'scroll'));
	});
}

// Collapsible call tree: each region's tbody carries its call path, so a collapsed row hides every
// tbody whose path extends its own. Filtering and sorting show rows outside the tree and switch
// collapsing off.
function callTree() {
	const bodies = (table) => Array.prototype.filter.call(table.tBodies, (body) => body.dataset.path !== undefined);

	function refresh(table) {
		const prefixes = bodies(table)
			.filter((body) => body.classList.contains('tree-collapsed'))
			.map((body) => body.dataset.path + ' > ');
		bodies(table).forEach((body) => {
			const path = body.dataset.path;
			body.classList.toggle(
				'tree-hidden',
				prefixes.some((prefix) => path.indexOf(prefix) === 0),
			);
		});
	}

	function setCollapsed(body, collapsed) {
		const button = body.querySelector('.tree-toggle');
		if (!button) return;
		body.classList.toggle('tree-collapsed', collapsed);
		button.setAttribute('aria-expanded', String(!collapsed));
		button.textContent = collapsed ? '▸' : '▾';
	}

	// Expand every collapsed ancestor, so a selected region can be shown.
	window.scopeProfilerRevealRegion = (body) => {
		const table = body.closest('table.region-stats');
		const path = body.dataset.path;
		if (!table || path === undefined) return;
		bodies(table).forEach((other) => {
			if (path.indexOf(other.dataset.path + ' > ') === 0) setCollapsed(other, false);
		});
		refresh(table);
	};

	document.querySelectorAll('.tree-toggle').forEach((button) => {
		button.addEventListener('click', (event) => {
			event.stopPropagation(); // the row's own click opens its detail
			const body = button.closest('tbody');
			setCollapsed(body, !body.classList.contains('tree-collapsed'));
			refresh(body.closest('table'));
		});
	});

	document.querySelectorAll('[data-tree-action]').forEach((button) => {
		button.addEventListener('click', () => {
			const table = document.getElementById(button.dataset.table);
			if (!table) return;
			const collapse = button.dataset.treeAction === 'collapse';
			// "Collapse all" keeps the top level open: a lone root row says nothing.
			bodies(table).forEach((body) => setCollapsed(body, collapse && Number(body.dataset.depth) > 0));
			refresh(table);
		});
	});
}

// Region filtering: each comma-separated term is a case-insensitive substring of the region name,
// "^" anchors a term to the start, and an empty box shows every region. The charts register
// through scopeProfilerOnRegionFilter and redraw when the filter changes.
function regionFilter() {
	const input = document.getElementById('region-filter');
	if (!input) return;
	// Table headers stick just below the bar (and the site's header); the bar's height changes when it wraps.
	const bar = input.closest('.filter-bar');
	if (bar && window.ResizeObserver) {
		new ResizeObserver(() => {
			document.documentElement.style.setProperty(
				'--filter-bar-height',
				`calc(var(--sl-nav-height, 3.5rem) + ${bar.offsetHeight}px)`,
			);
		}).observe(bar);
	}
	const count = document.getElementById('region-filter-count');
	const listeners = [];

	const terms = () =>
		input.value
			.split(',')
			.map((term) => term.trim().toLowerCase())
			.filter(Boolean);

	function matches(name, active) {
		const lowered = String(name == null ? '' : name).toLowerCase();
		return active.some((term) =>
			term.charAt(0) === '^' ? lowered.indexOf(term.slice(1)) === 0 : lowered.indexOf(term) !== -1,
		);
	}

	// The Regions/Depth sliders' region set, handed over by the chart module; null while they limit
	// nothing.
	let limit = null;
	const allowed = (region, active) => (!active.length || matches(region, active)) && (!limit || limit.has(region));

	function filterTables(active) {
		const limiting = active.length > 0 || limit !== null;
		let shown = 0;
		let total = 0;
		document.querySelectorAll('table.region-stats').forEach((table) => {
			let visible = 0;
			let filterable = 0;
			Array.prototype.forEach.call(table.tBodies, (tbody) => {
				const region = tbody.dataset.region;
				if (region === undefined) return;
				filterable += 1;
				const match = allowed(region, active);
				tbody.hidden = !match;
				if (match) visible += 1;
			});
			const empty = table.querySelector('tbody.region-empty');
			if (empty) empty.hidden = !filterable || visible > 0;
			table.classList.toggle('filtering', limiting);
			shown += visible;
			total += filterable;
		});
		// Every other per-region element: hotspots, load balance, the comparison table.
		document.querySelectorAll('[data-filter-region]').forEach((item) => {
			item.hidden = limiting && !allowed(item.dataset.filterRegion, active);
		});
		document.querySelectorAll('.hotspots').forEach((block) => {
			block.hidden = !block.querySelector('li[data-filter-region]:not([hidden])');
		});
		if (count) {
			count.textContent = !limiting || !total ? '' : `${shown} of ${total} region${total === 1 ? '' : 's'}`;
		}
	}

	function apply() {
		const active = terms();
		filterTables(active);
		listeners.forEach((listener) => {
			try {
				listener(active.slice());
			} catch (error) {
				/* one chart must not stop the rest */
			}
		});
	}

	// The chart module passes the sliders' region set here, so the tables keep to the same regions.
	window.scopeProfilerSetRegionLimit = (regions) => {
		limit = regions;
		filterTables(terms());
	};
	// Charts register here to redraw themselves when the filter changes.
	window.scopeProfilerOnRegionFilter = (listener) => {
		listeners.push(listener);
		listener(terms());
	};

	// Re-apply the filter to elements drawn since (the comparison tables).
	window.scopeProfilerRefilter = () => filterTables(terms());

	let timer = null;
	input.addEventListener('input', () => {
		window.clearTimeout(timer);
		timer = window.setTimeout(apply, 150);
	});
	apply();
}

function sortableTables() {
	document.querySelectorAll('table.region-stats').forEach((table) => {
		const headerRow = table.tHead.rows[0];
		Array.prototype.forEach.call(headerRow.cells, (th) => {
			const key = th.dataset.key;
			if (!key) return;
			th.addEventListener('click', () => {
				const ascending = th.dataset.sortDir !== 'asc';
				Array.prototype.forEach.call(headerRow.cells, (cell) => {
					delete cell.dataset.sortDir;
				});
				th.dataset.sortDir = ascending ? 'asc' : 'desc';
				// Sorted rows no longer follow the call tree, so drop its indentation.
				table.classList.add('flat');
				const bodies = Array.prototype.slice.call(table.tBodies);
				bodies.sort((a, b) => {
					const av = a.dataset[key];
					const bv = b.dataset[key];
					const an = parseFloat(av);
					const bn = parseFloat(bv);
					const cmp = !isNaN(an) && !isNaN(bn) ? an - bn : String(av).localeCompare(String(bv));
					return ascending ? cmp : -cmp;
				});
				bodies.forEach((tbody) => table.appendChild(tbody));
			});
		});
	});
}

function showAllButtons() {
	document.querySelectorAll('[data-show-all]').forEach((button) => {
		button.addEventListener('click', () => {
			const table = document.getElementById(button.dataset.showAll);
			if (!table) return;
			const all = table.classList.toggle('show-all');
			button.textContent = all ? button.dataset.lessLabel : button.dataset.moreLabel;
		});
	});
}

export function initReport() {
	if (window.__scopeProfilerReport) return;
	window.__scopeProfilerReport = true;
	selection();
	callTree();
	regionFilter();
	sortableTables();
	showAllButtons();
}
