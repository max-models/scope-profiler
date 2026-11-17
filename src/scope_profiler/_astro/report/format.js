// Small formatting helpers shared by the report components.

/** Python's `{:.6g}`, close enough: six significant digits, no trailing zeros. */
export function g6(value) {
	return value === null || value === undefined ? '-' : String(parseFloat(Number(value).toPrecision(6)));
}

/** `{:,}`: a count with thousands separators. */
export function count(value) {
	return Number(value).toLocaleString('en-US');
}

/** A tiny trend line of a region's call durations, as the points of an SVG polyline. */
export function sparkline(values, width = 90, height = 22) {
	if (!values || values.length < 2) return null;
	const lo = Math.min(...values);
	const hi = Math.max(...values);
	const span = hi - lo;
	const points = values
		.map((value, index) => {
			const x = (index * width) / (values.length - 1);
			const y = span === 0 ? height / 2 : height - 2 - ((value - lo) / span) * (height - 4);
			return `${x.toFixed(1)},${y.toFixed(1)}`;
		})
		.join(' ');
	return { points, fill: `0,${height} ${points} ${width},${height}`, width, height };
}

/** The slider readouts; the same wording as the standalone report. */
export const topLabel = (value, max) => (value >= max ? `all ${max}` : `top ${value} of ${max}`);
export const depthLabel = (value, max) =>
	value >= max ? 'all levels' : value === 0 ? 'top level only' : `${value} level${value > 1 ? 's' : ''} of children`;

/** The background of a matrix cell: red above the region's mean, blue below. */
export const shadeStyle = (shade) =>
	shade ? `background: rgba(${shade.sign > 0 ? '220,38,38' : '37,99,235'}, ${shade.alpha.toFixed(2)})` : undefined;
