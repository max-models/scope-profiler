// The lines of `text` after the first line containing `start`, up to the next line containing
// `end`: Sphinx's literalinclude `:start-after:` and `:end-before:`, for the generated outputs.
export function between(text: string, start: string, end: string): string {
	const lines = text.split('\n');
	const first = lines.findIndex((line) => line.includes(start)) + 1;
	const last = lines.findIndex((line, index) => index >= first && line.includes(end));
	return lines.slice(first, last < 0 ? undefined : last).join('\n');
}
