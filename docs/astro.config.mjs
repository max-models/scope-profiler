// @ts-check
// The shared max-models docs setup (@max-models/starlight-theme): served at
// https://max-models.github.io/scope-profiler/, with math, the API reference and a link check.
import { defineDocs } from '@max-models/starlight-theme';

// The page names of the old Sphinx site, which the README of earlier releases links to.
const guides = [
	'c',
	'concurrency',
	'configuration',
	'dashboard',
	'fortran',
	'hdf5_and_python_api',
	'likwid',
	'line_profiler',
	'mcp',
	'modes',
	'mpi',
	'notebook_magics',
	'overhead',
	'plot_cli',
	'plotly_package',
	'regions',
	'try_it',
];
const tutorials = [
	'01_getting_started',
	'02_postprocessing',
	'03_visualization',
	'04_profiling_modes',
	'05_custom_analysis',
	'06_custom_plots',
	'07_notebook_magics',
	'08_reports_and_comparison',
];
const dashed = (name) => name.replaceAll('_', '-');

export default defineDocs({
	repo: 'scope-profiler',
	title: 'scope-profiler',
	description: 'A lightweight, low-overhead profiling framework for Python and HPC applications.',
	pythonPackage: 'scope_profiler',
	nav: [
		{ label: 'Getting started', link: '/getting-started/installation/' },
		{ label: 'User guide', link: '/guide/configuration/', section: '/guide/' },
		{ label: 'Tutorials', link: '/tutorials/' },
		{ label: 'Reference', link: '/reference/api/', section: '/reference/' },
		{ label: 'Examples', link: '/examples/' },
		{ label: 'API', link: '/api/scope_profiler/', section: '/api/' },
	],
	sidebar: [
		{
			label: 'Getting started',
			items: [
				{ label: 'Installation', slug: 'getting-started/installation' },
				{ label: 'Quickstart', slug: 'getting-started/quickstart' },
				{ slug: 'guide/modes' },
				{ label: 'Try it: drag and drop a profile', slug: 'getting-started/try-it' },
			],
		},
		{
			label: 'User guide',
			items: [
				'configuration',
				'regions',
				'concurrency',
				'hdf5-and-python-api',
				'plot-cli',
				'plotly-package',
				'dashboard',
				'mcp',
				'notebook-magics',
				'mpi',
				'likwid',
				'line-profiler',
				'fortran',
				'c',
				'overhead',
			].map((page) => ({ slug: `guide/${page}` })),
		},
		{
			label: 'Interactive Jupyter tutorials',
			// the tutorials are notebooks in notebooks/tutorials/, built into the guide section with the guides
			items: [
				{ label: 'Overview', slug: 'tutorials' },
				...tutorials.map((page) => ({ slug: `guide/${dashed(page)}` })),
			],
		},
		{
			label: 'Reference',
			items: [
				{ label: 'Python API', slug: 'reference/api' },
				{ label: 'CLI', slug: 'reference/cli' },
			],
		},
		{ label: 'Examples & workflows', items: [{ slug: 'examples' }] },
		{ label: 'Development', items: [{ label: 'Building the docs', slug: 'development/docs' }] },
	],
	// the Fortran guide's code blocks; Shiki names the language fortran-free-form
	starlight: { expressiveCode: { shiki: { langAlias: { fortran: 'fortran-free-form' } } } },
	astro: {
		redirects: {
			'/installation.html': '/getting-started/installation/',
			'/quickstart.html': '/getting-started/quickstart/',
			'/api.html': '/reference/api/',
			'/cli.html': '/reference/cli/',
			'/examples.html': '/examples/',
			'/tutorials.html': '/tutorials/',
			...Object.fromEntries(
				guides.map((page) => [`/guide/${page}.html`, page === 'try_it' ? '/getting-started/try-it/' : `/guide/${dashed(page)}/`]),
			),
			'/guide/try-it/': '/getting-started/try-it/',
			// the tutorials moved into the guide section, next to the notebooks of the guide
			...Object.fromEntries(
				tutorials.flatMap((page) => [
					[`/tutorials/${page}.html`, `/guide/${dashed(page)}/`],
					[`/tutorials/${dashed(page)}/`, `/guide/${dashed(page)}/`],
				]),
			),
		},
	},
});
