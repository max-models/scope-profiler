# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import shutil
import sys

# Make the package importable for autodoc
sys.path.insert(0, os.path.abspath("../../src"))


def copy_tutorials(app):
    confdir = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(confdir, "..", "..", "tutorials")
    dst = os.path.join(confdir, "tutorials")

    if os.path.exists(dst):
        shutil.rmtree(dst)

    if os.path.exists(src):
        shutil.copytree(src, dst)


def copy_figures(app):
    # conf.py lives in docs/source/, figures/ is at repo root
    confdir = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(confdir, "..", "..", "figures")
    dst = os.path.join(confdir, "_static", "figures")

    if os.path.exists(dst):
        shutil.rmtree(dst)

    if os.path.exists(src):
        shutil.copytree(src, dst)


def setup(app):
    app.connect("builder-inited", copy_tutorials)
    app.connect("builder-inited", copy_figures)
    app.add_css_file("custom.css")


# -- Project information -----------------------------------------------------

project = "scope-profiler"
copyright = "2025, Max"
author = "Max"

# -- General configuration ---------------------------------------------------

extensions = [
    "nbsphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

exclude_patterns = []

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# MyST settings
myst_enable_extensions = [
    "colon_fence",
    "fieldlist",
]
myst_heading_anchors = 3

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "h5py": ("https://docs.h5py.org/en/stable", None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"

html_static_path = ["_static"]
templates_path = ["_templates"]

html_theme_options = {
    "repository_url": "https://github.com/max-models/scope-profiler",
    "repository_branch": "devel",
    "use_repository_button": True,
    "show_toc_level": 3,
    "secondary_sidebar_items": ["page-toc"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/max-models/scope-profiler",
            "icon": "fab fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/scope-profiler/",
            "icon": "fas fa-box",
            "type": "fontawesome",
        },
    ],
}
