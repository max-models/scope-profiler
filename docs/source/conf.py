# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import shutil

def copy_tutorials(app):
    src = os.path.abspath("../tutorials")
    dst = os.path.abspath("source/tutorials")

    # Remove existing target directory if it exists
    if os.path.exists(dst):
        shutil.rmtree(dst)

    shutil.copytree(src, dst)


def setup(app):
    app.connect("builder-inited", copy_tutorials)
    # app.add_stylesheet("my-styles.css")
    app.add_css_file("custom.css")


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "python-template"
copyright = "2025, Max"
author = "Max"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "myst_parser",  # enable Markdown support
]

exclude_patterns = []

# Recognize both .rst and .md
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Activate the theme.
html_theme = "sphinx_book_theme"


html_static_path = ["_static"]
templates_path = ["_templates"]


# html_sidebars = {"**": []}

html_theme_options = {
    "repository_branch": "devel",
    "show_toc_level": 3,
    "secondary_sidebar_items": ["page-toc"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/max-models/template-python",
            "icon": "fab fa-github",
            "type": "fontawesome",
        },
    ],
}

