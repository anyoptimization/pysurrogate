"""Sphinx configuration for the pysurrogate documentation.

Notebooks are executed out-of-band by the docs runner (jupyter-cache); Sphinx
itself never executes them (``nbsphinx_execute = 'never'``) — it only renders the
already-hydrated ``.ipynb`` files. Unlike the single-page pysampling docs, this is
a multi-page site: the stock left/right sidebars are kept (real toctree in
``index.md``), and autodoc/napoleon/numpydoc power the API reference page.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

import pysurrogate  # noqa: E402

project = "pysurrogate"
copyright = "2026, Julian Blank"
author = "Julian Blank"
release = pysurrogate.__version__

extensions = [
    "nbsphinx",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

exclude_patterns = ["build", "**.ipynb_checkpoints"]

html_theme = "sphinx_book_theme"
html_logo = "_static/pysurrogate.png"
html_title = "pysurrogate"
html_baseurl = "https://anyoptimization.com/projects/pysurrogate/"
# Mirrors pymoo's book-theme setup (minimal, stock knobs): light mode, GitHub
# repo/issue/source buttons in the navbar. Multi-page site — keep the stock left
# nav (the index.md toctree) and the right in-page TOC (no sidebar override).
html_theme_options = {
    "repository_url": "https://github.com/anyoptimization/pysurrogate",
    "repository_branch": "main",
    "path_to_docs": "docs/source",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_source_button": True,
    "use_download_button": False,
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
templates_path = ["_templates"]
# Shipped to the site root: /llms.txt (LLM-readable project summary).
html_extra_path = ["llms.txt"]
html_copy_source = False
html_sourcelink_suffix = ""

# Notebooks are pre-executed by the docs runner; Sphinx must never re-execute them.
nbsphinx_execute = "never"

# Google-style docstrings (no types in the docstring — annotations carry them).
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# autodoc defaults for the API page. `undoc-members` is off: the dataclasses
# (Prediction, Split, Result, Evaluation, Metric) document their fields in the
# class docstring's Attributes section (rendered by napoleon), so also emitting
# them as autodoc members would duplicate every attribute.
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"
