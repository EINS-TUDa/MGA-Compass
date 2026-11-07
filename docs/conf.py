"""Sphinx configuration for the MGA Compass / MGA Surveyor documentation.

Build locally from the repository root:

    uv run --all-packages --group docs sphinx-build -b html docs docs/_build/html

The reveal.js presentations under docs/presentation/ are not part of the
Sphinx build; the GitHub Pages workflow copies them into the built site at
/presentations/ after Sphinx runs.
"""

project = "MGA Compass"
author = "Sina Hajikazemi, Florian Steinke"
copyright = "2026, EINS, TU Darmstadt"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "sphinx_copybutton",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "presentation", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = ["colon_fence", "dollarmath", "deflist", "attrs_inline"]
myst_heading_anchors = 3

# The docstrings in both packages use single backticks for inline code
# (Markdown habit), so render them as code rather than reST's default
# "interpreted text" (italics).
default_role = "code"

autodoc_member_order = "bysource"
autodoc_typehints = "description"
# undoc-members: much of mga_compass.core has no docstrings yet; list those
# functions anyway so the reference is complete and the guides can link to them.
autodoc_default_options = {"members": True, "undoc-members": True, "show-inheritance": True}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
    "pypsa": ("https://docs.pypsa.org/latest", None),
}

html_theme = "furo"
html_title = "MGA Compass"
html_logo = "mga_compass_logo.svg"
html_favicon = "mga_compass_logo.svg"

_GITHUB_ICON = (
    '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">'
    '<path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17'
    '.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68'
    '-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64'
    '-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27'
    ' 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15'
    ' 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013'
    ' 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>'
)
html_theme_options = {
    # Shown at the top of every page until the developers have reviewed the docs.
    "announcement": (
        "<strong>Warning:</strong> this documentation was generated with AI assistance and has "
        "not yet been reviewed by the developers. It may contain errors; the code and the papers "
        "are authoritative."
    ),
    # "Edit this page" / "View source" links point at the repo's docs/ folder.
    "source_repository": "https://github.com/EINS-TUDa/MGA-Compass/",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {"name": "GitHub", "url": "https://github.com/EINS-TUDa/MGA-Compass", "html": _GITHUB_ICON, "class": ""},
    ],
}
