from __future__ import annotations

import importlib.metadata
from typing import Any

project = "ooipy"
copyright = "2024"
maintainer = "John Ragland"
version = release = importlib.metadata.version("ooipy")

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
]

source_suffix = [".rst", ".md"]
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]

html_theme = "furo"
html_theme_options: dict[str, Any] = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/Ocean-Data-Lab/ooipy",
            "class": "",
        },
    ],
    "source_repository": "https://github.com/Ocean-Data-Lab/ooipy",
    "source_branch": "main",
    "source_directory": "docs/",
}

html_favicon = "../imgs/OOIPY_favicon.ico"

myst_enable_extensions = [
    "colon_fence",
    "html_admonition",
    "html_image",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "obspy": ("https://docs.obspy.org/", "https://docs.obspy.org/objects.inv"),
    "xarray": ("https://xarray.pydata.org/en/stable/", None),
}

nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
]

always_document_param_types = True

bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "author_year"
