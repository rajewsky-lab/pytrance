project = "pyTrance"
author = "Leon Strenger"
release = "0.1.0"

extensions = [
    "autoapi.extension",
    "sphinx.ext.napoleon",      # parses NumPy-style docstrings
    "sphinx.ext.viewcode",      # adds [source] links
    "sphinx.ext.intersphinx",   # cross-links to numpy, python docs etc.
    "myst_nb",                 
]

# Point autoapi at your package source
autoapi_dirs = ["../pytrance"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_add_toctree_entry = False

napoleon_numpy_docstring = True
napoleon_google_docstring = False

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/rajewsky-lab/pyTrance",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "anndata": ("https://anndata.readthedocs.io/en/latest", None),
}

exclude_patterns = ["_build"]
