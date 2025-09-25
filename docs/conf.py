# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from pathlib import Path

# Add the source code to the path
sys.path.insert(0, os.path.abspath("../../src"))

# Try to import the package to check if it's available
package_available = False
try:
    import dacapo_monai

    package_available = True
    print(
        f"Successfully imported dacapo_monai version: {getattr(dacapo_monai, '__version__', 'unknown')}"
    )
except ImportError as e:
    print(f"Warning: Could not import dacapo_monai: {e}")
    print("Documentation will be built with limited functionality")
except Exception as e:
    print(f"Unexpected error importing dacapo_monai: {e}")
    print("Documentation will be built with limited functionality")

project = "DaCapo-MONAI"
copyright = "2025, DaCapo-MONAI Contributors"
author = "DaCapo-MONAI Contributors"
release = "0.1.0"
version = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
]

# Conditionally add autosummary only if package is available and not disabled by env
if package_available and not os.getenv("SPHINX_NO_AUTOSUMMARY"):
    extensions.append("sphinx.ext.autosummary")
else:
    print("Skipping autosummary extension due to import issues or environment setting")

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = "DaCapo-MONAI Documentation"
# html_logo = '_static/logo.png'  # Comment out until we have a logo
# html_favicon = '_static/favicon.ico'  # Comment out until we have a favicon

html_static_path = ["_static"]

html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#2563eb",
        "color-brand-content": "#2563eb",
    },
    "dark_css_variables": {
        "color-brand-primary": "#60a5fa",
        "color-brand-content": "#60a5fa",
    },
    "source_repository": "https://github.com/dacapo-toolbox/dacapo-monai/",
    "source_branch": "main",
    "source_directory": "docs/",
}

# -- Extension configuration -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "ignore-module-all": True,
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"

# Prevent Sphinx from failing on import errors
autodoc_mock_imports_extended = True

# Mock imports for packages that might not be available during documentation build
autodoc_mock_imports = [
    "torch",
    "monai",
    "dacapo_toolbox",
    "gunpowder",
    "funlib",
    "funlib.geometry",
    "funlib.persistence",
    "numpy",
    "zarr",
    "h5py",
    "scipy",
    "skimage",
    "PIL",
    "cv2",
    "matplotlib",
]

# Autosummary settings (only if enabled)
autosummary_generate = package_available and not os.getenv("SPHINX_NO_AUTOSUMMARY")
autosummary_imported_members = False  # Set to False to avoid import issues
autosummary_ignore_module_all = False


# Handle import failures gracefully
def autodoc_skip_member(app, what, name, obj, skip, options):
    """Skip members that can't be imported"""
    if skip:
        return True
    try:
        # Try to access the object to see if it can be imported
        _ = str(obj)
        return False
    except Exception:
        print(f"Skipping {name} due to import error")
        return True


# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "monai": ("https://docs.monai.io/en/stable/", None),
}

# MyST parser settings
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# Copy button settings
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# Mermaid settings
mermaid_output_format = "png"
mermaid_params = [
    "--theme",
    "default",
    "--width",
    "600",
    "--backgroundColor",
    "transparent",
]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_css_files = [
    "custom.css",
]

# Add custom JavaScript
html_js_files = [
    "custom.js",
]

# Suppress warnings for missing references during development
suppress_warnings = ["ref.any", "autosummary", "autodoc"]


def setup(app):
    """Sphinx setup hook"""
    app.connect("autodoc-skip-member", autodoc_skip_member)
    return {"version": "0.1", "parallel_read_safe": True}
