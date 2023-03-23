# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import pathlib
import sys


# -- Project information -----------------------------------------------------

ROOT_DIR = pathlib.Path(__file__).absolute().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / 'omnisafe'))

project = 'OmniSafe'
copyright = '2022, OmniSafe Team'
author = 'OmniSafe Team'
release = 'v1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

"""
Run before:
conda activate safe
pip install recommonmark
pip install sphinx_markdown_tables
pip install sphinx_design
pip install sphinx_copybutton
pip install sphinx-press-theme
pip install sphinx
"""

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    #'sphinx.ext.imgmath',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.extlinks',
    'recommonmark',
    'sphinx_markdown_tables',
    'sphinx_design',
    'sphinx_copybutton',
    'sphinx_autodoc_typehints',
]

if not os.getenv('READTHEDOCS', None):
    extensions.append('sphinxcontrib.spelling')

source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}
templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

html_theme_options = {
    'light_css_variables': {
        'color-brand-primary': '#4E98C8',
        'color-brand-content': '#67A4BA',
        'sd-color-success': '#5EA69C',
        'sd-color-info': '#76A2DB',
        'sd-color-warning': '#AD677E',
    },
}

math_number_all = True  # Set this option to True if you want all displayed math to be numbered. The default is False.
math_eqref_format = 'Eq.{number}'  # gets rendered as, for example, Eq.10.

# If True, displayed math equations are numbered across pages when numfig
# is enabled. The numfig_secnum_depth setting is respected. The eq, not
# numref, role must be used to reference equation numbers. Default is
# True.

# see http://www.sphinx-doc.org/en/master/usage/configuration.html#confval-numfig
# If true, figures, tables and code-blocks are automatically numbered if they have a caption.
# The numref role is enabled. Obeyed so far only by HTML and LaTeX builders. Default is False.
# The LaTeX builder always assigns numbers whether this option is enabled or not.
numfig_secnum_depth = 3

# A dictionary mapping 'figure', 'table', 'code-block' and 'section' to strings that are used for format of figure numbers.
# As a special character, %s will be replaced to figure number.
# Default is to use 'Fig. %s' for 'figure', 'Table %s' for 'table', 'Listing %s' for 'code-block' and 'Section' for 'section'.
