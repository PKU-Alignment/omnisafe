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
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
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

html_theme = 'press'
html_static_path = ['_static']
