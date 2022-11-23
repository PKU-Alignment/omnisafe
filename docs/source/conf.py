# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys


sys.path.insert(0, os.path.abspath('./omnisafe/'))
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

extensions = ['recommonmark', 'sphinx_markdown_tables', 'sphinx_design', 'sphinx_copybutton']


source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}
templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'press'
html_static_path = ['_static']
