"""Sphinx configuration following PAMI's Sphinx/Read the Docs structure."""
import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = 'cuPAMI'
author = 'Rage Uday Kiran'
copyright = '2026, Rage Uday Kiran'
# Read the package version without executing setup.py or importing CUDA.
tree = ast.parse((ROOT / 'setup.py').read_text(encoding='utf-8'))
release = next(
    ast.literal_eval(keyword.value)
    for node in ast.walk(tree) if isinstance(node, ast.Call)
    for keyword in node.keywords if keyword.arg == 'version'
)

extensions = [
    'myst_parser', 'sphinx.ext.autodoc', 'sphinx.ext.viewcode',
    'sphinx.ext.napoleon', 'sphinx.ext.duration',
]
source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}
root_doc = 'index'
exclude_patterns = ['_build', 'benchmarks', 'Thumbs.db', '.DS_Store']
html_theme = 'sphinx_rtd_theme'
html_title = f'{project} {release}'
html_theme_options = {'navigation_depth': 3}
myst_heading_anchors = 3
autodoc_member_order = 'bysource'
