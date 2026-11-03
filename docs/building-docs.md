# Building and publishing the documentation

The setup follows PAMI's Sphinx configuration: the Read the Docs theme,
API documentation from docstrings, and source-code links. MyST adds support
for cuPAMI's existing Markdown guides.

## Build locally

From the repository root:

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -b html -W --keep-going docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser. The build imports the local
source for API documentation and does not run the mining examples. CUDA,
CuPy, and the full package dependencies are not required to build the docs.

## Read the Docs

The root `.readthedocs.yaml` uses Ubuntu 24.04 and Python 3.13, installs
`docs/requirements.txt`, and builds with `docs/conf.py`.

After pushing these files to the branch connected to your existing Read the Docs
project, open its **Builds** page and trigger a build of **latest**. Future pushes
build automatically when the repository webhook is enabled. Open the build log
if a build fails; warnings are treated as build failures.

Configuration reference:
[Read the Docs v2](https://docs.readthedocs.com/platform/stable/config-file/v2.html).
