#!/bin/sh
set -eu
cd "$(dirname "$0")"

# Build into a fresh directory so older releases are never uploaded.
release_dir=$(mktemp -d)
trap 'python3 -c "import shutil, sys; shutil.rmtree(sys.argv[1])" "$release_dir"' EXIT

echo "Running setup"
python3 setup.py sdist --dist-dir "$release_dir" bdist_wheel --dist-dir "$release_dir"
python3 -m twine check "$release_dir"/*

echo "Uploading cuPAMI to TestPyPI"
python3 -m twine upload --repository-url https://test.pypi.org/legacy/ "$release_dir"/*
echo "Completed: https://test.pypi.org/project/cupami/"
