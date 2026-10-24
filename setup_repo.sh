#!/usr/bin/env bash
# Set up the existing checkout locally; no GitHub publication or repository creation.
set -euo pipefail
cd "$(dirname "$0")"
uv venv --python 3.12.13
uv pip install --python .venv/bin/python -r requirements-lock.txt
.venv/bin/python -m pytest -q
printf '\nReady. See README.md for benchmark reproduction commands.\n'
