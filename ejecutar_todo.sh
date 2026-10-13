#!/usr/bin/env bash
# Regenera todo el pipeline y corre las pruebas. Opcional: ./ejecutar_todo.sh --descargar
set -euo pipefail
cd "$(dirname "$0")"
.venv/bin/python run_all.py "$@"
.venv/bin/python -m pytest -q
