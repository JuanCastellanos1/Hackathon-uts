#!/usr/bin/env bash
# Crea el entorno virtual e instala las dependencias (Linux/macOS, solo la primera vez).
set -euo pipefail
cd "$(dirname "$0")"
PY=$(command -v python3.12 || command -v python3 || command -v python)
[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
echo "Instalación completa. Siguiente paso: ./ejecutar_todo.sh o ./demo.sh"
