#!/usr/bin/env bash
# Abre la demo interactiva (http://localhost:8501).
set -euo pipefail
cd "$(dirname "$0")"
.venv/bin/python -m streamlit run app/app.py
