#!/bin/bash
cd "$(dirname "$0")"

echo "Starting agent..."
.venv/bin/python -m streamlit run app.py
