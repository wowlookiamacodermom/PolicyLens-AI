#!/bin/bash
cd "$(dirname "$0")"
PY=$(command -v python3 || command -v python)
[ -z "$PY" ] && { echo "Install Python 3.10 or 3.11 from python.org, then run this again."; read -n1; exit 1; }
if [ ! -x .venv/bin/python ]; then echo "First-time setup, this can take a few minutes..."; "$PY" -m venv .venv || exit 1; fi
if [ ! -f .venv/policylens_ready.txt ]; then
  .venv/bin/python -m pip install --upgrade pip && .venv/bin/python -m pip install -r requirements.txt \
    && .venv/bin/python -m pip install -r requirements-ml.txt && echo ready > .venv/policylens_ready.txt || { echo "Setup failed."; read -n1; exit 1; }
fi
mkdir -p ~/.streamlit; [ -f ~/.streamlit/credentials.toml ] || printf '[general]\nemail = ""\n' > ~/.streamlit/credentials.toml
.venv/bin/python -m streamlit run app/streamlit_app.py
