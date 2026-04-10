#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Please install Python 3 first." >&2
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR" >/dev/null
fi

"$PYTHON_BIN" -m pip install --upgrade pip -q >/dev/null
"$PYTHON_BIN" -m pip install -r "$ROOT_DIR/requirements.txt" -q >/dev/null

cd "$ROOT_DIR"
"$PYTHON_BIN" -m scripts.run_partial_lv_mvp "$@"
