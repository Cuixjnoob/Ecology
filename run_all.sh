#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"

ensure_python_env() {
  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not found. Please install Python 3 first." >&2
    exit 1
  fi

  if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR" >/dev/null
  fi

  "$PYTHON_BIN" -m pip install --upgrade pip -q >/dev/null
  "$PYTHON_BIN" -m pip install -r "$ROOT_DIR/requirements.txt" -q >/dev/null
}

ensure_git_repo() {
  if [ ! -d "$ROOT_DIR/.git" ]; then
    echo "当前目录不是 Git 仓库：$ROOT_DIR" >&2
    exit 1
  fi
}

git_status() {
  ensure_git_repo
  cd "$ROOT_DIR"
  git status --short --branch
  echo "---"
  git remote -v
}

git_sync() {
  ensure_git_repo
  cd "$ROOT_DIR"

  local commit_message="${1:-auto sync: $(date '+%Y-%m-%d %H:%M:%S')}"

  if ! git diff --quiet || ! git diff --cached --quiet || [ -n "$(git ls-files --others --exclude-standard)" ]; then
    git add .
    git commit -m "$commit_message"
  else
    echo "没有新的本地修改，跳过 commit。"
  fi

  git pull --no-rebase
  git push
}

run_experiment() {
  ensure_python_env
  cd "$ROOT_DIR"
  "$PYTHON_BIN" -m scripts.run_partial_lv_mvp "$@"
}

COMMAND="${1:-run}"

case "$COMMAND" in
  run)
    shift || true
    run_experiment "$@"
    ;;
  status)
    git_status
    ;;
  sync)
    shift || true
    git_sync "$@"
    ;;
  *)
    echo "用法："
    echo "  ./run_all.sh run [实验参数...]"
    echo "  ./run_all.sh status"
    echo "  ./run_all.sh sync [提交信息]"
    exit 1
    ;;
esac
