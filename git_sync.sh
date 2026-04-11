#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ ! -d .git ]; then
  echo "错误：当前目录不是 Git 仓库：$ROOT_DIR" >&2
  exit 1
fi

COMMIT_MESSAGE="${1:-update: $(date '+%Y-%m-%d %H:%M:%S')}"

printf '\n[1/5] 当前仓库状态\n'
git status --short --branch

printf '\n[2/5] 拉取远程最新代码\n'
git pull --no-rebase

printf '\n[3/5] 暂存本地修改\n'
git add .

if git diff --cached --quiet; then
  printf '\n没有可提交的修改。\n'
else
  printf '\n[4/5] 提交本地修改\n'
  git commit -m "$COMMIT_MESSAGE"
fi

printf '\n[5/5] 推送到 GitHub\n'
git push

printf '\n同步完成。\n'
git status --short --branch
