#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
MESSAGE="${1:-chore: update $(date '+%Y-%m-%d %H:%M:%S')}"

if [[ ! -d .git ]]; then
  echo "错误：当前目录不是 Git 仓库：$ROOT_DIR"
  exit 1
fi

echo "==> 当前分支: $BRANCH"

git add -A

if git diff --cached --quiet; then
  echo "没有可提交的变更。"
  git status -sb
  exit 0
fi

echo "==> 提交信息: $MESSAGE"
git commit -m "$MESSAGE"
git push origin "$BRANCH"

echo "✅ 推送完成"
git status -sb
