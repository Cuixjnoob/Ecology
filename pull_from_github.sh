#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

BRANCH="${1:-$(git rev-parse --abbrev-ref HEAD)}"

if [[ ! -d .git ]]; then
  echo "错误：当前目录不是 Git 仓库：$ROOT_DIR"
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "错误：检测到未提交修改。请先提交/暂存，避免 pull 覆盖本地工作。"
  git status --short
  exit 1
fi

echo "==> 拉取远程分支: origin/$BRANCH"
git fetch origin "$BRANCH"
git pull --ff-only origin "$BRANCH"

echo "✅ 拉取完成"
git status -sb
