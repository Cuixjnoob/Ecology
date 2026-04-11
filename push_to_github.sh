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
# 显式加入 runs/，这样实验结果目录也会随代码一起提交到 GitHub。
if [[ -d runs ]]; then
  git add runs
fi

if git diff --cached --quiet; then
  echo "没有可提交的变更。"
  git status -sb
  exit 0
fi

echo "==> 提交信息: $MESSAGE"
git commit -m "$MESSAGE"

# GitHub HTTPS 在部分网络环境下会出现 HTTP/2 framing / SSL timeout。
# 这里使用更稳的 HTTP/1.1，并关闭低速超时限制，失败时自动重试 3 次。
echo "==> 推送到 GitHub"
for attempt in 1 2 3; do
  echo "   尝试 $attempt/3"
  if git \
    -c http.version=HTTP/1.1 \
    -c http.postBuffer=524288000 \
    -c http.lowSpeedLimit=0 \
    -c http.lowSpeedTime=999999 \
    push origin "$BRANCH"; then
    echo "✅ 推送完成"
    git status -sb
    exit 0
  fi
  if [[ "$attempt" != "3" ]]; then
    echo "   推送失败，5 秒后重试..."
    sleep 5
  fi
done

echo "❌ 推送仍失败。请稍后重试，或检查网络 / 代理设置。"
exit 1
