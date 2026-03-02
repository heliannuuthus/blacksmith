#!/usr/bin/env bash
# gsync - 同步 Git 项目到最新主分支
# 单仓库：在 .git 项目内运行，同步当前项目
# 多仓库：在非 .git 目录运行，遍历一级子目录中所有 Git 项目
set -uo pipefail

DIRTY_REPOS=()
ROOT_DIR="$PWD"

sync_repo() {
  local repo_dir="$1"
  local name
  name="$(basename "$repo_dir")"

  local default_branch current_branch

  default_branch="$(git -C "$repo_dir" symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@')"
  [ -z "$default_branch" ] && default_branch="main"

  current_branch="$(git -C "$repo_dir" branch --show-current 2>/dev/null)"

  # 工作区不干净则跳过
  if [ -n "$(git -C "$repo_dir" status --porcelain 2>/dev/null)" ]; then
    DIRTY_REPOS+=("$name ($current_branch)")
    echo "  ⚠ 跳过：存在未提交的改动"
    return
  fi

  # 切换到主分支
  if [ "$current_branch" != "$default_branch" ]; then
    git -C "$repo_dir" checkout "$default_branch" --quiet 2>/dev/null || {
      echo "  ⚠ 跳过：无法切换到 $default_branch"
      return
    }
  fi

  # 拉取最新
  if git -C "$repo_dir" pull origin "$default_branch" --quiet 2>/dev/null; then
    echo "  ✓ 已同步到最新 $default_branch"
  else
    echo "  ⚠ 拉取失败"
  fi
}

echo "========================================"
echo "  gsync - Git 分支同步"
echo "========================================"
echo ""

if [ -d ".git" ]; then
  echo "[$(basename "$PWD")]"
  sync_repo "$PWD"
else
  found=0
  for dir in "$ROOT_DIR"/*/; do
    [ -d "$dir/.git" ] || continue
    found=1
    echo "[$(basename "$dir")]"
    sync_repo "$dir"
    echo ""
  done

  if [ "$found" -eq 0 ]; then
    echo "当前目录及一级子目录下未找到 Git 项目。"
    exit 1
  fi
fi

# 汇总提示
if [ ${#DIRTY_REPOS[@]} -gt 0 ]; then
  echo ""
  echo "========================================"
  echo "  以下项目存在未提交的改动，未执行同步："
  echo "========================================"
  for repo in "${DIRTY_REPOS[@]}"; do
    echo "  • $repo"
  done
  echo ""
  echo "请先提交或 stash 后重新运行 gsync。"
fi
