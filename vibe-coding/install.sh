#!/usr/bin/env bash
# ============================================================================
# Vibe Coding Installer
# 将 rules / skills / agents / commands 安装到目标项目的 .cursor 目录
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-.}"

if [ "$TARGET" = "--help" ] || [ "$TARGET" = "-h" ]; then
  cat <<EOF
Usage: $(basename "$0") [TARGET_PROJECT_DIR]

将 vibe-coding 资源安装到目标项目的 .cursor/ 目录。

  TARGET_PROJECT_DIR  目标项目根目录 (默认: 当前目录)

Options:
  --rules-only     仅安装 rules
  --skills-only    仅安装 skills
  --agents-only    仅安装 agents
  --commands-only  仅安装 commands
  --dry-run        仅显示将要执行的操作
  -h, --help       显示帮助信息

示例:
  $(basename "$0") ~/projects/my-app          # 安装全部到 my-app
  $(basename "$0") --rules-only ~/projects/my-app  # 仅安装 rules
  $(basename "$0")                            # 安装全部到当前目录
EOF
  exit 0
fi

# 解析参数
DRY_RUN=false
INSTALL_RULES=true
INSTALL_SKILLS=true
INSTALL_AGENTS=true
INSTALL_COMMANDS=true

for arg in "$@"; do
  case "$arg" in
    --rules-only)    INSTALL_SKILLS=false; INSTALL_AGENTS=false; INSTALL_COMMANDS=false ;;
    --skills-only)   INSTALL_RULES=false; INSTALL_AGENTS=false; INSTALL_COMMANDS=false ;;
    --agents-only)   INSTALL_RULES=false; INSTALL_SKILLS=false; INSTALL_COMMANDS=false ;;
    --commands-only) INSTALL_RULES=false; INSTALL_SKILLS=false; INSTALL_AGENTS=false ;;
    --dry-run)       DRY_RUN=true ;;
    -*)              ;; # 忽略未知 flag
    *)               TARGET="$arg" ;;
  esac
done

TARGET="$(cd "$TARGET" 2>/dev/null && pwd || echo "$TARGET")"

echo "========================================"
echo "  Vibe Coding Installer"
echo "========================================"
echo "源目录:   $SCRIPT_DIR"
echo "目标项目: $TARGET"
echo ""

copy_dir() {
  local src="$1" dst="$2" label="$3"
  if [ ! -d "$src" ]; then
    echo "  [SKIP] $label: 源目录不存在"
    return
  fi

  local count
  count=$(find "$src" -type f | wc -l | tr -d ' ')

  if $DRY_RUN; then
    echo "  [DRY-RUN] $label: 将复制 $count 个文件到 $dst"
    return
  fi

  mkdir -p "$dst"
  cp -r "$src"/* "$dst"/ 2>/dev/null || true
  echo "  [OK] $label: 已安装 $count 个文件 -> $dst"
}

if $INSTALL_RULES; then
  copy_dir "$SCRIPT_DIR/rules" "$TARGET/.cursor/rules" "Rules"
fi

if $INSTALL_SKILLS; then
  copy_dir "$SCRIPT_DIR/skills" "$TARGET/.cursor/skills" "Skills"
fi

if $INSTALL_AGENTS; then
  copy_dir "$SCRIPT_DIR/agents" "$TARGET/.cursor/agents" "Agents"
fi

if $INSTALL_COMMANDS; then
  copy_dir "$SCRIPT_DIR/commands" "$TARGET/.cursor/commands" "Commands"
fi

echo ""
echo "安装完成！重启 Cursor 以加载新配置。"
echo ""
echo "使用方式:"
echo "  Rules   -> 自动应用或在 chat 中 @rule-name 引用"
echo "  Skills  -> Agent 自动发现或输入 /skill-name 调用"
echo "  Agents  -> Agent 根据任务自动委派子代理"
echo "  Commands -> 在 chat 输入 / 查看可用命令"
