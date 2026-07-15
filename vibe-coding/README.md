# Vibe coding（已迁出）

原 AI 开发资源已迁入 GitHub 私有库 **`heliannuuthus/workspace`**，统一使用根目录 `AGENTS.md` 与 `.agents/skills/`，供 Codex、Cursor 等 Agent 工具读取，并与 Git 子模块和多根工作区配合使用。

## 新位置

克隆：`git clone git@github.com:heliannuuthus/workspace.git`（本地目录可任意命名，例如 `workspace`）。

## 使用方式

1. 克隆后初始化 submodule，具体命令见该仓库根目录 `README.md`。
2. 从 workspace 根目录启动 Codex 或打开 Cursor，使其读取根 `AGENTS.md` 和 `.agents/skills/`。
3. 单独打开业务子项目时，使用该项目自己的 `AGENTS.md`。

blacksmith 内不再保留副本，避免与私有库双份维护。
