# Vibe coding（已迁出）

原 **rules / skills / agents / install.sh** 已全部迁入独立私有仓库 **`heliannuuthus-ai-dev-workspace`** 的 **`.claude/`** 目录，用于 Claude 与 Cursor 双 IDE，并与 Git 子模块 + 多根工作区配合使用。

## 新位置

独立仓库目录名 **`heliannuuthus-ai-dev-workspace`**（与 `blacksmith` 同级或任意路径均可）；已 `git init` 并完成首提交，推送到你的 GitHub 私有库即可。

## 使用方式

1. 将该目录初始化为 Git 仓库并推送到 **GitHub 私有库**（见该仓库根目录 `README.md`）。
2. **Claude**：直接打开该仓库（或把它嵌进你的工作区），使用 `.claude/`。
3. **Cursor**：在该仓库或目标项目根执行  
   `bash /path/to/heliannuuthus-ai-dev-workspace/.claude/install.sh .`  
   生成 `.cursor/`。

blacksmith 内不再保留副本，避免与私有库双份维护。
