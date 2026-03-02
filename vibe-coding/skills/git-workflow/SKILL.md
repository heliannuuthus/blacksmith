---
name: git-workflow
description: "Git 分支、提交和同步工作流规范。在创建分支、提交代码、推送代码、同步主分支时使用，也可通过 gs/gsync 触发主分支同步。"
---

# Git 工作流规范

## 分支命名

格式：`type/short-title`，全部小写，用连字符分隔单词。

| 类型 | 用途 | Gitmoji |
|------|------|---------|
| feat | 新功能 | :sparkles: |
| fix | 修复 Bug | :bug: |
| hotfix | 紧急修复 | :ambulance: |
| refactor | 代码重构 | :recycle: |
| style | UI/样式调整 | :lipstick: |
| docs | 文档变更 | :memo: |
| test | 测试相关 | :white_check_mark: |
| chore | 构建/配置/维护 | :wrench: |
| perf | 性能优化 | :zap: |
| ci | CI/CD 流水线 | :construction_worker: |
| deps | 依赖升降级 | :arrow_up: |
| security | 安全修复 | :lock: |
| arch | 架构变更 | :building_construction: |
| wip | 进行中的工作 | :construction: |

示例：
- `feat/user-authentication`
- `fix/login-redirect-loop`
- `refactor/handler-layout`
- `docs/api-design`

## 提交规则

### 提交前必须 Diff

提交前始终运行以下命令来了解完整改动：

```bash
git diff origin/main...HEAD    # 当前分支相对 main 的全部改动
git diff --cached              # 已暂存的改动
git diff                       # 未暂存的改动
```

根据 diff 结果撰写 commit message，确保描述准确反映实际变更。

### 提交信息格式

```
type(scope): 简要描述

可选的详细说明
```

`type` 与分支类型一致，`scope` 可选，小写。使用祈使语气。

### 禁止事项

1. **禁止 force push 到 main/master** — 没有例外
2. **禁止直接向 main/master 提交代码** — 必须通过 PR 以 squash merge 方式合并
3. **禁止在非 rebase 场景使用 force push** — 如 rebase 后需要强推，使用 `--force-with-lease`
4. **禁止跳过 hooks**（`--no-verify`）— 除非用户明确要求

### 允许 Force Push 的场景

仅当以下条件全部满足时：
- 目标分支不是 main/master
- 正在将自己的功能分支 rebase 到最新 main
- 使用 `--force-with-lease`（而非 `--force`）

## 创建 PR

创建 PR 前必须执行：

```bash
git fetch origin main
git diff origin/main...HEAD          # 查看分支完整改动
git log origin/main..HEAD --oneline  # 查看所有 commit
```

根据 diff 和 commit 历史生成 PR 描述：
- **标题**：`type(scope): 简要描述`，与分支类型一致
- **正文**：总结全部改动（不是只看最后一个 commit），包含 Summary 和 Test Plan
- 推送后使用 `gh pr create` 创建，目标分支为 main

## 同步规则

当用户输入 **gs**、**gsync** 或要求同步/准备开发下一个功能时：

运行脚本：`bash <skill-dir>/gsync.sh`

脚本行为：
- **在 Git 项目内运行**：仅同步当前项目到最新主分支
- **在非 Git 目录运行**：遍历一级子目录中所有 Git 项目，逐个同步
- 存在未提交改动的项目会被跳过，最后统一提示用户

同步完成后，如果用户提供了新分支名，直接创建：`git checkout -b type/title`；否则询问用户下一个分支的命名。
