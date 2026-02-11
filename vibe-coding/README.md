# Vibe Coding

可移植的 Cursor IDE vibe coding 资源集合。

聚焦技术栈：**React + TypeScript / Rust / Go / 架构设计 / 安全 / 云原生**

## 安装

### 在线安装（推荐）

```bash
curl -sSL https://raw.githubusercontent.com/heliannuuthus/blacksmith/master/vibe-coding/install.sh | bash -s -- ~
```

### 本地安装

```bash
# 安装全部到全局
./install.sh ~

# 安装到指定项目
./install.sh ~/projects/my-app

# 选择性安装
./install.sh --rules-only ~/projects/my-app
./install.sh --skills-only ~
./install.sh --agents-only ~

# 预览（不实际操作）
./install.sh --dry-run ~/projects/my-app
```

## 三层互补模型

| 层级 | 类型 | 数量 | 触发方式 | 职责 | 详情 |
|------|------|------|---------|------|------|
| **约束层** | Rules | 12 | 自动 (`alwaysApply` / `globs`) | 安全底线、编码规范 | [rules/README.md](rules/README.md) |
| **知识层** | Skills | 57 | Agent 自动发现 / `@skill-name` | 领域知识、工作流程 | [skills/README.md](skills/README.md) |
| **执行层** | Agents | 7 | Agent 自动委派 | 独立专家角色 | [agents/README.md](agents/README.md) |

### 设计决策

- **Rules** 仅保留必须始终遵守的约束 — `alwaysApply: true` 安全底线 + `globs` 精确匹配语言规范
- **Skills** 承载可被调用的知识和流程 — 含原 commands（`disable-model-invocation: true`）和从 rules 迁移的审计/云原生内容
- **Agents** 提供独立的专家视角 — 代码审查、安全评估、架构设计等需独立上下文的任务

## 资源来源

| 来源 | Stars | 提供内容 |
|------|-------|---------|
| [sanjeed5/awesome-cursor-rules-mdc](https://github.com/sanjeed5/awesome-cursor-rules-mdc) | 3,200+ | Rules: 语言最佳实践 |
| [matank001/cursor-security-rules](https://github.com/matank001/cursor-security-rules) | 350+ | Rules: 安全开发全系列 |
| [sickn33/antigravity-awesome-skills](https://github.com/sickn33/antigravity-awesome-skills) | 8,100+ | Skills: 领域技能 |
| [hamzafer/cursor-commands](https://github.com/hamzafer/cursor-commands) | 550+ | Skills: 工作流 |
| [alexhatzo/agent-squared](https://github.com/alexhatzo/agent-squared) | - | Agents: 多角色子代理 |
| [chrisboden/cursor-skills](https://github.com/chrisboden/cursor-skills) | - | Skills: skill-creator 等 |

## 官方文档

- [Cursor Rules](https://cursor.com/docs/context/rules) / [Cursor Skills](https://cursor.com/docs/context/skills) / [Cursor Subagents](https://cursor.com/docs/context/subagents)
- [Agent Skills Spec](https://agentskills.io/specification)
