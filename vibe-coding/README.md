# Vibe Coding

可移植的 Cursor IDE vibe coding 资源集合，涵盖 **Rules / Skills / Agents** 三大类。

聚焦技术栈：**React + TypeScript / Rust / Go / 架构设计 / 安全 / 云原生**

> **设计原则**：Rules 负责声明式约束（alwaysApply 安全底线 + globs 语言规范），Skills 负责按需的领域知识和流程，Agents 负责独立的专家角色。三者互补，不重叠。

## 目录结构

```
vibe-coding/
├── rules/                          # AI 行为规则 (.mdc) - 12 个
│   ├── lang/                       # 语言编码规范 (globs 匹配文件后缀)
│   │   ├── go.mdc                  # **/*.go
│   │   ├── rust.mdc                # **/*.rs
│   │   ├── react.mdc               # **/*.{jsx,tsx}
│   │   └── typescript.mdc          # **/*.{ts,tsx}
│   └── security/                   # 安全底线规则
│       ├── secure-development-principles.mdc  # alwaysApply
│       ├── path-traversal-prevention.mdc      # alwaysApply
│       ├── ssrf-prevention.mdc                # alwaysApply
│       ├── secure-sql-usage.mdc               # alwaysApply
│       ├── secure-mcp-usage.mdc               # alwaysApply
│       ├── secure-dev-golang.mdc              # **/*.go
│       ├── secure-dev-rust.mdc                # **/*.rs
│       └── secure-dev-node.mdc                # **/*.{tsx,ts,js}
├── skills/                         # Agent Skills (SKILL.md) - 57 个
│   ├── golang-pro/                 # Go 专家
│   ├── go-concurrency-patterns/    # Go 并发模式
│   ├── rust-pro/                   # Rust 专家
│   ├── rust-async-patterns/        # Rust 异步模式
│   ├── react-best-practices/       # React 最佳实践
│   ├── react-patterns/             # React 设计模式
│   ├── react-state-management/     # React 状态管理
│   ├── senior-architect/           # 高级架构师
│   ├── architecture-patterns/      # 架构模式
│   ├── architecture-decision-records/ # ADR 决策记录
│   ├── microservices-patterns/     # 微服务模式
│   ├── api-design-principles/      # API 设计原则
│   ├── monorepo-architect/         # Monorepo 架构
│   ├── security-auditor/           # 安全审计
│   ├── dangerous-flows/            # 危险数据流分析
│   ├── xxe-prevention/             # XXE 防护
│   ├── kubernetes-best-practices/  # K8s 最佳实践
│   ├── docker-best-practices/      # Docker 最佳实践
│   ├── terraform-best-practices/   # Terraform 最佳实践
│   ├── deslop/                     # 清理 AI 生成代码
│   ├── create-pr/                  # 创建 PR
│   ├── write-unit-tests/           # 编写单元测试
│   └── ...                         # 共 57 个, 详见 skills/ 目录
├── agents/                         # 自定义 Subagents - 7 个
│   ├── backend-architect.md        # 后端架构师
│   ├── cloud-architect.md          # 云架构师
│   ├── code-reviewer.md            # 代码审查员
│   ├── security-engineer.md        # 安全工程师
│   ├── deployment-engineer.md      # 部署工程师
│   ├── front-end-dev.md            # 前端开发者
│   └── debugger.md                 # 调试专家
├── install.sh                      # 安装脚本
└── README.md
```

## 快速开始

### 安装到目标项目

```bash
# 安装全部到目标项目
./install.sh ~/projects/my-app

# 安装到全局 (所有项目生效)
./install.sh ~

# 仅安装 rules
./install.sh --rules-only ~/projects/my-app

# 仅安装 skills
./install.sh --skills-only ~/projects/my-app

# 预览 (不实际操作)
./install.sh --dry-run ~/projects/my-app
```

### 手动安装

直接将对应目录复制到目标项目的 `.cursor/` 下：

```bash
cp -r rules/    ~/projects/my-app/.cursor/rules/
cp -r skills/   ~/projects/my-app/.cursor/skills/
cp -r agents/   ~/projects/my-app/.cursor/agents/
```

## 架构设计

### 三层互补模型

| 层级 | 类型 | 触发方式 | 职责 |
|------|------|---------|------|
| **约束层** | Rules | 自动 (`alwaysApply`) / 文件匹配 (`globs`) | 安全底线、编码规范 |
| **知识层** | Skills | Agent 自动发现 / `@skill-name` 手动引用 | 领域知识、工作流程 |
| **执行层** | Agents | Agent 根据任务自动委派 | 独立专家角色 |

### 设计决策

- **Rules 仅保留必须始终遵守的约束**：`alwaysApply: true` 安全底线 + `globs` 精确匹配语言规范
- **Skills 承载可被调用的知识和流程**：含原 commands（标记 `disable-model-invocation: true`）和从 rules 迁移的审计/云原生内容
- **Agents 提供独立的专家视角**：代码审查、安全评估、架构设计等需独立上下文的任务

## 资源来源

所有资源均来自全网高呼声开源项目：

| 来源 | Stars | 提供内容 | 链接 |
|------|-------|---------|------|
| sanjeed5/awesome-cursor-rules-mdc | 3,200+ | Rules: 语言最佳实践 | [GitHub](https://github.com/sanjeed5/awesome-cursor-rules-mdc) |
| matank001/cursor-security-rules | 350+ | Rules: 安全开发全系列 | [GitHub](https://github.com/matank001/cursor-security-rules) |
| sickn33/antigravity-awesome-skills | 8,100+ | Skills: 领域技能 (精选) | [GitHub](https://github.com/sickn33/antigravity-awesome-skills) |
| hamzafer/cursor-commands | 550+ | Skills: 工作流 (原 commands) | [GitHub](https://github.com/hamzafer/cursor-commands) |
| alexhatzo/agent-squared | - | Agents: 多角色子代理 | [GitHub](https://github.com/alexhatzo/agent-squared) |
| chrisboden/cursor-skills | - | Skills: skill-creator 等 | [GitHub](https://github.com/chrisboden/cursor-skills) |

### 官方文档

- [Cursor Rules](https://cursor.com/docs/context/rules)
- [Cursor Skills](https://cursor.com/docs/context/skills) / [Agent Skills Spec](https://agentskills.io/specification)
- [Cursor Subagents](https://cursor.com/docs/context/subagents)

## 统计

- **Rules**: 12 个 `.mdc` 文件 (4 语言规范 + 8 安全底线)
- **Skills**: 57 个 SKILL.md (7 语言 + 6 架构 + 9 安全 + 11 云原生 + 3 工程 + 21 工作流)
- **Agents**: 7 个子代理定义
