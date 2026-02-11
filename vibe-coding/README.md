# Vibe Coding

可移植的 Cursor IDE vibe coding 资源集合，涵盖 **Rules / Skills / Agents / Commands** 四大类。

聚焦技术栈：**React + TypeScript / Rust / Go / 架构设计 / 安全 / 云原生**

## 目录结构

```
vibe-coding/
├── rules/                          # AI 行为规则 (.mdc)
│   ├── lang/                       # 语言最佳实践
│   │   ├── go.mdc                  # Go 编码规范
│   │   ├── rust.mdc                # Rust 编码规范
│   │   ├── react.mdc               # React 最佳实践
│   │   └── typescript.mdc          # TypeScript 最佳实践
│   ├── security/                   # 安全规则
│   │   ├── secure-development-principles.mdc  # 通用安全原则 (always apply)
│   │   ├── secure-dev-golang.mdc   # Go 安全开发
│   │   ├── secure-dev-rust.mdc     # Rust 安全开发
│   │   ├── secure-dev-node.mdc     # Node.js 安全开发
│   │   ├── dangerous-flows.mdc     # 危险流识别与检测
│   │   ├── path-traversal-prevention.mdc
│   │   ├── ssrf-prevention.mdc
│   │   ├── xxe-prevention.mdc
│   │   ├── secure-sql-usage.mdc
│   │   └── secure-mcp-usage.mdc
│   └── cloud-native/               # 云原生规则
│       ├── kubernetes.mdc
│       ├── docker.mdc
│       └── terraform.mdc
├── skills/                          # Agent Skills (SKILL.md)
│   ├── # --- 语言 ---
│   ├── golang-pro/                  # Go 专家
│   ├── go-concurrency-patterns/     # Go 并发模式
│   ├── rust-pro/                    # Rust 专家
│   ├── rust-async-patterns/         # Rust 异步模式
│   ├── react-best-practices/        # React 最佳实践
│   ├── react-patterns/              # React 设计模式
│   ├── react-state-management/      # React 状态管理
│   ├── # --- 架构 ---
│   ├── senior-architect/            # 高级架构师
│   ├── architecture-patterns/       # 架构模式
│   ├── architecture-decision-records/  # ADR 决策记录
│   ├── microservices-patterns/      # 微服务模式
│   ├── api-design-principles/       # API 设计原则
│   ├── monorepo-architect/          # Monorepo 架构
│   ├── # --- 安全 ---
│   ├── security-auditor/            # 安全审计
│   ├── api-security-best-practices/ # API 安全
│   ├── auth-implementation-patterns/  # 认证实现模式
│   ├── backend-security-coder/      # 后端安全编码
│   ├── frontend-security-coder/     # 前端安全编码
│   ├── security-scanning-security-sast/  # SAST 安全扫描
│   ├── security-scanning-security-hardening/ # 安全加固
│   ├── # --- 云原生 ---
│   ├── kubernetes-architect/        # K8s 架构师
│   ├── k8s-manifest-generator/      # K8s 清单生成
│   ├── k8s-security-policies/       # K8s 安全策略
│   ├── helm-chart-scaffolding/      # Helm Chart 脚手架
│   ├── docker-expert/               # Docker 专家
│   ├── terraform-skill/             # Terraform 基础
│   ├── terraform-specialist/        # Terraform 专家
│   ├── observability-engineer/      # 可观测性工程师
│   ├── # --- 工程 ---
│   ├── code-review-excellence/      # Code Review 卓越实践
│   ├── skill-creator/               # Skill 创建指南
│   └── artifacts-builder/           # 制品构建器
├── agents/                          # 自定义 Subagents
│   ├── backend-architect.md         # 后端架构师
│   ├── cloud-architect.md           # 云架构师
│   ├── code-reviewer.md             # 代码审查员
│   ├── security-engineer.md         # 安全工程师
│   ├── deployment-engineer.md       # 部署工程师
│   ├── front-end-dev.md             # 前端开发者
│   └── debugger.md                  # 调试专家
├── commands/                        # 自定义斜杠命令
│   ├── # --- 代码质量 ---
│   ├── code-review.md               # /code-review
│   ├── refactor-code.md             # /refactor-code
│   ├── optimize-performance.md      # /optimize-performance
│   ├── lint-fix.md                  # /lint-fix
│   ├── add-error-handling.md        # /add-error-handling
│   ├── deslop.md                    # /deslop (清理 AI 生成代码)
│   ├── fix-compile-errors.md        # /fix-compile-errors
│   ├── # --- 安全 ---
│   ├── security-audit.md            # /security-audit
│   ├── security-review.md           # /security-review
│   ├── accessibility-audit.md       # /accessibility-audit
│   ├── # --- 测试 ---
│   ├── run-all-tests-and-fix.md     # /run-all-tests-and-fix
│   ├── write-unit-tests.md          # /write-unit-tests
│   ├── debug-issue.md               # /debug-issue
│   ├── # --- Git & PR ---
│   ├── create-pr.md                 # /create-pr
│   ├── generate-pr-description.md   # /generate-pr-description
│   ├── address-github-pr-comments.md  # /address-github-pr-comments
│   ├── light-review-existing-diffs.md # /light-review-existing-diffs
│   ├── git-commit.md                # /git-commit
│   ├── # --- 文档 & 基建 ---
│   ├── add-documentation.md         # /add-documentation
│   ├── generate-api-docs.md         # /generate-api-docs
│   ├── diagrams.md                  # /diagrams
│   ├── database-migration.md        # /database-migration
│   ├── docker-logs.md               # /docker-logs
│   ├── setup-new-feature.md         # /setup-new-feature
│   ├── onboard-new-developer.md     # /onboard-new-developer
│   └── overview.md                  # /overview
├── install.sh                       # 安装脚本
└── README.md
```

## 快速开始

### 安装到目标项目

```bash
# 安装全部到目标项目
./install.sh ~/projects/my-app

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
# 按需复制
cp -r rules/    ~/projects/my-app/.cursor/rules/
cp -r skills/   ~/projects/my-app/.cursor/skills/
cp -r agents/   ~/projects/my-app/.cursor/agents/
cp -r commands/  ~/projects/my-app/.cursor/commands/
```

## 使用方式

| 类别 | 目标路径 | 触发方式 |
|------|---------|---------|
| **Rules** | `.cursor/rules/` | 自动应用 / `@rule-name` 手动引用 |
| **Skills** | `.cursor/skills/<name>/SKILL.md` | Agent 自动发现 / `/skill-name` 手动调用 |
| **Agents** | `.cursor/agents/` | Agent 根据任务自动委派 |
| **Commands** | `.cursor/commands/` | 输入 `/command-name` 触发 |

## 资源来源

所有资源均来自全网高呼声开源项目，非 AI 编造：

| 来源 | Stars | 提供内容 | 链接 |
|------|-------|---------|------|
| **sanjeed5/awesome-cursor-rules-mdc** | 3,200+ | Rules: 语言最佳实践 + 云原生 | [GitHub](https://github.com/sanjeed5/awesome-cursor-rules-mdc) |
| **matank001/cursor-security-rules** | 350+ | Rules: 安全开发全系列 | [GitHub](https://github.com/matank001/cursor-security-rules) |
| **sickn33/antigravity-awesome-skills** | 8,100+ | Skills: 715 个领域技能 (精选 31 个) | [GitHub](https://github.com/sickn33/antigravity-awesome-skills) |
| **hamzafer/cursor-commands** | 550+ | Commands: 26 个斜杠命令 | [GitHub](https://github.com/hamzafer/cursor-commands) |
| **alexhatzo/agent-squared** | - | Agents: 多角色子代理 | [GitHub](https://github.com/alexhatzo/agent-squared) |
| **chrisboden/cursor-skills** | - | Skills: skill-creator 等 | [GitHub](https://github.com/chrisboden/cursor-skills) |

### 官方文档

- [Cursor Rules](https://cursor.com/docs/context/rules)
- [Cursor Skills](https://cursor.com/docs/context/skills) / [Agent Skills Spec](https://agentskills.io/specification)
- [Cursor Subagents](https://cursor.com/docs/context/subagents)
- [Cursor Commands](https://cursor.com/docs/context/commands)

## 统计

- **Rules**: 17 个 `.mdc` 文件 (4 语言 + 10 安全 + 3 云原生)
- **Skills**: 31 个 SKILL.md (7 语言 + 8 架构 + 7 安全 + 7 云原生 + 2 工程)
- **Agents**: 7 个子代理定义
- **Commands**: 26 个斜杠命令
