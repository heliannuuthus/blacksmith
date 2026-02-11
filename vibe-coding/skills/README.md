# Skills

Cursor Agent Skills，每个 skill 是一个独立目录，包含 `SKILL.md`（YAML frontmatter + Markdown 正文）。

> Skills 承载**按需激活的领域知识和工作流程**，Agent 根据任务描述自动发现匹配的 skill，也可手动 `@skill-name` 引用。

## 触发方式

- **自动发现** — Agent 根据 `description` 字段匹配当前任务
- **手动引用** — 在 chat 中 `@skill-name`
- `disable-model-invocation: true` — 标记为仅手动触发（工作流类 skill）

## 语言

| Skill | 说明 |
|-------|------|
| [golang-pro](golang-pro/) | Go 1.21+ 专家，现代模式与高级并发 |
| [go-concurrency-patterns](go-concurrency-patterns/) | Go 并发模式（goroutine、channel、sync、context） |
| [rust-pro](rust-pro/) | Rust 1.75+ 专家，async 模式与高级类型系统 |
| [rust-async-patterns](rust-async-patterns/) | Rust 异步编程（Tokio、async trait、并发模式） |
| [react-best-practices](react-best-practices/) | React/Next.js 性能优化（Vercel Engineering） |
| [react-patterns](react-patterns/) | React 现代模式（Hooks、组合、TypeScript） |
| [react-state-management](react-state-management/) | React 状态管理（Redux Toolkit、Zustand、Jotai、React Query） |

## 架构

| Skill | 说明 |
|-------|------|
| [senior-architect](senior-architect/) | 全栈架构设计（系统设计、技术决策、架构图生成） |
| [architecture-patterns](architecture-patterns/) | 架构模式（Clean Architecture、六边形、DDD） |
| [architecture-decision-records](architecture-decision-records/) | ADR 决策记录编写与维护 |
| [microservices-patterns](microservices-patterns/) | 微服务架构（服务边界、事件驱动、弹性模式） |
| [api-design-principles](api-design-principles/) | REST/GraphQL API 设计原则 |
| [monorepo-architect](monorepo-architect/) | Monorepo 架构（Nx、Turborepo、Bazel） |

## 安全

| Skill | 说明 |
|-------|------|
| [security-auditor](security-auditor/) | 综合安全审计（DevSecOps） |
| [api-security-best-practices](api-security-best-practices/) | API 安全模式（认证、授权、输入校验、限流） |
| [auth-implementation-patterns](auth-implementation-patterns/) | 认证授权实现（JWT、OAuth2、Session、RBAC） |
| [backend-security-coder](backend-security-coder/) | 后端安全编码实践 |
| [frontend-security-coder](frontend-security-coder/) | 前端安全编码实践（XSS 防护等） |
| [security-scanning-security-sast](security-scanning-security-sast/) | SAST 静态安全扫描 |
| [security-scanning-security-hardening](security-scanning-security-hardening/) | 多层安全加固 |
| [dangerous-flows](dangerous-flows/) | 危险数据流分析（原 rule 迁移） |
| [xxe-prevention](xxe-prevention/) | XXE 攻击防护（原 rule 迁移） |

## 云原生

| Skill | 说明 |
|-------|------|
| [kubernetes-architect](kubernetes-architect/) | K8s 架构设计 |
| [kubernetes-best-practices](kubernetes-best-practices/) | K8s 开发部署最佳实践（原 rule 迁移） |
| [k8s-manifest-generator](k8s-manifest-generator/) | K8s YAML 清单生成 |
| [k8s-security-policies](k8s-security-policies/) | K8s 安全策略（NetworkPolicy、RBAC、PodSecurity） |
| [helm-chart-scaffolding](helm-chart-scaffolding/) | Helm Chart 模板与打包 |
| [docker-expert](docker-expert/) | Docker 专家（多阶段构建、镜像优化、安全加固） |
| [docker-best-practices](docker-best-practices/) | Docker 容器化最佳实践（原 rule 迁移） |
| [terraform-skill](terraform-skill/) | Terraform IaC 基础 |
| [terraform-specialist](terraform-specialist/) | Terraform/OpenTofu 高级 IaC |
| [terraform-best-practices](terraform-best-practices/) | Terraform 最佳实践（原 rule 迁移） |
| [observability-engineer](observability-engineer/) | 可观测性（监控、日志、链路追踪） |

## 工程实践

| Skill | 说明 |
|-------|------|
| [code-review-excellence](code-review-excellence/) | Code Review 最佳实践 |
| [skill-creator](skill-creator/) | Skill 创建指南 |
| [artifacts-builder](artifacts-builder/) | Claude Artifacts 构建器 |

## 工作流

> 原 commands 迁移而来，标记 `disable-model-invocation: true`，需手动 `@skill-name` 触发。

| Skill | 说明 |
|-------|------|
| [create-pr](create-pr/) | 创建 GitHub PR |
| [generate-pr-description](generate-pr-description/) | 生成 PR 描述 |
| [address-github-pr-comments](address-github-pr-comments/) | 处理 PR 评论 |
| [light-review-existing-diffs](light-review-existing-diffs/) | 轻量 Diff 审查 |
| [git-commit](git-commit/) | Git 提交（Conventional Commits） |
| [write-unit-tests](write-unit-tests/) | 编写单元测试 |
| [run-all-tests-and-fix](run-all-tests-and-fix/) | 跑测试并修复 |
| [refactor-code](refactor-code/) | 代码重构 |
| [optimize-performance](optimize-performance/) | 性能优化分析 |
| [lint-fix](lint-fix/) | Lint 修复 |
| [fix-compile-errors](fix-compile-errors/) | 修复编译错误 |
| [add-error-handling](add-error-handling/) | 添加错误处理 |
| [deslop](deslop/) | 清理 AI 生成代码 |
| [add-documentation](add-documentation/) | 添加文档 |
| [generate-api-docs](generate-api-docs/) | 生成 API 文档 |
| [diagrams](diagrams/) | 生成架构图 |
| [database-migration](database-migration/) | 数据库迁移脚本 |
| [setup-new-feature](setup-new-feature/) | 新功能脚手架 |
| [onboard-new-developer](onboard-new-developer/) | 新人入职引导 |
| [overview](overview/) | 项目概览 |
| [accessibility-audit](accessibility-audit/) | 无障碍审计（WCAG） |

## 来源

| 仓库 | Stars | 内容 |
|------|-------|------|
| [sickn33/antigravity-awesome-skills](https://github.com/sickn33/antigravity-awesome-skills) | 8,100+ | 领域技能（精选） |
| [hamzafer/cursor-commands](https://github.com/hamzafer/cursor-commands) | 550+ | 工作流（原 commands 迁移） |
| [chrisboden/cursor-skills](https://github.com/chrisboden/cursor-skills) | - | skill-creator 等 |

## 官方文档

- [Cursor Skills](https://cursor.com/docs/context/skills)
- [Agent Skills Spec](https://agentskills.io/specification)
