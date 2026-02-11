# Agents

Cursor 自定义 Subagents，每个 agent 是一个 `.md` 文件（YAML frontmatter + Markdown 指令）。

> Agents 提供**独立的专家角色**，拥有隔离的上下文，由主 Agent 根据任务自动委派或用户手动选择。

## 触发方式

- **自动委派** — 主 Agent 根据 `description` 匹配当前任务，自动启动子代理
- **手动选择** — 在 Agent 面板中选择特定子代理执行任务

## Agent 列表

| Agent | 文件 | 职责 |
|-------|------|------|
| **backend-architect** | [backend-architect.md](backend-architect.md) | 后端系统架构与 API 设计（RESTful、微服务边界、数据库 Schema、可扩展性） |
| **cloud-architect** | [cloud-architect.md](cloud-architect.md) | 云基础设施设计与优化（AWS/Azure/GCP、Terraform IaC、成本优化、多区域部署） |
| **code-reviewer** | [code-reviewer.md](code-reviewer.md) | 代码质量审查（安全、可维护性、最佳实践） |
| **security-engineer** | [security-engineer.md](security-engineer.md) | API 安全审计（认证漏洞、授权缺陷、注入攻击、合规验证） |
| **deployment-engineer** | [deployment-engineer.md](deployment-engineer.md) | CI/CD 与部署自动化（Pipeline、Docker、K8s、GitHub Actions） |
| **frontend-developer** | [front-end-dev.md](front-end-dev.md) | 前端开发（React、状态管理、性能优化、无障碍、响应式设计） |
| **debugger** | [debugger.md](debugger.md) | 系统调试（错误分析、测试失败、堆栈追踪、异常行为排查） |

## Frontmatter 规范

```yaml
---
name: agent-name           # 代理标识
description: ...           # 职责描述（主 Agent 据此匹配委派）
tools: Read, Write, Edit, Bash, Grep  # 可用工具列表
---
```

> 注意：`model` 字段不是 Cursor 原生支持的，已移除。子代理继承主 Agent 的模型配置。

## 来源

| 仓库 | 内容 |
|------|------|
| [alexhatzo/agent-squared](https://github.com/alexhatzo/agent-squared) | 多角色子代理定义 |

## 官方文档

- [Cursor Subagents](https://cursor.com/docs/context/subagents)
