# Rules

Cursor AI 行为约束规则，采用 `.mdc` 格式（Markdown + YAML frontmatter）。

> Rules 只保留**必须始终遵守**的内容：安全底线（`alwaysApply: true`）和语言编码规范（`globs` 精确匹配）。

## 触发方式

- `alwaysApply: true` — 每次对话自动注入，无需手动引用
- `globs: **/*.go` — 当操作匹配文件时自动激活
- 手动引用 — 在 chat 中 `@rule-name`

## 语言规范

| 文件 | globs | 说明 |
|------|-------|------|
| [go.mdc](lang/go.mdc) | `**/*.go` | Go 编码规范，遵循 Google Style |
| [rust.mdc](lang/rust.mdc) | `**/*.rs` | Rust 惯用模式与最佳实践 |
| [react.mdc](lang/react.mdc) | `**/*.{jsx,tsx}` | React 现代最佳实践 + TypeScript |
| [typescript.mdc](lang/typescript.mdc) | `**/*.{ts,tsx}` | TypeScript 严格模式与类型安全 |

## 安全底线

| 文件 | 触发 | 说明 |
|------|------|------|
| [secure-development-principles.mdc](security/secure-development-principles.mdc) | `alwaysApply` | 通用安全开发原则（输入校验、密钥保护、安全协议等） |
| [path-traversal-prevention.mdc](security/path-traversal-prevention.mdc) | `alwaysApply` | 路径遍历攻击防护 |
| [ssrf-prevention.mdc](security/ssrf-prevention.mdc) | `alwaysApply` | SSRF 攻击防护 |
| [secure-sql-usage.mdc](security/secure-sql-usage.mdc) | `alwaysApply` | SQL 注入防护与安全查询 |
| [secure-mcp-usage.mdc](security/secure-mcp-usage.mdc) | `alwaysApply` | MCP 工具安全使用规范 |
| [secure-dev-golang.mdc](security/secure-dev-golang.mdc) | `**/*.go` | Go 安全编码实践 |
| [secure-dev-rust.mdc](security/secure-dev-rust.mdc) | `**/*.rs` | Rust 安全编码实践 |
| [secure-dev-node.mdc](security/secure-dev-node.mdc) | `**/*.{tsx,ts,js}` | Node.js/TypeScript 安全编码实践 |

## 来源

| 仓库 | Stars | 内容 |
|------|-------|------|
| [sanjeed5/awesome-cursor-rules-mdc](https://github.com/sanjeed5/awesome-cursor-rules-mdc) | 3,200+ | 语言最佳实践 |
| [matank001/cursor-security-rules](https://github.com/matank001/cursor-security-rules) | 350+ | 安全开发全系列 |

## 官方文档

- [Cursor Rules](https://cursor.com/docs/context/rules)
