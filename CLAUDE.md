# Blacksmith

Blacksmith 是个人工具箱仓库，包含 Python uv workspace、Rust CLI、浏览器脚本和迁移占位文档。

## 目录结构

| 路径 | 说明 |
|------|------|
| `apps/` | Python 应用，如 HowToCook 数据抓取工具 |
| `packages/` | Python 共享库，如 GitHub API client |
| `clis/` | Rust CLI workspace，如 `hostctl` |
| `scripts/` | 浏览器/用户脚本，如 TOTP 自动填入 |
| `vibe-coding/` | 旧资源迁移占位，实际 AI 资源在 workspace 根 `.claude/` |

## 常用命令

```bash
# Python workspace
uv sync --all-extras
uv run ruff format apps packages
uv run ruff check apps packages
uv run mypy apps packages
uv run pytest

# Rust CLI
cd clis
cargo fmt
cargo clippy --all-targets --all-features
cargo test
```

## 开发规则

- Python 项目使用 uv 管理，不混用 pipenv/poetry。
- Python 代码目标版本为 3.12，Ruff 行宽 100。
- Rust CLI workspace edition 为 2024，禁止 `unsafe_code`。
- GitHub token 检测顺序遵循 README：显式 token、`GH_TOKEN`/`GITHUB_TOKEN`、可选 `gh auth token`。
- 不要把 vibe-coding 资源重新复制回 `blacksmith/vibe-coding/`；根 `.claude/` 是单一来源。

## 验证 Checklist

1. Python 改动：`uv run ruff check apps packages`，必要时 `uv run pytest`。
2. Rust 改动：`cd clis && cargo test`，必要时 `cargo clippy`。
3. 脚本改动：检查 README/注释中的安装和使用路径。
