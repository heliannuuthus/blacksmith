# skill-downloader

下载 skill.md 文件到本地的 `~/.cluade/skills/` 目录的工具。

## 功能

- 支持多种 GitHub URL 格式
- 自动解析 markdown frontmatter 获取 skill 名称
- 自动创建同名目录并保存 skill.md
- 支持使用 gh 包的工具和 GitHub API

## 支持的 URL 格式

- `https://github.com/owner/repo/blob/branch/path/to/skill.md`
- `https://github.com/owner/repo.git`
- `https://github.com/owner/repo`
- `git@github.com:owner/repo.git`
- `owner/repo`
- `owner/repo/path/to/skill.md`
- `owner/repo@branch`
- `owner/repo@branch/path/to/skill.md`

## 前置要求

工具默认使用 `gh cli` 进行认证。首次使用前需要：

1. 安装 gh cli（如果未安装）：
   ```bash
   # Ubuntu/Debian
   sudo apt-get install gh
   
   # 或访问 https://cli.github.com/
   ```

2. 认证 gh cli：
   ```bash
   gh auth login
   ```

## 使用方法

```bash
# 下载 skill（默认使用 gh cli 凭证）
uv run skill-downloader download owner/repo

# 指定分支
uv run skill-downloader download owner/repo@main

# 指定路径
uv run skill-downloader download owner/repo/path/to/SKILL.md

# 使用完整 URL
uv run skill-downloader download https://github.com/owner/repo/blob/main/SKILL.md

# 从 raw URL 下载
uv run skill-downloader download-raw https://raw.githubusercontent.com/owner/repo/branch/SKILL.md

# 列出已安装的 skills
uv run skill-downloader list

# 使用 GitHub token（覆盖 gh cli）
export GITHUB_TOKEN=your_token
uv run skill-downloader download owner/repo
# 或
uv run skill-downloader --token your_token download owner/repo
```

## 配置

- `--token`: GitHub token（或通过 `GITHUB_TOKEN` 环境变量，覆盖 gh cli）
- `--skills-dir`: 指定 skills 目录（默认: `~/.cluade/skills`）
- `--retries`: 重试次数（默认: 3）
- `-v, --verbose`: 详细输出（`-v` INFO, `-vv` DEBUG）

**注意**：如果没有提供 `--token`，工具会默认使用 `gh cli` 凭证。如果 `gh cli` 未安装或未认证，工具会提示错误并退出。

## Skill 文件格式

skill.md 文件应包含 YAML frontmatter，其中 `name` 字段用于确定保存的目录名：

```markdown
---
name: my-skill
description: Description of the skill
---

# Skill Content

...
```

如果没有 `name` 字段，将使用仓库名或路径推断。
