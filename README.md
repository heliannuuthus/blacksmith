# Blacksmith

Python monorepo，使用 [uv](https://docs.astral.sh/uv/) 管理。

## 目录结构

```
blacksmith/
├── apps/                      # Python 应用
│   └── howtocook/            # HowToCook 数据抓取工具
├── clis/                      # Rust CLI 工具
│   └── hostctl/              # Hosts 文件管理工具（TUI）
├── packages/                  # 共享 Python 库
│   └── gh/                   # GitHub API 客户端
├── scripts/                   # 浏览器 / 用户脚本
│   └── totp-auto-paste/      # TOTP 验证码自动填入（Tampermonkey）
├── vibe-coding/              # Cursor IDE vibe coding 资源集合 → [详情](vibe-coding/README.md)
│   ├── rules/                # AI 行为规则 → [详情](vibe-coding/rules/README.md)
│   ├── skills/               # Agent 领域技能 → [详情](vibe-coding/skills/README.md)
│   ├── agents/               # 自定义子代理 → [详情](vibe-coding/agents/README.md)
│   └── install.sh            # 一键安装
└── pyproject.toml            # Python workspace 配置
```

## 快速开始

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装依赖（uv 在 .venv 内）
.venv/bin/uv sync

# 运行 CLI 工具
uv run howtocook --help
uv run howtocook categories
uv run howtocook dishes -o dishes.json

# 启用日志（调试模式）
uv run howtocook -v categories      # INFO 级别
uv run howtocook -vv categories     # DEBUG 级别
```

## 项目

### howtocook

抓取 [HowToCook](https://github.com/Anduin2017/HowToCook) 项目数据。

```bash
# 查看类别
uv run howtocook categories

# 抓取菜品
uv run howtocook dishes -o dishes.json

# 抓取技巧
uv run howtocook tips -o tips.json

# 结构化解析
uv run howtocook dishes -p -o dishes.json

# 使用 GitHub token（提高速率限制）
export GITHUB_TOKEN=your_token
uv run howtocook dishes -o dishes.json
```

作为库使用：

```python
from howtocook import HowToCookScraper

scraper = HowToCookScraper(token="optional")
dishes = scraper.fetch_all_dishes()
tips = scraper.fetch_all_tips()
```

### gh

GitHub API 客户端。

```python
from gh import GitHubClient

# 自动检测环境变量 GH_TOKEN / GITHUB_TOKEN
client = GitHubClient()

# 手动指定 token
client = GitHubClient(token="your_token")

# 使用 gh cli 凭证（需用户明确授权）
client = GitHubClient(use_gh_cli=True)

contents = client.get_contents("owner", "repo", "path")
file = client.get_file_content("owner", "repo", "file.md")
```

Token 检测顺序：
1. 手动传入的 token
2. 环境变量 `GH_TOKEN` / `GITHUB_TOKEN`
3. `gh auth token`（仅当 `use_gh_cli=True` 时）

## 开发

```bash
# 安装开发依赖
uv sync --all-extras

# 格式化
uv run ruff format apps packages

# 检查
uv run ruff check apps packages
uv run mypy apps packages

# 测试
uv run pytest
```

## 添加新项目

```bash
# 添加应用
cd apps && uv init my-app

# 添加共享库
cd packages && uv init my-lib
```

新项目会自动被 workspace 识别。
