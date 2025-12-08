# howtocook

抓取 [HowToCook](https://github.com/Anduin2017/HowToCook) 项目数据，支持增量同步和 LLM 优化。

## 安装

```bash
uv sync
```

## 命令

### 增量同步

```bash
# 同步菜品（增量）
uv run howtocook dishes

# 同步技巧
uv run howtocook tips

# 强制全量同步
uv run howtocook dishes --force

# 指定分类
uv run howtocook dishes -c meat_dish

# 解析 markdown 结构
uv run howtocook dishes -p

# 调整并发
uv run howtocook dishes -j 16
```

### LLM 工作流（detect）

一键完成：抓取 → 解析 → 评估 → 优化 → 保存

```bash
# 设置 API Key
export DASHSCOPE_API_KEY=sk-xxx

# 运行完整工作流
uv run howtocook detect

# 指定分类
uv run howtocook detect -c aquatic

# 自定义输出
uv run howtocook detect -o ./output
```

工作流程：
```
┌─────────┐     ┌─────────┐     ┌──────┐
│  fetch  │ ──► │ process │ ──► │ save │
│ (并行)  │     │  (LLM)  │     │      │
└─────────┘     └─────────┘     └──────┘
```

- **fetch**: 并行抓取 + markdown 解析
- **process**: qwen-plus 评估并优化
- **save**: 保存结构化 JSON

### 其他命令

```bash
# 查看分类
uv run howtocook categories

# 查看同步状态
uv run howtocook status -o datasets/howtocook/dishes

# 使用 gh cli 凭证
uv run howtocook --use-gh-cli dishes

# 详细日志
uv run howtocook -vv dishes
```

## 输出格式

```json
{
  "name": "酱炖蟹",
  "category": "aquatic",
  "path": "dishes/aquatic/酱炖蟹.md",
  "sha": "abc123",
  "html_url": "https://github.com/...",
  "parsed": {
    "title": "酱炖蟹的做法",
    "sections": {
      "必备原料和工具": ["螃蟹", "豆瓣酱", ...],
      "操作": ["步骤1", "步骤2", ...]
    }
  },
  "evaluation": {
    "score": 8,
    "issues": []
  },
  "refined": {
    "title": "酱炖蟹",
    "description": "秋日限定咸鲜风味",
    "difficulty": 3,
    "ingredients": [
      {"name": "螃蟹", "amount": "500g", "note": "首选河蟹"}
    ],
    "steps": [
      {"order": 1, "action": "刷洗干净", "duration": "", "tips": ""}
    ],
    "tips": ["咸口放 50g 豆瓣酱"]
  }
}
```

## 环境变量

| 变量 | 说明 |
|------|------|
| `GITHUB_TOKEN` | GitHub token（可选，提高限额）|
| `DASHSCOPE_API_KEY` | 千问 API Key（detect 必需）|
| `USE_GH_CLI` | 使用 gh cli 凭证（设为 true）|

## 依赖

- `gh`: GitHub API 客户端
- `click`: CLI 框架
- `markdown-it-py`: Markdown 解析
- `langchain` / `langgraph`: LLM 工作流
- `langchain-openai`: 千问 API
