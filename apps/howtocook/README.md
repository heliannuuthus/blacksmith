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
# 默认使用百炼
export LLM_PROVIDER=bailian
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
- **process**: 百炼默认使用 `qwen3.6-plus` 评估、清洗和打标
- **refine**: 百炼默认使用 `qwen3.7-max` 优化低分菜谱
- **save**: 保存结构化 JSON

百炼默认关闭深度思考，以降低批处理延迟和输出 Token 消耗；菜谱质量仍通过低分复审节点保障。

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
    "tags": {
      "cuisines": ["家常菜"],
      "flavors": ["咸鲜"],
      "scenes": ["晚餐", "宴客"]
    },
    "tips": ["咸口放 50g 豆瓣酱"]
  }
}
```

标签使用与 Helios Zwei 一致的三类结构：`cuisines`、`flavors`、`scenes`。工作流会将
模型输出限制在内置词表中，去重并丢弃未知标签，避免同义标签持续膨胀。

## 环境变量

| 变量 | 说明 |
|------|------|
| `GITHUB_TOKEN` | GitHub token（可选，提高限额）|
| `LLM_PROVIDER` | `bailian`（默认）、`openrouter` 或 `custom` |
| `DASHSCOPE_API_KEY` | 百炼 API Key |
| `BAILIAN_PROCESS_MODEL` | 百炼清洗模型，默认 `qwen3.6-plus` |
| `BAILIAN_REFINE_MODEL` | 百炼复审模型，默认 `qwen3.7-max` |
| `OPENROUTER_API_KEY` | OpenRouter API Key |
| `OPENROUTER_BASE_URL` | API 地址，默认 `https://openrouter.ai/api/v1` |
| `OPENROUTER_PROCESS_MODEL` | 清洗与打标模型，默认 `moonshotai/kimi-k2.6` |
| `OPENROUTER_REFINE_MODEL` | 低分复审模型，默认 `moonshotai/kimi-k3` |
| `OPENROUTER_HTTP_REFERER` | 可选的应用来源 URL |
| `OPENROUTER_APP_TITLE` | 可选的应用名称 |
| `CUSTOM_API_KEY` | OpenAI 兼容自定义服务的 API Key |
| `CUSTOM_BASE_URL` | 自定义服务地址，例如 `https://example.com/v1` |
| `CUSTOM_PROCESS_MODEL` | 自定义清洗模型，默认 `qwen3.7-max` |
| `CUSTOM_REFINE_MODEL` | 自定义复审模型，默认 `qwen3.7-max` |
| `LLM_CONCURRENCY` | LLM 并发数，默认 `5` |
| `LLM_REQUEST_TIMEOUT` | 单次 LLM 请求超时秒数，默认 `180` |
| `USE_GH_CLI` | 使用 gh cli 凭证（设为 true）|

## 可恢复批处理

使用自定义 OpenAI 兼容服务运行全量清洗：

```bash
bash apps/howtocook/scripts/run-detect-custom.sh
```

脚本会静默读取 API Key，不会写入文件。处理结果按 provider 和模型写入 checkpoint；
网络失败或手动中断后，重新执行同一命令即可恢复。切换模型会自动重新评估，避免混用结果。
可通过 `CUSTOM_BASE_URL`、`CUSTOM_PROCESS_MODEL`、`CUSTOM_REFINE_MODEL`、
`LLM_CONCURRENCY` 和 `LLM_REQUEST_TIMEOUT` 覆盖默认配置。

## 依赖

- `gh`: GitHub API 客户端
- `click`: CLI 框架
- `markdown-it-py`: Markdown 解析
- `langchain` / `langgraph`: LLM 工作流
- `langchain-openai`: OpenAI 兼容 API 客户端
