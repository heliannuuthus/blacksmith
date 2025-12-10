"""LangGraph workflow for recipe detection and refinement."""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated, Any, TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from .models import SyncManifest
from .parser import MarkdownParser
from .scraper import FileInfo, HowToCookScraper

# 加载 .env 文件（支持 GITHUB_TOKEN 和 DASHSCOPE_API_KEY）
load_dotenv()

logger = logging.getLogger(__name__)

QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_CONCURRENCY = 5  # LLM 调用并发度
LLM_MAX_RETRIES = 3  # LLM 调用最大重试次数
LLM_RETRY_MIN_WAIT = 2  # 重试最小等待时间（秒）
LLM_RETRY_MAX_WAIT = 30  # 重试最大等待时间（秒）
MANIFEST_FILE = ".detect-manifest.json"  # detect 工作流的增量同步 manifest
CHECKPOINT_DIR = ".detect-checkpoint"  # 临时存储目录


# ============ Pydantic Models ============

class Ingredient(BaseModel):
    """食材"""
    name: str = Field(description="食材名称")
    amount: str = Field(description="用量")
    note: str = Field(default="", description="备注")


class Step(BaseModel):
    """步骤"""
    order: int = Field(description="序号")
    action: str = Field(description="操作")
    duration: str = Field(default="", description="时间")
    tips: str = Field(default="", description="技巧")


# ============ State ============

class DetectState(TypedDict):
    """Workflow state."""
    category: str | None
    output_dir: str
    concurrency: int
    force: bool  # 强制全量处理，忽略增量检查
    parsed_recipes: list[dict]
    processed_recipes: list[dict]
    total: int
    fetched: int
    skipped: int  # 跳过的（已处理过的）
    refined: int
    saved: int
    errors: list[str]
    messages: Annotated[list, add_messages]


# ============ LLM ============

def get_llm(model: str = "qwen-plus", temperature: float = 0.2) -> ChatOpenAI:
    """Get Qwen LLM client."""
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY required")
    logger.debug("[LLM] 创建客户端: model=%s, temperature=%.2f", model, temperature)
    return ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base=QWEN_BASE_URL,
        temperature=temperature,
    )


def _do_invoke_llm(llm: ChatOpenAI, messages: list) -> Any:
    """Internal LLM invoke with retry."""
    return llm.invoke(messages)


def invoke_llm(llm: ChatOpenAI, messages: list, context: str = "") -> str:
    """Invoke LLM with logging and retry."""
    # 计算输入 token 估算（中文约 2 字符/token）
    input_text = str(messages)
    input_chars = len(input_text)
    input_tokens_est = input_chars // 2
    
    # INFO 级别：显示正在处理什么
    logger.info("[LLM] → %s", context)
    logger.debug("[LLM]   模型: %s, 输入: %d 字符 (~%d tokens)", llm.model_name, input_chars, input_tokens_est)
    
    start_time = time.time()
    attempt = 0
    last_error = None
    
    # 手动实现重试逻辑，以便记录详细日志
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            response = llm.invoke(messages)
            elapsed = time.time() - start_time
            
            output_text = response.content
            output_chars = len(output_text)
            output_tokens_est = output_chars // 2
            
            # INFO 级别：显示完成状态和耗时
            if attempt > 1:
                logger.info("[LLM] ✓ 完成 (%.1fs, ~%d tokens, 第%d次尝试)", elapsed, output_tokens_est, attempt)
            else:
                logger.info("[LLM] ✓ 完成 (%.1fs, ~%d tokens)", elapsed, output_tokens_est)
            
            # DEBUG 级别：详细信息
            logger.debug("[LLM]   响应: %d 字符, 速度: ~%.0f tokens/s", output_chars, output_tokens_est / elapsed if elapsed > 0 else 0)
            
            # 如果有 usage 信息（某些模型会返回）
            if hasattr(response, 'response_metadata') and response.response_metadata:
                usage = response.response_metadata.get('token_usage', {})
                if usage:
                    logger.debug("[LLM]   实际 tokens: 输入=%d, 输出=%d, 总计=%d",
                               usage.get('prompt_tokens', 0),
                               usage.get('completion_tokens', 0),
                               usage.get('total_tokens', 0))
            
            return response.content
            
        except Exception as e:
            last_error = e
            elapsed = time.time() - start_time
            
            if attempt < LLM_MAX_RETRIES:
                # 指数退避
                wait_time = min(LLM_RETRY_MIN_WAIT * (2 ** (attempt - 1)), LLM_RETRY_MAX_WAIT)
                logger.warning("[LLM] ⚠ 第%d次尝试失败 (%.1fs): %s, %d秒后重试...", 
                              attempt, elapsed, e, wait_time)
                time.sleep(wait_time)
            else:
                logger.error("[LLM] ✗ 全部%d次尝试失败 (%.1fs): %s", attempt, elapsed, e)
    
    # 所有重试都失败了
    raise last_error


# ============ Prompts ============

SYSTEM_PROMPT = """你是专业的中餐菜谱编辑。任务：
1. 评估菜谱数据完整性（1-10分）
2. 转换为标准化 JSON 格式
3. 保持原始信息准确，不编造

评分标准：
- 9-10: 完整精确，步骤详细
- 7-8: 基本完整，少量模糊
- 5-6: 缺少信息，需补充
- 1-4: 严重不完整"""

PROCESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """处理以下菜谱：

标题：{title}

内容：
{sections}

请返回 JSON（不要其他内容）：
```json
{{
  "evaluation": {{
    "score": 1-10,
    "issues": ["问题1", "问题2"]
  }},
  "recipe": {{
    "title": "菜名",
    "description": "一句话简介",
    "difficulty": 1-5,
    "servings": "份量",
    "ingredients": [
      {{"name": "食材", "amount": "用量", "note": "备注"}}
    ],
    "steps": [
      {{"order": 1, "action": "操作", "duration": "时间", "tips": "技巧"}}
    ],
    "tips": ["小贴士"]
  }}
}}
```

要求：
- 保留原始信息，不编造
- 统一单位（g/ml/个）
- 步骤有序号和时间"""),
])

# 优化提示词 - 用于低分菜谱的完善
REFINE_SYSTEM_PROMPT = """你是一位资深中餐大厨和菜谱编辑专家。你的任务是基于不完整的菜谱信息，生成一份专业、完整、可操作的菜谱。

你必须：
1. 保留原始菜谱中明确提供的所有信息
2. 基于菜名和已有信息，补充缺失的内容（食材用量、步骤细节、时间等）
3. 确保补充的内容符合中餐烹饪常识和该菜品的传统做法
4. 提供实用的烹饪技巧

注意：补充的内容应该合理、实用，不要凭空编造离谱的做法。"""

REFINE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", REFINE_SYSTEM_PROMPT),
    ("human", """请完善以下菜谱：

菜名：{title}

原始内容：
{sections}

评估问题：
{issues}

请生成一份完整的菜谱，返回 JSON（不要其他内容）：
```json
{{
  "title": "菜名",
  "description": "一句话简介，突出这道菜的特色",
  "difficulty": 1-5,
  "prep_time": "准备时间",
  "cook_time": "烹饪时间",
  "servings": "份量（如：2-3人份）",
  "ingredients": [
    {{"name": "食材名", "amount": "精确用量", "note": "选购/处理提示"}}
  ],
  "steps": [
    {{
      "order": 1,
      "action": "详细操作描述",
      "duration": "预计时间",
      "tips": "这一步的关键技巧"
    }}
  ],
  "tips": ["实用小贴士1", "小贴士2"]
}}
```

要求：
- 食材用量要精确（如：猪肉 300g、酱油 2汤匙）
- 步骤要详细可操作，包含火候、时间
- 补充原菜谱缺失的关键信息
- 保持菜品的正宗风味"""),
])


# ============ Utils ============

def format_sections(parsed: dict) -> str:
    """Format parsed sections to text."""
    lines = []
    for name, items in parsed.get("sections", {}).items():
        lines.append(f"## {name}")
        for item in items:
            lines.append(f"- {item}")
    return "\n".join(lines)


def parse_llm_json(content: str) -> dict:
    """Extract JSON from LLM response."""
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    return json.loads(content.strip())


def sanitize_filename(name: str) -> str:
    """Sanitize filename."""
    for char in '<>:"/\\|?*':
        name = name.replace(char, "_")
    return name.strip()


def load_manifest(output_dir: Path) -> SyncManifest:
    """Load detect manifest."""
    path = output_dir / MANIFEST_FILE
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return SyncManifest(**json.load(f))
    return SyncManifest()


def save_manifest(output_dir: Path, manifest: SyncManifest) -> None:
    """Save detect manifest."""
    path = output_dir / MANIFEST_FILE
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(mode="json"), f, ensure_ascii=False, indent=2, default=str)


# ============ Checkpoint (临时存储) ============

def get_checkpoint_dir(output_dir: Path) -> Path:
    """Get checkpoint directory path."""
    return output_dir / CHECKPOINT_DIR


def save_checkpoint(output_dir: Path, stage: str, data: dict) -> None:
    """Save checkpoint data for a stage."""
    checkpoint_dir = get_checkpoint_dir(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_file = checkpoint_dir / f"{stage}.json"
    with checkpoint_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.debug("[CHECKPOINT] 保存 %s: %d 条记录", stage, len(data.get("items", [])))


def load_checkpoint(output_dir: Path, stage: str) -> dict | None:
    """Load checkpoint data for a stage."""
    checkpoint_dir = get_checkpoint_dir(output_dir)
    checkpoint_file = checkpoint_dir / f"{stage}.json"
    
    if checkpoint_file.exists():
        with checkpoint_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
            logger.info("[CHECKPOINT] 恢复 %s: %d 条记录", stage, len(data.get("items", [])))
            return data
    return None


def clear_checkpoint(output_dir: Path, stage: str) -> None:
    """Clear checkpoint for a stage."""
    checkpoint_dir = get_checkpoint_dir(output_dir)
    checkpoint_file = checkpoint_dir / f"{stage}.json"
    
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        logger.debug("[CHECKPOINT] 清除 %s", stage)


def clear_all_checkpoints(output_dir: Path) -> None:
    """Clear all checkpoints."""
    checkpoint_dir = get_checkpoint_dir(output_dir)
    if checkpoint_dir.exists():
        import shutil
        shutil.rmtree(checkpoint_dir)
        logger.info("[CHECKPOINT] 清除所有临时数据")


def save_single_result(output_dir: Path, stage: str, path: str, result: dict) -> None:
    """Save a single processed result to checkpoint (for incremental saving)."""
    checkpoint_dir = get_checkpoint_dir(output_dir)
    stage_dir = checkpoint_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    
    # 使用 path 的 hash 作为文件名
    import hashlib
    filename = hashlib.md5(path.encode()).hexdigest() + ".json"
    result_file = stage_dir / filename
    
    with result_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def load_single_result(output_dir: Path, stage: str, path: str) -> dict | None:
    """Load a single processed result from checkpoint."""
    checkpoint_dir = get_checkpoint_dir(output_dir)
    stage_dir = checkpoint_dir / stage
    
    import hashlib
    filename = hashlib.md5(path.encode()).hexdigest() + ".json"
    result_file = stage_dir / filename
    
    if result_file.exists():
        with result_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


def has_checkpoint_result(output_dir: Path, stage: str, path: str) -> bool:
    """Check if a checkpoint result exists."""
    checkpoint_dir = get_checkpoint_dir(output_dir)
    stage_dir = checkpoint_dir / stage
    
    import hashlib
    filename = hashlib.md5(path.encode()).hexdigest() + ".json"
    result_file = stage_dir / filename
    
    return result_file.exists()


# ============ Nodes ============

def fetch_node(state: DetectState) -> DetectState:
    """Parallel fetch and parse recipes with incremental sync."""
    logger.info("=" * 50)
    logger.info("[FETCH] 开始获取菜谱...")
    logger.info("=" * 50)
    
    output_dir = Path(state["output_dir"])
    force = state.get("force", False)
    
    scraper = HowToCookScraper(
        token=os.environ.get("GITHUB_TOKEN"),
        use_gh_cli=os.environ.get("USE_GH_CLI", "").lower() == "true",
    )
    parser = MarkdownParser()
    
    category = state.get("category")
    logger.info("[FETCH] 目标分类: %s", category or "全部")
    logger.info("[FETCH] 并发数: %d", state.get("concurrency", 8))
    logger.info("[FETCH] 模式: %s", "强制全量" if force else "增量同步")
    
    # 加载 manifest 用于增量检查
    manifest = SyncManifest() if force else load_manifest(output_dir)
    if not force and manifest.files:
        logger.info("[FETCH] 已有 %d 个已处理记录", len(manifest.files))
    
    logger.info("[FETCH] 正在列出文件...")
    file_infos = list(scraper.list_dish_files(category=category))
    logger.info("[FETCH] 发现 %d 个菜谱文件", len(file_infos))
    
    # 增量检查：筛选需要处理的文件
    if force:
        to_fetch = file_infos
        skipped_count = 0
    else:
        to_fetch = [f for f in file_infos if manifest.needs_update(f.path, f.sha)]
        skipped_count = len(file_infos) - len(to_fetch)
    
    logger.info("[FETCH] 需要处理: %d 个, 跳过: %d 个 (已是最新)", len(to_fetch), skipped_count)
    
    if not to_fetch:
        logger.info("[FETCH] 所有文件已是最新，无需处理!")
        return {
            **state,
            "parsed_recipes": [],
            "total": len(file_infos),
            "fetched": 0,
            "skipped": skipped_count,
            "errors": [],
        }
    
    # 按分类统计待处理文件
    categories_count: dict[str, int] = {}
    for info in to_fetch:
        cat = info.category or "unknown"
        categories_count[cat] = categories_count.get(cat, 0) + 1
    logger.info("[FETCH] 待处理分类统计:")
    for cat, count in sorted(categories_count.items()):
        logger.info("[FETCH]   - %s: %d 个", cat, count)
    
    parsed = []
    errors = []
    fetch_count = 0
    
    def fetch_one(info: FileInfo) -> dict | None:
        try:
            dish = scraper.fetch_dish_by_path(info.path, info.category)
            return {
                "name": dish.name,
                "category": dish.category,
                "path": dish.path,
                "sha": dish.sha,
                "html_url": dish.html_url,
                "parsed": parser.parse_dish(dish),
            }
        except Exception as e:
            logger.error("[FETCH] ✗ 获取失败 %s: %s", info.path, e)
            return None
    
    logger.info("[FETCH] 开始并发下载...")
    with ThreadPoolExecutor(max_workers=state.get("concurrency", 8)) as pool:
        futures = {pool.submit(fetch_one, info): info for info in to_fetch}
        for future in as_completed(futures):
            result = future.result()
            fetch_count += 1
            if result:
                parsed.append(result)
                if fetch_count % 20 == 0 or fetch_count == len(to_fetch):
                    logger.info("[FETCH] 进度: %d/%d (%.1f%%)", fetch_count, len(to_fetch), fetch_count / len(to_fetch) * 100)
            else:
                errors.append(futures[future].path)
    
    logger.info("[FETCH] 完成! 成功: %d, 失败: %d, 跳过: %d", len(parsed), len(errors), skipped_count)
    if errors:
        logger.warning("[FETCH] 失败的文件:")
        for err_path in errors[:5]:  # 只显示前5个
            logger.warning("[FETCH]   - %s", err_path)
        if len(errors) > 5:
            logger.warning("[FETCH]   ... 还有 %d 个", len(errors) - 5)
    
    return {
        **state,
        "parsed_recipes": parsed,
        "total": len(file_infos),
        "fetched": len(parsed),
        "skipped": skipped_count,
        "errors": errors,
    }


def process_node(state: DetectState) -> DetectState:
    """Evaluate recipes with qwen-plus LLM (parallel with checkpoint)."""
    import threading
    
    recipes = state["parsed_recipes"]
    output_dir = Path(state["output_dir"])
    concurrency = LLM_CONCURRENCY
    
    logger.info("")
    logger.info("=" * 50)
    logger.info("[PROCESS] 开始评估菜谱质量...")
    logger.info("=" * 50)
    logger.info("[PROCESS] 模型: qwen-plus-2025-12-01")
    logger.info("[PROCESS] 待评估: %d 个菜谱", len(recipes))
    logger.info("[PROCESS] 并发数: %d", concurrency)
    
    llm = get_llm("qwen-plus-2025-12-01")
    
    # 线程安全的计数器和结果存储
    lock = threading.Lock()
    processed_count = 0
    restored_count = 0
    low_score_count = 0
    high_score_count = 0
    error_count = 0
    score_distribution = {i: 0 for i in range(11)}
    
    # 使用字典保持顺序（按索引存储）
    results: dict[int, dict] = {}
    
    def process_one(idx: int, recipe: dict) -> None:
        nonlocal processed_count, restored_count, low_score_count, high_score_count, error_count
        
        parsed = recipe["parsed"]
        sections_text = format_sections(parsed)
        recipe_name = parsed.get("title", recipe["name"])
        recipe_path = recipe["path"]
        
        # 检查是否有 checkpoint
        cached_result = load_single_result(output_dir, "process", recipe_path)
        if cached_result:
            with lock:
                results[idx] = cached_result
                score = cached_result.get("evaluation", {}).get("score", 0)
                score_distribution[score] = score_distribution.get(score, 0) + 1
                restored_count += 1
                processed_count += 1
                if score < 8:
                    low_score_count += 1
                else:
                    high_score_count += 1
                logger.info("[PROCESS] [%d/%d] ↺ %s (从缓存恢复, 分数: %d)", 
                           processed_count, len(recipes), recipe_name, score)
            return
        
        try:
            messages = PROCESS_PROMPT.format_messages(
                title=recipe_name,
                sections=sections_text,
            )
            response_content = invoke_llm(
                llm, messages,
                context=f"评估菜谱 {recipe_name}"
            )
            
            result = parse_llm_json(response_content)
            score = result.get("evaluation", {}).get("score", 5)
            issues = result.get("evaluation", {}).get("issues", [])
            
            processed_result = {
                **recipe,
                "evaluation": result.get("evaluation", {}),
                "refined": result.get("recipe", {}),
            }
            
            # 保存到 checkpoint
            save_single_result(output_dir, "process", recipe_path, processed_result)
            
            with lock:
                results[idx] = processed_result
                score_distribution[score] = score_distribution.get(score, 0) + 1
                processed_count += 1
                
                if score < 8:
                    low_score_count += 1
                    logger.info("[PROCESS] [%d/%d] ⚠ %s (分数: %d) - 需要优化", 
                               processed_count, len(recipes), recipe_name, score)
                    if issues:
                        for issue in issues[:2]:
                            logger.info("[PROCESS]         问题: %s", issue)
                else:
                    high_score_count += 1
                    logger.info("[PROCESS] [%d/%d] ✓ %s (分数: %d)", 
                               processed_count, len(recipes), recipe_name, score)
                
        except Exception as e:
            error_result = {
                **recipe,
                "evaluation": {"score": 0, "issues": [str(e)]},
                "refined": {},
            }
            # 不保存失败的到 checkpoint，下次重试
            with lock:
                results[idx] = error_result
                processed_count += 1
                low_score_count += 1
                error_count += 1
                score_distribution[0] = score_distribution.get(0, 0) + 1
                logger.error("[PROCESS] [%d/%d] ✗ %s 评估失败: %s", 
                            processed_count, len(recipes), recipe_name, e)
    
    # 并行执行
    logger.info("[PROCESS] 开始并行评估...")
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(process_one, idx, recipe) for idx, recipe in enumerate(recipes)]
        for future in as_completed(futures):
            future.result()  # 等待完成，异常已在 process_one 中处理
    
    # 按原始顺序排列结果
    processed = [results[i] for i in range(len(recipes))]
    
    if restored_count > 0:
        logger.info("[PROCESS] 从缓存恢复: %d 个", restored_count)
    
    # 输出统计
    logger.info("")
    logger.info("[PROCESS] 评估完成!")
    logger.info("[PROCESS] -" * 25)
    logger.info("[PROCESS] 高分 (≥8): %d 个 (%.1f%%)", high_score_count, high_score_count / len(recipes) * 100 if recipes else 0)
    logger.info("[PROCESS] 低分 (<8): %d 个 (%.1f%%)", low_score_count, low_score_count / len(recipes) * 100 if recipes else 0)
    logger.info("[PROCESS] 失败: %d 个", error_count)
    logger.info("[PROCESS] -" * 25)
    logger.info("[PROCESS] 评分分布:")
    for score in range(10, -1, -1):
        count = score_distribution.get(score, 0)
        if count > 0:
            bar = "█" * min(count, 30)
            logger.info("[PROCESS]   %2d分: %s %d", score, bar, count)
    
    return {
        **state,
        "processed_recipes": processed,
    }


def refine_node(state: DetectState) -> DetectState:
    """Refine low-score recipes with a more powerful model (parallel with checkpoint)."""
    import threading
    
    recipes = state["processed_recipes"]
    output_dir = Path(state["output_dir"])
    concurrency = LLM_CONCURRENCY
    
    logger.info("")
    logger.info("=" * 50)
    logger.info("[REFINE] 开始优化低分菜谱...")
    logger.info("=" * 50)
    
    # 筛选需要优化的菜谱（评分 < 8）
    low_score_indices = []
    for i, recipe in enumerate(recipes):
        score = recipe.get("evaluation", {}).get("score", 0)
        if score < 8:
            low_score_indices.append(i)
    
    if not low_score_indices:
        logger.info("[REFINE] 所有菜谱评分 ≥ 8，无需优化")
        logger.info("[REFINE] 跳过优化步骤")
        return state
    
    logger.info("[REFINE] 模型: qwen-max (更强大)")
    logger.info("[REFINE] 待优化: %d 个低分菜谱", len(low_score_indices))
    logger.info("[REFINE] 并发数: %d", concurrency)
    logger.info("")
    
    # 使用更强大的模型
    llm = get_llm("qwen-max", temperature=0.3)
    
    # 线程安全的计数器
    lock = threading.Lock()
    refined_count = 0
    restored_count = 0
    error_count = 0
    progress_count = 0
    
    def refine_one(idx: int) -> None:
        nonlocal refined_count, restored_count, error_count, progress_count
        
        recipe = recipes[idx]
        parsed = recipe["parsed"]
        sections_text = format_sections(parsed)
        issues = recipe.get("evaluation", {}).get("issues", [])
        issues_text = "\n".join(f"- {issue}" for issue in issues) if issues else "- 信息不完整"
        recipe_name = parsed.get("title", recipe["name"])
        recipe_path = recipe["path"]
        original_score = recipe.get("evaluation", {}).get("score", 0)
        
        # 检查是否有 checkpoint
        cached_result = load_single_result(output_dir, "refine", recipe_path)
        if cached_result and cached_result.get("refined_by") == "qwen-max":
            with lock:
                recipes[idx] = cached_result
                restored_count += 1
                refined_count += 1
                progress_count += 1
                ingredients_count = len(cached_result.get("refined", {}).get("ingredients", []))
                steps_count = len(cached_result.get("refined", {}).get("steps", []))
                logger.info("[REFINE] [%d/%d] ↺ %s (从缓存恢复: %d 食材, %d 步骤)", 
                           progress_count, len(low_score_indices), recipe_name, ingredients_count, steps_count)
            return
        
        try:
            messages = REFINE_PROMPT.format_messages(
                title=recipe_name,
                sections=sections_text,
                issues=issues_text,
            )
            response_content = invoke_llm(
                llm, messages,
                context=f"优化菜谱 {recipe_name} (原分数: {original_score})"
            )
            
            refined_recipe = parse_llm_json(response_content)
            
            refined_result = {
                **recipe,
                "refined": refined_recipe,
                "refined_by": "qwen-max",
            }
            
            # 保存到 checkpoint
            save_single_result(output_dir, "refine", recipe_path, refined_result)
            
            # 更新菜谱
            with lock:
                recipes[idx] = refined_result
                refined_count += 1
                progress_count += 1
                
                # 显示优化后的菜谱信息
                ingredients_count = len(refined_recipe.get("ingredients", []))
                steps_count = len(refined_recipe.get("steps", []))
                logger.info("[REFINE] [%d/%d] ✓ %s: %d 种食材, %d 个步骤", 
                           progress_count, len(low_score_indices), recipe_name, ingredients_count, steps_count)
            
        except Exception as e:
            # 不保存失败的到 checkpoint，下次重试
            with lock:
                recipes[idx] = {
                    **recipe,
                    "refine_error": str(e),
                }
                error_count += 1
                progress_count += 1
                logger.error("[REFINE] [%d/%d] ✗ %s 优化失败: %s", 
                            progress_count, len(low_score_indices), recipe_name, e)
    
    # 并行执行
    logger.info("[REFINE] 开始并行优化...")
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(refine_one, idx) for idx in low_score_indices]
        for future in as_completed(futures):
            future.result()
    
    if restored_count > 0:
        logger.info("[REFINE] 从缓存恢复: %d 个", restored_count)
    
    logger.info("")
    logger.info("[REFINE] 优化完成!")
    logger.info("[REFINE] -" * 25)
    logger.info("[REFINE] 成功: %d 个", refined_count)
    logger.info("[REFINE] 失败: %d 个", error_count)
    logger.info("[REFINE] 成功率: %.1f%%", refined_count / len(low_score_indices) * 100 if low_score_indices else 0)
    
    return {
        **state,
        "processed_recipes": recipes,
        "refined": refined_count,
    }


def save_node(state: DetectState) -> DetectState:
    """Save results to disk and update manifest."""
    output_dir = Path(state["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("")
    logger.info("=" * 50)
    logger.info("[SAVE] 开始保存结果...")
    logger.info("=" * 50)
    logger.info("[SAVE] 输出目录: %s", output_dir)
    
    # 加载现有 manifest（用于增量更新）
    manifest = load_manifest(output_dir)
    
    saved = 0
    error_count = 0
    categories_saved: dict[str, int] = {}
    
    recipes = state["processed_recipes"]
    logger.info("[SAVE] 待保存: %d 个菜谱", len(recipes))
    
    for i, recipe in enumerate(recipes, 1):
        try:
            cat = recipe.get("category", "unknown")
            cat_dir = output_dir / cat
            cat_dir.mkdir(parents=True, exist_ok=True)
            
            filename = sanitize_filename(recipe["name"]) + ".json"
            file_path = cat_dir / filename
            
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(recipe, f, ensure_ascii=False, indent=2)
            saved += 1
            categories_saved[cat] = categories_saved.get(cat, 0) + 1
            
            # 更新 manifest
            manifest.update_entry(recipe["path"], recipe["sha"], recipe.get("category"))
            
            # 每20个显示一次进度
            if i % 20 == 0 or i == len(recipes):
                logger.info("[SAVE] 进度: %d/%d (%.1f%%)", i, len(recipes), i / len(recipes) * 100)
            
        except Exception as e:
            logger.error("[SAVE] ✗ 保存失败 %s: %s", recipe["name"], e)
            error_count += 1
    
    # 保存 manifest
    if saved > 0:
        save_manifest(output_dir, manifest)
        logger.info("[SAVE] Manifest 已更新 (%d 条记录)", len(manifest.files))
    
    # 清理 checkpoint（保存成功后不再需要）
    if saved > 0 and error_count == 0:
        clear_all_checkpoints(output_dir)
    elif error_count > 0:
        logger.info("[SAVE] 保留 checkpoint 数据（有 %d 个错误，下次可恢复）", error_count)
    
    logger.info("")
    logger.info("[SAVE] 保存完成!")
    logger.info("[SAVE] -" * 25)
    logger.info("[SAVE] 成功: %d 个", saved)
    logger.info("[SAVE] 失败: %d 个", error_count)
    logger.info("[SAVE] -" * 25)
    logger.info("[SAVE] 按分类统计:")
    for cat, count in sorted(categories_saved.items()):
        logger.info("[SAVE]   - %s: %d 个", cat, count)
    
    return {**state, "saved": saved}


# ============ Workflow ============

def build_workflow() -> StateGraph:
    """Build detect workflow.
    
    Flow: fetch -> process -> refine -> save -> END
    
    - fetch: 并发获取和解析菜谱
    - process: 使用 qwen-plus 评估菜谱质量
    - refine: 使用 qwen-max 优化低分（<8分）菜谱
    - save: 保存结果到磁盘
    """
    wf = StateGraph(DetectState)
    
    wf.add_node("fetch", fetch_node)
    wf.add_node("process", process_node)
    wf.add_node("refine", refine_node)
    wf.add_node("save", save_node)
    
    wf.set_entry_point("fetch")
    wf.add_edge("fetch", "process")
    wf.add_edge("process", "refine")
    wf.add_edge("refine", "save")
    wf.add_edge("save", END)
    
    return wf


class RecipeDetector:
    """Recipe detection workflow executor."""
    
    def __init__(self):
        self.app = build_workflow().compile()
        logger.info("RecipeDetector 初始化完成")
    
    def run(
        self,
        output_dir: str = "datasets/howtocook/dishes",
        category: str | None = None,
        concurrency: int = 8,
        force: bool = False,
    ) -> dict[str, int]:
        """Run workflow."""
        import time
        start_time = time.time()
        
        logger.info("")
        logger.info("╔" + "═" * 48 + "╗")
        logger.info("║" + " 菜谱检测与优化工作流 ".center(46) + "║")
        logger.info("╚" + "═" * 48 + "╝")
        logger.info("")
        logger.info("配置:")
        logger.info("  输出目录: %s", output_dir)
        logger.info("  目标分类: %s", category or "全部")
        logger.info("  并发数量: %d", concurrency)
        logger.info("  同步模式: %s", "强制全量" if force else "增量")
        logger.info("")
        logger.info("工作流: fetch → process → refine → save")
        logger.info("")
        
        result = self.app.invoke({
            "category": category,
            "output_dir": output_dir,
            "concurrency": concurrency,
            "force": force,
            "parsed_recipes": [],
            "processed_recipes": [],
            "total": 0,
            "fetched": 0,
            "skipped": 0,
            "refined": 0,
            "saved": 0,
            "errors": [],
            "messages": [],
        })
        
        elapsed = time.time() - start_time
        
        logger.info("")
        logger.info("╔" + "═" * 48 + "╗")
        logger.info("║" + " 工作流执行完成 ".center(46) + "║")
        logger.info("╚" + "═" * 48 + "╝")
        logger.info("")
        logger.info("最终统计:")
        logger.info("  总菜谱数: %d", result["total"])
        logger.info("  跳过数量: %d (已是最新)", result.get("skipped", 0))
        logger.info("  获取成功: %d", result["fetched"])
        logger.info("  已优化数: %d", result["refined"])
        logger.info("  保存成功: %d", result["saved"])
        logger.info("  错误数量: %d", len(result["errors"]))
        logger.info("")
        logger.info("耗时: %.1f 秒 (%.1f 分钟)", elapsed, elapsed / 60)
        logger.info("")
        
        return {
            "total": result["total"],
            "skipped": result.get("skipped", 0),
            "fetched": result["fetched"],
            "refined": result["refined"],
            "saved": result["saved"],
            "errors": len(result["errors"]),
        }
