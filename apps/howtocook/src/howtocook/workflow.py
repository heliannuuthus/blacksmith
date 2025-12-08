"""LangGraph workflow for recipe detection and refinement."""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from .parser import MarkdownParser
from .scraper import FileInfo, HowToCookScraper

logger = logging.getLogger(__name__)

QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


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
    parsed_recipes: list[dict]
    processed_recipes: list[dict]
    total: int
    fetched: int
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
    return ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base=QWEN_BASE_URL,
        temperature=temperature,
    )


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


# ============ Nodes ============

def fetch_node(state: DetectState) -> DetectState:
    """Parallel fetch and parse recipes."""
    logger.info("Fetching recipes...")
    
    scraper = HowToCookScraper(
        token=os.environ.get("GITHUB_TOKEN"),
        use_gh_cli=os.environ.get("USE_GH_CLI", "").lower() == "true",
    )
    parser = MarkdownParser()
    
    file_infos = list(scraper.list_dish_files(category=state.get("category")))
    logger.info("Found %d files", len(file_infos))
    
    parsed = []
    errors = []
    
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
            logger.error("Fetch failed %s: %s", info.path, e)
            return None
    
    with ThreadPoolExecutor(max_workers=state.get("concurrency", 8)) as pool:
        futures = {pool.submit(fetch_one, info): info for info in file_infos}
        for future in as_completed(futures):
            result = future.result()
            if result:
                parsed.append(result)
            else:
                errors.append(futures[future].path)
    
    logger.info("Fetched %d, errors %d", len(parsed), len(errors))
    
    return {
        **state,
        "parsed_recipes": parsed,
        "total": len(file_infos),
        "fetched": len(parsed),
        "errors": errors,
    }


def process_node(state: DetectState) -> DetectState:
    """Evaluate and refine recipes with LLM."""
    recipes = state["parsed_recipes"]
    logger.info("Processing %d recipes with LLM...", len(recipes))
    
    llm = get_llm("qwen-plus")
    processed = []
    refined_count = 0
    
    for recipe in recipes:
        parsed = recipe["parsed"]
        sections_text = format_sections(parsed)
        
        try:
            response = llm.invoke(
                PROCESS_PROMPT.format_messages(
                    title=parsed.get("title", recipe["name"]),
                    sections=sections_text,
                )
            )
            
            result = parse_llm_json(response.content)
            score = result.get("evaluation", {}).get("score", 5)
            
            processed.append({
                **recipe,
                "evaluation": result.get("evaluation", {}),
                "refined": result.get("recipe", {}),
            })
            
            if score < 8:
                refined_count += 1
                
        except Exception as e:
            logger.warning("Process failed %s: %s", recipe["name"], e)
            processed.append({
                **recipe,
                "evaluation": {"score": 0, "issues": [str(e)]},
                "refined": {},
            })
    
    logger.info("Processed %d, refined %d", len(processed), refined_count)
    
    return {
        **state,
        "processed_recipes": processed,
        "refined": refined_count,
    }


def save_node(state: DetectState) -> DetectState:
    """Save results to disk."""
    output_dir = Path(state["output_dir"])
    saved = 0
    
    for recipe in state["processed_recipes"]:
        try:
            cat_dir = output_dir / recipe.get("category", "unknown")
            cat_dir.mkdir(parents=True, exist_ok=True)
            
            filename = sanitize_filename(recipe["name"]) + ".json"
            file_path = cat_dir / filename
            
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(recipe, f, ensure_ascii=False, indent=2)
            saved += 1
            
        except Exception as e:
            logger.error("Save failed %s: %s", recipe["name"], e)
    
    logger.info("Saved %d recipes", saved)
    return {**state, "saved": saved}


# ============ Workflow ============

def build_workflow() -> StateGraph:
    """Build detect workflow."""
    wf = StateGraph(DetectState)
    
    wf.add_node("fetch", fetch_node)
    wf.add_node("process", process_node)
    wf.add_node("save", save_node)
    
    wf.set_entry_point("fetch")
    wf.add_edge("fetch", "process")
    wf.add_edge("process", "save")
    wf.add_edge("save", END)
    
    return wf


class RecipeDetector:
    """Recipe detection workflow executor."""
    
    def __init__(self):
        self.app = build_workflow().compile()
        logger.info("Detector initialized")
    
    def run(
        self,
        output_dir: str = "datasets/howtocook/dishes",
        category: str | None = None,
        concurrency: int = 8,
    ) -> dict[str, int]:
        """Run workflow."""
        result = self.app.invoke({
            "category": category,
            "output_dir": output_dir,
            "concurrency": concurrency,
            "parsed_recipes": [],
            "processed_recipes": [],
            "total": 0,
            "fetched": 0,
            "refined": 0,
            "saved": 0,
            "errors": [],
            "messages": [],
        })
        
        return {
            "total": result["total"],
            "fetched": result["fetched"],
            "refined": result["refined"],
            "saved": result["saved"],
            "errors": len(result["errors"]),
        }
