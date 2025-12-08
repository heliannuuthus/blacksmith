"""Markdown parser for HowToCook recipes."""

import logging
from typing import Any

from markdown_it import MarkdownIt

from .models import Dish, Tip

logger = logging.getLogger(__name__)


class MarkdownParser:
    """Simple markdown parser - extracts basic structure for LLM processing."""

    def __init__(self):
        """Initialize markdown parser."""
        self.md = MarkdownIt()
        logger.debug("Markdown parser initialized")

    def parse_dish(self, dish: Dish) -> dict[str, Any]:
        """Parse dish markdown into sections."""
        logger.debug("Parsing dish: %s", dish.name)
        return self._parse(dish.content)

    def parse_tip(self, tip: Tip) -> dict[str, Any]:
        """Parse tip markdown into sections."""
        logger.debug("Parsing tip: %s", tip.name)
        return self._parse(tip.content)

    def _parse(self, content: str) -> dict[str, Any]:
        """
        Parse markdown into simple structured format.
        
        Returns:
            {
                "title": "标题",
                "sections": {
                    "章节名": ["内容1", "内容2", ...]
                }
            }
        """
        tokens = self.md.parse(content)

        result: dict[str, Any] = {
            "title": "",
            "sections": {},
        }

        current_section: str | None = None
        current_items: list[str] = []

        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token.type == "heading_open":
                # 保存上一个 section
                if current_section:
                    result["sections"][current_section] = current_items
                    current_items = []

                level = int(token.tag[1])
                
                # 获取标题文本
                if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                    text = self._get_text(tokens[i + 1])
                    current_section = text
                    
                    if level == 1 and not result["title"]:
                        result["title"] = text
                    
                    i += 2
                    continue

            elif token.type == "paragraph_open":
                if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                    text = self._get_text(tokens[i + 1])
                    if text:
                        current_items.append(text)
                    i += 2
                    continue

            elif token.type in ["bullet_list_open", "ordered_list_open"]:
                items = self._parse_list(tokens, i)
                current_items.extend(items["items"])
                i = items["end_index"]
                continue

            elif token.type == "fence":
                if token.content:
                    current_items.append(f"```{token.info}\n{token.content}```")

            i += 1

        # 保存最后一个 section
        if current_section:
            result["sections"][current_section] = current_items

        return result

    def _parse_list(self, tokens: list, start: int) -> dict[str, Any]:
        """Parse list tokens into items."""
        items = []
        i = start + 1
        
        while i < len(tokens) and tokens[i].type not in ["bullet_list_close", "ordered_list_close"]:
            if tokens[i].type == "list_item_open":
                i += 1
                if i < len(tokens) and tokens[i].type == "paragraph_open":
                    i += 1
                    if i < len(tokens) and tokens[i].type == "inline":
                        items.append(self._get_text(tokens[i]))
                        i += 1
                    i += 1  # paragraph_close
                i += 1  # list_item_close
            else:
                i += 1
        
        return {"items": items, "end_index": i + 1}

    def _get_text(self, inline_token: Any) -> str:
        """Extract plain text from inline token."""
        if not inline_token.children:
            return getattr(inline_token, "content", "")

        parts = []
        for child in inline_token.children:
            if child.type == "text":
                parts.append(child.content)
            elif child.type == "code_inline":
                parts.append(child.content)
            elif child.type == "softbreak":
                parts.append(" ")
            elif hasattr(child, "content") and child.content:
                parts.append(child.content)

        return "".join(parts)
