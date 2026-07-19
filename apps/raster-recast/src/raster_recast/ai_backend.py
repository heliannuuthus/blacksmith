"""Client-side FLUX-Text editing logic for RasterRecast's private AI worker."""

from __future__ import annotations

import io
import tempfile
import time
from dataclasses import dataclass
from multiprocessing.connection import Client
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageFilter

from raster_recast.core import Box, validate_box
from raster_recast.inference_worker import AUTH_KEY

AI_MODEL_DOWNLOAD_GB = 19.4
AI_DISK_RESERVE_GB = 30
DEFAULT_INFERENCE_STEPS = 28
DEFAULT_TIMEOUT_SECONDS = 30 * 60
DEFAULT_SOCKET_PATH = Path(tempfile.gettempdir()) / "raster-recast.sock"

ProgressCallback = Callable[[str], None]


class AIBackendUnavailable(RuntimeError):
    """Raised when the private FLUX-Text process cannot serve a request."""


@dataclass(frozen=True)
class AIModelProfile:
    label: str
    steps: int
    description: str


MODEL_PROFILE = AIModelProfile(
    label="FLUX-Text · Multi-Resolution",
    steps=DEFAULT_INFERENCE_STEPS,
    description="纯 AI 场景文字编辑｜中文笔画约束、字体风格迁移、局部背景重建",
)


def _connect(socket_path: Path):  # type: ignore[no-untyped-def]
    try:
        return Client(str(socket_path), family="AF_UNIX", authkey=AUTH_KEY)
    except (OSError, EOFError) as exc:
        raise AIBackendUnavailable(f"无法连接内置 FLUX-Text 引擎：{exc}") from exc


def ai_environment_status(socket_path: Path | None = None) -> tuple[bool, str]:
    """Check the private standalone worker without loading model weights."""
    path = socket_path or DEFAULT_SOCKET_PATH
    try:
        connection = _connect(path)
        try:
            connection.send({"operation": "health"})
            if not connection.poll(2):
                return False, "内置 FLUX-Text 引擎尚未响应"
            response = connection.recv()
        finally:
            connection.close()
    except (AIBackendUnavailable, EOFError, OSError):
        return False, "内置 FLUX-Text 引擎尚未启动"
    if response.get("type") != "ready":
        return False, "内置 FLUX-Text 引擎状态异常"
    return True, "FLUX-Text AI 引擎已就绪"


def build_edit_prompt(replacement: str) -> str:
    """Build the semantic condition used alongside FLUX-Text glyph guidance."""
    text = replacement.strip()
    if not text:
        raise ValueError("FLUX-Text 精确替换需要填写新文字")
    return (
        f'Replace only the text inside the mask with exactly "{text}". '
        "Preserve the original typeface style, stroke weight, color, spacing, baseline, "
        "perspective, lighting and material. Reconstruct the occluded background naturally. "
        "Do not alter anything outside the masked text region."
    )


def expand_box(image: Image.Image, box: Box, context: int) -> Box:
    """Expand a selection by context pixels while staying inside the image."""
    x, y, width, height = validate_box(image, box)
    context = max(0, context)
    left = max(0, x - context)
    top = max(0, y - context)
    right = min(image.width, x + width + context)
    bottom = min(image.height, y + height + context)
    return left, top, right - left, bottom - top


def prepare_fluxtext_input(
    image: Image.Image,
    expanded: Box,
    selection: Box,
) -> Image.Image:
    """Create an RGBA contextual crop whose transparent pixels are the edit mask."""
    ex, ey, expanded_width, expanded_height = expanded
    x, y, width, height = validate_box(image, selection)
    crop = image.convert("RGBA").crop((ex, ey, ex + expanded_width, ey + expanded_height))
    alpha = Image.new("L", crop.size, 255)
    draw = ImageDraw.Draw(alpha)
    left = x - ex
    top = y - ey
    draw.rectangle((left, top, left + width - 1, top + height - 1), fill=0)
    crop.putalpha(alpha)
    return crop


def merge_generated_selection(
    original: Image.Image,
    generated_crop: Image.Image,
    expanded: Box,
    selection: Box,
    *,
    feather: int = 3,
) -> Image.Image:
    """Merge only the selected portion of the AI result back at original resolution."""
    ex, ey, expanded_width, expanded_height = expanded
    x, y, width, height = validate_box(original, selection)
    resized = generated_crop.convert("RGB").resize(
        (expanded_width, expanded_height), Image.Resampling.LANCZOS
    )
    relative = (x - ex, y - ey, x - ex + width, y - ey + height)
    patch = resized.crop(relative)

    base = original.convert("RGBA")
    patch_rgba = patch.convert("RGBA")
    mask = Image.new("L", (width, height), 255)
    if feather > 0 and min(width, height) > feather * 2:
        inner = Image.new("L", (width, height), 0)
        inner.paste(255, (feather, feather, width - feather, height - feather))
        mask = inner.filter(ImageFilter.GaussianBlur(radius=feather))
    base.paste(patch_rgba, (x, y), mask)
    return base if original.mode == "RGBA" else base.convert("RGB")


class FluxTextBackend:
    """Client for the standalone, local FLUX-Text/MMGP process."""

    def __init__(self, socket_path: Path | None = None) -> None:
        self.socket_path = socket_path or DEFAULT_SOCKET_PATH

    def _generate(
        self,
        image: Image.Image,
        replacement: str,
        *,
        seed: int,
        progress: ProgressCallback,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> Image.Image:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        connection = _connect(self.socket_path)
        try:
            connection.send(
                {
                    "operation": "edit",
                    "image": buffer.getvalue(),
                    "replacement": replacement.strip(),
                    "prompt": build_edit_prompt(replacement),
                    "steps": MODEL_PROFILE.steps,
                    "seed": seed,
                }
            )
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if not connection.poll(min(1.0, deadline - time.monotonic())):
                    continue
                response = connection.recv()
                response_type = response.get("type")
                if response_type == "progress":
                    progress(str(response.get("message", "AI 正在处理…")))
                elif response_type == "error":
                    raise AIBackendUnavailable(
                        f"FLUX-Text 推理失败：{response.get('message', '未知错误')}"
                    )
                elif response_type == "result":
                    with Image.open(io.BytesIO(response["image"])) as result:
                        return result.convert("RGB")
                else:
                    raise AIBackendUnavailable(f"AI 引擎返回了未知消息：{response_type}")
        except (EOFError, OSError) as exc:
            raise AIBackendUnavailable(f"内置 FLUX-Text 引擎意外退出：{exc}") from exc
        finally:
            connection.close()
        raise AIBackendUnavailable("FLUX-Text 推理超时")

    def edit_region(
        self,
        image: Image.Image,
        box: Box,
        replacement: str,
        *,
        context: int = 96,
        seed: int = 1,
        progress: ProgressCallback | None = None,
    ) -> Image.Image:
        """Send one contextual region to the private worker and merge its AI result."""
        report = progress or (lambda _message: None)
        build_edit_prompt(replacement)
        available, reason = ai_environment_status(self.socket_path)
        if not available:
            raise AIBackendUnavailable(reason)

        expanded = expand_box(image, box, context)
        model_input = prepare_fluxtext_input(image, expanded, box)
        report(f"正在执行 {MODEL_PROFILE.steps} 步纯 AI 文字编辑…")
        generated = self._generate(
            model_input,
            replacement,
            seed=seed,
            progress=report,
        )
        report("正在将 AI 局部结果合回原图…")
        return merge_generated_selection(image, generated, expanded, box)
