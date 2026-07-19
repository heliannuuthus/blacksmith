"""Small image primitives shared by the GUI and AI backend."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

Box = tuple[int, int, int, int]


def validate_box(image: Image.Image, box: Box) -> Box:
    """Validate that an x/y/width/height selection is inside an image."""
    x, y, width, height = box
    if width <= 0 or height <= 0:
        raise ValueError("框选区域的宽和高必须大于 0")
    if x < 0 or y < 0 or x + width > image.width or y + height > image.height:
        raise ValueError("框选区域超出图片边界")
    return box


def save_image(image: Image.Image, output: Path) -> None:
    """Save an image, using high-quality settings for JPEG output."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() in {".jpg", ".jpeg"}:
        image.convert("RGB").save(output, quality=95, subsampling=0)
    else:
        image.save(output)
