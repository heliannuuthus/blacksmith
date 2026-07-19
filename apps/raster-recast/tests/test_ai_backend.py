import pytest
from PIL import Image
from raster_recast.ai_backend import (
    build_edit_prompt,
    expand_box,
    merge_generated_selection,
    prepare_fluxtext_input,
)


def test_build_edit_prompt_keeps_exact_chinese_and_digits() -> None:
    prompt = build_edit_prompt("张三2026")

    assert "张三2026" in prompt
    assert "exactly" in prompt
    assert "typeface style" in prompt
    assert "outside the masked text region" in prompt


def test_build_edit_prompt_requires_replacement() -> None:
    with pytest.raises(ValueError, match="需要填写新文字"):
        build_edit_prompt("  ")


def test_expand_box_is_clamped_to_image() -> None:
    image = Image.new("RGB", (200, 100))

    assert expand_box(image, (5, 10, 30, 20), 25) == (0, 0, 60, 55)


def test_prepare_fluxtext_input_encodes_selection_as_alpha_mask() -> None:
    image = Image.new("RGB", (100, 80), "white")
    prepared = prepare_fluxtext_input(image, (10, 10, 70, 50), (30, 25, 20, 15))

    assert prepared.mode == "RGBA"
    assert prepared.size == (70, 50)
    assert prepared.getpixel((0, 0))[3] == 255
    assert prepared.getpixel((20, 15))[3] == 0


def test_merge_generated_selection_changes_only_selected_box() -> None:
    original = Image.new("RGB", (100, 80), "white")
    generated = Image.new("RGB", (70, 60), "red")

    result = merge_generated_selection(
        original,
        generated,
        expanded=(10, 10, 70, 60),
        selection=(30, 25, 20, 15),
        feather=0,
    )

    assert result.crop((30, 25, 50, 40)).getcolors() == [(300, (255, 0, 0))]
    assert result.getpixel((29, 25)) == (255, 255, 255)
    assert result.getpixel((50, 39)) == (255, 255, 255)
