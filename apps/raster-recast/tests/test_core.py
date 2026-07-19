from pathlib import Path

from PIL import Image
from raster_recast.core import save_image, validate_box


def test_validate_box_accepts_selection_inside_image() -> None:
    image = Image.new("RGB", (100, 80))

    assert validate_box(image, (10, 12, 40, 20)) == (10, 12, 40, 20)


def test_save_image_creates_png(tmp_path: Path) -> None:
    image = Image.new("RGBA", (20, 20), (10, 20, 30, 128))
    output = tmp_path / "nested" / "output.png"

    save_image(image, output)

    with Image.open(output) as saved:
        assert saved.mode == "RGBA"
        assert saved.size == (20, 20)
