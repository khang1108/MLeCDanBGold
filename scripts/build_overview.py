"""Combine review sheets into a compact visual overview."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    """Write one overview image for quick manual screening."""

    root = Path("artifacts/benchmark/2026-09-06/review_sheets")
    files = sorted(root.glob("*.jpg"))
    font = ImageFont.load_default()
    tile_w, tile_h = 420, 290
    columns = 3
    canvas = Image.new("RGB", (columns * tile_w, ((len(files) + columns - 1) // columns) * tile_h), "#bbbbbb")
    for i, path in enumerate(files):
        image = Image.open(path).convert("RGB")
        image.thumbnail((tile_w - 10, tile_h - 35))
        x, y = (i % columns) * tile_w, (i // columns) * tile_h
        canvas.paste(image, (x + (tile_w - image.width) // 2, y + 3))
        ImageDraw.Draw(canvas).text((x + 4, y + tile_h - 25), path.stem, fill="black", font=font)
    canvas.save(root / "overview.jpg", quality=85)


if __name__ == "__main__":
    main()
