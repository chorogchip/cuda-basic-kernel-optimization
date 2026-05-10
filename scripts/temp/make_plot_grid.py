#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
GENERATED_OUTPUT_NAMES = {"all_plots_grid.png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine plot images into a compact square grid.")
    parser.add_argument("--input-dir", default="plots", help="Directory containing plot images.")
    parser.add_argument("--output", default="plots/all_plots_grid.png", help="Output image path.")
    parser.add_argument("--tile-size", type=int, default=440, help="Square tile size in pixels.")
    parser.add_argument("--gap", type=int, default=28, help="Gap between tiles in pixels.")
    parser.add_argument("--padding", type=int, default=40, help="Outer padding in pixels.")
    parser.add_argument("--label-height", type=int, default=34, help="Space reserved for each filename label.")
    parser.add_argument("--no-labels", action="store_true", help="Do not draw filename labels.")
    return parser.parse_args()


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.stem)]


def find_images(input_dir: Path, output: Path) -> list[Path]:
    output = output.resolve()
    images = [
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and path.resolve() != output
        and path.name not in GENERATED_OUTPUT_NAMES
    ]
    return sorted(images, key=natural_key)


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def fit_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> str:
    if draw.textlength(text, font=font) <= width:
        return text

    ellipsis = "..."
    available = max(0, width - int(draw.textlength(ellipsis, font=font)))
    shortened = ""
    for char in text:
        if draw.textlength(shortened + char, font=font) > available:
            break
        shortened += char
    return shortened.rstrip("_-. ") + ellipsis


def paste_tile(
    canvas: Image.Image,
    source_path: Path,
    xy: tuple[int, int],
    tile_size: int,
    label_height: int,
    font: ImageFont.ImageFont,
    draw_labels: bool,
) -> None:
    x, y = xy
    draw = ImageDraw.Draw(canvas)

    shadow_box = [x + 3, y + 5, x + tile_size + 3, y + tile_size + 5]
    tile_box = [x, y, x + tile_size, y + tile_size]
    draw.rounded_rectangle(shadow_box, radius=14, fill=(221, 226, 235))
    draw.rounded_rectangle(tile_box, radius=14, fill=(255, 255, 255), outline=(221, 226, 235), width=1)

    label_space = label_height if draw_labels else 0
    image_area = tile_size - label_space - 26
    image_box = (x + 13, y + 13, image_area, image_area)

    with Image.open(source_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        thumbnail = ImageOps.contain(image, (image_area, image_area), method=Image.Resampling.LANCZOS)

    px = image_box[0] + (image_area - thumbnail.width) // 2
    py = image_box[1] + (image_area - thumbnail.height) // 2
    canvas.paste(thumbnail, (px, py))

    if draw_labels:
        label = fit_text(draw, source_path.stem.replace("_", " "), font, tile_size - 30)
        bbox = draw.textbbox((0, 0), label, font=font)
        tx = x + (tile_size - (bbox[2] - bbox[0])) // 2
        ty = y + tile_size - label_height + 4
        draw.text((tx, ty), label, fill=(45, 55, 72), font=font)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output = Path(args.output)

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    images = find_images(input_dir, output)
    if not images:
        raise SystemExit(f"No images found in {input_dir}")

    cols = math.ceil(math.sqrt(len(images)))
    rows = cols
    canvas_size = args.padding * 2 + cols * args.tile_size + (cols - 1) * args.gap
    canvas = Image.new("RGB", (canvas_size, canvas_size), (246, 248, 251))
    font = load_font(19)

    for index, image_path in enumerate(images):
        row, col = divmod(index, cols)
        x = args.padding + col * (args.tile_size + args.gap)
        y = args.padding + row * (args.tile_size + args.gap)
        paste_tile(
            canvas,
            image_path,
            (x, y),
            args.tile_size,
            args.label_height,
            font,
            not args.no_labels,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)
    print(f"Wrote {output} ({len(images)} images, {rows}x{cols}, {canvas_size}x{canvas_size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
