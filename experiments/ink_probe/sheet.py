#!/usr/bin/env python3
"""Contact sheet of one surface: the CT of its middle layer beside each ink map given.

    uv run python sheet.py <stack.npy> <map>[:LO:HI] [<map>[:LO:HI] ...] --out sheet.jpg
        [--width 1600] [--crop Y0 Y1 X0 X1]

A map is an 8-bit image (0-255 = probability 0-1). It is shown as stored, or with probabilities
LO..HI spread over black..white when the suffix is given (the caption then says so); there is no
per-image stretch. A map on another pixel grid (a 9.6 um map of an 8.64 um render) is resized to
the stack's grid. --crop is in stack pixels. Panels sit side by side, or one above the other when
the shown area is more than 1.5 times as wide as it is high.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None


def load_map(spec: str, shape: tuple[int, int]) -> tuple[str, np.ndarray]:
    """(caption, uint8 image on the stack's grid) of one `<map>[:LO:HI]` argument."""
    parts = spec.rsplit(":", 2)
    path, lo, hi = (Path(parts[0]), float(parts[1]), float(parts[2])) if len(parts) == 3 else (Path(spec), 0.0, 1.0)
    img = Image.open(path).convert("L")
    if img.size != (shape[1], shape[0]):
        img = img.resize((shape[1], shape[0]), Image.BILINEAR)
    shown = np.clip((np.asarray(img, dtype=np.float32) / 255 - lo) / (hi - lo), 0, 1)
    caption = path.stem[-28:] + ("" if (lo, hi) == (0.0, 1.0) else f" [{lo:g}-{hi:g}]")
    return caption, (255 * shown).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stack", type=Path, help="(layer, y, x) uint8 .npy")
    ap.add_argument("maps", nargs="+", help="<map>[:LO:HI]")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--width", type=int, default=1600, help="pixel width of the sheet")
    ap.add_argument("--crop", type=int, nargs=4, metavar=("Y0", "Y1", "X0", "X1"))
    args = ap.parse_args()
    stack = np.load(args.stack, mmap_mode="r")
    panels = [("CT, middle layer", np.asarray(stack[stack.shape[0] // 2]))]
    panels += [load_map(spec, panels[0][1].shape) for spec in args.maps]
    if args.crop:
        y0, y1, x0, x1 = args.crop
        panels = [(name, img[y0:y1, x0:x1]) for name, img in panels]
    rows, cols = panels[0][1].shape
    across = cols <= 1.5 * rows  # side by side, else one above the other
    w = args.width // len(panels) if across else args.width
    h = round(w * rows / cols)
    step = (w, 0) if across else (0, h + 18)
    sheet = Image.new("L", (w + step[0] * (len(panels) - 1), h + 18 + step[1] * (len(panels) - 1)), 0)
    draw = ImageDraw.Draw(sheet)
    for i, (name, img) in enumerate(panels):
        sheet.paste(Image.fromarray(img).resize((w, h), Image.LANCZOS), (i * step[0], i * step[1] + 18))
        draw.text((i * step[0] + 4, i * step[1] + 3), name, fill=255)
    sheet.save(args.out, quality=90)
    print(args.out, sheet.size)


if __name__ == "__main__":
    main()
