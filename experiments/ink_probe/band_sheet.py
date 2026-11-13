#!/usr/bin/env python3
"""Contact sheets of the ink maps that screen_band.sh made, one strip per winding.

    uv run python band_sheet.py <work>/seg --out <work>/sheets [--scale 8] [--per-sheet 8]

Each map is reduced by --scale (block mean) and stretched so that probability 0.1 is black and
0.6 white. A strip runs along the winding: its long side is the way round the scroll, its short
side the scroll's axis. Writes <out>/sheet_NN.jpg and <out>/windings.json with, per winding, the
map's size, the share of it that holds papyrus (probability above 0) and the share of that with
probability above 0.5.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
LABEL = 150  # pixels kept left of each strip for its name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("seg", type=Path, help="folder with w###/preds/*.tif")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scale", type=int, default=8)
    ap.add_argument("--per-sheet", type=int, default=8)
    ap.add_argument("--width", type=int, default=2000, help="a longer strip continues on the next line")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report, rows = {}, []
    for folder in sorted(p for p in args.seg.iterdir() if p.is_dir()):
        maps = sorted((folder / "preds").glob("*.tif")) if (folder / "preds").is_dir() else []
        if not maps:
            continue
        image = Image.open(maps[0])
        full = np.asarray(image)
        papyrus = full > 0
        report[folder.name] = {
            "map": maps[0].name,
            "height": full.shape[0],
            "width": full.shape[1],
            "papyrus_share": float(papyrus.mean()),
            "above_half_share": float((full[papyrus] > 127).mean()) if papyrus.any() else 0.0,
        }
        small = np.asarray(image.reduce(args.scale), dtype=np.float32) / 255
        shown = (np.clip((small - 0.1) / 0.5, 0, 1) * 255).astype(np.uint8)
        for part, begin in enumerate(range(0, shown.shape[1], args.width)):
            piece = shown[:, begin : begin + args.width]
            if piece.any():
                rows.append((f"{folder.name}{'' if part == 0 else f' +{part}'}", piece))
    for number, begin in enumerate(range(0, len(rows), args.per_sheet)):
        chosen = rows[begin : begin + args.per_sheet]
        height = sum(piece.shape[0] + 6 for _, piece in chosen)
        sheet = Image.new("L", (LABEL + args.width, height), 40)
        pen = ImageDraw.Draw(sheet)
        y = 0
        for name, piece in chosen:
            sheet.paste(Image.fromarray(piece), (LABEL, y))
            stats = report[name.split()[0]]
            pen.text((6, y + 6), name, fill=255)
            pen.text((6, y + 22), f"{100 * stats['above_half_share']:.1f}% > 0.5", fill=200)
            y += piece.shape[0] + 6
        sheet.save(args.out / f"sheet_{number:02d}.jpg", quality=88)
    (args.out / "windings.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"{len(report)} windings, {len(rows)} strips on {-(-len(rows) // args.per_sheet)} sheets in {args.out}")


if __name__ == "__main__":
    main()
