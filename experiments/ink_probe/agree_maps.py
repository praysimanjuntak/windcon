#!/usr/bin/env python3
"""Do independent ink models light up the same places? Pairwise correlation of ink maps.

Pearson correlation of the maps' pixel values inside the surface's filled area (any layer > 0,
eroded by --margin pixels to drop the tiled rim), on the stack's pixel grid. Maps on another grid
are resized. Where two models trained on different data agree, the response is less likely to be
either model's artefact; PHerc. 1447's labelled windings give the level real ink reaches.

    uv run python agree_maps.py <stack.npy> <map> <map> [<map> ...] [--margin 32]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion

Image.MAX_IMAGE_PIXELS = None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stack", type=Path, help="(layer, y, x) uint8 .npy")
    ap.add_argument("maps", nargs="+", type=Path)
    ap.add_argument("--margin", type=int, default=32)
    args = ap.parse_args()
    stack = np.load(args.stack, mmap_mode="r")
    inside = binary_erosion(np.asarray(stack).max(axis=0) > 0, iterations=args.margin)
    values = []
    for path in args.maps:
        img = Image.open(path).convert("L")
        if img.size != (inside.shape[1], inside.shape[0]):
            img = img.resize((inside.shape[1], inside.shape[0]), Image.BILINEAR)
        values.append(np.asarray(img, dtype=np.float32)[inside])
    corr = np.corrcoef(np.stack(values))
    names = [p.stem[-24:] for p in args.maps]
    print(f"{int(inside.sum())} pixels")
    for i, name in enumerate(names):
        print(f"{name:>24s} " + " ".join(f"{corr[i, j]:+.2f}" for j in range(len(names))))


if __name__ == "__main__":
    main()
