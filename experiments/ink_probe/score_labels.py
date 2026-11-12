#!/usr/bin/env python3
"""Pixel agreement of ink maps with a human ink label: ROC AUC and average precision.

Scored inside the certainty mask only, with label > 127 as ink, as the dataset
YoussefMoNader/ink-8um-pherc1447-surfaces scores its own predictions. A map on another pixel grid
(a 9.6 um map of an 8.64 um render) is resized to the label's grid first.

    uv run python score_labels.py --label w058/labels/inklabels.png --mask w058/labels/mask.png \
        w058/predictions/v8in.png out/w058_hecate_fwd.png out/w058_hecate_rev.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import rankdata

Image.MAX_IMAGE_PIXELS = None


def auc_ap(score: np.ndarray, ink: np.ndarray) -> tuple[float, float]:
    n_pos = int(ink.sum())
    n_neg = len(ink) - n_pos
    auc = (rankdata(score)[ink].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    order = np.argsort(-score, kind="stable")
    hits = ink[order]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(auc), float(precision[hits].mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("maps", nargs="+", type=Path)
    ap.add_argument("--label", required=True, type=Path)
    ap.add_argument("--mask", required=True, type=Path)
    args = ap.parse_args()
    label = np.asarray(Image.open(args.label).convert("L"))
    inside = np.asarray(Image.open(args.mask).convert("L")) > 0
    ink = label[inside] > 127
    print(f"{int(inside.sum())} labelled pixels, {100 * ink.mean():.1f} % ink")
    for path in args.maps:
        img = Image.open(path)
        if img.mode not in ("L", "I;16", "F", "I"):
            img = img.convert("L")
        if img.size != (label.shape[1], label.shape[0]):
            img = img.resize((label.shape[1], label.shape[0]), Image.BILINEAR)
        auc, average_precision = auc_ap(np.asarray(img, dtype=np.float32)[inside], ink)
        print(f"{path.name:44s} AUC {auc:.3f}  AP {average_precision:.3f}")


if __name__ == "__main__":
    main()
