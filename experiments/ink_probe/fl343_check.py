#!/usr/bin/env python3
"""Compare ink maps of the PHerc. 343 First Letters segment with the submission's letter boxes.

The submission (github.com/Nieuwlaar/pherc343-first-letters) publishes its mesh, its image and a
sidecar JSON with the image area, the mean and standard deviation of each model's map and the 18
boxes it put around letters, but not the maps. This script takes maps made here with
run_fl343_models.sh, builds the submission's ensemble (mean of the per-model z-scores at depth
+1, reading side, stretched as in the sidecar), and reports for every model and for the ensemble
how much higher the map is inside the boxes than elsewhere on the papyrus. With --image it also
correlates the ensemble with the ink overlay of the submission's own image.

    uv run python fl343_check.py --maps <dir> --render <render>.npy \
        --sidecar <winner>/outputs/concat_w047-w048_R5B2_z9500-11000.json \
        [--image <winner>/outputs/concat_w047-w048_R5B2_z9500-11000.png] --out <prefix>

Writes <prefix>_ensemble.png (the submission's area in reading orientation, drawn by the rule in
its sidecar: ink in light yellow over the dimmed CT layer) and <prefix>_boxes.png (the same with
the boxes drawn), and prints the scores as JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import rankdata

Image.MAX_IMAGE_PIXELS = None
MODEL_FILES = {  # sidecar name -> file name part
    "dnative": "dnative",
    "hecate": "hecate",
    "ink9-42": "hybrid_3d2d-seed42",
    "ink9-43": "hybrid_3d2d-seed43",
}


def auc(inside: np.ndarray, outside: np.ndarray) -> float:
    """Probability that a pixel inside the boxes is brighter than one outside (ties count half)."""
    ranks = rankdata(np.concatenate([inside, outside]))
    count = len(inside)
    return float((ranks[:count].sum() - count * (count + 1) / 2) / (count * len(outside)))


def published_overlay(png: Path, layer: np.ndarray) -> tuple[np.ndarray, list[int], float]:
    """The ink overlay of the submission's image over the area, its offset in the image, and how well its
    CT matches `layer` (our render of the layer it shows, in reading orientation) there.

    The image draws the area at full size inside margins, so the offset is found by matching. Ink is
    light yellow over a grey CT layer: red minus blue is zero on the CT and grows with the ink.
    """
    rgb = np.asarray(Image.open(png).convert("RGB"), dtype=np.float32)
    grey, ours = rgb[..., 2], layer.astype(np.float32)
    height, width = ours.shape

    def pooled(image: np.ndarray) -> np.ndarray:  # 8 x 8 means
        h, w = image.shape[0] // 8 * 8, image.shape[1] // 8 * 8
        return image[:h, :w].reshape(h // 8, 8, w // 8, 8).mean((1, 3))

    def best(big: np.ndarray, small: np.ndarray, rows: range, cols: range, step: int = 1) -> tuple[float, int, int]:
        """Best correlation of `small` with `big` read from (row, column) at every `step`th pixel."""
        h, w = small.shape
        return max(
            (float(np.corrcoef(big[y : y + h * step : step, x : x + w * step : step].ravel(), small.ravel())[0, 1]), y, x)
            for y in rows
            for x in cols
        )

    small_grey, small_ours = pooled(grey), pooled(ours)
    _, y8, x8 = best(
        small_grey,
        small_ours,
        range(small_grey.shape[0] - small_ours.shape[0] + 1),
        range(small_grey.shape[1] - small_ours.shape[1] + 1),
    )
    # Around the coarse match, every 4th pixel places the area to the pixel.
    rows = range(max(0, 8 * y8 - 8), min(grey.shape[0] - height, 8 * y8 + 8) + 1)
    cols = range(max(0, 8 * x8 - 8), min(grey.shape[1] - width, 8 * x8 + 8) + 1)
    match, y, x = best(grey, ours[::4, ::4], rows, cols, step=4)
    window = rgb[y : y + height, x : x + width]
    return np.clip(window[..., 0] - window[..., 2], 0, None), [y, x], match


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--maps", required=True, type=Path)
    ap.add_argument("--render", required=True, type=Path, help="the render as (layer, y, x) .npy")
    ap.add_argument("--sidecar", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--image", type=Path, help="the submission's image, to compare its ink overlay with ours")
    ap.add_argument("--direction", default="_reverse", help="file suffix of the maps: _reverse (reading side) or ''")
    args = ap.parse_args()
    sidecar = json.loads(args.sidecar.read_text())
    area = sidecar["area"]["raw_render_px"]
    rows, cols = slice(area["row0"], area["row1"]), slice(area["col0"], area["col1"])

    def reading(image: np.ndarray) -> np.ndarray:
        """The submission's area of a raw render-frame image, mirrored into reading orientation."""
        crop = image[rows, cols]
        if sidecar["orientation"]["flip_lr"]:
            crop = crop[:, ::-1]
        if sidecar["orientation"]["flip_ud"]:
            crop = crop[::-1]
        return crop

    render = np.load(args.render, mmap_mode="r")
    shape = render.shape[1:]
    papyrus = reading(np.asarray((render > 0).all(0)))
    boxes = np.zeros(papyrus.shape, dtype=bool)
    letters = sidecar["letters"]["letters_area_px"]
    for letter in letters:
        x0, y0, x1, y1 = (round(v) for v in letter["box"])
        boxes[y0:y1, x0:x1] = True
    inside, outside = boxes & papyrus, ~boxes & papyrus

    stats = sidecar["ink"]["normalisation"]["per_model"]
    scores, z_maps = {}, []
    for model, part in MODEL_FILES.items():
        found = sorted(args.maps.glob(f"*__{part}__*{args.direction}.tif")) + sorted(
            args.maps.glob(f"*__{part}__*{args.direction}.png")
        )
        if args.direction == "":
            found = [path for path in found if not path.stem.endswith("_reverse")]
        if not found:
            print(f"no map of {model}")
            continue
        image = Image.open(found[0])
        if image.size != (shape[1], shape[0]):  # Hecate works on a 9.6 um grid
            image = image.resize((shape[1], shape[0]), Image.BILINEAR)
        z = (reading(np.asarray(image, dtype=np.float32)) / 255 - stats[model]["mean"]) / stats[model]["sd"]
        z_maps.append(z)
        scores[model] = {
            "file": found[0].name,
            "mean_z_in_boxes": float(z[inside].mean()),
            "mean_z_elsewhere": float(z[outside].mean()),
            "auc": auc(z[inside][::7], z[outside][::29]),
        }
    ensemble = np.mean(z_maps, axis=0)
    scores["ensemble"] = {
        "models": len(z_maps),
        "mean_z_in_boxes": float(ensemble[inside].mean()),
        "mean_z_elsewhere": float(ensemble[outside].mean()),
        "auc": auc(ensemble[inside][::7], ensemble[outside][::29]),
        "boxes": len(letters),
        "boxes_brighter_than_elsewhere": int(
            sum(
                ensemble[round(b[1]) : round(b[3]), round(b[0]) : round(b[2])].mean() > ensemble[outside].mean()
                for b in (letter["box"] for letter in letters)
            )
        ),
    }
    stretch = sidecar["ink"]["normalisation"]["ensemble_stretch"]
    stretched = np.clip((ensemble - stretch["p1"]) / (stretch["p99_5"] - stretch["p1"]), 0, 1)
    # The submission's image shows ink from `threshold` to `full` of the stretch, over the dimmed CT layer.
    style = sidecar["overlay"]
    opacity = np.clip((stretched - style["threshold"]) / (style["full"] - style["threshold"]), 0, 1) * papyrus
    layer = reading(np.asarray(render[sidecar["ct"]["layers"][0]]))
    if args.image:
        overlay, offset, match = published_overlay(args.image, layer)
        scores["published_image"] = {
            "area_offset_yx": offset,
            "ct_layer": sidecar["ct"]["layers"][0],
            "ct_correlation": match,
            "ink_correlation": float(np.corrcoef(overlay[papyrus], opacity[papyrus])[0, 1]),
        }
    colour = np.array([float(v) for v in style["cmap"].split(":")[1].split(",")])
    alpha = (style["alpha"] * opacity)[..., None]
    shown = (layer[..., None] * style["ct_dim"] * (1 - alpha) + colour * alpha).round().astype(np.uint8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(shown).save(f"{args.out}_ensemble.png")
    drawn = Image.fromarray(shown)
    pen = ImageDraw.Draw(drawn)
    for letter in letters:
        pen.rectangle([round(v) for v in letter["box"]], outline=(0, 200, 255), width=4)
    drawn.save(f"{args.out}_boxes.png")
    print(json.dumps(scores, indent=1))


if __name__ == "__main__":
    main()
