#!/usr/bin/env python3
"""Lay one rendered and inked winding out as the PHerc. 0306B First Letters submission's triage reads it.

That submission (github.com/SirProdigle/pherc0306b-first-letters, `discovery/`) found its line by scoring
windows of every winding of a spiral fit for rows of ink: `fl/remote/spiral/window_score.py`, then the
gates of `fl/scripts/tri_rank.py` and the review cards of `x_vast_plan/cards_from_tri.py`. Those scripts
read one folder per lane. This writes that folder from a 66-layer render (`render_mesh.py`,
`fetch_layers.py`) and the ink maps of both faces (`run_fl343_models.sh`), so the scripts run unchanged
(`triage_lane.sh`):

    <lane>/renders/<name>.zarr          the seven central layers, all the gates read of a render
    <lane>/maps/<name>/quality_tiles.npy  their tile rule (`window_score.render_qmask`) on the central 31 layers
    <lane>/maps/<name>/<name>_d0_fwd_z.npy, <name>_d0_reverse_z.npy
                                        fused maps as their `ens_view.py` makes them: per model a z-score over
                                        the rendered pixels of on-sheet tiles, the mean over models, 48 px of
                                        border blanked
    <lane>/maps/<name>/done_d0

    lane_layout.py <lane> <name> <stack.npy> --fwd <map.tif> ... --rev <map.tif> ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tifffile

TILE = 128  # pixels per side of a quality tile
ON_SHEET = 0.10  # a tile is on a sheet when its brightness profile through the layers varies by this share
BORDER = 48  # the models fire along a render's outer border


def quality_tiles(layers: np.ndarray, valid: np.ndarray, median: float) -> np.ndarray:
    """Per tile, the range of the mean brightness over the layers, relative to the render's median."""
    rows, columns = valid.shape[0] // TILE, valid.shape[1] // TILE
    scores = np.full((rows, columns), np.nan, np.float32)
    for i in range(rows):
        block = np.asarray(layers[:, i * TILE : (i + 1) * TILE]).astype(np.float32)
        for j in range(columns):
            inside = valid[i * TILE : (i + 1) * TILE, j * TILE : (j + 1) * TILE]
            if inside.mean() < 0.8:
                continue
            profile = block[:, :, j * TILE : (j + 1) * TILE][:, inside].mean(1)
            scores[i, j] = (profile.max() - profile.min()) / max(median, 1.0)
    return scores


def fused_map(files: list[str], valid: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Mean over models of each map's z-score, taken over the pixels in `reference`."""
    scores = []
    for file in files:
        ink = tifffile.imread(file).astype(np.float32)
        if ink.shape != valid.shape:
            ink = cv2.resize(ink, (valid.shape[1], valid.shape[0]), interpolation=cv2.INTER_LINEAR)
        sample = ink[reference]
        scores.append((ink - sample.mean()) / (sample.std() + 1e-6))
    fused = np.mean(scores, 0)
    fused[~valid] = 0
    fused[:BORDER] = 0
    fused[-BORDER:] = 0
    fused[:, :BORDER] = 0
    fused[:, -BORDER:] = 0
    return fused.astype(np.float16)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("lane", type=Path)
    ap.add_argument("name")
    ap.add_argument("stack", type=Path, help="the render as .npy, layers first (fetch_layers.py)")
    ap.add_argument("--fwd", nargs="+", default=[], help="ink maps of the layers as stored")
    ap.add_argument("--rev", nargs="+", default=[], help="ink maps of the layers reversed")
    args = ap.parse_args()

    stack = np.load(args.stack, mmap_mode="r")
    if stack.ndim != 3 or stack.shape[0] < 31:
        raise SystemExit(f"{args.stack}: expected layers x height x width with 31 layers or more, got {stack.shape}")
    centre = stack.shape[0] // 2
    middle = np.ascontiguousarray(stack[centre])
    valid = middle > 0
    if not valid.any():
        raise SystemExit(f"{args.stack}: the middle layer is empty")

    core = np.ascontiguousarray(stack[centre - 3 : centre + 4])
    render = args.lane / "renders" / f"{args.name}.zarr"
    (render / "0").mkdir(parents=True, exist_ok=True)
    (render / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (render / "0" / ".zarray").write_text(
        json.dumps(
            {
                "zarr_format": 2,
                "shape": list(core.shape),
                "chunks": list(core.shape),
                "dtype": "|u1",
                "compressor": None,
                "fill_value": 0,
                "order": "C",
                "filters": None,
                "dimension_separator": ".",
            }
        )
    )
    (render / "0" / "0.0.0").write_bytes(core.tobytes())

    maps = args.lane / "maps" / args.name
    maps.mkdir(parents=True, exist_ok=True)
    tiles = quality_tiles(stack[centre - 15 : centre + 16], valid, float(np.median(middle[valid])))
    np.save(maps / "quality_tiles.npy", tiles)
    on_sheet = np.zeros_like(valid)
    grid = np.kron(np.nan_to_num(tiles, nan=-1) >= ON_SHEET, np.ones((TILE, TILE), bool))
    on_sheet[: grid.shape[0], : grid.shape[1]] = grid[: valid.shape[0], : valid.shape[1]]
    reference = on_sheet & valid if (on_sheet & valid).sum() > 0.2 * valid.sum() else valid

    for side, files in (("fwd", args.fwd), ("reverse", args.rev)):
        if files:
            np.save(maps / f"{args.name}_d0_{side}_z.npy", fused_map(files, valid, reference))
    (maps / "done_d0").write_text("ok\n")
    measured = ~np.isnan(tiles)
    print(
        f"{args.name}: {valid.mean():.0%} of the grid rendered, "
        f"{(tiles[measured] >= ON_SHEET).mean() if measured.any() else 0:.0%} of its tiles on a sheet, "
        f"{len(args.fwd)} and {len(args.rev)} maps",
        flush=True,
    )


if __name__ == "__main__":
    main()
