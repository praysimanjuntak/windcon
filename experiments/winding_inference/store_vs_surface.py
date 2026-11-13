#!/usr/bin/env python3
"""Does a winding-inference store count the sheets a surface prediction shows?

The winding model was trained on PHerc. Paris 4. On another scan nobody has counted the windings,
but the scan's surface prediction marks its sheets, and that is a second, independent reading of
the same voxels. Along each ray of the store this script counts the model's windings between the
ray's first and last crossing, and the predicted sheets on the same stretch (the stretches of the
ray inside the prediction whose centres lie on it). It reports the two counts per 100 voxels and
their ratio, for all rays and for groups of rays by how far apart the predicted sheets are.

    uv run python store_vs_surface.py --store <store> --surface <URL of the prediction's level> \
        [--rays 2000] [--z-range LO HI] [--out report.json]

The prediction must be in the voxels of the volume the store was built on. A ratio under 1 where
sheets are close means the model merges sheets there, over 1 that it splits them or that the
prediction misses some.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from build_store import load_store

from windcon.zarr_io import ZarrArray

MARGIN = 8  # samples read beyond the first and last crossing, so that a sheet at either end is seen whole
GROUPS = (0, 12, 16, 20, 28, np.inf)  # voxels between predicted sheets on a ray (median gap)


def count_ray(surface: ZarrArray, origin: np.ndarray, step: np.ndarray, t: np.ndarray) -> tuple[int, int, float, float]:
    """Model windings, predicted sheets, length in voxels and the prediction's median sheet gap in voxels
    (nan under two sheets) between the first and last crossing of one ray."""
    samples = np.arange(int(np.floor(t[0])) - MARGIN, int(np.ceil(t[-1])) + MARGIN + 1)
    line = np.rint(origin + samples[:, None] * step).astype(int)  # z, y, x
    low, high = line.min(0), line.max(0) + 1
    block = surface.read(tuple((int(a), int(b)) for a, b in zip(low, high)))
    inside = block[line[:, 0] - low[0], line[:, 1] - low[1], line[:, 2] - low[2]] > 0
    edges = np.diff(np.concatenate([[0], inside.astype(np.int8), [0]]))
    centres = samples[0] + (np.flatnonzero(edges == 1) + np.flatnonzero(edges == -1) - 1) / 2
    centres = centres[(centres >= t[0]) & (centres < t[-1])]
    pitch = float(np.linalg.norm(step))
    gap = float(np.median(np.diff(centres)) * pitch) if len(centres) >= 2 else float("nan")
    return len(t) - 1, len(centres), float((t[-1] - t[0]) * pitch), gap


def summary(rows: np.ndarray) -> dict:
    model, sheets, length = rows[:, 0], rows[:, 1], rows[:, 2]
    both = sheets > 0
    return {
        "rays": len(rows),
        "model_windings_per_100_voxels": float(100 * model.sum() / length.sum()),
        "predicted_sheets_per_100_voxels": float(100 * sheets.sum() / length.sum()),
        "model_per_predicted_sheet": float(model.sum() / sheets.sum()) if sheets.sum() else None,
        "per_ray_median": float(np.median(model[both] / sheets[both])) if both.any() else None,
        "rays_with_the_same_count": float(np.mean(model == sheets)),
        "rays_within_one": float(np.mean(np.abs(model - sheets) <= 1)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--surface", required=True, help="URL of the surface prediction's level in the store's voxels")
    ap.add_argument("--rays", type=int, default=2000, help="rays drawn at random")
    ap.add_argument("--z-range", type=float, nargs=2, metavar=("LO", "HI"), help="only rays whose middle crossing is in here")
    ap.add_argument("--out", type=Path, help="write the report here as JSON")
    args = ap.parse_args()

    rays, _ = load_store(args.store)
    surface = ZarrArray(args.surface)
    ends = np.cumsum(rays["crossing_counts"])
    begins = ends - rays["crossing_counts"]
    usable = rays["crossing_counts"] >= 3
    if args.z_range:
        middle = rays["crossing_t"][np.where(usable, (begins + ends) // 2, 0)]  # a ray's middle crossing
        z = rays["ray_origin_zyx"][:, 0] + middle * rays["ray_step_zyx"][:, 0]
        usable &= (z >= args.z_range[0]) & (z < args.z_range[1])
    chosen = np.random.default_rng(0).choice(np.flatnonzero(usable), min(args.rays, int(usable.sum())), replace=False)
    rows = np.array(
        [
            count_ray(surface, rays["ray_origin_zyx"][i], rays["ray_step_zyx"][i], rays["crossing_t"][begins[i] : ends[i]])
            for i in chosen
        ]
    )
    report = {"store": str(args.store), "surface": args.surface, "all": summary(rows), "by_predicted_sheet_gap": {}}
    print(f"all {len(rows):,} rays: {json.dumps(report['all'])}")
    for low, high in itertools.pairwise(GROUPS):
        group = rows[(rows[:, 3] >= low) & (rows[:, 3] < high)]
        if len(group):
            name = f"{low:g}-{high:g} voxels" if np.isfinite(high) else f"{low:g} voxels and over"
            report["by_predicted_sheet_gap"][name] = summary(group)
            print(f"predicted sheets {name} apart: {json.dumps(report['by_predicted_sheet_gap'][name])}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
