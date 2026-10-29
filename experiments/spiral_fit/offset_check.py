#!/usr/bin/env python3
"""Signed offset of point collections from nearby verified patch surfaces (diagnostic).

For every point over a patch surface (within --max-normal voxels across it and about half a grid
cell along it), report d = signed offset along the patch normal facing the scroll axis: d > 0 means
the point lies on the axis side of the patch surface.

    uv run python offset_check.py --dataset data/PHercParis4 --pcl pcls/auto.json --z 10025 10425 10825
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_surfaces import PatchSurfaces  # noqa: E402


def load_points(pcl: Path, zs: set[int]) -> np.ndarray:
    doc = json.loads(pcl.read_text())
    pts = [p["p"] for c in doc["collections"].values() for p in c["points"].values()
           if round(p["p"][2]) in zs]
    return np.asarray(pts, dtype=np.float64).reshape(-1, 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--pcl", required=True, type=Path, nargs="+")
    ap.add_argument("--z", required=True, type=int, nargs="+")
    ap.add_argument("--max-normal", type=float, default=8.0)
    args = ap.parse_args()
    surfaces = PatchSurfaces(args.dataset, min(args.z) - 30, max(args.z) + 30)
    print(f"{len(surfaces.names)} patches, {len(surfaces.V):,} vertices")
    for pcl in args.pcl:
        P = load_points(pcl, set(args.z))
        label, normal, _ = surfaces.nearest(P, args.max_normal)
        o = normal[label >= 0]
        q = np.percentile(o, [10, 25, 50, 75, 90]) if len(o) else [np.nan] * 5
        print(f"{pcl.name}: {len(P):,} points in the slices, {len(o):,} over a patch "
              f"({len(o) / max(len(P), 1):.0%}); signed offset (+ = toward axis) "
              f"p10 {q[0]:.1f}  p25 {q[1]:.1f}  median {q[2]:.1f}  p75 {q[3]:.1f}  p90 {q[4]:.1f}")


if __name__ == "__main__":
    main()
