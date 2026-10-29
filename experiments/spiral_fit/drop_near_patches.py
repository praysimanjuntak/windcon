#!/usr/bin/env python3
"""Remove point-collection points that lie on or near a verified patch surface.

The fitter attaches a point to a patch surface within 2.5 voxels (pcl_link_distance_tolerance); a
collection attached to two patches becomes a same-winding link between them. Dropping every point
within --max-normal voxels of a patch surface (a margin over 2.5, since offsets are measured against
the nearest patch vertex's tangent plane) leaves the automatic paths only where no verified patch
lies, so they fill gaps without tying patches together. Each collection is split at removed points;
runs with fewer than 2 points are discarded.

    uv run python drop_near_patches.py --pcl pcls/auto_chains.json --dataset data/PHercParis4 \
        --out pcls/auto_chains_free.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_surfaces import PatchSurfaces  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pcl", required=True, type=Path)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-normal", type=float, default=4.0)
    ap.add_argument("--max-lateral", type=float, default=16.0)
    args = ap.parse_args()
    doc = json.loads(args.pcl.read_text())
    cols = [doc["collections"][k] for k in sorted(doc["collections"], key=int)]
    items = [sorted(c["points"].items(), key=lambda kv: int(kv[0])) for c in cols]
    xyz = np.array([pt["p"] for its in items for _, pt in its], dtype=np.float64)
    surfaces = PatchSurfaces(args.dataset, xyz[:, 2].min() - 30, xyz[:, 2].max() + 30)
    label, _, _ = surfaces.nearest(xyz, args.max_normal, args.max_lateral)
    near = label >= 0

    out, k, next_point = {}, 0, 0
    for col, its in zip(cols, items):
        flags = near[k:k + len(its)]
        k += len(its)
        run: list = []
        runs = []
        for (_, pt), drop in zip(its, flags):
            if drop:
                runs.append(run)
                run = []
            else:
                run.append(pt)
        runs.append(run)
        for n, r in enumerate(x for x in runs if len(x) >= 2):
            points = {}
            for pt in r:
                points[str(next_point)] = pt
                next_point += 1
            out[str(len(out))] = {k2: v for k2, v in col.items() if k2 != "points"} | {
                "name": f"{col['name']}_r{n}", "points": points}
    args.out.write_text(json.dumps({"vc_pointcollections_json_version": "1", "collections": out}))
    print(json.dumps({"collections_in": len(cols), "points_in": len(xyz),
                      "points_near_patch": int(near.sum()),
                      "collections_out": len(out), "points_out": next_point}))


if __name__ == "__main__":
    main()
