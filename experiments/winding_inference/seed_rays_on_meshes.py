#!/usr/bin/env python3
"""Seed rays for build_store.py on traced sheets instead of on a fit's windings.

For the check of the winding model against sheets that people traced (store_vs_meshes.py) the rays
should pass those sheets, and need no fit. This writes the seed file `build_store.py --rays`
reads: every --every-th node of each mesh inside --z, with the ray along the mesh normal turned
away from the umbilicus.

    uv run python seed_rays_on_meshes.py --umbilicus <umbilicus.json> --z 4600 5900 --every 6 \
        --mesh <.../w026_...> --mesh <.../w031_...> --out seeds.npz

A mesh named w<number>... gives its seeds that winding number, others 0. The normal is taken
between nodes --normal-span voxels apart, so that a 4-voxel grid gives the same normals as a
20-voxel one.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ink_probe"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spiral_fit"))
from make_outer_shell import axis_xy  # noqa: E402
from render_mesh import load_mesh  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--umbilicus", type=Path, required=True)
    ap.add_argument("--mesh", type=Path, action="append", required=True)
    ap.add_argument("--z", type=float, nargs=2, required=True, metavar=("BEGIN", "END"))
    ap.add_argument("--every", type=int, default=6, help="take every Nth node along both grid axes")
    ap.add_argument("--normal-span", type=float, default=20.0, help="voxels between the nodes a normal is taken from")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    seeds, directions, windings = [], [], []
    for folder in args.mesh:
        xyz, valid, nodes_per_voxel = load_mesh(folder)
        step = max(1, round(args.normal_span * nodes_per_voxel))
        along_columns = np.roll(xyz, -step, 1) - np.roll(xyz, step, 1)
        along_rows = np.roll(xyz, -step, 0) - np.roll(xyz, step, 0)
        whole = valid & np.roll(valid, -step, 1) & np.roll(valid, step, 1) & np.roll(valid, -step, 0) & np.roll(valid, step, 0)
        whole[:step] = whole[-step:] = False
        whole[:, :step] = whole[:, -step:] = False
        normal = np.cross(along_columns, along_rows)
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-9)
        pick = np.zeros_like(valid)
        pick[:: args.every, :: args.every] = True
        pick &= whole & (xyz[..., 2] >= args.z[0]) & (xyz[..., 2] < args.z[1])
        points, normals = xyz[pick], normal[pick]
        centre = axis_xy(args.umbilicus, points[:, 2])
        outward = np.sign((points[:, 0] - centre[:, 0]) * normals[:, 0] + (points[:, 1] - centre[:, 1]) * normals[:, 1])
        normals *= np.where(outward == 0, 1, outward)[:, None]
        number = re.match(r"w(\d+)", folder.name)
        seeds.append(points)
        directions.append(normals)
        windings.append(np.full(len(points), int(number.group(1)) if number else 0))
        print(f"{folder.name}: {len(points)} seeds")
    np.savez(
        args.out,
        seed_xyz=np.concatenate(seeds).astype(np.float32),
        direction_xyz=np.concatenate(directions).astype(np.float32),
        seed_winding=np.concatenate(windings).astype(np.int16),
    )
    print(f"{sum(len(s) for s in seeds)} seeds in {args.out}")


if __name__ == "__main__":
    main()
