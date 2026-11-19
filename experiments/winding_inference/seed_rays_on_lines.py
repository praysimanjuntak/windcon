#!/usr/bin/env python3
"""Seed rays for build_store.py through two successive turns of hand-traced fibres (PHerc. 0332's slice lines).

store_vs_lines.py tests a store wherever one of its rays happens to cross a fibre twice; a store seeded on a fit
gets a few hundred such pairs from 11,000 slabs. These seeds make every slab a test: a seed sits halfway between
the places where a fibre passes the same angle one turn apart, and its ray runs from the one place to the other.
The seeds are spread evenly over the distance between the two places (--per-bin seeds in each band of
--distances voxels), so that tightly and loosely wound places are tested alike.

    uv run python seed_rays_on_lines.py --umbilicus <umbilicus.json> --lines <folder of fibre .json> \
        [--line-scale 0.25] [--per-bin 40] --out seeds.npz

The seeds are in the voxels of --line-scale (level 2 of the 2.4 um scan with the default); build_store.py
--volume-scale reads them from another level.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spiral_fit"))
from make_outer_shell import axis_xy  # noqa: E402


def turn_pairs(xyz: np.ndarray, turn: np.ndarray, every: float) -> tuple[np.ndarray, np.ndarray]:
    """Places of one fibre at turn coordinates u and u + 1, for u every `every` turns where both exist."""
    order = np.argsort(turn)
    turn, xyz = turn[order], xyz[order]
    u = np.arange(turn[0], turn[-1] - 1.0, every)
    if len(u) == 0:
        return np.empty((0, 3)), np.empty((0, 3))
    inner = np.stack([np.interp(u, turn, xyz[:, k]) for k in range(3)], axis=1)
    outer = np.stack([np.interp(u + 1.0, turn, xyz[:, k]) for k in range(3)], axis=1)
    # Skip places where the trace has a gap: the nearest traced point must lie within a twentieth of a turn.
    traced = np.searchsorted(turn, u), np.searchsorted(turn, u + 1.0)
    near = np.ones(len(u), bool)
    for index, value in zip(traced, (u, u + 1.0), strict=True):
        index = np.clip(index, 1, len(turn) - 1)
        gap = np.minimum(np.abs(turn[index] - value), np.abs(turn[index - 1] - value))
        near &= gap < 0.05
    return inner[near], outer[near]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--umbilicus", type=Path, required=True)
    ap.add_argument("--lines", type=Path, required=True, help="folder of fibre json files with line_points")
    ap.add_argument("--line-scale", type=float, default=0.25, help="factor from fibre coordinates to seed voxels")
    ap.add_argument("--every", type=float, default=1 / 36, help="turns between candidate seeds along a fibre")
    ap.add_argument("--distances", type=float, nargs="+", default=[0, 8, 12, 16, 20, 25, 40],
                    help="edges of the distance bands (voxels between the two places)")
    ap.add_argument("--per-bin", type=int, default=40, help="seeds drawn from each distance band")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    inner, outer, fibre = [], [], []
    for path in sorted(args.lines.glob("*.json")):
        record = json.loads(path.read_text())
        xyz = np.asarray([p["position"] if isinstance(p, dict) else p for p in record["line_points"]], np.float64)
        xyz = xyz * args.line_scale
        centre = axis_xy(args.umbilicus, xyz[:, 2])
        turn = np.unwrap(np.arctan2(xyz[:, 1] - centre[:, 1], xyz[:, 0] - centre[:, 0])) / (2 * np.pi)
        a, b = turn_pairs(xyz, turn, args.every)
        inner.append(a)
        outer.append(b)
        fibre += [path.stem] * len(a)
    inner, outer, fibre = np.concatenate(inner), np.concatenate(outer), np.asarray(fibre)
    distance = np.linalg.norm(outer - inner, axis=1)

    rng = np.random.default_rng(args.seed)
    chosen = []
    for lo, hi in zip(args.distances[:-1], args.distances[1:], strict=True):
        band = np.flatnonzero((distance >= lo) & (distance < hi))
        chosen.append(rng.permutation(band)[: args.per_bin])
        print(f"{lo:4.0f}-{hi:<4.0f} voxels: {len(band):5d} places, {len(chosen[-1])} seeds")
    chosen = np.sort(np.concatenate(chosen))
    direction = (outer - inner)[chosen] / distance[chosen, None]
    np.savez(
        args.out,
        seed_xyz=(0.5 * (inner + outer))[chosen].astype(np.float32),
        direction_xyz=direction.astype(np.float32),
        seed_winding=np.zeros(len(chosen), np.int16),
        fibre=fibre[chosen],
        distance=distance[chosen].astype(np.float32),
    )
    print(f"{len(chosen)} seeds from {len(np.unique(fibre[chosen]))} fibres -> {args.out}")


if __name__ == "__main__":
    main()
