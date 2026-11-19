#!/usr/bin/env python3
"""A ray store against hand-traced fibres that run round the scroll, such as PHerc. 0332's slice lines.

A horizontal fibre follows one sheet of papyrus round several turns. A ray that crosses it twice, at two
successive turns, crosses one winding between the two places: the store should count one level there. Every
pair of successive crossings of the same fibre by the same ray is such a test, with no winding number needed.
The store's level at a place on a ray is interpolated between its crossings; the count of a pair is the
difference of the two levels, rounded.

    run.sh store_vs_lines.py --store <store> --umbilicus <umbilicus.json> --lines <folder of fibre .json> \
        [--line-scale 0.25] [--tolerance 2] [--out report.json]

Fibre points are scaled by --line-scale into the store's voxels (0.25 for lines traced on a 2.4 um scan and a
store made on its level 2).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np
from build_store import load_store
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spiral_fit"))
from make_outer_shell import axis_xy  # noqa: E402

RINGS = (0, 300, 600, 900, 1200, 1500, 2000, 3000)
DISTANCES = (0, 8, 12, 16, 20, 25, 40)


def load_fibres(folder: Path, scale: float, umbilicus: Path, z_range: tuple[float, float]):
    """Points (x, y, z) of every fibre in the z-range, its index, and its turn coordinate (unwrapped angle / 2 pi)."""
    points, owner, turn = [], [], []
    names = []
    for path in sorted(folder.glob("*.json")):
        record = json.loads(path.read_text())
        xyz = np.asarray([p["position"] if isinstance(p, dict) else p for p in record["line_points"]], np.float64)
        xyz = xyz * scale
        keep = (xyz[:, 2] >= z_range[0]) & (xyz[:, 2] <= z_range[1])
        if keep.sum() < 50:
            continue
        centre = axis_xy(umbilicus, xyz[:, 2])
        angle = np.unwrap(np.arctan2(xyz[:, 1] - centre[:, 1], xyz[:, 0] - centre[:, 0]))
        names.append(path.stem)
        points.append(xyz[keep])
        owner.append(np.full(keep.sum(), len(names) - 1))
        turn.append(angle[keep] / (2 * np.pi))
    return np.concatenate(points), np.concatenate(owner), np.concatenate(turn), names


def level_at(t: np.ndarray, crossing_t: np.ndarray, crossing_level: np.ndarray) -> np.ndarray:
    """The store's level at positions t of one ray, linear between its crossings, nan outside them."""
    return np.interp(t, crossing_t, crossing_level.astype(np.float64), left=np.nan, right=np.nan)


def summarise(counted: np.ndarray, truth: np.ndarray) -> dict:
    one = truth == 1
    c = counted[one]
    return {
        "pairs": len(counted),
        "pairs_one_winding_apart": int(one.sum()),
        "windings_per_true_winding": float(counted.sum() / max(truth.sum(), 1)),
        "of_those_counted_as_one": float(np.mean(c == 1)) if len(c) else None,
        "of_those_counted_as_none": float(np.mean(c <= 0)) if len(c) else None,
        "of_those_counted_as_two_or_more": float(np.mean(c >= 2)) if len(c) else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--umbilicus", type=Path, required=True)
    ap.add_argument("--lines", type=Path, required=True, help="folder of fibre json files with line_points")
    ap.add_argument("--line-scale", type=float, default=0.25, help="factor from fibre coordinates to store voxels")
    ap.add_argument("--tolerance", type=float, default=2.0, help="voxels between a fibre point and a ray")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--pairs-out", type=Path, help="write every pair (ray source, fibre, turns, counts) as npz")
    args = ap.parse_args()

    rays, manifest = load_store(args.store)
    origin = rays["ray_origin_zyx"][:, ::-1].astype(np.float64)
    step = rays["ray_step_zyx"][:, ::-1].astype(np.float64)
    offsets = np.concatenate([[0], np.cumsum(rays["crossing_counts"])])
    crossing_t, crossing_level = rays["crossing_t"], rays["crossing_level"]
    first = crossing_t[offsets[:-1]]
    last = crossing_t[offsets[1:] - 1]

    # Sample every ray once per voxel over the span of its crossings.
    lengths = np.ceil(last - first).astype(np.int64) + 1
    ray_of = np.repeat(np.arange(len(origin)), lengths)
    t = np.repeat(first, lengths) + (np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths))
    samples = origin[ray_of] + t[:, None] * step[ray_of]
    z_range = (samples[:, 2].min(), samples[:, 2].max())
    fibre_xyz, fibre_of, fibre_turn, names = load_fibres(args.lines, args.line_scale, args.umbilicus, z_range)
    print(f"{len(origin):,} rays ({len(samples):,} samples, z {z_range[0]:.0f}-{z_range[1]:.0f}); "
          f"{len(names)} fibres, {len(fibre_xyz):,} points", flush=True)

    tree = cKDTree(samples)
    near = tree.query_ball_point(fibre_xyz, r=args.tolerance, workers=-1)
    hits = defaultdict(list)  # (ray, fibre) -> [(t, turn)]
    for point, found in enumerate(near):
        for sample in found:
            hits[(int(ray_of[sample]), int(fibre_of[point]))].append((t[sample], fibre_turn[point]))

    counted, truth, ring_radius, distance, which, keys, model_spacing, places = [], [], [], [], [], [], [], []
    for (ray, fibre), found in hits.items():
        found = np.asarray(found)
        order = np.argsort(found[:, 1])
        found = found[order]
        # one crossing per turn: split where the fibre's turn coordinate jumps by more than a quarter turn
        breaks = np.flatnonzero(np.diff(found[:, 1]) > 0.25) + 1
        groups = np.split(found, breaks)
        crossings = np.array([[np.median(g[:, 0]), np.median(g[:, 1])] for g in groups])
        if len(crossings) < 2:
            continue
        crossings = crossings[np.argsort(crossings[:, 0])]
        span = slice(offsets[ray], offsets[ray + 1])
        level = level_at(crossings[:, 0], crossing_t[span], crossing_level[span])
        for a in range(len(crossings) - 1):
            b = a + 1
            if np.isnan(level[a]) or np.isnan(level[b]):
                continue
            true_windings = round(abs(crossings[b, 1] - crossings[a, 1]))
            if true_windings < 1:
                continue
            counted.append(round(abs(level[b] - level[a])))
            truth.append(true_windings)
            mid = origin[ray] + 0.5 * (crossings[a, 0] + crossings[b, 0]) * step[ray]
            centre = axis_xy(args.umbilicus, np.array([mid[2]]))[0]
            ring_radius.append(float(np.hypot(mid[0] - centre[0], mid[1] - centre[1])))
            distance.append(float(abs(crossings[b, 0] - crossings[a, 0]) * np.linalg.norm(step[ray]) / true_windings))
            which.append(ray)
            places.append((crossings[a, 0], crossings[b, 0]))
            # the model's own spacing of crossings round the pair, in samples of this store
            inside = crossing_t[span][(crossing_t[span] >= crossings[a, 0] - 12) & (crossing_t[span] <= crossings[b, 0] + 12)]
            model_spacing.append(float(np.median(np.diff(inside))) if len(inside) >= 2 else np.nan)
            keys.append((*rays["ray_source"][ray], fibre, round(crossings[a, 1]), round(crossings[b, 1])))
    counted, truth = np.asarray(counted), np.asarray(truth)
    ring_radius, distance = np.asarray(ring_radius), np.asarray(distance)
    report = {
        "store": str(args.store),
        "sample_step_voxels": manifest.get("sample_step_voxels"),
        "fibres": names,
        "rays_with_pairs": len(set(which)),
        "all": summarise(counted, truth),
        "by_ring_voxels": {},
        "by_sheet_distance_voxels": {},
    }
    for lo, hi in pairwise(RINGS):
        m = (ring_radius >= lo) & (ring_radius < hi)
        if m.sum():
            report["by_ring_voxels"][f"{lo}-{hi}"] = summarise(counted[m], truth[m])
    for lo, hi in zip(DISTANCES[:-1], DISTANCES[1:] + (np.inf,)):
        m = (distance >= lo) & (distance < hi)
        if m.sum():
            report["by_sheet_distance_voxels"][f"{lo}-{hi}"] = summarise(counted[m], truth[m])

    def line(name, s):
        if not s["pairs_one_winding_apart"]:
            return
        print(f"  {name:10s} pairs {s['pairs']:6d}  windings per true winding {s['windings_per_true_winding']:.3f}  "
              f"one {s['of_those_counted_as_one']:.1%}  none {s['of_those_counted_as_none']:.1%}  "
              f"two+ {s['of_those_counted_as_two_or_more']:.1%}")

    print(f"successive crossings of the same fibre by the same ray ({report['rays_with_pairs']:,} rays):")
    line("all", report["all"])
    print(" by distance from the umbilicus, voxels:")
    for k, s in report["by_ring_voxels"].items():
        line(k, s)
    print(" by distance between the two crossings per winding, voxels:")
    for k, s in report["by_sheet_distance_voxels"].items():
        line(k, s)
    if args.out:
        args.out.write_text(json.dumps(report, indent=1) + "\n")
    if args.pairs_out:
        np.savez(
            args.pairs_out,
            key=np.asarray(keys, dtype=np.int64).reshape(-1, 6),
            counted=counted,
            truth=truth,
            ring_radius=ring_radius,
            distance=distance,
            model_spacing_samples=np.asarray(model_spacing),
            ray=np.asarray(which, dtype=np.int64),
            t=np.asarray(places, dtype=np.float64).reshape(-1, 2),
            fibres=np.asarray(names),
        )


if __name__ == "__main__":
    main()
