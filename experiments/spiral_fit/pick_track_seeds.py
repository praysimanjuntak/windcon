#!/usr/bin/env python3
"""Seed tracks for villa's grow_track_graph.py inside a z-window.

The grower draws its random seeds from the longest 30 % of all tracks of the scroll and has no z
filter. This prints the first --count rows of that same shuffled order whose track lies entirely
inside the window, for the grower's --seeds. Rows in --exclude-patches (folders named
band-seed<row>-*, as the grower writes them) are skipped. With --umbilicus and --radius only
tracks whose median distance from the umbilicus lies in that range are taken (winding_sense.sh
seeds near the axis, where a patch passes a full turn soonest).

Run from villa/spiral-fitting with its environment:
    .venv/bin/python <this> tracks/<name>.dbm.vctracks --z 9900 10900 --count 8000 > seeds.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from grow_track_graph import PackedTracks, random_top_arclength_rows  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tracks", type=Path, help="packed .vctracks directory")
    ap.add_argument("--z", type=int, nargs=2, required=True, metavar=("BEGIN", "END"))
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--random-seed", type=int, default=0, help="the grower's --random-seed")
    ap.add_argument("--exclude-patches", type=Path, help="folder of band-seed<row>-* patches to skip")
    ap.add_argument("--umbilicus", type=Path, help="umbilicus.json, for --radius")
    ap.add_argument("--radius", type=float, nargs=2, metavar=("INNER", "OUTER"),
                    help="keep tracks whose median distance from the umbilicus is in this range, voxels")
    args = ap.parse_args()
    tracks = PackedTracks(args.tracks)
    excluded = set()
    if args.exclude_patches:
        excluded = {int(m.group(1)) for p in args.exclude_patches.iterdir()
                    if (m := re.match(r"band-seed(\d+)", p.name))}
    rows, cutoff, _, top = random_top_arclength_rows(tracks, random_seed=args.random_seed, excluded=excluded)
    z = np.memmap(args.tracks / "z_bounds.i32", mode="r", dtype="<i4", shape=(tracks.track_count, 2))
    inside = (z[rows, 0] >= args.z[0]) & (z[rows, 1] < args.z[1])
    picked = rows[inside]
    if args.radius:
        import json

        axis = sorted((q["z"], q["x"], q["y"]) for q in json.loads(args.umbilicus.read_text())["control_points"])
        zs, xs, ys = (np.array(column, dtype=np.float64) for column in zip(*axis))
        offsets = np.memmap(args.tracks / "offsets.i64", mode="r", dtype="<i8")
        points = np.memmap(args.tracks / "coordinates.i32", mode="r", dtype="<i4").reshape(-1, 3)  # z, y, x
        near = []
        for row in picked:
            zyx = np.asarray(points[offsets[row] : offsets[row + 1]], dtype=np.float64)
            radius = np.hypot(zyx[:, 1] - np.interp(zyx[:, 0], zs, ys), zyx[:, 2] - np.interp(zyx[:, 0], zs, xs))
            if args.radius[0] <= np.median(radius) < args.radius[1]:
                near.append(row)
                if len(near) == args.count:
                    break
        picked = np.array(near, dtype=np.int64)
    picked = picked[: args.count]
    print(f"{top} tracks at least {cutoff:.0f} voxels long, {int(inside.sum())} inside z {args.z}, "
          f"{len(picked)} picked", file=sys.stderr)
    print(*picked.tolist())


if __name__ == "__main__":
    main()
