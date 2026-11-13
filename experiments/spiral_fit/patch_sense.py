#!/usr/bin/env python3
"""Which way does a scroll wind? Read it off surface patches that run more than a turn.

`fit_spiral.py` needs the sense (`spiral_outward_sense` in spiral-scroll.json) and no score of a
fit shows a wrong one: a fit of the wrong sense follows the sheets locally and loses two windings
per turn in the gaps between its tracks. A patch that follows one sheet for more than a turn does
show it: where the patch overlaps itself, the sheet one turn further on lies outside or inside.

For every tifxyz patch in a folder this script counts the turns along the patch's grid, pairs
each node with the node one turn later at the same height, and takes the difference of their
distances from the umbilicus. Patches under --min-turns have no such pairs and are skipped.

    uv run python patch_sense.py --umbilicus <umbilicus.json> --patches <folder> [--out report.json]

Reported per patch: turns, pairs, the median difference and the share of pairs that are further
out. A patch votes when it has 50 pairs, four in five of them agree, and the difference is under
a quarter of the patch's radius (a patch that runs past the umbilicus sweeps the angle without
following a sheet round). A scroll whose patches are further out one turn of growing
atan2(y - cy, x - cx) later is `CW` in spiral-scroll.json's terms, as PHerc. Paris 4 is; the
other way `ACW`, as PHerc. 0332.
winding_sense.sh grows such patches from a scroll's surface tracks with villa's grow_track_graph.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ink_probe"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "winding_inference"))
from make_outer_shell import axis_xy  # noqa: E402
from mesh_windings import turns  # noqa: E402
from render_mesh import load_mesh  # noqa: E402

HEIGHT = 8.0  # voxels of height that count like 0.02 turns when a node looks for its partner


def one_turn_later(folder: Path, umbilicus: Path, min_turns: float) -> dict | None:
    """Radius one turn of growing angle later minus radius now, over the nodes that have such a partner."""
    xyz, valid, _ = load_mesh(folder)
    if valid.sum() < 200:
        return None
    centre = axis_xy(umbilicus, xyz[..., 2].ravel()).reshape(*valid.shape, 2)
    theta = np.arctan2(xyz[..., 1] - centre[..., 1], xyz[..., 0] - centre[..., 0])
    radius = np.hypot(xyz[..., 0] - centre[..., 0], xyz[..., 1] - centre[..., 1])
    turn, valid = turns(theta, valid)
    along = (theta / (2 * np.pi) + turn)[valid]
    if np.ptp(along) < min_turns:
        return None
    radius, z = radius[valid], xyz[..., 2][valid]
    where = np.stack([along, z * 0.02 / HEIGHT], axis=-1)
    distance, partner = cKDTree(where).query(where + [1.0, 0.0], distance_upper_bound=0.02)
    paired = np.isfinite(distance)
    if paired.sum() < 30:
        return None
    change = radius[partner[paired]] - radius[paired]
    share = float(np.mean(change > 0))
    clear = paired.sum() >= 50 and abs(share - 0.5) >= 0.3 and abs(np.median(change)) < 0.25 * np.median(radius)
    return {
        "turns": float(np.ptp(along)),
        "pairs": int(paired.sum()),
        "radius_one_turn_later_median": float(np.median(change)),
        "share_of_pairs_further_out": share,
        "votes": bool(clear),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--umbilicus", type=Path, required=True)
    ap.add_argument("--patches", type=Path, required=True, help="folder of tifxyz patches")
    ap.add_argument("--min-turns", type=float, default=1.01)
    ap.add_argument("--out", type=Path, help="write the report here as JSON")
    args = ap.parse_args()

    folders = sorted(p for p in args.patches.iterdir() if (p / "x.tif").exists())
    report = {}
    for folder in folders:
        row = one_turn_later(folder, args.umbilicus, args.min_turns)
        if row:
            report[folder.name] = row
            print(f"{folder.name[:40]:40s} turns {row['turns']:4.2f}  pairs {row['pairs']:6d}  radius one turn later "
                  f"{row['radius_one_turn_later_median']:+7.1f}  ({row['share_of_pairs_further_out']:.0%} of pairs further out)"
                  f"{'' if row['votes'] else '  no vote'}")
    medians = np.array([row["radius_one_turn_later_median"] for row in report.values() if row["votes"]])
    if not len(medians):
        raise SystemExit(f"none of {len(folders)} patches gives a vote ({len(report)} run {args.min_turns} turns)")
    outward = int((medians > 0).sum())
    summary = {
        "patches": len(folders),
        "patches_over_a_turn": len(report),
        "patches_voting": len(medians),
        "further_out_one_turn_later": outward,
        "further_in_one_turn_later": len(medians) - outward,
        "median_of_the_voting_patches_medians_voxels": float(np.median(medians)),
        "spiral_outward_sense": "CW" if outward > len(medians) - outward else "ACW" if outward < len(medians) - outward else None,
    }
    print(json.dumps(summary))
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "patches": report}, indent=1) + "\n")


if __name__ == "__main__":
    main()
