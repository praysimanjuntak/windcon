#!/usr/bin/env python3
"""Score a fitted spiral on hand-traced lines that each follow one sheet round the scroll.

PHerc. 0332's spiral dataset (dl.ash2txt.org/datasets/spiral_datasets/PHerc0332/20251211183505)
has 51 such lines in `slice_lines/`, each about three turns of one sheet, and scores fits with
them (`eval/slice_eval.py`): along a line the fit's winding number, with the turns taken out,
must stay constant. This script reads the same quantity from a checkpoint of the villa commit
windcon uses, and adds the fit's windings per turn along each line.

    run.sh line_windings.py --checkpoint <run>/checkpoint_fitted.ckpt --umbilicus <umbilicus.json> \
        --lines <folder of line JSONs> --scale 4 [--step 8] [--out report.json]

A line JSON holds `line_points` with positions (x, y, z) in voxels of the scan's level 0;
--scale divides them into the fit's frame (4 for a fit at level 2). As in slice_eval.py, lines
under one turn or with a radius that changes by more than 60 voxels per turn are left out, lines
need 200 points in the fit's z-range, and points more than three windings from their line's
median are counted and not scored. Reported: the share of points within a quarter and within
half a winding of their line's median, the median and 90th percentile of that distance, steps of
more than half a winding between consecutive points per 1,000 points, and the windings per turn.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from store_vs_fit import CHUNK, fitted_transform


def load_lines(folder: Path, umbilicus: Path, scale: float, step: int) -> list[tuple[str, np.ndarray]]:
    """(name, points z y x in the fit's frame) for every line that is not degenerate."""
    axis = sorted((p["z"], p["x"], p["y"]) for p in json.loads(umbilicus.read_text())["control_points"])
    zs, xs, ys = (np.array(column, dtype=np.float64) for column in zip(*axis))
    lines, dropped = [], []
    for path in sorted(folder.glob("*.json")):
        points = json.loads(path.read_text())["line_points"]
        xyz = np.array([p["position"] if isinstance(p, dict) else p for p in points], dtype=np.float64)[::step] / scale
        dx, dy = xyz[:, 0] - np.interp(xyz[:, 2], zs, xs), xyz[:, 1] - np.interp(xyz[:, 2], zs, ys)
        angle, radius = np.unwrap(np.arctan2(dy, dx)), np.hypot(dx, dy)
        turns = np.ptp(angle) / (2 * np.pi)
        drift = abs(np.polyfit(angle / (2 * np.pi), radius, 1)[0]) if turns > 0.1 else np.inf
        if turns < 1 or drift > 60:
            dropped.append(path.stem)
            continue
        lines.append((path.stem, np.ascontiguousarray(xyz[:, ::-1])))
    if dropped:
        print(f"left out {len(dropped)} degenerate lines: {', '.join(dropped)}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--umbilicus", type=Path, required=True)
    ap.add_argument("--lines", type=Path, required=True, help="folder of line JSONs")
    ap.add_argument("--scale", type=float, default=1.0, help="level-0 voxels per voxel of the fit's frame")
    ap.add_argument("--step", type=int, default=8, help="use every Nth point of a line")
    ap.add_argument("--out", type=Path, help="write the report here as JSON")
    args = ap.parse_args()

    import torch
    from vesuvius.neural_tracing.winding_models import infer_winding_volume as villa

    sys.path.insert(0, villa._spiral_scripts_dir())
    from sample_spiral import get_theta_and_radii

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lines = load_lines(args.lines, args.umbilicus, args.scale, args.step)
    rows, pooled, steps, off_model = {}, [], 0, 0
    with torch.inference_mode():
        transform, dr_per_winding, (z_begin, z_end) = fitted_transform(args.checkpoint, args.umbilicus, device)
        for name, zyx in lines:
            zyx = zyx[(zyx[:, 0] >= z_begin + 2) & (zyx[:, 0] < z_end - 2)]
            if len(zyx) < 200:
                continue
            points = torch.from_numpy(zyx).to(device, torch.float32)
            spiral = torch.cat([transform(points[i : i + CHUNK]) for i in range(0, len(points), CHUNK)])
            theta, _, shifted = get_theta_and_radii(spiral[:, 1:], dr_per_winding)
            theta = theta.cpu().numpy().astype(np.float64)
            along = np.unwrap(theta) / (2 * np.pi)  # turns round the scroll along the line
            # The winding number grows by one per turn along a sheet: take the turns out.
            winding = (shifted / dr_per_winding).cpu().numpy().astype(np.float64) - np.rint(along - theta / (2 * np.pi))
            residual = winding - np.median(winding)
            scored = np.abs(residual) <= 3
            off_model += int((~scored).sum())
            residual, winding, along = residual[scored], winding[scored], along[scored]
            jumps = int(np.sum(np.abs(np.diff(winding)) > 0.5))
            rows[name] = {
                "z": float(zyx[:, 0].mean()),
                "points": len(residual),
                "turns": float(np.ptp(along)),
                "within_a_quarter": float(np.mean(np.abs(residual) < 0.25)),
                "within_half": float(np.mean(np.abs(residual) < 0.5)),
                "steps_over_half_a_winding": jumps,
                "fit_windings_per_turn": float(1 + np.polyfit(along, winding, 1)[0]),
            }
            pooled.append(np.abs(residual))
            steps += jumps
    if not rows:
        raise SystemExit("no line has 200 points in the fit's z-range")
    pooled = np.concatenate(pooled)
    per_turn = np.array([row["fit_windings_per_turn"] for row in rows.values()])
    summary = {
        "lines": len(rows),
        "points": len(pooled),
        "turns": float(sum(row["turns"] for row in rows.values())),
        "within_a_quarter": float(np.mean(pooled < 0.25)),
        "within_half": float(np.mean(pooled < 0.5)),
        "distance_median": float(np.median(pooled)),
        "distance_p90": float(np.percentile(pooled, 90)),
        "steps_over_half_a_winding_per_1000_points": 1000 * steps / len(pooled),
        "line_median_within_a_quarter": float(np.median([row["within_a_quarter"] for row in rows.values()])),
        "fit_windings_per_turn_median": float(np.median(per_turn)),
        "fit_windings_per_turn_10th_90th": [float(np.percentile(per_turn, q)) for q in (10, 90)],
        "points_over_three_windings_off": off_model,
    }
    print(json.dumps(summary))
    if args.out:
        args.out.write_text(json.dumps({"checkpoint": str(args.checkpoint), "all": summary, "lines": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
