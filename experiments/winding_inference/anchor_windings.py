#!/usr/bin/env python3
"""Score a fitted spiral on points whose winding number people fixed.

PHerc. 0332's spiral dataset (dl.ash2txt.org/datasets/spiral_datasets/PHerc0332/20251211183505)
has `anchors/anchors_{train,held}_records.json`: points on hand-annotated fibres, each with the
absolute winding number of its sheet (`wind_a`), 24,452 in all. A fit that counts as the people
did has a coordinate that differs from those numbers by one constant. This script reads the
fit's coordinate at the anchors inside its z-range and reports

- the fit's windings per true winding (a straight line through coordinate against number) in
  ranges of the true winding, and
- the median of coordinate minus number in rings of distance from the umbilicus.

    run.sh anchor_windings.py --checkpoint <run>/checkpoint_fitted.ckpt --umbilicus <umbilicus.json> \
        --anchors <anchors_train_records.json> --anchors <anchors_held_records.json> [--out report.json]

The anchors are in the fit's frame (x, y, z). Fits that take the anchors as input are not scored
fairly by the train records; the fits here take none.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from mesh_windings import winding_coordinate
from store_vs_fit import fitted_transform

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spiral_fit"))
from make_outer_shell import axis_xy  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--umbilicus", type=Path, required=True)
    ap.add_argument("--anchors", type=Path, action="append", required=True)
    ap.add_argument("--ranges", type=float, nargs="+", default=[0, 20, 40, 70], help="edges of the true-winding ranges")
    ap.add_argument("--ring", type=float, default=100.0, help="width of the rings round the umbilicus, voxels")
    ap.add_argument("--out", type=Path, help="write the report here as JSON")
    args = ap.parse_args()

    import torch
    from vesuvius.neural_tracing.winding_models import infer_winding_volume as villa

    sys.path.insert(0, villa._spiral_scripts_dir())
    records = [record for path in args.anchors for record in json.loads(path.read_text())]
    points = np.array([record["p"] for record in records], dtype=np.float64)
    number = np.array([record["wind_a"] for record in records], dtype=np.float64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.inference_mode():
        transform, dr_per_winding, (z_begin, z_end) = fitted_transform(args.checkpoint, args.umbilicus, device)
        inside = (points[:, 2] >= z_begin + 2) & (points[:, 2] < z_end - 2)
        points, number = points[inside], number[inside]
        if len(points) < 20:
            raise SystemExit("under 20 anchors in the fit's z-range")
        coordinate, _ = winding_coordinate(transform, dr_per_winding, points, device)
    centre = axis_xy(args.umbilicus, points[:, 2])
    radius = np.hypot(points[:, 0] - centre[:, 0], points[:, 1] - centre[:, 1])
    report = {"checkpoint": str(args.checkpoint), "anchors": len(points), "by_true_winding": {}, "by_radius": {}}
    for low, high in zip(args.ranges[:-1], args.ranges[1:]):
        chosen = (number >= low) & (number < high)
        if chosen.sum() >= 30:
            report["by_true_winding"][f"{low:g}-{high:g}"] = {
                "anchors": int(chosen.sum()),
                "fit_windings_per_true_winding": float(np.polyfit(number[chosen], coordinate[chosen], 1)[0]),
                "fit_minus_true_median": float(np.median(coordinate[chosen] - number[chosen])),
            }
    for low in np.arange(0, radius.max(), args.ring):
        chosen = (radius >= low) & (radius < low + args.ring)
        if chosen.sum() >= 15:
            report["by_radius"][f"{low:g}-{low + args.ring:g}"] = {
                "anchors": int(chosen.sum()),
                "true_winding_median": float(np.median(number[chosen])),
                "fit_coordinate_median": float(np.median(coordinate[chosen])),
                "fit_minus_true_median": float(np.median(coordinate[chosen] - number[chosen])),
            }
    print(json.dumps(report, indent=1))
    if args.out:
        args.out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
