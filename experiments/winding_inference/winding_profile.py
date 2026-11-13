#!/usr/bin/env python3
"""A fit's winding coordinate against distance from the umbilicus.

At one height (default: the middle of the fit's z-range) the coordinate is read on circles round
the umbilicus, and its median over the directions is printed per radius. The difference between
two radii is the number of windings the fit has between them; divided into the distance it is the
fit's pitch there. Two fits of one scroll can be compared without any annotation.

    run.sh winding_profile.py --umbilicus <umbilicus.json> --checkpoint <run>/checkpoint_fitted.ckpt \
        [--checkpoint <another>] [--radii 200 400 ...] [--out report.json]
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
    ap.add_argument("--checkpoint", type=Path, action="append", required=True)
    ap.add_argument("--umbilicus", type=Path, required=True)
    ap.add_argument("--radii", type=float, nargs="+", default=list(range(200, 3000, 200)))
    ap.add_argument("--z", type=float, help="height of the circles (default: the middle of each fit's range)")
    ap.add_argument("--directions", type=int, default=72)
    ap.add_argument("--out", type=Path, help="write the report here as JSON")
    args = ap.parse_args()

    import torch
    from vesuvius.neural_tracing.winding_models import infer_winding_volume as villa

    sys.path.insert(0, villa._spiral_scripts_dir())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    angles = np.arange(args.directions) * 2 * np.pi / args.directions
    report = {"radii": args.radii, "fits": {}}
    print("radius".ljust(40) + " ".join(f"{radius:7.0f}" for radius in args.radii))
    with torch.inference_mode():
        for checkpoint in args.checkpoint:
            transform, dr_per_winding, (z_begin, z_end) = fitted_transform(checkpoint, args.umbilicus, device)
            z = args.z if args.z is not None else 0.5 * (z_begin + z_end)
            centre = axis_xy(args.umbilicus, np.array([z]))[0]
            medians = []
            for radius in args.radii:
                circle = np.stack(
                    [centre[0] + radius * np.cos(angles), centre[1] + radius * np.sin(angles), np.full(len(angles), z)], axis=-1
                )
                winding, _ = winding_coordinate(transform, dr_per_winding, circle, device)
                medians.append(float(np.median(winding)))
            name = f"{checkpoint.parent.name}/{checkpoint.name}"
            report["fits"][name] = {"z": z, "winding_median": medians}
            print(name[-39:].ljust(40) + " ".join(f"{value:7.1f}" for value in medians), flush=True)
    if args.out:
        args.out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
