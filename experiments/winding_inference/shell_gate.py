#!/usr/bin/env python3
"""Where does fit_spiral.py accept winding-inference rays? Rebuild its table of the outer shell.

fit_spiral.py counts a pair of ray points only if both lie where its table of the outer shell
(`ShellPolarMap`: the shell mesh's nodes binned by slice and by 720 angles) has a confidence of
at least `shell_min_confidence` (0.25) and inside the table's radius. The confidence is the
table's occupancy, smoothed over 4 slices and 1 angle bin and divided by its maximum. A shell mesh
that covers part of the circumference, or has its rows 20 slices apart, therefore passes only part
of a window. This script builds the table with villa's own class and reports the share of the
window that passes, by angle, and the share of a store's crossings.

    run.sh shell_gate.py --shell <outer_shell tifxyz> --umbilicus <umbilicus.json> \
        --z-begin 9900 --z-end 10900 [--store <store>] [--min-confidence 0.25]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from build_store import load_store
from PIL import Image

SECTORS = 36


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shell", type=Path, required=True)
    ap.add_argument("--umbilicus", type=Path, required=True)
    ap.add_argument("--z-begin", type=int, required=True)
    ap.add_argument("--z-end", type=int, required=True)
    ap.add_argument("--margin", type=int, default=160, help="the fit's model_flow_bounds_z_margin")
    ap.add_argument("--min-confidence", type=float, default=0.25, help="the fit's shell_min_confidence")
    ap.add_argument("--store", type=Path, help="also report the share of this store's crossings that pass")
    args = ap.parse_args()

    import torch
    from vesuvius.neural_tracing.winding_models import infer_winding_volume as villa

    sys.path.insert(0, villa._spiral_scripts_dir())
    from fit_spiral import ShellPolarMap
    from umbilicus import json_umbilicus_z_to_yx

    xyz = np.stack([np.asarray(Image.open(args.shell / f"{axis}.tif"), dtype=np.float32) for axis in "xyz"], axis=-1)
    nodes = xyz[(xyz > 0).all(-1)]
    centre_yx = json_umbilicus_z_to_yx(str(args.umbilicus), coordinate_scale=1.0)
    config = {"shell_table_smooth_sigma_z": 4.0, "shell_table_smooth_sigma_theta": 1.0,
              "shell_min_confidence": args.min_confidence}  # fmt: skip
    table = ShellPolarMap(
        SimpleNamespace(valid_zyxs=torch.from_numpy(np.ascontiguousarray(nodes[:, ::-1]))), centre_yx,
        args.z_begin - args.margin, args.z_end + args.margin, 720, "cpu", config=config,
    )  # fmt: skip

    in_window = nodes[(nodes[:, 2] >= args.z_begin) & (nodes[:, 2] < args.z_end)]
    offset = in_window[:, [1, 0]] - centre_yx(in_window[:, 2])  # y, x
    sector = (np.mod(np.arctan2(offset[:, 0], offset[:, 1]), 2 * np.pi) / (2 * np.pi) * SECTORS).astype(int) % SECTORS
    print(f"shell nodes in the window: {len(in_window):,}; per 10-degree sector: "
          + " ".join(str(int((sector == k).sum())) for k in range(SECTORS)))  # fmt: skip

    confidence = table.lookup_table[1, args.z_begin - table.z_min : args.z_end - table.z_min, :-1].numpy()
    passes = confidence >= args.min_confidence
    print(f"cells of the window that pass at {args.min_confidence:g}: {passes.mean():.1%}; per 10-degree sector: "
          + " ".join(f"{share:.2f}" for share in passes.reshape(len(passes), SECTORS, -1).mean((0, 2))))  # fmt: skip

    if args.store:
        rays, _ = load_store(args.store)
        index = np.repeat(np.arange(len(rays["crossing_counts"])), rays["crossing_counts"])
        points = rays["ray_origin_zyx"][index] + rays["crossing_t"][:, None] * rays["ray_step_zyx"][index]
        points = torch.from_numpy(points[(points[:, 0] >= args.z_begin) & (points[:, 0] < args.z_end)].astype(np.float32))
        table_radius, radius, _, valid = table.lookup(points)
        inside = radius <= table_radius
        print(f"{args.store.name}: {len(points):,} crossings in the window; the table passes {valid.float().mean():.1%}, "
              f"{inside.float().mean():.1%} are inside its radius, {(valid & inside).float().mean():.1%} count")


if __name__ == "__main__":
    main()
