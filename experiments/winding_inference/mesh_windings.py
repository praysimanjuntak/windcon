#!/usr/bin/env python3
"""Where does a fitted spiral put a known sheet, and how far apart are its windings there?

A fit gives every point of the scan a winding coordinate: whole numbers on its sheets, k + 0.5
half-way between sheets k and k + 1. For a tifxyz mesh that is known to follow one sheet of
papyrus, this script reads that coordinate at every node and reports

  spread     the 5th-95th percentile range over the mesh, in windings: under one means the fit
             keeps the sheet on one of its windings
  phase      how far the mesh's median lies from the fit's nearest sheet, in windings (0 to 0.5)
  spacing    the distance between the fit's sheets along the mesh normal, in voxels
  per turn   the fit's windings passed in one turn along the mesh (a straight line through the
             coordinate against the angle): 1 when the fit counts as the sheet does, 2 when it
             packs two windings into each of the sheet's

Along one sheet the coordinate grows by one with every turn round the scroll; the turns are
counted along the mesh's grid and taken out. Only the largest connected part of the mesh inside
the fit's z-range is read.

and, with --surface, two readings of a surface prediction along the same normals, on up to 2,000
lines of 121 voxels: the distance between its neighbouring sheets (the median gap on each line
that meets at least three), and how far the mesh itself is from the nearest predicted sheet.

    run.sh mesh_windings.py --checkpoint <run>/checkpoint_fitted.ckpt --umbilicus <umbilicus.json> \
        --mesh <mesh.tifxyz> [--mesh <another>] [--surface <URL of the prediction's level 0>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from store_vs_fit import CHUNK, fitted_transform

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ink_probe"))
from render_mesh import load_mesh, node_normals  # noqa: E402

from windcon.zarr_io import ZarrArray  # noqa: E402

REACH = 60  # voxels read each way along a normal for the prediction's sheet spacing


def winding_coordinate(transform, dr_per_winding, xyz: np.ndarray, device) -> tuple[np.ndarray, np.ndarray]:
    """The fit's winding coordinate and its angle round the spiral (radians) at scan points (x, y, z)."""
    import torch
    from sample_spiral import get_theta_and_radii

    zyx = torch.from_numpy(np.ascontiguousarray(xyz[:, ::-1])).to(device, torch.float32)
    spiral = torch.cat([transform(zyx[i : i + CHUNK]) for i in range(0, len(zyx), CHUNK)])
    theta, _, shifted = get_theta_and_radii(spiral[:, 1:], dr_per_winding)
    return (shifted / dr_per_winding).cpu().numpy().astype(np.float64), theta.cpu().numpy().astype(np.float64)


def turns(theta: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Whole turns round the spiral of each node relative to a start node, counted along the grid, and the
    mask of the largest connected part of `mask`, which is the part that is counted."""
    from scipy import ndimage

    labels, count = ndimage.label(mask)
    if count == 0:
        return np.zeros(mask.shape, dtype=np.int64), mask
    largest = labels == 1 + np.argmax(ndimage.sum(mask, labels, range(1, count + 1)))
    turn = np.zeros(mask.shape, dtype=np.int64)
    seen = np.zeros(mask.shape, dtype=bool)
    start = tuple(np.argwhere(largest)[0])
    seen[start] = True
    frontier = [start]
    while frontier:
        following = []
        for r, c in frontier:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and largest[nr, nc] and not seen[nr, nc]:
                    step = theta[nr, nc] - theta[r, c]
                    turn[nr, nc] = turn[r, c] + (step < -np.pi) - (step > np.pi)
                    seen[nr, nc] = True
                    following.append((nr, nc))
        frontier = following
    return turn, largest


def read_prediction(surface: ZarrArray, points: np.ndarray, normals: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """Along the normal at up to 2,000 of the points: the distance between neighbouring predicted sheets (per
    line, the median gap between the centres of the stretches the line spends inside the prediction), and the
    distance from the point to the nearest predicted voxel on its line (REACH + 1 where there is none)."""
    chosen = rng.choice(len(points), min(2000, len(points)), replace=False)
    offsets = np.arange(-REACH, REACH + 1)
    spacings, distances = [], []
    for index in chosen:
        line = np.rint(points[index] + offsets[:, None] * normals[index]).astype(int)  # x, y, z
        low, high = line.min(0), line.max(0) + 1
        block = surface.read(((low[2], high[2]), (low[1], high[1]), (low[0], high[0])))
        inside = block[line[:, 2] - low[2], line[:, 1] - low[1], line[:, 0] - low[0]] > 0
        distances.append(np.abs(offsets[inside]).min() if inside.any() else REACH + 1)
        edges = np.diff(np.concatenate([[0], inside.astype(np.int8), [0]]))
        begins, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        if len(begins) >= 3:
            spacings.append(np.median(np.diff((begins + ends - 1) / 2)))
    return np.array(spacings, dtype=np.float64), np.array(distances, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--umbilicus", type=Path, required=True)
    ap.add_argument("--mesh", type=Path, action="append", required=True)
    ap.add_argument("--surface", help="URL of a surface prediction's level-0 array, for the sheets' own spacing")
    ap.add_argument("--out", type=Path, help="write the report here as JSON")
    args = ap.parse_args()

    import torch
    from vesuvius.neural_tracing.winding_models import infer_winding_volume as villa

    sys.path.insert(0, villa._spiral_scripts_dir())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    surface = ZarrArray(args.surface) if args.surface else None
    report = {}
    with torch.inference_mode():
        transform, dr_per_winding, (z_begin, z_end) = fitted_transform(args.checkpoint, args.umbilicus, device)
        for path in args.mesh:
            xyz, valid, _ = load_mesh(path)
            all_normals = node_normals(xyz, valid)
            valid &= (xyz[..., 2] >= z_begin) & (xyz[..., 2] < z_end)
            if not valid.any():
                print(path.name, "has no node in the fit's z-range", flush=True)
                continue
            grid_winding = np.zeros(valid.shape)
            grid_theta = np.zeros(valid.shape)
            grid_winding[valid], grid_theta[valid] = winding_coordinate(transform, dr_per_winding, xyz[valid], device)
            # Every turn along the sheet adds one winding at the spiral's seam: count the turns and take them out.
            turn, valid = turns(grid_theta, valid)
            points, normals = xyz[valid], all_normals[valid]
            winding, theta = (grid_winding - turn)[valid], grid_theta[valid]
            along = theta / (2 * np.pi) + turn[valid]  # turns round the scroll, continuous over the mesh
            ahead, _ = winding_coordinate(transform, dr_per_winding, points + 4 * normals, device)
            behind, theta_behind = winding_coordinate(transform, dr_per_winding, points - 4 * normals, device)
            same_turn = np.abs(theta_behind - theta) < np.pi  # not across the seam
            rate = np.abs(ahead - behind)[same_turn] / 8
            spacing = 1 / rate[rate > 1e-4]
            median = float(np.median(winding))
            row = {
                "nodes": len(points),
                "winding_median": median,
                "spread": float(np.percentile(winding, 95) - np.percentile(winding, 5)),
                "phase": float(abs(median - round(median))),
                "share_within_half_a_winding_of_the_median": float(np.mean(np.abs(winding - median) < 0.5)),
                "fit_spacing_voxels": [float(np.percentile(spacing, q)) for q in (10, 50, 90)],
                "turns_spanned": float(np.ptp(along)),
                "fit_windings_per_turn": float(1 + np.polyfit(along, winding, 1)[0]),
            }
            if surface is not None:
                # The same lines for a mesh whatever else is read in the run.
                sheets, distance = read_prediction(surface, points, normals, np.random.default_rng(0))
                row["prediction_spacing_voxels"] = [float(np.percentile(sheets, q)) for q in (10, 50, 90)]
                row["prediction_lines"] = len(sheets)
                row["share_within_2_voxels_of_a_predicted_sheet"] = float(np.mean(distance <= 2))
                row["distance_to_predicted_sheet_voxels"] = [float(np.percentile(distance, q)) for q in (50, 90)]
            report[path.name] = row
            print(path.name, json.dumps(row), flush=True)
    if args.out:
        args.out.write_text(json.dumps({"checkpoint": str(args.checkpoint), "meshes": report}, indent=1) + "\n")


if __name__ == "__main__":
    main()
