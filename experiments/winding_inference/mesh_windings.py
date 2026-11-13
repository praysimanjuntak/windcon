#!/usr/bin/env python3
"""Where does a fitted spiral put a known sheet, and how far apart are its windings there?

A fit gives every point of the scan a winding coordinate: whole numbers on its sheets, k + 0.5
half-way between sheets k and k + 1. For a tifxyz mesh that is known to follow one sheet of
papyrus, this script reads that coordinate at every node and reports

  spread     the 5th-95th percentile range over the mesh, in windings: under one means the fit
             keeps the sheet on one of its windings
  phase      how far the mesh's median lies from the fit's nearest sheet, in windings (0 to 0.5)
  spacing    the distance between the fit's sheets along the mesh normal at the mesh, in voxels,
             and the fit's windings per 100 voxels over 60 voxels each way along the normal
  per turn   the fit's windings passed in one turn along the mesh (a straight line through the
             coordinate against the angle): 1 when the fit counts as the sheet does, 2 when it
             packs two windings into each of the sheet's

Along one sheet the coordinate grows by one with every turn round the scroll; the turns are
counted along the mesh's grid and taken out. Only the largest connected part of the mesh inside
the fit's z-range is read.

and, with --surface, readings of a surface prediction along the same normals, on up to 2,000
lines of 121 voxels: its sheets per 100 voxels, the distance between neighbouring sheets (the
median gap on each line that meets at least three), and how far the mesh itself is from the
nearest predicted sheet, with the share of the lines' own length that is as near for comparison.

--profile N gives the coordinate's median in three bands of height by N stretches of the mesh's
turns, which shows where along a sheet a fit leaves it; --distance-to gives the distance to the
nearest node of another mesh in the same table.

    run.sh mesh_windings.py --checkpoint <run>/checkpoint_fitted.ckpt --umbilicus <umbilicus.json> \
        --mesh <mesh.tifxyz> [--mesh <another>] [--surface <URL of the prediction's level 0>]
"""

from __future__ import annotations

import argparse
import itertools
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


def by_height_and_stretch(values: np.ndarray, along: np.ndarray, z: np.ndarray, stretches: int) -> list[list]:
    """Medians of `values` in three bands of height with equal node counts (rows, low z first) and `stretches`
    equal stretches of the turns along the mesh (columns); None where a cell has under 20 nodes."""
    edges = np.linspace(along.min(), along.max() + 1e-9, stretches + 1)
    heights = np.percentile(z, [0, 100 / 3, 200 / 3, 100])
    heights[-1] += 1
    table = []
    for low, high in itertools.pairwise(heights):
        band = (z >= low) & (z < high)
        cells = [band & (along >= a) & (along < b) for a, b in itertools.pairwise(edges)]
        table.append([float(np.median(values[cell])) if cell.sum() >= 20 else None for cell in cells])
    return table


def read_prediction(surface: ZarrArray, points: np.ndarray, normals: np.ndarray, rng) -> dict:
    """Readings of a surface prediction along the normal at up to 2,000 of the points, REACH voxels each way."""
    chosen = rng.choice(len(points), min(2000, len(points)), replace=False)
    offsets = np.arange(-REACH, REACH + 1)
    gaps, distances, near, sheets = [], [], [], 0
    for index in chosen:
        line = np.rint(points[index] + offsets[:, None] * normals[index]).astype(int)  # x, y, z
        low, high = line.min(0), line.max(0) + 1
        block = surface.read(((low[2], high[2]), (low[1], high[1]), (low[0], high[0])))
        inside = block[line[:, 2] - low[2], line[:, 1] - low[1], line[:, 0] - low[0]] > 0
        distances.append(np.abs(offsets[inside]).min() if inside.any() else REACH + 1)
        near.append(np.convolve(inside, np.ones(5), mode="same")[2:-2].astype(bool).mean())
        edges = np.diff(np.concatenate([[0], inside.astype(np.int8), [0]]))
        begins, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        sheets += len(begins)
        if len(begins) >= 3:  # the median gap between the centres of the stretches inside the prediction
            gaps.append(np.median(np.diff((begins + ends - 1) / 2)))
    distances = np.array(distances)
    return {
        "prediction_lines": len(chosen),
        "predicted_sheets_per_100_voxels": float(100 * sheets / (len(chosen) * len(offsets))),
        "prediction_spacing_voxels": [float(np.percentile(gaps, q)) for q in (10, 50, 90)] if gaps else None,
        "share_within_2_voxels_of_a_predicted_sheet": float(np.mean(distances <= 2)),
        "the_same_for_a_surface_placed_at_random": float(np.mean(near)),
        "distance_to_predicted_sheet_voxels": [float(np.percentile(distances, q)) for q in (50, 90)],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--umbilicus", type=Path, required=True)
    ap.add_argument("--mesh", type=Path, action="append", required=True)
    ap.add_argument("--surface", help="URL of a surface prediction's level-0 array, for the sheets' own spacing")
    ap.add_argument("--sectors", type=int, default=0,
                    help="also give the windings per turn in this many equal sectors of the angle round the "
                         "umbilicus, for a mesh that passes a sector on two turns or more")
    ap.add_argument("--profile", type=int, default=0,
                    help="also give the coordinate's median in three bands of height by this many stretches of "
                         "the mesh's turns")
    ap.add_argument("--distance-to", type=Path,
                    help="with --profile: another tifxyz mesh; the distance from each node to its nearest node, "
                         "in the same table")
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
            # The fit's count over the stretch on which the prediction's sheets are counted.
            ends = points[None] + REACH * np.array([1.0, -1.0])[:, None, None] * normals[None]
            (far_a, theta_a), (far_b, theta_b) = (winding_coordinate(transform, dr_per_winding, e, device) for e in ends)
            whole = (np.abs(theta_a - theta_b) < np.pi) & ((ends[..., 2] >= z_begin) & (ends[..., 2] < z_end)).all(0)
            median = float(np.median(winding))
            row = {
                "nodes": len(points),
                "winding_median": median,
                "spread": float(np.percentile(winding, 95) - np.percentile(winding, 5)),
                "phase": float(abs(median - round(median))),
                "share_within_half_a_winding_of_the_median": float(np.mean(np.abs(winding - median) < 0.5)),
                "fit_spacing_voxels": [float(np.percentile(spacing, q)) for q in (10, 50, 90)],
                "fit_windings_per_100_voxels": float(100 * np.abs(far_a - far_b)[whole].mean() / (2 * REACH)),
                "turns_spanned": float(np.ptp(along)),
                "fit_windings_per_turn": float(1 + np.polyfit(along, winding, 1)[0]),
            }
            if args.sectors:
                # Between two passes of the sheet through one sector the fit should count one winding.
                sector = np.minimum((theta / (2 * np.pi) * args.sectors).astype(int), args.sectors - 1)
                row["fit_windings_per_turn_by_sector"] = [
                    float(1 + np.polyfit(along[sector == k], winding[sector == k], 1)[0])
                    if (sector == k).sum() > 20 and np.ptp(along[sector == k]) > 1.5 else None
                    for k in range(args.sectors)
                ]
            if args.profile:
                row["winding_by_height_and_stretch"] = by_height_and_stretch(winding, along, points[:, 2], args.profile)
                if args.distance_to:
                    from scipy.spatial import cKDTree

                    other, other_valid, _ = load_mesh(args.distance_to)
                    distance = cKDTree(other[other_valid]).query(points)[0]
                    row["distance_to_other_mesh_by_height_and_stretch"] = by_height_and_stretch(
                        distance, along, points[:, 2], args.profile
                    )
            if surface is not None:
                # The same lines for a mesh whatever else is read in the run.
                row.update(read_prediction(surface, points, normals, np.random.default_rng(0)))
            report[path.name] = row
            print(path.name, json.dumps(row), flush=True)
    if args.out:
        args.out.write_text(json.dumps({"checkpoint": str(args.checkpoint), "meshes": report}, indent=1) + "\n")


if __name__ == "__main__":
    main()
