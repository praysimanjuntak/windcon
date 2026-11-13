#!/usr/bin/env python3
"""Does a winding-inference store count the windings between sheets that people traced?

Consecutive crossings of a ray are one winding apart by the model's count. Where a scroll has
traced sheets that follow one another without a gap, a ray that passes two of them shows what the
model counts between sheets a person told apart. PHerc. 0139 has 37 such meshes in a row, w023 to
w059, each a little over one turn, registered on its 9.362 um scan; Paris 4 has one mesh of 6.4
turns.

For each ray this script finds where it passes the meshes (the ray sample nearest to a mesh,
within --within voxels of it; passes closer together than --merge voxels are one sheet), reads the
model's level at each pass (the crossings numbered along the ray, interpolated between them), and
compares the level difference between two passes that follow one another on the ray with the
traced windings between them:

- two passes of one mesh: the turns between them along the mesh;
- passes of two meshes named w<number>... whose numbers differ by one at most: one winding (the
  end of each mesh overlaps the start of the next, so the numbers alone would miscount there);
- with --number-step N, for a scroll where only every Nth winding is traced (PHerc. 1447: w058,
  w060, w062) and every mesh starts at the same angle: the difference of the numbers, a mesh's
  second turn counting as the next number;
- anything else is left out.

    uv run python store_vs_meshes.py --store <store> --umbilicus <umbilicus.json> --voxel-um 9.362 \
        --mesh <.../w023_...> --mesh <.../w024_...> [...] [--out report.json]

Reported: the model's windings per traced winding, the share of pairs one traced winding apart
between which the model counts one (within a half), none or two and more, and the same by the
distance between the two sheets on the ray. With --surface, a surface prediction is a second
witness: does it have a sheet between two traced neighbours where the model counts two?

With --checkpoint the same pairs are counted by a fitted spiral instead of the model: the fit's
winding coordinate at the two passes (run.sh store_vs_meshes.py ..., villa's environment). The
store then only supplies the lines along which the sheets are met.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
from build_store import load_store
from mesh_windings import turns
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ink_probe"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spiral_fit"))
from make_outer_shell import axis_xy  # noqa: E402
from render_mesh import load_mesh  # noqa: E402

from windcon.zarr_io import ZarrArray  # noqa: E402

GROUPS_UM = (0, 115, 145, 175, 230, np.inf)  # distance between the two sheets on the ray
SAMPLE = 0.5  # ray samples between which the nearest mesh point is looked up


def mesh_points(folder: Path, umbilicus: Path, upsample: int, z_range: tuple[float, float]):
    """Points (z, y, x) on a mesh, `upsample` per grid cell edge, with the turns along the mesh at each point
    (continuous: one more for every turn round the umbilicus) and its angle round the umbilicus in turns."""
    xyz, valid, _ = load_mesh(folder)
    valid &= (xyz[..., 2] >= z_range[0]) & (xyz[..., 2] < z_range[1])
    if not valid.any():
        return np.zeros((0, 3), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32)
    centre = axis_xy(umbilicus, xyz[..., 2].ravel()).reshape(*valid.shape, 2)
    theta = np.arctan2(xyz[..., 1] - centre[..., 1], xyz[..., 0] - centre[..., 0])
    turn, valid = turns(theta, valid)
    along = theta / (2 * np.pi) + turn
    whole = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
    rows, cols = np.nonzero(whole)
    steps = (np.arange(upsample) + 0.5) / upsample
    u, v = (a.ravel() for a in np.meshgrid(steps, steps, indexing="ij"))
    weights = np.stack([(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v])  # corner, point in the cell
    corners = ((rows, cols), (rows + 1, cols), (rows, cols + 1), (rows + 1, cols + 1))
    points = sum(xyz[r, c][:, None, :] * w[None, :, None] for (r, c), w in zip(corners, weights))
    turns_along = sum(along[r, c][:, None] * w[None, :] for (r, c), w in zip(corners, weights))
    points = points.reshape(-1, 3)
    centre = axis_xy(umbilicus, points[:, 2])
    angle = np.arctan2(points[:, 1] - centre[:, 1], points[:, 0] - centre[:, 0]) / (2 * np.pi)
    return points[:, ::-1].astype(np.float32), turns_along.ravel().astype(np.float32), angle.astype(np.float32)


def passes(tree: cKDTree, origin: np.ndarray, step: np.ndarray, t: np.ndarray, within: float, merge: float):
    """(position along the ray in samples, index of the nearest mesh point) of every sheet the ray passes between
    its first and last crossing, in the order met; passes closer than `merge` samples are one sheet."""
    samples = np.arange(t[0], t[-1] + SAMPLE / 2, SAMPLE)
    distance, index = tree.query(origin + samples[:, None] * step, distance_upper_bound=within)
    runs, _ = ndimage.label(np.isfinite(distance))
    found = []
    for members in ndimage.find_objects(runs):
        best = members[0].start + int(np.argmin(distance[members]))
        if found and samples[best] - found[-1][0] < merge:
            if distance[best] < found[-1][2]:
                found[-1] = (samples[best], index[best], distance[best])
            continue
        found.append((samples[best], index[best], distance[best]))
    return [(position, point) for position, point, _ in found]


def predicted_sheets(surface: ZarrArray, origin: np.ndarray, step: np.ndarray, first: float, last: float) -> np.ndarray:
    """Positions along the ray, in samples, of the centres of its stretches inside a surface prediction."""
    samples = np.arange(int(np.floor(first)), int(np.ceil(last)) + 1)
    line = np.rint(origin + samples[:, None] * step).astype(int)  # z, y, x
    low, high = line.min(0), line.max(0) + 1
    block = surface.read(tuple((int(a), int(b)) for a, b in zip(low, high)))
    inside = block[line[:, 0] - low[0], line[:, 1] - low[1], line[:, 2] - low[2]] > 0
    edges = np.diff(np.concatenate([[0], inside.astype(np.int8), [0]]))
    return samples[0] + (np.flatnonzero(edges == 1) + np.flatnonzero(edges == -1) - 1) / 2


def summary(rows: np.ndarray) -> dict:
    """rows: traced windings between two passes, the windings counted between them (by the model or by a fit),
    sheet distance in micrometres, predicted sheets strictly between the passes (nan without a prediction)."""
    traced, model, between = rows[:, 0], np.abs(rows[:, 1]), rows[:, 3]
    one = traced == 1
    result = {
        "pairs": len(rows),
        "windings_per_traced_winding": float(model.sum() / traced.sum()),
        "pairs_one_traced_winding_apart": int(one.sum()),
        "of_those_counted_as_one": float(np.mean(np.abs(model[one] - 1) < 0.5)) if one.any() else None,
        "of_those_counted_as_none": float(np.mean(model[one] < 0.5)) if one.any() else None,
        "of_those_counted_as_two_or_more": float(np.mean(model[one] >= 1.5)) if one.any() else None,
    }
    clear, crowded = one & (between == 0), one & (between >= 1)
    if clear.any() or crowded.any():
        result["of_those_with_a_predicted_sheet_between"] = float(crowded.sum() / (clear.sum() + crowded.sum()))
        if clear.any():
            result["counted_as_one_where_the_prediction_has_no_sheet_between"] = float(np.mean(np.abs(model[clear] - 1) < 0.5))
        if crowded.any():
            result["counted_as_two_or_more_where_the_prediction_has_a_sheet_between"] = float(np.mean(model[crowded] >= 1.5))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--umbilicus", type=Path, required=True)
    ap.add_argument("--mesh", type=Path, action="append", required=True)
    ap.add_argument("--voxel-um", type=float, required=True, help="voxel size of the volume the store was built on")
    ap.add_argument("--within", type=float, default=4.0, help="a ray passes a mesh if it comes this near, voxels")
    ap.add_argument("--merge", type=float, default=6.0, help="passes closer than this are one sheet, voxels")
    ap.add_argument("--upsample", type=int, default=4, help="mesh points per grid cell edge")
    ap.add_argument("--surface", help="URL of a surface prediction's level in the store's voxels: also count its "
                                      "sheets between two traced ones, as a second witness where the model counts two")
    ap.add_argument("--number-step", type=int, default=1,
                    help="the traced windings are this many numbers apart (meshes must start at one angle)")
    ap.add_argument("--checkpoint", type=Path, help="count with this fitted spiral instead of the store's model")
    ap.add_argument("--out", type=Path, help="write the report here as JSON")
    args = ap.parse_args()

    rays, _ = load_store(args.store)
    ends = np.cumsum(rays["crossing_counts"])
    begins = ends - rays["crossing_counts"]
    owner = np.repeat(np.arange(len(ends)), rays["crossing_counts"])
    crossing_z = rays["ray_origin_zyx"][owner, 0] + rays["crossing_t"] * rays["ray_step_zyx"][owner, 0]
    z_range = (float(crossing_z.min()) - 20, float(crossing_z.max()) + 20)

    clouds = [mesh_points(path, args.umbilicus, args.upsample, z_range) for path in args.mesh]
    points = np.concatenate([c[0] for c in clouds])
    along, angle = np.concatenate([c[1] for c in clouds]), np.concatenate([c[2] for c in clouds])
    mesh = np.repeat(np.arange(len(clouds)), [len(c[0]) for c in clouds])
    numbers = [re.match(r"w(\d+)", path.name) for path in args.mesh]
    number = np.array([float(m.group(1)) if m else np.nan for m in numbers])
    # a mesh's number plus the whole turns from its start: its second turn lies on the next winding
    lap = np.concatenate([np.floor(c[1] - c[1].min() + 1e-6) if len(c[1]) else c[1] for c in clouds])
    print(f"{len(args.mesh)} meshes, {len(points):,} points in z {z_range[0]:.0f}-{z_range[1]:.0f}", flush=True)
    tree = cKDTree(points)
    surface = ZarrArray(args.surface) if args.surface else None
    clearance = args.within + 2  # voxels: a predicted sheet this near a pass is the traced sheet itself

    met = []  # per ray: crossings, origin, step, voxels per sample, passes
    for i in np.flatnonzero(rays["crossing_counts"] >= 3):
        t = rays["crossing_t"][begins[i] : ends[i]].astype(np.float64)
        origin, step = rays["ray_origin_zyx"][i].astype(np.float64), rays["ray_step_zyx"][i].astype(np.float64)
        pitch = float(np.linalg.norm(step))
        found = passes(tree, origin, step, t, args.within, args.merge / pitch)
        if len(found) >= 2:
            met.append((t, origin, step, pitch, found))

    fit = None
    if args.checkpoint:
        import torch
        from vesuvius.neural_tracing.winding_models import infer_winding_volume as villa

        sys.path.insert(0, villa._spiral_scripts_dir())
        from mesh_windings import winding_coordinate
        from store_vs_fit import fitted_transform

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        where = np.concatenate([origin + np.array([p for p, _ in found])[:, None] * step for _, origin, step, _, found in met])
        with torch.inference_mode():
            transform, dr_per_winding, (z_begin, z_end) = fitted_transform(args.checkpoint, args.umbilicus, device)
            winding, theta = winding_coordinate(transform, dr_per_winding, where[:, ::-1], device)
        fit = (winding, theta / (2 * np.pi), (where[:, 0] >= z_begin) & (where[:, 0] < z_end))

    rows, left_out, first = [], 0, 0  # first: index of a ray's first pass among all passes
    for t, origin, step, pitch, found in met:
        centres = None  # the prediction's sheets on this ray, read once if a pair needs them
        for j, ((t_a, a), (t_b, b)) in enumerate(itertools.pairwise(found), start=first):
            if mesh[a] == mesh[b]:
                swing = angle[b] - angle[a]
                traced = abs(round(float(along[b] - along[a] - (swing - round(swing)))))
            elif args.number_step > 1:
                apart = abs(number[mesh[a]] + lap[a] - number[mesh[b]] - lap[b])  # nan when a mesh has no number
                traced = int(apart) if 1 <= apart <= args.number_step else 0
            elif abs(number[mesh[a]] - number[mesh[b]]) <= 1:
                traced = 1
            else:
                traced = 0
            if traced == 0 or (fit is not None and not (fit[2][j] and fit[2][j + 1])):
                left_out += 1
                continue
            if fit is None:
                level = np.interp([t_a, t_b], t, np.arange(len(t)))
                counted = level[1] - level[0]
            else:  # the coordinate jumps by one where the spiral's angle wraps: take that out
                counted = fit[0][j + 1] - fit[0][j] + round(float(fit[1][j + 1] - fit[1][j]))
            between = np.nan
            if surface is not None:
                if centres is None:
                    centres = predicted_sheets(surface, origin, step, found[0][0], found[-1][0])
                between = np.sum((centres > t_a + clearance / pitch) & (centres < t_b - clearance / pitch))
            rows.append((traced, counted, (t_b - t_a) * pitch * args.voxel_um / traced, between))
        first += len(found)
    if not rows:
        raise SystemExit("no ray passes two traced windings")
    rows = np.array(rows)
    report = {
        "store": str(args.store),
        "counted_by": str(args.checkpoint) if args.checkpoint else "the store's model",
        "rays": int((rays["crossing_counts"] >= 3).sum()),
        "pairs_left_out": left_out,
        "traced_sheet_distance_um_10_50_90": [float(np.percentile(rows[:, 2], q)) for q in (10, 50, 90)],
        "all": summary(rows),
        "by_sheet_distance_um": {},
    }
    for low, high in itertools.pairwise(GROUPS_UM):
        group = rows[(rows[:, 2] >= low) & (rows[:, 2] < high)]
        if len(group) >= 20:
            report["by_sheet_distance_um"][f"{low:g}-{high:g}"] = summary(group)
    print(json.dumps(report, indent=1))
    if args.out:
        args.out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
