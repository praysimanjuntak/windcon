#!/usr/bin/env python3
"""Where a fit puts a verified patch between two sheets, what do its neighbours and the scan say?

Input: patch_windings.py --dump-quads (every quad: centre, patch, the fit's raw winding coordinate
and theta, offset from its own snapped sheet) and its per-patch phase report. A patch is *on* a
sheet when |phase| < 0.2 and *between* sheets when |phase| > 0.35. For sampled quads of between-sheet
patches, quads of other patches in the same slice (|dz| <= 2) within --radius voxels, displaced
mostly along the scroll radius, are neighbours. Per neighbour pair:

* the fit's winding difference dw (raw coordinates, unwrapped across theta = 0 like villa);
* the bright bands the CT shows strictly between the two quads (profile along the segment,
  averaged across it; peaks with a real dark gap on both sides), so the scan's winding
  difference is that count + 1 when the two quads lie on different sheets.

On-sheet patches' quads give the control. The question is whether the fit's |dw| for between-sheet
pairs lies half a winding above or below the scan's count, and whether those pairs differ from the
control in spacing.

    uv run python phase_conflicts.py --quads quads.npz --phase pw_all.json \
        --umbilicus data/PHercParis4/umbilicus.json --out phase_conflicts.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from windcon.cli import PARIS4_L2
from windcon.evaluate import read_crop
from windcon.zarr_io import ZarrArray

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_surfaces import umbilicus_xy  # noqa: E402

AIR = 40.0  # windcon.tracer.TraceParams.air_level: intensity of empty space


def count_gaps(vol: ZarrArray, a: np.ndarray, b: np.ndarray, z: int) -> int | None:
    """Dark inter-sheet gaps crossed going from point a to point b (xy) in slice z: the scan's
    winding difference between the sheets through a and b (0 on one, even steeply inclined, sheet).
    A gap is a run of at least 2 px darker than halfway between air and the profile's bright level,
    reaching below 35 % of that range. None if the profile has no bright level."""
    lo = np.floor(np.minimum(a, b)).astype(int) - 8
    hi = np.ceil(np.maximum(a, b)).astype(int) + 9
    img = ndimage.gaussian_filter(read_crop(vol, z, lo[0], hi[0], lo[1], hi[1], 2), 1.0)
    d = b - a
    length = float(np.linalg.norm(d))
    if length < 4:
        return None
    t = d / length
    n = np.array([-t[1], t[0]])
    s = np.arange(0.0, length + 1e-9, 0.5)
    u = np.arange(-2.0, 2.01, 1.0)
    pts = a[None, None, :] + s[None, :, None] * t + u[:, None, None] * n - lo
    prof = ndimage.map_coordinates(img, [pts[..., 1].ravel(), pts[..., 0].ravel()], order=1,
                                   mode="nearest").reshape(len(u), len(s)).mean(axis=0)
    bright = float(np.percentile(prof, 90))
    if bright - AIR < 20:
        return None
    dark = prof < AIR + 0.5 * (bright - AIR)
    edges = np.diff(np.r_[0, dark.astype(int), 0])
    gaps = 0
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if (stop - start) * 0.5 >= 2.0 and prof[start:stop].min() < AIR + 0.35 * (bright - AIR):
            gaps += 1
    return gaps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quads", required=True, type=Path)
    ap.add_argument("--phase", required=True, type=Path, help="patch_windings.py --out report")
    ap.add_argument("--umbilicus", required=True, type=Path)
    ap.add_argument("--radius", type=float, default=40.0)
    ap.add_argument("--per-patch", type=int, default=30, help="quads sampled per patch")
    ap.add_argument("--ct-pairs", type=int, default=400, help="pairs per class whose CT bands are counted")
    ap.add_argument("--volume", default=PARIS4_L2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    with np.load(args.quads) as d:
        names, xyz, patch, s, theta = d["names"], d["xyz"].astype(np.float64), d["patch"], d["s"].astype(np.float64), d["theta"].astype(np.float64)
    per_patch = json.loads(args.phase.read_text())["per_patch"]
    phase = np.array([per_patch[str(n)]["phase"] for n in names])
    cls = np.where(np.abs(phase) < 0.2, 0, np.where(np.abs(phase) > 0.35, 2, 1))  # 0 on, 1 mid, 2 between
    axis = umbilicus_xy(args.umbilicus)
    ax = axis(xyz[:, 2])
    radius_of = np.hypot(xyz[:, 0] - ax[:, 0], xyz[:, 1] - ax[:, 1])
    tree = cKDTree(xyz)
    vol = ZarrArray(args.volume)
    report = {"patches": {"on": int((cls == 0).sum()), "mid": int((cls == 1).sum()), "between": int((cls == 2).sum())},
              "params": {"radius": args.radius, "per_patch": args.per_patch}, "classes": {}}
    for label, want in (("between", 2), ("on", 0)):
        sample = []
        for p in np.flatnonzero(cls == want):
            idx = np.flatnonzero(patch == p)
            sample.extend(rng.choice(idx, min(args.per_patch, len(idx)), replace=False))
        sample = np.array(sample)
        pairs = []  # (a, b, dw, |dr|)
        for a, nbrs in zip(sample, tree.query_ball_point(xyz[sample], args.radius)):
            nbrs = np.array([b for b in nbrs if patch[b] != patch[a] and abs(xyz[b, 2] - xyz[a, 2]) <= 2.0])
            if not len(nbrs):
                continue
            dr = radius_of[nbrs] - radius_of[a]
            dist = np.linalg.norm(xyz[nbrs, :2] - xyz[a, :2], axis=1)
            radial = np.abs(dr) >= 0.8 * dist  # mostly along the scroll radius
            if not radial.any():
                continue
            b = nbrs[radial][np.argmin(dist[radial])]  # the nearest radial neighbour
            dtheta = theta[b] - theta[a]
            dw = s[b] - s[a] + float(dtheta > np.pi) - float(dtheta < -np.pi)
            pairs.append((a, b, dw, abs(radius_of[b] - radius_of[a])))
        if not pairs:
            continue
        a_i, b_i, dw, dr = (np.array(v) for v in zip(*pairs))
        nb_cls = cls[patch[b_i]]
        frac = np.abs(dw - np.round(dw))
        row = {
            "sampled_quads": int(len(sample)),
            "quads_with_radial_neighbour": int(len(pairs)),
            "neighbour_patch_on_sheet": round(float(np.mean(nb_cls == 0)), 4),
            "neighbour_patch_between": round(float(np.mean(nb_cls == 2)), 4),
            "fit_dw_fraction_near_half": round(float(np.mean(frac > 0.35)), 4),
            "fit_abs_dw_p25_50_75": [round(float(v), 2) for v in np.percentile(np.abs(dw), [25, 50, 75])],
            "radial_gap_px_p25_50_75": [round(float(v), 1) for v in np.percentile(dr, [25, 50, 75])],
        }
        # CT check: the scan's winding difference (dark gaps crossed) against the fit's, on pairs
        # with an on-sheet neighbour (between class) or any neighbour (control).
        pick = np.flatnonzero(nb_cls == 0) if want == 2 else np.arange(len(pairs))
        pick = rng.choice(pick, min(args.ct_pairs, len(pick)), replace=False) if len(pick) else pick
        ct = []
        for k in pick:
            n_gaps = count_gaps(vol, xyz[a_i[k], :2], xyz[b_i[k], :2], int(round(xyz[a_i[k], 2])))
            if n_gaps is not None:
                ct.append((abs(dw[k]), n_gaps, dr[k]))
        if ct:
            fit_w, scan_w, gap = (np.array(v, dtype=float) for v in zip(*ct))
            diff = fit_w - scan_w
            apart = scan_w >= 1
            row["ct_pairs"] = len(ct)
            row["scan_windings_0_1_2plus"] = [round(float(np.mean(scan_w == 0)), 3), round(float(np.mean(scan_w == 1)), 3),
                                              round(float(np.mean(scan_w >= 2)), 3)]
            row["fit_minus_scan_windings_p10_25_50_75_90"] = [round(float(v), 2) for v in np.percentile(diff, [10, 25, 50, 75, 90])]
            row["fit_agrees_with_scan_within_0.35"] = round(float(np.mean(np.abs(diff) < 0.35)), 4)
            row["fit_fewer_windings_than_scan_by_0.35_or_more"] = round(float(np.mean(diff <= -0.35)), 4)
            row["fit_more_windings_than_scan_by_0.35_or_more"] = round(float(np.mean(diff >= 0.35)), 4)
            if apart.any():
                row["scan_px_per_winding_p25_50_75"] = [round(float(v), 1) for v in np.percentile(gap[apart] / scan_w[apart], [25, 50, 75])]
                row["fit_px_per_winding_p25_50_75"] = [round(float(v), 1) for v in np.percentile(gap[apart] / np.maximum(fit_w[apart], 0.25), [25, 50, 75])]
        report["classes"][label] = row
        print(label, json.dumps(row))
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
