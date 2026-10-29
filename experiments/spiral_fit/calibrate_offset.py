#!/usr/bin/env python3
"""Calibrate the inward shift of traced paths against human same-winding arcs.

For human arcs drawn in one slice (outside the fit window, so the fit evaluation stays untouched),
trace every band of the arc's crop, smooth the paths, and measure the signed offset of each arc point
from the nearest traced vertex along the path normal facing the scroll axis (+ = the human point
lies toward the axis). The median is the shift to apply; the spread shows the centring noise.

    uv run python experiments/spiral_fit/calibrate_offset.py --gt data/paris4/same_windings.json \
        --umbilicus data/paris4/umbilicus.json --z 12418
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from windcon import evaluate
from windcon.annotations import load_collections
from windcon.chains import inward_normals, smooth_path
from windcon.cli import PARIS4_L2
from windcon.tracer import SliceField, TraceParams, trace_region
from windcon.zarr_io import ZarrArray


def umbilicus_xy(path: Path, z: float) -> np.ndarray:
    pts = sorted((p["z"], p["x"], p["y"]) for p in json.loads(path.read_text())["control_points"])
    zs, xs, ys = (np.array(v, float) for v in zip(*pts))
    return np.array([np.interp(z, zs, xs), np.interp(z, zs, ys)])


def offsets(arc_xy: np.ndarray, paths: list[np.ndarray], center: np.ndarray, max_dist: float) -> np.ndarray:
    verts, normals = [], []
    for xy in paths:
        n, _ = inward_normals(xy, np.repeat(center[None], len(xy), axis=0))
        verts.append(xy)
        normals.append(n)
    V, N = np.concatenate(verts), np.concatenate(normals)
    dist, idx = cKDTree(V).query(arc_xy, k=1, distance_upper_bound=max_dist)
    ok = np.isfinite(dist)
    return np.einsum("ij,ij->i", arc_xy[ok] - V[idx[ok]], N[idx[ok]])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--umbilicus", required=True, type=Path)
    ap.add_argument("--z", type=int, nargs="+", required=True)
    ap.add_argument("--sigma", type=float, nargs="+", default=[0.0, 3.0, 6.0])
    ap.add_argument("--max-dist", type=float, default=6.0)
    args = ap.parse_args()
    vol = ZarrArray(PARIS4_L2)
    arcs = [a for a in evaluate.flat_z_arcs(load_collections(args.gt)) if a.z in set(args.z)]
    raw: list[tuple[np.ndarray, list[np.ndarray], np.ndarray]] = []
    for arc in arcs:
        x0, x1, y0, y1 = evaluate.arc_box(arc, 128)
        field = SliceField(evaluate.read_crop(vol, arc.z, x0, x1, y0, y1, 2), TraceParams())
        origin = np.array([x0, y0], dtype=np.float64)
        paths = [p.xy + origin for p in trace_region(field)]
        raw.append((arc.xy, paths, umbilicus_xy(args.umbilicus, arc.z)))
    print(f"{len(arcs)} arcs in slices {sorted({a.z for a in arcs})}")
    for sigma in args.sigma:
        d = np.concatenate([offsets(a, [smooth_path(p, sigma) for p in ps], c, args.max_dist)
                            for a, ps, c in raw])
        q = np.percentile(d, [10, 25, 50, 75, 90])
        print(f"sigma {sigma:>4} vertices: {len(d)} arc points matched; offset (+ = toward axis) "
              f"p10 {q[0]:.2f} p25 {q[1]:.2f} median {q[2]:.2f} p75 {q[3]:.2f} p90 {q[4]:.2f}")


if __name__ == "__main__":
    main()
