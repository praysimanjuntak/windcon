#!/usr/bin/env python3
"""Smooth, shift and link per-slice traced paths into multi-slice same-winding chains.

Reads the auto_z<Z>.npz files of trace_window.py. Writes into --out:
  auto_chains.json   one same-winding collection per chain (one sheet followed through z)
  auto_singles.json  one collection per smoothed, shifted path, without linking
  chains_paths.npz   the processed paths and the links between them (for check_links.py)
  chains_stats.json  link and chain statistics

    uv run python experiments/spiral_fit/build_chains.py --raw-dir auto_paths_z5 \
        --umbilicus data/PHercParis4/umbilicus.json --out pcls --delta 1.9
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from windcon.annotations import write_same_winding
from windcon.chains import build_chains, link_slices, shift_inward, smooth_path


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def umbilicus(path: Path):
    pts = sorted((p["z"], p["x"], p["y"]) for p in json.loads(path.read_text())["control_points"])
    zs, xs, ys = (np.array(v, float) for v in zip(*pts))
    return lambda z: np.array([np.interp(z, zs, xs), np.interp(z, zs, ys)])


def resample(xy: np.ndarray, spacing: float) -> np.ndarray:
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    n = max(int(round(cum[-1] / spacing)), 1)
    s = np.linspace(0.0, cum[-1], n + 1)
    return np.stack([np.interp(s, cum, xy[:, 0]), np.interp(s, cum, xy[:, 1])], axis=1)


def displacement(a: np.ndarray, b: np.ndarray, r: float) -> float:
    """Median distance from b's vertices to a, over the vertices that lie alongside a."""
    d, _ = cKDTree(a).query(b, k=1)
    d = d[d <= r]
    return float(np.median(d)) if len(d) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", required=True, type=Path)
    ap.add_argument("--umbilicus", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--delta", type=float, required=True, help="inward shift (px) onto the sheet face")
    ap.add_argument("--sigma", type=float, default=3.0, help="smoothing along each path (vertices)")
    ap.add_argument("--spacing", type=float, default=20.0, help="exported point spacing (px)")
    ap.add_argument("--max-dz", type=int, default=5, help="only link slices at most this far apart")
    ap.add_argument("--r-match", type=float, default=2.5)
    ap.add_argument("--r-sep", type=float, default=5.0)
    ap.add_argument("--min-votes", type=int, default=8)
    ap.add_argument("--min-chain-paths", type=int, default=1,
                    help="write only chains made of at least this many per-slice paths")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    centre = umbilicus(args.umbilicus)

    zs, paths = [], []
    for f in sorted(args.raw_dir.glob("auto_z*.npz"), key=lambda p: int(np.load(p)["z"])):
        with np.load(f) as npz:  # NpzFile decompresses an array on every access: read each once
            xy_all, offsets, z = npz["xy"].astype(np.float64), npz["offsets"], int(npz["z"])
        c = centre(z)
        slice_paths = []
        for a, b in zip(offsets[:-1], offsets[1:]):
            xy = smooth_path(xy_all[a:b], args.sigma)
            slice_paths.append(shift_inward(xy, np.repeat(c[None], len(xy), axis=0), args.delta))
        zs.append(z)
        paths.append(slice_paths)
    log(f"{len(zs)} slices, {sum(map(len, paths)):,} paths")

    links, pair_stats = [], []
    for k in range(len(zs) - 1):
        if zs[k + 1] - zs[k] > args.max_dz:
            links.append([])
            continue
        pairs = link_slices(paths[k], paths[k + 1], r_match=args.r_match, r_sep=args.r_sep,
                            min_votes=args.min_votes)
        links.append(pairs)
        disp = [displacement(paths[k][i], paths[k + 1][j], args.r_match) for i, j in pairs[:500]]
        pair_stats.append({"z": [zs[k], zs[k + 1]], "links": len(pairs),
                           "linked_fraction": round(len(pairs) / max(len(paths[k]), 1), 3),
                           "median_displacement_px": round(float(np.nanmedian(disp)), 2) if disp else None})
    chains = build_chains([len(p) for p in paths], links)
    log(f"{sum(map(len, links)):,} links -> {len(chains):,} chains")
    # Processed paths and links, for check_links.py and drop_near_patches.py.
    flat = [(s, i) for s in range(len(zs)) for i in range(len(paths[s]))]
    chain_of = {node: n for n, chain in enumerate(chains) for node in chain}
    lengths = [len(paths[s][i]) for s, i in flat]
    np.savez_compressed(
        args.out / "chains_paths.npz",
        xy=np.concatenate([paths[s][i] for s, i in flat]).astype(np.float32),
        offsets=np.cumsum([0] + lengths).astype(np.int64),
        path_slice=np.array([s for s, _ in flat], dtype=np.int32),
        path_index=np.array([i for _, i in flat], dtype=np.int32),
        path_chain=np.array([chain_of[node] for node in flat], dtype=np.int64),
        zs=np.array(zs, dtype=np.int64),
        links=np.array([(s, i, j) for s, pairs in enumerate(links) for i, j in pairs],
                       dtype=np.int64).reshape(-1, 3),
    )

    def chain_points(chain):
        pts, last = [], None
        for s, i in chain:
            xy = resample(paths[s][i], args.spacing)
            if last is not None and np.linalg.norm(xy[-1] - last) < np.linalg.norm(xy[0] - last):
                xy = xy[::-1]  # start each slice's path next to where the previous one ended
            pts.append(np.column_stack([xy, np.full(len(xy), float(zs[s]))]))
            last = xy[-1]
        return np.concatenate(pts)

    kept = [c for c in chains if len(c) >= args.min_chain_paths]
    write_same_winding(args.out / "auto_chains.json",
                       [(f"chain_{n:06d}", chain_points(c)) for n, c in enumerate(kept)])
    write_same_winding(args.out / "auto_singles.json",
                       [(f"single_z{zs[s]}_{i:05d}", chain_points([(s, i)])) for s, i in flat])

    sizes = np.array([len(c) for c in chains])
    extents = np.array([zs[c[-1][0]] - zs[c[0][0]] for c in chains])
    bins = [(1, 1), (2, 2), (3, 4), (5, 9), (10, 19), (20, 10**9)]
    stats = {
        "slices": len(zs), "paths": int(sizes.sum()), "chains": len(chains),
        "chains_written": len(kept), "paths_in_written_chains": int(sum(len(c) for c in kept)),
        "params": {k: getattr(args, k) for k in ("delta", "sigma", "spacing", "max_dz", "r_match",
                                                 "r_sep", "min_votes", "min_chain_paths")},
        "paths_per_chain": {f"{lo}-{hi}" if hi < 10**9 else f">={lo}":
                            {"chains": int(((sizes >= lo) & (sizes <= hi)).sum()),
                             "paths": int(sizes[(sizes >= lo) & (sizes <= hi)].sum())}
                            for lo, hi in bins},
        "z_extent_voxels": {"median": float(np.median(extents)), "p90": float(np.percentile(extents, 90)),
                            "max": int(extents.max())},
        "slice_pairs": pair_stats,
    }
    (args.out / "chains_stats.json").write_text(json.dumps(stats, indent=1))
    log(json.dumps({k: v for k, v in stats.items() if k != "slice_pairs"}))


if __name__ == "__main__":
    main()
