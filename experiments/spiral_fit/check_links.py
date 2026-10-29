#!/usr/bin/env python3
"""Check traced paths and cross-slice links against the verified patches (independent precision).

A path is *on* a patch when its points lie over that patch's surface within --on voxels. Each path
is assigned its majority patch P (at least half of its points on P). Then:

* path switch: within the part of the path that P covers, at least --run consecutive points lie
  more than --off voxels from P's surface (the path left P's sheet);
* link (a in slice z, b in slice z + dz), a assigned to P: b is *confirmed* when most of its covered
  points are on P and *contradicted* when most lie more than --off voxels from P (b is on another
  sheet). Links whose partner P does not cover are unverifiable.

The verified patches are human-checked surfaces, so this measures precision on the patch-covered
part of the scroll only.

    uv run python check_links.py --chains-dir pcls --dataset data/PHercParis4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_surfaces import PatchSurfaces  # noqa: E402

SPACING = 4.0  # px between checked points along a path


def resample(xy: np.ndarray, spacing: float) -> np.ndarray:
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    n = max(int(round(cum[-1] / spacing)), 1)
    s = np.linspace(0.0, cum[-1], n + 1)
    return np.stack([np.interp(s, cum, xy[:, 0]), np.interp(s, cum, xy[:, 1])], axis=1)


def longest_run(flags: np.ndarray) -> int:
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chains-dir", required=True, type=Path, help="folder with chains_paths.npz")
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--on", type=float, default=3.0)
    ap.add_argument("--off", type=float, default=7.0)
    ap.add_argument("--run", type=int, default=3, help="consecutive off points that make a switch")
    ap.add_argument("--max-links", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    with np.load(args.chains_dir / "chains_paths.npz") as npz:  # read every array once
        d = {k: npz[k] for k in npz.files}
    zs, offsets = d["zs"], d["offsets"]
    n_paths = len(offsets) - 1
    surfaces = PatchSurfaces(args.dataset, int(zs.min()) - 30, int(zs.max()) + 30)
    print(f"{len(surfaces.names)} patches, {len(surfaces.V):,} vertices; {n_paths:,} paths, "
          f"{len(d['links']):,} links", flush=True)

    pts, owner = [], []
    for p in range(n_paths):
        xy = resample(d["xy"][offsets[p]:offsets[p + 1]].astype(np.float64), SPACING)
        pts.append(np.column_stack([xy, np.full(len(xy), float(zs[d["path_slice"][p]]))]))
        owner.append(np.full(len(xy), p))
    pts_all, owner = np.concatenate(pts), np.concatenate(owner)
    label, _, _ = surfaces.nearest(pts_all, args.on)
    on = label >= 0
    print(f"{on.mean():.1%} of {len(pts_all):,} path points lie on a verified patch", flush=True)

    # Majority patch per path.
    base = len(surfaces.names) + 1
    uniq, counts = np.unique(owner[on].astype(np.int64) * base + label[on], return_counts=True)
    path_of, patch_of = uniq // base, uniq % base
    order = np.lexsort((-counts, path_of))
    first = np.concatenate([[True], path_of[order][1:] != path_of[order][:-1]])
    assigned = {int(p): int(q) for p, q, c in
                zip(path_of[order][first], patch_of[order][first], counts[order][first])
                if c >= 0.5 * len(pts[int(p)])}
    print(f"{len(assigned):,} paths ({len(assigned) / n_paths:.1%}) have a majority patch", flush=True)

    # Path switches.
    switch_of, covered_len, off_len = {}, 0, 0
    for p, q in assigned.items():
        lab, nrm, _ = surfaces.nearest_on(q, pts[p], max_normal=25.0)
        cov = lab >= 0
        off = cov & (np.abs(nrm) > args.off)
        covered_len += int(cov.sum())
        off_len += int(off.sum())
        switch_of[p] = longest_run(off) >= args.run
    report = {
        "paths": n_paths,
        "points_on_patch_fraction": round(float(on.mean()), 4),
        "paths_with_majority_patch": len(assigned),
        "path_switch_fraction": round(sum(switch_of.values()) / max(len(assigned), 1), 4),
        "path_off_length_fraction": round(off_len / max(covered_len, 1), 4),
    }
    sizes = np.bincount(d["path_chain"])
    linked = sizes[d["path_chain"]] >= 2
    for name, want in (("unlinked", False), ("linked", True)):
        group = [p for p in assigned if linked[p] == want]
        report[f"path_switch_fraction_{name}"] = round(
            sum(switch_of[p] for p in group) / max(len(group), 1), 4)
        report[f"paths_{name}_with_majority_patch"] = len(group)

    # Links.
    links = d["links"]
    rng = np.random.default_rng(args.seed)
    sample = rng.choice(len(links), min(args.max_links, len(links)), replace=False)
    path_id = {(int(s), int(i)): n for n, (s, i) in enumerate(zip(d["path_slice"], d["path_index"]))}
    outcome = {"confirmed": 0, "contradicted": 0, "ambiguous": 0, "unverifiable": 0}
    for s, i, j in links[sample]:
        a, b = path_id[(int(s), int(i))], path_id[(int(s) + 1, int(j))]
        q = assigned.get(a)
        if q is None:
            outcome["unverifiable"] += 1
            continue
        lab, nrm, _ = surfaces.nearest_on(q, pts[b], max_normal=25.0)
        cov = lab >= 0
        if cov.sum() < 3:
            outcome["unverifiable"] += 1
            continue
        on_frac = (cov & (np.abs(nrm) <= args.on)).sum() / cov.sum()
        off_frac = (cov & (np.abs(nrm) > args.off)).sum() / cov.sum()
        outcome["confirmed" if on_frac >= 0.6 else "contradicted" if off_frac >= 0.6 else "ambiguous"] += 1
    verifiable = outcome["confirmed"] + outcome["contradicted"] + outcome["ambiguous"]
    report |= {
        "links_checked": int(len(sample)), **{f"links_{k}": v for k, v in outcome.items()},
        "link_contradicted_fraction": round(outcome["contradicted"] / max(verifiable, 1), 4),
    }
    (args.chains_dir / "check_links.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
