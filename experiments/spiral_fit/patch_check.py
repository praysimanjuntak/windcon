#!/usr/bin/env python3
"""Which verified patches do the automatic paths leave? (a label-quality check, no fit involved)

A path is assigned to the patch it mostly lies on (at least half of its points within --on voxels
of that patch's surface); it *leaves* the patch when, inside the patch's extent, at least --run
consecutive points lie more than --off voxels from the surface (check_links.py's path switch).
Paths switch sheets at a low base rate, so a patch that many of its paths leave is likely the one
that switches. Writes per-patch counts and compares patch sets given as folders of patch names.

    uv run python patch_check.py --chains-dir pcls --dataset data/PHercParis4 --out patch_check.json \
        --minus verified_patches_few10 --sets verified_patches_few10_orac471 ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_links import SPACING, longest_run, resample  # noqa: E402
from patch_surfaces import PatchSurfaces  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chains-dir", required=True, type=Path, help="folder with chains_paths.npz")
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--on", type=float, default=3.0)
    ap.add_argument("--off", type=float, default=7.0)
    ap.add_argument("--run", type=int, default=3)
    ap.add_argument("--min-paths", type=int, default=3, help="patches with fewer assigned paths are unscored")
    ap.add_argument("--sets", nargs="*", default=[], help="folders (in --dataset) of patch names to compare")
    ap.add_argument("--minus", help="folder (in --dataset) whose patches are removed from every set")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    with np.load(args.chains_dir / "chains_paths.npz") as npz:
        xy_all, offsets, zs, path_slice = (npz[k] for k in ("xy", "offsets", "zs", "path_slice"))
    surfaces = PatchSurfaces(args.dataset, int(zs.min()) - 30, int(zs.max()) + 30)
    pts, owner = [], []
    for p in range(len(offsets) - 1):
        xy = resample(xy_all[offsets[p]:offsets[p + 1]].astype(np.float64), SPACING)
        pts.append(np.column_stack([xy, np.full(len(xy), float(zs[path_slice[p]]))]))
        owner.append(np.full(len(xy), p))
    pts_all, owner = np.concatenate(pts), np.concatenate(owner)
    label, _, _ = surfaces.nearest(pts_all, args.on)
    on = label >= 0
    base = len(surfaces.names) + 1
    uniq, counts = np.unique(owner[on].astype(np.int64) * base + label[on], return_counts=True)
    path_of, patch_of = uniq // base, uniq % base
    order = np.lexsort((-counts, path_of))
    first = np.concatenate([[True], path_of[order][1:] != path_of[order][:-1]])
    assigned = [(int(p), int(q)) for p, q, c in zip(path_of[order][first], patch_of[order][first], counts[order][first])
                if c >= 0.5 * len(pts[int(p)])]
    per_patch: dict[str, list[int]] = {}
    for p, q in assigned:
        lab, nrm, _ = surfaces.nearest_on(q, pts[p], max_normal=25.0)
        off = (lab >= 0) & (np.abs(nrm) > args.off)
        entry = per_patch.setdefault(surfaces.names[q], [0, 0])
        entry[0] += 1
        entry[1] += int(longest_run(off) >= args.run)

    def summary(names: set[str] | None) -> dict:
        rows = [(n, a, s) for n, (a, s) in per_patch.items() if a >= args.min_paths and (names is None or n in names)]
        a = sum(r[1] for r in rows)
        s = sum(r[2] for r in rows)
        return {"patches_scored": len(rows), "paths": a, "paths_leaving_fraction": round(s / max(a, 1), 4),
                "patches_left_by_20pct_or_more": round(sum(r[2] >= 0.2 * r[1] for r in rows) / max(len(rows), 1), 4)}

    report = {"params": {"on": args.on, "off": args.off, "run": args.run, "min_paths": args.min_paths},
              "all": summary(None), "sets": {}}
    print(f"all patches: {report['all']}")
    minus = {p.name for p in (args.dataset / args.minus).iterdir()} if args.minus else set()
    for folder in args.sets:
        names = {p.name for p in (args.dataset / folder).iterdir()} - minus
        report["sets"][folder] = summary(names)
        print(f"{folder}: {report['sets'][folder]}")
    report["per_patch"] = {n: {"paths": a, "leaving": s} for n, (a, s) in sorted(per_patch.items())}
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
