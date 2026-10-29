#!/usr/bin/env python3
"""Agreement of fits with the automatic paths, scored as single-slice paths and as chains.

For runs scored by agree_eval.sh: the fraction of path points on their strip's winding (winding
test only), for the same points scored once per single-slice path (auto_singles.json) and once per
chain (auto_chains.json), binned by the number of paths in the point's chain. A chain's points all
pass only if the fit puts every linked path on one winding, so wrong links (and fit errors between
slices) show up as chain scores below the single-path scores of the same paths.

    uv run python agree_summary.py runs/fullnodt_A_s1 ... --chains-dir pcls --out agree.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BINS = ((1, 1), (2, 2), (3, 4), (5, 9), (10, 19), (20, 10**9))


def label(lo: int, hi: int) -> str:
    return (str(lo) if lo == hi else f"{lo}-{hi}") if hi < 10**9 else f">={lo}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--chains-dir", required=True, type=Path, help="build_chains.py output folder")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    stats = json.loads((args.chains_dir / "chains_stats.json").read_text())
    if stats["params"]["min_chain_paths"] != 1:
        raise SystemExit("needs build_chains.py --min-chain-paths 1 (chain_<n> = n-th chain)")
    with np.load(args.chains_dir / "chains_paths.npz") as npz:
        zs, path_slice, path_index, path_chain = (npz[k] for k in ("zs", "path_slice", "path_index", "path_chain"))
    chain_size = np.bincount(path_chain)
    size_of = {f"single_z{zs[s]}_{i:05d}": int(chain_size[c])
               for s, i, c in zip(path_slice.tolist(), path_index.tolist(), path_chain.tolist())}
    size_of |= {f"chain_{n:06d}": int(k) for n, k in enumerate(chain_size.tolist())}

    report = {}
    for run in args.runs:
        row = {}
        for key in ("singles", "chains"):
            f = run / f"agree_{key}" / "satisfied_fitted.json"
            if not f.exists():
                continue
            sat = np.zeros(len(BINS)); tot = np.zeros(len(BINS)); full = np.zeros(len(BINS)); n = np.zeros(len(BINS))
            for e in json.loads(f.read_text())["pcls"]:
                size = size_of[e["name"]]
                b = next(k for k, (lo, hi) in enumerate(BINS) if lo <= size <= hi)
                sat[b] += e["satisfied_points"]; tot[b] += e["total_points"]
                n[b] += 1; full[b] += e["satisfied_points"] == e["total_points"]
            row[key] = {
                "points_satisfied_fraction": round(float(sat.sum() / tot.sum()), 4),
                "by_chain_paths": {label(*BINS[k]): {
                    "points": int(tot[k]),
                    "points_satisfied_fraction": round(float(sat[k] / max(tot[k], 1)), 4),
                    "collections_fully_satisfied_fraction": round(float(full[k] / max(n[k], 1)), 4),
                } for k in range(len(BINS))},
            }
        f = run / "agree_relwind" / "satisfied_fitted.json"
        if f.exists():  # relative-winding pairs: a pair agrees when all its points pass
            entries = json.loads(f.read_text())["pcls"]
            row["relwind"] = {
                "pairs": len(entries),
                "pairs_fully_satisfied_fraction": round(
                    sum(e["satisfied_points"] == e["total_points"] for e in entries) / max(len(entries), 1), 4),
                "points_satisfied_fraction": round(
                    sum(e["satisfied_points"] for e in entries) / max(sum(e["total_points"] for e in entries), 1), 4),
            }
            print(f"{run.name}: relative-winding pairs agreeing with the fit "
                  f"{100 * row['relwind']['pairs_fully_satisfied_fraction']:.1f} % of {len(entries):,}")
        report[run.name] = row
        if {"singles", "chains"} <= row.keys():
            cells = [f"{lab}: {100 * row['singles']['by_chain_paths'][lab]['points_satisfied_fraction']:.1f}"
                     f" -> {100 * row['chains']['by_chain_paths'][lab]['points_satisfied_fraction']:.1f}"
                     for lab in (label(*b) for b in BINS)]
            print(f"{run.name}: singles -> chains, % points on their winding, by paths per chain: "
                  + "; ".join(cells))
    args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
