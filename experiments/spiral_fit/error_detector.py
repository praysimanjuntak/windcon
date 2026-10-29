#!/usr/bin/env python3
"""Do the automatic chains flag where a fit is wrong? (paths as a fit check, not a fit input)

For a fit that never trained on the chains, agree_eval.sh scores every chain against the fit
(winding test only): a chain the fit puts on one winding agrees, one it spreads over several
windings disagrees. Each verified patch then gets a *check score*: the agreement of the chain
points lying on it (within --radius voxels), weighting every point by its chain's agreeing fraction.
The patches are independent ground truth for the fit (held out, or at least not the chains), so
the question is how well the check score predicts each patch's satisfaction in that fit.

With --flag-out the chains the fit disagrees with (agreement below --flag-below) are also written,
worst first, as a point-collection document of auto_chains.json's points: the places to inspect,
for VC3D or any reader of the spiral-fit point-collection format. This needs no verified patches;
without the run's satisfied_fitted.json the patch scoring is skipped.

    uv run python error_detector.py --run runs/sparse_A_s1 --chains-dir pcls --dataset data/PHercParis4 \
        [--patches eval_patches|train] [--out error_detector.json] [--flag-out flagged.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from windcon.annotations import load_collections, write_same_winding  # noqa: E402
from patch_surfaces import patch_vertices, umbilicus_xy  # noqa: E402


def auc(score: np.ndarray, positive: np.ndarray, weight: np.ndarray) -> float:
    """Weighted probability that a random positive outranks a random negative (ties count half)."""
    order = np.argsort(score, kind="stable")
    s, y, w = score[order], positive[order], weight[order]
    neg_below = 0.0
    total = 0.0
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        wp = w[i:j][y[i:j]].sum()
        wn = w[i:j][~y[i:j]].sum()
        total += wp * (neg_below + 0.5 * wn)
        neg_below += wn
        i = j
    return float(total / max(w[y].sum() * w[~y].sum(), 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--chains-dir", required=True, type=Path)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--patches", default="eval_patches", choices=["eval_patches", "train"],
                    help="score held-out patches (eval_patches/) or the run's own training patches")
    ap.add_argument("--z", type=int, nargs=2, default=(9900, 10900))
    ap.add_argument("--radius", type=float, default=3.5)
    ap.add_argument("--min-points", type=int, default=20, help="patches with fewer chain points are unscored")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--per-patch", type=Path, help="also write every scored patch's check score (JSON)")
    ap.add_argument("--flag-out", type=Path, help="write the chains the fit disagrees with (point collections)")
    ap.add_argument("--flag-below", type=float, default=0.5, help="agreement below which a chain is flagged")
    args = ap.parse_args()

    with np.load(args.chains_dir / "chains_paths.npz") as d:
        xy, offsets, zs, path_slice, path_chain = (d[k] for k in ("xy", "offsets", "zs", "path_slice", "path_chain"))
    per_vertex_chain = np.repeat(path_chain, np.diff(offsets))
    pts = np.column_stack([xy, np.repeat(zs[path_slice], np.diff(offsets)).astype(np.float64)])
    agree = np.full(int(path_chain.max()) + 1, np.nan)
    for e in json.loads((args.run / "agree_chains" / "satisfied_fitted.json").read_text())["pcls"]:
        agree[int(e["name"].removeprefix("chain_"))] = e["satisfied_points"] / max(e["total_points"], 1)
    vertex_agree = agree[per_vertex_chain]
    keep = np.isfinite(vertex_agree)  # chains the fitter dropped (outside the window) are unscored
    tree = cKDTree(pts[keep])
    vertex_agree = vertex_agree[keep]

    if args.flag_out:
        flagged = sorted((agree[int(c.name.removeprefix("chain_"))], c.name, c.xyz)
                         for c in load_collections(args.chains_dir / "auto_chains.json")
                         if agree[int(c.name.removeprefix("chain_"))] < args.flag_below)
        write_same_winding(args.flag_out, [(f"{name}_agree{a:.3f}", xyz) for a, name, xyz in flagged])
        print(f"flagged {len(flagged)} of {int(np.isfinite(agree).sum())} scored chains -> {args.flag_out}")

    axis = umbilicus_xy(args.dataset / "umbilicus.json")
    sat_file = (args.run / "eval_patches" if args.patches == "eval_patches" else args.run) / "satisfied_fitted.json"
    if not sat_file.exists():
        print(f"{sat_file} not found: patch scoring skipped")
        return
    sat = {e["id"]: (e["satisfied_area"], e["total_area"]) for e in json.loads(sat_file.read_text())["patches"]}
    rows = []
    for name, (s, t) in sat.items():
        if t <= 0:
            continue
        verts, _ = patch_vertices(args.dataset / "verified_patches" / name, *args.z)
        if len(verts) == 0:
            continue
        hits = tree.query_ball_point(verts, args.radius)
        idx = np.unique(np.concatenate([np.asarray(h, dtype=np.int64) for h in hits])) if len(hits) else []
        if len(idx) < args.min_points:
            continue
        coverage = float(np.mean([len(h) > 0 for h in hits]))
        c = verts.mean(axis=0)
        radius = float(np.linalg.norm(c[:2] - axis(c[2])))
        rows.append((float(vertex_agree[idx].mean()), coverage, radius, s / t, t, name))
    names = [r[5] for r in rows]
    rows = [r[:5] for r in rows]
    score, coverage, radius, frac, area = (np.array(v) for v in zip(*rows))
    good = frac >= 0.5
    two_sided = lambda x: round(max(auc(x, good, area), 1 - auc(x, good, area)), 3)  # noqa: E731
    report = {"run": args.run.name, "patches": args.patches, "scored_patches": len(rows),
              "scored_area_fraction_satisfied": round(float((frac * area).sum() / area.sum()), 4),
              "auc_area_weighted": round(auc(score, good, area), 3),
              "baseline_auc_chain_coverage": two_sided(coverage),
              "baseline_auc_distance_from_axis": two_sided(radius), "bins": []}
    edges = np.quantile(score, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    edges[-1] += 1e-9
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (score >= lo) & (score < hi)
        report["bins"].append({"check_score": [round(float(lo), 3), round(float(hi), 3)], "patches": int(m.sum()),
                               "area_satisfied": round(float((frac[m] * area[m]).sum() / area[m].sum()), 4)})
    print(json.dumps(report, indent=1))
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))
    if args.per_patch:
        args.per_patch.write_text(json.dumps([
            {"patch": n, "check_score": round(float(sc), 4), "chain_coverage": round(float(cv), 4),
             "satisfied_fraction": round(float(f), 4), "area": float(a)}
            for n, sc, cv, f, a in zip(names, score, coverage, frac, area)]))


if __name__ == "__main__":
    main()
