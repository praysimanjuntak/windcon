#!/usr/bin/env python3
"""Held-out agreement split by verified-patch coverage (diagnostic).

A held-out arc or eval fibre counts as covered when at least half of its points inside the fit window
lie over a verified patch surface (within --max-normal voxels across it, as in offset_check.py). For
each run, the share of points that pass the fitter's satisfaction test in its eval_pcls pass, for
covered and uncovered collections: in the full family the patches pin the fit down where they
exist, so the automatic paths can add information mainly on uncovered ground.

    uv run python coverage_split.py runs/fullnodt_A_s1 ... --dataset data/PHercParis4 \
        --heldout pcls/heldout_same.json --fibers fiber_points.json --z 9900 10900
(fiber_points.json from dump_fibers.py)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_surfaces import PatchSurfaces  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--heldout", required=True, type=Path, help="heldout_same.json")
    ap.add_argument("--fibers", type=Path, help="dump_fibers.py output")
    ap.add_argument("--z", type=int, nargs=2, default=(9900, 10900), help="fit window")
    ap.add_argument("--max-normal", type=float, default=8.0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    z0, z1 = args.z
    doc = json.loads(args.heldout.read_text())["collections"]
    groups = {"arcs": {c["name"]: np.array([p["p"] for p in c["points"].values()], dtype=np.float64)
                       for c in doc.values()}}
    if args.fibers:
        groups["fibers"] = {name: np.asarray(pts, dtype=np.float64).reshape(-1, 3)
                            for name, pts in json.loads(args.fibers.read_text()).items()}
    for g in groups:  # points inside the fit window only, collections with at least two
        groups[g] = {n: P[(P[:, 2] >= z0) & (P[:, 2] < z1)] for n, P in groups[g].items()}
        groups[g] = {n: P for n, P in groups[g].items() if len(P) >= 2}
    zs = np.concatenate([P[:, 2] for g in groups.values() for P in g.values()])
    surfaces = PatchSurfaces(args.dataset, int(zs.min()) - 30, int(zs.max()) + 30)
    covered = {g: {n: bool((surfaces.nearest(P, args.max_normal)[0] >= 0).mean() >= 0.5) for n, P in c.items()}
               for g, c in groups.items()}
    report: dict = {"collections": {}, "runs": {}}
    for g, cov in covered.items():
        report["collections"][g] = {
            ("covered" if k else "uncovered"): [sum(v == k for v in cov.values()),
                                                 int(sum(len(groups[g][n]) for n, v in cov.items() if v == k))]
            for k in (True, False)}
        print(g, "collections, points:", report["collections"][g])

    def key_of(e: dict) -> tuple[str, str] | None:
        src = Path(str(e.get("source_file") or ""))
        if src.name == "same_windings.json" and e.get("name") in covered["arcs"]:
            return "arcs", e["name"]
        if "fibers" in covered and src.parent.name == "fibers" and src.name in covered["fibers"]:
            return "fibers", src.name
        return None

    for run in args.runs:
        f = run / "eval_pcls" / "satisfied_fitted.json"
        if not f.exists():
            continue
        sat = {(g, k): [0, 0] for g in groups for k in (True, False)}
        for e in json.loads(f.read_text())["pcls"]:
            gk = key_of(e)
            if gk:
                s = sat[(gk[0], covered[gk[0]][gk[1]])]
                s[0] += e["satisfied_points"]
                s[1] += e["total_points"]
        row = {f"{g}_{'covered' if k else 'uncovered'}": round(s / max(t, 1), 4) for (g, k), (s, t) in sat.items()}
        report["runs"][run.name] = row
        print(run.name + ": " + ", ".join(f"{k} {100 * v:.1f}" for k, v in row.items()))
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
