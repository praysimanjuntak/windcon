#!/usr/bin/env python3
"""How do fits satisfy each annotation-targeting arm's extra patches? (select_patches.py arms)

For every run given, the satisfied fraction of the area of: the patches each arm added to the few
family (random, targeted, oracle), and the common held-out set. Fits that trained on every patch
are scored on their own training patches (satisfied_fitted.json); other fits on their held-out
patches (eval_patches/satisfied_fitted.json). If fits trained on every patch also fail the oracle
picks, those patches cannot be satisfied together with the rest under the fit's model.

    uv run python picks_in_fits.py runs/fullnodt_A_s1 runs/few_A_s1 --dataset data/PHercParis4 \
        [--out picks_in_fits.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--base", default="verified_patches_few10")
    ap.add_argument("--count", type=int, default=471)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    D = args.dataset
    base = {p.name for p in (D / args.base).iterdir()}
    sets = {f"{arm}_picks": {p.name for p in (D / f"{args.base}_{arm}{args.count}").iterdir()} - base
            for arm in ("rand", "targ", "orac")}
    sets["eval_al"] = {p.name for p in (D / "verified_patches_eval_al").iterdir()}
    report = {}
    print(f"{'run':16s}{'scored on':>16s}" + "".join(f"{k:>14s}" for k in sets))
    for run in args.runs:
        held = run / "eval_patches" / "satisfied_fitted.json"
        own = run / "satisfied_fitted.json"
        f, kind = (held, "held-out") if held.exists() else (own, "own training")
        sat = {e["id"]: (e["satisfied_area"], e["total_area"])
               for e in json.loads(f.read_text())["patches"] if e["total_area"] > 0}
        row = {}
        for key, names in sets.items():
            area = sum(sat[n][1] for n in names if n in sat)
            row[key] = round(sum(sat[n][0] for n in names if n in sat) / max(area, 1), 4)
        report[run.name] = {"scored_on": kind, **row}
        print(f"{run.name:16s}{kind:>16s}" + "".join(f"{100 * v:14.1f}" for v in row.values()))
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
