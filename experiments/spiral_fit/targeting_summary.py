#!/usr/bin/env python3
"""Annotation-targeting test: held-out patch area of each fit over select_patches.py's common set.

Every fit is scored from its own eval_patches/satisfied_fitted.json restricted to the patches in
verified_patches_eval_al, so fits scored on a larger held-out set (few_A on eval_few) compare
exactly with the arms scored on eval_al.

    uv run python targeting_summary.py runs/few_A_s1 runs/few_rand_s1 runs/few_targ_s1 runs/few_orac_s1 \
        --dataset data/PHercParis4 [--out targeting.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    eval_set = {p.name for p in (args.dataset / "verified_patches_eval_al").iterdir()}
    report = {"eval_patches": len(eval_set), "runs": {}}
    for run in args.runs:
        f = run / "eval_patches" / "satisfied_fitted.json"
        if not f.exists():
            continue
        entries = [e for e in json.loads(f.read_text())["patches"] if e["id"] in eval_set and e["total_area"] > 0]
        area = sum(e["total_area"] for e in entries)
        sat = sum(e["satisfied_area"] for e in entries)
        report["runs"][run.name] = {"patches": len(entries), "area_satisfied": round(sat / max(area, 1), 4)}
        print(f"{run.name:14s} {len(entries):5d} patches  held-out area satisfied {100 * sat / max(area, 1):5.1f} %")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
