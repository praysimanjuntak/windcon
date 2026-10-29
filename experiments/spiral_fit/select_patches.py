#!/usr/bin/env python3
"""Pick extra training patches for the annotation-targeting test (few family).

Starting from the few family's training patches (verified_patches_few10), add --count more from the
held-out ones, three ways:
  rand   uniformly at random (the baseline);
  targ   the lowest check scores of a fit that never saw the chains (error_detector.py --per-patch):
         where the automatic chains say the fit is wrong, no ground truth used;
  orac   the lowest satisfaction in that fit (ground truth; the upper bound).
Every arm then scores on the same held-out set: the held-out patches minus all three extra sets.
Writes folders of symlinks next to verified_patches: verified_patches_few10_<arm><count>, and
verified_patches_eval_al with the common held-out set.

    uv run python select_patches.py --dataset data/PHercParis4 --per-patch runs/few_A_s1/per_patch.json \
        --satisfied runs/few_A_s1/eval_patches/satisfied_fitted.json --count 471
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def link_folder(folder: Path, names: list[str], source: Path) -> None:
    if folder.exists():
        raise SystemExit(f"{folder} exists; delete it to re-select")
    tmp = folder.with_name(folder.name + ".part")
    tmp.mkdir()
    for n in sorted(names):
        (tmp / n).symlink_to(source / n)
    tmp.rename(folder)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--per-patch", required=True, type=Path, help="error_detector.py --per-patch output")
    ap.add_argument("--satisfied", required=True, type=Path, help="the same fit's eval_patches/satisfied_fitted.json")
    ap.add_argument("--base", default="verified_patches_few10")
    ap.add_argument("--count", type=int, default=471)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    src = args.dataset / "verified_patches"
    base = {p.name for p in (args.dataset / args.base).iterdir()}
    pool = sorted(p.name for p in src.iterdir() if p.name not in base)
    scored = {r["patch"]: r for r in json.loads(args.per_patch.read_text())}
    sat = {e["id"]: e for e in json.loads(args.satisfied.read_text())["patches"] if e["total_area"] > 0}

    # Targeted and oracle picks come from the scored patches (those with chain points on them), so
    # all three arms draw from the same pool.
    candidates = [n for n in pool if n in scored and n in sat]
    rand = random.Random(args.seed).sample(candidates, args.count)
    targ = sorted(candidates, key=lambda n: (scored[n]["check_score"], n))[:args.count]
    orac = sorted(candidates, key=lambda n: (sat[n]["fraction"], n))[:args.count]
    extra = set(rand) | set(targ) | set(orac)
    evaluation = [n for n in pool if n not in extra]
    for arm, names in (("rand", rand), ("targ", targ), ("orac", orac)):
        link_folder(args.dataset / f"{args.base}_{arm}{args.count}", list(base | set(names)), src)
    link_folder(args.dataset / "verified_patches_eval_al", evaluation, src)

    def area_satisfied(names):
        a = sum(sat[n]["total_area"] for n in names if n in sat)
        return round(sum(sat[n]["satisfied_area"] for n in names if n in sat) / max(a, 1), 4)

    report = {"base": len(base), "pool": len(pool), "candidates": len(candidates), "count": args.count,
              "overlap_targ_orac": len(set(targ) & set(orac)), "overlap_targ_rand": len(set(targ) & set(rand)),
              "eval_patches": len(evaluation),
              "fit_satisfaction_of_picks": {arm: area_satisfied(names) for arm, names in
                                            (("rand", rand), ("targ", targ), ("orac", orac))},
              "fit_satisfaction_of_eval_set": area_satisfied(evaluation)}
    (args.dataset / "select_patches.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
