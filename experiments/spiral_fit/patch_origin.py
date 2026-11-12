#!/usr/bin/env python3
"""Satisfied patch area split by where a verified patch came from.

Most folders in the published `verified_patches/` are named `band-seed*`: sheets grown
automatically from surface tracks by villa's `grow_track_graph.py`. The rest (`auto_grown_*_sel_*`,
`same_wrap*_lasagna`, `*_fill_sel_*`, ...) were grown or traced in VC3D and selected by a person.
A fit is scored on its held-out patches (eval_patches/satisfied_fitted.json) when that pass exists,
otherwise on its own training patches (satisfied_fitted.json). The held-out pass of a fit that
trained on some published patches (the band-seed ones, say) scores those too: an origin whose
scored patches are mostly the fit's own training patches is marked with `*`.

    uv run python patch_origin.py runs/sparse_A_s1 runs/fullnodt_A_s1 [--out patch_origin.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ORIGINS = ("band_seed", "curated")


def origin(patch_id: str) -> str:
    return "band_seed" if patch_id.startswith("band-seed") else "curated"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    report = {}
    print(f"{'run':20s}" + "".join(f"{o + ' n':>14s}{o + ' %':>14s}" for o in ORIGINS))
    for run in args.runs:
        held = run / "eval_patches" / "satisfied_fitted.json"
        own = run / "satisfied_fitted.json"
        f = held if held.exists() else own
        if not f.exists():
            continue
        trained = {e["id"] for e in json.loads(own.read_text()).get("patches", [])} if own.exists() else set()
        area = {o: [0.0, 0.0, 0, 0] for o in ORIGINS}  # satisfied, total, patches, training patches
        for e in json.loads(f.read_text())["patches"]:
            if e["total_area"] > 0:
                a = area[origin(e["id"])]
                a[0] += e["satisfied_area"]
                a[1] += e["total_area"]
                a[2] += 1
                a[3] += e["id"] in trained
        if not any(a[2] for a in area.values()):
            continue
        row = {}
        for o, (sat, total, n, n_trained) in area.items():
            row[o] = {"patches": n, "area_satisfied": round(sat / max(total, 1), 4),
                      "trained_on": 2 * n_trained > n}
        report[run.name] = row
        print(f"{run.name:20s}" + "".join(
            f"{row[o]['patches']:14d}{100 * row[o]['area_satisfied']:13.1f}{'*' if row[o]['trained_on'] else ' '}"
            for o in ORIGINS))
    print("* the fit trained on these patches; the others were held out")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
