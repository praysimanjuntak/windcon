#!/usr/bin/env python3
"""Markdown tables of the spiral-fit A/B experiment from collect_metrics.py output.

    python summarize_ab.py results_ab.json
Each cell lists the per-seed values (seed 1 / seed 2) of a metric.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict


def get(row: dict, *path: str):
    cur = row
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


HUMAN_FAMILIES = {"full", "fullnodt"}  # families that train on all verified patches
HUMAN_PCL_FAMILIES = HUMAN_FAMILIES | {"few"}  # families that train on the human point collections
AUTO_VARIANTS = {"B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "M", "N", "O", "P", "Q",
                 "Eg", "Gg", "Ig", "IgDT"}  # variants that train on automatic paths
METRICS = [
    # label, path in a collect_metrics row, families/variants whose fits trained on the inputs
    ("held-out patch area", ("patches", "satisfied_area_fraction"), HUMAN_FAMILIES),
    ("held-out arcs: points", ("pcls", "same_winding", "points_satisfied_fraction"), set()),
    ("held-out arcs: whole arcs", ("pcls", "same_winding", "fully_satisfied_fraction"), set()),
    ("eval fibers: points", ("pcls", "fibers", "points_satisfied_fraction"), set()),
    ("relative windings: points", ("pcls", "relative", "points_satisfied_fraction"), HUMAN_PCL_FAMILIES),
    ("training patch area", ("train_inputs", "satisfied_area_fraction"), HUMAN_PCL_FAMILIES),
    ("auto paths: points", ("diag_default", "auto", "points_satisfied_fraction"), AUTO_VARIANTS),
    ("auto paths: points, no scan test", ("diag_scanfree", "auto", "points_satisfied_fraction"),
     AUTO_VARIANTS),
]
ORDER = ("sparse_A", "sparse_L", "sparse_B", "sparse_C", "sparse_D", "sparse_E", "sparse_Eg",
         "sparse_F", "sparse_H", "sparse_J", "sparse_K", "sparse_M", "sparse_N", "sparse_O",
         "sparse_P", "sparse_Q", "few_A", "few_Ig", "few_rand", "few_targ", "few_orac", "few_AR10",
         "few_oracR10", "few_ALR", "few_oracLR", "few_ANW", "few_oracNW", "few_AP", "few_oracP",
         "few_ADT", "few_oracDT", "few_ASD", "few_oracSD", "full_A", "full_B", "fullnodt_A",
         "fullnodt_ADT", "fullnodt_D", "fullnodt_E", "fullnodt_Eg", "fullnodt_G", "fullnodt_Gg",
         "fullnodt_I", "fullnodt_Ig", "fullnodt_IgDT")


def main() -> None:
    rows = json.load(open(sys.argv[1]))
    by_regime: dict[str, list[dict]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["run"]):
        by_regime[row["run"].rsplit("_s", 1)[0]].append(row)
    regimes = [r for r in ORDER if r in by_regime]
    print("| metric | " + " | ".join(regimes) + " |")
    print("|---|" + "---|" * len(regimes))
    for label, path, trained_in in METRICS:
        cells = []
        for regime in regimes:
            family, variant = regime.split("_")
            vals = [get(row, *path) for row in by_regime[regime]]
            vals = [v for v in vals if v is not None]
            if not vals or (path[0] == "train_inputs" and family == "sparse"):
                cells.append("–")
                continue
            text = " / ".join(f"{100 * v:.1f}" for v in vals)
            if len(vals) > 1:
                text += f" (mean {100 * sum(vals) / len(vals):.1f})"
            trained = family in trained_in or variant in trained_in
            cells.append(f"*{text}*" if trained else text)
        print(f"| {label} | " + " | ".join(cells) + " |")
    print("\nValues are % (one per seed, in seed order; mean in brackets); italics = the scored inputs"
          " were training inputs of that fit.")


if __name__ == "__main__":
    main()
