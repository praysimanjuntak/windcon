#!/usr/bin/env python3
"""Does a score that needs no annotation rank fits the way held-out truth does?

    python rank_scores.py results_ab.json
The score is the share of the raw automatic paths a fit keeps on one winding (the `auto` group of
datasets/diag): whole paths, their points, and their points without the scan test. It is compared
by Spearman rank correlation with each held-out truth, over fits that trained on no automatic path
and over fits that did, for all families together and for each family with at least 8 fits.
"""

from __future__ import annotations

import json
import sys

from summarize_ab import get

SCORES = [
    ("whole paths", ("diag_default", "auto", "fully_satisfied_fraction")),
    ("path points", ("diag_default", "auto", "points_satisfied_fraction")),
    ("path points, no scan test", ("diag_scanfree", "auto", "points_satisfied_fraction")),
]
TRUTHS = [
    ("arcs", ("pcls", "same_winding", "points_satisfied_fraction")),
    ("eval fibres", ("pcls", "fibers", "points_satisfied_fraction")),
    ("relative windings", ("pcls", "relative", "points_satisfied_fraction")),
    ("held-out patch area", ("patches", "satisfied_area_fraction")),
]


def ranks(values: list[float]) -> list[float]:
    """Ranks with ties given their mean rank."""
    order = sorted(range(len(values)), key=values.__getitem__)
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2
        i = j + 1
    return out


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    var = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return cov / var if var else float("nan")


def main() -> None:
    rows = [r for r in json.load(open(sys.argv[1])) if get(r, *SCORES[0][1]) is not None]
    groups = [("all families", rows)]
    for family in sorted({r["run"].split("_")[0] for r in rows}):
        groups.append((family, [r for r in rows if r["run"].split("_")[0] == family]))
    print("| fits | n | score | " + " | ".join(label for label, _ in TRUTHS) + " |")
    print("|---|---|---|" + "---|" * len(TRUTHS))
    for name, group in groups:
        for trained, what in ((False, "trained on no automatic path"), (True, "trained on automatic paths")):
            fits = [r for r in group if bool(get(r, "train_pcls", "auto")) == trained]
            if len(fits) < 8:
                continue
            for label, score in SCORES:
                cells = []
                for _, truth in TRUTHS:
                    pairs = [(get(r, *score), get(r, *truth)) for r in fits]
                    pairs = [p for p in pairs if p[1] is not None]
                    cells.append(f"{spearman(*map(list, zip(*pairs))):+.2f} ({len(pairs)})" if len(pairs) >= 8 else "–")
                print(f"| {name}, {what} | {len(fits)} | {label} | " + " | ".join(cells) + " |")
    print("\nSpearman rank correlation (number of fits with that truth); a family's held-out patch area"
          " exists only where its patches were held out.")


if __name__ == "__main__":
    main()
