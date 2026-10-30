#!/usr/bin/env python3
"""Figure: the fit check as a calibration chart. For each fit, its scored verified patches are split
into fifths by check score (error_detector.py: the chains' agreement with the fit on each patch, no
patch labels used), and each bar is the share of that fifth's patch area the fit actually satisfies.

    uv run python plot_fit_check.py results/spiral_fit/error_detector/sparse_A_s1.json:"no human data" \
        results/spiral_fit/error_detector/few_A_s1.json:"10 % of patches" --out fit_check.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reports", nargs="+", help="error_detector.py --out report, then :label")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    fits = []
    for item in args.reports:
        path, _, label = item.partition(":")
        report = json.loads(Path(path).read_text())
        fits.append((label or report["run"], report))
    width = 0.8 / len(fits)
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=120, layout="constrained")
    for i, (label, report) in enumerate(fits):
        shares = [100 * b["area_satisfied"] for b in report["bins"]]
        x = np.arange(len(shares)) + (i - (len(fits) - 1) / 2) * width
        ax.bar(x, shares, width, label=f"{label} (AUC {report['auc_area_weighted']:.2f})",
               color=plt.cm.viridis(i / max(len(fits) - 1, 1)))
    ax.set_xticks(range(5), ["lowest", "2nd", "3rd", "4th", "highest"])
    ax.set_xlabel("fifth of the scored patches by check score (chain agreement, no patch labels)")
    ax.set_ylabel("patch area the fit satisfies (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.out)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
