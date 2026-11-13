#!/usr/bin/env python3
"""How far apart are the sheets of a scroll, and where? Read off its surface prediction.

The winding model keeps its count where sheets are about 175 um or more apart and loses windings
under about 115 um (transfer_check.py), and a spiral fit holds a sheet more easily where sheets
are loose. This script measures the spacing before any fit exists: on a few slices of the
prediction it walks outwards from the umbilicus at every angle, and takes the distances between
the centres of neighbouring predicted sheets. A radial line meets tilted sheets at an angle, so
these distances are upper bounds of the spacing along the sheet normal.

    uv run python sheet_gaps.py --surface <URL of the prediction's level 0> --umbilicus <umbilicus.json> \
        --z 6000 9000 12000 --voxel-um 8.64 [--out report.json] [--drop-cache]

Per slice it prints the median gap in rings of 500 voxels round the umbilicus, the share of gaps
of 175 um or more and of under 115 um, and the sheets met along the median line. One slice of an
8,600-voxel-wide prediction is about 2,000 chunks; --drop-cache deletes them afterwards.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from windcon.zarr_io import ZarrArray

RING = 500  # voxels
LOOSE_UM, TIGHT_UM = 175.0, 115.0


def axis_xy(path: Path, z: float) -> tuple[float, float]:
    """Umbilicus position (x, y) at z, interpolated between its control points."""
    points = sorted((p["z"], p["x"], p["y"]) for p in json.loads(path.read_text())["control_points"])
    zs, xs, ys = (np.array(column, dtype=np.float64) for column in zip(*points))
    return float(np.interp(z, zs, xs)), float(np.interp(z, zs, ys))


def radial_gaps(image: np.ndarray, centre: tuple[float, float], angles: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gaps between neighbouring sheets along every radial line (voxels), the radius of each gap's middle, and
    the sheets met on each line."""
    reach = int(np.hypot(*image.shape))
    radius = np.arange(reach)
    gaps, middles, counts = [], [], []
    for angle in np.arange(angles) * 2 * np.pi / angles:
        x = np.rint(centre[0] + radius * np.cos(angle)).astype(int)
        y = np.rint(centre[1] + radius * np.sin(angle)).astype(int)
        inside = (x >= 0) & (x < image.shape[1]) & (y >= 0) & (y < image.shape[0])
        line = np.zeros(reach, dtype=bool)
        line[inside] = image[y[inside], x[inside]]
        edges = np.diff(np.concatenate([[0], line.astype(np.int8), [0]]))
        centres = (np.flatnonzero(edges == 1) + np.flatnonzero(edges == -1) - 1) / 2
        counts.append(len(centres))
        if len(centres) >= 2:
            gaps.append(np.diff(centres))
            middles.append((centres[1:] + centres[:-1]) / 2)
    if not gaps:
        return np.zeros(0), np.zeros(0), np.array(counts)
    return np.concatenate(gaps), np.concatenate(middles), np.array(counts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--surface", required=True, help="URL of the surface prediction's level-0 array")
    ap.add_argument("--umbilicus", type=Path, required=True)
    ap.add_argument("--z", type=int, nargs="+", required=True, help="slices to read")
    ap.add_argument("--voxel-um", type=float, required=True, help="the scan's voxel size")
    ap.add_argument("--angles", type=int, default=720)
    ap.add_argument("--out", type=Path, help="write the report here as JSON")
    ap.add_argument("--drop-cache", action="store_true", help="delete the downloaded chunks afterwards")
    args = ap.parse_args()

    surface = ZarrArray(args.surface)
    report = {"surface": args.surface, "voxel_um": args.voxel_um, "slices": {}}
    for z in args.z:
        image = surface.read(((z, z + 1), (0, surface.shape[1]), (0, surface.shape[2])))[0] > 0
        gaps, middles, counts = radial_gaps(image, axis_xy(args.umbilicus, z), args.angles)
        if not len(gaps):
            print(f"z {z}: no sheets predicted")
            continue
        microns = gaps * args.voxel_um
        rings = {}
        for low in range(0, int(middles.max()) + 1, RING):
            ring = microns[(middles >= low) & (middles < low + RING)]
            if len(ring) >= 50:
                rings[f"{low}-{low + RING}"] = {"median_um": float(np.median(ring)), "gaps": len(ring),
                                                "loose": float(np.mean(ring >= LOOSE_UM)),
                                                "tight": float(np.mean(ring < TIGHT_UM))}  # fmt: skip
        row = {
            "median_gap_um": float(np.median(microns)),
            "mean_gap_um": float(microns.mean()),
            "mean_gap_um_without_the_widest_tenth": float(microns[microns <= np.percentile(microns, 90)].mean()),
            "gap_um_10th_90th": [float(np.percentile(microns, q)) for q in (10, 90)],
            f"share_of_gaps_{LOOSE_UM:g}_um_or_more": float(np.mean(microns >= LOOSE_UM)),
            f"share_of_gaps_under_{TIGHT_UM:g}_um": float(np.mean(microns < TIGHT_UM)),
            "sheets_on_the_median_line": float(np.median(counts)),
            "rings_voxels": rings,
        }
        report["slices"][str(z)] = row
        print(f"z {z}: median gap {row['median_gap_um']:.0f} um (10th-90th {row['gap_um_10th_90th'][0]:.0f}-"
              f"{row['gap_um_10th_90th'][1]:.0f}; mean {row['mean_gap_um']:.0f}, without the widest tenth "
              f"{row['mean_gap_um_without_the_widest_tenth']:.0f}), {np.mean(microns >= LOOSE_UM):.0%} of gaps "
              f"{LOOSE_UM:g} um or more, {np.mean(microns < TIGHT_UM):.0%} under {TIGHT_UM:g} um, "
              f"{np.median(counts):.0f} sheets on the median line")
        print("   rings (voxels: median um, loose share): " + "  ".join(
            f"{name}: {ring['median_um']:.0f}, {ring['loose']:.0%}" for name, ring in rings.items()), flush=True)  # fmt: skip
    if args.out:
        args.out.write_text(json.dumps(report, indent=1) + "\n")
    if args.drop_cache:
        shutil.rmtree(surface.cache)


if __name__ == "__main__":
    main()
