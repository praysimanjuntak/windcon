#!/usr/bin/env python3
"""Outer shell of a scroll from the mask of its scan, as the tifxyz fit_spiral.py reads as outer_shell/.

Only Paris 4 has a published outer shell. A masked scan is zero outside the scroll, so the shell
can be read off it: on every 20th slice of a coarse pyramid level, a ray is cast from the
umbilicus at every angle and the farthest non-zero voxel of the scroll's largest connected piece
is kept. Rows of the grid are slices, columns are angles.

    uv run python make_outer_shell.py --volume <URL of the masked zarr> --mask-level 5 --fit-level 2 \
        --umbilicus <umbilicus.json> --z-begin 9900 --z-end 10900 --out <dataset>/outer_shell
    ... --compare <published outer_shell>   # radius differences at the published shell's points

Coordinates are voxels of the fit level. fit_spiral.py bins the points by slice and angle, so only
their positions matter, not the grid.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from windcon.zarr_io import ZarrArray  # noqa: E402

ROW_STEP = 20  # slices between grid rows, as in the published Paris 4 shell


def axis_xy(path: Path, z: np.ndarray) -> np.ndarray:
    """Umbilicus position (x, y) at the given z, interpolated between its control points."""
    points = sorted((p["z"], p["x"], p["y"]) for p in json.loads(path.read_text())["control_points"])
    zs, xs, ys = (np.array(column, dtype=np.float64) for column in zip(*points))
    return np.stack([np.interp(z, zs, xs), np.interp(z, zs, ys)], axis=-1)


def envelope(mask: np.ndarray, centre_xy: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Distance from the centre to the farthest pixel of `mask` along each angle, in pixels; nan if none."""
    reach = np.arange(0.0, float(np.hypot(*mask.shape)), 0.5)
    xs = centre_xy[0] + reach[None, :] * np.cos(angles)[:, None]
    ys = centre_xy[1] + reach[None, :] * np.sin(angles)[:, None]
    hit = ndimage.map_coordinates(mask.astype(np.uint8), [ys, xs], order=0, mode="constant", cval=0) > 0
    last = hit.shape[1] - 1 - hit[:, ::-1].argmax(1)
    return np.where(hit.any(1), reach[last] + 0.5, np.nan)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--volume", required=True, help="URL of the masked scan's zarr group")
    ap.add_argument("--mask-level", type=int, required=True, help="pyramid level the mask is read at")
    ap.add_argument("--fit-level", type=int, required=True, help="pyramid level whose voxels the fit uses")
    ap.add_argument("--umbilicus", required=True, type=Path)
    ap.add_argument("--z-begin", type=int, required=True)
    ap.add_argument("--z-end", type=int, required=True)
    ap.add_argument("--margin", type=int, default=160, help="slices added at both ends (the flow field's z margin)")
    ap.add_argument("--angles", type=int, default=720)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--compare", type=Path, help="a published outer_shell folder to measure against")
    args = ap.parse_args()
    factor = 2 ** (args.mask_level - args.fit_level)
    volume = ZarrArray(f"{args.volume.rstrip('/')}/{args.mask_level}")
    angles = np.arange(args.angles) * 2 * np.pi / args.angles
    rows = np.arange(args.z_begin - args.margin, args.z_end + args.margin + 1, ROW_STEP)
    rows = rows[(rows >= 0) & (rows < volume.shape[0] * factor)]
    radius = np.full((len(rows), args.angles), np.nan)
    centres = axis_xy(args.umbilicus, rows.astype(np.float64))
    for index, z in enumerate(rows):
        level_z = min(int(z // factor), volume.shape[0] - 1)
        image = volume.read(((level_z, level_z + 1), (0, volume.shape[1]), (0, volume.shape[2])))[0]
        mask = ndimage.binary_fill_holes(ndimage.binary_closing(image > 0, iterations=2))
        labels, count = ndimage.label(mask)
        if count == 0:
            continue
        mask = labels == 1 + np.argmax(ndimage.sum(mask, labels, range(1, count + 1)))
        # A voxel i of the mask level covers fit voxels i * factor .. (i + 1) * factor - 1.
        centre = (centres[index] + 0.5) / factor - 0.5
        radius[index] = envelope(mask, centre, angles) * factor
    valid = np.isfinite(radius)
    x = np.where(valid, centres[:, :1] + radius * np.cos(angles)[None], -1.0)
    y = np.where(valid, centres[:, 1:] + radius * np.sin(angles)[None], -1.0)
    z = np.where(valid, rows[:, None].astype(np.float64), -1.0)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, values in (("x", x), ("y", y), ("z", z)):
        Image.fromarray(values.astype(np.float32)).save(args.out / f"{name}.tif")
    (args.out / "meta.json").write_text(json.dumps({
        "format": "tifxyz", "type": "seg", "uuid": "outer_shell_from_mask", "scale": [1 / ROW_STEP, 1 / ROW_STEP],
        "source": f"make_outer_shell.py: farthest non-zero voxel of {args.volume} level {args.mask_level} per angle",
    }, indent=1))  # fmt: skip
    print(f"{args.out}: {len(rows)} slices x {args.angles} angles, {valid.mean():.1%} of the nodes set, "
          f"radius median {np.nanmedian(radius):.0f} (5th-95th percentile {np.nanpercentile(radius, 5):.0f}-"
          f"{np.nanpercentile(radius, 95):.0f}) voxels")

    if args.compare:
        px, py, pz = (np.asarray(Image.open(args.compare / f"{axis}.tif"), dtype=np.float64) for axis in "xyz")
        keep = (px > 0) & (pz >= rows[0]) & (pz <= rows[-1])
        centre = axis_xy(args.umbilicus, pz[keep])
        theirs = np.hypot(px[keep] - centre[:, 0], py[keep] - centre[:, 1])
        angle = np.mod(np.arctan2(py[keep] - centre[:, 1], px[keep] - centre[:, 0]), 2 * np.pi)
        row = np.clip(np.rint((pz[keep] - rows[0]) / ROW_STEP).astype(int), 0, len(rows) - 1)
        column = np.rint(angle / (2 * np.pi) * args.angles).astype(int) % args.angles
        difference = radius[row, column] - theirs
        difference = difference[np.isfinite(difference)]
        print(f"compared at {len(difference):,} published points: mask shell minus published, voxels: median "
              f"{np.median(difference):+.0f}, 5th-95th percentile {np.percentile(difference, 5):+.0f} to "
              f"{np.percentile(difference, 95):+.0f}, within 30: {np.mean(np.abs(difference) <= 30):.1%}")


if __name__ == "__main__":
    main()
