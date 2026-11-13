#!/usr/bin/env python3
"""An outer shell for fit_spiral.py from the rays alone: the surface on which they count winding N.

fit_spiral.py pins one winding (`shell_outer_winding_idx`, 130 by default) to its `outer_shell/`
mesh. Paris 4's published shell is a traced sheet; other scrolls have none, and the boundary of a
scan's mask is not one winding (outer wraps break off, so the count to it changes with the angle).
A winding-inference store says how many windings lie between the points of each ray. Summed
outwards from the umbilicus that gives a winding count at every radius, slice and angle, and the
radius where the count reaches N is a surface a fit can pin winding N to.

    uv run python count_shell.py --store <store> --umbilicus <umbilicus.json> --winding 130 \
        --z-begin 9500 --z-end 11000 --out <dataset>/outer_shell [--mask-shell <tifxyz>]

Between two neighbouring crossings of a ray lies one winding. In cells of 10 degrees by 100 slices
the script divides the number of such steps in each ring of 100 voxels by the radial distance they
cover, which is the windings per radial voxel there, fills rings without rays from the nearest one
with rays (the core, and beyond the outermost seeds), and sums outwards. The radii are smoothed
over neighbouring cells and written as a tifxyz grid: rows are slices 20 apart, columns angles.
With --mask-shell it also prints the count at that shell, per sector.

Use the result with `"shell_outer_winding_idx": N` and `"shell_min_confidence": 0` (rows 20
slices apart do not pass fit_spiral.py's default confidence gate).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from build_store import load_store
from PIL import Image
from scipy import ndimage

SECTORS, BAND, RING, ROW_STEP = 36, 100, 100, 20
MIN_STEPS = 30  # steps a ring of a cell needs for a density of its own


def axis_xy(path: Path, z: np.ndarray) -> np.ndarray:
    """Umbilicus position (x, y) at the given z, interpolated between its control points."""
    points = sorted((p["z"], p["x"], p["y"]) for p in json.loads(path.read_text())["control_points"])
    zs, xs, ys = (np.array(column, dtype=np.float64) for column in zip(*points))
    return np.stack([np.interp(z, zs, xs), np.interp(z, zs, ys)], axis=-1)


def steps(store: Path, umbilicus: Path) -> tuple[np.ndarray, ...]:
    """One row per pair of neighbouring crossings that runs outwards: slice, angle, radius of its middle, and
    the radial distance it covers."""
    rays, _ = load_store(store)
    index = np.repeat(np.arange(len(rays["crossing_counts"])), rays["crossing_counts"])
    points = rays["ray_origin_zyx"][index] + rays["crossing_t"][:, None] * rays["ray_step_zyx"][index]  # z, y, x
    offset = points[:, [2, 1]] - axis_xy(umbilicus, points[:, 0])
    radius = np.hypot(offset[:, 0], offset[:, 1])
    angle = np.mod(np.arctan2(offset[:, 1], offset[:, 0]), 2 * np.pi)
    pair = (index[1:] == index[:-1]) & (radius[1:] - radius[:-1] > 0.5)
    return (points[:-1, 0][pair], angle[:-1][pair], ((radius[1:] + radius[:-1]) / 2)[pair], (radius[1:] - radius[:-1])[pair])


def count_profiles(z, angle, middle, covered, z_begin: int, z_end: int, reach: int) -> tuple[np.ndarray, np.ndarray]:
    """Windings from the umbilicus to the outer edge of every ring, per cell (band, sector, ring), and the
    share of rings that had steps of their own."""
    bands = max(1, int(np.ceil((z_end - z_begin) / BAND)))
    rings = int(np.ceil(reach / RING))
    cell = (
        np.clip(((z - z_begin) // BAND).astype(int), 0, bands - 1),
        np.minimum((angle / (2 * np.pi) * SECTORS).astype(int), SECTORS - 1),
        np.minimum((middle // RING).astype(int), rings - 1),
    )
    number = np.zeros((bands, SECTORS, rings))
    distance = np.zeros((bands, SECTORS, rings))
    np.add.at(number, cell, 1.0)
    np.add.at(distance, cell, covered)
    # A cell's ring takes in its neighbours in slice and angle: steps are few near the core.
    def with_neighbours(cells: np.ndarray) -> np.ndarray:
        padded = np.pad(np.pad(cells, ((1, 1), (0, 0), (0, 0)), mode="edge"), ((0, 0), (1, 1), (0, 0)), mode="wrap")
        return sum(padded[b : b + bands, s : s + SECTORS] for b in range(3) for s in range(3))

    number, distance = with_neighbours(number), with_neighbours(distance)
    known = number >= MIN_STEPS
    density = np.where(known, number / np.maximum(distance, 1e-9), np.nan)  # windings per radial voxel
    for band in range(bands):
        for sector in range(SECTORS):
            line, have = density[band, sector], np.flatnonzero(known[band, sector])
            if not len(have):
                raise SystemExit(f"no rays in slices {z_begin + band * BAND}+, sector {sector}: the store does not cover the window")
            line[:] = np.interp(np.arange(rings), have, line[have])  # nearest value beyond the ends
    return np.cumsum(density * RING, axis=-1), known


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--umbilicus", type=Path, required=True)
    ap.add_argument("--winding", type=int, default=130, help="the winding the shell is to be")
    ap.add_argument("--z-begin", type=int, required=True)
    ap.add_argument("--z-end", type=int, required=True)
    ap.add_argument("--margin", type=int, default=160, help="slices added at both ends (the flow field's z margin)")
    ap.add_argument("--angles", type=int, default=720)
    ap.add_argument("--mask-shell", type=Path, help="a shell from the scan's mask: print the count the rays give at it")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    z, angle, middle, covered = steps(args.store, args.umbilicus)
    inside = (z >= args.z_begin) & (z < args.z_end)
    z, angle, middle, covered = z[inside], angle[inside], middle[inside], covered[inside]
    reach = int(np.percentile(middle, 99.9)) + 4 * RING
    count, known = count_profiles(z, angle, middle, covered, args.z_begin, args.z_end, reach)
    edges = (np.arange(count.shape[-1]) + 1) * RING
    outermost = edges[np.array([[np.flatnonzero(k).max() for k in band] for band in known])]

    # The radius at which each cell's count reaches the winding; beyond the last ring the last density goes on.
    last_density = (count[..., -1] - count[..., -2]) / RING
    radius = np.empty(count.shape[:2])
    for band, sector in np.ndindex(radius.shape):
        line = count[band, sector]
        radius[band, sector] = (
            np.interp(args.winding, line, edges) if line[-1] >= args.winding
            else edges[-1] + (args.winding - line[-1]) / last_density[band, sector]
        )  # fmt: skip
    beyond = radius > outermost
    radius = ndimage.gaussian_filter(radius, sigma=(1.0, 1.0), mode=("nearest", "wrap"))
    print(f"{len(z):,} steps between neighbouring crossings in the window; winding {args.winding} lies at a radius of "
          f"{np.median(radius):.0f} voxels (5th-95th percentile {np.percentile(radius, 5):.0f}-{np.percentile(radius, 95):.0f}); "
          f"in {beyond.mean():.0%} of the cells that is beyond the outermost ring with rays")
    whole = count[:, :, :].mean(0)
    print("windings from the umbilicus, all slices, by radius: " + ", ".join(
        f"{int(edge)}: {np.median(whole[:, k]):.0f}" for k, edge in enumerate(edges) if edge % 500 == 0))  # fmt: skip

    if args.mask_shell:
        sx, sy, sz = (np.asarray(Image.open(args.mask_shell / f"{a}.tif"), dtype=np.float64) for a in "xyz")
        keep = (sx > 0) & (sz >= args.z_begin) & (sz < args.z_end)
        offset = np.stack([sx[keep], sy[keep]], -1) - axis_xy(args.umbilicus, sz[keep])
        shell_radius = np.hypot(offset[:, 0], offset[:, 1])
        sector = np.minimum((np.mod(np.arctan2(offset[:, 1], offset[:, 0]), 2 * np.pi) / (2 * np.pi) * SECTORS).astype(int), SECTORS - 1)
        band = np.clip(((sz[keep] - args.z_begin) // BAND).astype(int), 0, count.shape[0] - 1)
        at_shell = np.array([
            np.interp(r, edges, count[b, s]) if r <= edges[-1] else count[b, s, -1] + (r - edges[-1]) * last_density[b, s]
            for r, b, s in zip(shell_radius, band, sector)
        ])  # fmt: skip
        by_sector = [np.median(at_shell[sector == k]) for k in range(SECTORS) if (sector == k).any()]
        print(f"count at {args.mask_shell.name}: median {np.median(at_shell):.0f}, by 10-degree sector "
              f"{min(by_sector):.0f} to {max(by_sector):.0f}")

    rows = np.arange(args.z_begin - args.margin, args.z_end + args.margin + 1, ROW_STEP)
    angles = np.arange(args.angles) * 2 * np.pi / args.angles
    # Linear between cell centres: clamped along the slices, wrapped round the angle.
    wrapped = np.concatenate([radius[:, -1:], radius, radius[:, :1]], axis=1)
    band_at = np.clip((rows - args.z_begin) / BAND - 0.5, 0, radius.shape[0] - 1)
    sector_at = angles / (2 * np.pi) * SECTORS + 0.5
    grid = ndimage.map_coordinates(wrapped, np.meshgrid(band_at, sector_at, indexing="ij"), order=1, mode="nearest")
    centres = axis_xy(args.umbilicus, rows.astype(np.float64))
    x = centres[:, :1] + grid * np.cos(angles)[None]
    y = centres[:, 1:] + grid * np.sin(angles)[None]
    zz = np.broadcast_to(rows[:, None].astype(np.float64), grid.shape)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, values in (("x", x), ("y", y), ("z", zz)):
        Image.fromarray(values.astype(np.float32)).save(args.out / f"{name}.tif")
    (args.out / "meta.json").write_text(json.dumps({
        "format": "tifxyz", "type": "seg", "uuid": f"winding_{args.winding}_by_ray_count", "scale": [1 / ROW_STEP, 1 / ROW_STEP],
        "source": f"count_shell.py: radius at which the rays of {args.store} count winding {args.winding}",
    }, indent=1))  # fmt: skip
    print(f"{args.out}: {len(rows)} slices x {args.angles} angles")


if __name__ == "__main__":
    main()
